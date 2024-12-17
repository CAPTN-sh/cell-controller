import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass, asdict, field
import enum
import json
import pickle
import time
from typing import Any, Dict, List, Tuple
import torch
from torch import nn

from cellcontroller.common.thresholder import Thresholder, ThresholderConfig
from cellcontroller.common.broker import MqttBrokerConfig
from cellcontroller.common.client import MqttPublisher, MqttSubscriber, make_inactive_mqtt_config
from cellcontroller.common.utils import LOGGER

from cellcontroller.adet.autoencoders.ae_factory import AEFactory
from cellcontroller.adet.autoencoders.dae import evaluate_dae
from cellcontroller.adet.data_preprocessing import (
    apply_preprocessing_for_inference,
    extract_feature_indices,
    pack_into_sequences,
    scale_test_data,
)
from cellcontroller.classifier import CellEventClassifier
from cellcontroller.structs import CellPayload, CellReport


class CellMqttPublisher(MqttPublisher):
    def __init__(self, broker_config: MqttBrokerConfig) -> None:
        super().__init__(make_inactive_mqtt_config(broker_config=broker_config, id="cell_pub"))


class CellMqttSubscriber(MqttSubscriber):
    def __init__(self, broker_config: MqttBrokerConfig, cell_topic: str) -> None:
        super().__init__(make_inactive_mqtt_config(broker_config=broker_config, id="cell_sub"))
        self.cell_topic = cell_topic

    def on_message(self, _, __, msg) -> None:
        try:
            mqtt_payload = json.loads(msg.payload.decode('utf8'))
        except json.JSONDecodeError:
            return
        p = CellPayload.from_mqtt_payload(mqtt_payload)
        self.message_queues[self.cell_topic].put_nowait(p)


class CellReportStorageAppendResult(enum.Enum):
    SKIP = -1
    SAMPLE = 0
    SEQUENCE = 1


class CellReportStorage:
    def __init__(self, routers: List[str] | None, same_routers_for_seq: bool, maxlen: int, label_len: int) -> None:
        self.routers = routers
        self.maxlen = maxlen
        self.label_len = label_len
        if self.routers and same_routers_for_seq:
            self.deques = {r: deque(maxlen=self.maxlen) for r in self.routers}
            self.counters = {r: 0 for r in self.routers}
            self.mix_routers = False
        else:
            self.deques = {"main": deque(maxlen=self.maxlen)}
            self.counters = {"main": 0}
            self.mix_routers = True

    def append(self, report_dict: Dict[str, Any]) -> CellReportStorageAppendResult:
        r = report_dict["identity"]  # checked before that it's not empty
        if self.routers and r not in self.routers:
            return CellReportStorageAppendResult.SKIP

        if not self.mix_routers:
            self.deques[r].append(report_dict)
            self.counters[r] += 1
            if self.counters[r] == self.label_len:
                self.counters[r] = 0
                return CellReportStorageAppendResult.SEQUENCE
            else:
                return CellReportStorageAppendResult.SAMPLE
        else:
            self.deques["main"].append(report_dict)
            self.counters["main"] += 1
            if self.counters["main"] == self.label_len:
                self.counters["main"] = 0
                return CellReportStorageAppendResult.SEQUENCE
            else:
                return CellReportStorageAppendResult.SAMPLE

    def get(self, router: str | None = None) -> List[Dict[str, Any]]:
        if router and router not in self.routers and not self.mix_routers:
            raise ValueError(
                f"CellReportStorage:get: router {router} is not in the non-empty routers list in non-mixing mode; "
                f"was get called before append?"
            )

        if not router or self.mix_routers:
            return [*self.deques["main"]]
        else:
            return [*self.deques[router]]


@dataclass
class CellAnomalyDetectorObjectPathsConfig:
    ad_base_seq_model_path: str | None
    ad_prob_seq_model_path: str | None
    ad_point_model_path: str | None
    ad_imputer_path: str | None
    ad_encoder_path: str | None
    ad_scaler_path: str | None
    ec_baselines_path: str | None


@dataclass
class CellAnomalyDetectionResult:
    # mobile operator
    operator: str = "vodafone"
    # unique routers involved in the sequence
    routers: List[str] = field(default_factory=lambda: [])
    # number of samples in the sequence
    seq_len: int = 1
    # label length (amount of non-overlapping new samples)
    new_samples: int = 1
    # technical anomaly for each sample
    tech_anomaly: bool = False
    # if 2/3 models detect anomaly
    det_anomaly: bool = False
    # events detected
    events: List[str] = field(default_factory=lambda: [])
    # loss / threshold ratios from all models (>=1.0 is anomaly)
    lt_ratios: List[float] = field(default_factory=lambda: [])
    # missing features
    missing: List[str] = field(default_factory=lambda: [])
    # whether to impute
    imputing: bool = True


class CellAnomalyDetector:
    # was selected during the training
    MAX_SEQUENCE_LEN = 8

    def __init__(
        self,
        config: CellAnomalyDetectorObjectPathsConfig,
        sequence_len: int = 8,
        label_len: int = 4,
        operator: str = "vodafone",
        active_routers: List[str] | None = None,
        adet_seq_same_routers: bool = False,
        with_lte: bool = True,
        imputing: bool = True,
    ) -> None:
        self.ad_base_seq_model = None
        self.ad_prob_seq_model = None
        self.ad_point_model = None
        self.ad_imputer = None
        self.ad_encoder = None
        self.ad_scaler = None
        self.event_classifier = None
        self.is_ad = self._maybe_load_ad_objects(
            ad_base_seq_model_path=config.ad_base_seq_model_path,
            ad_prob_seq_model_path=config.ad_prob_seq_model_path,
            ad_point_model_path=config.ad_point_model_path,
            ad_imputer_path=config.ad_imputer_path,
            ad_encoder_path=config.ad_encoder_path,
            ad_scaler_path=config.ad_scaler_path,
            ec_baselines_path=config.ec_baselines_path,
        )

        if sequence_len > self.MAX_SEQUENCE_LEN:
            LOGGER.warning("WARNING: CellAnomalyDetector: sequence_len is greater than MAX_SEQUENCE_LEN, clipping..")
        self.sequence_len = max(1, min(sequence_len, self.MAX_SEQUENCE_LEN))
        if label_len > self.sequence_len:
            LOGGER.warning("WARNING: CellAnomalyDetector: label_len is greater than sequence_len, clipping..")
        self.label_len = max(1, min(label_len, self.sequence_len))

        self.operator = operator
        self.active_routers = active_routers
        self.with_lte = with_lte
        self.imputing = imputing

        self.feature_indicies = None

        self.cell_reports = CellReportStorage(
            routers=self.active_routers,
            same_routers_for_seq=adet_seq_same_routers if self.active_routers is not None else False,
            maxlen=self.sequence_len,
            label_len=self.label_len,
        )

    def detect(self, cell_report: Dict[str, Any]) -> CellAnomalyDetectionResult | None:
        if self.feature_indicies is None:
            self.feature_indicies = extract_feature_indices(cell_report)

        res_base_seq = None
        res_prob_seq = None
        events = []
        router = cell_report["identity"] if self.active_routers else None
        append_res = self.cell_reports.append(cell_report)
        if append_res == CellReportStorageAppendResult.SEQUENCE:
            cell_report_sequence = self.cell_reports.get(router)
            events = self.event_classifier.identify_events(cell_report_sequence)

            with ThreadPoolExecutor() as executor:
                future_base_seq = executor.submit(self._ad_inference, cell_report_sequence, self.ad_base_seq_model)
                future_prob_seq = executor.submit(self._ad_inference, cell_report_sequence, self.ad_prob_seq_model)
                future_point = executor.submit(self._ad_inference, [cell_report], self.ad_point_model)

                for future in as_completed([future_base_seq, future_prob_seq, future_point]):
                    result = future.result()
                    if future == future_base_seq:
                        res_base_seq = result
                    elif future == future_prob_seq:
                        res_prob_seq = result
                    elif future == future_point:
                        res_point = result
        elif append_res == CellReportStorageAppendResult.SAMPLE:
            res_point = self._ad_inference([cell_report], self.ad_point_model)
            events = self.event_classifier.identify_events(cell_report)
        else:
            return None

        res = CellAnomalyDetectionResult(operator=self.operator)
        if res_base_seq and res_prob_seq:
            res.routers = list(set([cell_report["identity"] for cell_report in cell_report_sequence]))
            res.seq_len = self.sequence_len
            res.new_samples = self.label_len
            # anomaly is true if 2/3 models detect anomaly
            res.det_anomaly = sum([res_base_seq[1] > 0.0, res_prob_seq[1] > 0.0, res_point[1] > 0.0]) >= 2
            res.lt_ratios = [float(res_base_seq[0]), float(res_prob_seq[0]), float(res_point[0])]
        else:
            res.routers = [cell_report["identity"]]
            res.seq_len = 1
            res.new_samples = 1
            res.det_anomaly = res_point[1] > 0.0
            res.lt_ratios = [float(res_point[0])]
        res.tech_anomaly = res_point[2] > 0
        res.events = events
        res.missing = [k for k, v in cell_report.items() if v is None]
        res.imputing = self.imputing
        return res

    def _ad_inference(self, samples: List[Dict[str, Any]], model: nn.Module) -> Tuple[float, float, int]:
        X_test, n_tech_anomalies = apply_preprocessing_for_inference(
            samples,
            self.feature_indicies,
            self.ad_imputer,
            self.ad_encoder,
            self.operator,
            self.with_lte,
        )

        if X_test.shape[0] == 0:
            return 0.0, 0.0, n_tech_anomalies

        X_test = scale_test_data(X_test, self.ad_scaler)

        if len(samples) > 1:
            # NOTE: overlap here is always 0 since the overlap is controlled by the self.label_len token parameter
            #       and the seq_len is fixed to the MAX_SEQUENCE_LEN -- the size selected during the training
            #       if self.sequence_len < self.MAX_SEQUENCE_LEN and e.g. self.label_len = 0, the sequence will be padded with zeros
            X_test = pack_into_sequences(X_test, seq_len=self.MAX_SEQUENCE_LEN, overlap=0)
            assert X_test.shape[0] == 1, (
                f"CellAnomalyDetector: seq len should be 1, got {X_test.shape[0]}, "
                "the sequence assumed to be inferred as soon as it is filled"
            )

        _, loss_array, score = evaluate_dae(
            X_test=X_test,
            model=model,
            threshold=model.threshold,
            scaler=self.ad_scaler,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            save_path=None,
            columns=self.feature_indicies.keys(),
        )

        return loss_array.mean() / model.threshold, score, n_tech_anomalies

    def _identify_events(self, cell_reports: Dict[str, Any] | List[Dict[str, Any]]) -> List[str]:
        if self.event_classifier:
            return self.event_classifier.identify_events(cell_reports)
        return []

    def _maybe_load_ad_objects(
        self,
        ad_base_seq_model_path: str | None,
        ad_prob_seq_model_path: str | None,
        ad_point_model_path: str | None,
        ad_imputer_path: str | None,
        ad_encoder_path: str | None,
        ad_scaler_path: str | None,
        ec_baselines_path: str | None,
    ) -> bool:
        tmp_l = [
            ad_base_seq_model_path,
            ad_prob_seq_model_path,
            ad_point_model_path,
            ad_imputer_path,
            ad_encoder_path,
            ad_scaler_path,
        ]

        tmp_l_nones = tmp_l.count(None)
        if tmp_l_nones == len(tmp_l):
            return False
        elif 0 < tmp_l_nones < len(tmp_l) and ad_imputer_path is not None:
            raise ValueError(
                "CellAnomalyDetector:_maybe_load_ad_objects: "
                "all or none of the paths should be provided except imputer"
            )
        else:
            self.ad_base_seq_model = AEFactory.create_ae(ad_base_seq_model_path)
            assert (
                self.ad_base_seq_model.threshold is not None
            ), "CellAnomalyDetector: ad_base_seq_model threshold is None"

            self.ad_prob_seq_model = AEFactory.create_ae(ad_prob_seq_model_path)
            assert (
                self.ad_prob_seq_model.threshold is not None
            ), "CellAnomalyDetector: ad_prob_seq_model threshold is None"

            self.ad_point_model = AEFactory.create_ae(ad_point_model_path)
            assert self.ad_point_model.threshold is not None, "CellAnomalyDetector: ad_point_model threshold is None"

            self.ad_imputer = self._load_pickled_object(ad_imputer_path)
            self.ad_encoder = self._load_pickled_object(ad_encoder_path)
            self.ad_scaler = self._load_pickled_object(ad_scaler_path)

            if ec_baselines_path is not None:
                self.event_classifier = CellEventClassifier(ec_baselines_path)

            return True

    def _load_pickled_object(self, filepath: str) -> Any:
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"CellAnomalyDetector:_load_pickled_object: file {filepath} not found")


class CellController:
    """
    CellController is an endpoint class that is responsible for processing cell feedback reports and propagating the results to the application-based data controller over MQTT.
    It does:
    - Deep Autoencoder-based anomaly detection on cell reports of all given operators and routers via the given configurations
    - Trend detection by thresholding on the certain values of cell reports of given active routers via the given configurations
    - Data class switch detection (LTE <-> NSA <-> SA) on cell reports of given active routers

    NOTE: the routes are hardcoded for Vodafone and Telekom operators. If the `active_routers` argument is not specified, all routers are processed.
    If some new routers are added, the hardcoded router lists as well as mappings in data preprocessing for anomaly detection should be updated.

    :param sub_broker_config: MqttBrokerConfig: Subscriber broker configuration
    :param pub_broker_config: MqttBrokerConfig | None: Publisher broker configuration. If None, no publishing is done
    :param cell_topic: str: cell topic to subscribe
    :param feed_controller_topic: str | None: Feed controller (application layer) topic to publish. If None, no publishing is done
    :param adet_publish_all: bool: Publish all anomaly detection results. Default is False
    :param operators: List[str]: Operators to process cell reports for (vodafone, telekom). Default is ["vodafone", "telekom"]
    :param active_routers: List[str] | None: Active routers to process cell reports for. If None, all routers are processed
    :param adet_seq_same_routers: bool: Anomaly detection sequence is for the same routers only. Default is False
    :param adop_cfgs: List[cellAnomalyDetectorObjectPathsConfig] | None: Configurations of paths to the pickled objects used to load the ad objects.
        One for one operator. Cut if there are more configs than operators. If None, no anomaly detection is performed
    :param thr_cfgs: Dict[str, str] | None: Configurations for thresholders. Consists of `CellReport`'s NUMERICAL keys and trend directions (a, d).
        Trends are published to feed controller topic if threshold is exceeded
    :param sequence_len: int: Length of the cell report sequence for anomaly detection. Default is 8
    :param label_len: int: Length of the start token for anomaly detection. Default is 4
    :param imputing: bool: Impute missing values in the cell reports. Default is True
    :param warmup: float: Warmup time before starting to process cell reports in seconds. Default is 3.0
    :param verbose: bool: Verbosity flag. Default is False
    """

    VODAFONE_ROUTERS = ["5G-D2-WAVELAB", "CAU-8388", "CAU-8395"]
    TELEKOM_ROUTERS = ["5G-DTAG-WAVELAB", "CAU-D4", "CAU-0C"]

    def __init__(
        self,
        sub_broker_config: MqttBrokerConfig,
        pub_broker_config: MqttBrokerConfig | None,
        cell_topic: str,
        feed_controller_topic: str | None = None,
        adet_publish_all: bool = False,
        operators: List[str] = ["vodafone", "telekom"],
        active_routers: List[str] | None = None,
        adet_seq_same_routers: bool = False,
        adop_cfgs: List[CellAnomalyDetectorObjectPathsConfig] | None = None,
        thr_cfgs: Dict[str, ThresholderConfig] | None = None,
        sequence_len: int = 8,
        label_len: int = 4,
        imputing: bool = True,
        warmup: float = 3.0,
        verbose: bool = False,
    ) -> None:
        self.subscriber = CellMqttSubscriber(broker_config=sub_broker_config, cell_topic=cell_topic)
        self.subscriber.start()

        self.cell_topic = cell_topic
        self.subscriber.subscribe([cell_topic])

        if pub_broker_config:
            self.publisher = CellMqttPublisher(broker_config=pub_broker_config)
            self.publisher.start()
            self.feed_controller_topic = feed_controller_topic
        else:
            self.publisher = None
            self.feed_controller_topic = None

        self.adet_publish_all = adet_publish_all if adet_publish_all is not None else False

        self.operators = self._validate_operators(operators)
        # TODO: add mqtt topic for changing active routers (with threaded_wrapper)
        self.active_routers = self._validate_routers(active_routers)
        self.adet_seq_same_routers = adet_seq_same_routers if adet_seq_same_routers is not None else False
        self.sequence_len = sequence_len
        self.label_len = label_len
        self.imputing = imputing

        self.anomaly_detectors = {}
        if adop_cfgs is not None:
            self.anomaly_detectors = self._prepare_anomaly_detectors(adop_cfgs)

        self.thresholders = {}
        if thr_cfgs is not None:
            self.thresholders = self._prepare_thresholders(thr_cfgs)

        self.router_data_classes = {}

        self.warmup = warmup
        self.verbose = verbose

    async def controller_coro(self) -> None:
        await asyncio.sleep(self.warmup)
        self.subscriber.clean_message_queue(self.cell_topic)
        self.is_running = True
        LOGGER.info(f"INFO: CellController's main coroutine is starting...")

        while self.is_running:
            try:
                cell_payload = await self.subscriber.await_message(self.cell_topic, timeout=0.05)
                if cell_payload and cell_payload.tx_packets > 0:
                    self._process_report(cell_payload.to_report(), cell_payload.operator)
            except asyncio.exceptions.CancelledError:
                self.is_running = False

    def _process_report(self, cell_report: CellReport, operator: str) -> None:
        # operator-wise
        if not operator or operator not in self.operators:
            return

        ## anomaly detection
        if self.anomaly_detectors and operator in self.anomaly_detectors:
            res = self.anomaly_detectors[operator].detect(asdict(cell_report))
            if res:
                # if anomaly check if active routers are not none and then if at least one of them belongs to the routers list
                # that corresponds to the payload operator. If active routers are none, publish anyway
                if self.adet_publish_all or (not self.adet_publish_all and res.det_anomaly):
                    if self.active_routers:
                        if self._validate_routers(res.routers, self.active_routers):
                            self._publish_to_feed_controller({"cell_anomaly": {"result": asdict(res)}})
                    else:
                        self._publish_to_feed_controller({"cell_anomaly": {"result": asdict(res)}})

                if self.verbose:
                    LOGGER.info(f"INFO: {res}")

        # router-wise
        if self._validate_routers([cell_report.identity], self.active_routers):
            router = cell_report.identity

            ## data class switch
            if router not in self.router_data_classes:
                self.router_data_classes[router] = cell_report.data_class

            if self.router_data_classes[router] != cell_report.data_class:
                self._publish_to_feed_controller(
                    {
                        "cell_dc_switch": {
                            "router": router,
                            "old": self.router_data_classes[router],
                            "new": cell_report.data_class,
                        }
                    }
                )
                self.router_data_classes[router] = cell_report.data_class
                if self.verbose:
                    LOGGER.info(
                        f"INFO: Data class switch: {self.router_data_classes[router]} -> {cell_report.data_class}"
                    )

            ## thresholding
            if self.thresholders:
                keys = []
                for key, thresholder in self.thresholders[router].items():
                    val = getattr(cell_report, key, None)
                    if val is not None:
                        if thresholder.check_trend(val):
                            keys.append(key)
                            if self.verbose:
                                LOGGER.info(f"INFO: Threshold exceed: {key}, values: {thresholder.values}")
                if keys:
                    self._publish_to_feed_controller({"cell_exceed": {"router": router, "keys": keys}})

    def _publish_to_feed_controller(self, actions: Dict[str, Any]) -> None:
        if self.publisher and self.feed_controller_topic:
            self.publisher.publish(self.feed_controller_topic, json.dumps({"all": actions}), source="cell_controller")

    def _prepare_anomaly_detectors(
        self,
        adop_cfgs: List[CellAnomalyDetectorObjectPathsConfig],
    ) -> Dict[str, CellAnomalyDetector]:
        ads = {
            op: CellAnomalyDetector(
                config=cfg,
                sequence_len=self.sequence_len,
                label_len=self.label_len,
                operator=op,
                active_routers=self._get_active_routers_for_anomaly_detector(op),
                adet_seq_same_routers=self.adet_seq_same_routers,
                with_lte=True,  # FIXME: hardcoded
                imputing=self.imputing,
            )
            for op, cfg in zip(self.operators, adop_cfgs)
            if cfg
        }

        valid_ads = {op: ad for op, ad in ads.items() if ad.is_ad}
        if len(valid_ads) != len(self.operators):
            LOGGER.error(
                f"CellController:_prepare_anomaly_detectors: anomaly detectors for "
                f"{set(self.operators) - set(valid_ads.keys())} are not loaded"
            )

        LOGGER.info(f"CellController:_prepare_anomaly_detectors: anomaly detectors are prepared")
        return valid_ads

    def _prepare_thresholders(self, cfgs: Dict[str, ThresholderConfig]) -> Dict[str, Dict[str, Thresholder]]:
        ts = {}
        for op in self.operators:
            routers = self.VODAFONE_ROUTERS if op == "vodafone" else self.TELEKOM_ROUTERS
            rs = {}
            for f, cfg in cfgs.items():
                if f not in CellReport.__annotations__:
                    raise ValueError(f"CellController:_prepare_thresholders: {f} is not a valid field")
                if not CellReport.__annotations__[f] in [int, float]:
                    raise ValueError(
                        f"CellController:_prepare_thresholders: {f}'s value is not of int or float type "
                        f"but {CellReport.__annotations__[f]}"
                    )
                rs.update({f: Thresholder(**asdict(cfg))})
            if self.active_routers is not None:
                ts.update({r: copy.deepcopy(rs) for r in self.active_routers})
            else:
                ts.update({r: copy.deepcopy(rs) for r in routers})

        LOGGER.info(f"CellController:_prepare_thresholders: thresholders are prepared {ts}")
        return ts

    def _validate_operators(self, operators: List[str]) -> List[str]:
        if not all([op.lower() in ["vodafone", "telekom"] for op in operators]):
            raise ValueError("CellController:_validate_operators: operators should be either 'vodafone' or 'telekom'")
        return [op.lower() for op in operators]

    def _validate_routers(
        self,
        routers: List[str] | str | None = None,
        validation_routers: List[str] | None = None,
    ) -> List[str] | None:
        if isinstance(routers, str):
            if routers != "all_valid":
                raise ValueError(
                    "CellController:_validate_routers: router var should be either 'all_valid' or a list of routers or null"
                )
            else:
                return None
        if routers is not None:
            if validation_routers is not None:
                return [r for r in routers if r in validation_routers]
            else:
                return [r for r in routers if r in self.VODAFONE_ROUTERS + self.TELEKOM_ROUTERS]
        return None

    def _get_active_routers_for_anomaly_detector(self, operator: str) -> List[str] | None:
        if not self.active_routers:
            return None

        routers = []
        for r in self.active_routers:
            if operator == "vodafone" and r in self.VODAFONE_ROUTERS:
                routers.append(r)
            elif operator == "telekom" and r in self.TELEKOM_ROUTERS:
                routers.append(r)
        return routers
