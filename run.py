import argparse
import os
import asyncio
import yaml

from cellcontroller.controller import CellController, CellAnomalyDetectorObjectPathsConfig
from cellcontroller.common.broker import MqttBrokerConfig
from cellcontroller.common.thresholder import ThresholderConfig

try:
    import uvloop
except ImportError:
    uvloop = None


def get_resource_path(base_path: str, resource_path: str) -> str:
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(base_path)
    path = os.path.join(base_path, resource_path)
    return path


async def main(config_path: str) -> None:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    sub_broker_config = MqttBrokerConfig(
        broker_host=config_data["sub_broker"]["broker_host"],
        broker_port=config_data["sub_broker"]["broker_port"],
        username=config_data["sub_broker"]["username"],
        password=config_data["sub_broker"]["password"],
        is_tls=config_data["sub_broker"]["is_tls"],
        tls_cafile=config_data["sub_broker"]["tls_cafile"],
        protocol=config_data["sub_broker"]["protocol"],
    )

    cell_topic = config_data["topics"]["cell_topic"]

    if "pub_broker" in config_data:
        pub_broker_config = MqttBrokerConfig(
            broker_host=config_data["pub_broker"]["broker_host"],
            broker_port=config_data["pub_broker"]["broker_port"],
            username=config_data["pub_broker"]["username"],
            password=config_data["pub_broker"]["password"],
            is_tls=config_data["pub_broker"]["is_tls"],
            tls_cafile=config_data["pub_broker"]["tls_cafile"],
            protocol=config_data["pub_broker"]["protocol"],
        )
        feed_controller_topic = config_data["topics"]["feed_controller_topic"]
        adet_publish_all = config_data["adet_publish_all"]
    else:
        pub_broker_config = None
        feed_controller_topic = None
        adet_publish_all = False

    operators = config_data["operators"]
    active_routers = config_data["active_routers"]
    adet_seq_same_routers = config_data["adet_seq_same_routers"] or False

    resource_path = config_data.get("resource_path", os.getcwd())

    # NOTE: Inner paths to anomaly detector models/imputers/scalers/baselines are hardcoded
    ad_cfg_vodafone = CellAnomalyDetectorObjectPathsConfig(
        ad_base_seq_model_path=get_resource_path(resource_path, "vodafone/base_seq_model.pt"),
        ad_prob_seq_model_path=get_resource_path(resource_path, "vodafone/prob_seq_model.pt"),
        ad_point_model_path=get_resource_path(resource_path, "vodafone/point_model.pt"),
        ad_imputer_path=get_resource_path(resource_path, "vodafone/imputer.pkl"),
        ad_encoder_path=get_resource_path(resource_path, "vodafone/encoder.pkl"),
        ad_scaler_path=get_resource_path(resource_path, "vodafone/scaler.pkl"),
        ec_baselines_path=get_resource_path(resource_path, "event_classifier/baselines.pkl"),
    )

    ad_cfg_telekom = CellAnomalyDetectorObjectPathsConfig(
        ad_base_seq_model_path=get_resource_path(resource_path, "telekom/base_seq_model.pt"),
        ad_prob_seq_model_path=get_resource_path(resource_path, "telekom/prob_seq_model.pt"),
        ad_point_model_path=get_resource_path(resource_path, "telekom/point_model.pt"),
        ad_imputer_path=get_resource_path(resource_path, "telekom/imputer.pkl"),
        ad_encoder_path=get_resource_path(resource_path, "telekom/encoder.pkl"),
        ad_scaler_path=get_resource_path(resource_path, "telekom/scaler.pkl"),
        ec_baselines_path=None,
    )

    thr_cfgs = {}
    for key, cfg in config_data["thresholders"].items():
        thr_cfgs[key] = ThresholderConfig(
            max_window_size=cfg["window_size"],
            k=cfg["k"],
            trend_direction=cfg["trend_direction"],
            warmup_iterations=cfg["warmup_iterations"],
            excesses_allowed=cfg["excesses_allowed"],
        )

    controller = CellController(
        sub_broker_config=sub_broker_config,
        pub_broker_config=pub_broker_config,
        cell_topic=cell_topic,
        feed_controller_topic=feed_controller_topic,
        adet_publish_all=adet_publish_all,
        active_routers=active_routers,
        adet_seq_same_routers=adet_seq_same_routers,
        operators=operators,
        adop_cfgs=[ad_cfg_vodafone, ad_cfg_telekom],
        thr_cfgs=thr_cfgs,
        sequence_len=config_data["sequence_len"],
        label_len=config_data["label_len"],
        imputing=config_data.get("imputing", False),
        warmup=config_data.get("warmup", 0),
        verbose=config_data.get("verbose", False),
    )

    await controller.controller_coro()


if __name__ == "__main__":
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-c, --config", dest='config_path', type=str, required=True, help="path to the yaml configuration file")
    # fmt: on
    args = parser.parse_args()

    try:
        asyncio.run(main(args.config_path))
    except KeyboardInterrupt:
        exit(0)
