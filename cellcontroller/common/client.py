from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field, fields
from datetime import datetime
import json
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion, MQTTErrorCode
import secrets
from typing import Any, Dict, List, Self

from cellcontroller.common.broker import MqttBrokerConfig
from cellcontroller.common.utils import LOGGER, int_to_mqtt_protocol, sleep_until_condition_with_intervals


@dataclass
class MqttGstWebrtcAppTopics:
    gcc: str = "gstwebrtcapp/gcc"
    stats: str = "gstwebrtcapp/stats"
    state: str = "gstwebrtcapp/state"
    actions: str = "gstwebrtcapp/actions"
    controller: str = ""

    def nullify(self) -> None:
        for f in fields(self):
            setattr(self, f.name, "")


@dataclass
class MqttExternalEstimationTopics:
    bandwidth: str = ""
    rtt: str = ""


@dataclass
class MqttConfig:
    id: str = ""
    broker_host: str = "0.0.0.0"
    broker_port: int = 1883
    keepalive: int = 20
    username: str | None = None
    password: str | None = None
    is_tls: bool = False
    tls_cafile: str | None = None
    protocol: int = 4
    topics: MqttGstWebrtcAppTopics = field(default_factory=lambda: MqttGstWebrtcAppTopics)
    external_topics: MqttExternalEstimationTopics | None = None


def make_inactive_mqtt_config(
    broker_config: MqttBrokerConfig,
    id: str,
    controller_topic: str | None = None,
) -> MqttConfig:
    return MqttConfig(
        id=id,
        broker_host=broker_config.broker_host,
        broker_port=broker_config.broker_port,
        keepalive=broker_config.keepalive,
        username=broker_config.username,
        password=broker_config.password,
        is_tls=broker_config.is_tls,
        protocol=broker_config.protocol,
        topics=MqttGstWebrtcAppTopics(
            gcc="",
            stats="",
            state="",
            actions="",
            controller=controller_topic or "",
        ),
        external_topics=None,
    )


@dataclass
class MqttMessage:
    timestamp: str
    id: str
    msg: str
    source: str
    topic: str

    @classmethod
    def from_payload(cls, payload: Any, topic: str) -> Self:
        fnames = [f.name for f in fields(cls) if f.name != 'topic']
        if isinstance(payload, dict) and all(k in payload for k in fnames):
            # a proper call by publisher
            return cls(**{k: v for k, v in payload.items() if k in fnames}, topic=topic)
        else:
            # unknown message format, put as is
            return cls(
                timestamp=datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3],
                id="unknown",
                msg=payload,
                source="unknown",
                topic=topic,
            )


class MqttClient(ABC):
    @abstractmethod
    def __init__(self, config: MqttConfig = MqttConfig("")) -> None:
        self.id = config.id + ("_" if config.id else "") + secrets.token_hex(4)
        self.id_init = config.id
        self.broker_host = config.broker_host
        self.broker_port = config.broker_port
        self.keepalive = config.keepalive
        self.username = config.username
        self.password = config.password
        self.is_tls = config.is_tls
        self.tls_cafile = config.tls_cafile
        self.protocol = int_to_mqtt_protocol(config.protocol)
        self.topics = config.topics
        self.external_topics = config.external_topics

        self.message_queue = None
        self.client = None

    def start(self, connection_timeout: float = 5.0) -> None:
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=self.id,
            protocol=self.protocol,
        )

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        if self.is_tls:
            self.client.tls_set(self.tls_cafile)

        try:
            ec = self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            if ec != MQTTErrorCode.MQTT_ERR_SUCCESS:
                raise Exception(f"error code: {ec}")
        except Exception as e:
            LOGGER.error(f"ERROR: MQTT client {self.id} failed to connect to the broker, reason: {e}")
            return

        if self.client.loop_start() != MQTTErrorCode.MQTT_ERR_SUCCESS:
            raise Exception(f"ERROR: MQTT client {self.id} failed to start the loop")

        if not sleep_until_condition_with_intervals(100, connection_timeout, lambda: self.client.is_connected()):
            raise Exception(f"ERROR: MQTT client {self.id} has not been connected after {connection_timeout} seconds")
        else:
            LOGGER.info(f"OK: MQTT client {self.id} has been started")

    def stop(self) -> None:
        _ = self.client.loop_stop()
        if self.client.is_connected():
            _ = self.client.disconnect()
        LOGGER.info(f"INFO: MQTT client {self.id} has been stopped")


class MqttPublisher(MqttClient):
    def __init__(
        self,
        config: MqttConfig = MqttConfig(""),
    ) -> None:
        super().__init__(config)

    def publish(self, topic: str, msg: str, id: str = "", source: str = "") -> None:
        if not self.client.is_connected():
            raise Exception(f"ERROR: MQTT publisher {self.id} is not connected to the broker")
        self.client.publish(
            topic,
            json.dumps(
                {
                    'timestamp': datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3],
                    'id': id or self.id,
                    'msg': msg,
                    'source': source,
                }
            ),
        )
        LOGGER.debug(f"INFO: MQTT publisher {self.id} has published message: {msg} to {topic}")


class MqttSubscriber(MqttClient):
    def __init__(
        self,
        config: MqttConfig = MqttConfig(""),
    ) -> None:
        super().__init__(config)

        self.message_queues: Dict[str, asyncio.Queue] = {}
        for f in fields(self.topics):
            topic = getattr(self.topics, f.name, "")
            if topic:
                self.message_queues[topic] = asyncio.Queue()
        if self.external_topics is not None:
            for f in fields(self.external_topics):
                ext_topic = getattr(self.external_topics, f.name, "")
                if ext_topic:
                    self.message_queues[ext_topic] = asyncio.Queue()

        self.subscriptions = []

    def on_message(self, _, __, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode('utf8'))
        except json.JSONDecodeError:
            return
        topic = msg.topic
        mqtt_message = MqttMessage.from_payload(payload, topic)
        if topic not in self.message_queues:
            self.message_queues[topic] = asyncio.Queue()
        self.message_queues[topic].put_nowait(mqtt_message)
        LOGGER.debug(f"INFO: MQTT subscriber {self.id} has received message: {payload} from topic {topic}")

    def subscribe(self, topics: List[str], qos: int = 1) -> None:
        if not self.client.is_connected():
            raise Exception(f"ERROR: MQTT subscriber {self.id} is not connected to the broker")
        for topic in topics:
            if topic:
                self.client.subscribe(topic, qos=qos)
                self.client.on_message = self.on_message
                self._add_subscription(topic)
                LOGGER.info(f"OK: MQTT subscriber {self.id} has successfully subscribed to {topic}")

    def _add_subscription(self, topic: str) -> None:
        if topic not in self.subscriptions:
            self.message_queues[topic] = asyncio.Queue()
            self.subscriptions.append(topic)

    def unsubscribe(self, topics: List[str]) -> None:
        for topic in topics:
            if topic in self.subscriptions:
                self.client.unsubscribe(topic)
                _ = self.message_queues.pop(topic, None)
                self.subscriptions.remove(topic)

    def get_message(self, topic: str) -> MqttMessage | None:
        queue = self.message_queues.get(topic, None)
        if queue is None:
            LOGGER.error(f"ERROR: No message queue for topic {topic}")
            return None
        if self.client.is_connected() and not queue.empty():
            return queue.get_nowait()
        return None

    async def await_message(self, topic: str, timeout: float | None = None) -> MqttMessage:
        queue = self.message_queues.get(topic, None)
        if queue is None:
            LOGGER.error(f"ERROR: No message queue for topic {topic}")
            return None
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.message_queues[topic].get(), timeout)
            else:
                return await self.message_queues[topic].get()
        except asyncio.TimeoutError:
            return None

    def clean_message_queue(self, topic: str) -> None:
        queue = self.message_queues.get(topic, None)
        if queue is None:
            return
        while not queue.empty():
            _ = queue.get_nowait()

    def stop(self) -> None:
        self.client.unsubscribe(self.subscriptions)
        for topic in self.subscriptions:
            self.clean_message_queue(topic)
        self.message_queues.clear()
        self.subscriptions.clear()
        super().stop()


@dataclass
class MqttPair:
    publisher: MqttPublisher
    subscriber: MqttSubscriber
