from dataclasses import dataclass, fields
from typing import Any, Dict, Self

from cellcontroller.common.utils import LOGGER


@dataclass
class MqttBrokerConfig:
    broker_host: str = "0.0.0.0"
    broker_port: int = 1883
    keepalive: int = 20
    username: str | None = None
    password: str | None = None
    is_tls: bool = False
    tls_cafile: str | None = None
    protocol: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Self:
        for key in config_dict.keys():
            if key not in {field.name: field.type for field in fields(cls)}:
                LOGGER.warning(f"MqttBrokerConfig.from_dict: invalid field name: {key}")
                continue

        return cls(
            broker_host=config_dict.get('broker_host', cls.broker_host),
            broker_port=config_dict.get('broker_port', cls.broker_port),
            keepalive=config_dict.get('keepalive', cls.keepalive),
            username=config_dict.get('username', cls.username),
            password=config_dict.get('password', cls.password),
            is_tls=config_dict.get('is_tls', cls.is_tls),
            tls_cafile=config_dict.get('tls_cafile', cls.tls_cafile),
            protocol=config_dict.get('protocol', cls.protocol),
        )
