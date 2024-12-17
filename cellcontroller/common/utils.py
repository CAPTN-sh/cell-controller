import logging
from paho.mqtt.client import MQTTProtocolVersion
import pickle
import time
from typing import Any, Callable


# logger
class LoggerMaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
info_handler.addFilter(LoggerMaxLevelFilter(logging.WARNING))

error_handler = logging.StreamHandler()
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s - %(filename)s:%(lineno)d - %(funcName)s()'))

LOGGER.addHandler(info_handler)
LOGGER.addHandler(error_handler)


def sleep_until_condition_with_intervals(
    num_intervals: int,
    sleeping_time_sec: float,
    condition_func: Callable[[], bool],
) -> bool:
    """
    Sleep until condition_func returns True or num_intervals is reached

    :param num_intervals: number of intervals
    :param sleeping_time_sec: sleeping time in seconds
    :param condition_func: callable that returns bool
    :return: True if condition_func returned True before num_intervals is reached, False otherwise
    """
    tick_interval_sec = sleeping_time_sec / num_intervals
    for _ in range(num_intervals):
        time.sleep(tick_interval_sec)
        if condition_func():
            return True
    return False


# MQTT
def int_to_mqtt_protocol(protocol_int: int) -> MQTTProtocolVersion:
    match protocol_int:
        case 3:
            return MQTTProtocolVersion.MQTTv31
        case 4:
            return MQTTProtocolVersion.MQTTv311
        case 5:
            return MQTTProtocolVersion.MQTTv5
        case _:
            return MQTTProtocolVersion.MQTTv311
