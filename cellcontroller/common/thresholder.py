from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class ThresholderConfig:
    max_window_size: int = 8
    k: float = 0.05
    trend_direction: str = "d"
    warmup_iterations: int = 10
    excesses_allowed: int = 1


class Thresholder:
    def __init__(
        self,
        max_window_size: int,
        k: float,
        trend_direction: str,
        warmup_iterations: int = 10,
        excesses_allowed: int = 1,
    ) -> None:
        self.max_window_size = max_window_size
        self.k = k
        self.trend_direction = self._validate_trend_direction(trend_direction)
        self.warmup_iterations = max(0, warmup_iterations)
        self.excesses_allowed = max(1, excesses_allowed)

        self.values: Deque[int | float] = deque(maxlen=max_window_size)

        self.num_iterations = 0
        self.num_excesses = 0

        self.threshold = 0.0

    def check_trend(self, val: int | float) -> bool:
        self.values.append(val)
        weighted_average = self._get_weighted_average()
        self._update_threshold(weighted_average)

        self.num_iterations += 1
        if self.num_iterations <= self.warmup_iterations:
            return False

        is_trend = False
        match self.trend_direction:
            case "i":
                is_trend = weighted_average > self.threshold
            case "d":
                is_trend = weighted_average < -self.threshold

        if is_trend:
            self.num_excesses += 1
            if self.num_excesses >= self.excesses_allowed:
                self.num_excesses = 0
                return True
        else:
            self.num_excesses = 0

        return False

    def _get_weighted_average(self) -> float:
        window_size = min(self.max_window_size, len(self.values))
        if window_size < 2:
            return 0.0
        weighted_sum = 0.0
        for i in range(1, window_size):
            weight = 2**-i
            weighted_sum += weight * (self.values[-i] - self.values[-i - 1])
        return weighted_sum

    def _update_threshold(self, weighted_average: float) -> None:
        self.threshold = self.threshold + self.k * (abs(weighted_average) - self.threshold)

    def _validate_trend_direction(self, td: str) -> str:
        if td not in ["i", "d"]:
            raise ValueError(f"Thresholder:_validate_trend_direction: invalid trend direction {td}")
        return td
