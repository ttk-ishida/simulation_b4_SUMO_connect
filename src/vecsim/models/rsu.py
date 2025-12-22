from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RSU:
    rsu_id: str
    x_m: float
    y_m: float
    coverage_radius_m: float

    def distance_m(self, x_m: float, y_m: float) -> float:
        return math.hypot(x_m - self.x_m, y_m - self.y_m)

    def in_coverage(self, x_m: float, y_m: float) -> bool:
        return self.distance_m(x_m, y_m) <= self.coverage_radius_m
