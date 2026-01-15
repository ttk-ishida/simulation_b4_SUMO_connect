from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class RSU:
    rsu_id: str
    x_m: float
    y_m: float
    coverage_radius_m: float
    coverage_half_width_m: Optional[float] = None
    coverage_half_height_m: Optional[float] = None

    def distance_m(self, x_m: float, y_m: float) -> float:
        return math.hypot(x_m - self.x_m, y_m - self.y_m)

    def in_coverage(self, x_m: float, y_m: float) -> bool:
        if self.coverage_half_width_m is not None and self.coverage_half_height_m is not None:
            return (
                (self.x_m - self.coverage_half_width_m) <= x_m <= (self.x_m + self.coverage_half_width_m)
                and (self.y_m - self.coverage_half_height_m) <= y_m <= (self.y_m + self.coverage_half_height_m)
            )
        return self.distance_m(x_m, y_m) <= self.coverage_radius_m

    def coverage_rect_bounds(self) -> Optional[tuple[float, float, float, float]]:
        if self.coverage_half_width_m is None or self.coverage_half_height_m is None:
            return None
        return (
            self.x_m - self.coverage_half_width_m,
            self.x_m + self.coverage_half_width_m,
            self.y_m - self.coverage_half_height_m,
            self.y_m + self.coverage_half_height_m,
        )
