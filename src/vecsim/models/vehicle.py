from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .rsu import RSU


@dataclass
class Vehicle:
    veh_id: str
    x_m: float = 0.0
    y_m: float = 0.0
    speed_mps: float = 0.0

    in_coverage: bool = False
    coverage_enter_t: Optional[float] = None
    last_seen_t: Optional[float] = None

    def update_state(self, x_m: float, y_m: float, speed_mps: float, now: float, rsu: RSU) -> None:
        self.x_m = x_m
        self.y_m = y_m
        self.speed_mps = speed_mps
        self.last_seen_t = now

        inside = rsu.in_coverage(x_m, y_m)
        if inside and not self.in_coverage:
            self.in_coverage = True
            self.coverage_enter_t = now
        elif (not inside) and self.in_coverage:
            self.in_coverage = False

    def estimate_remaining_stay_time(self, now: float, default_s: float = 5.0) -> float:
        """Stub: remaining coverage time estimate [s].

        You will later replace this with a trajectory-based estimator.
        For now we return a constant default.
        """
        return default_s
