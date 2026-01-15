from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from .rsu import RSU


@dataclass
class Vehicle:
    veh_id: str
    x_m: float = 0.0
    y_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0

    in_coverage: bool = False
    coverage_enter_t: Optional[float] = None
    last_seen_t: Optional[float] = None

    def update_state(self, x_m: float, y_m: float, speed_mps: float, heading_deg: float, now: float, rsu: RSU) -> None:
        self.x_m = x_m
        self.y_m = y_m
        self.speed_mps = speed_mps
        self.heading_deg = heading_deg
        self.last_seen_t = now

        inside = rsu.in_coverage(x_m, y_m)
        if inside and not self.in_coverage:
            self.in_coverage = True
            self.coverage_enter_t = now
        elif (not inside) and self.in_coverage:
            self.in_coverage = False

    def estimate_remaining_stay_time(self, rsu: RSU, cfg: dict[str, Any]) -> float:
        """Estimate remaining coverage time [s] using straight-line approximation."""
        rsu_cfg = cfg.get("rsu", {}) or {}
        min_speed = float(rsu_cfg.get("min_speed_mps", 0.1))
        eps = float(rsu_cfg.get("speed_eps_mps", 1e-3))
        speed = max(float(self.speed_mps), min_speed)

        rect = rsu.coverage_rect_bounds()
        if rect is not None:
            x_min, x_max, y_min, y_max = rect

            # Determine dominant axis based on heading (SUMO: 0=East, 90=North).
            theta = math.radians(float(self.heading_deg))
            dx = math.cos(theta)
            dy = math.sin(theta)

            if abs(dx) >= abs(dy):
                d_out = (x_max - self.x_m) if dx >= 0 else (self.x_m - x_min)
            else:
                d_out = (y_max - self.y_m) if dy >= 0 else (self.y_m - y_min)

            d_out = max(0.0, float(d_out))
            return float(d_out / (speed + eps))

        # Circular coverage: ray-circle intersection along heading.
        theta = math.radians(float(self.heading_deg))
        dx = math.cos(theta)
        dy = math.sin(theta)
        fx = self.x_m - rsu.x_m
        fy = self.y_m - rsu.y_m

        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - (rsu.coverage_radius_m ** 2)
        disc = b * b - c
        if disc <= 0.0:
            return 0.0
        t_out = -b + math.sqrt(disc)
        if t_out <= 0.0:
            return 0.0
        return float(t_out / (speed + eps))
