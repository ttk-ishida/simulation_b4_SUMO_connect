from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..models.rsu import RSU
from ..models.vehicle import Vehicle
from .channel import sinr_linear


def uplink_rate_bps(
    uplink_cfg: Dict[str, Any],
    vehicle: Vehicle,
    rsu: RSU,
    rng: np.random.Generator,
) -> float:
    """Compute uplink rate [bit/s]."""
    rate_model = uplink_cfg.get("rate_model", "fixed")
    if rate_model == "fixed":
        return float(uplink_cfg.get("fixed_rate_bps", 2e6))

    if rate_model == "shannon":
        sh = uplink_cfg.get("shannon", {}) or {}
        bw = float(sh.get("bandwidth_hz", 10e6))
        sinr = sinr_linear(rsu.distance_m(vehicle.x_m, vehicle.y_m), sh, rng)
        # Shannon capacity: C = B log2(1+SINR)
        return float(bw * math.log2(1.0 + max(sinr, 0.0)))

    raise ValueError(f"Unknown uplink rate_model: {rate_model}")


def uplink_time_s(input_bits: float, rate_bps: float) -> float:
    if rate_bps <= 0:
        return float("inf")
    return float(input_bits / rate_bps)


def compute_uplink_time_for_task(
    cfg: Dict[str, Any],
    vehicle: Vehicle,
    rsu: RSU,
    input_bits: float,
    rng: np.random.Generator,
) -> float:
    uplink_cfg = (cfg.get("comm", {}) or {}).get("uplink", {}) or {}
    if not bool(uplink_cfg.get("enabled", True)):
        return 0.0
    rate = uplink_rate_bps(uplink_cfg, vehicle, rsu, rng)
    return uplink_time_s(input_bits, rate)
