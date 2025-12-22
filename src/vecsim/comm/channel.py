from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pathloss import pathloss_db


def received_power_w(tx_power_w: float, pl_db: float) -> float:
    # Rx = Tx * 10^(-PL/10)
    return float(tx_power_w) * (10 ** (-pl_db / 10.0))


def sinr_linear(distance_m: float, shannon_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    """Compute a simple SINR (no interference) as Rx/noise."""
    tx_power_w = float(shannon_cfg.get("tx_power_w", 0.2))
    noise_power_w = float(shannon_cfg.get("noise_power_w", 1e-9))
    pl_cfg = shannon_cfg.get("pathloss", {"name": "log_distance", "params": {}})
    pl_db = pathloss_db(distance_m, pl_cfg, rng)
    rx = received_power_w(tx_power_w, pl_db)
    return float(rx / max(noise_power_w, 1e-30))
