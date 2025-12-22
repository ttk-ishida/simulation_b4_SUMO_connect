from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def pathloss_db(distance_m: float, model_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    """Compute pathloss in dB for a given distance (simple models)."""
    name = model_cfg.get("name", "log_distance")
    params = model_cfg.get("params", {}) or {}
    d = max(float(distance_m), 1e-3)

    if name == "log_distance":
        pl0_db = float(params.get("pl0_db", 60.0))
        d0 = float(params.get("d0_m", 1.0))
        n = float(params.get("n", 2.7))
        sigma = float(params.get("shadow_sigma_db", 0.0))
        shadow = float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0
        return pl0_db + 10.0 * n * math.log10(d / max(d0, 1e-3)) + shadow

    raise ValueError(f"Unknown pathloss model: {name}")


def db_to_linear(db: float) -> float:
    return 10 ** (db / 10.0)


def linear_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))
