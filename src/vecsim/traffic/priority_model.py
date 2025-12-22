from __future__ import annotations

from typing import Any, Dict

import numpy as np


def sample_priority(model_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    """Sample continuous priority P in [0,1]."""
    name = model_cfg.get("name", "uniform")
    params = model_cfg.get("params", {}) or {}

    if name == "uniform":
        return float(rng.random())

    if name == "beta":
        a = float(params.get("a", 2.0))
        b = float(params.get("b", 5.0))
        return float(rng.beta(a, b))

    if name == "two_point":
        # P is either low or high with probability p_high
        p_high = float(params.get("p_high", 0.2))
        low = float(params.get("low", 0.2))
        high = float(params.get("high", 0.9))
        return high if (rng.random() < p_high) else low

    raise ValueError(f"Unknown priority model: {name}")
