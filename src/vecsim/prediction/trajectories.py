from __future__ import annotations

from typing import Any, Dict

import numpy as np


def synthetic_entry_probability(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Return a synthetic probability that a candidate vehicle will enter the RSU area within horizon.

    This is a placeholder for real trajectory prediction. Implementations can:
      - use distance to intersection,
      - use route intention from SUMO,
      - or simply sample a noisy probability.
    """
    pred = cfg.get("prediction", {}) or {}
    est = pred.get("inflow_estimator", {}) or {}
    params = est.get("params", {}) or {}
    base = float(params.get("base_entry_prob", 0.3))
    noise_std = float(params.get("noise_std", 0.05))
    p = base + float(rng.normal(0.0, noise_std))
    return float(max(0.0, min(1.0, p)))
