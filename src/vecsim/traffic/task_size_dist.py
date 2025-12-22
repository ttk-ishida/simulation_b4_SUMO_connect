from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_cycles(dist_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    """Sample required CPU cycles based on config."""
    name = dist_cfg.get("name", "lognormal")
    params = dist_cfg.get("params", {}) or {}

    if name == "uniform":
        lo = float(params.get("low_cycles", 1e8))
        hi = float(params.get("high_cycles", 1e9))
        return float(rng.uniform(lo, hi))

    if name == "lognormal":
        mean = float(params.get("mean_cycles", 5e8))
        sigma = float(params.get("sigma", 0.6))
        lo = float(params.get("min_cycles", 1e7))
        hi = float(params.get("max_cycles", 1e10))
        # Convert mean to lognormal parameters approximately (treat mean as exp(mu))
        x = float(rng.lognormal(mean=math.log(max(mean, 1e-9)), sigma=sigma))
        return _clip(x, lo, hi)

    if name == "mixture_lognormal":
        w_light = float(params.get("w_light", 0.8))
        light = params.get("light", {}) or {}
        heavy = params.get("heavy", {}) or {}
        use_light = rng.random() < w_light
        comp = light if use_light else heavy
        sub = {"name": "lognormal", "params": {
            "mean_cycles": float(comp.get("mean_cycles", 5e8)),
            "sigma": float(comp.get("sigma", 0.6)),
            "min_cycles": float(comp.get("min_cycles", 1e7)),
            "max_cycles": float(comp.get("max_cycles", 1e10)),
        }}
        return sample_cycles(sub, rng)

    raise ValueError(f"Unknown cycles distribution: {name}")


def expected_cycles(dist_cfg: Dict[str, Any]) -> float:
    """Compute E[cycles] for supported distributions.

    For clipped distributions, we return an approximate mean (ignoring clipping),
    which is typically sufficient for alpha(t) scaling.
    """
    name = dist_cfg.get("name", "lognormal")
    params = dist_cfg.get("params", {}) or {}

    if name == "uniform":
        lo = float(params.get("low_cycles", 1e8))
        hi = float(params.get("high_cycles", 1e9))
        return 0.5 * (lo + hi)

    if name == "lognormal":
        mean = float(params.get("mean_cycles", 5e8))
        sigma = float(params.get("sigma", 0.6))
        # Treat given mean_cycles as exp(mu) (approx); then E[X]=exp(mu+sigma^2/2)
        mu = math.log(max(mean, 1e-9))
        return float(math.exp(mu + 0.5 * sigma * sigma))

    if name == "mixture_lognormal":
        w_light = float(params.get("w_light", 0.8))
        light = params.get("light", {}) or {}
        heavy = params.get("heavy", {}) or {}
        e_light = expected_cycles({"name": "lognormal", "params": {
            "mean_cycles": float(light.get("mean_cycles", 5e8)),
            "sigma": float(light.get("sigma", 0.5)),
        }})
        e_heavy = expected_cycles({"name": "lognormal", "params": {
            "mean_cycles": float(heavy.get("mean_cycles", 2e9)),
            "sigma": float(heavy.get("sigma", 0.6)),
        }})
        return float(w_light * e_light + (1.0 - w_light) * e_heavy)

    raise ValueError(f"Unknown cycles distribution: {name}")
