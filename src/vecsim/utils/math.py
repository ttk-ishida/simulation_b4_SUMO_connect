from __future__ import annotations

import math
from typing import Optional


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def inv(x: float, eps: float = 1e-9) -> float:
    return 1.0 / max(x, eps)


def sigmoid_like_rho(rho: float) -> float:
    """Alpha mapping: rho/(1+rho)."""
    rho = max(0.0, rho)
    return rho / (1.0 + rho)
