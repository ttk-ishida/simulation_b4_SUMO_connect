from __future__ import annotations

from typing import Any, Dict

from ..traffic.task_size_dist import expected_cycles


def expected_compute_time_s(cfg: Dict[str, Any]) -> float:
    cycles_cfg = cfg.get("cycles_dist", {}) or {}
    mec = cfg.get("mec", {}) or {}
    f = float(mec.get("cpu_freq_hz", 3e9))
    e_cycles = float(expected_cycles(cycles_cfg))
    return float(e_cycles / f)


def rho_future(lambda_task_hat_per_s: float, e_compute_time_s: float) -> float:
    """rho_future = lambda_hat_task * E[C]."""
    return float(max(0.0, lambda_task_hat_per_s) * max(0.0, e_compute_time_s))


def alpha_from_rho(rho: float) -> float:
    """alpha = rho / (1 + rho), mapping to (0,1)."""
    rho = max(0.0, float(rho))
    return float(rho / (1.0 + rho))
