from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from .trajectories import synthetic_entry_probability


def expected_inflow_vehicles(
    candidate_vehicle_ids: Iterable[str],
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Compute expected number of vehicles entering within horizon (placeholder)."""
    s = 0.0
    for _vid in candidate_vehicle_ids:
        s += synthetic_entry_probability(cfg, rng)
    return float(s)


def lambda_task_hat(
    expected_inflow: float,
    horizon_s: float,
    mu_tasks_per_s_per_vehicle: float,
) -> float:
    """Estimate future task arrival rate [tasks/s] from expected inflow and per-vehicle task rate."""
    if horizon_s <= 0:
        return 0.0
    lambda_veh = expected_inflow / horizon_s
    return float(mu_tasks_per_s_per_vehicle * lambda_veh)
