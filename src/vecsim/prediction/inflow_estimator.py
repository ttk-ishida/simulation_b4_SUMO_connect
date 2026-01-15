from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import csv
import functools
from bisect import bisect_right

import numpy as np

from .trajectories import synthetic_entry_probability


def _profile_segments(cfg: Dict[str, Any]) -> List[dict[str, Any]]:
    pred = cfg.get("prediction", {}) or {}
    profile = pred.get("inflow_profile", {}) or {}
    segments = profile.get("segments", []) or []
    return list(segments)


def expected_inflow_from_profile(
    now_s: float,
    horizon_s: float,
    segments: List[dict[str, Any]],
) -> float:
    if horizon_s <= 0.0:
        return 0.0
    if not segments:
        return 0.0
    t0 = float(now_s)
    t1 = float(now_s + horizon_s)
    exp = 0.0
    for seg in segments:
        s = float(seg.get("t_start_s", 0.0))
        e = float(seg.get("t_end_s", 0.0))
        rate = float(seg.get("rate_veh_per_s", 0.0))
        if e <= s or rate <= 0.0:
            continue
        overlap = max(0.0, min(t1, e) - max(t0, s))
        exp += overlap * rate
    return float(exp)


def expected_inflow_vehicles(
    candidate_vehicle_ids: Iterable[str],
    now_s: float,
    horizon_s: float,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Compute expected number of vehicles entering within horizon."""
    pred = cfg.get("prediction", {}) or {}
    est = pred.get("inflow_estimator", {}) or {}
    name = str(est.get("name", "synthetic_from_edges"))

    if name == "from_timeseries":
        ts_cfg = pred.get("inflow_timeseries", {}) or {}
        path = str(ts_cfg.get("path", ""))
        if not path:
            return 0.0
        times, values = _load_timeseries(path)
        if not times:
            return 0.0
        idx = max(0, bisect_right(times, float(now_s)) - 1)
        return float(values[idx])

    if name == "from_profile":
        segments = _profile_segments(cfg)
        return expected_inflow_from_profile(now_s=now_s, horizon_s=horizon_s, segments=segments)

    s = 0.0
    for _vid in candidate_vehicle_ids:
        s += synthetic_entry_probability(cfg, rng)
    return float(s)


@functools.lru_cache(maxsize=64)
def _load_timeseries(path: str) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    values: List[float] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row.get("t_s", "0"))
                    v = float(row.get("n_true", "0"))
                except ValueError:
                    continue
                times.append(t)
                values.append(v)
    except FileNotFoundError:
        return [], []
    return times, values


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


def apply_inflow_error(
    n_true: float,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Apply relative error to N_true to generate N_hat."""
    pred = cfg.get("prediction", {}) or {}
    err_cfg = pred.get("error", {}) or {}
    sigma = float(err_cfg.get("sigma", 0.0))
    dist = str(err_cfg.get("dist", "uniform"))
    if sigma <= 0.0:
        return float(max(0.0, n_true))

    if dist == "normal":
        e = float(rng.normal(0.0, sigma))
        e = max(-sigma, min(sigma, e))
    else:
        e = float(rng.uniform(-sigma, sigma))
    return float(max(0.0, n_true * (1.0 + e)))
