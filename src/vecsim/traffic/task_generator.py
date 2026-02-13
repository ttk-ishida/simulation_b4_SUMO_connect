from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..models.task import Task
from .task_size_dist import sample_cycles
from .priority_model import sample_priority


def poisson_event(rate_per_s: float, dt_s: float, rng: np.random.Generator) -> bool:
    """Return True if an event occurs within dt for a Poisson process."""
    if rate_per_s <= 0 or dt_s <= 0:
        return False
    p = 1.0 - math.exp(-rate_per_s * dt_s)
    return bool(rng.random() < p)


def sample_input_bits(dist_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    name = dist_cfg.get("name", "lognormal")
    params = dist_cfg.get("params", {}) or {}
    if name == "uniform":
        lo = float(params.get("low_bits", 1e5))
        hi = float(params.get("high_bits", 1e7))
        return float(rng.uniform(lo, hi))
    if name == "lognormal":
        mean = float(params.get("mean_bits", 2e6))
        sigma = float(params.get("sigma", 0.6))
        lo = float(params.get("min_bits", 1e5))
        hi = float(params.get("max_bits", 2e7))
        x = float(rng.lognormal(mean=math.log(max(mean, 1e-9)), sigma=sigma))
        return max(lo, min(hi, x))
    raise ValueError(f"Unknown input_bits_dist: {name}")


def sample_relative_deadline(deadline_cfg: Dict[str, Any], rng: np.random.Generator) -> float:
    name = deadline_cfg.get("name", "uniform")
    params = deadline_cfg.get("params", {}) or {}
    if name == "uniform":
        lo = float(params.get("low_s", 0.5))
        hi = float(params.get("high_s", 3.0))
        return float(rng.uniform(lo, hi))
    if name == "constant":
        return float(params.get("value_s", 1.0))
    raise ValueError(f"Unknown deadline_dist: {name}")


@dataclass
class TaskFactory:
    """Creates tasks based on config."""
    cfg: Dict[str, Any]
    rng: np.random.Generator
    next_id: int = 0

    def make_task(
        self,
        vehicle_id: str,
        now: float,
        predicted_stay_s: float,
        mec_cpu_freq_hz: float,
    ) -> Task:
        self.next_id += 1
        task_id = f"t{self.next_id:08d}"

        pr_cfg = (self.cfg.get("priority", {}) or {}).get("model", {"name": "uniform"})
        cycles_cfg = self.cfg.get("cycles_dist", {}) or {}
        tg = self.cfg.get("task_generation", {}) or {}

        input_bits = sample_input_bits(tg.get("input_bits_dist", {}), self.rng)
        rel_deadline = sample_relative_deadline(tg.get("deadline_dist", {}), self.rng)
        priority = sample_priority(pr_cfg, self.rng)
        cycles = sample_cycles(cycles_cfg, self.rng)

        deadline_t = now + rel_deadline
        stay_deadline_t = now + float(predicted_stay_s)
        eff_deadline_t = min(deadline_t, stay_deadline_t)
        # eff_deadline_t = deadline_t


        compute_time_s = float(cycles / mec_cpu_freq_hz)

        return Task(
            task_id=task_id,
            vehicle_id=vehicle_id,
            created_t=now,
            relative_deadline_s=float(rel_deadline),
            deadline_t=deadline_t,
            stay_deadline_t=stay_deadline_t,
            effective_deadline_t=eff_deadline_t,
            priority=float(priority),
            input_bits=float(input_bits),
            cycles=float(cycles),
            compute_time_s=compute_time_s,
        )
