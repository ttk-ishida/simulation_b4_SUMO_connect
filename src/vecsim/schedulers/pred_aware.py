from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task
from ..utils.math import clip


@dataclass
class PredAwareScheduler:
    name: str = "pred_aware"
    eps: float = 1e-9

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        alpha = clip(float(ctx.alpha), 0.0, 1.0)

        max_r = 0.0
        for t in tasks:
            r = max(0.0, float(t.effective_deadline_t - ctx.now))
            if r > max_r:
                max_r = r

        def score_and_keys(t: Task) -> tuple[float, float, float]:
            R = max(0.0, float(t.effective_deadline_t - ctx.now))
            C = float(t.compute_time_s)
            F_dense = clip(float(t.priority), 0.0, 1.0)
            # E-EDF side: smaller effective remaining time -> larger score
            F_sparse = 1.0 - (R / (max_r + self.eps))
            score = alpha * F_dense + (1.0 - alpha) * F_sparse
            # Tie-breaker: smaller R and smaller C first
            return (score, R, C)

        def sort_key(t: Task) -> tuple[float, float, float]:
            score, R, C = score_and_keys(t)
            return (-score, R, C)

        return sorted(tasks, key=sort_key)
