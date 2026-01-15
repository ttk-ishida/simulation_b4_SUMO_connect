from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task
from ..utils.math import clip


@dataclass
class PredictionAwareScheduler:
    """Pred-aware scoring without tie-breaks (score only)."""
    name: str = "prediction_aware"
    eps: float = 1e-9

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        alpha = clip(float(ctx.alpha), 0.0, 1.0)
        max_r = 0.0
        for t in tasks:
            r = max(0.0, float(t.effective_deadline_t - ctx.now))
            if r > max_r:
                max_r = r

        def score(t: Task) -> float:
            R = max(0.0, float(t.effective_deadline_t - ctx.now))
            F_dense = clip(float(t.priority), 0.0, 1.0)
            # E-EDF side: smaller effective remaining time -> larger score
            F_sparse = 1.0 - (R / (max_r + self.eps))
            return alpha * F_dense + (1.0 - alpha) * F_sparse

        return sorted(tasks, key=lambda t: -score(t))
