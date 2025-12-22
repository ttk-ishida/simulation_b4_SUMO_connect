from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import SchedulerContext, remaining_time, s_total, slack_next
from ..models.task import Task
from ..utils.math import clip, inv


def priority_weight(p: float, cfg: Dict[str, Any]) -> float:
    wcfg = (cfg.get("priority_weight", {}) or {})
    name = wcfg.get("name", "linear")
    params = wcfg.get("params", {}) or {}
    p = clip(float(p), 0.0, 1.0)
    if name == "linear":
        return p
    if name == "power":
        gamma = float(params.get("gamma", 2.0))
        return p ** gamma
    return p


@dataclass
class PredAwareScheduler:
    name: str = "pred_aware"
    eps: float = 1e-9

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        cfg = ctx.cfg or {}
        alpha = clip(float(ctx.alpha), 0.0, 1.0)

        def score(t: Task) -> float:
            R = remaining_time(t, ctx.now)
            if R <= self.eps:
                return -1e30  # already impossible
            st = ctx.w0 + s_total(t)
            S = R - st  # slack if executed next

            # Sparse-side: success+margin (0 if miss)
            F_sparse = max(0.0, 1.0 - st / R)

            # Dense-side: prioritize high priority and urgent slack
            wP = priority_weight(t.priority, cfg)
            U = inv(S + self.eps, eps=self.eps)  # higher if slack small
            F_dense = wP * U

            return (1.0 - alpha) * F_sparse + alpha * F_dense

        return sorted(tasks, key=score, reverse=True)
