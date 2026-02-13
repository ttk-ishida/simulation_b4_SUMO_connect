from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import SchedulerContext
from ..models.task import Task
from ..utils.math import clip

_logged_score_stats = False

_logged_r_stats = False


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
        global _logged_r_stats, _logged_score_stats
        if tasks and (not _logged_r_stats or not _logged_score_stats):
            rs = [max(0.0, float(t.effective_deadline_t - ctx.now)) for t in tasks]
            if not _logged_r_stats:
                r_min = float(np.min(rs))
                r_med = float(np.median(rs))
                r_max = float(np.max(rs))
                r_std = float(np.std(rs))
                print(f"[R stats] n={len(rs)} min={r_min:.6f} med={r_med:.6f} max={r_max:.6f} std={r_std:.6f}")
                _logged_r_stats = True
            if not _logged_score_stats:
                ps = np.array([clip(float(t.priority), 0.0, 1.0) for t in tasks], dtype=float)
                rs = np.array(rs, dtype=float)
                if np.std(ps) > 0 and np.std(rs) > 0:
                    # F_sparse before mixing: 1 - R/max_r
                    f_sparse = 1.0 - (rs / (max_r + self.eps))
                    # Correlation between score components and priority
                    score = alpha * ps + (1.0 - alpha) * f_sparse
                    corr = float(np.corrcoef(score, ps)[0, 1])
                    print(f"[Score corr] n={len(ps)} corr(score, priority)={corr:.6f}")
                _logged_score_stats = True

        def score(t: Task) -> float:
            R = max(0.0, float(t.effective_deadline_t - ctx.now))
            F_dense = clip(float(t.priority), 0.0, 1.0)
            # E-EDF side: smaller effective remaining time -> larger score
            F_sparse = 1.0 - (R / (max_r + self.eps))
            return alpha * F_dense + (1.0 - alpha) * F_sparse

        return sorted(tasks, key=lambda t: -score(t))
