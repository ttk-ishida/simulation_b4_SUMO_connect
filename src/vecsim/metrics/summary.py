from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..models.task import Task, TaskStatus


def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    idx = int(q * (len(xs_sorted) - 1))
    return float(xs_sorted[idx])


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    scheduler: str
    seed: int
    n_created: int
    n_finished: int
    n_success: int
    success_rate: float

    # Priority split
    high_thr: float
    high_success_rate: float
    low_success_rate: float

    # Weighted success (continuous priority in [0,1])
    weighted_success_sum: float
    weighted_success_norm: float  # divide by sum of priorities

    # Timings
    mean_response_time_s: Optional[float]
    p95_response_time_s: Optional[float]
    p99_response_time_s: Optional[float]

    # Margin / slack style score (only for success)
    mean_margin: Optional[float]

    # Utility (success * (priority + beta*margin))
    utility_beta: float
    utility_sum: float
    utility_mean_per_task: float


def _task_margin(task: Task) -> Optional[float]:
    """Margin in [0,1] where 1=very early, 0=just meets deadline.

    margin = max(0, 1 - (response_time)/(allowed_time))
    allowed_time = effective_deadline - created_t
    """
    if task.cpu_finish_t is None:
        return None
    allowed = task.effective_deadline_t - task.created_t
    if allowed <= 0:
        return 0.0
    rt = task.cpu_finish_t - task.created_t
    return max(0.0, 1.0 - (rt / allowed))


def summarize_tasks(
    tasks: Iterable[Task],
    run_name: str,
    scheduler: str,
    seed: int,
    high_thr: float,
    utility_beta: float = 0.5,
) -> RunSummary:
    tasks = list(tasks)
    created = [t for t in tasks if t.status in (TaskStatus.CREATED, TaskStatus.UL_IN_PROGRESS, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.FINISHED, TaskStatus.DEADLINE_MISS, TaskStatus.DROPPED)]
    finished = [t for t in tasks if t.status in (TaskStatus.FINISHED, TaskStatus.DEADLINE_MISS)]
    success = [t for t in tasks if t.status == TaskStatus.FINISHED and (t.cpu_finish_t is not None) and (t.cpu_finish_t <= t.effective_deadline_t)]

    n_created = len(created) if created else len(tasks)
    n_finished = len(finished)
    n_success = len(success)
    sr = (n_success / n_created) if n_created > 0 else 0.0

    # High/Low split by threshold
    highs = [t for t in tasks if float(t.priority) >= high_thr]
    lows = [t for t in tasks if float(t.priority) < high_thr]

    def _sr(sub: List[Task]) -> float:
        if not sub:
            return 0.0
        s = sum(1 for t in sub if t.status == TaskStatus.FINISHED and (t.cpu_finish_t is not None) and (t.cpu_finish_t <= t.effective_deadline_t))
        return s / len(sub)

    high_sr = _sr(highs)
    low_sr = _sr(lows)

    # Weighted success
    sum_p = sum(float(t.priority) for t in tasks)
    sum_p_succ = sum(float(t.priority) for t in success)
    w_norm = (sum_p_succ / sum_p) if sum_p > 0 else 0.0

    # Latency
    rts = [float(t.cpu_finish_t - t.created_t) for t in finished if t.cpu_finish_t is not None]
    mean_rt = float(np.mean(rts)) if rts else None
    p95 = _percentile(rts, 0.95) if rts else None
    p99 = _percentile(rts, 0.99) if rts else None

    # Margin and Utility
    margins = []
    util_sum = 0.0
    for t in tasks:
        m = _task_margin(t)
        if m is None:
            continue
        if t.status == TaskStatus.FINISHED and (t.cpu_finish_t is not None) and (t.cpu_finish_t <= t.effective_deadline_t):
            margins.append(m)
            util_sum += float(t.priority) + float(utility_beta) * float(m)
        else:
            # failed tasks contribute 0
            pass

    mean_margin = float(np.mean(margins)) if margins else None
    util_mean = util_sum / max(1, n_created)

    return RunSummary(
        run_name=run_name,
        scheduler=scheduler,
        seed=seed,
        n_created=n_created,
        n_finished=n_finished,
        n_success=n_success,
        success_rate=float(sr),
        high_thr=float(high_thr),
        high_success_rate=float(high_sr),
        low_success_rate=float(low_sr),
        weighted_success_sum=float(sum_p_succ),
        weighted_success_norm=float(w_norm),
        mean_response_time_s=mean_rt,
        p95_response_time_s=p95,
        p99_response_time_s=p99,
        mean_margin=mean_margin,
        utility_beta=float(utility_beta),
        utility_sum=float(util_sum),
        utility_mean_per_task=float(util_mean),
    )


def write_summary_csv(summary: RunSummary, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(summary)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        w.writeheader()
        w.writerow(d)
