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
    scenario: str
    run_name: str
    scheduler: str
    seed: int
    alpha_true_mean: Optional[float]
    n_created: int
    n_finished: int
    n_dropped: int
    n_success: int
    success_rate: float
    throughput_per_s: float

    # Failure breakdown
    miss_uplink: int
    miss_queue: int
    miss_cpu: int
    miss_coverage: int

    # Priority split
    high_thr: float
    high_success_rate: float
    low_success_rate: float

    # Weighted success (continuous priority in [0,1])
    weighted_success_sum: float
    weighted_success_norm: float  # divide by sum of priorities
    alpha_true_weighted_success_norm: Optional[float]

    # Timings
    mean_response_time_s: Optional[float]
    p95_response_time_s: Optional[float]
    p99_response_time_s: Optional[float]

    # Margin (only for success)
    mean_margin: Optional[float]

    # Utility (success * (priority + beta*margin))
    utility_beta: float
    utility_sum: float
    utility_mean_per_task: float


def _task_margin(task: Task) -> Optional[float]:
    """Margin at CPU start: R_i(t_start) - C_i."""
    if task.cpu_start_t is None:
        return None
    eff_start = task.meta.get("effective_deadline_at_start")
    if eff_start is None:
        return None
    return float(eff_start) - float(task.cpu_start_t) - float(task.compute_time_s)


def summarize_tasks(
    tasks: Iterable[Task],
    scenario: str,
    run_name: str,
    scheduler: str,
    seed: int,
    high_thr: float,
    utility_beta: float = 0.5,
    sim_duration_s: Optional[float] = None,
    alpha_true_mean: Optional[float] = None,
) -> RunSummary:
    tasks = list(tasks)
    created = [t for t in tasks if t.status in (TaskStatus.CREATED, TaskStatus.UL_IN_PROGRESS, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.FINISHED, TaskStatus.DEADLINE_MISS, TaskStatus.DROPPED)]
    finished = [t for t in tasks if t.status in (TaskStatus.FINISHED, TaskStatus.DEADLINE_MISS)]
    dropped = [t for t in tasks if t.status == TaskStatus.DROPPED]
    success = [t for t in tasks if t.status == TaskStatus.FINISHED and (t.cpu_finish_t is not None) and (t.cpu_finish_t <= t.effective_deadline_t)]

    n_created = len(created) if created else len(tasks)
    n_finished = len(finished)
    n_dropped = len(dropped)
    n_success = len(success)
    sr = (n_success / n_created) if n_created > 0 else 0.0
    tp = (n_finished / sim_duration_s) if sim_duration_s else 0.0

    miss_uplink = sum(1 for t in tasks if t.status == TaskStatus.DEADLINE_MISS and t.meta.get("miss_where") == "uplink")
    miss_queue = sum(1 for t in tasks if t.status == TaskStatus.DEADLINE_MISS and t.meta.get("miss_where") == "queue")
    miss_cpu = sum(1 for t in tasks if t.status == TaskStatus.DEADLINE_MISS and t.meta.get("miss_where") == "cpu")
    miss_coverage = sum(1 for t in tasks if t.status == TaskStatus.DEADLINE_MISS and t.meta.get("miss_where") == "coverage")

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
    a_weighted = None
    if tasks:
        alpha_vals = []
        for t in tasks:
            a = t.meta.get("alpha_true_at_create")
            alpha_vals.append(a if a is not None else alpha_true_mean)
        if any(a is not None for a in alpha_vals):
            total_w = 0.0
            succ_w = 0.0
            for t, a in zip(tasks, alpha_vals):
                if a is None:
                    continue
                a = max(0.0, min(1.0, float(a)))
                w = (1.0 - a) + a * float(t.priority)
                total_w += w
                if t.status == TaskStatus.FINISHED and (t.cpu_finish_t is not None) and (t.cpu_finish_t <= t.effective_deadline_t):
                    succ_w += w
            if total_w > 0:
                a_weighted = succ_w / total_w

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
        scenario=str(scenario),
        run_name=run_name,
        scheduler=scheduler,
        seed=seed,
        alpha_true_mean=alpha_true_mean,
        n_created=n_created,
        n_finished=n_finished,
        n_dropped=n_dropped,
        n_success=n_success,
        success_rate=float(sr),
        throughput_per_s=float(tp),
        miss_uplink=int(miss_uplink),
        miss_queue=int(miss_queue),
        miss_cpu=int(miss_cpu),
        miss_coverage=int(miss_coverage),
        high_thr=float(high_thr),
        high_success_rate=float(high_sr),
        low_success_rate=float(low_sr),
        weighted_success_sum=float(sum_p_succ),
        weighted_success_norm=float(w_norm),
        alpha_true_weighted_success_norm=a_weighted,
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
    write_header = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        if write_header:
            w.writeheader()
        w.writerow(d)
