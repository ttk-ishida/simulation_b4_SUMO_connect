from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..models.task import Task, TaskStatus
from ..sim.events import SimEvent


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


def _build_event_index(events: Iterable[SimEvent]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        info = idx.setdefault(ev.task_id, {
            "created_t": None,
            "finish_t": None,
            "finish_type": None,
            "miss_where": None,
            "dropped_t": None,
        })
        if ev.etype == "TASK_CREATED":
            info["created_t"] = float(ev.t)
        elif ev.etype in ("CPU_FINISHED", "DEADLINE_MISS"):
            t = float(ev.t)
            if info["finish_t"] is None or t < float(info["finish_t"]):
                info["finish_t"] = t
                info["finish_type"] = ev.etype
                if ev.etype == "DEADLINE_MISS":
                    info["miss_where"] = (ev.details or {}).get("where")
        elif ev.etype == "DROPPED":
            info["dropped_t"] = float(ev.t)
    return idx


def summarize_tasks_windowed(
    tasks: Iterable[Task],
    events: Iterable[SimEvent],
    window_s: float,
    step_s: float,
    scenario: str,
    run_name: str,
    scheduler: str,
    seed: int,
    high_thr: float,
    utility_beta: float = 0.5,
    sim_duration_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    tasks_list = list(tasks)
    event_idx = _build_event_index(events)
    if sim_duration_s is None:
        sim_duration_s = max((float(t.created_t) for t in tasks_list), default=0.0)

    window_s = float(window_s)
    step_s = float(step_s) if step_s > 0 else float(window_s)
    if window_s <= 0.0:
        return []

    rows: List[Dict[str, Any]] = []
    t = window_s
    while t <= sim_duration_s + 1e-9:
        win_start = t - window_s
        window_tasks: List[Task] = []
        for task in tasks_list:
            created_t = event_idx.get(task.task_id, {}).get("created_t")
            if created_t is None:
                created_t = float(task.created_t)
            if created_t <= win_start or created_t > t:
                continue
            window_tasks.append(task)

        n_created = len(window_tasks)
        if n_created == 0:
            rows.append({
                "t_s": float(t),
                "window_s": float(window_s),
                "scenario": str(scenario),
                "run_name": run_name,
                "scheduler": scheduler,
                "seed": int(seed),
                "alpha_true_mean": None,
                "n_created": 0,
                "n_finished": 0,
                "n_dropped": 0,
                "n_success": 0,
                "success_rate": 0.0,
                "throughput_per_s": 0.0,
                "miss_uplink": 0,
                "miss_queue": 0,
                "miss_cpu": 0,
                "miss_coverage": 0,
                "high_thr": float(high_thr),
                "high_success_rate": 0.0,
                "low_success_rate": 0.0,
                "weighted_success_sum": 0.0,
                "weighted_success_norm": 0.0,
                "alpha_true_weighted_success_norm": None,
                "mean_response_time_s": None,
                "p95_response_time_s": None,
                "p99_response_time_s": None,
                "mean_margin": None,
                "utility_beta": float(utility_beta),
                "utility_sum": 0.0,
                "utility_mean_per_task": 0.0,
            })
            t += step_s
            continue

        n_finished = 0
        n_success = 0
        n_dropped = 0
        miss_uplink = 0
        miss_queue = 0
        miss_cpu = 0
        miss_coverage = 0
        successes: List[Task] = []
        finished_tasks: List[Task] = []
        for task in window_tasks:
            info = event_idx.get(task.task_id, {})
            dropped_t = info.get("dropped_t")
            finish_t = info.get("finish_t")
            finish_type = info.get("finish_type")
            miss_where = info.get("miss_where")

            if dropped_t is not None and float(dropped_t) <= t:
                n_dropped += 1
                continue

            if finish_t is not None and float(finish_t) <= t:
                n_finished += 1
                finished_tasks.append(task)
                if finish_type == "CPU_FINISHED":
                    n_success += 1
                    successes.append(task)
                elif finish_type == "DEADLINE_MISS":
                    if miss_where == "uplink":
                        miss_uplink += 1
                    elif miss_where == "queue":
                        miss_queue += 1
                    elif miss_where == "cpu":
                        miss_cpu += 1
                    elif miss_where == "coverage":
                        miss_coverage += 1

        sr = (n_success / n_created) if n_created > 0 else 0.0
        tp = (n_finished / window_s) if window_s > 0 else 0.0

        highs = [t for t in window_tasks if float(t.priority) >= high_thr]
        lows = [t for t in window_tasks if float(t.priority) < high_thr]

        def _sr_subset(sub: List[Task]) -> float:
            if not sub:
                return 0.0
            s = 0
            for tsk in sub:
                info = event_idx.get(tsk.task_id, {})
                finish_t = info.get("finish_t")
                finish_type = info.get("finish_type")
                if finish_t is not None and float(finish_t) <= t and finish_type == "CPU_FINISHED":
                    s += 1
            return s / len(sub)

        high_sr = _sr_subset(highs)
        low_sr = _sr_subset(lows)

        sum_p = sum(float(t.priority) for t in window_tasks)
        sum_p_succ = sum(float(t.priority) for t in successes)
        w_norm = (sum_p_succ / sum_p) if sum_p > 0 else 0.0

        alpha_vals = [t.meta.get("alpha_true_at_create") for t in window_tasks if t.meta.get("alpha_true_at_create") is not None]
        alpha_true_mean = float(sum(alpha_vals) / len(alpha_vals)) if alpha_vals else None
        a_weighted = None
        if window_tasks:
            alpha_list = []
            for tsk in window_tasks:
                a = tsk.meta.get("alpha_true_at_create")
                alpha_list.append(a if a is not None else alpha_true_mean)
            if any(a is not None for a in alpha_list):
                total_w = 0.0
                succ_w = 0.0
                for tsk, a in zip(window_tasks, alpha_list):
                    if a is None:
                        continue
                    a = max(0.0, min(1.0, float(a)))
                    w = (1.0 - a) + a * float(tsk.priority)
                    total_w += w
                    info = event_idx.get(tsk.task_id, {})
                    finish_t = info.get("finish_t")
                    finish_type = info.get("finish_type")
                    if finish_t is not None and float(finish_t) <= t and finish_type == "CPU_FINISHED":
                        succ_w += w
                if total_w > 0:
                    a_weighted = succ_w / total_w

        rts = [float(tsk.cpu_finish_t - tsk.created_t) for tsk in finished_tasks if tsk.cpu_finish_t is not None and float(tsk.cpu_finish_t) <= t]
        mean_rt = float(np.mean(rts)) if rts else None
        p95 = _percentile(rts, 0.95) if rts else None
        p99 = _percentile(rts, 0.99) if rts else None

        margins = []
        util_sum = 0.0
        for tsk in window_tasks:
            info = event_idx.get(tsk.task_id, {})
            finish_t = info.get("finish_t")
            finish_type = info.get("finish_type")
            if finish_t is None or float(finish_t) > t or finish_type != "CPU_FINISHED":
                continue
            m = _task_margin(tsk)
            if m is None:
                continue
            margins.append(m)
            util_sum += float(tsk.priority) + float(utility_beta) * float(m)

        mean_margin = float(np.mean(margins)) if margins else None
        util_mean = util_sum / max(1, n_created)

        rows.append({
            "t_s": float(t),
            "window_s": float(window_s),
            "scenario": str(scenario),
            "run_name": run_name,
            "scheduler": scheduler,
            "seed": int(seed),
            "alpha_true_mean": alpha_true_mean,
            "n_created": int(n_created),
            "n_finished": int(n_finished),
            "n_dropped": int(n_dropped),
            "n_success": int(n_success),
            "success_rate": float(sr),
            "throughput_per_s": float(tp),
            "miss_uplink": int(miss_uplink),
            "miss_queue": int(miss_queue),
            "miss_cpu": int(miss_cpu),
            "miss_coverage": int(miss_coverage),
            "high_thr": float(high_thr),
            "high_success_rate": float(high_sr),
            "low_success_rate": float(low_sr),
            "weighted_success_sum": float(sum_p_succ),
            "weighted_success_norm": float(w_norm),
            "alpha_true_weighted_success_norm": a_weighted,
            "mean_response_time_s": mean_rt,
            "p95_response_time_s": p95,
            "p99_response_time_s": p99,
            "mean_margin": mean_margin,
            "utility_beta": float(utility_beta),
            "utility_sum": float(util_sum),
            "utility_mean_per_task": float(util_mean),
        })

        t += step_s

    return rows


def write_summary_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
