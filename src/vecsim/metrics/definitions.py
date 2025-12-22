from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..models.task import Task, TaskStatus


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    priority: float
    success: bool
    deadline_miss: bool
    response_time_s: Optional[float]


def outcome_from_task(task: Task) -> TaskOutcome:
    success = task.status == TaskStatus.FINISHED and (task.cpu_finish_t is not None) and (task.cpu_finish_t <= task.effective_deadline_t)
    deadline_miss = (task.cpu_finish_t is not None) and (task.cpu_finish_t > task.effective_deadline_t)
    return TaskOutcome(
        task_id=task.task_id,
        priority=float(task.priority),
        success=bool(success),
        deadline_miss=bool(deadline_miss),
        response_time_s=task.response_time(),
    )


def success_rate(outcomes: Iterable[TaskOutcome]) -> float:
    o = list(outcomes)
    if not o:
        return 0.0
    return sum(1 for x in o if x.success) / len(o)


def weighted_success(outcomes: Iterable[TaskOutcome]) -> float:
    """Sum of priority of successful tasks (continuous priority in [0,1])."""
    s = 0.0
    for x in outcomes:
        if x.success:
            s += float(x.priority)
    return s


def tail_latency_p95(outcomes: Iterable[TaskOutcome]) -> Optional[float]:
    vals = sorted([x.response_time_s for x in outcomes if x.response_time_s is not None])
    if not vals:
        return None
    idx = int(0.95 * (len(vals) - 1))
    return float(vals[idx])
