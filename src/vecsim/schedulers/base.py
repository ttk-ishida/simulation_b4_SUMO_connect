from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol

from ..models.task import Task


@dataclass(frozen=True)
class SchedulerContext:
    now: float
    w0: float            # time until CPU can start next task [s]
    alpha: float = 0.0   # predicted congestion weight in [0,1]
    cfg: Dict[str, Any] | None = None


class Scheduler(Protocol):
    name: str

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        ...


def remaining_time(task: Task, now: float) -> float:
    return task.effective_deadline_t - now


def s_total(task: Task) -> float:
    """Total service time if scheduled next: uplink + compute."""
    return float(task.uplink_time_s + task.compute_time_s)


def slack_next(task: Task, ctx: SchedulerContext) -> float:
    """Slack if this task is executed next: R - (W0 + s_total)."""
    R = remaining_time(task, ctx.now)
    return float(R - (ctx.w0 + s_total(task)))
