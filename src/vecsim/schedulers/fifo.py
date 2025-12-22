from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class FIFOScheduler:
    name: str = "fifo"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        # Arrival/enqueue order: prefer enqueued_t if set, else created_t.
        return sorted(tasks, key=lambda t: (t.enqueued_t if t.enqueued_t is not None else t.created_t))
