from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class EDFScheduler:
    name: str = "edf"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        # Earliest deadline first (ascending)
        return sorted(tasks, key=lambda t: t.deadline_t)
