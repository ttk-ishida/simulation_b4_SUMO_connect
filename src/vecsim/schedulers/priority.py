from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class PriorityScheduler:
    """Rank tasks by priority only (descending)."""
    name: str = "priority"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        return sorted(tasks, key=lambda t: -t.priority)
