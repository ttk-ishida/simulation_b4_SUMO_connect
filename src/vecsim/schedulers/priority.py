from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task

_logged_priority_order = False


@dataclass
class PriorityScheduler:
    """Rank tasks by priority only (descending)."""
    name: str = "priority"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        global _logged_priority_order
        ordered = sorted(tasks, key=lambda t: -t.priority)
        if (not _logged_priority_order) and ordered:
            ids = [t.task_id for t in ordered[:10]]
            print(f"[Priority order] n={len(ordered)} ids={ids}")
            _logged_priority_order = True
        return ordered
