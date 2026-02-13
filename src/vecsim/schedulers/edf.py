from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task

_logged_edf_order = False


@dataclass
class EDFScheduler:
    name: str = "edf"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        # Earliest deadline first (ascending)
        global _logged_edf_order
        ordered = sorted(tasks, key=lambda t: t.deadline_t)
        if (not _logged_edf_order) and ordered:
            ids = [t.task_id for t in ordered[:10]]
            print(f"[EDF order] n={len(ordered)} ids={ids}")
            _logged_edf_order = True
        return ordered
