from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class VEDFScheduler:
    """Single-core simplification of V-EDF style ranking.

    Default: priority desc, then effective deadline asc.
    """
    name: str = "v_edf"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        return sorted(tasks, key=lambda t: (-t.priority, t.effective_deadline_t))
