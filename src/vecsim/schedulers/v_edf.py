from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class VEDFScheduler:
    """Single-core simplification of V-EDF style ranking.

    Default: deadline asc, then priority desc.
    """
    name: str = "v_edf"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        return sorted(tasks, key=lambda t: (t.deadline_t, -t.priority))
