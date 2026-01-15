from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import SchedulerContext
from ..models.task import Task


@dataclass
class EEDFScheduler:
    """Effective-deadline EDF (min(deadline, stay) first)."""
    name: str = "e_edf"

    def rank(self, tasks: List[Task], ctx: SchedulerContext) -> List[Task]:
        # Smaller effective deadline first (equivalently smaller remaining R_i).
        return sorted(tasks, key=lambda t: t.effective_deadline_t)
