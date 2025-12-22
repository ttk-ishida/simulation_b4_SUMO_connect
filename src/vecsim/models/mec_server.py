from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .queue import TaskQueue
from .task import Task, TaskStatus


@dataclass
class MECServer:
    """Single-core non-preemptive MEC server with one queue."""
    cpu_freq_hz: float
    queue: TaskQueue
    non_preemptive: bool = True

    current_task: Optional[Task] = None
    remaining_compute_s: float = 0.0

    def w0(self) -> float:
        """Time until the CPU can start the next task (state variable)."""
        return self.remaining_compute_s if self.current_task is not None else 0.0

    def is_idle(self) -> bool:
        return self.current_task is None

    def start_next_if_idle(self, now: float) -> Optional[Task]:
        if self.current_task is not None:
            return None
        t = self.queue.pop_next()
        if t is None:
            return None
        t.status = TaskStatus.RUNNING
        t.cpu_start_t = now
        self.current_task = t
        self.remaining_compute_s = float(t.compute_time_s)
        return t

    def step(self, dt: float, now: float) -> Optional[Task]:
        """Advance server execution by dt seconds.

        Returns:
            Finished task if any, else None.
        """
        if self.current_task is None:
            return None
        self.remaining_compute_s -= dt
        if self.remaining_compute_s > 0:
            return None

        finished = self.current_task
        finished.status = TaskStatus.FINISHED
        finished.cpu_finish_t = now
        self.current_task = None
        self.remaining_compute_s = 0.0
        return finished
