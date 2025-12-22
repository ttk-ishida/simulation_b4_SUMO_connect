from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from .task import Task


@dataclass
class TaskQueue:
    """Simple task queue (ordering is controlled by schedulers)."""
    tasks: List[Task] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    def enqueue(self, task: Task) -> None:
        self.tasks.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        self.tasks.extend(list(tasks))

    def remove(self, task_id: str) -> Optional[Task]:
        for i, t in enumerate(self.tasks):
            if t.task_id == task_id:
                return self.tasks.pop(i)
        return None

    def reorder(self, key: Callable[[Task], float], reverse: bool = True) -> None:
        """Reorder tasks in-place by a score key (descending by default)."""
        self.tasks.sort(key=key, reverse=reverse)

    def pop_next(self) -> Optional[Task]:
        if not self.tasks:
            return None
        return self.tasks.pop(0)
