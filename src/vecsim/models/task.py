from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    UL_IN_PROGRESS = "UL_IN_PROGRESS"
    READY = "READY"              # arrived at MEC queue
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    DROPPED = "DROPPED"
    DEADLINE_MISS = "DEADLINE_MISS"


@dataclass
class Task:
    """A single offloaded task.

    Time-related fields are absolute simulation times [s].
    """
    task_id: str
    vehicle_id: str
    created_t: float

    # Constraints
    relative_deadline_s: float        # relative deadline from creation [s]
    deadline_t: float                 # created_t + relative_deadline
    stay_deadline_t: float            # created_t + predicted_stay_time (coverage constraint)
    effective_deadline_t: float       # min(deadline_t, stay_deadline_t)

    # Work & comm
    priority: float                   # continuous in [0,1]
    input_bits: float                 # uplink input size [bit]
    cycles: float                     # required CPU cycles [cycle]

    # Computed durations (can be filled later)
    uplink_time_s: float = 0.0        # [s]
    compute_time_s: float = 0.0       # [s]

    # Runtime bookkeeping
    status: TaskStatus = TaskStatus.CREATED
    enqueued_t: Optional[float] = None
    cpu_start_t: Optional[float] = None
    cpu_finish_t: Optional[float] = None
    drop_reason: Optional[str] = None

    # Optional extra info
    meta: dict = field(default_factory=dict)

    def remaining_until_deadline(self, now: float) -> float:
        return self.effective_deadline_t - now

    def is_deadline_missed(self, now: float) -> bool:
        return now > self.effective_deadline_t

    def response_time(self) -> Optional[float]:
        if self.cpu_finish_t is None:
            return None
        return self.cpu_finish_t - self.created_t
