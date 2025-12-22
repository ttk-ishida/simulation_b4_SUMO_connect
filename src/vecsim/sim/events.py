from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


EventType = Literal[
    "TASK_CREATED",
    "UL_STARTED",
    "UL_FINISHED",
    "ENQUEUED",
    "CPU_STARTED",
    "CPU_FINISHED",
    "DROPPED",
    "DEADLINE_MISS",
]


@dataclass(frozen=True)
class SimEvent:
    """A minimal event record for later metrics."""
    t: float
    etype: EventType
    task_id: str
    vehicle_id: Optional[str] = None
    details: Optional[dict] = None
