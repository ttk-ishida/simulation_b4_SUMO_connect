from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from ..sim.events import SimEvent


@dataclass
class EventRecorder:
    """Collects and optionally persists simulation events (raw logs)."""
    events: List[SimEvent] = field(default_factory=list)

    def record(self, ev: SimEvent) -> None:
        self.events.append(ev)

    def extend(self, evs: Iterable[SimEvent]) -> None:
        self.events.extend(list(evs))

    def dump_jsonl(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps({
                    "t": ev.t,
                    "etype": ev.etype,
                    "task_id": ev.task_id,
                    "vehicle_id": ev.vehicle_id,
                    "details": ev.details or {},
                }, ensure_ascii=False) + "\n")
