from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Summary:
    scheduler: str
    run_name: str
    n_tasks: int
    success_rate: float


def read_events_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Minimal summary based on CPU_FINISHED and DEADLINE_MISS events.
    finished = {}
    created = set()
    for ev in events:
        et = ev.get("etype")
        tid = ev.get("task_id")
        if et == "TASK_CREATED":
            created.add(tid)
        if et == "CPU_FINISHED":
            finished[tid] = True
        if et == "DEADLINE_MISS":
            finished[tid] = False

    n = len(created) if created else len(finished)
    if n == 0:
        return {"n_tasks": 0, "success_rate": 0.0}
    succ = sum(1 for v in finished.values() if v is True)
    return {"n_tasks": n, "success_rate": succ / max(1, n)}


def write_summary_csv(summary_rows: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_path, index=False)
