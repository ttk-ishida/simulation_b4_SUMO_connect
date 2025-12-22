from __future__ import annotations

from typing import Any, Dict

from .fifo import FIFOScheduler
from .edf import EDFScheduler
from .v_edf import VEDFScheduler
from .pred_aware import PredAwareScheduler


def make_scheduler(cfg: Dict[str, Any]):
    scfg = cfg.get("scheduler", {}) or {}
    name = scfg.get("name", "fifo")
    params = scfg.get("params", {}) or {}

    if name == "fifo":
        return FIFOScheduler()

    if name == "edf":
        return EDFScheduler(**{k: v for k, v in params.items() if k in {"use_effective_deadline"}})

    if name == "v_edf":
        return VEDFScheduler()

    if name == "pred_aware":
        return PredAwareScheduler(eps=float(params.get("eps", 1e-9)))

    raise ValueError(f"Unknown scheduler name: {name}")
