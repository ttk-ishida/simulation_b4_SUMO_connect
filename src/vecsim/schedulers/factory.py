from __future__ import annotations

from typing import Any, Dict

from .fifo import FIFOScheduler
from .edf import EDFScheduler
from .e_edf import EEDFScheduler
from .v_edf import VEDFScheduler
from .pred_aware import PredAwareScheduler
from .prediction_aware import PredictionAwareScheduler
from .priority import PriorityScheduler


def make_scheduler(cfg: Dict[str, Any]):
    scfg = cfg.get("scheduler", {}) or {}
    name = scfg.get("name", "fifo")
    params = scfg.get("params", {}) or {}

    if name == "fifo":
        return FIFOScheduler()

    if name == "edf":
        return EDFScheduler()

    if name == "e_edf":
        return EEDFScheduler()

    if name == "v_edf":
        return VEDFScheduler()

    if name == "pred_aware":
        return PredAwareScheduler(eps=float(params.get("eps", 1e-9)))
    if name == "prediction_aware":
        return PredictionAwareScheduler(eps=float(params.get("eps", 1e-9)))
    if name == "priority":
        return PriorityScheduler()

    raise ValueError(f"Unknown scheduler name: {name}")
