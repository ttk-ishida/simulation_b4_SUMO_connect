from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MuSegment:
    t_start_s: float
    t_end_s: float
    mu: float  # tasks/s/vehicle


def parse_mu_schedule(cfg: Dict[str, Any]) -> Optional[List[MuSegment]]:
    """Parse scenario burst schedule from config.

    Expected structure (configs/scenario/traffic_burst.yaml):
      burst:
        enabled: true
        mu_schedule:
          - {t_start_s: 0, t_end_s: 200, mu: 0.03}
          - ...
    """
    burst = cfg.get("burst", {}) or {}
    if not bool(burst.get("enabled", False)):
        return None
    sched = burst.get("mu_schedule", None)
    if not sched:
        return None
    out: List[MuSegment] = []
    for seg in sched:
        out.append(MuSegment(
            t_start_s=float(seg["t_start_s"]),
            t_end_s=float(seg["t_end_s"]),
            mu=float(seg["mu"]),
        ))
    return out


def mu_at_time(default_mu: float, schedule: Optional[List[MuSegment]], now_s: float) -> float:
    if not schedule:
        return float(default_mu)
    for seg in schedule:
        if seg.t_start_s <= now_s < seg.t_end_s:
            return float(seg.mu)
    return float(default_mu)
