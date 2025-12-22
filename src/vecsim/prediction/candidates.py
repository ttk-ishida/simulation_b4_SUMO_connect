from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from ..models.rsu import RSU
from ..models.vehicle import Vehicle


@dataclass(frozen=True)
class CandidatePolicy:
    """Defines which vehicles are candidates for inflow prediction."""
    # Vehicles within (coverage_radius + margin) are considered candidates.
    margin_m: float = 150.0


def select_candidate_vehicle_ids(
    vehicles: Mapping[str, Vehicle],
    rsu: RSU,
    cfg: Dict[str, Any],
) -> List[str]:
    pred = cfg.get("prediction", {}) or {}
    pol_cfg = pred.get("candidate_policy", {}) or {}
    margin = float(pol_cfg.get("margin_m", 150.0))
    r = float(rsu.coverage_radius_m) + margin

    out: List[str] = []
    for vid, v in vehicles.items():
        if rsu.distance_m(v.x_m, v.y_m) <= r:
            out.append(vid)
    return out
