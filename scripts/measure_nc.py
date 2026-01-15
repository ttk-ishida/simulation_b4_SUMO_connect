#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from vecsim.models.rsu import RSU
from vecsim.sim.traci_api import TraciAPI
from vecsim.utils.yamlconf import load_merged


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    scenario_dir = Path(cfg.get("paths", {}).get("scenario_dir", "scenarios"))
    sumocfg = cfg.get("sumo", {}).get("sumocfg_path")
    if sumocfg and not str(sumocfg).startswith("/"):
        cfg["sumo"]["sumocfg_path"] = str(scenario_dir / sumocfg)
    return cfg


def _percentile(xs: List[int], q: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(q * (len(xs_sorted) - 1))
    return float(xs_sorted[idx])


def main() -> int:
    p = argparse.ArgumentParser(description="Measure max/p95 vehicles within RSU coverage (N_C).")
    p.add_argument("--config", default="configs/base.yaml", help="Base config YAML")
    p.add_argument("--scenario", required=True, help="Scenario config YAML")
    p.add_argument("--use-p95", action="store_true", help="Use p95 instead of max as N_C")
    args = p.parse_args()

    cfg = load_merged(args.config, args.scenario, None)
    cfg = _resolve_paths(cfg)

    sim = cfg.get("sim", {}) or {}
    duration_s = float(sim.get("duration_s", 600.0))
    step_s = float(sim.get("step_length_s", 0.2))
    n_steps = int(duration_s / step_s)

    rsu_cfg = cfg.get("rsu", {}) or {}
    rect_cfg = (rsu_cfg.get("coverage_rect", {}) or {})
    half_w = rect_cfg.get("half_width_m", None)
    half_h = rect_cfg.get("half_height_m", None)
    rsu = RSU(
        rsu_id=str(rsu_cfg.get("id", "rsu0")),
        x_m=float(rsu_cfg.get("x_m", 0.0)),
        y_m=float(rsu_cfg.get("y_m", 0.0)),
        coverage_radius_m=float(rsu_cfg.get("coverage_radius_m", 250.0)),
        coverage_half_width_m=float(half_w) if half_w is not None else None,
        coverage_half_height_m=float(half_h) if half_h is not None else None,
    )

    sumocfg_path = cfg.get("sumo", {}).get("sumocfg_path")
    if not sumocfg_path:
        raise SystemExit("sumo.sumocfg_path is not set in config.")

    traci = TraciAPI()
    traci.start(["sumo", "-c", str(sumocfg_path)])

    counts: List[int] = []
    try:
        for _ in range(n_steps):
            traci.simulation_step()
            count = 0
            for vid in traci.get_all_vehicle_ids():
                snap = traci.get_vehicle_snapshot(vid)
                if rsu.in_coverage(snap.x_m, snap.y_m):
                    count += 1
            counts.append(count)
    finally:
        traci.close()

    max_n = max(counts) if counts else 0
    p95 = _percentile(counts, 0.95)
    n_c = p95 if args.use_p95 else max_n
    print(f"N_C={n_c:.0f} (max={max_n}, p95={p95:.1f}, samples={len(counts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
