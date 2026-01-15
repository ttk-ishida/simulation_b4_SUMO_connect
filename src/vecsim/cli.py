from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging_conf import setup_logging
from .sim.runner import RunArgs, run_once
from .utils.yamlconf import load_yaml

log = logging.getLogger(__name__)


def _cartesian_product(keys: List[str], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []

    def rec(i: int, cur: Dict[str, Any]) -> None:
        if i == len(keys):
            points.append(dict(cur))
            return
        k = keys[i]
        for v in grid[k]:
            cur[k] = v
            rec(i + 1, cur)
        cur.pop(k, None)

    rec(0, {})
    return points


def cmd_run(ns: argparse.Namespace) -> int:
    args = RunArgs(
        config_path=ns.config,
        scenario_config_path=ns.scenario,
        scheduler_config_path=ns.scheduler,
        overrides=None,
    )
    return run_once(args)


def cmd_sweep(ns: argparse.Namespace) -> int:
    sweep_doc = load_yaml(ns.config)
    sweep_cfg = sweep_doc.get("sweep", {}) or {}

    base_config = str(sweep_cfg["base_config"])
    scenario_config = sweep_cfg.get("scenario_config", None)
    scenario_configs = sweep_cfg.get("scenario_configs", None)
    if scenario_config is None and scenario_configs is None:
        raise SystemExit("Sweep config must set scenario_config or scenario_configs.")
    if scenario_configs is None:
        scenario_configs = [str(scenario_config)]
    else:
        scenario_configs = [str(x) for x in scenario_configs]
    scheduler_configs = list(sweep_cfg["scheduler_configs"])
    runs_per_point = int(sweep_cfg.get("runs_per_point", 1))
    seeds = list(sweep_cfg.get("seeds", []))
    grid = sweep_cfg.get("grid", {}) or {}
    sweep_name = str(sweep_cfg.get("name", "sweep"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path("results/summaries") / f"{sweep_name}_{ts}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    points = _cartesian_product(keys, grid)

    if not points:
        points = [{}]

    total = 0
    for scen_cfg in scenario_configs:
        for sched_cfg in scheduler_configs:
            for point in points:
                for r in range(runs_per_point):
                    total += 1
                    seed = int(seeds[(total - 1) % len(seeds)]) if seeds else None

                    overrides = dict(point)
                    if seed is not None:
                        overrides["sim.seed"] = seed
                        if "sim.task_seed" not in overrides:
                            overrides["sim.task_seed"] = seed
                        if "sim.noise_seed" not in overrides:
                            overrides["sim.noise_seed"] = seed
                    overrides["output.summary_dir"] = str(summary_dir)

                    log.info("Sweep run #%d | scenario=%s | scheduler=%s | overrides=%s", total, scen_cfg, sched_cfg, overrides)
                    run_once(RunArgs(
                        config_path=base_config,
                        scenario_config_path=scen_cfg,
                        scheduler_config_path=str(sched_cfg),
                        overrides=overrides,
                    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vecsim", description="VEC single-queue scheduling simulator (SUMO+TraCI)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--log-file", default=None, help="Optional log file path")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single simulation")
    p_run.add_argument("--config", required=True, help="Base config YAML (e.g., configs/base.yaml)")
    p_run.add_argument("--scenario", required=True, help="Scenario config YAML (e.g., configs/scenario/x_intersection_2lane.yaml)")
    p_run.add_argument("--scheduler", required=True, help="Scheduler config YAML (e.g., configs/scheduler/pred_aware.yaml)")
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="Run a sweep experiment")
    p_sweep.add_argument("--config", required=True, help="Sweep config YAML (e.g., configs/sweep/mu_sweep.yaml)")
    p_sweep.set_defaults(func=cmd_sweep)

    return p


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()
    setup_logging(level=ns.log_level, log_file=ns.log_file)
    rc = ns.func(ns)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
