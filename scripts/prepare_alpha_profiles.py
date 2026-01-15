#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# NOTE: run with PYTHONPATH=src so these imports resolve.
from vecsim.models.rsu import RSU
from vecsim.prediction.alpha import alpha_from_rho, expected_compute_time_s, rho_future
from vecsim.sim.traci_api import TraciAPI
from vecsim.utils.yamlconf import load_merged, load_yaml


@dataclass(frozen=True)
class ProfileResult:
    target_alpha: float
    rate_veh_per_s: float
    mean_alpha: float
    routes_path: str
    sumocfg_path: str
    scenario_path: str


def _alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.1f}".replace(".", "p")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_sumo_cmd(sumocfg_path: Path) -> List[str]:
    sumo_home = os.environ.get("SUMO_HOME")
    sumo_bin = "sumo"
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo"
        if candidate.exists():
            sumo_bin = str(candidate)
    return [sumo_bin, "-c", str(sumocfg_path)]


def _write_sumocfg(
    template_path: Path,
    out_path: Path,
    route_file: Path,
    duration_s: float,
    step_length_s: float,
) -> None:
    tree = ET.parse(template_path)
    root = tree.getroot()

    net_path = (template_path.parent / "net.net.xml").resolve()
    add_path = (template_path.parent / "add.add.xml").resolve()

    inp = root.find("input")
    if inp is None:
        raise ValueError("sumocfg missing <input> section")

    net = inp.find("net-file")
    if net is not None:
        net.set("value", str(net_path))
    routes = inp.find("route-files")
    if routes is not None:
        routes.set("value", str(route_file.resolve()))
    add = inp.find("additional-files")
    if add is not None:
        add.set("value", str(add_path))

    time = root.find("time")
    if time is not None:
        begin = time.find("begin")
        if begin is not None:
            begin.set("value", "0")
        end = time.find("end")
        if end is not None:
            end.set("value", f"{duration_s:.3f}")
        step = time.find("step-length")
        if step is not None:
            step.set("value", f"{step_length_s:.3f}")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _poisson_depart_times(rate_per_s: float, duration_s: float, rng: np.random.Generator) -> List[float]:
    if rate_per_s <= 0.0:
        return []
    t = 0.0
    times: List[float] = []
    while True:
        t += rng.exponential(1.0 / rate_per_s)
        if t > duration_s:
            break
        times.append(float(t))
    return times


def _write_routes(
    out_path: Path,
    rate_total_per_s: float,
    duration_s: float,
    traffic_seed: int,
) -> None:
    directions = [
        ("W_to_E", "flow_WE"),
        ("E_to_W", "flow_EW"),
        ("S_to_N", "flow_SN"),
        ("N_to_S", "flow_NS"),
    ]
    rate_dir = float(rate_total_per_s) / max(1, len(directions))

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    ET.SubElement(
        root,
        "vType",
        id="car",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5.0",
        minGap="2.5",
        maxSpeed="13.9",
    )

    # Routes match scenarios/x_intersection_2lane/routes.rou.xml
    ET.SubElement(root, "route", id="W_to_E", edges="w2c c2e")
    ET.SubElement(root, "route", id="E_to_W", edges="e2c c2w")
    ET.SubElement(root, "route", id="S_to_N", edges="s2c c2n")
    ET.SubElement(root, "route", id="N_to_S", edges="n2c c2s")

    departures: List[Tuple[float, str]] = []
    for i, (route_id, _) in enumerate(directions):
        rng = np.random.default_rng(int(traffic_seed) + i * 9973)
        for depart_t in _poisson_depart_times(rate_dir, duration_s, rng):
            departures.append((float(depart_t), route_id))

    departures.sort(key=lambda x: x[0])
    for veh_id, (depart_t, route_id) in enumerate(departures, start=1):
        ET.SubElement(
            root,
            "vehicle",
            id=f"veh_{veh_id:08d}",
            type="car",
            route=route_id,
            depart=f"{depart_t:.3f}",
            departLane="random",
            departSpeed="max",
        )

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _collect_coverage_counts(
    sumocfg_path: Path,
    rsu: RSU,
    duration_s: float,
    step_length_s: float,
) -> List[int]:
    traci = TraciAPI()
    traci.start(_build_sumo_cmd(sumocfg_path))
    counts: List[int] = []
    n_steps = int(duration_s / step_length_s)
    now = 0.0
    try:
        for _ in range(n_steps):
            traci.simulation_step()
            count = 0
            for vid in traci.get_all_vehicle_ids():
                snap = traci.get_vehicle_snapshot(vid)
                inside = rsu.in_coverage(snap.x_m, snap.y_m)
                if inside:
                    count += 1
            counts.append(int(count))
            now += step_length_s
    finally:
        traci.close()
    return counts


def _mean_alpha(
    coverage_counts: List[int],
    horizon_s: float,
    mu: float,
    e_comp_s: float,
    duration_s: float,
    step_length_s: float,
    warmup_s: float,
    cfg: Dict[str, Any],
) -> float:
    series = _alpha_series(
        coverage_counts=coverage_counts,
        horizon_s=horizon_s,
        mu=mu,
        e_comp_s=e_comp_s,
        duration_s=duration_s,
        step_length_s=step_length_s,
        warmup_s=warmup_s,
        cfg=cfg,
    )
    if not series:
        return 0.0
    return float(sum(a for _, _, a in series) / len(series))


def _alpha_series(
    coverage_counts: List[int],
    horizon_s: float,
    mu: float,
    e_comp_s: float,
    duration_s: float,
    step_length_s: float,
    warmup_s: float,
    cfg: Dict[str, Any],
) -> List[Tuple[float, int, float]]:
    if horizon_s <= 0.0:
        return []
    if not coverage_counts:
        return []
    steps_ahead = max(1, int(math.ceil(horizon_s / step_length_s)))
    start_idx = max(0, int(math.ceil(warmup_s / step_length_s)))
    end_idx = len(coverage_counts) - 1 - steps_ahead
    out: List[Tuple[float, int, float]] = []
    if end_idx < start_idx:
        return []
    for idx in range(start_idx, end_idx + 1):
        t = idx * step_length_s
        n_true = coverage_counts[idx + steps_ahead]
        pred = cfg.get("prediction", {}) or {}
        mapping = (pred.get("alpha_mapping", {}) or {}).get("name", "sigmoid")
        if mapping == "sigmoid_capacity":
            cap_cfg = pred.get("capacity", {}) or {}
            n_c = float(cap_cfg.get("n_c", 1.0))
            rho = n_true / max(n_c, 1e-9)
        else:
            lambda_veh = n_true / horizon_s
            lambda_task = mu * lambda_veh
            rho = rho_future(lambda_task, e_comp_s)
        alpha = alpha_from_rho(rho, cfg=cfg)
        out.append((float(t), int(n_true), float(alpha)))
    return out


def _evaluate_rate(
    rate_veh_per_s: float,
    template_sumocfg: Path,
    rsu: RSU,
    duration_s: float,
    step_length_s: float,
    warmup_s: float,
    horizon_s: float,
    mu: float,
    e_comp_s: float,
    cfg: Dict[str, Any],
    traffic_seed: int,
) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        routes_path = tmpdir_path / "routes.rou.xml"
        sumocfg_path = tmpdir_path / "sumo.sumocfg"
        _write_routes(routes_path, rate_veh_per_s, duration_s, traffic_seed)
        _write_sumocfg(template_sumocfg, sumocfg_path, routes_path, duration_s, step_length_s)
        counts = _collect_coverage_counts(sumocfg_path, rsu, duration_s, step_length_s)
        return _mean_alpha(counts, horizon_s, mu, e_comp_s, duration_s, step_length_s, warmup_s, cfg=cfg)


def _find_rate_for_alpha(
    target_alpha: float,
    template_sumocfg: Path,
    rsu: RSU,
    duration_s: float,
    step_length_s: float,
    warmup_s: float,
    horizon_s: float,
    mu: float,
    e_comp_s: float,
    cfg: Dict[str, Any],
    traffic_seed: int,
    rate_lo: float,
    rate_hi: float,
    tol: float,
    max_iter: int,
) -> Tuple[float, float]:
    cache: Dict[float, float] = {}

    def eval_rate(rate: float) -> float:
        if rate in cache:
            return cache[rate]
        val = _evaluate_rate(
            rate_veh_per_s=rate,
            template_sumocfg=template_sumocfg,
            rsu=rsu,
            duration_s=duration_s,
            step_length_s=step_length_s,
            warmup_s=warmup_s,
            horizon_s=horizon_s,
            mu=mu,
            e_comp_s=e_comp_s,
            cfg=cfg,
            traffic_seed=traffic_seed,
        )
        cache[rate] = val
        return val

    lo = float(rate_lo)
    hi = float(rate_hi)
    a_lo = eval_rate(lo)
    a_hi = eval_rate(hi)

    expand = 0
    while a_hi < target_alpha and expand < 10:
        lo, a_lo = hi, a_hi
        hi *= 2.0
        a_hi = eval_rate(hi)
        expand += 1

    if target_alpha <= a_lo:
        return lo, a_lo
    if target_alpha >= a_hi:
        return hi, a_hi

    mid = (lo + hi) / 2.0
    a_mid = eval_rate(mid)
    for _ in range(max_iter):
        if abs(a_mid - target_alpha) <= tol:
            break
        if a_mid < target_alpha:
            lo = mid
        else:
            hi = mid
        mid = (lo + hi) / 2.0
        a_mid = eval_rate(mid)
    return mid, a_mid


def _inverse_sigmoid(alpha: float, a: float, b: float) -> float:
    alpha = min(0.999999, max(1e-6, float(alpha)))
    return float(b + (1.0 / a) * math.log(alpha / (1.0 - alpha)))


def _initial_rate_from_alpha(cfg: Dict[str, Any], target_alpha: float, horizon_s: float) -> float:
    pred = cfg.get("prediction", {}) or {}
    mapping = pred.get("alpha_mapping", {}) or {}
    name = str(mapping.get("name", "sigmoid_capacity"))
    params = mapping.get("params", {}) or {}
    a = float(params.get("a", 10.0))
    b = float(params.get("b", 0.8))
    if name == "sigmoid_capacity":
        cap_cfg = pred.get("capacity", {}) or {}
        n_c = float(cap_cfg.get("n_c", 1.0))
        rho_target = _inverse_sigmoid(target_alpha, a=a, b=b)
        n_target = max(0.0, rho_target * n_c)
        if horizon_s <= 0:
            return 0.0
        return float(n_target / horizon_s)
    rho_target = _inverse_sigmoid(target_alpha, a=a, b=b)
    if horizon_s <= 0:
        return 0.0
    return float(rho_target / horizon_s)


def _write_scenario_config(
    out_path: Path,
    sumocfg_path: Path,
    inflow_series_path: Path,
    duration_s: float,
    horizon_s: float,
    rate_veh_per_s: float,
    target_alpha: float,
    mean_alpha: float,
) -> None:
    rel_sumocfg = str(sumocfg_path)
    rel_series = str(inflow_series_path)
    doc = {
        "sumo": {
            "sumocfg_path": rel_sumocfg,
            "gui": False,
            "additional_traci_cmd": [],
        },
        "prediction": {
            "inflow_estimator": {"name": "from_timeseries"},
            "inflow_timeseries": {"path": rel_series, "horizon_s": float(horizon_s)},
        },
        "traffic_profile": {
            "target_alpha": float(target_alpha),
            "mean_alpha": float(mean_alpha),
            "rate_veh_per_s": float(rate_veh_per_s),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        import yaml  # local import to keep dependency localized

        yaml.safe_dump(doc, f, sort_keys=False)


def _parse_targets(spec: str) -> List[float]:
    if not spec:
        return [i / 10.0 for i in range(11)]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate 11 traffic profiles via alpha-targeted binary search.")
    p.add_argument("--base-config", default="configs/base.yaml", help="Base config YAML")
    p.add_argument("--scenario-template", default="configs/scenario/x_intersection_2lane.yaml", help="Scenario config YAML")
    p.add_argument("--out-scenario-dir", default="configs/scenario/alpha_profiles", help="Output scenario config dir")
    p.add_argument("--out-route-dir", default="scenarios/x_intersection_2lane/alpha_profiles", help="Output route/sumocfg dir")
    p.add_argument("--targets", default="", help="Comma-separated alpha targets (default 0.0..1.0)")
    p.add_argument("--traffic-seed", type=int, default=123, help="Seed for Poisson arrival generation")
    p.add_argument("--rate-lo", type=float, default=0.05, help="Initial lower bound for rate [veh/s]")
    p.add_argument("--rate-hi", type=float, default=1.0, help="Initial upper bound for rate [veh/s]")
    p.add_argument("--tol", type=float, default=0.02, help="Alpha tolerance")
    p.add_argument("--max-iter", type=int, default=12, help="Binary search iterations")
    args = p.parse_args()

    cfg = load_merged(args.base_config, args.scenario_template, None)
    sim = cfg.get("sim", {}) or {}
    rsu_cfg = cfg.get("rsu", {}) or {}
    pred = cfg.get("prediction", {}) or {}
    task = cfg.get("task_generation", {}) or {}

    duration_s = float(sim.get("duration_s", 600.0))
    step_length_s = float(sim.get("step_length_s", 0.2))
    warmup_s = float(sim.get("warmup_s", 0.0))
    horizon_s = float(pred.get("horizon_s", 5.0))
    mu = float(task.get("mu_tasks_per_s_per_vehicle", 0.1))

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

    e_comp_s = expected_compute_time_s(cfg)

    scenario_template_cfg = load_yaml(args.scenario_template)
    sumocfg_rel = scenario_template_cfg.get("sumo", {}).get("sumocfg_path", None)
    if not sumocfg_rel:
        raise SystemExit("Scenario template missing sumo.sumocfg_path")
    template_sumocfg = (Path(cfg.get("paths", {}).get("scenario_dir", "scenarios")) / sumocfg_rel).resolve()

    scenario_dir = Path(cfg.get("paths", {}).get("scenario_dir", "scenarios"))
    out_scenario_dir = Path(args.out_scenario_dir)
    out_route_dir = Path(args.out_route_dir)
    _ensure_dir(out_scenario_dir)
    _ensure_dir(out_route_dir)

    results: List[ProfileResult] = []
    targets = _parse_targets(args.targets)
    for target in targets:
        tag = _alpha_tag(target)
        # Binary search is disabled; use inverse-sigmoid init + proportional refinement.
        # rate, mean_alpha = _find_rate_for_alpha(...)
        rate = _initial_rate_from_alpha(cfg, target, horizon_s)
        rate = max(float(args.rate_lo), min(float(args.rate_hi), float(rate)))
        mean_alpha = 0.0
        for _ in range(int(args.max_iter)):
            mean_alpha = _evaluate_rate(
                rate_veh_per_s=rate,
                template_sumocfg=template_sumocfg,
                rsu=rsu,
                duration_s=duration_s,
                step_length_s=step_length_s,
                warmup_s=warmup_s,
                horizon_s=horizon_s,
                mu=mu,
                e_comp_s=e_comp_s,
                cfg=cfg,
                traffic_seed=args.traffic_seed,
            )
            if abs(mean_alpha - target) <= args.tol:
                break
            if mean_alpha > 0:
                rate = rate * (target / mean_alpha)
            rate = max(float(args.rate_lo), min(float(args.rate_hi), float(rate)))

        prof_dir = out_route_dir / tag
        _ensure_dir(prof_dir)
        routes_path = prof_dir / "routes.rou.xml"
        sumocfg_path = prof_dir / "sumo.sumocfg"
        series_path = prof_dir / "n_true_series.csv"
        _write_routes(routes_path, rate, duration_s, args.traffic_seed)
        _write_sumocfg(template_sumocfg, sumocfg_path, routes_path, duration_s, step_length_s)

        counts = _collect_coverage_counts(sumocfg_path, rsu, duration_s, step_length_s)
        series = _alpha_series(
            coverage_counts=counts,
            horizon_s=horizon_s,
            mu=mu,
            e_comp_s=e_comp_s,
            duration_s=duration_s,
            step_length_s=step_length_s,
            warmup_s=warmup_s,
            cfg=cfg,
        )
        with series_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "n_true", "alpha_true"])
            for t_s, n_true, a in series:
                w.writerow([f"{t_s:.3f}", n_true, f"{a:.6f}"])

        scenario_path = out_scenario_dir / f"{tag}.yaml"
        rel_sumocfg = Path(os.path.relpath(sumocfg_path, scenario_dir))
        rel_series = Path(os.path.relpath(series_path, scenario_dir))
        _write_scenario_config(
            scenario_path,
            sumocfg_path=rel_sumocfg,
            inflow_series_path=rel_series,
            duration_s=duration_s,
            horizon_s=horizon_s,
            rate_veh_per_s=rate,
            target_alpha=target,
            mean_alpha=mean_alpha,
        )

        results.append(ProfileResult(
            target_alpha=target,
            rate_veh_per_s=rate,
            mean_alpha=mean_alpha,
            routes_path=str(routes_path),
            sumocfg_path=str(sumocfg_path),
            scenario_path=str(scenario_path),
        ))
        print(f"[{tag}] rate={rate:.4f} veh/s | mean_alpha={mean_alpha:.3f}")

    # Write summary CSV
    summary_path = out_scenario_dir / "alpha_profile_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_alpha", "rate_veh_per_s", "mean_alpha", "routes_path", "sumocfg_path", "scenario_path"])
        for r in results:
            w.writerow([r.target_alpha, r.rate_veh_per_s, r.mean_alpha, r.routes_path, r.sumocfg_path, r.scenario_path])

    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
