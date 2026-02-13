#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# NOTE: run with PYTHONPATH=src so these imports resolve.
from vecsim.models.rsu import RSU
from vecsim.prediction.alpha import alpha_from_rho, expected_compute_time_s, rho_future
from vecsim.sim.traci_api import TraciAPI
from vecsim.utils.yamlconf import load_merged, load_yaml


@dataclass(frozen=True)
class ProfileResult:
    name: str
    mean_alpha: float
    routes_path: str
    sumocfg_path: str
    scenario_path: str
    series_path: str


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


def _write_routes_piecewise(
    out_path: Path,
    segments: List[Dict[str, float]],
    duration_s: float,
    traffic_seed: int,
) -> None:
    directions = [
        ("W_to_E", "flow_WE"),
        ("E_to_W", "flow_EW"),
        ("S_to_N", "flow_SN"),
        ("N_to_S", "flow_NS"),
    ]

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

    ET.SubElement(root, "route", id="W_to_E", edges="w2c c2e")
    ET.SubElement(root, "route", id="E_to_W", edges="e2c c2w")
    ET.SubElement(root, "route", id="S_to_N", edges="s2c c2n")
    ET.SubElement(root, "route", id="N_to_S", edges="n2c c2s")

    departures: List[Tuple[float, str]] = []
    for i, (route_id, _) in enumerate(directions):
        for seg_idx, seg in enumerate(segments):
            t0 = float(seg["t_start_s"])
            t1 = float(seg["t_end_s"])
            rate = float(seg["rate_veh_per_s"])
            if t1 <= t0 or rate <= 0.0:
                continue
            rng = np.random.default_rng(int(traffic_seed) + i * 9973 + seg_idx * 101)
            for dt in _poisson_depart_times(rate / max(1, len(directions)), t1 - t0, rng):
                departures.append((t0 + float(dt), route_id))

    departures = [(t, r) for t, r in departures if 0.0 <= t <= duration_s]
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
    try:
        for _ in range(n_steps):
            traci.simulation_step()
            count = 0
            for vid in traci.get_all_vehicle_ids():
                snap = traci.get_vehicle_snapshot(vid)
                if rsu.in_coverage(snap.x_m, snap.y_m):
                    count += 1
            counts.append(int(count))
    finally:
        traci.close()
    return counts


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
    horizon_s: float,
    notes: str,
    segments: List[Dict[str, float]],
    mean_alpha: float,
) -> None:
    doc = {
        "sumo": {
            "sumocfg_path": str(sumocfg_path),
            "gui": False,
            "additional_traci_cmd": [],
        },
        "prediction": {
            "inflow_estimator": {"name": "from_timeseries"},
            "inflow_timeseries": {"path": str(inflow_series_path), "horizon_s": float(horizon_s)},
        },
        "traffic_profile": {
            "notes": notes,
            "mean_alpha": float(mean_alpha),
            "rate_veh_per_s_segments": segments,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(doc, f, sort_keys=False)


def _mean_alpha(series: List[Tuple[float, int, float]]) -> float:
    if not series:
        return 0.0
    return float(sum(a for _, _, a in series) / len(series))


def _clamp_rate(rate: float, rate_lo: float, rate_hi: float) -> float:
    return max(float(rate_lo), min(float(rate_hi), float(rate)))


def _mean_alpha_in_window(
    series: List[Tuple[float, int, float]],
    t_start: float,
    t_end: float,
    buffer_s: float = 0.0,
) -> float:
    if not series:
        return 0.0
    t0 = float(t_start) + float(buffer_s)
    t1 = float(t_end) - float(buffer_s)
    if t1 <= t0:
        t0 = float(t_start)
        t1 = float(t_end)
    vals = [a for t, _, a in series if t0 <= float(t) < t1]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _adjust_rate(rate: float, target_alpha: float, mean_alpha: float, step: float) -> float:
    if mean_alpha <= 0.0:
        return float(rate)
    ratio = float(target_alpha) / float(mean_alpha)
    if ratio <= 0.0:
        return float(rate)
    return float(rate) * float(ratio ** float(step))


def _evaluate_profile_series(
    segments: List[Dict[str, float]],
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
) -> List[Tuple[float, int, float]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        routes_path = tmpdir_path / "routes.rou.xml"
        sumocfg_path = tmpdir_path / "sumo.sumocfg"
        _write_routes_piecewise(routes_path, segments, duration_s, traffic_seed)
        _write_sumocfg(template_sumocfg, sumocfg_path, routes_path, duration_s, step_length_s)
        counts = _collect_coverage_counts(sumocfg_path, rsu, duration_s, step_length_s)
        return _alpha_series(
            coverage_counts=counts,
            horizon_s=horizon_s,
            mu=mu,
            e_comp_s=e_comp_s,
            duration_s=duration_s,
            step_length_s=step_length_s,
            warmup_s=warmup_s,
            cfg=cfg,
        )


def _evaluate_rate_for_segment(
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
    if duration_s <= 0.0:
        return 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        routes_path = tmpdir_path / "routes.rou.xml"
        sumocfg_path = tmpdir_path / "sumo.sumocfg"
        segments = [{"t_start_s": 0.0, "t_end_s": float(duration_s), "alpha": 0.0, "rate_veh_per_s": float(rate_veh_per_s)}]
        _write_routes_piecewise(routes_path, segments, duration_s, traffic_seed)
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
        return _mean_alpha(series)


def _calibrate_rate_for_target(
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
    max_iter: int,
    tol: float,
) -> Tuple[float, float]:
    rate = _initial_rate_from_alpha(cfg, target_alpha, horizon_s)
    rate = max(float(rate_lo), min(float(rate_hi), float(rate)))
    mean_alpha = 0.0
    for _ in range(max_iter):
        mean_alpha = _evaluate_rate_for_segment(
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
        if abs(mean_alpha - target_alpha) <= tol:
            break
        if mean_alpha > 0:
            rate = rate * (target_alpha / mean_alpha)
        rate = max(float(rate_lo), min(float(rate_hi), float(rate)))
    return rate, mean_alpha


def main() -> int:
    p = argparse.ArgumentParser(description="Generate time-varying alpha profiles for Experiment 3.")
    p.add_argument("--base-config", default="configs/base.yaml", help="Base config YAML")
    p.add_argument("--scenario-template", default="configs/scenario/x_intersection_2lane.yaml", help="Scenario config YAML")
    p.add_argument("--out-scenario-dir", default="configs/scenario/alpha_change_exp3_v2", help="Output scenario config dir")
    p.add_argument("--out-route-dir", default="scenarios/x_intersection_2lane/alpha_change_exp3_v2", help="Output route/sumocfg dir")
    p.add_argument("--traffic-seed", type=int, default=123, help="Seed for Poisson arrival generation")
    p.add_argument("--rate-lo", type=float, default=0.05, help="Lower bound for rate [veh/s]")
    p.add_argument("--rate-hi", type=float, default=15.0, help="Upper bound for rate [veh/s]")
    p.add_argument("--low-alpha", type=float, default=0.2, help="Low-band representative alpha")
    p.add_argument("--high-alpha", type=float, default=0.8, help="High-band representative alpha")
    p.add_argument("--random-low", type=float, default=0.1, help="Random alpha min")
    p.add_argument("--random-high", type=float, default=0.9, help="Random alpha max")
    p.add_argument("--random-seed", type=int, default=42, help="Seed for random alpha sequence")
    p.add_argument("--segment-s", type=float, default=300.0, help="Segment length for alternating/random profiles [s]")
    p.add_argument("--calib-iters", type=int, default=6, help="Iterations for per-segment rate calibration")
    p.add_argument("--calib-tol", type=float, default=0.02, help="Tolerance for per-segment alpha calibration")
    p.add_argument("--no-calibrate", action="store_true", help="Disable per-segment rate calibration")
    p.add_argument("--profile-calib-iters", type=int, default=10, help="Iterations for full-profile calibration")
    p.add_argument("--profile-calib-tol", type=float, default=0.02, help="Tolerance for full-profile calibration")
    p.add_argument("--profile-calib-step", type=float, default=0.6, help="Damping step for full-profile calibration")
    p.add_argument("--segment-buffer-s", type=float, default=0.0, help="Ignore buffer near segment boundaries [s]")
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

    mid_t = duration_s / 2.0
    low_a = float(args.low_alpha)
    high_a = float(args.high_alpha)

    profiles: Dict[str, List[Dict[str, float]]] = {
        "alpha_mid_high_to_low": [
            {"t_start_s": 0.0, "t_end_s": mid_t, "alpha": high_a},
            {"t_start_s": mid_t, "t_end_s": duration_s, "alpha": low_a},
        ],
        "alpha_mid_low_to_high": [
            {"t_start_s": 0.0, "t_end_s": mid_t, "alpha": low_a},
            {"t_start_s": mid_t, "t_end_s": duration_s, "alpha": high_a},
        ],
    }

    alt_segments: List[Dict[str, float]] = []
    seg_len = float(args.segment_s)
    t = 0.0
    toggle = True
    while t < duration_s:
        a = low_a if toggle else high_a
        alt_segments.append({"t_start_s": t, "t_end_s": min(duration_s, t + seg_len), "alpha": a})
        toggle = not toggle
        t += seg_len
    alt_name = f"alpha_alternate_{int(seg_len)}"
    profiles[alt_name] = alt_segments

    rng = np.random.default_rng(int(args.random_seed))
    rand_segments: List[Dict[str, float]] = []
    t = 0.0
    while t < duration_s:
        a = float(rng.uniform(float(args.random_low), float(args.random_high)))
        rand_segments.append({"t_start_s": t, "t_end_s": min(duration_s, t + seg_len), "alpha": a})
        t += seg_len
    rand_name = f"alpha_random_{int(seg_len)}"
    profiles[rand_name] = rand_segments

    results: List[ProfileResult] = []
    for name, segs in profiles.items():
        segments: List[Dict[str, float]] = []
        for seg in segs:
            a = float(seg["alpha"])
            seg_len = float(seg["t_end_s"]) - float(seg["t_start_s"])
            if args.no_calibrate:
                rate = _initial_rate_from_alpha(cfg, a, horizon_s)
                rate = _clamp_rate(rate, args.rate_lo, args.rate_hi)
                _ = 0.0
            else:
                rate, _ = _calibrate_rate_for_target(
                    target_alpha=a,
                    template_sumocfg=template_sumocfg,
                    rsu=rsu,
                    duration_s=seg_len,
                    step_length_s=step_length_s,
                    warmup_s=warmup_s,
                    horizon_s=horizon_s,
                    mu=mu,
                    e_comp_s=e_comp_s,
                    cfg=cfg,
                    traffic_seed=int(args.traffic_seed) + int(seg["t_start_s"]),
                    rate_lo=float(args.rate_lo),
                    rate_hi=float(args.rate_hi),
                    max_iter=int(args.calib_iters),
                    tol=float(args.calib_tol),
                )
            segments.append({
                "t_start_s": float(seg["t_start_s"]),
                "t_end_s": float(seg["t_end_s"]),
                "alpha": a,
                "rate_veh_per_s": rate,
            })

        if not args.no_calibrate and int(args.profile_calib_iters) > 0:
            for _ in range(int(args.profile_calib_iters)):
                series_tmp = _evaluate_profile_series(
                    segments=segments,
                    template_sumocfg=template_sumocfg,
                    rsu=rsu,
                    duration_s=duration_s,
                    step_length_s=step_length_s,
                    warmup_s=warmup_s,
                    horizon_s=horizon_s,
                    mu=mu,
                    e_comp_s=e_comp_s,
                    cfg=cfg,
                    traffic_seed=int(args.traffic_seed),
                )
                max_err = 0.0
                for seg in segments:
                    t0 = float(seg["t_start_s"])
                    t1 = float(seg["t_end_s"])
                    target = float(seg["alpha"])
                    mean_a = _mean_alpha_in_window(series_tmp, t0, t1, args.segment_buffer_s)
                    max_err = max(max_err, abs(mean_a - target))
                    seg["rate_veh_per_s"] = _clamp_rate(
                        _adjust_rate(float(seg["rate_veh_per_s"]), target, mean_a, args.profile_calib_step),
                        args.rate_lo,
                        args.rate_hi,
                    )
                if max_err <= float(args.profile_calib_tol):
                    break

        prof_dir = out_route_dir / name
        _ensure_dir(prof_dir)
        routes_path = prof_dir / "routes.rou.xml"
        sumocfg_path = prof_dir / "sumo.sumocfg"
        series_path = prof_dir / "n_true_series.csv"

        _write_routes_piecewise(routes_path, segments, duration_s, args.traffic_seed)
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
        segment_summary_path = prof_dir / "alpha_segment_summary.csv"
        with segment_summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_start_s", "t_end_s", "target_alpha", "mean_alpha"])
            for seg in segments:
                t0 = float(seg["t_start_s"])
                t1 = float(seg["t_end_s"])
                target = float(seg.get("alpha", 0.0))
                mean_a = _mean_alpha_in_window(series, t0, t1, args.segment_buffer_s)
                w.writerow([f"{t0:.3f}", f"{t1:.3f}", f"{target:.6f}", f"{mean_a:.6f}"])
        with series_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "n_true", "alpha_true"])
            for t_s, n_true, a in series:
                w.writerow([f"{t_s:.3f}", n_true, f"{a:.6f}"])

        scenario_path = out_scenario_dir / f"{name}.yaml"
        rel_sumocfg = Path(os.path.relpath(sumocfg_path, scenario_dir))
        rel_series = Path(os.path.relpath(series_path, scenario_dir))
        notes = f"generated exp3 profile: {name}"
        mean_alpha = _mean_alpha(series)
        _write_scenario_config(
            scenario_path,
            sumocfg_path=rel_sumocfg,
            inflow_series_path=rel_series,
            horizon_s=horizon_s,
            notes=notes,
            segments=segments,
            mean_alpha=mean_alpha,
        )

        results.append(ProfileResult(
            name=name,
            mean_alpha=mean_alpha,
            routes_path=str(routes_path),
            sumocfg_path=str(sumocfg_path),
            scenario_path=str(scenario_path),
            series_path=str(series_path),
        ))
        print(f"[{name}] mean_alpha={mean_alpha:.3f}")

    summary_path = out_scenario_dir / "alpha_change_exp3_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "mean_alpha", "routes_path", "sumocfg_path", "scenario_path", "series_path"])
        for r in results:
            w.writerow([r.name, r.mean_alpha, r.routes_path, r.sumocfg_path, r.scenario_path, r.series_path])

    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
