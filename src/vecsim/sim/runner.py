from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_right
import csv

import numpy as np

from ..metrics.recorder import EventRecorder
from ..metrics.summary import summarize_tasks, write_summary_csv
from ..models.mec_server import MECServer
from ..models.queue import TaskQueue
from ..models.rsu import RSU
from ..models.task import Task, TaskStatus
from ..models.vehicle import Vehicle
from ..prediction.alpha import alpha_from_rho, expected_compute_time_s, rho_future
from ..prediction.candidates import select_candidate_vehicle_ids
from ..prediction.inflow_estimator import apply_inflow_error, expected_inflow_vehicles, lambda_task_hat
from ..schedulers.base import SchedulerContext, s_total
from ..schedulers.factory import make_scheduler
from ..traffic.mu_schedule import mu_at_time, parse_mu_schedule
from ..traffic.task_generator import TaskFactory, poisson_event
from ..comm.uplink import compute_uplink_time_for_task
from ..utils.yamlconf import apply_overrides, load_merged
from .clock import SimClock
from .events import SimEvent
from .seed import set_global_seed
from .traci_api import TraciAPI

log = logging.getLogger(__name__)


@dataclass
class RunArgs:
    config_path: str
    scenario_config_path: Optional[str] = None
    scheduler_config_path: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


@dataclass
class InTransit:
    task_id: str
    ul_finish_t: float


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    scenario_dir = Path(cfg.get("paths", {}).get("scenario_dir", "scenarios"))
    sumocfg = cfg.get("sumo", {}).get("sumocfg_path")
    if sumocfg and not str(sumocfg).startswith("/"):
        cfg["sumo"]["sumocfg_path"] = str(scenario_dir / sumocfg)
    return cfg


def build_sumo_cmd(cfg: Dict[str, Any]) -> List[str]:
    sumo_home = os.environ.get("SUMO_HOME")
    sumo_bin = "sumo-gui" if cfg.get("sumo", {}).get("gui", False) else "sumo"
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / sumo_bin
        if candidate.exists():
            sumo_bin = str(candidate)

    sumocfg_path = cfg.get("sumo", {}).get("sumocfg_path")
    if not sumocfg_path:
        raise ValueError("sumo.sumocfg_path is not set (see configs/scenario/*.yaml).")

    cmd = [sumo_bin, "-c", str(sumocfg_path)]
    extra = cfg.get("sumo", {}).get("additional_traci_cmd", []) or []
    cmd.extend([str(x) for x in extra])
    return cmd


def _drop_expired_in_queue(queue: TaskQueue, now: float, rec: EventRecorder) -> None:
    keep: list[Task] = []
    for t in queue.tasks:
        if t.meta.get("coverage_out_t") is not None:
            t.status = TaskStatus.DEADLINE_MISS
            t.meta["miss_where"] = "coverage"
            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "coverage"}))
        elif now > t.effective_deadline_t:
            t.status = TaskStatus.DEADLINE_MISS
            t.meta["miss_where"] = "queue"
            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "queue"}))
        else:
            keep.append(t)
    queue.tasks = keep


def _update_task_deadline(task: Task, now: float, vehicle: Optional[Vehicle], rsu: RSU, cfg: Dict[str, Any]) -> None:
    remain_deadline = float(task.deadline_t - now)
    if vehicle is None:
        stay_s = 0.0
        task.meta["coverage_out_t"] = now
    else:
        stay_s = float(vehicle.estimate_remaining_stay_time(rsu, cfg))
        if not vehicle.in_coverage:
            task.meta["coverage_out_t"] = now
    task.stay_deadline_t = now + max(0.0, stay_s)
    eff_remain = min(remain_deadline, stay_s)
    # eff_remain = remain_deadline
    task.effective_deadline_t = now + max(0.0, eff_remain)


def _maybe_drop_impossible_next(queue: TaskQueue, mec: MECServer, now: float, rec: EventRecorder, enabled: bool) -> None:
    if not enabled:
        return
    keep: list[Task] = []
    for t in queue.tasks:
        R = t.effective_deadline_t - now
        need = mec.w0() + s_total(t)
        if R <= 0 or need > R:
            t.status = TaskStatus.DROPPED
            t.drop_reason = "impossible_if_next"
            rec.record(SimEvent(t=now, etype="DROPPED", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"reason": t.drop_reason}))
        else:
            keep.append(t)
    queue.tasks = keep


def run_once(args: RunArgs) -> int:
    # 1) Load + merge configs
    cfg = load_merged(args.config_path, args.scenario_config_path, args.scheduler_config_path)
    cfg = apply_overrides(cfg, args.overrides)
    cfg = _resolve_paths(cfg)

    # 2) Seed / RNG
    sim_cfg = cfg.get("sim", {}) or {}
    seed = int(sim_cfg.get("seed", 0))
    task_seed = int(sim_cfg.get("task_seed", seed))
    noise_seed = int(sim_cfg.get("noise_seed", seed))
    set_global_seed(seed)
    rng_task = np.random.default_rng(task_seed)
    rng_pred = np.random.default_rng(noise_seed)

    # 3) Sim clock
    step_length = float(cfg.get("sim", {}).get("step_length_s", 0.2))
    duration_s = float(cfg.get("sim", {}).get("duration_s", 600.0))
    n_steps = int(duration_s / step_length)
    clock = SimClock(step_length_s=step_length)

    # 4) Entities: RSU, MEC, Scheduler
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

    mec_cfg = cfg.get("mec", {}) or {}
    queue = TaskQueue()
    mec = MECServer(cpu_freq_hz=float(mec_cfg.get("cpu_freq_hz", 3e9)), queue=queue, non_preemptive=bool(mec_cfg.get("non_preemptive", True)))

    scheduler = make_scheduler(cfg)
    sched_params = (cfg.get("scheduler", {}) or {}).get("params", {}) or {}

    # 5) Traffic/task generation
    tf = TaskFactory(cfg=cfg, rng=rng_task)
    default_mu = float((cfg.get("task_generation", {}) or {}).get("mu_tasks_per_s_per_vehicle", 0.05))
    mu_schedule = parse_mu_schedule(cfg)  # from scenario burst config
    # 6) Prediction for alpha(t)
    pred_cfg = cfg.get("prediction", {}) or {}
    horizon_s = float(sched_params.get("horizon_s", pred_cfg.get("horizon_s", 5.0)))
    e_comp_s = expected_compute_time_s(cfg)
    high_thr = float((cfg.get("priority", {}) or {}).get("high_threshold", 0.7))
    alpha_true_series: Optional[Tuple[List[float], List[float]]] = None
    ts_cfg = pred_cfg.get("inflow_timeseries", {}) or {}
    ts_path = str(ts_cfg.get("path", ""))
    if ts_path:
        scenario_dir = Path(str((cfg.get("paths", {}) or {}).get("scenario_dir", "scenarios")))
        full_path = scenario_dir / ts_path
        times: List[float] = []
        alpha_vals: List[float] = []
        try:
            with full_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        t = float(row.get("t_s", ""))
                        a = float(row.get("alpha_true", ""))
                    except (TypeError, ValueError):
                        continue
                    times.append(t)
                    alpha_vals.append(a)
        except FileNotFoundError:
            times, alpha_vals = [], []
        if times and alpha_vals:
            alpha_true_series = (times, alpha_vals)

    # 7) Outputs
    out_cfg = cfg.get("output", {}) or {}
    run_name = str(out_cfg.get("run_name", "dev"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(str(out_cfg.get("save_dir", "results/raw")))
    summary_dir = Path(str(out_cfg.get("summary_dir", "results/summaries")))
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"events_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.jsonl"
    decision_path = raw_dir / f"decision_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.jsonl"
    candidate_path = raw_dir / f"candidates_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.jsonl"
    outcome_path = raw_dir / f"outcomes_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.jsonl"
    scenario_tag = str(args.scenario_config_path or "no_scenario").replace("/", "__")
    summary_path = summary_dir / f"summary_{run_name}_{scheduler.name}_{scenario_tag}.csv"

    utility_beta = float(out_cfg.get("utility_beta", 0.5))

    # Admission rule
    drop_impossible = bool((((mec_cfg.get("queue", {}) or {}).get("admission", {}) or {}).get("drop_if_impossible_next", True)))
    drop_impossible_scheds = {"v_edf"}

    # 8) State containers
    vehicles: dict[str, Vehicle] = {}
    tasks: dict[str, Task] = {}               # task_id -> Task (for evaluation)
    in_transit: list[InTransit] = []          # task_id with uplink finish time

    rec = EventRecorder()
    traci = TraciAPI()
    log_rankings = scheduler.name in {"priority", "prediction_aware"}
    decision_seq = 0
    decision_by_task: dict[str, str] = {}
    decision_alpha_by_task: dict[str, float] = {}
    decision_t_by_task: dict[str, float] = {}
    if log_rankings:
        decision_f = decision_path.open("w", encoding="utf-8")
        candidate_f = candidate_path.open("w", encoding="utf-8")
        outcome_f = outcome_path.open("w", encoding="utf-8")
    else:
        decision_f = candidate_f = outcome_f = None
    sumo_cmd = build_sumo_cmd(cfg)

    log.info("Run start: scheduler=%s, steps=%d, dt=%.3f, duration=%.1f, seed=%d", scheduler.name, n_steps, step_length, duration_s, seed)
    traci.start(sumo_cmd)

    try:
        for _ in range(n_steps):
            now = clock.t
            traci.simulation_step()

            # --- Vehicle lifecycle ---
            for vid in traci.get_departed_vehicles():
                vehicles[vid] = Vehicle(veh_id=vid)
            for vid in traci.get_arrived_vehicles():
                vehicles.pop(vid, None)

            # Update snapshots
            for vid in list(vehicles.keys()):
                snap = traci.get_vehicle_snapshot(vid)
                vehicles[vid].update_state(snap.x_m, snap.y_m, snap.speed_mps, snap.heading_deg, now, rsu)

            # --- Task generation rate mu(t) ---
            mu = mu_at_time(default_mu=default_mu, schedule=mu_schedule, now_s=now)

            # --- Generate tasks (Poisson, only while in RSU coverage) ---
            alpha_true_now = None
            if alpha_true_series is not None:
                times, vals = alpha_true_series
                idx = max(0, bisect_right(times, float(now)) - 1)
                alpha_true_now = float(vals[idx])
            for v in vehicles.values():
                if not v.in_coverage:
                    continue
                if poisson_event(rate_per_s=mu, dt_s=step_length, rng=rng_task):
                    tg_cfg = cfg.get("task_generation", {}) or {}
                    burst_size = int(tg_cfg.get("burst_size", 1))
                    burst_size = max(1, burst_size)
                    # Remaining stay time estimate (straight-line approximation)
                    stay_s = v.estimate_remaining_stay_time(rsu, cfg)
                    for _ in range(burst_size):
                        task = tf.make_task(vehicle_id=v.veh_id, now=now, predicted_stay_s=stay_s, mec_cpu_freq_hz=mec.cpu_freq_hz)
                        if alpha_true_now is not None:
                            task.meta["alpha_true_at_create"] = alpha_true_now

                        # Uplink time (explicitly modeled)
                        t_ul = compute_uplink_time_for_task(cfg, v, rsu, task.input_bits, rng_task)
                        task.uplink_time_s = float(t_ul)

                        tasks[task.task_id] = task
                        rec.record(SimEvent(t=now, etype="TASK_CREATED", task_id=task.task_id, vehicle_id=task.vehicle_id))
                        rec.record(SimEvent(t=now, etype="UL_STARTED", task_id=task.task_id, vehicle_id=task.vehicle_id))

                        task.status = TaskStatus.UL_IN_PROGRESS
                        in_transit.append(InTransit(task_id=task.task_id, ul_finish_t=now + task.uplink_time_s))

            # --- Refresh effective deadlines based on current vehicle state ---
            for t in tasks.values():
                if t.status in (TaskStatus.FINISHED, TaskStatus.DEADLINE_MISS, TaskStatus.DROPPED):
                    continue
                v = vehicles.get(t.vehicle_id)
                _update_task_deadline(t, now, v, rsu, cfg)

            # --- Uplink completion -> enqueue ---
            if in_transit:
                ignore_ul_coverage = bool(cfg.get("sim", {}).get("ignore_ul_coverage", False))
                def _ul_diag(t: Task, now_s: float) -> dict:
                    stay_deadline_t = float(getattr(t, "stay_deadline_t", t.deadline_t))
                    remain_deadline = float(t.deadline_t - now_s)
                    remain_stay = float(stay_deadline_t - now_s)
                    r_eff = min(remain_deadline, remain_stay)
                    t_ul = float(t.uplink_time_s)
                    c_s = float(t.compute_time_s)
                    min_need = t_ul + c_s
                    bottleneck = "deadline" if remain_deadline <= remain_stay else "stay"
                    margin_s = r_eff - min_need
                    v = vehicles.get(t.vehicle_id)
                    speed_mps = float(v.speed_mps) if v is not None else None
                    dist_to_exit_m = None
                    if v is not None:
                        rect = rsu.coverage_rect_bounds()
                        theta = math.radians(float(v.heading_deg))
                        dx = math.cos(theta)
                        dy = math.sin(theta)
                        if rect is not None:
                            x_min, x_max, y_min, y_max = rect
                            if abs(dx) >= abs(dy):
                                d_out = (x_max - v.x_m) if dx >= 0 else (v.x_m - x_min)
                            else:
                                d_out = (y_max - v.y_m) if dy >= 0 else (v.y_m - y_min)
                            dist_to_exit_m = max(0.0, float(d_out))
                        else:
                            fx = v.x_m - rsu.x_m
                            fy = v.y_m - rsu.y_m
                            b = fx * dx + fy * dy
                            c = fx * fx + fy * fy - (rsu.coverage_radius_m ** 2)
                            disc = b * b - c
                            if disc > 0.0:
                                t_out = -b + math.sqrt(disc)
                                if t_out > 0.0:
                                    dist_to_exit_m = float(t_out)
                    return {
                        "remain_deadline_raw_s": remain_deadline,
                        "remain_stay_raw_s": remain_stay,
                        "r_eff_s": float(max(0.0, r_eff)),
                        "bottleneck": bottleneck,
                        "t_ul_s": t_ul,
                        "c_s": c_s,
                        "min_need_s": min_need,
                        "margin_s": margin_s,
                        "feasible": bool(min_need <= r_eff),
                        "speed_mps": speed_mps,
                        "dist_to_exit_m": dist_to_exit_m,
                    }
                still: list[InTransit] = []
                for it in in_transit:
                    if it.ul_finish_t <= now:
                        t = tasks[it.task_id]
                        ul_details = _ul_diag(t, now)
                        # NOTE: UL完了時のcoverage判定を一時的に無効化する場合はここをFalseにする。
                        if False and (not ignore_ul_coverage) and t.meta.get("coverage_out_t") is not None:
                            t.status = TaskStatus.DEADLINE_MISS
                            t.meta["miss_where"] = "coverage"
                            rec.record(SimEvent(t=now, etype="UL_FINISHED", task_id=t.task_id, vehicle_id=t.vehicle_id, details=ul_details))
                            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "coverage"}))
                            continue
                        # If already missed deadline by uplink completion, mark miss
                        if now > t.effective_deadline_t:
                            t.status = TaskStatus.DEADLINE_MISS
                            t.meta["miss_where"] = "uplink"
                            rec.record(SimEvent(t=now, etype="UL_FINISHED", task_id=t.task_id, vehicle_id=t.vehicle_id, details=ul_details))
                            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "uplink"}))
                            continue

                        # Optional drop-before-enqueue for specific schedulers
                        if drop_impossible and scheduler.name in drop_impossible_scheds:
                            R = t.effective_deadline_t - now
                            need = mec.w0() + s_total(t)
                            if R <= 0 or need > R:
                                t.status = TaskStatus.DROPPED
                                t.drop_reason = "impossible_if_next"
                                rec.record(SimEvent(t=now, etype="DROPPED", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"reason": t.drop_reason}))
                                continue

                        t.status = TaskStatus.READY
                        t.enqueued_t = now
                        rec.record(SimEvent(t=now, etype="UL_FINISHED", task_id=t.task_id, vehicle_id=t.vehicle_id, details=ul_details))
                        rec.record(SimEvent(t=now, etype="ENQUEUED", task_id=t.task_id, vehicle_id=t.vehicle_id))
                        queue.enqueue(t)
                    else:
                        still.append(it)
                in_transit = still

            # --- Queue sanitation: expired/impossible ---
            _drop_expired_in_queue(queue, now, rec)
            _maybe_drop_impossible_next(queue, mec, now, rec, enabled=False)

            # --- Prediction alpha(t) ---
            cand_ids = select_candidate_vehicle_ids(vehicles, rsu, cfg)
            exp_in_true = expected_inflow_vehicles(
                candidate_vehicle_ids=cand_ids,
                now_s=now,
                horizon_s=horizon_s,
                cfg=cfg,
                rng=rng_pred,
            )
            exp_in_hat = apply_inflow_error(exp_in_true, cfg=cfg, rng=rng_pred)
            lam_hat = lambda_task_hat(expected_inflow=exp_in_hat, horizon_s=horizon_s, mu_tasks_per_s_per_vehicle=mu)
            mapping = (pred_cfg.get("alpha_mapping", {}) or {}).get("name", "sigmoid")
            if mapping == "sigmoid_capacity":
                cap_cfg = pred_cfg.get("capacity", {}) or {}
                n_c = float(cap_cfg.get("n_c", 1.0))
                rho = exp_in_hat / max(n_c, 1e-9)
            else:
                rho = rho_future(lam_hat, e_comp_s)
            alpha = alpha_from_rho(rho, cfg=cfg)

            # --- Scheduler reorder ---
            if len(queue) > 1:
                ctx = SchedulerContext(now=now, w0=mec.w0(), alpha=alpha, cfg=sched_params)
                queue.tasks = scheduler.rank(list(queue.tasks), ctx)

            # --- CPU start/step ---
            if mec.is_idle():
                if log_rankings and len(queue) > 0:
                    decision_seq += 1
                    decision_id = f"d{decision_seq:08d}"
                    cand = list(queue.tasks)
                    # Priority order
                    priority_order = sorted(cand, key=lambda t: -t.priority)
                    # Proposed order (prediction_aware)
                    max_r = 0.0
                    for t in cand:
                        r = max(0.0, float(t.effective_deadline_t - now))
                        if r > max_r:
                            max_r = r

                    def prop_score(t: Task) -> float:
                        r = max(0.0, float(t.effective_deadline_t - now))
                        f_dense = float(t.priority)
                        f_sparse = 1.0 - (r / (max_r + 1e-9))
                        return alpha * f_dense + (1.0 - alpha) * f_sparse

                    proposed_order = sorted(cand, key=lambda t: -prop_score(t))
                    if scheduler.name == "priority":
                        chosen = priority_order[0]
                    else:
                        chosen = proposed_order[0]
                    decision_by_task[chosen.task_id] = decision_id
                    decision_alpha_by_task[chosen.task_id] = float(alpha)
                    decision_t_by_task[chosen.task_id] = float(now)
                    # Top-K diff
                    k = 10
                    topk_pri = [t.task_id for t in priority_order[:k]]
                    topk_prop = [t.task_id for t in proposed_order[:k]]
                    diff_positions = [i for i in range(min(k, len(topk_pri), len(topk_prop))) if topk_pri[i] != topk_prop[i]]
                    decision_obj = {
                        "t": float(now),
                        "algo": scheduler.name,
                        "alpha": float(alpha),
                        "queue_snapshot_id": decision_id,
                        "chosen_task_id": chosen.task_id,
                        "chosen_rank_in_algo": (priority_order.index(chosen) + 1) if scheduler.name == "priority" else (proposed_order.index(chosen) + 1),
                        "chosen_rank_in_priority": priority_order.index(chosen) + 1,
                        "topk_diff": {
                            "k": k,
                            "diff_positions": diff_positions,
                            "diff_count": len(diff_positions),
                            "priority_topk": topk_pri,
                            "proposed_topk": topk_prop,
                        },
                    }
                    decision_f.write(json.dumps(decision_obj, ensure_ascii=False) + "\n")
                    # Candidate logs: top-K per algo + ensure chosen is included
                    for algo_name, ordered in (("priority", priority_order), ("prediction_aware", proposed_order)):
                        top = ordered[:k]
                        if chosen not in top:
                            top = top + [chosen]
                        for rank, t in enumerate(top, start=1):
                            r = max(0.0, float(t.effective_deadline_t - now))
                            f_dense = float(t.priority)
                            f_sparse = 1.0 - (r / (max_r + 1e-9))
                            p_score = prop_score(t)
                            cand_obj = {
                                "t": float(now),
                                "algo": algo_name,
                                "queue_snapshot_id": decision_id,
                                "task_id": t.task_id,
                                "rank_in_algo": rank,
                                "prio_score": float(t.priority),
                                "prop_score": float(p_score),
                                "prop_terms": {
                                    "alpha": float(alpha),
                                    "f_dense": f_dense,
                                    "f_sparse": f_sparse,
                                    "r_eff_s": r,
                                    "max_r": float(max_r),
                                },
                                "sort_key": (-p_score,),
                            }
                            candidate_f.write(json.dumps(cand_obj, ensure_ascii=False) + "\n")
                started = mec.start_next_if_idle(now)
                if started is not None:
                    started.meta["effective_deadline_at_start"] = started.effective_deadline_t
                    rec.record(SimEvent(t=now, etype="CPU_STARTED", task_id=started.task_id, vehicle_id=started.vehicle_id))

            finished = mec.step(dt=step_length, now=now + step_length)
            if finished is not None:
                if log_rankings:
                    decision_id = decision_by_task.get(finished.task_id)
                    alpha_at_decision = decision_alpha_by_task.get(finished.task_id)
                    decision_t = decision_t_by_task.get(finished.task_id)
                if finished.cpu_finish_t is not None:
                    v = vehicles.get(finished.vehicle_id)
                    _update_task_deadline(finished, finished.cpu_finish_t, v, rsu, cfg)
                if finished.meta.get("coverage_out_t") is not None:
                    finished.status = TaskStatus.DEADLINE_MISS
                    finished.meta["miss_where"] = "coverage"
                    rec.record(SimEvent(t=finished.cpu_finish_t or (now + step_length), etype="DEADLINE_MISS", task_id=finished.task_id, vehicle_id=finished.vehicle_id, details={"where": "coverage"}))
                if finished.cpu_finish_t is not None and finished.cpu_finish_t > finished.effective_deadline_t:
                    finished.status = TaskStatus.DEADLINE_MISS
                    finished.meta["miss_where"] = "cpu"
                    rec.record(SimEvent(t=finished.cpu_finish_t, etype="DEADLINE_MISS", task_id=finished.task_id, vehicle_id=finished.vehicle_id, details={"where": "cpu"}))
                else:
                    rec.record(SimEvent(t=finished.cpu_finish_t or (now + step_length), etype="CPU_FINISHED", task_id=finished.task_id, vehicle_id=finished.vehicle_id))
                if log_rankings and finished.cpu_finish_t is not None:
                    tardiness_s = max(0.0, float(finished.cpu_finish_t - finished.effective_deadline_t))
                    met_deadline = bool(finished.cpu_finish_t <= finished.effective_deadline_t)
                    margin = None
                    if finished.cpu_start_t is not None:
                        eff_start = finished.meta.get("effective_deadline_at_start")
                        if eff_start is not None:
                            margin = float(eff_start) - float(finished.cpu_start_t) - float(finished.compute_time_s)
                    reward = 0.0
                    if met_deadline and margin is not None:
                        reward = float(finished.priority) + float(utility_beta) * float(margin)
                    out_obj = {
                        "task_id": finished.task_id,
                        "decision_id": decision_id,
                        "decision_t": decision_t,
                        "decision_alpha": alpha_at_decision,
                        "started_at": finished.cpu_start_t,
                        "finished_at": finished.cpu_finish_t,
                        "met_deadline": met_deadline,
                        "tardiness_s": tardiness_s,
                        "reward": reward,
                    }
                    outcome_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            clock.tick()

        # --- Persist raw + summary ---
        rec.dump_jsonl(str(raw_path))
        scenario_name = args.scenario_config_path or ""
        alpha_true_mean = None
        if alpha_true_series is not None:
            _, vals = alpha_true_series
            if vals:
                alpha_true_mean = float(sum(vals) / len(vals))
        summary = summarize_tasks(
            tasks=tasks.values(),
            scenario=scenario_name,
            run_name=run_name,
            scheduler=scheduler.name,
            seed=seed,
            high_thr=high_thr,
            utility_beta=utility_beta,
            sim_duration_s=duration_s,
            alpha_true_mean=alpha_true_mean,
        )
        write_summary_csv(summary, str(summary_path))

        log.info("Run finished. Raw events: %s", raw_path)
        log.info("Run summary: %s", summary_path)
        return 0
    finally:
        if log_rankings and decision_f is not None:
            decision_f.close()
            candidate_f.close()
            outcome_f.close()
        traci.close()


def main() -> None:
    raise SystemExit("Use `python -m vecsim.cli ...`")
