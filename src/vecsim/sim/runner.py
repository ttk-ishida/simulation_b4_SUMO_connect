from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from ..prediction.inflow_estimator import expected_inflow_vehicles, lambda_task_hat
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
        if now > t.effective_deadline_t:
            t.status = TaskStatus.DEADLINE_MISS
            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "queue"}))
        else:
            keep.append(t)
    queue.tasks = keep


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
    seed = int(cfg.get("sim", {}).get("seed", 0))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    # 3) Sim clock
    step_length = float(cfg.get("sim", {}).get("step_length_s", 0.2))
    duration_s = float(cfg.get("sim", {}).get("duration_s", 600.0))
    n_steps = int(duration_s / step_length)
    clock = SimClock(step_length_s=step_length)

    # 4) Entities: RSU, MEC, Scheduler
    rsu_cfg = cfg.get("rsu", {}) or {}
    rsu = RSU(
        rsu_id=str(rsu_cfg.get("id", "rsu0")),
        x_m=float(rsu_cfg.get("x_m", 0.0)),
        y_m=float(rsu_cfg.get("y_m", 0.0)),
        coverage_radius_m=float(rsu_cfg.get("coverage_radius_m", 250.0)),
    )

    mec_cfg = cfg.get("mec", {}) or {}
    queue = TaskQueue()
    mec = MECServer(cpu_freq_hz=float(mec_cfg.get("cpu_freq_hz", 3e9)), queue=queue, non_preemptive=bool(mec_cfg.get("non_preemptive", True)))

    scheduler = make_scheduler(cfg)
    sched_params = (cfg.get("scheduler", {}) or {}).get("params", {}) or {}

    # 5) Traffic/task generation
    tf = TaskFactory(cfg=cfg, rng=rng)
    default_mu = float((cfg.get("task_generation", {}) or {}).get("mu_tasks_per_s_per_vehicle", 0.05))
    mu_schedule = parse_mu_schedule(cfg)  # from scenario burst config
    # 6) Prediction for alpha(t)
    pred_cfg = cfg.get("prediction", {}) or {}
    horizon_s = float(sched_params.get("horizon_s", pred_cfg.get("horizon_s", 5.0)))
    e_comp_s = expected_compute_time_s(cfg)
    high_thr = float((cfg.get("priority", {}) or {}).get("high_threshold", 0.7))

    # 7) Outputs
    out_cfg = cfg.get("output", {}) or {}
    run_name = str(out_cfg.get("run_name", "dev"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(str(out_cfg.get("save_dir", "results/raw")))
    summary_dir = Path(str(out_cfg.get("summary_dir", "results/summaries")))
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"events_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.jsonl"
    summary_path = summary_dir / f"summary_{run_name}_{scheduler.name}_seed{seed}_{timestamp}.csv"

    utility_beta = float(out_cfg.get("utility_beta", 0.5))

    # Admission rule
    drop_impossible = bool((((mec_cfg.get("queue", {}) or {}).get("admission", {}) or {}).get("drop_if_impossible_next", True)))

    # 8) State containers
    vehicles: dict[str, Vehicle] = {}
    tasks: dict[str, Task] = {}               # task_id -> Task (for evaluation)
    in_transit: list[InTransit] = []          # task_id with uplink finish time

    rec = EventRecorder()
    traci = TraciAPI()
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
                vehicles[vid].update_state(snap.x_m, snap.y_m, snap.speed_mps, now, rsu)

            # --- Task generation rate mu(t) ---
            mu = mu_at_time(default_mu=default_mu, schedule=mu_schedule, now_s=now)

            # --- Generate tasks (Poisson, only while in RSU coverage) ---
            for v in vehicles.values():
                if not v.in_coverage:
                    continue
                if poisson_event(rate_per_s=mu, dt_s=step_length, rng=rng):
                    # Remaining stay time estimate (stub now; replace later)
                    stay_s = v.estimate_remaining_stay_time(now, default_s=horizon_s)
                    task = tf.make_task(vehicle_id=v.veh_id, now=now, predicted_stay_s=stay_s, mec_cpu_freq_hz=mec.cpu_freq_hz)

                    # Uplink time (explicitly modeled)
                    t_ul = compute_uplink_time_for_task(cfg, v, rsu, task.input_bits, rng)
                    task.uplink_time_s = float(t_ul)

                    tasks[task.task_id] = task
                    rec.record(SimEvent(t=now, etype="TASK_CREATED", task_id=task.task_id, vehicle_id=task.vehicle_id))
                    rec.record(SimEvent(t=now, etype="UL_STARTED", task_id=task.task_id, vehicle_id=task.vehicle_id))

                    task.status = TaskStatus.UL_IN_PROGRESS
                    in_transit.append(InTransit(task_id=task.task_id, ul_finish_t=now + task.uplink_time_s))

            # --- Uplink completion -> enqueue ---
            if in_transit:
                still: list[InTransit] = []
                for it in in_transit:
                    if it.ul_finish_t <= now:
                        t = tasks[it.task_id]
                        # If already missed deadline by uplink completion, mark miss
                        if now > t.effective_deadline_t:
                            t.status = TaskStatus.DEADLINE_MISS
                            rec.record(SimEvent(t=now, etype="UL_FINISHED", task_id=t.task_id, vehicle_id=t.vehicle_id))
                            rec.record(SimEvent(t=now, etype="DEADLINE_MISS", task_id=t.task_id, vehicle_id=t.vehicle_id, details={"where": "uplink"}))
                            continue

                        t.status = TaskStatus.READY
                        t.enqueued_t = now
                        rec.record(SimEvent(t=now, etype="UL_FINISHED", task_id=t.task_id, vehicle_id=t.vehicle_id))
                        rec.record(SimEvent(t=now, etype="ENQUEUED", task_id=t.task_id, vehicle_id=t.vehicle_id))
                        queue.enqueue(t)
                    else:
                        still.append(it)
                in_transit = still

            # --- Queue sanitation: expired/impossible ---
            _drop_expired_in_queue(queue, now, rec)
            _maybe_drop_impossible_next(queue, mec, now, rec, enabled=drop_impossible)

            # --- Prediction alpha(t) ---
            cand_ids = select_candidate_vehicle_ids(vehicles, rsu, cfg)
            exp_in = expected_inflow_vehicles(candidate_vehicle_ids=cand_ids, cfg=cfg, rng=rng)
            lam_hat = lambda_task_hat(expected_inflow=exp_in, horizon_s=horizon_s, mu_tasks_per_s_per_vehicle=mu)
            rho = rho_future(lam_hat, e_comp_s)
            alpha = alpha_from_rho(rho)

            # --- Scheduler reorder ---
            if len(queue) > 1:
                ctx = SchedulerContext(now=now, w0=mec.w0(), alpha=alpha, cfg=sched_params)
                queue.tasks = scheduler.rank(list(queue.tasks), ctx)

            # --- CPU start/step ---
            if mec.is_idle():
                started = mec.start_next_if_idle(now)
                if started is not None:
                    rec.record(SimEvent(t=now, etype="CPU_STARTED", task_id=started.task_id, vehicle_id=started.vehicle_id))

            finished = mec.step(dt=step_length, now=now + step_length)
            if finished is not None:
                if finished.cpu_finish_t is not None and finished.cpu_finish_t > finished.effective_deadline_t:
                    finished.status = TaskStatus.DEADLINE_MISS
                    rec.record(SimEvent(t=finished.cpu_finish_t, etype="DEADLINE_MISS", task_id=finished.task_id, vehicle_id=finished.vehicle_id, details={"where": "cpu"}))
                else:
                    rec.record(SimEvent(t=finished.cpu_finish_t or (now + step_length), etype="CPU_FINISHED", task_id=finished.task_id, vehicle_id=finished.vehicle_id))

            clock.tick()

        # --- Persist raw + summary ---
        rec.dump_jsonl(str(raw_path))
        summary = summarize_tasks(
            tasks=tasks.values(),
            run_name=run_name,
            scheduler=scheduler.name,
            seed=seed,
            high_thr=high_thr,
            utility_beta=utility_beta,
        )
        write_summary_csv(summary, str(summary_path))

        log.info("Run finished. Raw events: %s", raw_path)
        log.info("Run summary: %s", summary_path)
        return 0
    finally:
        traci.close()


def main() -> None:
    raise SystemExit("Use `python -m vecsim.cli ...`")
