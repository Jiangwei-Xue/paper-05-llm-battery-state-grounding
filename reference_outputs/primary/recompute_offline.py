#!/usr/bin/env python3
"""Offline recomputation for the SEGAN PV--battery scheduling study.

This script uses only saved task, request, response, action, and replay records.
It performs no provider or network calls.

Outputs:
- branch- and block-level F1/F0 metrics
- normalized, stratified, robust, and permutation summaries
- violation-severity and timing analyses
- exact F1--F0 paired temporal analysis
- independent replay verification and hand-constructed tests
- F1 same-action-space and expanded-system lexicographic gate analyses
- zero, heuristic, stale-optimal, random, and direct-MPC baselines
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats
from scipy.optimize import Bounds, LinearConstraint, milp

DT = 0.25
TARGET_KWH = 325.0
ENGINEERING_TERMINAL_TOL_KWH = 1.0
POWER_TOL_KW = 1e-4
ENERGY_TOL_KWH = 1e-5
EXPORT_TOL_KW = 1e-4
ACTIVITY_THRESHOLDS_KWH = (1.0, 5.0, 10.0)
BOOTSTRAP_SEED = 20260716
PERMUTATION_SEED = 20260821
RANDOM_BASELINE_SEED = 20260822
BOOTSTRAP_RESAMPLES = 10000
PERMUTATION_RESAMPLES = 100000
LEX_DISTANCE_TOL_KWH = 1e-4
LEX_THROUGHPUT_TOL_KWH = 1e-4
MILP_TIME_LIMIT_SECONDS = 10.0
MODE_TIE_BREAK_WEIGHT = 1e-8
BRANCHES = ("C1", "C2", "S1", "S2")


def json_safe(value: Any) -> Any:
    """Convert non-finite derived statistics to explicit JSON nulls."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_c: float
    eta_d: float
    initial_soc_kwh: float
    reserve_kwh: float
    export_limit_kw: float
    degradation_cost_per_kwh: float


@dataclass
class GateResult:
    status: str
    success: bool
    gate_type: str
    terminal_tolerance_kwh: float
    distance_kwh: float | None = None
    throughput_kwh: float | None = None
    physical_cost_usd: float | None = None
    curtailment_kwh: float | None = None
    curtailment_price_term_usd: float | None = None
    exact_retained_fraction: float | None = None
    correlation: float | None = None
    solve_seconds: float | None = None
    stage1_seconds: float | None = None
    stage2_seconds: float | None = None
    stage3_seconds: float | None = None
    action_kw: list[float] | None = None
    curtailment_kw: list[float] | None = None
    soc_kwh: list[float] | None = None
    message: str | None = None
    distance_lower_bound_kwh: float | None = None
    distance_optimality_gap_kwh: float | None = None
    global_distance_optimality_proven: bool | None = None
    lp_overlap_intervals: int | None = None
    lp_overlap_kwh: float | None = None
    selected_mode_source: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_digest(action: Sequence[float]) -> str:
    arr = np.asarray(action, dtype=np.float64)
    return sha256_bytes(arr.tobytes())


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def quantile_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(resamples, len(a)))].mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def summarize_values(values: Sequence[float], *, bootstrap: bool = True) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
        "positive": int(np.sum(a > 1e-12)),
        "zero": int(np.sum(np.abs(a) <= 1e-12)),
        "negative": int(np.sum(a < -1e-12)),
    }
    for proportion in (0.1, 0.2):
        out[f"trimmed_mean_{int(proportion*100)}pct"] = float(stats.trim_mean(a, proportiontocut=proportion))
    if bootstrap:
        lo, hi = quantile_ci(a)
        out["bootstrap_ci95"] = [lo, hi]
    order_value = np.sort(a)[::-1]
    order_abs = a[np.argsort(np.abs(a))[::-1]]
    for k in (1, 2, 5):
        out[f"mean_drop_largest_{k}"] = float(order_value[k:].mean()) if len(a) > k else float("nan")
        out[f"mean_drop_largest_abs_{k}"] = float(order_abs[k:].mean()) if len(a) > k else float("nan")
    return out


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3 or len(np.unique(xa[mask])) < 2 or len(np.unique(ya[mask])) < 2:
        return {"n": int(mask.sum()), "rho": None, "pvalue": None}
    result = stats.spearmanr(xa[mask], ya[mask])
    return {"n": int(mask.sum()), "rho": float(result.statistic), "pvalue": float(result.pvalue)}


def mean_or_none(values: Iterable[float | None]) -> float | None:
    a = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(a)) if a else None


def task_suffix(task: dict[str, Any], battery: Battery) -> dict[str, np.ndarray]:
    rows = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    update = task["visible_update"]
    n = len(rows)
    return {
        "load_kw": np.asarray([float(row["load_kw"]) for row in rows], dtype=float),
        "pv_kw": np.asarray([float(row["pv_forecast_kw"]) for row in rows], dtype=float),
        "price_usd_mwh": np.asarray([float(row["dam_price_usd_mwh"]) for row in rows], dtype=float),
        "capacity_kwh": np.full(n, float(update.get("usable_capacity_kwh", battery.capacity_kwh))),
        "max_charge_kw": np.full(n, float(update.get("max_charge_kw", battery.max_charge_kw))),
        "max_discharge_kw": np.full(n, float(update.get("max_discharge_kw", battery.max_discharge_kw))),
        "reserve_kwh": np.full(n, float(update.get("reserve_soc_kwh", battery.reserve_kwh))),
        "export_limit_kw": np.full(n, float(update.get("export_limit_kw", battery.export_limit_kw))),
    }


def independent_replay(task: dict[str, Any], action_kw: Sequence[float], initial_soc_kwh: float, battery: Battery) -> dict[str, Any]:
    data = task_suffix(task, battery)
    action = np.asarray(action_kw, dtype=float)
    n = len(data["load_kw"])
    if len(action) != n or not np.isfinite(action).all():
        raise ValueError("invalid action horizon")
    soc = float(initial_soc_kwh)
    trace: list[dict[str, float]] = []
    cost = 0.0
    total_violation_energy = 0.0
    violation_intervals = 0
    for t in range(n):
        u = float(action[t])
        charge = max(u, 0.0)
        discharge = max(-u, 0.0)
        next_soc = soc + battery.eta_c * charge * DT - discharge * DT / battery.eta_d
        charge_excess = max(charge - data["max_charge_kw"][t], 0.0)
        discharge_excess = max(discharge - data["max_discharge_kw"][t], 0.0)
        soc_lower = max(data["reserve_kwh"][t] - next_soc, 0.0)
        soc_upper = max(next_soc - data["capacity_kwh"][t], 0.0)
        grid = data["load_kw"][t] - data["pv_kw"][t] + u
        export_excess = max(-data["export_limit_kw"][t] - grid, 0.0)
        dynamic_bad = (
            charge_excess > POWER_TOL_KW
            or discharge_excess > POWER_TOL_KW
            or soc_lower > ENERGY_TOL_KWH
            or soc_upper > ENERGY_TOL_KWH
            or export_excess > EXPORT_TOL_KW
        )
        violation_intervals += int(dynamic_bad)
        total_violation_energy += DT * (charge_excess + discharge_excess + export_excess) + soc_lower + soc_upper
        cost += DT / 1000.0 * data["price_usd_mwh"][t] * grid
        cost += battery.degradation_cost_per_kwh * DT * (charge + discharge)
        trace.append({
            "offset": t,
            "requested_power_kw": u,
            "requested_charge_kw": charge,
            "requested_discharge_kw": discharge,
            "soc_start_kwh": soc,
            "soc_end_kwh": next_soc,
            "charge_power_exceedance_kw": charge_excess,
            "discharge_power_exceedance_kw": discharge_excess,
            "soc_lower_violation_kwh": soc_lower,
            "soc_upper_violation_kwh": soc_upper,
            "grid_kw": grid,
            "export_violation_kw": export_excess,
        })
        soc = next_soc
    terminal_signed = soc - TARGET_KWH
    terminal_abs = abs(terminal_signed)
    dynamic_ok = violation_intervals == 0
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": float(soc),
        "terminal_signed_error_kwh": float(terminal_signed),
        "terminal_absolute_error_kwh": float(terminal_abs),
        "dynamic_constraints_ok": bool(dynamic_ok),
        "engineering_feasible": bool(dynamic_ok and terminal_abs <= ENGINEERING_TERMINAL_TOL_KWH),
        "strict_feasible": bool(dynamic_ok and terminal_abs <= 1e-6),
        "violation_interval_count": int(violation_intervals),
        "total_violation_energy": float(total_violation_energy),
        "action_energy_kwh": float(DT * np.abs(action).sum()),
        "physical_cost_usd": float(cost),
        "trace": trace,
    }


def replay_severity(replay: dict[str, Any]) -> dict[str, Any]:
    trace = replay["trace"]
    c = np.asarray([r["charge_power_exceedance_kw"] for r in trace], dtype=float)
    d = np.asarray([r["discharge_power_exceedance_kw"] for r in trace], dtype=float)
    sl = np.asarray([r["soc_lower_violation_kwh"] for r in trace], dtype=float)
    su = np.asarray([r["soc_upper_violation_kwh"] for r in trace], dtype=float)
    ex = np.asarray([r["export_violation_kw"] for r in trace], dtype=float)
    power = c + d
    return {
        "power_violation_any": bool(np.any(power > POWER_TOL_KW)),
        "power_violation_peak_kw": float(power.max(initial=0.0)),
        "power_violation_energy_kwh": float(DT * power.sum()),
        "soc_lower_violation_any": bool(np.any(sl > ENERGY_TOL_KWH)),
        "soc_lower_violation_peak_kwh": float(sl.max(initial=0.0)),
        "soc_lower_violation_sum_kwh_steps": float(sl.sum()),
        "soc_upper_violation_any": bool(np.any(su > ENERGY_TOL_KWH)),
        "soc_upper_violation_peak_kwh": float(su.max(initial=0.0)),
        "soc_upper_violation_sum_kwh_steps": float(su.sum()),
        "export_violation_any": bool(np.any(ex > EXPORT_TOL_KW)),
        "export_violation_peak_kw": float(ex.max(initial=0.0)),
        "export_violation_energy_kwh": float(DT * ex.sum()),
        "terminal_violation_any": bool(replay["terminal_absolute_error_kwh"] > ENGINEERING_TERMINAL_TOL_KWH),
        "terminal_absolute_error_kwh": float(replay["terminal_absolute_error_kwh"]),
        "terminal_signed_error_kwh": float(replay["terminal_signed_error_kwh"]),
        "violation_interval_count": int(replay["violation_interval_count"]),
        "total_violation_energy": float(replay["total_violation_energy"]),
        "dynamic_constraints_ok": bool(replay["dynamic_constraints_ok"]),
        "engineering_feasible": bool(replay["engineering_feasible"]),
        "physical_cost_usd": float(replay["physical_cost_usd"]),
    }


def action_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) != len(b):
        raise ValueError("unequal action horizons")
    return float(DT * np.abs(a - b).sum())


def grouping_ed(actions: Sequence[np.ndarray], pair_a: tuple[int, int], pair_b: tuple[int, int]) -> float:
    cross = [action_distance(actions[i], actions[j]) for i in pair_a for j in pair_b]
    within_a = action_distance(actions[pair_a[0]], actions[pair_a[1]])
    within_b = action_distance(actions[pair_b[0]], actions[pair_b[1]])
    return float(2.0 * np.mean(cross) - within_a - within_b)


def get_attempt_timing(branch: dict[str, Any]) -> dict[str, Any]:
    attempts = branch.get("response", {}).get("attempts", [])
    if not attempts:
        return {"attempt_count": 0, "latency_seconds": None, "request_sent_utc": None, "response_received_utc": None}
    start = parse_time(attempts[0]["request_sent_utc"])
    end = parse_time(attempts[-1]["response_received_utc"])
    return {
        "attempt_count": len(attempts),
        "latency_seconds": float((end - start).total_seconds()),
        "request_sent_utc": start.isoformat(),
        "response_received_utc": end.isoformat(),
        "request_start_epoch": start.timestamp(),
        "response_end_epoch": end.timestamp(),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_inputs(project: Path) -> tuple[Battery, dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    config = yaml.safe_load((project / "configs/frozen_protocol.yaml").read_text(encoding="utf-8"))
    b = config["battery"]
    battery = Battery(
        capacity_kwh=float(b["energy_capacity_kwh"]),
        max_charge_kw=float(b["max_charge_kw"]),
        max_discharge_kw=float(b["max_discharge_kw"]),
        eta_c=float(b["charge_efficiency"]),
        eta_d=float(b["discharge_efficiency"]),
        initial_soc_kwh=float(b["initial_soc_kwh"]),
        reserve_kwh=float(b["reserve_soc_kwh"]),
        export_limit_kw=float(b["export_limit_kw"]),
        degradation_cost_per_kwh=float(b["degradation_cost_per_kwh"]),
    )
    f1_plan_path = project / "segan_revision_major_v2/reviews/e2b_protocol_v2/E2B_V2_F1_RUN_PLAN.jsonl"
    f0_plan_path = project / "segan_revision_major_v2/reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"
    f1_plan = load_jsonl(f1_plan_path)
    f0_plan = load_jsonl(f0_plan_path)
    f1_plan_by_block = {row["block_id"]: row for row in f1_plan}
    f0_plan_by_block = {row["block_id"]: row for row in f0_plan}
    tasks: dict[str, dict[str, Any]] = {}
    for row in [*f1_plan, *f0_plan]:
        scenario_id = row["scenario_id"]
        if scenario_id in tasks:
            continue
        source = project / row["source_record_path"]
        source_record = json.loads(source.read_text(encoding="utf-8"))
        task = source_record["task"]
        if task["scenario_id"] != scenario_id:
            raise ValueError(f"scenario mismatch in {source}")
        tasks[scenario_id] = {
            "task": task,
            "source_record_path": str(row["source_record_path"]),
            "source_record_sha256": sha256_file(source),
        }
    f1_dir = project / "segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records"
    f0_dir = project / "segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records"
    f1_records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(f1_dir.glob("*.json"))]
    f0_records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(f0_dir.glob("*.json"))]
    return battery, tasks, f1_records, f0_records, f1_plan_by_block, f0_plan_by_block


def task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    iv = task["v2_prefix_intervention"]
    c = float(iv["canonical_event_soc_kwh"])
    s = float(iv["stale_event_soc_kwh"])
    gap = c - s
    activation = int(task["visible_update"]["activation_step"])
    horizon = len(task["model_visible_episode"]["stage_2"]["remaining_timeseries"])
    return {
        "event_family": str(task["event_family"]),
        "difficulty": str(task["difficulty"]),
        "season": str(task["season"]),
        "location_id": str(task["location_id"]),
        "activation_step": activation,
        "horizon_intervals": horizon,
        "horizon_hours": horizon * DT,
        "canonical_soc_kwh": c,
        "stale_soc_kwh": s,
        "soc_gap_signed_kwh": gap,
        "soc_gap_abs_kwh": abs(gap),
        "soc_gap_direction": "canonical_higher" if gap > 0 else "canonical_lower" if gap < 0 else "equal",
        "divergence_kwh": float(iv["divergence_kwh"]),
    }


def block_and_branch_rows(
    tier: str,
    records: list[dict[str, Any]],
    plan_by_block: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    battery: Battery,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    branch_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    for record in records:
        block_id = record["block_id"]
        plan = plan_by_block[block_id]
        scenario_id = record["scenario_id"]
        task = tasks[scenario_id]["task"]
        meta = task_metadata(task)
        if int(plan["horizon"]) != meta["horizon_intervals"]:
            raise ValueError("plan/task horizon mismatch")
        actions: dict[str, np.ndarray] = {}
        for position, branch_name in enumerate(record["branch_order"], start=1):
            branch = record["branches"][branch_name]
            parsed = branch["parsed"]
            if parsed.get("ok") is not True:
                continue
            action = np.asarray(parsed["dense_action_kw"], dtype=float)
            actions[branch_name] = action
            carrier = "canonical" if branch_name.startswith("C") else "stale"
            carrier_initial = meta["canonical_soc_kwh"] if carrier == "canonical" else meta["stale_soc_kwh"]
            for replay_name, initial in (("carrier_consistent", carrier_initial), ("authoritative", meta["canonical_soc_kwh"])):
                independent = independent_replay(task, action, initial, battery)
                archived = branch["score"][f"{replay_name}_replay"]
                trace_fields = (
                    "requested_power_kw", "soc_start_kwh", "soc_end_kwh",
                    "charge_power_exceedance_kw", "discharge_power_exceedance_kw",
                    "soc_lower_violation_kwh", "soc_upper_violation_kwh",
                    "grid_kw", "export_violation_kw",
                )
                max_trace_diff = 0.0
                for arow, brow in zip(independent["trace"], archived["trace"], strict=True):
                    for field in trace_fields:
                        max_trace_diff = max(max_trace_diff, abs(float(arow[field]) - float(brow[field])))
                scalar_diffs = {
                    field: abs(float(independent[field]) - float(archived[field]))
                    for field in (
                        "terminal_soc_kwh", "terminal_signed_error_kwh", "terminal_absolute_error_kwh",
                        "total_violation_energy", "action_energy_kwh", "physical_cost_usd",
                    )
                }
                exact_flags = {
                    field: independent[field] == archived[field]
                    for field in ("dynamic_constraints_ok", "engineering_feasible", "strict_feasible", "violation_interval_count")
                }
                replay_checks.append({
                    "tier": tier,
                    "block_id": block_id,
                    "scenario_id": scenario_id,
                    "model_condition": record["model_condition"],
                    "branch": branch_name,
                    "replay_type": replay_name,
                    "max_trace_abs_diff": max_trace_diff,
                    **{f"abs_diff_{k}": v for k, v in scalar_diffs.items()},
                    "all_flag_matches": all(exact_flags.values()),
                    **{f"match_{k}": v for k, v in exact_flags.items()},
                })
            authoritative = independent_replay(task, action, meta["canonical_soc_kwh"], battery)
            carrier_replay = independent_replay(task, action, carrier_initial, battery)
            timing = get_attempt_timing(branch)
            severity_auth = replay_severity(authoritative)
            severity_carrier = replay_severity(carrier_replay)
            action_energy = float(DT * np.abs(action).sum())
            row = {
                "tier": tier,
                "block_id": block_id,
                "scenario_id": scenario_id,
                "model_condition": record["model_condition"],
                "branch": branch_name,
                "carrier": carrier,
                "duplicate_index": int(branch_name[1]),
                "branch_position": position,
                "run_order": int(plan["run_order"]),
                **meta,
                "parsed": True,
                "action_digest": action_digest(action),
                "action_energy_kwh": action_energy,
                "exact_zero": bool(action_energy <= 1e-12),
                **{f"nontrivial_{int(th)}kwh": bool(action_energy >= th) for th in ACTIVITY_THRESHOLDS_KWH},
                "carrier_consistent_raw_feasible": carrier_replay["engineering_feasible"],
                "authoritative_raw_feasible": authoritative["engineering_feasible"],
                **{f"carrier_{k}": v for k, v in severity_carrier.items()},
                **{f"authoritative_{k}": v for k, v in severity_auth.items()},
                "returned_model": branch["response"].get("returned_model"),
                "system_fingerprint": branch["response"].get("system_fingerprint"),
                "prompt_sha256": branch["prompt_sha256"],
                "request_payload_sha256": branch["request_payload_bytes_sha256"],
                **timing,
            }
            branch_rows.append(row)
        if set(actions) != set(BRANCHES):
            continue
        c1, c2, s1, s2 = (actions[x] for x in BRANCHES)
        cross = [action_distance(c, s) for c in (c1, c2) for s in (s1, s2)]
        B = float(np.mean(cross))
        Wc = action_distance(c1, c2)
        Ws = action_distance(s1, s2)
        ED = 2.0 * B - Wc - Ws
        h = len(c1)
        pair_max_distance = h * DT * (2.0 * battery.max_charge_kw)
        mean_c = (c1 + c2) / 2.0
        mean_s = (s1 + s2) / 2.0
        signed_net_delta = float(DT * np.sum(mean_c - mean_s))
        soc_direction_score = -float(np.sign(meta["soc_gap_signed_kwh"])) * signed_net_delta
        def auth(branch: str) -> dict[str, Any]:
            return record["branches"][branch]["score"]["authoritative_replay"]
        terminal_alignment = float(np.mean([auth("S1")["terminal_absolute_error_kwh"], auth("S2")["terminal_absolute_error_kwh"]]) - np.mean([auth("C1")["terminal_absolute_error_kwh"], auth("C2")["terminal_absolute_error_kwh"]]))
        mpc = np.asarray(record["same_information_mpc"]["action_kw"], dtype=float)
        mpc_alignment = float(np.mean([action_distance(s1, mpc), action_distance(s2, mpc)]) - np.mean([action_distance(c1, mpc), action_distance(c2, mpc)]))
        br = [x for x in branch_rows if x["tier"] == tier and x["block_id"] == block_id]
        brmap = {x["branch"]: x for x in br}
        contrasts: dict[str, float] = {}
        sev_fields = [
            "power_violation_peak_kw", "power_violation_energy_kwh",
            "soc_lower_violation_peak_kwh", "soc_lower_violation_sum_kwh_steps",
            "soc_upper_violation_peak_kwh", "soc_upper_violation_sum_kwh_steps",
            "export_violation_peak_kw", "export_violation_energy_kwh",
            "terminal_absolute_error_kwh", "violation_interval_count", "total_violation_energy",
        ]
        for field in sev_fields:
            sval = np.mean([brmap["S1"][f"authoritative_{field}"], brmap["S2"][f"authoritative_{field}"]])
            cval = np.mean([brmap["C1"][f"authoritative_{field}"], brmap["C2"][f"authoritative_{field}"]])
            contrasts[f"alignment_{field}"] = float(sval - cval)
        starts = [brmap[b]["request_start_epoch"] for b in BRANCHES if brmap[b].get("request_start_epoch") is not None]
        ends = [brmap[b]["response_end_epoch"] for b in BRANCHES if brmap[b].get("response_end_epoch") is not None]
        def time_gap(a: str, b: str) -> float | None:
            va = brmap[a].get("request_start_epoch"); vb = brmap[b].get("request_start_epoch")
            return abs(float(va)-float(vb)) if va is not None and vb is not None else None
        block_rows.append({
            "tier": tier,
            "block_id": block_id,
            "scenario_id": scenario_id,
            "model_condition": record["model_condition"],
            "run_order": int(plan["run_order"]),
            "branch_order": "-".join(record["branch_order"]),
            **meta,
            "B_kwh": B,
            "Wc_kwh": Wc,
            "Ws_kwh": Ws,
            "ED_kwh": ED,
            "B_avg_kw": B / (h * DT),
            "Wc_avg_kw": Wc / (h * DT),
            "Ws_avg_kw": Ws / (h * DT),
            "ED_avg_kw": ED / (h * DT),
            "B_fraction_pairwise_max": B / pair_max_distance,
            "ED_fraction_theoretical_max": ED / (2.0 * pair_max_distance),
            "B_per_soc_gap": B / meta["soc_gap_abs_kwh"],
            "ED_per_soc_gap": ED / meta["soc_gap_abs_kwh"],
            "B_per_battery_capacity": B / battery.capacity_kwh,
            "ED_per_battery_capacity": ED / battery.capacity_kwh,
            "terminal_alignment_kwh": terminal_alignment,
            "mpc_alignment_kwh": mpc_alignment,
            "soc_direction_net_energy_kwh": soc_direction_score,
            "canonical_minus_stale_net_energy_kwh": signed_net_delta,
            "block_span_seconds": max(ends) - min(starts) if starts and ends else None,
            "C_duplicate_start_gap_seconds": time_gap("C1", "C2"),
            "S_duplicate_start_gap_seconds": time_gap("S1", "S2"),
            "mean_latency_seconds": mean_or_none(brmap[b]["latency_seconds"] for b in BRANCHES),
            **contrasts,
        })
    return pd.DataFrame(branch_rows), pd.DataFrame(block_rows), replay_checks


# ---------- MILP gate implementation ----------


def _gate_problem_arrays(task: dict[str, Any], battery: Battery, reference: np.ndarray, initial_soc: float, allow_curtailment: bool, terminal_tolerance: float, distance_limit: float | None = None, throughput_limit: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LinearConstraint, dict[str, slice | int], dict[str, np.ndarray]]:
    data = task_suffix(task, battery)
    n = len(reference)
    if n != len(data["load_kw"]):
        raise ValueError("reference length mismatch")
    # c,d,q,e(0..n),a,z
    idx: dict[str, slice | int] = {
        "c": slice(0, n),
        "d": slice(n, 2*n),
        "q": slice(2*n, 3*n),
        "e": slice(3*n, 4*n+1),
        "a": slice(4*n+1, 5*n+1),
        "z": slice(5*n+1, 6*n+1),
    }
    N = 6*n+1
    lower = np.full(N, -np.inf)
    upper = np.full(N, np.inf)
    lower[idx["c"]] = 0.0; upper[idx["c"]] = data["max_charge_kw"]
    lower[idx["d"]] = 0.0; upper[idx["d"]] = data["max_discharge_kw"]
    lower[idx["q"]] = 0.0; upper[idx["q"]] = data["pv_kw"] if allow_curtailment else 0.0
    e0 = 3*n
    lower[e0] = upper[e0] = initial_soc
    for t in range(n):
        lower[e0+t+1] = data["reserve_kwh"][t]
        upper[e0+t+1] = data["capacity_kwh"][t]
    lower[e0+n] = max(lower[e0+n], TARGET_KWH-terminal_tolerance)
    upper[e0+n] = min(upper[e0+n], TARGET_KWH+terminal_tolerance)
    lower[idx["a"]] = 0.0
    lower[idx["z"]] = 0.0; upper[idx["z"]] = 1.0
    integrality = np.zeros(N, dtype=int)
    integrality[idx["z"]] = 1
    extra = int(distance_limit is not None) + int(throughput_limit is not None)
    M = sparse.lil_matrix((6*n+extra, N), dtype=float)
    lb = np.full(6*n+extra, -np.inf)
    ub = np.full(6*n+extra, np.inf)
    row = 0
    for t in range(n):
        c = t; d = n+t; q = 2*n+t; e_t = 3*n+t; e_next = e_t+1; a = 4*n+1+t; z = 5*n+1+t
        M[row, e_next] = 1.0; M[row, e_t] = -1.0; M[row, c] = -battery.eta_c*DT; M[row, d] = DT/battery.eta_d
        lb[row] = ub[row] = 0.0; row += 1
        M[row, c] = 1.0; M[row, d] = -1.0; M[row, q] = 1.0
        lb[row] = -data["export_limit_kw"][t] - data["load_kw"][t] + data["pv_kw"][t]; row += 1
        M[row, c] = 1.0; M[row, z] = -data["max_charge_kw"][t]; ub[row] = 0.0; row += 1
        M[row, d] = 1.0; M[row, z] = data["max_discharge_kw"][t]; ub[row] = data["max_discharge_kw"][t]; row += 1
        M[row, c] = 1.0; M[row, d] = -1.0; M[row, a] = -1.0; ub[row] = reference[t]; row += 1
        M[row, c] = -1.0; M[row, d] = 1.0; M[row, a] = -1.0; ub[row] = -reference[t]; row += 1
    if distance_limit is not None:
        M[row, idx["a"]] = DT; ub[row] = distance_limit; row += 1
    if throughput_limit is not None:
        M[row, idx["c"]] = DT; M[row, idx["d"]] = DT; ub[row] = throughput_limit; row += 1
    return lower, upper, integrality, np.zeros(N), LinearConstraint(M.tocsr(), lb, ub), idx, data


def _solve_milp(c: np.ndarray, lower: np.ndarray, upper: np.ndarray, integrality: np.ndarray, constraint: LinearConstraint, label: str) -> tuple[Any, float]:
    start = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraint,
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-9, "presolve": True},
    )
    elapsed = time.perf_counter() - start
    if not result.success or result.x is None:
        raise RuntimeError(f"{label}: status={result.status}; {result.message}")
    return result, elapsed


def solve_gate(task: dict[str, Any], battery: Battery, reference_action: Sequence[float], initial_soc: float, *, allow_curtailment: bool, terminal_tolerance: float, full_lexicographic: bool = True, feasible_action: Sequence[float] | None = None) -> GateResult:
    reference = np.asarray(reference_action, dtype=float)
    gate_type = "expanded_system" if allow_curtailment else "battery_only"
    overall_start = time.perf_counter()
    try:
        distance_upper = None
        if feasible_action is not None:
            distance_upper = action_distance(reference, feasible_action) + LEX_DISTANCE_TOL_KWH
        lower, upper, integ, _, constraint, idx, data = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=distance_upper)
        n = len(reference); N = len(lower)
        zero_mode_mask = np.zeros(n, dtype=bool)
        anchor_mode = np.zeros(n, dtype=float)
        if feasible_action is not None:
            feasible_array = np.asarray(feasible_action, dtype=float)
            zero_mode_mask = np.abs(reference) <= 1e-9
            anchor_mode = (feasible_array > 0.0).astype(float)
            z_indices = np.arange(idx["z"].start, idx["z"].stop)
            lower[z_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper[z_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ[z_indices[zero_mode_mask]] = 0
        c1 = np.zeros(N); c1[idx["a"]] = DT
        raw_charge_mode = (reference > 0.0).astype(float)
        c1[idx["z"]] = MODE_TIE_BREAK_WEIGHT * (1.0 - 2.0 * raw_charge_mode)
        r1, t1 = _solve_milp(c1, lower, upper, integ, constraint, "gate distance")
        dopt = float(DT * np.sum(r1.x[idx["a"]]))
        if not full_lexicographic:
            x = r1.x
            c = x[idx["c"]]; d = x[idx["d"]]; q = x[idx["q"]]; e = x[idx["e"]]
            action = c-d
            return GateResult(status="optimal_distance_only", success=True, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance, distance_kwh=float(DT*np.abs(action-reference).sum()), throughput_kwh=float(DT*(c+d).sum()), physical_cost_usd=_physical_cost(data, battery, action, q), curtailment_kwh=float(DT*q.sum()), curtailment_price_term_usd=float(DT/1000*np.sum(data["price_usd_mwh"]*q)), exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)), correlation=_correlation(action,reference), solve_seconds=time.perf_counter()-overall_start, stage1_seconds=t1, action_kw=action.tolist(), curtailment_kw=q.tolist(), soc_kwh=e.tolist())
        lower2, upper2, integ2, _, constraint2, idx2, data2 = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=dopt+LEX_DISTANCE_TOL_KWH)
        if feasible_action is not None:
            z2_indices = np.arange(idx2["z"].start, idx2["z"].stop)
            lower2[z2_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper2[z2_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ2[z2_indices[zero_mode_mask]] = 0
        c2 = np.zeros(len(lower2)); c2[idx2["c"]] = DT; c2[idx2["d"]] = DT
        c2[idx2["z"]] = MODE_TIE_BREAK_WEIGHT * (1.0 - 2.0 * raw_charge_mode)
        r2, t2 = _solve_milp(c2, lower2, upper2, integ2, constraint2, "gate throughput")
        topt = float(DT * np.sum(r2.x[idx2["c"]] + r2.x[idx2["d"]]))
        lower3, upper3, integ3, _, constraint3, idx3, data3 = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=dopt+LEX_DISTANCE_TOL_KWH, throughput_limit=topt+LEX_THROUGHPUT_TOL_KWH)
        if feasible_action is not None:
            z3_indices = np.arange(idx3["z"].start, idx3["z"].stop)
            lower3[z3_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper3[z3_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ3[z3_indices[zero_mode_mask]] = 0
        # The first two stages include a deterministic raw-sign mode tie-break.
        # Fix that selected binary mode for the final cost tie-break; this avoids
        # re-opening a highly degenerate mixed-integer optimal face.
        selected_mode = np.rint(r2.x[idx2["z"]]).astype(float)
        lower3[idx3["z"]] = selected_mode
        upper3[idx3["z"]] = selected_mode
        integ3[idx3["z"]] = 0
        c3 = np.zeros(len(lower3))
        c3[idx3["c"]] = DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
        c3[idx3["d"]] = -DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
        c3[idx3["q"]] = DT/1000*data3["price_usd_mwh"]
        r3, t3 = _solve_milp(c3, lower3, upper3, integ3, constraint3, "gate cost fixed-mode")
        x = r3.x; c = x[idx3["c"]]; d = x[idx3["d"]]; q = x[idx3["q"]]; e = x[idx3["e"]]
        action = c-d
        return GateResult(
            status="optimal", success=True, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance,
            distance_kwh=float(DT*np.abs(action-reference).sum()), throughput_kwh=float(DT*(c+d).sum()),
            physical_cost_usd=_physical_cost(data3,battery,action,q), curtailment_kwh=float(DT*q.sum()),
            curtailment_price_term_usd=float(DT/1000*np.sum(data3["price_usd_mwh"]*q)),
            exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)), correlation=_correlation(action,reference),
            solve_seconds=time.perf_counter()-overall_start, stage1_seconds=t1, stage2_seconds=t2, stage3_seconds=t3,
            action_kw=action.tolist(), curtailment_kw=q.tolist(), soc_kwh=e.tolist(),
        )
    except Exception as exc:
        return GateResult(status="error", success=False, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance, solve_seconds=time.perf_counter()-overall_start, message=f"{type(exc).__name__}: {exc}")


def _physical_cost(data: dict[str, np.ndarray], battery: Battery, action: np.ndarray, curtailment: np.ndarray) -> float:
    grid = data["load_kw"] - data["pv_kw"] + curtailment + action
    return float(DT/1000*np.sum(data["price_usd_mwh"]*grid) + battery.degradation_cost_per_kwh*DT*np.sum(np.abs(action)))


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a,b)[0,1])



def _solve_gate_lp_mode(
    task: dict[str, Any],
    battery: Battery,
    reference: np.ndarray,
    initial_soc: float,
    *,
    allow_curtailment: bool,
    terminal_tolerance: float,
    fixed_mode: np.ndarray | None,
    zero_mode_anchor: np.ndarray,
    mode_source: str,
) -> tuple[GateResult | None, dict[str, Any]]:
    """Solve the lexicographic gate with relaxed or fixed charge/discharge modes."""
    overall = time.perf_counter()
    lower, upper, integ, _, constraint, idx, data = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance
    )
    integ[:] = 0
    if fixed_mode is not None:
        mode = np.asarray(fixed_mode, dtype=float)
        zidx = np.arange(idx["z"].start, idx["z"].stop)
        lower[zidx] = mode
        upper[zidx] = mode
    N = len(lower)
    c1 = np.zeros(N); c1[idx["a"]] = DT
    try:
        r1, t1 = _solve_milp(c1, lower, upper, integ, constraint, f"LP gate distance {mode_source}")
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}"}
    dopt = float(DT * np.sum(r1.x[idx["a"]]))
    lower2, upper2, integ2, _, constraint2, idx2, data2 = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance,
        distance_limit=dopt + LEX_DISTANCE_TOL_KWH,
    )
    integ2[:] = 0
    if fixed_mode is not None:
        zidx2 = np.arange(idx2["z"].start, idx2["z"].stop)
        lower2[zidx2] = mode; upper2[zidx2] = mode
    c2 = np.zeros(len(lower2)); c2[idx2["c"]] = DT; c2[idx2["d"]] = DT
    try:
        r2, t2 = _solve_milp(c2, lower2, upper2, integ2, constraint2, f"LP gate throughput {mode_source}")
    except Exception as exc:
        return None, {"distance_lower_bound_kwh": dopt, "error": f"{type(exc).__name__}: {exc}"}
    ch2 = np.asarray(r2.x[idx2["c"]], dtype=float)
    dis2 = np.asarray(r2.x[idx2["d"]], dtype=float)
    action2 = ch2 - dis2
    overlap = np.minimum(ch2, dis2)
    overlap_count = int(np.sum(overlap > 1e-6))
    overlap_kwh = float(DT * overlap.sum())
    topt = float(DT * np.sum(ch2 + dis2))
    if fixed_mode is None and overlap_count > 0:
        return None, {
            "distance_lower_bound_kwh": dopt,
            "throughput_lower_bound_kwh": topt,
            "lp_action_kw": action2,
            "lp_overlap_intervals": overlap_count,
            "lp_overlap_kwh": overlap_kwh,
            "stage1_seconds": t1,
            "stage2_seconds": t2,
        }
    selected_mode = np.asarray(fixed_mode, dtype=float) if fixed_mode is not None else (action2 > 1e-9).astype(float)
    zero_action = np.abs(action2) <= 1e-9
    selected_mode[zero_action] = zero_mode_anchor[zero_action]
    lower3, upper3, integ3, _, constraint3, idx3, data3 = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance,
        distance_limit=dopt + LEX_DISTANCE_TOL_KWH,
        throughput_limit=topt + LEX_THROUGHPUT_TOL_KWH,
    )
    integ3[:] = 0
    zidx3 = np.arange(idx3["z"].start, idx3["z"].stop)
    lower3[zidx3] = selected_mode; upper3[zidx3] = selected_mode
    c3 = np.zeros(len(lower3))
    c3[idx3["c"]] = DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
    c3[idx3["d"]] = -DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
    c3[idx3["q"]] = DT/1000*data3["price_usd_mwh"]
    try:
        r3, t3 = _solve_milp(c3, lower3, upper3, integ3, constraint3, f"LP gate cost {mode_source}")
    except Exception as exc:
        return None, {"distance_lower_bound_kwh": dopt, "error": f"{type(exc).__name__}: {exc}"}
    x = r3.x
    ch = np.asarray(x[idx3["c"]], dtype=float); dis = np.asarray(x[idx3["d"]], dtype=float)
    q = np.asarray(x[idx3["q"]], dtype=float); e = np.asarray(x[idx3["e"]], dtype=float)
    action = ch - dis
    distance = float(DT * np.abs(action-reference).sum())
    result = GateResult(
        status="global_distance_optimum_lp_integral" if fixed_mode is None else "fixed_mode_feasible",
        success=True,
        gate_type="expanded_system" if allow_curtailment else "battery_only",
        terminal_tolerance_kwh=terminal_tolerance,
        distance_kwh=distance,
        throughput_kwh=float(DT*np.sum(ch+dis)),
        physical_cost_usd=_physical_cost(data3,battery,action,q),
        curtailment_kwh=float(DT*q.sum()),
        curtailment_price_term_usd=float(DT/1000*np.sum(data3["price_usd_mwh"]*q)),
        exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)),
        correlation=_correlation(action,reference),
        solve_seconds=time.perf_counter()-overall,
        stage1_seconds=t1,stage2_seconds=t2,stage3_seconds=t3,
        action_kw=action.tolist(),curtailment_kw=q.tolist(),soc_kwh=e.tolist(),
        distance_lower_bound_kwh=dopt,
        distance_optimality_gap_kwh=max(distance-dopt,0.0),
        global_distance_optimality_proven=fixed_mode is None,
        lp_overlap_intervals=overlap_count,
        lp_overlap_kwh=overlap_kwh,
        selected_mode_source=mode_source,
    )
    return result, {"distance_lower_bound_kwh": dopt, "lp_overlap_intervals": overlap_count, "lp_overlap_kwh": overlap_kwh}


def solve_gate_hybrid(
    task: dict[str, Any],
    battery: Battery,
    reference_action: Sequence[float],
    initial_soc: float,
    *,
    allow_curtailment: bool,
    terminal_tolerance: float,
    full_lexicographic: bool = True,
    feasible_action: Sequence[float] | None = None,
) -> GateResult:
    """Complete-coverage projection with an LP certificate and fixed-mode upper bound.

    If the LP relaxation has no simultaneous charge/discharge, it is an exact
    globally distance-optimal solution to the mixed-integer gate. Otherwise,
    the LP distance is retained as a lower bound and the best feasible solution
    from a disclosed mode library is returned as an upper bound.
    """
    reference = np.asarray(reference_action, dtype=float)
    if feasible_action is None:
        raise ValueError("hybrid gate requires a feasible reference action")
    feasible = np.asarray(feasible_action, dtype=float)
    anchor = (feasible > 0.0).astype(float)
    relaxed, info = _solve_gate_lp_mode(
        task,battery,reference,initial_soc,allow_curtailment=allow_curtailment,
        terminal_tolerance=terminal_tolerance,fixed_mode=None,
        zero_mode_anchor=anchor,mode_source="lp_relaxation",
    )
    if relaxed is not None:
        if not full_lexicographic:
            relaxed.status = "global_distance_optimum_lp_integral_distance_focus"
        return relaxed
    lower_bound = float(info.get("distance_lower_bound_kwh", float("nan")))
    lp_action = np.asarray(info.get("lp_action_kw", feasible), dtype=float)
    heuristic = heuristic_action(task,battery,initial_soc)
    modes: dict[str,np.ndarray] = {}
    raw_mode = (reference > 0.0).astype(float)
    raw_mode[np.abs(reference)<=1e-9] = anchor[np.abs(reference)<=1e-9]
    lp_mode = (lp_action > 0.0).astype(float)
    lp_mode[np.abs(lp_action)<=1e-9] = anchor[np.abs(lp_action)<=1e-9]
    heur_mode = (heuristic > 0.0).astype(float)
    modes["raw_sign_with_mpc_zero_anchor"] = raw_mode
    modes["lp_net_sign_with_mpc_zero_anchor"] = lp_mode
    modes["direct_mpc_sign"] = anchor
    modes["terminal_heuristic_sign"] = heur_mode
    unique: dict[bytes,tuple[str,np.ndarray]] = {}
    for name,mode in modes.items(): unique.setdefault(mode.astype(np.uint8).tobytes(),(name,mode))
    candidates: list[GateResult] = []
    errors=[]
    for name,mode in unique.values():
        result, meta = _solve_gate_lp_mode(
            task,battery,reference,initial_soc,allow_curtailment=allow_curtailment,
            terminal_tolerance=terminal_tolerance,fixed_mode=mode,
            zero_mode_anchor=anchor,mode_source=name,
        )
        if result is not None:
            result.distance_lower_bound_kwh=lower_bound
            result.distance_optimality_gap_kwh=max(float(result.distance_kwh)-lower_bound,0.0)
            result.global_distance_optimality_proven=result.distance_optimality_gap_kwh<=LEX_DISTANCE_TOL_KWH
            result.lp_overlap_intervals=int(info.get("lp_overlap_intervals",0))
            result.lp_overlap_kwh=float(info.get("lp_overlap_kwh",0.0))
            candidates.append(result)
        else:
            errors.append({"mode":name,"meta":meta})
    if not candidates:
        return GateResult(status="error",success=False,gate_type="expanded_system" if allow_curtailment else "battery_only",terminal_tolerance_kwh=terminal_tolerance,distance_lower_bound_kwh=lower_bound,lp_overlap_intervals=int(info.get("lp_overlap_intervals",0)),lp_overlap_kwh=float(info.get("lp_overlap_kwh",0.0)),message=canonical_json(errors))
    candidates.sort(key=lambda r:(float(r.distance_kwh),float(r.throughput_kwh),float(r.physical_cost_usd)))
    best=candidates[0]
    best.status="feasible_mode_library_upper_bound" if not best.global_distance_optimality_proven else "global_distance_optimum_mode_library"
    if not full_lexicographic:
        best.status += "_distance_focus"
    return best

def solve_cost_optimal(task: dict[str, Any], battery: Battery, initial_soc: float, *, allow_curtailment: bool, terminal_tolerance: float = 0.0) -> GateResult:
    data = task_suffix(task,battery); n=len(data["load_kw"])
    reference=np.zeros(n)
    lower,upper,integ,_,constraint,idx,data = _gate_problem_arrays(task,battery,reference,initial_soc,allow_curtailment,terminal_tolerance)
    # Remove the absolute-value constraints and variables indirectly by zero objective; they are harmless.
    c=np.zeros(len(lower))
    c[idx["c"]]=DT/1000*data["price_usd_mwh"]+battery.degradation_cost_per_kwh*DT
    c[idx["d"]]=-DT/1000*data["price_usd_mwh"]+battery.degradation_cost_per_kwh*DT
    c[idx["q"]]=DT/1000*data["price_usd_mwh"]
    start=time.perf_counter()
    try:
        r,t=_solve_milp(c,lower,upper,integ,constraint,"cost optimum")
        x=r.x; ch=x[idx["c"]]; dis=x[idx["d"]]; q=x[idx["q"]]; e=x[idx["e"]]; action=ch-dis
        return GateResult(status="optimal",success=True,gate_type="expanded_system_optimum" if allow_curtailment else "battery_only_optimum",terminal_tolerance_kwh=terminal_tolerance,distance_kwh=None,throughput_kwh=float(DT*(ch+dis).sum()),physical_cost_usd=_physical_cost(data,battery,action,q),curtailment_kwh=float(DT*q.sum()),curtailment_price_term_usd=float(DT/1000*np.sum(data["price_usd_mwh"]*q)),solve_seconds=time.perf_counter()-start,stage1_seconds=t,action_kw=action.tolist(),curtailment_kw=q.tolist(),soc_kwh=e.tolist())
    except Exception as exc:
        return GateResult(status="error",success=False,gate_type="expanded_system_optimum" if allow_curtailment else "battery_only_optimum",terminal_tolerance_kwh=terminal_tolerance,solve_seconds=time.perf_counter()-start,message=f"{type(exc).__name__}: {exc}")


def heuristic_action(task: dict[str, Any], battery: Battery, initial_soc: float) -> np.ndarray:
    data=task_suffix(task,battery); n=len(data["load_kw"]); soc=float(initial_soc); out=np.zeros(n)
    for t in range(n):
        remaining=n-t; delta=TARGET_KWH-soc
        if delta>=0:
            desired=delta/(battery.eta_c*DT*remaining)
        else:
            desired=delta*battery.eta_d/(DT*remaining)
        export_floor=-data["export_limit_kw"][t]-data["load_kw"][t]+data["pv_kw"][t]
        max_charge_by_soc=max(data["capacity_kwh"][t]-soc,0)/(battery.eta_c*DT)
        max_discharge_by_soc=max(soc-data["reserve_kwh"][t],0)*battery.eta_d/DT
        lo=max(-data["max_discharge_kw"][t],export_floor,-max_discharge_by_soc)
        hi=min(data["max_charge_kw"][t],max_charge_by_soc)
        u=float(np.clip(desired,lo,hi)) if lo<=hi else float(hi)
        out[t]=u
        soc += battery.eta_c*max(u,0)*DT-max(-u,0)*DT/battery.eta_d
    return out


def deterministic_random_action(scenario_id: str, task: dict[str, Any], battery: Battery) -> np.ndarray:
    seed=int(hashlib.sha256(f"{RANDOM_BASELINE_SEED}:{scenario_id}".encode()).hexdigest()[:16],16)%(2**32)
    rng=np.random.default_rng(seed); data=task_suffix(task,battery)
    return rng.uniform(-battery.max_discharge_kw,battery.max_charge_kw,size=len(data["load_kw"]))


def gate_analysis(f1_records: list[dict[str, Any]], f1_plan_by_block: dict[str,dict[str,Any]], tasks: dict[str,dict[str,Any]], battery: Battery, output_dir: Path) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    # Scenario-level deterministic reference actions and cost optima.
    scenario_refs: dict[str,dict[str,Any]]={}
    first_by_scenario: dict[str,dict[str,Any]]={}
    for record in f1_records:
        first_by_scenario.setdefault(record["scenario_id"],record)
    print(f"[gate] preparing {len(first_by_scenario)} scenario baselines",flush=True)
    for i,(sid,record) in enumerate(sorted(first_by_scenario.items()),start=1):
        task=tasks[sid]["task"]; meta=task_metadata(task); canonical=meta["canonical_soc_kwh"]; stale=meta["stale_soc_kwh"]
        direct=np.asarray(record["same_information_mpc"]["action_kw"],dtype=float)
        stale_opt=solve_cost_optimal(task,battery,stale,allow_curtailment=False)
        same_opt=solve_cost_optimal(task,battery,canonical,allow_curtailment=False)
        sys_opt=solve_cost_optimal(task,battery,canonical,allow_curtailment=True)
        scenario_refs[sid]={
            "zero":np.zeros_like(direct),
            "heuristic":heuristic_action(task,battery,canonical),
            "stale_optimal":np.asarray(stale_opt.action_kw,dtype=float) if stale_opt.success else np.full_like(direct,np.nan),
            "random_static_range":deterministic_random_action(sid,task,battery),
            "direct_mpc":direct,
            "same_space_optimum":same_opt,
            "expanded_system_optimum":sys_opt,
            "archived_mpc_objective_usd":float(record["same_information_mpc"]["objective_usd"]),
            "stale_optimum_result":stale_opt,
        }
        if i%10==0 or i==len(first_by_scenario): print(f"[gate] baseline optima {i}/{len(first_by_scenario)}",flush=True)
    cache: dict[tuple[str,bool,float,str,bool],GateResult]={}
    action_sidecars=[]; llm_rows=[]
    total=len(f1_records)*4
    done=0
    for record in sorted(f1_records,key=lambda r:(r["model_condition"],r["scenario_id"])):
        sid=record["scenario_id"]; task=tasks[sid]["task"]; canonical=task_metadata(task)["canonical_soc_kwh"]
        opt_same=scenario_refs[sid]["same_space_optimum"]; opt_sys=scenario_refs[sid]["expanded_system_optimum"]
        for branch in BRANCHES:
            reference=np.asarray(record["branches"][branch]["parsed"]["dense_action_kw"],dtype=float)
            refhash=action_digest(reference)
            results={}
            for allow,typ in ((False,"battery_only"),(True,"expanded_system")):
                key=(sid,allow,0.0,refhash,True)
                if key not in cache: cache[key]=solve_gate_hybrid(task,battery,reference,canonical,allow_curtailment=allow,terminal_tolerance=0.0,full_lexicographic=True,feasible_action=scenario_refs[sid]["direct_mpc"])
                results[typ]=cache[key]
            engkey=(sid,False,ENGINEERING_TERMINAL_TOL_KWH,refhash,False)
            if engkey not in cache: cache[engkey]=solve_gate_hybrid(task,battery,reference,canonical,allow_curtailment=False,terminal_tolerance=ENGINEERING_TERMINAL_TOL_KWH,full_lexicographic=False,feasible_action=scenario_refs[sid]["direct_mpc"])
            eng=cache[engkey]
            raw_replay=independent_replay(task,reference,canonical,battery)
            row={"scenario_id":sid,"model_condition":record["model_condition"],"block_id":record["block_id"],"branch":branch,"carrier":"canonical" if branch.startswith("C") else "stale","raw_action_digest":refhash,"raw_action_energy_kwh":float(DT*np.abs(reference).sum()),"authoritative_raw_feasible":raw_replay["engineering_feasible"],"D_feasible_set_engineering_kwh":eng.distance_kwh,"D_feasible_set_status":eng.status}
            for typ,res in results.items():
                prefix=f"gate_{typ}_"
                row.update({prefix+"status":res.status,prefix+"distance_kwh":res.distance_kwh,prefix+"distance_lower_bound_kwh":res.distance_lower_bound_kwh,prefix+"distance_optimality_gap_kwh":res.distance_optimality_gap_kwh,prefix+"global_distance_optimality_proven":res.global_distance_optimality_proven,prefix+"lp_overlap_intervals":res.lp_overlap_intervals,prefix+"selected_mode_source":res.selected_mode_source,prefix+"throughput_kwh":res.throughput_kwh,prefix+"cost_usd":res.physical_cost_usd,prefix+"curtailment_kwh":res.curtailment_kwh,prefix+"curtailment_price_term_usd":res.curtailment_price_term_usd,prefix+"exact_retained_fraction":res.exact_retained_fraction,prefix+"correlation":res.correlation,prefix+"solve_seconds":res.solve_seconds})
                optimum=opt_same if typ=="battery_only" else opt_sys
                row[prefix+"regret_usd"]=(res.physical_cost_usd-optimum.physical_cost_usd) if res.success and optimum.success else None
                row[prefix+"regret_fraction_abs_opt"]=(res.physical_cost_usd-optimum.physical_cost_usd)/max(abs(optimum.physical_cost_usd),1.0) if res.success and optimum.success else None
                action_sidecars.append({"kind":"llm","scenario_id":sid,"model_condition":record["model_condition"],"block_id":record["block_id"],"branch":branch,"gate_type":typ,"result":res.__dict__})
            if results["battery_only"].success and results["expanded_system"].success:
                row["expanded_minus_battery_distance_kwh"]=results["expanded_system"].distance_kwh-results["battery_only"].distance_kwh
                row["expanded_distance_reduction_kwh"]=results["battery_only"].distance_kwh-results["expanded_system"].distance_kwh
            llm_rows.append(row); done+=1
            if done%40==0 or done==total: print(f"[gate] F1 branches {done}/{total}; cache={len(cache)}",flush=True)
    baseline_rows=[]
    for i,(sid,refs) in enumerate(sorted(scenario_refs.items()),start=1):
        task=tasks[sid]["task"]; canonical=task_metadata(task)["canonical_soc_kwh"]
        for baseline in ("zero","heuristic","stale_optimal","random_static_range","direct_mpc"):
            reference=np.asarray(refs[baseline],dtype=float)
            if not np.isfinite(reference).all():
                baseline_rows.append({"scenario_id":sid,"baseline":baseline,"status":"reference_unavailable"}); continue
            raw=independent_replay(task,reference,canonical,battery)
            row={"scenario_id":sid,"baseline":baseline,"raw_action_energy_kwh":float(DT*np.abs(reference).sum()),"authoritative_raw_feasible":raw["engineering_feasible"],"raw_terminal_error_kwh":raw["terminal_absolute_error_kwh"],"raw_total_violation_energy":raw["total_violation_energy"]}
            for allow,typ in ((False,"battery_only"),(True,"expanded_system")):
                key=(sid,allow,0.0,action_digest(reference),True)
                if key not in cache: cache[key]=solve_gate_hybrid(task,battery,reference,canonical,allow_curtailment=allow,terminal_tolerance=0.0,full_lexicographic=True,feasible_action=scenario_refs[sid]["direct_mpc"])
                res=cache[key]; opt=refs["same_space_optimum"] if not allow else refs["expanded_system_optimum"]
                prefix=f"gate_{typ}_"
                row.update({prefix+"status":res.status,prefix+"distance_kwh":res.distance_kwh,prefix+"distance_lower_bound_kwh":res.distance_lower_bound_kwh,prefix+"distance_optimality_gap_kwh":res.distance_optimality_gap_kwh,prefix+"global_distance_optimality_proven":res.global_distance_optimality_proven,prefix+"lp_overlap_intervals":res.lp_overlap_intervals,prefix+"selected_mode_source":res.selected_mode_source,prefix+"throughput_kwh":res.throughput_kwh,prefix+"cost_usd":res.physical_cost_usd,prefix+"curtailment_kwh":res.curtailment_kwh,prefix+"solve_seconds":res.solve_seconds,prefix+"regret_usd":res.physical_cost_usd-opt.physical_cost_usd if res.success and opt.success else None,prefix+"regret_fraction_abs_opt":(res.physical_cost_usd-opt.physical_cost_usd)/max(abs(opt.physical_cost_usd),1.0) if res.success and opt.success else None,prefix+"exact_retained_fraction":res.exact_retained_fraction})
                action_sidecars.append({"kind":"baseline","scenario_id":sid,"baseline":baseline,"gate_type":typ,"result":res.__dict__})
            baseline_rows.append(row)
        if i%10==0 or i==len(scenario_refs): print(f"[gate] baseline gates {i}/{len(scenario_refs)}",flush=True)
    sidecar_path=output_dir/"gate_action_sidecars.jsonl.gz"
    with gzip.open(sidecar_path,"wt",encoding="utf-8") as f:
        for row in action_sidecars: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    llm_df=pd.DataFrame(llm_rows); base_df=pd.DataFrame(baseline_rows)
    summary={"unique_gate_cache_entries":len(cache),"llm_rows":len(llm_df),"baseline_rows":len(base_df),"llm_success_counts":{},"baseline_success_counts":{},"optimum_verification":{}}
    for typ in ("battery_only","expanded_system"):
        summary["llm_success_counts"][typ]=int((llm_df[f"gate_{typ}_status"].astype(str)!="error").sum())
        summary["baseline_success_counts"][typ]=int((base_df[f"gate_{typ}_status"].astype(str)!="error").sum())
    diffs=[]
    for sid,refs in scenario_refs.items():
        if refs["same_space_optimum"].success: diffs.append(refs["same_space_optimum"].physical_cost_usd-refs["archived_mpc_objective_usd"])
    summary["optimum_verification"]={"n":len(diffs),"max_abs_cost_diff_usd":float(np.max(np.abs(diffs))) if diffs else None,"mean_cost_diff_usd":float(np.mean(diffs)) if diffs else None}
    return llm_df,base_df,summary


def hand_constructed_replay_tests(battery: Battery) -> list[dict[str,Any]]:
    # Direct small-horizon checker tests, independent of project task generation.
    tests=[]
    def run(name,rows,actions,e0,expected):
        task={"model_visible_episode":{"stage_2":{"remaining_timeseries":rows}},"visible_update":{"activation_step":0}}
        got=independent_replay(task,actions,e0,battery)
        checks={k:(got[k]==v if isinstance(v,bool) else abs(float(got[k])-float(v))<=1e-8) for k,v in expected.items()}
        tests.append({"test":name,"pass":bool(all(checks.values())),"checks":{k:bool(v) for k,v in checks.items()},"observed":{k:(bool(got[k]) if isinstance(got[k], (bool, np.bool_)) else float(got[k])) for k in expected}})
    base=[{"load_kw":100.0,"pv_forecast_kw":0.0,"dam_price_usd_mwh":100.0} for _ in range(4)]
    run("efficiency_and_terminal_exact",base,[100.0]*4,230.0,{"terminal_soc_kwh":325.0,"strict_feasible":True})
    run("terminal_mismatch_zero",base,[0.0]*4,250.0,{"engineering_feasible":False,"terminal_absolute_error_kwh":75.0})
    run("charge_power_violation",base,[300.0,0,0,0],250.0,{"dynamic_constraints_ok":False})
    run("soc_upper_violation",base,[250.0]*4,450.0,{"dynamic_constraints_ok":False})
    run("soc_lower_violation",base,[-250.0]*4,100.0,{"dynamic_constraints_ok":False})
    export_rows=[{"load_kw":0.0,"pv_forecast_kw":400.0,"dam_price_usd_mwh":100.0} for _ in range(4)]
    run("export_violation",export_rows,[0.0]*4,250.0,{"dynamic_constraints_ok":False})
    boundary_task={"model_visible_episode":{"stage_2":{"remaining_timeseries":base}},"visible_update":{"activation_step":0,"max_charge_kw":250.0}}
    got=independent_replay(boundary_task,[250.0,0,0,0],250.0,battery)
    tests.append({"test":"power_exact_boundary","pass":bool(got["trace"][0]["charge_power_exceedance_kw"]==0.0),"observed":float(got["trace"][0]["charge_power_exceedance_kw"])})
    return tests


def build_summary_tables(f1_branch: pd.DataFrame,f1_block: pd.DataFrame,f0_branch: pd.DataFrame,f0_block: pd.DataFrame) -> tuple[dict[str,Any],pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    summary: dict[str,Any]={}
    # Core and robust distributions.
    summary["f1_core"]={}
    metrics=["B_kwh","Wc_kwh","Ws_kwh","ED_kwh","B_avg_kw","Wc_avg_kw","Ws_avg_kw","ED_avg_kw","B_per_soc_gap","ED_per_soc_gap","B_per_battery_capacity","ED_per_battery_capacity","ED_fraction_theoretical_max","terminal_alignment_kwh","mpc_alignment_kwh","soc_direction_net_energy_kwh"]
    for model,g in f1_block.groupby("model_condition"):
        summary["f1_core"][model]={m:summarize_values(g[m]) for m in metrics}
    # Stratified summary rows.
    strat_rows=[]
    strata=["divergence_kwh","soc_gap_direction","event_family","difficulty","season","location_id"]
    temp=f1_block.copy()
    temp["horizon_group"]=pd.cut(temp["horizon_intervals"],bins=[-np.inf,87,90,np.inf],labels=["81-87","88-90","91-92"])
    strata.append("horizon_group")
    for model,gm in temp.groupby("model_condition"):
        for strat in strata:
            for level,g in gm.groupby(strat,observed=False):
                for metric in ("ED_kwh","ED_avg_kw","ED_per_soc_gap","soc_direction_net_energy_kwh","terminal_alignment_kwh","alignment_total_violation_energy"):
                    s=summarize_values(g[metric],bootstrap=False)
                    strat_rows.append({"model_condition":model,"stratum":strat,"level":str(level),"metric":metric,**s})
    strat_df=pd.DataFrame(strat_rows)
    # Correlations.
    corr_rows=[]
    for model,g in f1_block.groupby("model_condition"):
        for x in ("soc_gap_abs_kwh","soc_gap_signed_kwh","horizon_intervals","run_order","block_span_seconds","mean_latency_seconds"):
            for y in ("ED_kwh","ED_avg_kw","ED_per_soc_gap","soc_direction_net_energy_kwh"):
                corr_rows.append({"model_condition":model,"x":x,"y":y,**safe_spearman(g[x],g[y])})
        corr_rows.append({"model_condition":model,"x":"C_duplicate_start_gap_seconds","y":"Wc_kwh",**safe_spearman(g["C_duplicate_start_gap_seconds"],g["Wc_kwh"])})
        corr_rows.append({"model_condition":model,"x":"S_duplicate_start_gap_seconds","y":"Ws_kwh",**safe_spearman(g["S_duplicate_start_gap_seconds"],g["Ws_kwh"])})
    corr_df=pd.DataFrame(corr_rows)
    # Timing by branch position.
    timing_rows=[]
    for model,g in f1_branch.groupby("model_condition"):
        for pos,gp in g.groupby("branch_position"):
            timing_rows.append({"model_condition":model,"branch_position":int(pos),"n":len(gp),"mean_latency_seconds":float(gp["latency_seconds"].mean()),"median_latency_seconds":float(gp["latency_seconds"].median()),"mean_action_energy_kwh":float(gp["action_energy_kwh"].mean()),"exact_zero_rate":float(gp["exact_zero"].mean())})
    timing_df=pd.DataFrame(timing_rows)
    # Activity threshold sensitivity.
    summary["activity_threshold_sensitivity"]={}
    for tier,df in (("F1",f1_branch),("F0",f0_branch)):
        summary["activity_threshold_sensitivity"][tier]={}
        for model,g in df.groupby("model_condition"):
            summary["activity_threshold_sensitivity"][tier][model]={f"{int(th)}_kwh":int(g[f"nontrivial_{int(th)}kwh"].sum()) for th in ACTIVITY_THRESHOLDS_KWH}
    # Violation mechanism counts and severity.
    summary["violation_severity"]={}
    for model,g in f1_branch.groupby("model_condition"):
        fields=["authoritative_power_violation_peak_kw","authoritative_power_violation_energy_kwh","authoritative_soc_lower_violation_peak_kwh","authoritative_soc_upper_violation_peak_kwh","authoritative_export_violation_peak_kw","authoritative_export_violation_energy_kwh","authoritative_terminal_absolute_error_kwh","authoritative_total_violation_energy"]
        summary["violation_severity"][model]={field:summarize_values(g[field]) for field in fields}
        summary["violation_severity"][model]["mechanism_counts"]={
            "power":int(g["authoritative_power_violation_any"].sum()),
            "soc_lower":int(g["authoritative_soc_lower_violation_any"].sum()),
            "soc_upper":int(g["authoritative_soc_upper_violation_any"].sum()),
            "export":int(g["authoritative_export_violation_any"].sum()),
            "terminal":int(g["authoritative_terminal_violation_any"].sum()),
            "multiple_dynamic_types":int(((g[["authoritative_power_violation_any","authoritative_soc_lower_violation_any","authoritative_soc_upper_violation_any","authoritative_export_violation_any"]].sum(axis=1))>=2).sum()),
        }
        summary["violation_severity"][model]["stale_authoritative_raw_feasible"]={"stale_branches":int((g["carrier"]=="stale").sum()),"feasible":int(g.loc[g["carrier"]=="stale","authoritative_raw_feasible"].sum())}
    return summary,strat_df,corr_df,timing_df


def permutation_analysis(f1_block: pd.DataFrame, records: list[dict[str,Any]]) -> dict[str,Any]:
    recmap={(r["scenario_id"],r["model_condition"]):r for r in records}
    rng=np.random.default_rng(PERMUTATION_SEED)
    out={}
    pairings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
    for model,g in f1_block.groupby("model_condition"):
        matrices=[]
        observed=[]
        for _,row in g.iterrows():
            r=recmap[(row["scenario_id"],model)]
            acts=[np.asarray(r["branches"][b]["parsed"]["dense_action_kw"],dtype=float) for b in BRANCHES]
            vals=np.asarray([grouping_ed(acts,*pair) for pair in pairings],dtype=float)
            matrices.append(vals); observed.append(vals[0])
        mat=np.vstack(matrices); obs=float(np.mean(observed))
        choices=rng.integers(0,3,size=(PERMUTATION_RESAMPLES,len(mat)))
        null=mat[np.arange(len(mat))[None,:],choices].mean(axis=1)
        p_upper=float((1+np.sum(null>=obs-1e-12))/(PERMUTATION_RESAMPLES+1))
        p_two=float((1+np.sum(np.abs(null-np.mean(null))>=abs(obs-np.mean(null))-1e-12))/(PERMUTATION_RESAMPLES+1))
        out[model]={"observed_mean_ED_kwh":obs,"null_mean":float(null.mean()),"null_ci95":[float(np.quantile(null,.025)),float(np.quantile(null,.975))],"permutation_p_upper":p_upper,"permutation_p_two_sided_centered":p_two,"resamples":PERMUTATION_RESAMPLES,"unique_pairings_per_block":3,"scenario_pairing_values":mat.tolist()}
    return out


def temporal_pairs(f1_branch:pd.DataFrame,f1_block:pd.DataFrame,f0_branch:pd.DataFrame,f0_block:pd.DataFrame,project:Path) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    # Match by scenario/model/branch and verify payload identity directly from records.
    f1_rec={};f0_rec={}
    for p in (project/"segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records").glob("*.json"):
        r=json.loads(p.read_text());f1_rec[(r["scenario_id"],r["model_condition"])]=r
    for p in (project/"segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records").glob("*.json"):
        r=json.loads(p.read_text());f0_rec[(r["scenario_id"],r["model_condition"])]=r
    rows=[]
    for key,r0 in sorted(f0_rec.items()):
        r1=f1_rec[key]
        for b in BRANCHES:
            a1=np.asarray(r1["branches"][b]["parsed"]["dense_action_kw"],dtype=float);a0=np.asarray(r0["branches"][b]["parsed"]["dense_action_kw"],dtype=float)
            fp1=r1["branches"][b]["response"].get("system_fingerprint")
            fp0=r0["branches"][b]["response"].get("system_fingerprint")
            fp_available=bool(fp1) and bool(fp0)
            rows.append({"scenario_id":key[0],"model_condition":key[1],"branch":b,"prompt_hash_equal":r1["branches"][b]["prompt_sha256"]==r0["branches"][b]["prompt_sha256"],"request_payload_hash_equal":r1["branches"][b]["request_payload_bytes_sha256"]==r0["branches"][b]["request_payload_bytes_sha256"],"returned_model_equal":r1["branches"][b]["response"]["returned_model"]==r0["branches"][b]["response"]["returned_model"],"f1_system_fingerprint":fp1,"f0_system_fingerprint":fp0,"system_fingerprint_available_both":fp_available,"system_fingerprint_equal_when_available":bool(fp_available and fp1==fp0),"exact_action_repeat":bool(np.array_equal(a1,a0)),"cross_window_action_distance_kwh":action_distance(a1,a0),"f1_action_energy_kwh":float(DT*np.abs(a1).sum()),"f0_action_energy_kwh":float(DT*np.abs(a0).sum())})
    pair_df=pd.DataFrame(rows)
    b1=f1_block.set_index(["scenario_id","model_condition"]);b0=f0_block.set_index(["scenario_id","model_condition"])
    brows=[]
    for key in b0.index:
        x1=b1.loc[key];x0=b0.loc[key]
        row={"scenario_id":key[0],"model_condition":key[1]}
        for m in ("B_kwh","Wc_kwh","Ws_kwh","ED_kwh","ED_avg_kw","ED_per_soc_gap","terminal_alignment_kwh","mpc_alignment_kwh","soc_direction_net_energy_kwh"):
            row[f"F1_{m}"]=float(x1[m]);row[f"F0_{m}"]=float(x0[m]);row[f"delta_{m}"]=float(x0[m]-x1[m])
        subset=pair_df[(pair_df.scenario_id==key[0])&(pair_df.model_condition==key[1])]
        row["complete_block_exact_repeat"]=bool(subset.exact_action_repeat.all())
        brows.append(row)
    block_pair_df=pd.DataFrame(brows)
    summary={}
    for model,g in pair_df.groupby("model_condition"):
        gb=block_pair_df[block_pair_df.model_condition==model]
        fp_avail=int(g.system_fingerprint_available_both.sum())
        summary[model]={"branches":len(g),"exact_action_repeats":int(g.exact_action_repeat.sum()),"complete_block_repeats":int(gb.complete_block_exact_repeat.sum()),"prompt_hash_equal":int(g.prompt_hash_equal.sum()),"request_payload_hash_equal":int(g.request_payload_hash_equal.sum()),"returned_model_equal":int(g.returned_model_equal.sum()),"system_fingerprint_available_both":fp_avail,"system_fingerprint_equal_when_available":int(g.system_fingerprint_equal_when_available.sum()),"system_fingerprint_missing_both":int(((g.f1_system_fingerprint.isna()|g.f1_system_fingerprint.eq(""))&(g.f0_system_fingerprint.isna()|g.f0_system_fingerprint.eq(""))).sum()),"f1_system_fingerprint_values":sorted(str(x) for x in g.f1_system_fingerprint.dropna().unique()),"f0_system_fingerprint_values":sorted(str(x) for x in g.f0_system_fingerprint.dropna().unique()),"changed_distance_kwh":summarize_values(g.loc[~g.exact_action_repeat,"cross_window_action_distance_kwh"],bootstrap=False),"ED_drift_kwh":summarize_values(gb["delta_ED_kwh"],bootstrap=False),"ED_sign_transitions":Counter(("pos" if a>1e-12 else "neg" if a<-1e-12 else "zero","pos" if b>1e-12 else "neg" if b<-1e-12 else "zero") for a,b in zip(gb.F1_ED_kwh,gb.F0_ED_kwh))}
        summary[model]["ED_sign_transitions"]={f"{a}->{b}":n for (a,b),n in summary[model]["ED_sign_transitions"].items()}
    return pair_df,block_pair_df,summary


def gate_summaries(llm:pd.DataFrame,base:pd.DataFrame) -> dict[str,Any]:
    out={"llm":{},"baselines":{}}
    for model,g in llm.groupby("model_condition"):
        out["llm"][model]={}
        for typ in ("battery_only","expanded_system"):
            fields=[f"gate_{typ}_distance_kwh",f"gate_{typ}_cost_usd",f"gate_{typ}_regret_usd",f"gate_{typ}_regret_fraction_abs_opt",f"gate_{typ}_curtailment_kwh",f"gate_{typ}_solve_seconds",f"gate_{typ}_exact_retained_fraction"]
            out["llm"][model][typ]={"solved":int(g[f"gate_{typ}_status"].notna().sum()-g[f"gate_{typ}_status"].astype(str).eq("error").sum()),**{field:summarize_values(g[field]) for field in fields}}
        out["llm"][model]["D_feasible_set_engineering_kwh"]=summarize_values(g["D_feasible_set_engineering_kwh"])
        out["llm"][model]["expanded_distance_reduction_kwh"]=summarize_values(g["expanded_distance_reduction_kwh"])
    for baseline,g in base.groupby("baseline"):
        out["baselines"][baseline]={}
        for typ in ("battery_only","expanded_system"):
            out["baselines"][baseline][typ]={field:summarize_values(g[field]) for field in (f"gate_{typ}_distance_kwh",f"gate_{typ}_cost_usd",f"gate_{typ}_regret_usd",f"gate_{typ}_regret_fraction_abs_opt",f"gate_{typ}_curtailment_kwh",f"gate_{typ}_solve_seconds")}
        out["baselines"][baseline]["authoritative_raw_feasible_count"]=int(g.authoritative_raw_feasible.fillna(False).sum()) if "authoritative_raw_feasible" in g else 0
    # Matched comparisons: LLM branch vs zero/heuristic on each scenario.
    for typ in ("battery_only","expanded_system"):
        bwide=base.pivot(index="scenario_id",columns="baseline",values=[f"gate_{typ}_distance_kwh",f"gate_{typ}_cost_usd",f"gate_{typ}_regret_usd"])
        for model,g in llm.groupby("model_condition"):
            for baseline in ("zero","heuristic","stale_optimal","random_static_range"):
                key=f"{model}_minus_{baseline}_{typ}"
                vals={}
                for metric in ("distance_kwh","cost_usd","regret_usd"):
                    left=g[f"gate_{typ}_{metric}"].to_numpy(float)
                    right=np.asarray([bwide.loc[sid,(f"gate_{typ}_{metric}",baseline)] for sid in g.scenario_id],dtype=float)
                    vals[metric]=summarize_values(left-right)
                out.setdefault("matched_llm_minus_baseline",{})[key]=vals
    return out


def write_markdown_report(path:Path,summary:dict[str,Any],gate_summary:dict[str,Any],gate_meta:dict[str,Any],replay_summary:dict[str,Any],temporal_summary:dict[str,Any],permutation:dict[str,Any]) -> None:
    def fmt(x:Any,d=2):
        if x is None:return "NA"
        if isinstance(x,(int,np.integer)):return str(int(x))
        return f"{float(x):.{d}f}"
    lines=["# Offline recomputation report","",f"Generated: {datetime.now(timezone.utc).isoformat()}","","All analyses use saved records only. Provider calls: **0**. Network attempts: **0**.","","## 1. Recomputed F1 carrier metrics",""]
    for model,rep in summary["f1_core"].items():
        ed=rep["ED_kwh"];ned=rep["ED_avg_kw"];soc=rep["ED_per_soc_gap"]
        lines += [f"### {model}","",f"- ED: mean {fmt(ed['mean'])} kWh, median {fmt(ed['median'])}, 95% benchmark-resampling interval [{fmt(ed['bootstrap_ci95'][0])}, {fmt(ed['bootstrap_ci95'][1])}], support +/0/- = {ed['positive']}/{ed['zero']}/{ed['negative']}.",f"- Horizon-normalized ED: mean {fmt(ned['mean'])} kW; median {fmt(ned['median'])} kW.",f"- SOC-gap-normalized ED: mean {fmt(soc['mean'],3)}; median {fmt(soc['median'],3)}.",f"- Mean after removing the largest 1/2/5 ED values: {fmt(ed['mean_drop_largest_1'])}, {fmt(ed['mean_drop_largest_2'])}, {fmt(ed['mean_drop_largest_5'])} kWh.",f"- 10% and 20% trimmed means: {fmt(ed['trimmed_mean_10pct'])} and {fmt(ed['trimmed_mean_20pct'])} kWh.",""]
    lines += ["## 2. Random-pairing sensitivity",""]
    for model,rep in permutation.items():
        lines.append(f"- {model}: observed mean ED {fmt(rep['observed_mean_ED_kwh'])} kWh; random-pairing null 95% interval [{fmt(rep['null_ci95'][0])}, {fmt(rep['null_ci95'][1])}]; upper-tail p={rep['permutation_p_upper']:.5f}.")
    lines += ["","## 3. Independent replay verification","",f"- Saved branch replays checked: {replay_summary['rows_checked']}.",f"- Maximum trace absolute difference: {replay_summary['max_trace_abs_diff']:.3e}.",f"- Maximum scalar absolute difference: {replay_summary['max_scalar_abs_diff']:.3e}.",f"- Rows with any Boolean/count mismatch: {replay_summary['flag_mismatch_rows']}.",f"- Hand-constructed tests passed: {replay_summary['hand_tests_passed']}/{replay_summary['hand_tests_total']}.","","## 4. F1 gate and baseline recomputation","",f"- F1 branch rows: {gate_meta['llm_rows']}; baseline rows: {gate_meta['baseline_rows']}.",f"- Battery-only F1 gates solved: {gate_meta['llm_success_counts']['battery_only']}/{gate_meta['llm_rows']}.",f"- Expanded-system F1 gates solved: {gate_meta['llm_success_counts']['expanded_system']}/{gate_meta['llm_rows']}.",f"- Independent same-space optimum versus archived MPC objective: maximum absolute difference {fmt(gate_meta['optimum_verification']['max_abs_cost_diff_usd'],6)} USD.",""]
    for model,rep in gate_summary["llm"].items():
        b=rep["battery_only"];s=rep["expanded_system"]
        lines += [f"### {model}","",f"- Minimum engineering-feasible same-space correction: median {fmt(rep['D_feasible_set_engineering_kwh']['median'])} kWh; mean {fmt(rep['D_feasible_set_engineering_kwh']['mean'])} kWh.",f"- Exact-terminal battery-only gate correction: median {fmt(b['gate_battery_only_distance_kwh']['median'])} kWh; mean {fmt(b['gate_battery_only_distance_kwh']['mean'])} kWh.",f"- Expanded-system gate correction: median {fmt(s['gate_expanded_system_distance_kwh']['median'])} kWh; mean {fmt(s['gate_expanded_system_distance_kwh']['mean'])} kWh.",f"- Battery-only post-gate regret: median {fmt(b['gate_battery_only_regret_usd']['median'])} USD.",f"- Expanded-system curtailment: median {fmt(s['gate_expanded_system_curtailment_kwh']['median'])} kWh.",""]
    lines += ["## 5. Same-payload temporal rerun",""]
    for model,rep in temporal_summary.items():
        lines.append(f"- {model}: exact branch repeats {rep['exact_action_repeats']}/{rep['branches']}; exact four-branch blocks {rep['complete_block_repeats']}/15; request-payload hashes {rep['request_payload_hash_equal']}/{rep['branches']}; system fingerprints equal {rep['system_fingerprint_equal_when_available']}/{rep['branches']}.")
    lines += ["","## Interpretation boundary","","The recomputation characterizes the frozen challenge set and the two evaluated hosted model conditions. Bootstrap intervals are resampling-stability intervals for these benchmark scenarios. The offline gate results measure deterministic repair and economic outcomes under the disclosed optimization formulations; they do not add new hosted-model evidence.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--project",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    project=args.project.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    print("[1/6] loading frozen records",flush=True)
    battery,tasks,f1_records,f0_records,f1_plan,f0_plan=load_inputs(project)
    print(f"[1/6] loaded tasks={len(tasks)} F1 blocks={len(f1_records)} F0 blocks={len(f0_records)}",flush=True)
    print("[2/6] recomputing branch/block metrics and independent replays",flush=True)
    f1_branch,f1_block,replay1=block_and_branch_rows("F1",f1_records,f1_plan,tasks,battery)
    f0_branch,f0_block,replay0=block_and_branch_rows("F0",f0_records,f0_plan,tasks,battery)
    f1_branch.to_csv(out/"f1_branch_metrics.csv",index=False);f1_block.to_csv(out/"f1_block_metrics.csv",index=False);f0_branch.to_csv(out/"f0_branch_metrics.csv",index=False);f0_block.to_csv(out/"f0_block_metrics.csv",index=False)
    replay_df=pd.DataFrame([*replay1,*replay0]);replay_df.to_csv(out/"independent_replay_comparison.csv",index=False)
    hand=hand_constructed_replay_tests(battery);(out/"independent_replay_hand_tests.json").write_text(json.dumps(hand,indent=2,sort_keys=True)+"\n")
    scalar_cols=[c for c in replay_df if c.startswith("abs_diff_")]
    replay_summary={"rows_checked":len(replay_df),"max_trace_abs_diff":float(replay_df.max_trace_abs_diff.max()),"max_scalar_abs_diff":float(replay_df[scalar_cols].max().max()),"flag_mismatch_rows":int((~replay_df.all_flag_matches).sum()),"hand_tests_passed":sum(bool(x["pass"]) for x in hand),"hand_tests_total":len(hand)}
    print(f"[2/6] replay max diff={replay_summary['max_trace_abs_diff']:.3e}; flag mismatches={replay_summary['flag_mismatch_rows']}",flush=True)
    print("[3/6] robust, stratified, timing, permutation, and temporal analyses",flush=True)
    summary,strat,corr,timing=build_summary_tables(f1_branch,f1_block,f0_branch,f0_block)
    strat.to_csv(out/"f1_stratified_summary.csv",index=False);corr.to_csv(out/"f1_correlation_summary.csv",index=False);timing.to_csv(out/"f1_timing_by_branch_position.csv",index=False)
    permutation=permutation_analysis(f1_block,f1_records)
    temporal_branch,temporal_block,temporal_summary=temporal_pairs(f1_branch,f1_block,f0_branch,f0_block,project)
    temporal_branch.to_csv(out/"f1_f0_branch_pairs.csv",index=False);temporal_block.to_csv(out/"f1_f0_block_pairs.csv",index=False)
    print("[4/6] solving F1 gates and matched baselines",flush=True)
    gate_llm,gate_base,gate_meta=gate_analysis(f1_records,f1_plan,tasks,battery,out)
    gate_llm.to_csv(out/"f1_gate_branch_results.csv",index=False);gate_base.to_csv(out/"gate_baseline_results.csv",index=False)
    gate_summary=gate_summaries(gate_llm,gate_base)
    print("[5/6] writing reports and manifests",flush=True)
    complete={"schema_version":"segan_offline_recompute_v1","created_utc":datetime.now(timezone.utc).isoformat(),"provider_calls":0,"network_attempts":0,"inputs":{"project_archive_expected":"energy-agent-reliability_complete_20260820.tar.gz","F1_blocks":len(f1_records),"F1_branches":len(f1_branch),"F0_blocks":len(f0_records),"F0_branches":len(f0_branch),"unique_tasks":len(tasks)},"analysis":summary,"permutation":permutation,"temporal":temporal_summary,"independent_replay":replay_summary,"gate":gate_summary,"gate_metadata":gate_meta,"interpretation_boundary":"Benchmark-set offline recomputation; no new hosted-model observations."}
    (out/"OFFLINE_RECOMPUTE_SUMMARY.json").write_text(json.dumps(json_safe(complete),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    write_markdown_report(out/"OFFLINE_RECOMPUTE_REPORT.md",summary,gate_summary,gate_meta,replay_summary,temporal_summary,permutation)
    readme="""# SEGAN offline recomputation package\n\nThis package was generated entirely from saved F1/F0 tasks and responses. It performs no provider or network calls.\n\n## Main files\n\n- `OFFLINE_RECOMPUTE_REPORT.md`: concise result narrative.\n- `OFFLINE_RECOMPUTE_SUMMARY.json`: complete machine-readable summary.\n- `f1_branch_metrics.csv`, `f1_block_metrics.csv`: primary recomputed metrics.\n- `f1_stratified_summary.csv`: SOC gap, direction, event, horizon, and other strata.\n- `f1_correlation_summary.csv`: dose, horizon, timing, and latency associations.\n- `f1_gate_branch_results.csv`: F1 branch-level same-space and expanded-system gates.\n- `gate_baseline_results.csv`: zero, heuristic, stale-optimal, random, and direct-MPC baselines.\n- `gate_action_sidecars.jsonl.gz`: full gated action, curtailment, and SOC vectors.\n- `independent_replay_comparison.csv`: independent versus archived replay checks.\n- `f1_f0_branch_pairs.csv`, `f1_f0_block_pairs.csv`: exact-payload temporal pairing.\n- `recompute_offline.py`: full regeneration script.\n\nBootstrap intervals are scenario-resampling stability intervals for the frozen challenge set.\n"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    # Copy script.
    import shutil
    shutil.copy2(Path(__file__),out/"recompute_offline.py")
    manifest=[]
    for p in sorted(out.iterdir()):
        if p.is_file():manifest.append({"path":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    (out/"SHA256_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("[6/6] complete",flush=True)
    print(json.dumps({"status":"complete","output":str(out),"files":len(manifest)+1,"gate_llm_rows":len(gate_llm),"gate_baseline_rows":len(gate_base)},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
