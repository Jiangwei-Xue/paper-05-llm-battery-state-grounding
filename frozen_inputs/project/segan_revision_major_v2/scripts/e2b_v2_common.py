#!/usr/bin/env python3
"""Provider-free E2b-v2 task, prompt, scoring, and evidence primitives."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "runs/experiments_v2/e2/e2_20260801T054708Z/records"
PROMPT_TEMPLATE = ROOT / "protocol/E2B_PROMPT_TEMPLATE_V2.txt"
MODEL_CONFIG = PROJECT / "experiments_v2/configs/models.yaml"
STEP_HOURS = 0.25
TERMINAL_TARGET_KWH = 325.0
ENGINEERING_TERMINAL_TOLERANCE_KWH = 1.0
POWER_TOLERANCE_KW = 1e-4
ENERGY_TOLERANCE_KWH = 1e-5
EXPORT_TOLERANCE_KW = 1e-4

sys.path.insert(0, str(PROJECT / "experiments_v2/src"))
sys.path.insert(0, str(PROJECT / "src"))

from experiments_v2.e2 import e2_states  # type: ignore[import-not-found]  # noqa: E402

from energy_agent_reliability.config import load_protocol  # noqa: E402
from energy_agent_reliability.online_gate_v6 import (  # noqa: E402
    economic_dispatch,
    limits_at,
    task_frame,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unique_tasks() -> list[dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(SOURCE.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        task = record["task"]
        scenario_id = str(task["scenario_id"])
        if scenario_id not in tasks:
            tasks[scenario_id] = {
                "scenario_id": scenario_id,
                "source_record_path": str(path.relative_to(PROJECT)),
                "source_record_sha256": sha256(path),
                "task_sha256": digest(task),
                "task": task,
            }
    return [tasks[key] for key in sorted(tasks)]


def battery_config() -> Any:
    protocol = load_protocol(PROJECT / "configs/frozen_protocol.yaml")
    return protocol.battery.model_copy(update={"terminal_soc_kwh": TERMINAL_TARGET_KWH})


def carrier_for(task: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch not in {"C1", "C2", "S1", "S2"}:
        raise ValueError(f"unknown E2b-v2 branch: {branch}")
    return cast(dict[str, Any], e2_states(task)["C" if branch.startswith("C") else "S"])


def visible_suffix_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    activation = int(task["visible_update"]["activation_step"])
    frame = task_frame(task).iloc[activation:]
    battery = battery_config()
    rows: list[dict[str, Any]] = []
    for offset, (timestamp, values) in enumerate(frame.iterrows()):
        limits = limits_at(task, activation + offset, battery, apply_event=True)
        rows.append(
            {
                "offset": offset,
                "timestamp_utc": str(timestamp).replace(" ", "T").replace("+00:00", "Z"),
                "load_kw": float(values["load_kw"]),
                "pv_forecast_kw": float(values["pv_kw"]),
                "price_usd_mwh": float(values["price_usd_mwh"]),
                "usable_capacity_kwh": limits["usable_capacity_kwh"],
                "charge_limit_kw": limits["max_charge_kw"],
                "discharge_limit_kw": limits["max_discharge_kw"],
                "reserve_kwh": limits["reserve_soc_kwh"],
                "export_limit_kw": limits["export_limit_kw"],
            }
        )
    return rows


def model_visible_contract(task: dict[str, Any], branch: str) -> dict[str, Any]:
    carrier = carrier_for(task, branch)
    rows = visible_suffix_rows(task)
    return {
        "protocol_id": "energybench-e2b-stage2-only-carrier-sensitivity-v2",
        "scenario_id": str(task["scenario_id"]),
        "carrier_state": {
            "soc_kwh": float(carrier["soc_kwh"]),
            "usable_capacity_kwh": float(carrier["usable_capacity_kwh"]),
            "charge_limit_kw": float(carrier["charge_limit_kw"]),
            "discharge_limit_kw": float(carrier["discharge_limit_kw"]),
            "reserve_kwh": float(carrier["reserve_kwh"]),
            "export_limit_kw": float(carrier["export_limit_kw"]),
        },
        "suffix_horizon_intervals": len(rows),
        "terminal_target_kwh": TERMINAL_TARGET_KWH,
        "timeseries": rows,
    }


def render_prompt(task: dict[str, Any], branch: str) -> str:
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{TASK_CONTRACT_JSON}}", canonical_json(model_visible_contract(task, branch)))


def request_payload(model_spec: dict[str, Any], prompt: str) -> dict[str, Any]:
    provider = str(model_spec["provider"])
    payload: dict[str, Any] = {
        "model": str(model_spec["configured_model_id"]),
        "stream": False,
        "temperature": float(model_spec["temperature"]),
        "max_tokens": int(model_spec["max_output_tokens"]),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "qwen":
        payload["enable_thinking"] = False
    else:
        raise ValueError(f"unsupported frozen provider: {provider}")
    return payload


def dense_to_segments(actions: list[float], *, zero_tolerance: float = 1e-9) -> dict[str, Any]:
    segments: list[dict[str, int | float]] = []
    for offset, value in enumerate(actions):
        power = 0.0 if abs(float(value)) <= zero_tolerance else float(value)
        if power == 0.0:
            continue
        segments.append(
            {
                "start_offset": offset,
                "end_offset_exclusive": offset + 1,
                "power_kw": power,
            }
        )
    return {"actions": segments}


def _safe_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score contains non-finite value")
    return float(value)


def raw_replay(
    task: dict[str, Any],
    actions_kw: list[float],
    *,
    initial_soc_kwh: float,
) -> dict[str, Any]:
    rows = visible_suffix_rows(task)
    if len(actions_kw) != len(rows):
        raise ValueError("action length does not match suffix horizon")
    battery = battery_config()
    eta_c = float(battery.charge_efficiency)
    eta_d = float(battery.discharge_efficiency)
    degradation = float(battery.degradation_cost_per_kwh)
    soc = float(initial_soc_kwh)
    trace: list[dict[str, Any]] = []
    total_violation_energy = 0.0
    total_cost = 0.0
    violation_intervals = 0
    for row, requested in zip(rows, actions_kw, strict=True):
        power = _safe_number(float(requested))
        charge = max(power, 0.0)
        discharge = max(-power, 0.0)
        next_soc = soc + eta_c * charge * STEP_HOURS - discharge * STEP_HOURS / eta_d
        charge_excess = max(charge - float(row["charge_limit_kw"]), 0.0)
        discharge_excess = max(discharge - float(row["discharge_limit_kw"]), 0.0)
        soc_lower = max(float(row["reserve_kwh"]) - next_soc, 0.0)
        soc_upper = max(next_soc - float(row["usable_capacity_kwh"]), 0.0)
        grid = float(row["load_kw"]) - float(row["pv_forecast_kw"]) + power
        export_excess = max(-float(row["export_limit_kw"]) - grid, 0.0)
        interval_violation = any(
            value > tolerance
            for value, tolerance in (
                (charge_excess, POWER_TOLERANCE_KW),
                (discharge_excess, POWER_TOLERANCE_KW),
                (soc_lower, ENERGY_TOLERANCE_KWH),
                (soc_upper, ENERGY_TOLERANCE_KWH),
                (export_excess, EXPORT_TOLERANCE_KW),
            )
        )
        violation_intervals += int(interval_violation)
        total_violation_energy += STEP_HOURS * (
            charge_excess + discharge_excess + export_excess
        ) + soc_lower + soc_upper
        total_cost += STEP_HOURS / 1000.0 * float(row["price_usd_mwh"]) * grid
        total_cost += degradation * STEP_HOURS * (charge + discharge)
        trace.append(
            {
                "offset": int(row["offset"]),
                "timestamp_utc": row["timestamp_utc"],
                "requested_power_kw": power,
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
            }
        )
        soc = next_soc
    terminal_signed = soc - TERMINAL_TARGET_KWH
    terminal_abs = abs(terminal_signed)
    dynamic_constraints_ok = violation_intervals == 0
    thresholds = {
        "1e-6_kwh": terminal_abs <= 1e-6,
        "1_kwh": terminal_abs <= 1.0,
        "5_kwh": terminal_abs <= 5.0,
        "10_kwh": terminal_abs <= 10.0,
        "25_kwh": terminal_abs <= 25.0,
    }
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": soc,
        "terminal_signed_error_kwh": terminal_signed,
        "terminal_absolute_error_kwh": terminal_abs,
        "terminal_thresholds": thresholds,
        "dynamic_constraints_ok": dynamic_constraints_ok,
        "engineering_feasible": bool(dynamic_constraints_ok and thresholds["1_kwh"]),
        "strict_feasible": bool(dynamic_constraints_ok and thresholds["1e-6_kwh"]),
        "violation_interval_count": violation_intervals,
        "total_violation_energy": total_violation_energy,
        "action_energy_kwh": STEP_HOURS * float(np.abs(np.asarray(actions_kw)).sum()),
        "segment_nonzero_interval_count": sum(abs(float(value)) > 1e-9 for value in actions_kw),
        "physical_cost_usd": total_cost,
        "trace": trace,
    }


def score_branch(task: dict[str, Any], branch: str, actions_kw: list[float]) -> dict[str, Any]:
    states = e2_states(task)
    carrier_initial = float(states["C" if branch.startswith("C") else "S"]["soc_kwh"])
    authoritative_initial = float(states["C"]["soc_kwh"])
    return {
        "carrier_consistent_replay": raw_replay(
            task, actions_kw, initial_soc_kwh=carrier_initial
        ),
        "authoritative_replay": raw_replay(
            task, actions_kw, initial_soc_kwh=authoritative_initial
        ),
    }


def same_information_mpc(task: dict[str, Any]) -> dict[str, Any]:
    states = e2_states(task)
    activation = int(task["visible_update"]["activation_step"])
    solution = economic_dispatch(
        task,
        battery_config(),
        initial_soc_kwh=float(states["C"]["soc_kwh"]),
        start_step=activation,
        apply_event=True,
        allow_curtailment=False,
    )
    actions = [float(value) for value in solution.action_kw]
    replay = raw_replay(task, actions, initial_soc_kwh=float(states["C"]["soc_kwh"]))
    return {
        "status": solution.solver_status,
        "action_kw": actions,
        "objective_usd": float(solution.objective_usd),
        "raw_replay": replay,
    }


def action_distance_kwh(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("unequal action horizons")
    return STEP_HOURS * float(np.abs(np.asarray(left) - np.asarray(right)).sum())
