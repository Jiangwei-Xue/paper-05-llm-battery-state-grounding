#!/usr/bin/env python3
"""Independent PV--BESS equation checker for saved SEGAN evidence.

This module deliberately does not import the project simulator, replay scorer,
gate implementation, or the earlier offline recomputation module.  It reads
frozen JSON evidence and evaluates the physical equations directly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DT_HOURS = 0.25
TARGET_SOC_KWH = 325.0
POWER_TOL_KW = 1e-4
ENERGY_TOL_KWH = 1e-5
EXPORT_TOL_KW = 1e-4
ENGINEERING_TERMINAL_TOL_KWH = 1.0
BRANCHES = ("C1", "C2", "S1", "S2")


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    reserve_kwh: float
    export_limit_kw: float
    degradation_cost_per_kwh: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_battery(project: Path) -> Battery:
    config = yaml.safe_load((project / "configs/frozen_protocol.yaml").read_text(encoding="utf-8"))
    row = config["battery"]
    return Battery(
        capacity_kwh=float(row["energy_capacity_kwh"]),
        max_charge_kw=float(row["max_charge_kw"]),
        max_discharge_kw=float(row["max_discharge_kw"]),
        charge_efficiency=float(row["charge_efficiency"]),
        discharge_efficiency=float(row["discharge_efficiency"]),
        reserve_kwh=float(row["reserve_soc_kwh"]),
        export_limit_kw=float(row["export_limit_kw"]),
        degradation_cost_per_kwh=float(row["degradation_cost_per_kwh"]),
    )


def task_arrays(task: dict[str, Any], battery: Battery) -> dict[str, np.ndarray]:
    rows = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    update = task["visible_update"]
    length = len(rows)
    return {
        "load_kw": np.asarray([float(row["load_kw"]) for row in rows], dtype=float),
        "pv_kw": np.asarray([float(row["pv_forecast_kw"]) for row in rows], dtype=float),
        "price_usd_mwh": np.asarray([float(row["dam_price_usd_mwh"]) for row in rows], dtype=float),
        "capacity_kwh": np.full(length, float(update.get("usable_capacity_kwh", battery.capacity_kwh))),
        "max_charge_kw": np.full(length, float(update.get("max_charge_kw", battery.max_charge_kw))),
        "max_discharge_kw": np.full(length, float(update.get("max_discharge_kw", battery.max_discharge_kw))),
        "reserve_kwh": np.full(length, float(update.get("reserve_soc_kwh", battery.reserve_kwh))),
        "export_limit_kw": np.full(length, float(update.get("export_limit_kw", battery.export_limit_kw))),
    }


def evaluate(
    task: dict[str, Any],
    action_kw: Sequence[float],
    initial_soc_kwh: float,
    battery: Battery,
    *,
    curtailment_kw: Sequence[float] | None = None,
    terminal_tolerance_kwh: float = ENGINEERING_TERMINAL_TOL_KWH,
) -> dict[str, Any]:
    arrays = task_arrays(task, battery)
    action = np.asarray(action_kw, dtype=float)
    length = len(arrays["load_kw"])
    curtailment = np.zeros(length, dtype=float) if curtailment_kw is None else np.asarray(curtailment_kw, dtype=float)
    if len(action) != length or len(curtailment) != length:
        raise ValueError("action or curtailment horizon mismatch")
    if not np.isfinite(action).all() or not np.isfinite(curtailment).all() or not math.isfinite(initial_soc_kwh):
        raise ValueError("non-finite checker input")

    soc = float(initial_soc_kwh)
    trace: list[dict[str, float]] = []
    cost = 0.0
    total_violation_energy = 0.0
    dynamic_bad_count = 0
    first_violation: int | None = None
    category_counts = {
        "charge_power": 0,
        "discharge_power": 0,
        "soc_lower": 0,
        "soc_upper": 0,
        "export": 0,
        "curtailment_lower": 0,
        "curtailment_upper": 0,
    }
    for step in range(length):
        requested = float(action[step])
        charge = max(requested, 0.0)
        discharge = max(-requested, 0.0)
        curtailed = float(curtailment[step])
        next_soc = soc + battery.charge_efficiency * charge * DT_HOURS - discharge * DT_HOURS / battery.discharge_efficiency
        charge_excess = max(charge - arrays["max_charge_kw"][step], 0.0)
        discharge_excess = max(discharge - arrays["max_discharge_kw"][step], 0.0)
        soc_lower = max(arrays["reserve_kwh"][step] - next_soc, 0.0)
        soc_upper = max(next_soc - arrays["capacity_kwh"][step], 0.0)
        curtailment_lower = max(-curtailed, 0.0)
        curtailment_upper = max(curtailed - arrays["pv_kw"][step], 0.0)
        grid = arrays["load_kw"][step] - arrays["pv_kw"][step] + curtailed + requested
        export_excess = max(-arrays["export_limit_kw"][step] - grid, 0.0)
        flags = {
            "charge_power": charge_excess > POWER_TOL_KW,
            "discharge_power": discharge_excess > POWER_TOL_KW,
            "soc_lower": soc_lower > ENERGY_TOL_KWH,
            "soc_upper": soc_upper > ENERGY_TOL_KWH,
            "export": export_excess > EXPORT_TOL_KW,
            "curtailment_lower": curtailment_lower > POWER_TOL_KW,
            "curtailment_upper": curtailment_upper > POWER_TOL_KW,
        }
        for category, active in flags.items():
            category_counts[category] += int(active)
        dynamic_bad = any(flags.values())
        if dynamic_bad and first_violation is None:
            first_violation = step
        dynamic_bad_count += int(dynamic_bad)
        total_violation_energy += (
            DT_HOURS * (charge_excess + discharge_excess + export_excess + curtailment_lower + curtailment_upper)
            + soc_lower
            + soc_upper
        )
        cost += DT_HOURS / 1000.0 * arrays["price_usd_mwh"][step] * grid
        cost += battery.degradation_cost_per_kwh * DT_HOURS * (charge + discharge)
        trace.append(
            {
                "offset": step,
                "requested_power_kw": requested,
                "requested_charge_kw": charge,
                "requested_discharge_kw": discharge,
                "curtailment_kw": curtailed,
                "soc_start_kwh": soc,
                "soc_end_kwh": next_soc,
                "charge_power_exceedance_kw": charge_excess,
                "discharge_power_exceedance_kw": discharge_excess,
                "soc_lower_violation_kwh": soc_lower,
                "soc_upper_violation_kwh": soc_upper,
                "grid_kw": grid,
                "export_violation_kw": export_excess,
                "curtailment_lower_violation_kw": curtailment_lower,
                "curtailment_upper_violation_kw": curtailment_upper,
            }
        )
        soc = next_soc
    terminal_signed = soc - TARGET_SOC_KWH
    terminal_abs = abs(terminal_signed)
    dynamic_ok = dynamic_bad_count == 0
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": float(soc),
        "terminal_signed_error_kwh": float(terminal_signed),
        "terminal_absolute_error_kwh": float(terminal_abs),
        "dynamic_constraints_ok": dynamic_ok,
        "engineering_feasible": bool(dynamic_ok and terminal_abs <= terminal_tolerance_kwh),
        "strict_feasible": bool(dynamic_ok and terminal_abs <= 1e-6),
        "violation_interval_count": int(dynamic_bad_count),
        "first_violation_timestep": first_violation,
        "category_counts": category_counts,
        "simultaneous_violation_categories": int(sum(value > 0 for value in category_counts.values())),
        "total_violation_energy": float(total_violation_energy),
        "action_energy_kwh": float(DT_HOURS * np.abs(action).sum()),
        "throughput_kwh": float(DT_HOURS * np.abs(action).sum()),
        "curtailment_kwh": float(DT_HOURS * curtailment.sum()),
        "physical_cost_usd": float(cost),
        "trace": trace,
    }


def severity(result: dict[str, Any]) -> dict[str, Any]:
    trace = result["trace"]
    fields = {
        "charge_power_violation_peak_kw": "charge_power_exceedance_kw",
        "discharge_power_violation_peak_kw": "discharge_power_exceedance_kw",
        "soc_lower_violation_peak_kwh": "soc_lower_violation_kwh",
        "soc_upper_violation_peak_kwh": "soc_upper_violation_kwh",
        "export_violation_peak_kw": "export_violation_kw",
    }
    output = {name: float(max((row[source] for row in trace), default=0.0)) for name, source in fields.items()}
    output.update(
        {
            "soc_lower_violation_sum_kwh_steps": float(sum(row["soc_lower_violation_kwh"] for row in trace)),
            "soc_upper_violation_sum_kwh_steps": float(sum(row["soc_upper_violation_kwh"] for row in trace)),
            "power_violation_energy_kwh": float(
                DT_HOURS
                * sum(row["charge_power_exceedance_kw"] + row["discharge_power_exceedance_kw"] for row in trace)
            ),
            "export_violation_energy_kwh": float(DT_HOURS * sum(row["export_violation_kw"] for row in trace)),
            "terminal_soc_error_kwh": float(result["terminal_absolute_error_kwh"]),
            "terminal_soc_signed_error_kwh": float(result["terminal_signed_error_kwh"]),
            "violation_count": int(result["violation_interval_count"]),
            "first_violation_timestep": result["first_violation_timestep"],
            "simultaneous_violation_categories": int(result["simultaneous_violation_categories"]),
            "engineering_feasible": bool(result["engineering_feasible"]),
            "dynamic_constraints_ok": bool(result["dynamic_constraints_ok"]),
            "physical_cost_usd": float(result["physical_cost_usd"]),
        }
    )
    return output


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_task(project: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = project / str(plan["source_record_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    public_manifest_path = project.parents[1] / "manifests/PUBLIC_RECORD_MANIFEST.jsonl"
    expected = plan["source_record_sha256"]
    if public_manifest_path.is_file():
        public_relative = f"frozen_inputs/project/{plan['source_record_path']}"
        public_hashes = {row["path"]: row["sha256"] for row in load_jsonl(public_manifest_path)}
        expected = public_hashes.get(public_relative)
    if expected is None or actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["task"], actual


def task_initial_soc(task: dict[str, Any]) -> tuple[float, float]:
    intervention = task["v2_prefix_intervention"]
    return float(intervention["canonical_event_soc_kwh"]), float(intervention["stale_event_soc_kwh"])


def compare_saved_replay(actual: dict[str, Any], archived: dict[str, Any]) -> dict[str, Any]:
    trace_fields = (
        "requested_power_kw",
        "soc_start_kwh",
        "soc_end_kwh",
        "charge_power_exceedance_kw",
        "discharge_power_exceedance_kw",
        "soc_lower_violation_kwh",
        "soc_upper_violation_kwh",
        "grid_kw",
        "export_violation_kw",
    )
    if len(actual["trace"]) != len(archived["trace"]):
        return {"match": False, "reason": "trace_length", "max_abs_difference": None}
    max_diff = 0.0
    for left, right in zip(actual["trace"], archived["trace"], strict=True):
        for field in trace_fields:
            max_diff = max(max_diff, abs(float(left[field]) - float(right[field])))
    scalar_fields = (
        "terminal_soc_kwh",
        "terminal_signed_error_kwh",
        "terminal_absolute_error_kwh",
        "total_violation_energy",
        "action_energy_kwh",
        "physical_cost_usd",
    )
    scalar_diff = max(abs(float(actual[field]) - float(archived[field])) for field in scalar_fields)
    flag_fields = ("dynamic_constraints_ok", "engineering_feasible", "strict_feasible", "violation_interval_count")
    flag_match = all(actual[field] == archived[field] for field in flag_fields)
    return {
        "match": bool(max_diff <= 1e-9 and scalar_diff <= 1e-9 and flag_match),
        "max_abs_difference": float(max(max_diff, scalar_diff)),
        "flag_match": flag_match,
    }


def iter_record_groups(project: Path) -> Iterable[tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]]:
    definitions = (
        (
            "F1",
            project / "segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records",
            project / "segan_revision_major_v2/reviews/e2b_protocol_v2/E2B_V2_F1_RUN_PLAN.jsonl",
        ),
        (
            "F0",
            project / "segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records",
            project / "segan_revision_major_v2/reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl",
        ),
        (
            "QWEN37",
            project / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records",
            project / "segan_revision_major_v2/reviews/qwen37plus_f1f0_v1/QWEN37PLUS_FORMAL_RUN_PLAN.jsonl",
        ),
    )
    for tier, record_dir, plan_path in definitions:
        plans = {row["block_id"]: row for row in load_jsonl(plan_path)}
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(record_dir.glob("*.json"))]
        yield tier, records, plans


def synthetic_task(
    *,
    length: int,
    load_kw: float = 100.0,
    pv_kw: float = 0.0,
    capacity_kwh: float = 500.0,
    reserve_kwh: float = 75.0,
    max_charge_kw: float = 250.0,
    max_discharge_kw: float = 250.0,
    export_limit_kw: float = 150.0,
) -> dict[str, Any]:
    rows = [
        {"load_kw": load_kw, "pv_forecast_kw": pv_kw, "dam_price_usd_mwh": 50.0}
        for _ in range(length)
    ]
    return {
        "model_visible_episode": {"stage_2": {"remaining_timeseries": rows}},
        "visible_update": {
            "usable_capacity_kwh": capacity_kwh,
            "reserve_soc_kwh": reserve_kwh,
            "max_charge_kw": max_charge_kw,
            "max_discharge_kw": max_discharge_kw,
            "export_limit_kw": export_limit_kw,
        },
    }


def fixture_results(battery: Battery) -> list[dict[str, Any]]:
    cases: list[tuple[str, dict[str, Any], list[float], float, bool]] = [
        ("normal_feasible", synthetic_task(length=4), [0.0, 0.0, 0.0, 0.0], TARGET_SOC_KWH, True),
        ("soc_upper", synthetic_task(length=2, capacity_kwh=326.0), [10.0, 10.0], 325.0, False),
        ("soc_lower", synthetic_task(length=2, reserve_kwh=324.0), [-10.0, -10.0], 325.0, False),
        ("charge_power", synthetic_task(length=2, max_charge_kw=5.0), [6.0, 0.0], 323.575, False),
        ("discharge_power", synthetic_task(length=2, max_discharge_kw=5.0), [-6.0, 0.0], 326.57894736842104, False),
        ("export", synthetic_task(length=2, load_kw=0.0, pv_kw=200.0, export_limit_kw=150.0), [0.0, 0.0], 325.0, False),
        ("terminal", synthetic_task(length=2), [0.0, 0.0], 300.0, False),
        ("capacity_derating_edge", synthetic_task(length=2, capacity_kwh=300.0), [0.0, 0.0], 325.0, False),
        ("reserve_edge", synthetic_task(length=2, reserve_kwh=326.0), [0.0, 0.0], 325.0, False),
        ("exact_boundary", synthetic_task(length=2, capacity_kwh=325.0, reserve_kwh=325.0), [0.0, 0.0], 325.0, True),
        ("boundary_plus_tolerance", synthetic_task(length=2, capacity_kwh=325.0 + ENERGY_TOL_KWH / 2), [0.0, 0.0], 325.0, True),
        ("boundary_beyond_tolerance", synthetic_task(length=2, capacity_kwh=325.0 - 2 * ENERGY_TOL_KWH), [0.0, 0.0], 325.0, False),
    ]
    output = []
    for name, task, action, initial, expected in cases:
        result = evaluate(task, action, initial, battery)
        output.append(
            {
                "fixture": name,
                "expected_engineering_feasible": expected,
                "observed_engineering_feasible": result["engineering_feasible"],
                "pass": result["engineering_feasible"] is expected,
                "terminal_soc_kwh": result["terminal_soc_kwh"],
                "violation_interval_count": result["violation_interval_count"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--extra-gate-sidecars", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    offline_root = args.offline_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    battery = load_battery(project)

    rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    task_by_scenario: dict[str, dict[str, Any]] = {}
    for tier, records, plans in iter_record_groups(project):
        for record in records:
            plan = plans[record["block_id"]]
            task, source_hash = source_task(project, plan)
            scenario_id = str(record["scenario_id"])
            task_by_scenario[scenario_id] = task
            canonical_soc, stale_soc = task_initial_soc(task)
            for branch_name in BRANCHES:
                branch = record["branches"][branch_name]
                if branch["parsed"].get("ok") is not True:
                    rows.append(
                        {
                            "kind": "hosted_raw",
                            "tier": tier,
                            "scenario_id": scenario_id,
                            "block_id": record["block_id"],
                            "branch": branch_name,
                            "status": "not_applicable_parser_failure",
                            "source_record_sha256": source_hash,
                        }
                    )
                    continue
                action = branch["parsed"]["dense_action_kw"]
                carrier = "canonical" if branch_name.startswith("C") else "stale"
                for replay_name, initial_soc in (
                    ("carrier_consistent", canonical_soc if carrier == "canonical" else stale_soc),
                    ("authoritative", canonical_soc),
                ):
                    checked = evaluate(task, action, initial_soc, battery)
                    archived = branch["score"][f"{replay_name}_replay"]
                    comparison = compare_saved_replay(checked, archived)
                    rows.append(
                        {
                            "kind": "hosted_raw",
                            "tier": tier,
                            "scenario_id": scenario_id,
                            "block_id": record["block_id"],
                            "branch": branch_name,
                            "carrier": carrier,
                            "replay_type": replay_name,
                            "status": "match" if comparison["match"] else "discrepancy",
                            "source_record_sha256": source_hash,
                            **comparison,
                        }
                    )
                    severity_rows.append(
                        {
                            "tier": tier,
                            "scenario_id": scenario_id,
                            "block_id": record["block_id"],
                            "model_condition": record["model_condition"],
                            "event_family": task["event_family"],
                            "difficulty": task["difficulty"],
                            "branch": branch_name,
                            "carrier": carrier,
                            "replay_type": replay_name,
                            "soc_gap_kwh": abs(canonical_soc - stale_soc),
                            "soc_gap_signed_kwh": canonical_soc - stale_soc,
                            **severity(checked),
                        }
                    )

    gate_paths = [offline_root / "gate_action_sidecars.jsonl.gz", *[path.resolve() for path in args.extra_gate_sidecars]]
    for gate_path in gate_paths:
        with gzip.open(gate_path, "rt", encoding="utf-8") as handle:
            gate_lines = list(handle)
        for line in gate_lines:
            if not line.strip():
                continue
            record = json.loads(line)
            result = record["result"]
            scenario_id = str(record["scenario_id"])
            if not result.get("success"):
                rows.append({"kind": "gate", "scenario_id": scenario_id, "status": "solver_unsuccessful"})
                continue
            task = task_by_scenario[scenario_id]
            canonical_soc, _ = task_initial_soc(task)
            terminal_tolerance = float(result.get("terminal_tolerance_kwh", 0.0))
            checked = evaluate(
                task,
                result["action_kw"],
                canonical_soc,
                battery,
                curtailment_kw=result["curtailment_kw"],
                terminal_tolerance_kwh=terminal_tolerance,
            )
            stored_soc = np.asarray(result["soc_kwh"], dtype=float)
            checked_soc = np.asarray([checked["initial_soc_kwh"], *[row["soc_end_kwh"] for row in checked["trace"]]])
            soc_diff = float(np.max(np.abs(stored_soc - checked_soc)))
            metric_diff = max(
                abs(float(result["physical_cost_usd"]) - checked["physical_cost_usd"]),
                abs(float(result["throughput_kwh"]) - checked["throughput_kwh"]),
                abs(float(result["curtailment_kwh"]) - checked["curtailment_kwh"]),
            )
            match = bool(
                checked["dynamic_constraints_ok"]
                and checked["terminal_absolute_error_kwh"] <= terminal_tolerance + ENERGY_TOL_KWH
                and soc_diff <= 2e-5
                and metric_diff <= 2e-5
            )
            rows.append(
                {
                    "kind": "gate",
                    "scenario_id": scenario_id,
                    "model_condition": record.get("model_condition"),
                    "branch": record.get("branch"),
                    "baseline": record.get("baseline"),
                    "gate_type": record["gate_type"],
                    "status": "match" if match else "discrepancy",
                    "match": match,
                    "soc_trace_max_abs_difference_kwh": soc_diff,
                    "metric_max_abs_difference": metric_diff,
                }
            )

    witness_path = project / "segan_revision_major_v2/reviews/v9_provider_free_audit/positive_control/E2_DETERMINISTIC_POSITIVE_CONTROL_WITNESSES.jsonl"
    if witness_path.exists():
        for witness in load_jsonl(witness_path):
            scenario_id = str(witness["scenario_id"])
            witness_task: dict[str, Any] | None = task_by_scenario.get(scenario_id)
            if witness_task is None:
                rows.append({"kind": "positive_control", "scenario_id": scenario_id, "status": "missing_task"})
                continue
            horizon = int(witness["horizon"])
            action = np.full(horizon, float(witness["action"]["power_kw"]), dtype=float)
            checked = evaluate(
                witness_task,
                action.tolist(),
                float(witness["canonical_event_soc_kwh"]),
                battery,
                terminal_tolerance_kwh=1e-6,
            )
            expected = witness["replay"]
            max_diff = max(
                abs(checked["terminal_soc_kwh"] - float(expected["terminal_soc_kwh"])),
                abs(checked["terminal_absolute_error_kwh"] - float(expected["terminal_soc_gap_kwh"])),
            )
            match = bool(checked["engineering_feasible"] == witness["feasible"] and max_diff <= 2e-5)
            rows.append(
                {
                    "kind": "positive_control",
                    "scenario_id": scenario_id,
                    "status": "match" if match else "discrepancy",
                    "match": match,
                    "max_abs_difference": max_diff,
                    "source_record_sha256": witness["source_record_sha256"],
                }
            )

    fixtures = fixture_results(battery)
    row_path = output / "INDEPENDENT_PHYSICS_CROSSCHECK_ROW_LEVEL.jsonl"
    row_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    severity_path = output / "UNIFIED_VIOLATION_SEVERITY.jsonl"
    severity_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in severity_rows), encoding="utf-8")
    (output / "INDEPENDENT_PHYSICS_FIXTURES.json").write_text(json.dumps(fixtures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    discrepancies = [row for row in rows if row["status"] in {"discrepancy", "missing_task", "solver_unsuccessful"}]
    report = {
        "schema_version": "independent_physics_crosscheck_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "checker_independence": {
            "imports_project_simulator": False,
            "imports_project_scorer": False,
            "imports_gate_implementation": False,
            "imports_prior_offline_recompute": False,
        },
        "rows": len(rows),
        "hosted_raw_replay_rows": sum(row["kind"] == "hosted_raw" for row in rows),
        "gate_rows": sum(row["kind"] == "gate" for row in rows),
        "positive_control_rows": sum(row["kind"] == "positive_control" for row in rows),
        "parser_not_applicable_rows": sum(row["status"] == "not_applicable_parser_failure" for row in rows),
        "discrepancies": len(discrepancies),
        "fixture_passed": sum(item["pass"] for item in fixtures),
        "fixture_total": len(fixtures),
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "status": "PASS" if not discrepancies and all(item["pass"] for item in fixtures) else "HOLD",
        "artifacts": {
            "row_level": row_path.name,
            "violation_severity": severity_path.name,
            "fixtures": "INDEPENDENT_PHYSICS_FIXTURES.json",
        },
    }
    (output / "INDEPENDENT_PHYSICS_CROSSCHECK.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Independent physics cross-check",
        "",
        f"- Status: **{report['status']}**",
        f"- Saved replay comparisons: **{report['hosted_raw_replay_rows']}**",
        f"- Saved gate trajectories: **{report['gate_rows']}**",
        f"- Deterministic positive-control witnesses: **{report['positive_control_rows']}**",
        f"- Hand-constructed boundary fixtures: **{report['fixture_passed']}/{report['fixture_total']}**",
        f"- Discrepancies: **{report['discrepancies']}**",
        "- Provider calls: **0**; network attempts: **0**.",
        "",
        "The checker implements the SOC, power, reserve, capacity, export, curtailment, terminal, and cost equations directly. It does not import the project simulator, scorer, gate, or prior recomputation code.",
    ]
    (output / "INDEPENDENT_PHYSICS_CROSSCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
