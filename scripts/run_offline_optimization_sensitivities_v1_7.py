#!/usr/bin/env python3
"""Run provider-free gate objective-order and deterministic-reference sensitivity.

The sample and parameter grid are fixed in code before solving. Results are a
diagnostic tier and do not replace the registered gate or admission protocol.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse  # type: ignore[import-untyped]
from scipy.optimize import LinearConstraint  # type: ignore[import-untyped]

SCHEMA = "offline_optimization_sensitivities_v1"
COST_FACE_TOLERANCE_USD = 1e-6
STRATIFIED_SCENARIO_COUNT = 20


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("offline_recompute_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_sidecars(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind") == "llm" and row.get("gate_type") == "battery_only":
                    result[(str(row["block_id"]), str(row["branch"]), "battery_only")] = row["result"]
    return result


def select_scenarios(tasks: dict[str, dict[str, Any]]) -> list[str]:
    ordered = sorted(
        tasks,
        key=lambda sid: (
            str(tasks[sid]["task"]["event_family"]),
            str(tasks[sid]["task"]["difficulty"]),
            sid,
        ),
    )
    by_event: dict[str, list[str]] = {}
    for sid in ordered:
        by_event.setdefault(str(tasks[sid]["task"]["event_family"]), []).append(sid)
    selected: list[str] = []
    event_names = sorted(by_event)
    cursor = 0
    while len(selected) < min(STRATIFIED_SCENARIO_COUNT, len(ordered)):
        event = event_names[cursor % len(event_names)]
        index = cursor // len(event_names)
        if index < len(by_event[event]):
            selected.append(by_event[event][index])
        cursor += 1
    return selected


def model_actions(
    f1_records: list[dict[str, Any]], qwen_records: list[dict[str, Any]]
) -> dict[tuple[str, str], tuple[str, np.ndarray]]:
    out: dict[tuple[str, str], tuple[str, np.ndarray]] = {}
    for record in [*f1_records, *qwen_records]:
        branch = record["branches"]["C1"]
        if branch["parsed"].get("ok") is not True:
            continue
        out[(str(record["scenario_id"]), str(record["model_condition"]))] = (
            str(record["block_id"]), np.asarray(branch["parsed"]["dense_action_kw"], dtype=float)
        )
    return out


def cost_vector(module: Any, data: dict[str, np.ndarray], battery: Any, idx: dict[str, Any], nvar: int) -> np.ndarray:
    c = np.zeros(nvar)
    c[idx["c"]] = module.DT / 1000 * data["price_usd_mwh"] + battery.degradation_cost_per_kwh * module.DT
    c[idx["d"]] = -module.DT / 1000 * data["price_usd_mwh"] + battery.degradation_cost_per_kwh * module.DT
    c[idx["q"]] = module.DT / 1000 * data["price_usd_mwh"]
    return c


def append_upper(constraint: LinearConstraint, coefficients: np.ndarray, upper: float) -> LinearConstraint:
    matrix = sparse.vstack([constraint.A, sparse.csr_matrix(coefficients.reshape(1, -1))], format="csr")
    lower = np.concatenate([np.asarray(constraint.lb, dtype=float), [-np.inf]])
    uppers = np.concatenate([np.asarray(constraint.ub, dtype=float), [upper]])
    return LinearConstraint(matrix, lower, uppers)


def cost_then_distance(module: Any, task: dict[str, Any], battery: Any, reference: np.ndarray, initial_soc: float) -> dict[str, Any]:
    start = time.perf_counter()
    lower, upper, integ, _, constraint, idx, data = module._gate_problem_arrays(
        task, battery, reference, initial_soc, False, 0.0
    )
    c_cost = cost_vector(module, data, battery, idx, len(lower))
    first, first_seconds = module._solve_milp(c_cost, lower, upper, integ, constraint, "cost-first")
    cost_opt = float(c_cost @ first.x)
    second_constraint = append_upper(constraint, c_cost, cost_opt + COST_FACE_TOLERANCE_USD)
    c_distance = np.zeros(len(lower))
    c_distance[idx["a"]] = module.DT
    second, second_seconds = module._solve_milp(
        c_distance, lower, upper, integ, second_constraint, "distance-within-cost-face"
    )
    action = second.x[idx["c"]] - second.x[idx["d"]]
    curtailment = second.x[idx["q"]]
    return {
        "status": "optimal",
        "cost_optimum_usd": cost_opt,
        "physical_cost_usd": module._physical_cost(data, battery, action, curtailment),
        "distance_kwh": float(module.DT * np.abs(action - reference).sum()),
        "throughput_kwh": float(module.DT * np.sum(second.x[idx["c"]] + second.x[idx["d"]])),
        "curtailment_kwh": float(module.DT * np.maximum(curtailment, 0).sum()),
        "retained_fraction": float(np.mean(np.abs(action - reference) <= 1e-6)),
        "stage1_seconds": first_seconds,
        "stage2_seconds": second_seconds,
        "runtime_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_module(args.reference_script.resolve())
    battery, tasks, f1_records, _, _, _ = module.load_inputs(project)
    qwen_dir = project / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records"
    qwen_records = [load_json(path) for path in sorted(qwen_dir.glob("qwen37f1_*.json"))]
    actions = model_actions(f1_records, qwen_records)
    selected = select_scenarios(tasks)
    sidecars = load_gate_sidecars([
        args.offline_root / "gate_action_sidecars.jsonl.gz",
        args.qwen_root / "gate_action_sidecars.jsonl.gz",
    ])
    plan = {
        "schema_version": SCHEMA,
        "selected_scenario_ids": selected,
        "selection_rule": "round-robin by sorted event family, then difficulty and scenario id; fixed count 20",
        "branch": "C1",
        "models": sorted({model for _, model in actions}),
        "gate_action_space": "battery_only",
        "objective_orders": [
            "distance_throughput_cost_saved", "minimum_intervention_distance_only",
            "cost_then_distance", "direct_cost_optimal",
        ],
        "cost_face_tolerance_usd": COST_FACE_TOLERANCE_USD,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    (output / "objective_order_sample_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for scenario_id in selected:
        task = tasks[scenario_id]["task"]
        initial = float(task["v2_prefix_intervention"]["canonical_event_soc_kwh"])
        direct = module.solve_cost_optimal(task, battery, initial, allow_curtailment=False, terminal_tolerance=0.0)
        for (sid, model), (block, reference) in sorted(actions.items()):
            if sid != scenario_id:
                continue
            common = {"scenario_id": sid, "event_family": task["event_family"], "difficulty": task["difficulty"], "model_condition": model, "block_id": block, "branch": "C1"}
            saved = sidecars[(block, "C1", "battery_only")]
            rows.append({**common, "objective_order": "distance_throughput_cost_saved", "status": saved["status"], "physical_cost_usd": saved["physical_cost_usd"], "distance_kwh": saved["distance_kwh"], "throughput_kwh": saved["throughput_kwh"], "curtailment_kwh": saved["curtailment_kwh"], "retained_fraction": saved["exact_retained_fraction"], "runtime_seconds": saved["solve_seconds"]})
            minimum = module.solve_gate_hybrid(task, battery, reference, initial, allow_curtailment=False, terminal_tolerance=0.0, full_lexicographic=False, feasible_action=direct.action_kw)
            rows.append({**common, "objective_order": "minimum_intervention_distance_only", "status": minimum.status, "physical_cost_usd": minimum.physical_cost_usd, "distance_kwh": minimum.distance_kwh, "throughput_kwh": minimum.throughput_kwh, "curtailment_kwh": minimum.curtailment_kwh, "retained_fraction": minimum.exact_retained_fraction, "runtime_seconds": minimum.solve_seconds})
            rows.append({**common, "objective_order": "direct_cost_optimal", "status": direct.status, "physical_cost_usd": direct.physical_cost_usd, "distance_kwh": float(module.action_distance(reference, direct.action_kw)), "throughput_kwh": direct.throughput_kwh, "curtailment_kwh": direct.curtailment_kwh, "retained_fraction": float(np.mean(np.abs(np.asarray(direct.action_kw)-reference)<=1e-6)), "runtime_seconds": direct.solve_seconds})
            try:
                result = cost_then_distance(module, task, battery, reference, initial)
                rows.append({**common, "objective_order": "cost_then_distance", **result})
            except Exception as exc:
                errors.append({**common, "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "objective_order_sensitivity_rows.csv", index=False)
    summary = frame.groupby(["model_condition", "objective_order"], dropna=False).agg(
        n=("scenario_id", "count"), mean_cost_usd=("physical_cost_usd", "mean"),
        median_cost_usd=("physical_cost_usd", "median"), mean_distance_kwh=("distance_kwh", "mean"),
        median_distance_kwh=("distance_kwh", "median"), mean_retained_fraction=("retained_fraction", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    ).reset_index()
    summary.to_csv(output / "objective_order_sensitivity_summary.csv", index=False)

    reference_rows: list[dict[str, Any]] = []
    for scenario_id, wrapped in sorted(tasks.items()):
        task = wrapped["task"]
        initial = float(task["v2_prefix_intervention"]["canonical_event_soc_kwh"])
        for degradation in (0.0, 0.008, 0.02):
            test_battery = replace(battery, degradation_cost_per_kwh=degradation)
            for tolerance in (0.0, 1.0):
                result = module.solve_cost_optimal(task, test_battery, initial, allow_curtailment=False, terminal_tolerance=tolerance)
                reference_rows.append({
                    "scenario_id": scenario_id, "event_family": task["event_family"], "difficulty": task["difficulty"],
                    "degradation_cost_per_kwh": degradation, "terminal_tolerance_kwh": tolerance,
                    "status": result.status, "physical_cost_usd": result.physical_cost_usd,
                    "throughput_kwh": result.throughput_kwh, "runtime_seconds": result.solve_seconds,
                })
    reference_frame = pd.DataFrame(reference_rows)
    reference_frame.to_csv(output / "deterministic_reference_boundary_sensitivity.csv", index=False)
    complete: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "objective_order": {"sample_scenarios": len(selected), "rows": len(rows), "errors": errors, "status": "PASS" if not errors else "PARTIAL"},
        "deterministic_reference": {"scenarios": len(tasks), "rows": len(reference_rows), "failed": int((reference_frame["status"] == "error").sum())},
        "interpretation_boundary": "Provider-free post-hoc sensitivity. It does not redefine the registered gate or scenario admission.",
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    (output / "OFFLINE_OPTIMIZATION_SENSITIVITIES.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Offline optimization sensitivities v1", "", f"Objective-order sample: {len(selected)} scenarios, C1 only, {len(rows)} result rows.", f"Objective-order solver errors: {len(errors)}.", f"Deterministic-reference grid: {len(reference_rows)} rows across 60 scenarios, terminal tolerance 0/1 kWh, and degradation 0/0.008/0.02 USD/kWh.", "", "These are provider-free post-hoc diagnostics. They do not redefine the registered gate or admission protocol."]
    (output / "OFFLINE_OPTIMIZATION_SENSITIVITIES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": complete["objective_order"]["status"], "rows": len(rows), "reference_rows": len(reference_rows), "provider_calls": 0, "network_attempts": 0}))


if __name__ == "__main__":
    main()
