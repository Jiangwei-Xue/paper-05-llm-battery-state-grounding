"""Failure decomposition for model runs that are permanently excluded from formal analysis."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .baselines import BaselineError, perfect_information_baseline, rolling_horizon_baseline
from .battery import BatteryOverrides, SimulationResult, simulate_dispatch
from .config import BatteryConfig, FrozenProtocol
from .episode import canonical_typed_state, execute_two_stage_episode
from .events import dynamic_override_schedule
from .metrics import (
    action_success_from_simulation,
    normalized_regret,
    operational_feasibility_success,
)
from .provenance import read_jsonl, sha256_file, utc_now, write_jsonl
from .state_methods import StateMethod, validate_explicit_state

BaselineView = Literal["recorded", "recomputed"]
TOLERANCE = 1e-8


def analyze_exploratory_run(
    *,
    project_root: Path,
    source_run_dir: Path,
    protocol: FrozenProtocol,
    baseline_view: BaselineView,
) -> dict[str, Any]:
    """Reconstruct per-episode state, execution, constraint, baseline, and usage outcomes."""
    source_report_path = source_run_dir / "EXPLORATORY_EXCLUDED_REPORT.json"
    source_report = _read_json(source_report_path)
    if source_report.get("excluded_from_formal_analysis") is not True:
        raise ValueError("Failure decomposition accepts excluded exploratory runs only.")
    pilot_path = project_root / "scenarios" / "frozen_pilot_tasks.jsonl"
    admission_path = project_root / "scenarios" / "admission_results.jsonl"
    pilots = {row["scenario_id"]: row for row in read_jsonl(pilot_path)}
    admissions = {row["scenario_id"]: row for row in read_jsonl(admission_path)}
    record_paths = [source_run_dir / name for name in source_report["record_files"]]
    rows = [
        decompose_episode(
            project_root=project_root,
            protocol=protocol,
            record=_read_json(record_path),
            candidate=pilots[_read_json(record_path)["scenario_id"]],
            admission=admissions[_read_json(record_path)["scenario_id"]],
            baseline_view=baseline_view,
        )
        for record_path in record_paths
    ]
    rows.sort(key=lambda item: item["scenario_id"])
    return {
        "schema_version": "exploratory_failure_decomposition_v1",
        "created_utc": utc_now(),
        "status": "complete",
        "run_kind": "exploratory_excluded_analysis",
        "excluded_from_formal_analysis": True,
        "formal_analysis_eligible": False,
        "source_run_dir": str(source_run_dir),
        "source_report_sha256": sha256_file(source_report_path),
        "source_pilot_sha256": sha256_file(pilot_path),
        "baseline_view": baseline_view,
        "episode_count": len(rows),
        "cost_to_mpc_definition": "(episode_cost-mpc_cost)/max(abs(mpc_cost),1); null unless MPC is feasible",
        "gx_f_table": gx_f_table(rows),
        "state_failure_field_classification": _counter_rows(
            reason for row in rows for reason in row["state_failure_fields"]
        ),
        "state_failure_root_classification": _counter_rows(
            code for row in rows for code in row["state_failure_root_codes"]
        ),
        "infeasible_episodes": [
            {
                "scenario_id": row["scenario_id"],
                "constraint_failure_codes": row["constraint_failure_codes"],
                "terminal_soc_error": row["terminal_soc_error"],
            }
            for row in rows
            if not row["operational_feasibility_success"]
        ],
        "baseline_error_classification": _counter_rows(
            row["baseline_error_code"] for row in rows if row["baseline_error_code"] is not None
        ),
        "stage_usage_latency": [
            {
                "scenario_id": row["scenario_id"],
                "stage1": row["stage1"],
                "stage2": row["stage2"],
            }
            for row in rows
        ],
        "episodes": rows,
    }


def decompose_episode(
    *,
    project_root: Path,
    protocol: FrozenProtocol,
    record: dict[str, Any],
    candidate: dict[str, Any],
    admission: dict[str, Any],
    baseline_view: BaselineView,
) -> dict[str, Any]:
    frame = _load_frame(project_root, protocol, candidate)
    stage1_parse = record["stage1"]["parse"]
    stage2_parse = record["stage2"]["parse"]
    parser_success = bool(stage1_parse.get("ok") and stage2_parse.get("ok"))
    expected_stage1 = int(candidate["model_visible_episode"]["stage_1"]["required_dispatch_plan_steps"])
    expected_stage2 = int(candidate["model_visible_episode"]["stage_2"]["required_dispatch_plan_steps"])
    dispatch_execution_success = bool(
        parser_success
        and len(stage1_parse.get("dispatch_kw", [])) == expected_stage1
        and len(stage2_parse.get("dispatch_kw", [])) == expected_stage2
    )
    state_method = str(record["state_method"])
    if state_method not in {"rolling_summary", "visible_carry", "typed_state", "canonical_typed_carry"}:
        raise ValueError(f"Unknown state method in record: {state_method}")

    state_success = False
    state_failure_fields: list[str] = ["parser_failure"] if not parser_success else []
    state_root_codes: list[str] = ["STATE_UNAVAILABLE_AFTER_PARSE_FAILURE"] if not parser_success else []
    operational_success = False
    constraint_codes: list[str] = ["DISPATCH_UNAVAILABLE_AFTER_PARSE_FAILURE"] if not parser_success else []
    terminal_error: float | None = None
    episode_cost: float | None = None
    if parser_success:
        stage1_actions = [float(value) for value in stage1_parse["dispatch_kw"]]
        stage2_actions = [float(value) for value in stage2_parse["dispatch_kw"]]
        executed = execute_two_stage_episode(
            frame,
            stage1_actions,
            stage2_actions,
            protocol.battery,
            candidate["visible_update"],
        )
        simulation = SimulationResult(
            executed.trace,
            executed.total_cost,
            executed.feasible,
            executed.terminal_soc_gap_kwh,
        )
        expected_state = _expected_state(protocol, candidate, frame, stage1_actions)
        validation = validate_explicit_state(
            stage2_parse.get("parsed_state"),
            expected_state,
            method=cast(StateMethod, state_method),
        )
        state_success = validation.passed
        state_failure_fields = validation.reasons
        state_root_codes = state_failure_root_codes(stage2_parse.get("parsed_state"), validation.reasons)
        action_success = action_success_from_simulation(simulation)
        operational_success = operational_feasibility_success(simulation, action_success)
        constraint_codes = constraint_failure_codes(
            simulation,
            frame,
            protocol.battery,
            candidate["visible_update"],
        )
        terminal_error = float(executed.terminal_soc_gap_kwh)
        episode_cost = float(executed.total_cost)

    if baseline_view == "recorded":
        baseline = recorded_baseline_status(record, admission, episode_cost)
    else:
        baseline = recomputed_baseline_status(
            frame,
            protocol.battery,
            candidate["visible_update"],
            episode_cost,
        )
    return {
        "scenario_id": record["scenario_id"],
        "event_family": record["event_family"],
        "difficulty": record["difficulty"],
        "state_method": state_method,
        "parser_success": parser_success,
        "state_governance_success": state_success,
        "dispatch_execution_success": dispatch_execution_success,
        "operational_feasibility_success": operational_success,
        "mpc_status": baseline["mpc_status"],
        "pi_status": baseline["pi_status"],
        "baseline_error_code": baseline["baseline_error_code"],
        "constraint_failure_codes": constraint_codes,
        "state_failure_fields": state_failure_fields,
        "state_failure_root_codes": state_root_codes,
        "terminal_soc_error": terminal_error,
        "episode_cost": episode_cost,
        "mpc_cost": baseline["mpc_cost"],
        "pi_cost": baseline["pi_cost"],
        "cost_to_mpc": baseline["cost_to_mpc"],
        "stage1": stage_usage(record["stage1"]),
        "stage2": stage_usage(record["stage2"]),
        "excluded_from_formal_analysis": True,
        "formal_analysis_eligible": False,
    }


def recorded_baseline_status(
    record: dict[str, Any], admission: dict[str, Any], episode_cost: float | None
) -> dict[str, Any]:
    pi_cost = _optional_float(admission.get("perfect_information_cost"))
    pi_status = "feasible" if admission.get("admitted") and pi_cost is not None else "error"
    baseline = record.get("baseline", {})
    mpc_cost = _optional_float(baseline.get("rolling_horizon_mpc_cost"))
    error_code: str | None = None
    if baseline.get("status") != "pass":
        mpc_status = "solver_error"
        message = str(baseline.get("error_message", ""))
        error_code = "MPC_HIGHS_INFEASIBLE" if "infeasible" in message.lower() else "MPC_RECORDED_ERROR"
        mpc_cost = None
    elif baseline.get("rolling_horizon_mpc_feasible") is True:
        mpc_status = "feasible"
    else:
        mpc_status = "simulation_infeasible"
        error_code = "MPC_RECORDED_SIMULATION_INFEASIBLE"
    cost_to_mpc = (
        normalized_regret(episode_cost, mpc_cost)
        if episode_cost is not None and mpc_cost is not None and mpc_status == "feasible"
        else None
    )
    return {
        "pi_status": pi_status,
        "mpc_status": mpc_status,
        "baseline_error_code": error_code,
        "pi_cost": pi_cost,
        "mpc_cost": mpc_cost,
        "cost_to_mpc": cost_to_mpc,
    }


def recomputed_baseline_status(
    frame: pd.DataFrame,
    config: BatteryConfig,
    visible_update: dict[str, Any],
    episode_cost: float | None,
) -> dict[str, Any]:
    schedule = dynamic_override_schedule(visible_update, len(frame))
    codes: list[str] = []
    pi_cost: float | None = None
    mpc_cost: float | None = None
    try:
        pi = perfect_information_baseline(frame, config, overrides_by_step=schedule)
        pi_status = "feasible" if pi.simulation.feasible else "simulation_infeasible"
        pi_cost = float(pi.simulation.total_cost)
        if pi_status != "feasible":
            codes.append("PI_SIMULATION_INFEASIBLE")
    except (BaselineError, ValueError) as exc:
        pi_status = "error"
        codes.append(getattr(exc, "code", "PI_INPUT_OR_RUNTIME_ERROR"))
    try:
        mpc = rolling_horizon_baseline(
            frame,
            config,
            overrides_by_step=schedule,
            lookahead_steps=96,
        )
        mpc_status = "feasible" if mpc.simulation.feasible else "simulation_infeasible"
        mpc_cost = float(mpc.simulation.total_cost)
        if mpc_status != "feasible":
            codes.append("MPC_SIMULATION_INFEASIBLE")
    except (BaselineError, ValueError) as exc:
        mpc_status = "error"
        codes.append(getattr(exc, "code", "MPC_INPUT_OR_RUNTIME_ERROR"))
    cost_to_mpc = (
        normalized_regret(episode_cost, mpc_cost)
        if episode_cost is not None and mpc_cost is not None and mpc_status == "feasible"
        else None
    )
    return {
        "pi_status": pi_status,
        "mpc_status": mpc_status,
        "baseline_error_code": ";".join(codes) if codes else None,
        "pi_cost": pi_cost,
        "mpc_cost": mpc_cost,
        "cost_to_mpc": cost_to_mpc,
    }


def constraint_failure_codes(
    simulation: SimulationResult,
    frame: pd.DataFrame,
    config: BatteryConfig,
    visible_update: dict[str, Any],
) -> list[str]:
    trace = simulation.trace
    codes: set[str] = set()
    if len(trace) == 0:
        return ["EMPTY_EXECUTION_TRACE"]
    requested = trace["requested_action_kw"].to_numpy(dtype=float)
    realized = trace["charge_kw"].to_numpy(dtype=float) - trace["discharge_kw"].to_numpy(dtype=float)
    clipped = np.abs(requested - realized) > TOLERANCE
    if clipped.any():
        codes.add("ACTION_CLIPPED")
    if float(trace["violation_cost"].sum()) > TOLERANCE:
        codes.add("ACTION_VIOLATION_PENALTY")
    schedule = dynamic_override_schedule(visible_update, len(frame))
    max_charge = np.asarray(
        [item.max_charge_kw if item.max_charge_kw is not None else config.max_charge_kw for item in schedule]
    )
    max_discharge = np.asarray(
        [
            item.max_discharge_kw if item.max_discharge_kw is not None else config.max_discharge_kw
            for item in schedule
        ]
    )
    charge = trace["charge_kw"].to_numpy(dtype=float)
    discharge = trace["discharge_kw"].to_numpy(dtype=float)
    positive = requested > TOLERANCE
    negative = requested < -TOLERANCE
    if np.any(positive & (requested > max_charge + TOLERANCE)):
        codes.add("CHARGE_POWER_LIMIT")
    if np.any(positive & (charge + TOLERANCE < np.minimum(requested, max_charge))):
        codes.add("SOC_CAPACITY_LIMIT")
    requested_discharge = -requested
    if np.any(negative & (requested_discharge > max_discharge + TOLERANCE)):
        codes.add("DISCHARGE_POWER_LIMIT")
    if np.any(negative & (discharge + TOLERANCE < np.minimum(requested_discharge, max_discharge))):
        codes.add("SOC_RESERVE_LIMIT")
    if simulation.terminal_soc_gap_kwh > 1e-6:
        codes.add("TERMINAL_SOC_MISMATCH")
    if (trace["soc_kwh"] < trace["reserve_soc_kwh"] - TOLERANCE).any():
        codes.add("SOC_BELOW_RESERVE")
    if (trace["soc_kwh"] > trace["available_capacity_kwh"] + TOLERANCE).any():
        codes.add("SOC_ABOVE_CAPACITY")
    if not np.isfinite(trace.to_numpy(dtype=float)).all():
        codes.add("NONFINITE_EXECUTION_VALUE")
    if not simulation.feasible and not codes:
        codes.add("UNCLASSIFIED_SIMULATOR_INFEASIBILITY")
    return sorted(codes)


def state_failure_root_codes(state: Any, reasons: list[str]) -> list[str]:
    if not reasons:
        return []
    codes: set[str] = set()
    if isinstance(state, dict) and set(state) == {"carrier_type", "value"}:
        codes.add("STATE_CARRIER_WRAPPER_AT_OUTPUT_BOUNDARY")
    if "required_fields_present" in reasons:
        codes.add("STATE_REQUIRED_FIELDS_MISSING")
    if "no_exclusion_as_active_constraint" in reasons:
        codes.add("STATE_EXCLUSION_BOUNDARY_INVALID")
    if any(reason.endswith("_correct") for reason in reasons):
        codes.add("STATE_VALUE_MISMATCH")
    if "no_forbidden_tokens" in reasons:
        codes.add("STATE_FORBIDDEN_TOKEN")
    if not codes:
        codes.add("STATE_SCHEMA_OR_CONTENT_FAILURE")
    return sorted(codes)


def stage_usage(stage: dict[str, Any]) -> dict[str, Any]:
    response = stage.get("response") or {}
    usage = response.get("usage") or {}
    return {
        "status": response.get("status"),
        "returned_model": response.get("returned_model"),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
        "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", 0)),
        "latency_seconds": _optional_float(response.get("elapsed_seconds")),
    }


def gx_f_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for governance, feasibility in ((True, True), (True, False), (False, True), (False, False)):
        matched = [
            str(row["scenario_id"])
            for row in rows
            if row["state_governance_success"] is governance
            and row["operational_feasibility_success"] is feasibility
        ]
        table.append(
            {
                "G_state_governance_success": governance,
                "F_operational_feasibility_success": feasibility,
                "count": len(matched),
                "scenario_ids": matched,
            }
        )
    return table


def write_failure_decomposition(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "EXPLORATORY_FAILURE_DECOMPOSITION.json"
    jsonl_path = output_dir / "EXPLORATORY_FAILURE_DECOMPOSITION.jsonl"
    csv_path = output_dir / "EXPLORATORY_FAILURE_DECOMPOSITION.csv"
    md_path = output_dir / "EXPLORATORY_FAILURE_DECOMPOSITION.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(jsonl_path, list(report["episodes"]))
    _write_csv(csv_path, list(report["episodes"]))
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    artifacts = [json_path, jsonl_path, csv_path, md_path]
    manifest = {
        "schema_version": "exploratory_analysis_hash_manifest_v1",
        "created_utc": utc_now(),
        "excluded_from_formal_analysis": True,
        "files": [
            {"path": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        ],
    }
    manifest_path = output_dir / "hash_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _expected_state(
    protocol: FrozenProtocol,
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    stage1_actions: list[float],
) -> dict[str, Any]:
    activation = int(candidate["visible_update"]["activation_step"])
    step_hours = _frame_step_hours(frame.index)
    prefix = simulate_dispatch(
        frame.iloc[:activation],
        np.asarray(stage1_actions, dtype=float)[:activation],
        protocol.battery,
        overrides=BatteryOverrides(),
        require_terminal_soc=False,
        step_hours=step_hours,
    )
    return canonical_typed_state(
        config=protocol.battery,
        current_time=frame.index[activation],
        visible_update=candidate["visible_update"],
        prefix=prefix,
    )


def _frame_step_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Failure decomposition requires at least two timestamps.")
    return float(np.median(index.to_series().diff().dropna().dt.total_seconds()) / 3600.0)


def _load_frame(
    project_root: Path, protocol: FrozenProtocol, candidate: dict[str, Any]
) -> pd.DataFrame:
    source = project_root / str(candidate["source_data"])
    frame = pd.read_parquet(source)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    start = pd.Timestamp(candidate["start_timestamp"]).tz_convert("UTC")
    end = start + pd.Timedelta(hours=int(candidate["horizon_hours"])) - pd.Timedelta(minutes=15)
    window = frame[(frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] <= end)].copy()
    window = window.sort_values("timestamp_utc").set_index("timestamp_utc")
    if len(window) != int(candidate["horizon_hours"]) * 4:
        raise ValueError(f"Incomplete episode frame for {candidate['scenario_id']}: {len(window)} rows")
    return window


def _counter_rows(values: Any) -> list[dict[str, Any]]:
    counts = Counter(str(value) for value in values)
    return [{"code": code, "count": count} for code, count in sorted(counts.items())]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "event_family",
        "difficulty",
        "state_method",
        "parser_success",
        "state_governance_success",
        "dispatch_execution_success",
        "operational_feasibility_success",
        "mpc_status",
        "pi_status",
        "baseline_error_code",
        "constraint_failure_codes",
        "terminal_soc_error",
        "cost_to_mpc",
        "stage1_prompt_tokens",
        "stage1_completion_tokens",
        "stage1_total_tokens",
        "stage1_latency_seconds",
        "stage2_prompt_tokens",
        "stage2_completion_tokens",
        "stage2_total_tokens",
        "stage2_latency_seconds",
        "excluded_from_formal_analysis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "constraint_failure_codes": ";".join(row["constraint_failure_codes"]),
                    "stage1_prompt_tokens": row["stage1"]["prompt_tokens"],
                    "stage1_completion_tokens": row["stage1"]["completion_tokens"],
                    "stage1_total_tokens": row["stage1"]["total_tokens"],
                    "stage1_latency_seconds": row["stage1"]["latency_seconds"],
                    "stage2_prompt_tokens": row["stage2"]["prompt_tokens"],
                    "stage2_completion_tokens": row["stage2"]["completion_tokens"],
                    "stage2_total_tokens": row["stage2"]["total_tokens"],
                    "stage2_latency_seconds": row["stage2"]["latency_seconds"],
                }
            )


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exploratory Failure Decomposition",
        "",
        "This report is permanently excluded from formal analysis.",
        "",
        f"- Source run: `{report['source_run_dir']}`",
        f"- Baseline view: `{report['baseline_view']}`",
        f"- Episodes: {report['episode_count']}",
        "- Formal-analysis eligible: `false`",
        "",
        "## Episode decomposition",
        "",
        "| scenario_id | event | difficulty | state method | parser | G | dispatch | F | PI | MPC | baseline error | constraint failures | terminal SOC error | cost to MPC |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|",
    ]
    for row in report["episodes"]:
        lines.append(
            "| {scenario_id} | {event_family} | {difficulty} | {state_method} | {parser} | {g} | "
            "{dispatch} | {f} | {pi} | {mpc} | {baseline} | {constraints} | {terminal} | {cost} |".format(
                scenario_id=row["scenario_id"],
                event_family=row["event_family"],
                difficulty=row["difficulty"],
                state_method=row["state_method"],
                parser=int(row["parser_success"]),
                g=int(row["state_governance_success"]),
                dispatch=int(row["dispatch_execution_success"]),
                f=int(row["operational_feasibility_success"]),
                pi=row["pi_status"],
                mpc=row["mpc_status"],
                baseline=row["baseline_error_code"] or "",
                constraints=", ".join(row["constraint_failure_codes"]),
                terminal=_format_number(row["terminal_soc_error"]),
                cost=_format_number(row["cost_to_mpc"]),
            )
        )
    lines.extend(["", "## G x F table", "", "| G | F | count |", "|---:|---:|---:|"])
    for cell in report["gx_f_table"]:
        lines.append(
            f"| {int(cell['G_state_governance_success'])} | "
            f"{int(cell['F_operational_feasibility_success'])} | {cell['count']} |"
        )
    lines.extend(_classification_section("State failure fields", report["state_failure_field_classification"]))
    lines.extend(_classification_section("State failure root codes", report["state_failure_root_classification"]))
    lines.extend(_classification_section("Baseline errors", report["baseline_error_classification"]))
    lines.extend(["", "## Stage usage and latency", "", "| scenario_id | S1 tokens | S1 seconds | S2 tokens | S2 seconds |", "|---|---:|---:|---:|---:|"])
    for row in report["episodes"]:
        lines.append(
            f"| {row['scenario_id']} | {row['stage1']['total_tokens']} | "
            f"{_format_number(row['stage1']['latency_seconds'])} | {row['stage2']['total_tokens']} | "
            f"{_format_number(row['stage2']['latency_seconds'])} |"
        )
    return "\n".join(lines) + "\n"


def _classification_section(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", f"## {title}", "", "| code | count |", "|---|---:|"]
    if not rows:
        lines.append("| none | 0 |")
    else:
        lines.extend(f"| {row['code']} | {row['count']} |" for row in rows)
    return lines


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"
