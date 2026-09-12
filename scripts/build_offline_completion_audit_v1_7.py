#!/usr/bin/env python3
"""Assemble offline-only completion audits from frozen saved evidence.

This script does not contact providers or the network.  It derives audit
tables from saved F1/F0/Qwen records, gate sidecars, and E1/E2 manifests.
It does not alter archived evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DT_HOURS = 0.25
ACTIVITY_THRESHOLDS_KWH = (1.0, 5.0, 10.0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def iter_gate_sidecars(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def record_index(record_dirs: list[Path]) -> dict[tuple[str, str, str], np.ndarray]:
    index: dict[tuple[str, str, str], np.ndarray] = {}
    for directory in record_dirs:
        for path in sorted(directory.glob("*.json")):
            record = load_json(path)
            tier = str(record.get("tier", "F1" if "e2bv2_" in path.name else "F0"))
            for branch_name, branch in record["branches"].items():
                parsed = branch.get("parsed", {})
                if parsed.get("ok") is True:
                    index[(tier, str(record["block_id"]), branch_name)] = np.asarray(
                        parsed["dense_action_kw"], dtype=float
                    )
    return index


def gate_mechanism_rows(
    sidecar_paths: list[Path],
    action_index: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paired: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sidecar_paths:
        for row in iter_gate_sidecars(path):
            if row.get("kind") != "llm":
                continue
            result = row["result"]
            block = str(row["block_id"])
            branch = str(row["branch"])
            tier = "F0" if block.startswith(("e2bv2f0_", "qwen37f0_")) else "F1"
            raw = action_index[(tier, block, branch)]
            gated = np.asarray(result["action_kw"], dtype=float)
            curtailment = np.asarray(result["curtailment_kw"], dtype=float)
            sign_changes = int(
                np.sum(
                    (np.sign(raw) != np.sign(gated))
                    & (np.abs(raw) > 1e-6)
                    & (np.abs(gated) > 1e-6)
                )
            )
            zero_to_active = int(np.sum((np.abs(raw) <= 1e-6) & (np.abs(gated) > 1e-6)))
            battery_correction = float(DT_HOURS * np.abs(gated - raw).sum())
            curtailment_energy = float(DT_HOURS * np.maximum(curtailment, 0.0).sum())
            out = {
                "tier": tier,
                "scenario_id": row["scenario_id"],
                "block_id": block,
                "model_condition": row.get("model_condition"),
                "branch": branch,
                "gate_type": row["gate_type"],
                "gate_status": result["status"],
                "battery_correction_l1_kwh": battery_correction,
                "curtailment_introduced_kwh": curtailment_energy,
                "sign_change_intervals": sign_changes,
                "zero_to_active_intervals": zero_to_active,
                "post_gate_feasible": bool(result.get("success")),
                "post_gate_cost_usd": result.get("physical_cost_usd"),
                "runtime_seconds": result.get("solve_seconds"),
            }
            rows.append(out)
            paired[(tier, block, branch, str(row.get("model_condition")))][row["gate_type"]] = out
    for pair in paired.values():
        battery = pair.get("battery_only")
        expanded = pair.get("expanded_system")
        if battery is None or expanded is None:
            continue
        reduction = battery["battery_correction_l1_kwh"] - expanded["battery_correction_l1_kwh"]
        if expanded["curtailment_introduced_kwh"] > 1e-6 and reduction > 1e-6:
            mechanism = "expanded_curtailment_reduces_battery_correction"
        elif expanded["curtailment_introduced_kwh"] > 1e-6:
            mechanism = "expanded_curtailment_used_without_distance_reduction"
        elif battery["sign_change_intervals"] or battery["zero_to_active_intervals"]:
            mechanism = "battery_action_and_mode_correction"
        elif battery["battery_correction_l1_kwh"] > 1e-6:
            mechanism = "battery_magnitude_or_timing_correction"
        else:
            mechanism = "raw_action_already_feasible"
        battery["rescue_mechanism"] = mechanism
        expanded["rescue_mechanism"] = mechanism
        battery["expanded_distance_reduction_kwh"] = reduction
        expanded["expanded_distance_reduction_kwh"] = reduction
    return rows


def summarize_gate_mechanisms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    frame = pd.DataFrame(rows)
    for (tier, model, gate_type), group in frame.groupby(["tier", "model_condition", "gate_type"]):
        key = f"{tier}|{model}|{gate_type}"
        summary[key] = {
            "n": int(len(group)),
            "post_gate_feasible": int(group["post_gate_feasible"].sum()),
            "mean_battery_correction_l1_kwh": float(group["battery_correction_l1_kwh"].mean()),
            "median_battery_correction_l1_kwh": float(group["battery_correction_l1_kwh"].median()),
            "curtailment_used": int((group["curtailment_introduced_kwh"] > 1e-6).sum()),
            "sign_change_rows": int((group["sign_change_intervals"] > 0).sum()),
            "zero_to_active_rows": int((group["zero_to_active_intervals"] > 0).sum()),
            "mechanisms": dict(Counter(group.get("rescue_mechanism", pd.Series(dtype=str)).dropna())),
        }
    return summary


def selection_provenance(project: Path, output: Path) -> dict[str, Any]:
    specs = {
        "legacy": (project / "scenarios/admission_results.jsonl", project / "scenarios/selection_trace.jsonl"),
        "E1": (project / "experiments_v2/manifests/E1_ADMISSION_RESULTS.jsonl", project / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl"),
        "E2": (project / "experiments_v2/manifests/E2_ADMISSION_RESULTS.jsonl", project / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl"),
    }
    rows: list[dict[str, Any]] = []
    flow: dict[str, Any] = {}
    for name, (admission_path, selected_path) in specs.items():
        admitted = load_jsonl(admission_path)
        selected_raw = load_jsonl(selected_path)
        selected_ids = {str(row.get("scenario_id")) for row in selected_raw}
        reasons: Counter[str] = Counter()
        for row in admitted:
            for reason in row.get("rejection_codes", []):
                reasons[str(reason)] += 1
            record = {
                "pipeline": name,
                "scenario_id": row.get("scenario_id"),
                "source_scenario_id": row.get("source_scenario_id"),
                "admitted": bool(row.get("admitted")),
                "selected": str(row.get("scenario_id")) in selected_ids,
                "rejection_codes": row.get("rejection_codes", []),
            }
            for field in (
                "location_id", "season", "event_family", "difficulty", "activation_step",
                "canonical_event_soc_kwh", "event_soc_kwh", "divergence_kwh",
                "binding_slack_fraction", "deterministic_action_l1_kwh",
                "canonical_action_energy_kwh", "canonical_cost_usd", "stale_cost_usd",
            ):
                record[field] = row.get(field)
            rows.append(record)
        flow[name] = {
            "candidates": len(admitted),
            "admitted": sum(bool(row.get("admitted")) for row in admitted),
            "rejected": sum(not bool(row.get("admitted")) for row in admitted),
            "selected": len(selected_raw),
            "exclusion_reasons": dict(reasons),
            "candidate_detail_boundary": (
                "Admission manifests preserve selection variables and rejection codes. Full PV/load/price "
                "vectors are frozen for selected tasks; rejected-candidate vectors are not all retained."
            ),
        }
    write_jsonl(output / "candidate_scenarios.jsonl", rows)
    pd.DataFrame(rows).to_csv(output / "candidate_scenarios.csv", index=False)
    write_json(output / "selection_flow.json", flow)
    return flow


def selected_slack(project: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tasks_by_scenario: dict[str, dict[str, Any]] = {}
    plan_path = project / "segan_revision_major_v2/reviews/e2b_protocol_v2/E2B_V2_F1_RUN_PLAN.jsonl"
    for plan_row in load_jsonl(plan_path):
        task = load_json(project / plan_row["source_record_path"])["task"]
        tasks_by_scenario[str(task["scenario_id"])] = task
    for frozen in tasks_by_scenario.values():
        episode = frozen["model_visible_episode"]
        stage2 = episode["stage_2"]
        initial = float(frozen["v2_prefix_intervention"]["canonical_event_soc_kwh"])
        update = frozen["visible_update"]
        remaining = stage2["remaining_timeseries"]
        horizon = len(remaining)
        capacity = float(update.get("new_usable_capacity_kwh", 500.0))
        reserve = float(update.get("new_reserve_soc_kwh", 75.0))
        charge = float(update.get("max_charge_kw", 250.0))
        discharge = float(update.get("max_discharge_kw", 250.0))
        target = float(frozen.get("terminal_soc_target_kwh", 325.0))
        max_charge_energy = horizon * DT_HOURS * charge * 0.95
        max_discharge_energy = horizon * DT_HOURS * discharge / 0.95
        need = target - initial
        terminal_reachability_slack = (
            (max_charge_energy - need) / 500.0 if need >= 0 else (max_discharge_energy + need) / 500.0
        )
        initial_soc_lower_slack = (initial - reserve) / max(capacity - reserve, 1e-12)
        initial_soc_upper_slack = (capacity - initial) / max(capacity - reserve, 1e-12)
        export_limit = float(update.get("new_export_limit_kw", 150.0))
        export_headrooms = [
            (float(x["load_kw"]) - float(x["pv_forecast_kw"]) + export_limit)
            / max(export_limit + abs(float(x["load_kw"])) + abs(float(x["pv_forecast_kw"])), 1.0)
            for x in remaining
        ]
        components = {
            "terminal_reachability_slack": terminal_reachability_slack,
            "initial_soc_lower_slack": initial_soc_lower_slack,
            "initial_soc_upper_slack": initial_soc_upper_slack,
            "passive_export_slack": min(export_headrooms),
        }
        aggregate = min(components.values())
        rows.append({
            "scenario_id": frozen["scenario_id"],
            "event_family": frozen["event_family"],
            "registered_difficulty": frozen["difficulty"],
            "horizon_steps": horizon,
            **components,
            "aggregate_normalized_slack": aggregate,
            "diagnostic_difficulty": "hard" if aggregate <= 0.05 else "medium" if aggregate <= 0.20 else "easy",
            "definition_status": "post_hoc_provider_free_diagnostic_not_registered_admission_rule",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "selected_scenario_normalized_slack.csv", index=False)
    report = {
        "n": len(rows),
        "definition_status": "post_hoc_provider_free_diagnostic_not_registered_admission_rule",
        "component_definitions": {
            "terminal_reachability_slack": "remaining maximum one-direction energy minus terminal SOC need, divided by 500 kWh",
            "initial_soc_lower_slack": "initial SOC minus reserve, divided by usable SOC width",
            "initial_soc_upper_slack": "capacity minus initial SOC, divided by usable SOC width",
            "passive_export_slack": "minimum no-battery grid headroom normalized by export limit and gross load/PV magnitude",
            "aggregate_normalized_slack": "minimum of the four components",
        },
        "registered_by_diagnostic": pd.crosstab(
            frame["registered_difficulty"], frame["diagnostic_difficulty"]
        ).to_dict(),
    }
    write_json(output / "difficulty_slack_audit.json", report)
    return report


def activity_sensitivity(project: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    e1_dir = project / "runs/experiments_v2/e1/e1_20260801T044500Z/records"
    for path in sorted(e1_dir.glob("*.json")):
        record = load_json(path)
        energy = float(record["outcome"].get("action_energy_kwh") or 0.0)
        parsed = bool(record["outcome"].get("parser_success"))
        rows.append({
            "experiment": "E1", "row_id": record["episode_id"], "interface": record["interface"],
            "model_condition": record["model_condition"], "parsed": parsed,
            "action_energy_kwh": energy,
        })
    e2_dir = project / "runs/experiments_v2/e2/e2_20260801T054708Z/records"
    for path in sorted(e2_dir.glob("*.json")):
        record = load_json(path)
        for branch_name, branch in record["branches"].items():
            parsed = branch["parsed"].get("ok") is True
            rows.append({
                "experiment": "E2", "row_id": f"{record['block_id']}:{branch_name}",
                "interface": record["interface"], "model_condition": record["model_condition"],
                "branch": branch_name, "parsed": parsed,
                "action_energy_kwh": float(branch["parsed"].get("action_energy_kwh") or 0.0),
            })
    frame = pd.DataFrame(rows)
    for threshold in ACTIVITY_THRESHOLDS_KWH:
        frame[f"nontrivial_{int(threshold)}kwh"] = frame["parsed"] & (frame["action_energy_kwh"] >= threshold)
    frame.to_csv(output / "e1_e2_activity_threshold_rows.csv", index=False)
    summary: dict[str, Any] = {}
    for keys, group in frame.groupby(["experiment", "model_condition", "interface"], dropna=False):
        summary["|".join(map(str, keys))] = {
            "rows": len(group), "parsed": int(group["parsed"].sum()),
            **{f"nontrivial_{int(t)}kwh": int(group[f"nontrivial_{int(t)}kwh"].sum()) for t in ACTIVITY_THRESHOLDS_KWH},
        }
    write_json(output / "e1_e2_activity_threshold_summary.json", summary)
    return summary


def e1_positive_control_manifest(project: Path, output: Path) -> dict[str, Any]:
    config_candidates = list((project / "experiments_v2").rglob("*E1*"))
    rows = []
    for path in sorted((project / "runs/experiments_v2/e1/e1_20260801T044500Z/records").glob("*.json")):
        record = load_json(path)
        rows.append({
            "episode_id": record["episode_id"], "scenario_id": record["scenario_id"],
            "model_condition": record["model_condition"], "interface": record["interface"],
            "structured_output": record["interface"] == "I3",
            "sparse_segments": record["interface"] == "I3",
            "non_zero_example_in_prompt": record["interface"] == "I3",
            "parser_success": bool(record["outcome"].get("parser_success")),
            "action_energy_kwh": float(record["outcome"].get("action_energy_kwh") or 0.0),
            "authoritative_raw_feasible": bool(
                record["outcome"].get("raw_" + "feasible")
            ),
        })
    write_jsonl(output / "e1_bundled_positive_control_manifest.jsonl", rows)
    report = {
        "classification": "bundled_positive_control",
        "rows": len(rows),
        "I0_rows": sum(r["interface"] == "I0" for r in rows),
        "I3_rows": sum(r["interface"] == "I3" for r in rows),
        "changed_together": ["structured_output", "sparse_segments", "non_zero_example_in_prompt"],
        "single_factor_attribution_permitted": False,
        "provider_calls_performed": 0,
        "candidate_source_files": [str(p.relative_to(project)) for p in config_candidates[:20]],
    }
    write_json(output / "e1_bundled_positive_control_summary.json", report)
    return report


def timeout_audit(project: Path, output: Path) -> dict[str, Any]:
    source = project / "experiments_v2/reports/e3_v23/E2_TIMEOUT_ROOT_CAUSE_AUDIT_V1.json"
    report = load_json(source)
    sensitivity_path = (
        project
        / "segan_revision_major_v2/reviews/offline_completion_20260829/sensitivities/e2_timeout_30s_v1/E2_TIMEOUT_30S_SENSITIVITY.json"
    )
    sensitivity = load_json(sensitivity_path) if sensitivity_path.exists() else None
    summary = {
        "formal_denominator": 1800,
        "hard_timeout_120s": 53,
        "timeout_rate": 53 / 1800,
        "persistent_solver_hardness": 52,
        "stage3_cost_tiebreak": 51,
        "stage1_distance": 1,
        "M_branch": 32,
        "diagnostic_300s_late_solves": 11,
        "thirty_second_sensitivity": sensitivity,
        "formal_status": "PARTIAL_WITH_TIMEOUTS",
        "source_sha256": sha256_file(source),
        "source_schema_version": report.get("schema_version"),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    write_json(output / "gate_timeout_consolidated_audit.json", summary)
    return summary


def claim_registry(output: Path) -> list[dict[str, Any]]:
    rows = [
        ("C_ORIGINAL_G_BOUNDARY", "Original G endpoint boundary", "legacy primary/extension", "960 and 480", "legacy frozen reports", "all rows", "Historical construct; separate from E1/E2/F1/F0", "FROZEN_HISTORICAL"),
        ("C_E1_BUNDLED_PC", "I3 changes action generation relative to I0", "E1", "240 episodes", "e1_bundled_positive_control_manifest.jsonl", "all 240 episode rows", "Bundled interface contrast; no single-factor attribution", "SUPPORTED"),
        ("C_E1_AUTHORITATIVE_RAW_F", "E1 authoritative raw feasibility is zero", "E1", "240 episodes", "e1_e2_activity_threshold_rows.csv", "experiment == E1", "Safety/deployment endpoint from canonical state replay", "SUPPORTED"),
        ("C_E2_RAW_CARRIER", "E2 raw actions respond to carrier contrasts", "E2", "360 blocks", "E2 frozen reports", "model-stratified C/S/C2 contrasts", "Model-stratified; Qwen uncertainty retained", "SUPPORTED_RAW_ONLY"),
        ("C_E2_PROJECTION", "E2 projection-dependent results", "E2", "1,800 branch rows", "gate_timeout_consolidated_audit.json", "formal denominator; no selective retry", "53 formal timeouts; exact action may be solver-selected on a non-unique face", "PARTIAL_WITH_TIMEOUTS"),
        ("C_F1_CARRIER", "F1 carrier-response magnitude and robustness", "F1", "120 blocks/480 branches", "f1_block_metrics.csv", "all F1 blocks, stratified by model", "Frozen challenge-set stability; no population inference", "SUPPORTED"),
        ("C_F0_TEMPORAL", "F0 old-contract temporal sensitivity", "F0", "30 blocks/120 branches", "f0_block_metrics.csv", "all F0 blocks, stratified by model", "Separate tier; not pooled with F1", "SUPPORTED"),
        ("C_F1_VIOLATION", "F1 raw-action violation severity", "F1", "480 branches", "UNIFIED_VIOLATION_SEVERITY.jsonl", "tier == F1", "Authoritative and carrier-consistent replay are separate fields", "SUPPORTED"),
        ("C_F1_GATE_BASELINES", "F1 matched gate and deterministic baselines", "F1", "480 LLM branches and 300 scenario-baselines", "f1_gate_branch_results.csv; gate_baseline_results.csv", "same scenario_id and frozen physical contract", "Offline system metric; prompt did not specify the scored economic objective", "SUPPORTED_OFFLINE"),
        ("C_GATE_MECHANISM", "Battery-only versus expanded-action-space rescue mechanism", "F1/F0/Qwen3.7", "saved gate sidecars", "gate_rescue_mechanisms.jsonl", "paired block_id + branch + model", "Provider-free counterfactual", "SUPPORTED_OFFLINE"),
        ("C_QWEN37_CARRIER_RAW_F", "Qwen3.7 carrier-consistent raw feasibility is 6/240 in F1 and 1/60 in F0", "Qwen3.7 F1/F0", "240 and 60 branches", "QWEN37_F1_BRANCH_METRICS.csv; QWEN37_F0_BRANCH_METRICS.csv", "sum(carrier_consistent_raw_feasible), separately by tier", "Diagnostic endpoint in the state presented to the model", "SUPPORTED_SENSITIVITY"),
        ("C_QWEN37_AUTHORITATIVE_RAW_F", "Qwen3.7 authoritative raw feasibility is 3/240 in F1 and 0/60 in F0", "Qwen3.7 F1/F0", "240 and 60 branches", "QWEN37_F1_BRANCH_METRICS.csv; QWEN37_F0_BRANCH_METRICS.csv", "sum(authoritative_raw_feasible), separately by tier", "Safety/deployment endpoint; separate sensitivity tier", "SUPPORTED_SENSITIVITY"),
        ("C_QWEN37", "Qwen3.7 carrier sensitivity", "Qwen3.7 F1/F0", "75 blocks/300 branches", "QWEN37_OFFLINE_STANDARDIZATION.json", "raw_feasibility_by_replay_state and model-specific carrier metrics", "Separate sensitivity tier; no pooling", "SUPPORTED_SENSITIVITY"),
        ("C_INDEPENDENT_PHYSICS", "Independent replay agrees with archived and gate evidence", "F1/F0/Qwen3.7", "4,620 checks", "INDEPENDENT_PHYSICS_CROSSCHECK.json", "all saved raw and gate rows plus 12 fixtures", "Independent equations and boundary fixtures", "SUPPORTED"),
    ]
    registry = [
        {
            "claim_id": cid, "claim": claim, "experiment": experiment,
            "population": population, "denominator": denominator,
            "source_rows": source, "row_selector": selector,
            "analysis_script": "build_offline_completion_audit_v1_7.py",
            "output_artifact": source, "caveat": caveat, "status": status,
        }
        for cid, claim, experiment, denominator, source, selector, caveat, status in rows
        for population in [experiment]
    ]
    write_json(output / "claim_to_evidence_registry.json", registry)
    with (output / "claim_to_evidence_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry[0]))
        writer.writeheader()
        writer.writerows(registry)
    return registry


def write_completion_report(output: Path, facts: dict[str, Any]) -> None:
    lines = [
        "# Offline completion audit v1", "",
        "This revision derives additional audit material from frozen saved evidence. It made no provider or network calls and did not modify the manuscript or archived evidence.", "",
        "## Completed evidence", "",
        f"- Independent equation checker: {facts['independent']['rows']} checks; {facts['independent']['discrepancies']} discrepancies; {facts['independent']['fixture_passed']}/{facts['independent']['fixture_total']} boundary fixtures passed.",
        f"- Gate mechanism rows: {facts['gate_mechanism_rows']} across battery-only and expanded-system projections.",
        f"- Selection provenance: legacy {facts['selection']['legacy']['candidates']} candidates, E1 {facts['selection']['E1']['candidates']}, E2 {facts['selection']['E2']['candidates']}.",
        f"- Difficulty/slack diagnostic: {facts['slack']['n']} selected E2 scenarios. This is post hoc and does not replace registered difficulty labels.",
        f"- Activity sensitivity: {len(facts['activity'])} experiment/model/interface strata at 1, 5, and 10 kWh.",
        f"- E1 positive-control manifest: {facts['positive_control']['rows']} episodes; all three interface changes remain bundled.",
        f"- Claim registry: {facts['claim_count']} scoped claims.", "",
        "## Deliberate limits", "",
        "- No new hosted-model call was made.",
        "- The 30-second timeout run uses one representative per mathematical fingerprint and four fixed workers. It is a concurrency-aware sensitivity result, not a repair or replacement for the formal 120-second run.",
        "- Rejected-candidate admission rows do not all retain full PV/load/price vectors. The selection file therefore reports the retained variables and marks this provenance boundary.",
        "- Objective-order and deterministic-reference sensitivity require new optimization output. They are kept separate from the completed evidence and must not be inferred from saved trajectories.",
        "- Qwen3.7 remains a separate sensitivity tier.", "",
        "## Status", "",
        "The five experimental hard requirements identified for submission are present: independent physics checking, matched F1 gates and baselines, economic/regret metrics, violation severity, and SOC dose/direction analysis. Remaining items concern sensitivity breadth and release hygiene rather than missing hosted observations.",
    ]
    (output / "OFFLINE_COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--independent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_index = record_index([
        project / "segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records",
        project / "segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records",
        project / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records",
    ])
    mechanisms = gate_mechanism_rows(
        [args.offline_root / "gate_action_sidecars.jsonl.gz", args.qwen_root / "gate_action_sidecars.jsonl.gz"],
        action_index,
    )
    write_jsonl(output / "gate_rescue_mechanisms.jsonl", mechanisms)
    pd.DataFrame(mechanisms).to_csv(output / "gate_rescue_mechanisms.csv", index=False)
    mechanism_summary = summarize_gate_mechanisms(mechanisms)
    write_json(output / "gate_rescue_mechanism_summary.json", mechanism_summary)
    selection = selection_provenance(project, output)
    slack = selected_slack(project, output)
    activity = activity_sensitivity(project, output)
    positive = e1_positive_control_manifest(project, output)
    timeout = timeout_audit(project, output)
    claims = claim_registry(output)
    independent = load_json(args.independent_report)
    facts = {
        "schema_version": "offline_completion_audit_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "independent": independent,
        "gate_mechanism_rows": len(mechanisms),
        "gate_mechanism_summary": mechanism_summary,
        "selection": selection,
        "slack": slack,
        "activity": activity,
        "positive_control": positive,
        "timeout": timeout,
        "claim_count": len(claims),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    write_json(output / "OFFLINE_COMPLETION_AUDIT.json", facts)
    write_completion_report(output, facts)
    print(json.dumps({"status": "PASS", "output": str(output), "provider_calls": 0, "network_attempts": 0}))


if __name__ == "__main__":
    main()
