#!/usr/bin/env python3
"""Recompute the F0 formal raw-action analysis from saved row evidence only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from e2b_v2_common import action_distance_kwh, canonical_json, score_branch, sha256

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PLAN_PATH = ROOT / "reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summary(values: list[float]) -> dict[str, float | int | None]:
    clean = finite(values)
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    ordered = sorted(clean)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p25": ordered[(len(ordered) - 1) // 4],
        "p75": ordered[(3 * (len(ordered) - 1)) // 4],
        "min": ordered[0],
        "max": ordered[-1],
    }


def branch_metrics(task: dict[str, Any], branch: str, item: dict[str, Any]) -> dict[str, Any]:
    parsed = item.get("parsed", {})
    response = item.get("response", {})
    actions = [float(value) for value in parsed.get("dense_action_kw", [])] if parsed.get("ok") else []
    saved = item.get("score", {}).get("carrier_consistent_replay", {})
    authoritative = item.get("score", {}).get("authoritative_replay", {})
    replay_match = False
    if parsed.get("ok") is True:
        recomputed = score_branch(task, branch, actions)
        replay_match = canonical_json(recomputed) == canonical_json(item.get("score", {}))
    return {
        "branch": branch,
        "response_status": response.get("status"),
        "parser_ok": parsed.get("ok") is True,
        "horizon": int(parsed.get("horizon", len(actions))),
        "action_length": len(actions),
        "nontrivial": any(abs(value) > 1e-9 for value in actions),
        "exact_zero": all(abs(value) <= 1e-9 for value in actions),
        "action_energy_kwh": float(saved.get("action_energy_kwh", 0.0)) if saved else None,
        "raw_feasible": bool(saved.get("engineering_feasible", False)) if saved else False,
        "strict_feasible": bool(saved.get("strict_feasible", False)) if saved else False,
        "terminal_error_kwh": float(saved.get("terminal_absolute_error_kwh")) if saved else None,
        "terminal_signed_error_kwh": float(saved.get("terminal_signed_error_kwh")) if saved else None,
        "physical_cost_usd": float(saved.get("physical_cost_usd")) if saved else None,
        "violation_intervals": int(saved.get("violation_interval_count", 0)) if saved else None,
        "authoritative_terminal_error_kwh": float(authoritative.get("terminal_absolute_error_kwh")) if authoritative else None,
        "replay_match": replay_match,
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        result[value] = summarize_rows(subset)
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    branches = {branch: [row["branches"][branch] for row in rows] for branch in ("C1", "C2", "S1", "S2")}
    out: dict[str, Any] = {"blocks": len(rows), "branch_rows": len(rows) * 4}
    out["raw_feasible"] = {branch: sum(int(item["raw_feasible"]) for item in values) for branch, values in branches.items()}
    out["nontrivial"] = {branch: sum(int(item["nontrivial"]) for item in values) for branch, values in branches.items()}
    out["exact_zero"] = {branch: sum(int(item["exact_zero"]) for item in values) for branch, values in branches.items()}
    out["parser_ok"] = {branch: sum(int(item["parser_ok"]) for item in values) for branch, values in branches.items()}
    out["terminal_error_kwh"] = {branch: summary([item["terminal_error_kwh"] for item in values]) for branch, values in branches.items()}
    out["action_energy_kwh"] = {branch: summary([item["action_energy_kwh"] for item in values]) for branch, values in branches.items()}
    out["physical_cost_usd"] = {branch: summary([item["physical_cost_usd"] for item in values]) for branch, values in branches.items()}
    out["pair_distances_kwh"] = {
        "C1_S1": summary([row["pair_metrics"]["C1_S1_distance_kwh"] for row in rows]),
        "C1_C2_null": summary([row["pair_metrics"]["C1_C2_distance_kwh"] for row in rows]),
        "S1_S2_null": summary([row["pair_metrics"]["S1_S2_distance_kwh"] for row in rows]),
        "null_adjusted_C1_S1": summary([row["pair_metrics"]["null_adjusted_C1_S1_kwh"] for row in rows]),
    }
    out["first_action_differences"] = {
        "C1_S1_nonzero": sum(int(row["pair_metrics"]["C1_S1_first_action_abs_kw"] > 1e-9) for row in rows),
        "C1_S1_abs_kw": summary([row["pair_metrics"]["C1_S1_first_action_abs_kw"] for row in rows]),
        "C1_C2_abs_kw": summary([row["pair_metrics"]["C1_C2_first_action_abs_kw"] for row in rows]),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default="f0_formal_20260820T_authorized_v1")
    args = parser.parse_args()
    run_root = ROOT / "runs/f0_formal_v1" / args.run_label
    records_dir = run_root / "records"
    out_dir = ROOT / "reviews/f0_formal_v1"
    plan_rows = load_jsonl(PLAN_PATH)
    plan = {str(row["block_id"]): row for row in plan_rows}
    rows: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        plan_row = plan[str(record["block_id"])]
        source_path = PROJECT / str(plan_row["source_record_path"])
        source_record = json.loads(source_path.read_text(encoding="utf-8"))
        task = source_record["task"]
        if sha256(source_path) != str(plan_row["source_record_sha256"]):
            raise SystemExit(f"source hash mismatch: {source_path}")
        task_hash = __import__("hashlib").sha256(canonical_json(task).encode()).hexdigest()
        if task_hash != str(plan_row["task_sha256"]):
            raise SystemExit(f"task hash mismatch: {source_path}")
        branch_values = {branch: branch_metrics(task, branch, record["branches"][branch]) for branch in ("C1", "C2", "S1", "S2")}
        actions = {branch: [float(value) for value in record["branches"][branch]["parsed"].get("dense_action_kw", [])] for branch in branch_values}
        pair = {
            "C1_S1_distance_kwh": action_distance_kwh(actions["C1"], actions["S1"]),
            "C1_C2_distance_kwh": action_distance_kwh(actions["C1"], actions["C2"]),
            "S1_S2_distance_kwh": action_distance_kwh(actions["S1"], actions["S2"]),
            "null_adjusted_C1_S1_kwh": action_distance_kwh(actions["C1"], actions["S1"]) - 0.5 * (action_distance_kwh(actions["C1"], actions["C2"]) + action_distance_kwh(actions["S1"], actions["S2"])),
            "C1_S1_first_action_abs_kw": abs(actions["C1"][0] - actions["S1"][0]) if actions["C1"] and actions["S1"] else None,
            "C1_C2_first_action_abs_kw": abs(actions["C1"][0] - actions["C2"][0]) if actions["C1"] and actions["C2"] else None,
            "C1_S1_action_energy_difference_kwh": abs(branch_values["C1"]["action_energy_kwh"] - branch_values["S1"]["action_energy_kwh"]),
        }
        rows.append({
            "block_id": record["block_id"], "scenario_id": record["scenario_id"], "model_condition": plan_row["model_condition"],
            "event_family": plan_row["event_family"], "source_record_path": plan_row["source_record_path"],
            "source_record_sha256": plan_row["source_record_sha256"], "task_sha256": plan_row["task_sha256"],
            "branches": branch_values, "pair_metrics": pair,
        })
    if len(rows) != 30:
        raise SystemExit(f"expected 30 records, found {len(rows)}")
    row_path = out_dir / "F0_FORMAL_ROW_LEVEL_ANALYSIS.jsonl"
    row_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    analysis = {
        "schema_version": "f0_formal_raw_action_analysis_v1",
        "status": "PASS",
        "evidence_status": "F0_FORMAL_RAW_ACTION_ANALYSIS_ONLY",
        "run_label": args.run_label,
        "analysis_source": {"run_root": str(run_root.relative_to(ROOT)), "plan": str(PLAN_PATH.relative_to(ROOT)), "plan_sha256": sha256(PLAN_PATH), "row_level": str(row_path.relative_to(ROOT)), "run_report": str((run_root / "F0_FORMAL_EXECUTION_REPORT.json").relative_to(ROOT)), "run_report_sha256": sha256(run_root / "F0_FORMAL_EXECUTION_REPORT.json")},
        "denominator": {"blocks": len(rows), "branch_rows": len(rows) * 4, "planned_provider_calls": 120, "provider_calls_performed_during_analysis": 0, "network_attempts_during_analysis": 0},
        "coverage": {"models": sorted({str(row["model_condition"]) for row in rows}), "scenarios": len({str(row["scenario_id"]) for row in rows}), "event_families": sorted({str(row["event_family"]) for row in rows})},
        "overall": summarize_rows(rows),
        "by_model": grouped(rows, "model_condition"),
        "by_event_family": grouped(rows, "event_family"),
        "by_scenario": grouped(rows, "scenario_id"),
        "raw_action_interpretation": {
            "raw_feasibility_definition": "carrier_consistent_replay.engineering_feasible",
            "null_adjusted_formula": "d(C1,S1) - 0.5 * [d(C1,C2) + d(S1,S2)]",
            "distance_units": "kWh action-energy L1 distance at 15-minute resolution",
            "projection_claims_included": False,
            "mismatched_payload_nulls_are_retained": True,
        },
    }
    (out_dir / "F0_FORMAL_ANALYSIS.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# F0 formal raw-action analysis", "", "Status: **PASS**", "", "This report is recomputed from the 30 saved formal records and does not pool F0 with E1, E2, E2b, or F1.", "", f"- Blocks: **{len(rows)}**", f"- Branch rows: **{len(rows) * 4}**", "- Raw-feasibility definition: **carrier-consistent deterministic replay**", "- Provider calls during analysis: **0**", "- Network attempts during analysis: **0**", "", "## Interpretation boundary", "", "The carrier contrast is reported with identical-payload nulls retained. A positive null-adjusted distance is descriptive evidence within this frozen challenge set; it is not a population estimate or a model ranking. Projection, fallback, and post-gate claims are outside this report.", "", "## Headline counts", "", f"- Raw-feasible branch rows: **{analysis['overall']['raw_feasible']}**", f"- Nontrivial branch rows: **{analysis['overall']['nontrivial']}**", f"- C1–S1 distance: **{analysis['overall']['pair_distances_kwh']['C1_S1']}**", f"- C1–C2 null distance: **{analysis['overall']['pair_distances_kwh']['C1_C2_null']}**", f"- Null-adjusted C1–S1 distance: **{analysis['overall']['pair_distances_kwh']['null_adjusted_C1_S1']}**", "", "Machine-readable details are in `F0_FORMAL_ANALYSIS.json` and `F0_FORMAL_ROW_LEVEL_ANALYSIS.jsonl`."]
    (out_dir / "F0_FORMAL_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
