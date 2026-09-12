#!/usr/bin/env python3
"""Standardize saved Qwen3.7 Plus F1/F0 evidence without provider calls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("segan_reference_recompute_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_gate(module: Any, llm: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    summary = cast(dict[str, Any], module.gate_summaries(llm, baseline))
    summary["tier_breakdown"] = {}
    for tier_value, group in llm.groupby("tier"):
        tier = str(tier_value)
        summary["tier_breakdown"][tier] = {}
        for gate_type in ("battery_only", "expanded_system"):
            summary["tier_breakdown"][tier][gate_type] = {
                field: module.summarize_values(group[field])
                for field in (
                    f"gate_{gate_type}_distance_kwh",
                    f"gate_{gate_type}_cost_usd",
                    f"gate_{gate_type}_regret_usd",
                    f"gate_{gate_type}_curtailment_kwh",
                    f"gate_{gate_type}_solve_seconds",
                )
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_module(args.reference_script.resolve())
    battery = module.load_inputs(project)[0]
    public_manifest_path = project.parents[1] / "manifests/PUBLIC_RECORD_MANIFEST.jsonl"
    public_record_hashes = {
        row["path"]: row["sha256"] for row in load_jsonl(public_manifest_path)
    }

    plan_path = project / "segan_revision_major_v2/reviews/qwen37plus_f1f0_v1/QWEN37PLUS_FORMAL_RUN_PLAN.jsonl"
    plan_rows = load_jsonl(plan_path)
    plan = {row["block_id"]: row for row in plan_rows}
    records_dir = project / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(records_dir.glob("*.json"))]
    tasks: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        scenario_id = row["scenario_id"]
        if scenario_id in tasks:
            continue
        source = project / row["source_record_path"]
        public_relative = f"frozen_inputs/project/{row['source_record_path']}"
        expected_public_hash = public_record_hashes.get(public_relative)
        observed_source_hash = module.sha256_file(source)
        if expected_public_hash is None or observed_source_hash != expected_public_hash:
            raise ValueError(f"source hash mismatch: {source}")
        source_record = json.loads(source.read_text(encoding="utf-8"))
        tasks[scenario_id] = {
            "task": source_record["task"],
            "source_record_path": row["source_record_path"],
            "source_record_sha256": row["source_record_sha256"],
            "source_record_public_sha256": observed_source_hash,
        }

    f1_records = [record for record in records if record["tier"] == "F1"]
    f0_records = [record for record in records if record["tier"] == "F0"]
    f1_plan = {key: value for key, value in plan.items() if value["tier"] == "F1"}
    f0_plan = {key: value for key, value in plan.items() if value["tier"] == "F0"}
    f1_branch, f1_block, replay_f1 = module.block_and_branch_rows("QWEN37_F1", f1_records, f1_plan, tasks, battery)
    f0_branch, f0_block, replay_f0 = module.block_and_branch_rows("QWEN37_F0", f0_records, f0_plan, tasks, battery)
    f1_branch.to_csv(output / "QWEN37_F1_BRANCH_METRICS.csv", index=False)
    f1_block.to_csv(output / "QWEN37_F1_BLOCK_METRICS.csv", index=False)
    f0_branch.to_csv(output / "QWEN37_F0_BRANCH_METRICS.csv", index=False)
    f0_block.to_csv(output / "QWEN37_F0_BLOCK_METRICS.csv", index=False)
    replay = pd.DataFrame([*replay_f1, *replay_f0])
    replay.to_csv(output / "QWEN37_INDEPENDENT_REPLAY_COMPARISON.csv", index=False)

    robust, stratified, correlations, timing = module.build_summary_tables(f1_branch, f1_block, f0_branch, f0_block)
    stratified.to_csv(output / "QWEN37_STRATIFIED_SUMMARY.csv", index=False)
    correlations.to_csv(output / "QWEN37_CORRELATION_SUMMARY.csv", index=False)
    timing.to_csv(output / "QWEN37_TIMING_BY_BRANCH_POSITION.csv", index=False)
    permutation = module.permutation_analysis(f1_block, f1_records)

    gate_llm, gate_baseline, gate_meta = module.gate_analysis(records, plan, tasks, battery, output)
    tier_by_block = {row["block_id"]: row["tier"] for row in plan_rows}
    gate_llm.insert(0, "tier", gate_llm["block_id"].map(tier_by_block))
    gate_llm.to_csv(output / "QWEN37_GATE_BRANCH_RESULTS.csv", index=False)
    gate_baseline.to_csv(output / "QWEN37_GATE_BASELINE_RESULTS.csv", index=False)
    gate_summary = summarize_gate(module, gate_llm, gate_baseline)

    replay_scalar_columns = [str(column) for column in replay if str(column).startswith("abs_diff_")]
    replay_summary = {
        "rows": len(replay),
        "maximum_trace_abs_difference": float(replay["max_trace_abs_diff"].max()),
        "maximum_scalar_abs_difference": float(replay[replay_scalar_columns].max().max()),
        "flag_mismatch_rows": int((~replay["all_flag_matches"]).sum()),
    }
    summary = {
        "schema_version": "qwen37_offline_standardization_v1_7",
        "created_utc": datetime.now(UTC).isoformat(),
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "input_run": "qwen37plus_formal_20260826T",
        "denominator": {
            "F1_blocks": len(f1_records),
            "F1_branches": len(f1_branch),
            "F0_blocks": len(f0_records),
            "F0_branches": len(f0_branch),
        },
        "raw_feasibility_by_replay_state": {
            "F1": {
                "carrier_consistent_raw_feasible": int(
                    f1_branch["carrier_consistent_raw_feasible"].sum()
                ),
                "authoritative_raw_feasible": int(
                    f1_branch["authoritative_raw_feasible"].sum()
                ),
                "denominator": len(f1_branch),
            },
            "F0": {
                "carrier_consistent_raw_feasible": int(
                    f0_branch["carrier_consistent_raw_feasible"].sum()
                ),
                "authoritative_raw_feasible": int(
                    f0_branch["authoritative_raw_feasible"].sum()
                ),
                "denominator": len(f0_branch),
            },
        },
        "robust_and_stratified": robust,
        "random_pairing": permutation,
        "gate": gate_summary,
        "gate_metadata": gate_meta,
        "independent_replay": replay_summary,
        "pooling_boundary": "Separate sensitivity tier; no pooling with historical F1/F0.",
    }
    (output / "QWEN37_OFFLINE_STANDARDIZATION.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    core = robust["f1_core"]["qwen37_plus_direct"]
    gate = gate_summary["llm"]["qwen37_plus_direct"]
    lines = [
        "# Qwen3.7 Plus offline standardization",
        "",
        "All calculations use the 300 saved formal responses. Provider calls: **0**; network attempts: **0**.",
        "",
        "- F1 carrier-consistent raw feasibility: "
        f"**{summary['raw_feasibility_by_replay_state']['F1']['carrier_consistent_raw_feasible']}/{len(f1_branch)}**.",
        "- F1 authoritative raw feasibility (safety/deployment endpoint): "
        f"**{summary['raw_feasibility_by_replay_state']['F1']['authoritative_raw_feasible']}/{len(f1_branch)}**.",
        "- F0 carrier-consistent raw feasibility: "
        f"**{summary['raw_feasibility_by_replay_state']['F0']['carrier_consistent_raw_feasible']}/{len(f0_branch)}**.",
        "- F0 authoritative raw feasibility (safety/deployment endpoint): "
        f"**{summary['raw_feasibility_by_replay_state']['F0']['authoritative_raw_feasible']}/{len(f0_branch)}**.",
        f"- F1 carrier ED: mean **{core['ED_kwh']['mean']:.2f} kWh**, median **{core['ED_kwh']['median']:.2f} kWh**.",
        f"- Exact-terminal battery-only correction: mean **{gate['battery_only']['gate_battery_only_distance_kwh']['mean']:.2f} kWh**.",
        f"- Exact-terminal expanded-system correction: mean **{gate['expanded_system']['gate_expanded_system_distance_kwh']['mean']:.2f} kWh**.",
        f"- Replay agreement: {replay_summary['rows']} rows; maximum trace difference {replay_summary['maximum_trace_abs_difference']:.3e}.",
        "",
        "This is a separate sensitivity tier and is not pooled with the historical confirmatory F1/F0 conditions.",
    ]
    (output / "QWEN37_OFFLINE_STANDARDIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
