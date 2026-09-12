#!/usr/bin/env python3
"""Regenerate E2 paired carrier results from saved block evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("C", "M", "S", "K", "C2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs/experiments_v2/e2" / args.run_label
    records = [
        json.loads(path.read_text()) for path in sorted((run_dir / "records").glob("*.json"))
    ]
    if len(records) != 360:
        raise SystemExit(f"E2 analysis requires 360 blocks, found {len(records)}")
    sidecars = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "projection_sidecars").glob("*.json"))
    ]
    if len(sidecars) != 1800:
        raise SystemExit(f"E2 analysis requires 1800 projection sidecars, found {len(sidecars)}")
    rows = [_paired_row(record) for record in records]
    scenario_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_model[(row["scenario_id"], row["model_condition"])].append(row)
    reduced = [_reduce_cell(key, values) for key, values in sorted(scenario_model.items())]
    scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reduced:
        scenario[row["scenario_id"]].append(row)
    cluster_rows = [_reduce_scenario(key, values) for key, values in sorted(scenario.items())]
    report: dict[str, Any] = {
        "schema_version": "experiments_v2_e2_analysis_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "E2_ANALYZED",
        "denominator_blocks": len(records),
        "independent_scenario_clusters": len(cluster_rows),
        "provider_repetitions_per_scenario_model": 3,
        "failure_as_zero": True,
        "branch_results": [_branch_summary(branch, records) for branch in BRANCHES],
        "model_branch_results": [
            _branch_summary(
                branch, [record for record in records if record["model_condition"] == model], model
            )
            for model in ("deepseek_formal", "qwen_flash")
            for branch in BRANCHES
        ],
        "co_primary": {
            "scenario_median_delta_action_sensitivity_kwh": _estimate(cluster_rows, "delta_action"),
            "canonical_minus_stale_raw_feasibility": _estimate(cluster_rows, "feasibility_delta"),
            "stale_minus_canonical_terminal_error_kwh": _estimate(
                cluster_rows, "terminal_error_delta"
            ),
        },
        "co_primary_by_model": {
            model: {
                "scenario_median_delta_action_sensitivity_kwh": _estimate(
                    [row for row in reduced if row["model_condition"] == model],
                    "delta_action",
                ),
                "canonical_vs_c2_action_l1_kwh": _estimate(
                    [row for row in reduced if row["model_condition"] == model],
                    "null_action",
                ),
                "delta_action_minus_null_kwh": _estimate(
                    [row for row in reduced if row["model_condition"] == model],
                    "delta_action_net",
                ),
                "canonical_minus_stale_raw_feasibility": _estimate(
                    [row for row in reduced if row["model_condition"] == model],
                    "feasibility_delta",
                ),
                "stale_minus_canonical_terminal_error_kwh": _estimate(
                    [row for row in reduced if row["model_condition"] == model],
                    "terminal_error_delta",
                ),
            }
            for model in ("deepseek_formal", "qwen_flash")
        },
        "paired_null": {
            "canonical_vs_c2_action_l1_kwh": _estimate(cluster_rows, "null_action"),
            "delta_action_minus_null_kwh": _estimate(cluster_rows, "delta_action"),
        },
        "scenario_model_reduced_rows": reduced,
        "scenario_cluster_rows": cluster_rows,
        "provider_cost": _cost(records),
        "projection_results": [_projection_summary(branch, sidecars) for branch in BRANCHES],
        "paired_parseability": {
            "C_and_S": sum(
                record["outcomes"]["C"]["parser_success"]
                and record["outcomes"]["S"]["parser_success"]
                for record in records
            ),
            "C_and_C2": sum(
                record["outcomes"]["C"]["parser_success"]
                and record["outcomes"]["C2"]["parser_success"]
                for record in records
            ),
            "denominator_blocks": len(records),
        },
        "provider_calls_performed_by_analysis": 0,
        "network_attempts_by_analysis": 0,
    }
    report["paired_null"]["delta_action_minus_null_kwh"] = _estimate(
        cluster_rows, "delta_action_net"
    )
    _write(run_dir / "E2_ANALYSIS.json", report)
    _write(ROOT / "experiments_v2/reports/E2_ANALYSIS.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "blocks": len(records),
                "co_primary": report["co_primary"],
                "paired_null": report["paired_null"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _paired_row(record: dict[str, Any]) -> dict[str, Any]:
    outcomes = record["outcomes"]
    return {
        "block_id": record["block_id"],
        "scenario_id": record["scenario_id"],
        "model_condition": record["model_condition"],
        "repetition": record["repetition"],
        "delta_action": _action_l1(record, "C", "S"),
        "null_action": _action_l1(record, "C", "C2"),
        "feasibility_delta": int(outcomes["C"]["raw_feasible"])
        - int(outcomes["S"]["raw_feasible"]),
        "terminal_error_delta": _terminal(outcomes["S"]) - _terminal(outcomes["C"]),
        "c_vs_m_action": _action_l1(record, "C", "M"),
        "c_vs_k_action": _action_l1(record, "C", "K"),
    }


def _action_l1(record: dict[str, Any], left: str, right: str) -> float:
    a = record["branches"][left]["parsed"]
    b = record["branches"][right]["parsed"]
    if not a["ok"] or not b["ok"]:
        return 0.0
    av = np.asarray(a["dense_action_kw"], dtype=float)
    bv = np.asarray(b["dense_action_kw"], dtype=float)
    return float(0.25 * np.abs(av - bv).sum())


def _terminal(outcome: dict[str, Any]) -> float:
    value = outcome.get("terminal_soc_error_kwh")
    return float(value) if value is not None else 250.0


def _reduce_cell(key: tuple[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3:
        raise SystemExit(f"E2 cell does not have three repetitions: {key}")
    fields = (
        "delta_action",
        "null_action",
        "feasibility_delta",
        "terminal_error_delta",
        "c_vs_m_action",
        "c_vs_k_action",
    )
    result: dict[str, Any] = {
        "scenario_id": key[0],
        "model_condition": key[1],
        "repetitions": 3,
    }
    result.update({field: float(np.median([row[field] for row in rows])) for field in fields})
    result["delta_action_net"] = float(result["delta_action"]) - float(result["null_action"])
    return result


def _reduce_scenario(scenario_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise SystemExit(f"E2 scenario lacks two model conditions: {scenario_id}")
    fields = (
        "delta_action",
        "null_action",
        "delta_action_net",
        "feasibility_delta",
        "terminal_error_delta",
        "c_vs_m_action",
        "c_vs_k_action",
    )
    return {
        "scenario_id": scenario_id,
        **{field: float(np.mean([row[field] for row in rows])) for field in fields},
    }


def _estimate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows])
    rng = np.random.default_rng(20260716)
    boot = np.asarray(
        [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(10_000)]
    )
    return {
        "estimate": float(np.mean(values)),
        "interval_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "scenario_clusters": len(values),
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20260716,
    }


def _branch_summary(
    branch: str, records: list[dict[str, Any]], model: str | None = None
) -> dict[str, Any]:
    denominator = len(records)
    return {
        "branch": branch,
        "model_condition": model,
        "blocks": denominator,
        **{
            field: {
                "numerator": sum(
                    bool(record["outcomes"][branch].get(field, False)) for record in records
                ),
                "denominator": denominator,
            }
            for field in ("parser_success", "nontrivial_action", "raw_feasible", "exact_zero")
        },
    }


def _cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, float | int | str]] = {}
    for model in ("deepseek_formal", "qwen_flash"):
        stages = [record["stage1"] for record in records if record["model_condition"] == model]
        stages.extend(
            stage
            for record in records
            if record["model_condition"] == model
            for stage in record["branches"].values()
        )
        valid = [
            stage["estimated_provider_cost"]
            for stage in stages
            if stage["response"].get("status") != "skipped"
        ]
        result[model] = {
            "amount": float(sum(float(value.get("amount", 0.0)) for value in valid)),
            "currency": str(
                next((value.get("currency") for value in valid if value.get("currency")), "unknown")
            ),
            "stage_calls": len(valid),
        }
    return result


def _projection_summary(branch: str, sidecars: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in sidecars if row["branch"] == branch]
    computed = [row for row in selected if row["status"] == "computed"]
    distances = [float(row["projection_distance_kwh"]) for row in computed]
    return {
        "branch": branch,
        "sidecars": len(selected),
        "computed": len(computed),
        "not_computed": len(selected) - len(computed),
        "projection_distance_mean_kwh": float(np.mean(distances)) if distances else None,
        "projection_distance_median_kwh": float(np.median(distances)) if distances else None,
        "simultaneous_intervals_total": sum(int(row["simultaneous_intervals"]) for row in computed),
        "sidecars_above_internal_overlap_tolerance": sum(
            int(row["simultaneous_intervals"]) > 0 for row in computed
        ),
        "maximum_numerical_overlap_kw": max(
            (float(row["maximum_simultaneous_kw"]) for row in computed), default=0.0
        ),
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


if __name__ == "__main__":
    main()
