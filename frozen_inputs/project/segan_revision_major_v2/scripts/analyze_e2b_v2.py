#!/usr/bin/env python3
"""Locked E2b-v2 analysis over saved Stage-2 branch records."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from e2b_v2_common import action_distance_kwh, canonical_json  # noqa: E402

BRANCHES = ("C1", "C2", "S1", "S2")
DEFAULT_BOOTSTRAP_SEED = 20260716
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
COMPLETE_BLOCK_GATE = 0.80


def _branch_action(record: dict[str, Any], branch: str) -> list[float] | None:
    value = record.get("branches", {}).get(branch, {})
    parsed = value.get("parsed", {})
    if parsed.get("ok") is not True:
        return None
    action = parsed.get("dense_action_kw")
    if not isinstance(action, list):
        return None
    return [float(item) for item in action]


def block_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    actions = {branch: _branch_action(record, branch) for branch in BRANCHES}
    if any(actions[branch] is None for branch in BRANCHES):
        return None
    dense = {branch: actions[branch] for branch in BRANCHES}
    lengths = {len(value) for value in dense.values() if value is not None}
    if len(lengths) != 1:
        raise ValueError("complete block has unequal horizons")
    c1, c2 = dense["C1"], dense["C2"]
    s1, s2 = dense["S1"], dense["S2"]
    assert c1 is not None and c2 is not None and s1 is not None and s2 is not None
    cross = [
        action_distance_kwh(c1, s1),
        action_distance_kwh(c1, s2),
        action_distance_kwh(c2, s1),
        action_distance_kwh(c2, s2),
    ]
    between = statistics.mean(cross)
    within_c = action_distance_kwh(c1, c2)
    within_s = action_distance_kwh(s1, s2)
    horizon = len(c1)

    def terminal_error(branch: str) -> float:
        score = record["branches"][branch]["score"]["authoritative_replay"]
        return float(score["terminal_absolute_error_kwh"])

    def mpc_distance(branch: str) -> float:
        mpc = [float(value) for value in record["same_information_mpc"]["action_kw"]]
        action = dense[branch]
        assert action is not None
        return action_distance_kwh(action, mpc)

    return {
        "scenario_id": str(record["scenario_id"]),
        "model_condition": str(record["model_condition"]),
        "horizon": horizon,
        "between_carrier_B_kwh": between,
        "within_canonical_Wc_kwh": within_c,
        "within_stale_Ws_kwh": within_s,
        "energy_distance_ED_kwh": 2.0 * between - within_c - within_s,
        "normalized_between_difference_kw": between / (horizon * 0.25),
        "terminal_alignment_kwh": statistics.mean([terminal_error("S1"), terminal_error("S2")])
        - statistics.mean([terminal_error("C1"), terminal_error("C2")]),
        "mpc_alignment_kwh": statistics.mean([mpc_distance("S1"), mpc_distance("S2")])
        - statistics.mean([mpc_distance("C1"), mpc_distance("C2")]),
    }


def _bootstrap(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int | list[float]]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if len(values) == 0:
        return {"n_scenarios": 0, "mean": float("nan"), "median": float("nan"), "ci95": []}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "n_scenarios": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "positive": int((values > 1e-12).sum()),
        "zero": int((np.abs(values) <= 1e-12).sum()),
        "negative": int((values < -1e-12).sum()),
    }


def analyze(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    parse_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    complete_by_model: Counter[str] = Counter()
    planned_by_model: Counter[str] = Counter()
    metrics_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_codes: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        model = str(record["model_condition"])
        planned_by_model[model] += 1
        for branch in BRANCHES:
            parsed = record.get("branches", {}).get(branch, {}).get("parsed", {})
            parse_by_model[model]["planned"] += 1
            if parsed.get("ok") is True:
                parse_by_model[model]["parsed"] += 1
            else:
                for code in parsed.get("diagnostics", ["missing_branch"]):
                    failure_codes[model][str(code)] += 1
        metrics = block_metrics(record)
        if metrics is not None:
            complete_by_model[model] += 1
            metrics_by_model[model].append(metrics)
    model_reports: dict[str, Any] = {}
    for model in sorted(planned_by_model):
        blocks = planned_by_model[model]
        complete = complete_by_model[model]
        rate = complete / blocks if blocks else 0.0
        rows = metrics_by_model[model]
        model_reports[model] = {
            "planned_blocks": blocks,
            "planned_branches": parse_by_model[model]["planned"],
            "parsed_branches": parse_by_model[model]["parsed"],
            "complete_blocks": complete,
            "complete_block_rate": rate,
            "responsiveness_status": "confirmatory" if rate >= COMPLETE_BLOCK_GATE else "exploratory_interface_gate_failed",
            "parser_failure_codes": dict(failure_codes[model]),
            "metrics": {
                field: _bootstrap(
                    rows,
                    field,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
                for field in (
                    "between_carrier_B_kwh",
                    "within_canonical_Wc_kwh",
                    "within_stale_Ws_kwh",
                    "energy_distance_ED_kwh",
                    "normalized_between_difference_kw",
                    "terminal_alignment_kwh",
                    "mpc_alignment_kwh",
                )
            },
        }
    return {
        "schema_version": "e2b_v2_analysis_v1",
        "analysis_population": "F1",
        "records": len(records),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "complete_block_gate": COMPLETE_BLOCK_GATE,
        "model_reports": model_reports,
        "pooled_results_are_descriptive_only": True,
        "parser_failures_encoded_as_zero_action": False,
        "provider_calls_performed_by_analysis": 0,
        "network_attempts_by_analysis": 0,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(load_records(args.records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"status": "complete", "records": report["records"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
