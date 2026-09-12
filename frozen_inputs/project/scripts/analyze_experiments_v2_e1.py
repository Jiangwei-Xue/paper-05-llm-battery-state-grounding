#!/usr/bin/env python3
"""Regenerate E1 interface results from saved row evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs/experiments_v2/e1" / args.run_label
    records = [json.loads(path.read_text()) for path in sorted((run_dir / "records").glob("*.json"))]
    if len(records) != 240:
        raise SystemExit(f"E1 analysis requires 240 records, found {len(records)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[str(record["interface"])].append(record)
        grouped[f"{record['model_condition']}|{record['interface']}"] .append(record)
        key = (str(record["scenario_id"]), str(record["model_condition"]), int(record["repetition"]))
        paired[key][str(record["interface"])] = record
    pair_rows = [_pair(key, cell) for key, cell in sorted(paired.items())]
    report = {
        "schema_version": "experiments_v2_e1_analysis_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "E1_ANALYZED",
        "denominator_logical_episodes": len(records),
        "independent_scenario_clusters": len({record["scenario_id"] for record in records}),
        "failure_as_zero": True,
        "interface_results": [_summary(key, grouped[key]) for key in ("I0", "I3")],
        "model_interface_results": [
            _summary(key, grouped[key])
            for key in sorted(grouped)
            if "|" in key
        ],
        "paired_blocks": len(pair_rows),
        "paired_contrasts_I3_minus_I0": {
            endpoint: sum(row[endpoint] for row in pair_rows) / len(pair_rows)
            for endpoint in ("nontrivial_action", "raw_feasible", "parser_success")
        },
        "discordant_pair_counts": {
            endpoint: sum(row[endpoint] != 0 for row in pair_rows)
            for endpoint in ("nontrivial_action", "raw_feasible", "parser_success")
        },
        "provider_calls_performed_by_analysis": 0,
        "network_attempts_by_analysis": 0,
    }
    _write(run_dir / "E1_ANALYSIS.json", report)
    _write(ROOT / "experiments_v2/reports/E1_ANALYSIS.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _summary(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(records)
    return {
        "cell": name,
        "episodes": denominator,
        "parser_success": _rate(records, "parser_success"),
        "nontrivial_action": _rate(records, "nontrivial_action"),
        "raw_feasible": _rate(records, "raw_feasible"),
        "exact_zero": _rate(records, "exact_zero"),
    }


def _rate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    numerator = sum(bool(record["outcome"].get(field, False)) for record in records)
    return {"numerator": numerator, "denominator": len(records), "rate": numerator / len(records)}


def _pair(key: tuple[str, str, int], cell: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(cell) != {"I0", "I3"}:
        raise SystemExit(f"Incomplete E1 pair: {key}")
    return {
        "scenario_id": key[0],
        "model_condition": key[1],
        "repetition": key[2],
        **{
            endpoint: int(bool(cell["I3"]["outcome"].get(endpoint, False)))
            - int(bool(cell["I0"]["outcome"].get(endpoint, False)))
            for endpoint in ("nontrivial_action", "raw_feasible", "parser_success")
        },
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    main()
