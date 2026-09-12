#!/usr/bin/env python3
"""Create the release-level feasibility semantics and endpoint counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    required = {
        "carrier_consistent_raw_feasible",
        "authoritative_raw_feasible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing feasibility fields in {path}: {sorted(missing)}")
    return {
        "denominator": int(len(frame)),
        "carrier_consistent_raw_feasible": int(
            frame["carrier_consistent_raw_feasible"].astype(bool).sum()
        ),
        "authoritative_raw_feasible": int(
            frame["authoritative_raw_feasible"].astype(bool).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "feasibility_semantics_v1_7",
        "definitions": {
            "carrier_consistent_raw_feasible": (
                "Raw parsed action replayed from the state carried in the model-facing branch. "
                "This is a diagnostic of whether the action is feasible in the state presented to the model."
            ),
            "authoritative_raw_feasible": (
                "The same raw parsed action replayed from canonical system state. "
                "This is the safety/deployment-facing primary feasibility endpoint."
            ),
        },
        "terminal_tolerance_kwh": 1.0,
        "counts": {
            "historical_F1": counts(args.primary / "f1_branch_metrics.csv"),
            "historical_F0": counts(args.primary / "f0_branch_metrics.csv"),
            "qwen37_F1": counts(args.qwen / "QWEN37_F1_BRANCH_METRICS.csv"),
            "qwen37_F0": counts(args.qwen / "QWEN37_F0_BRANCH_METRICS.csv"),
        },
        "naming_rule": (
            "Release artifacts must not use an unqualified raw-feasibility field. "
            "Every field and prose statement must identify the replay state."
        ),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    (output / "FEASIBILITY_SEMANTICS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Raw-action feasibility semantics",
        "",
        "The release reports two replay endpoints and never uses an unqualified raw-feasibility field.",
        "",
        "- `carrier_consistent_raw_feasible`: replay starts from the state carried in the model-facing branch. It diagnoses the action in the world represented to the model.",
        "- `authoritative_raw_feasible`: replay starts from canonical system state. It is the safety/deployment-facing primary endpoint.",
        "",
        "Both endpoints use the engineering terminal tolerance of 1 kWh. Exact-terminal gates are separate optimization diagnostics and are not substituted for either raw-action endpoint.",
        "",
        "| Tier | Denominator | Carrier-consistent | Authoritative |",
        "|---|---:|---:|---:|",
    ]
    for tier, row in report["counts"].items():
        lines.append(
            f"| {tier} | {row['denominator']} | "
            f"{row['carrier_consistent_raw_feasible']} | "
            f"{row['authoritative_raw_feasible']} |"
        )
    (output / "FEASIBILITY_SEMANTICS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **report["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
