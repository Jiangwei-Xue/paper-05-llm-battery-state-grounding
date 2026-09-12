#!/usr/bin/env python3
"""Regenerate release summary tables from reproduced row-level outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--reproduced-root", type=Path)
    args = parser.parse_args()
    root = args.release.resolve()
    reproduced = args.reproduced_root.resolve() if args.reproduced_root else root / "reproduced_outputs"
    output = reproduced / "tables"
    output.mkdir(parents=True, exist_ok=True)
    feasibility = json.loads(
        (reproduced / "semantics/FEASIBILITY_SEMANTICS.json").read_text(encoding="utf-8")
    )
    sensitivity = pd.read_csv(
        reproduced / "sensitivities/objective_order_sensitivity_summary.csv"
    )
    slash = "\\"
    lines = [
        slash + "begin{tabular}{lrrr}",
        slash + "toprule",
        "Tier & Denominator & Carrier-consistent & Authoritative " + slash + slash,
        slash + "midrule",
    ]
    for tier, row in feasibility["counts"].items():
        label = tier.replace("_", slash + "_")
        lines.append(
            f"{label} & {row['denominator']} & {row['carrier_consistent_raw_feasible']} & "
            f"{row['authoritative_raw_feasible']} " + slash + slash
        )
    lines.extend([slash + "bottomrule", slash + "end{tabular}", ""])
    (output / "table_feasibility_by_replay_state.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    lines = [
        slash + "begin{tabular}{llrr}",
        slash + "toprule",
        "Model & Objective order & Mean cost (USD) & Mean correction (kWh) "
        + slash
        + slash,
        slash + "midrule",
    ]
    for _, row in sensitivity.iterrows():
        model = str(row["model_condition"]).replace("_", slash + "_")
        order = str(row["objective_order"]).replace("_", slash + "_")
        lines.append(
            f"{model} & {order} & {float(row['mean_cost_usd']):.2f} & "
            f"{float(row['mean_distance_kwh']):.2f} " + slash + slash
        )
    lines.extend([slash + "bottomrule", slash + "end{tabular}", ""])
    (output / "table_objective_order_sensitivity.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    combined = (
        (output / "table_feasibility_by_replay_state.tex").read_text(encoding="utf-8")
        + "\n"
        + (output / "table_objective_order_sensitivity.tex").read_text(encoding="utf-8")
    )
    (output / "REGENERATED_SUMMARY_TABLES.tex").write_text(combined, encoding="utf-8")
    print(json.dumps({"status": "PASS", "tables": 3, "provider_calls": 0, "network_attempts": 0}))


if __name__ == "__main__":
    main()
