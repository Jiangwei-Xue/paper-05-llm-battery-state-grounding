#!/usr/bin/env python3
"""Adversarial acceptance audit for the standalone v1.7 evidence release."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def check(name: str, passed: bool, evidence: Any, boundary: str | None = None) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "boundary": boundary,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_json(path: Path) -> int:
    return sum(1 for item in path.glob("*.json") if item.is_file())


def forbidden_feasibility_tokens(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    pattern = re.compile(r"(?<!carrier_consistent_)(?<!authoritative_)raw_feasible(?:_count)?")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl", ".md", ".tex"}:
            continue
        if "expected_outputs" in path.parts or "frozen_inputs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            findings.append({"path": str(path.relative_to(root)), "token": match.group(0)})
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--reproduced-root", type=Path)
    args = parser.parse_args()
    root = args.release.resolve()
    workspace = args.project.resolve() if args.project else root / "frozen_inputs/project"
    outputs = args.reproduced_root.resolve() if args.reproduced_root else root / "reproduced_outputs"
    primary = outputs / "primary"
    audit = outputs / "audit"
    semantics = outputs / "semantics"
    checks: list[dict[str, Any]] = []

    record_counts = {
        "F1": count_json(workspace / "segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records"),
        "F0": count_json(workspace / "segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records"),
        "Qwen37": count_json(workspace / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records"),
        "E1": count_json(workspace / "runs/experiments_v2/e1/e1_20260801T044500Z/records"),
        "E2": count_json(workspace / "runs/experiments_v2/e2/e2_20260801T054708Z/records"),
    }
    checks.append(check("bottom-level denominators", record_counts == {"F1": 120, "F0": 30, "Qwen37": 75, "E1": 240, "E2": 360}, record_counts))

    registry = read_json(audit / "claim_to_evidence_registry.json")
    required_claim_fields = {"claim_id", "denominator", "source_rows", "row_selector", "status"}
    traceable = bool(registry) and all(required_claim_fields <= set(row) for row in registry)
    checks.append(check("claim registry row traceability", traceable, {"claims": len(registry), "required_fields": sorted(required_claim_fields)}))

    semantics_report = read_json(semantics / "FEASIBILITY_SEMANTICS.json")
    q1 = semantics_report["counts"]["qwen37_F1"]
    q0 = semantics_report["counts"]["qwen37_F0"]
    semantic_counts_ok = (
        q1["carrier_consistent_raw_feasible"] == 6
        and q1["authoritative_raw_feasible"] == 3
        and q1["denominator"] == 240
        and q0["carrier_consistent_raw_feasible"] == 1
        and q0["authoritative_raw_feasible"] == 0
        and q0["denominator"] == 60
    )
    forbidden = forbidden_feasibility_tokens(outputs)
    checks.append(check("qualified raw-action feasibility", semantic_counts_ok and not forbidden, {"qwen37_F1": q1, "qwen37_F0": q0, "forbidden_tokens": forbidden}))

    terminal_ok = semantics_report["terminal_tolerance_kwh"] == 1.0
    gate = pd.read_csv(primary / "f1_gate_branch_results.csv")
    gate_exact_columns = [column for column in gate if column.startswith("gate_") and "distance" in column]
    checks.append(check("terminal endpoint separation", terminal_ok and bool(gate_exact_columns), {"raw_engineering_tolerance_kwh": 1.0, "gate_contract": "exact terminal equality", "gate_columns": gate_exact_columns[:8]}))

    baseline = pd.read_csv(primary / "gate_baseline_results.csv")
    cost_columns = ["gate_battery_only_cost_usd", "gate_battery_only_regret_usd"]
    cost_complete = all(baseline[column].notna().all() for column in cost_columns)
    checks.append(check("cost and regret denominator", cost_complete and len(baseline) == 300, {"rows": len(baseline), "missing": {column: int(baseline[column].isna().sum()) for column in cost_columns}}))

    direct = baseline[baseline["baseline"] == "direct_mpc"]
    direct_max_regret = float(np.max(np.abs(direct["gate_battery_only_regret_usd"]))) if len(direct) else float("inf")
    checks.append(check("direct MPC regret identity", len(direct) == 60 and direct_max_regret <= 1e-6, {"rows": len(direct), "maximum_absolute_regret_usd": direct_max_regret}))

    scenario_counts = Counter(baseline["scenario_id"])
    baseline_labels = sorted(str(value) for value in baseline["baseline"].unique())
    matched = len(scenario_counts) == 60 and set(scenario_counts.values()) == {5}
    checks.append(check("matched gate baselines", matched, {"scenarios": len(scenario_counts), "baselines": baseline_labels, "rows_per_scenario": sorted(set(scenario_counts.values()))}))

    pooled_columns = [column for column in gate if "pooled" in column.lower()]
    model_counts = gate.groupby("model_condition").size().to_dict()
    checks.append(check("no F1 F0 Qwen pooling", not pooled_columns and set(model_counts.values()) == {240}, {"F1_model_rows": model_counts, "pooled_columns": pooled_columns}, "Cross-tier summaries remain prohibited."))

    blocks = pd.read_csv(primary / "f1_block_metrics.csv")
    dose_event = pd.crosstab(blocks["divergence_kwh"], blocks["event_family"])
    dose_horizon = blocks.groupby("divergence_kwh")["horizon_intervals"].agg(["min", "max", "mean"]).reset_index()
    confounded = bool((dose_event == 0).any().any()) or len(dose_horizon["mean"].round(6).unique()) > 1
    checks.append(check("dose-response confounding disclosed", True, {"dose_by_event": dose_event.to_dict(), "dose_by_horizon": dose_horizon.to_dict("records"), "confounding_present": confounded}, "Dose results are descriptive strata, not an isolated causal dose effect."))

    robustness = read_json(primary / "OFFLINE_RECOMPUTE_SUMMARY.json")["analysis"]["f1_core"]
    reversal: dict[str, Any] = {}
    for model, report in robustness.items():
        ed = report["ED_kwh"]
        reversal[model] = {
            "mean": ed["mean"],
            "median": ed["median"],
            "trimmed_mean_10pct": ed["trimmed_mean_10pct"],
            "mean_drop_largest_5": ed["mean_drop_largest_5"],
            "sparse_support": {"positive": ed["positive"], "zero": ed["zero"], "negative": ed["negative"]},
        }
    checks.append(check("outlier robustness disclosed", True, reversal, "Means are not interpreted as broad response when medians or trimmed means collapse."))

    timeout = read_json(audit / "gate_timeout_consolidated_audit.json")
    timeout_ok = timeout["formal_denominator"] == 1800 and timeout["hard_timeout_120s"] == 53 and timeout["formal_status"] == "PARTIAL_WITH_TIMEOUTS"
    checks.append(check("timeouts retained in denominator", timeout_ok, timeout, "Archived E2 projection evidence is excluded from complete-rate claims."))

    table_path = outputs / "tables/REGENERATED_SUMMARY_TABLES.tex"
    table_text = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
    table_ok = bool(table_text.strip()) and "raw_feasible" not in table_text
    checks.append(check("tables rebuilt from reproduced data", table_ok, {"path": str(table_path.relative_to(outputs)), "bytes": len(table_text.encode())}))

    failures = [row for row in checks if row["status"] == "FAIL"]
    report = {
        "schema_version": "offline_evidence_release_v1_7_acceptance",
        "status": "PASS_WITH_BOUNDARIES" if not failures else "FAIL",
        "checks": checks,
        "failed_checks": [row["check"] for row in failures],
        "experimental_materials_freeze_ready": not failures,
        "historical_boundary": "Rejected candidates do not all retain complete PV/load/price vectors; selection-rule counterfactuals are limited to retained evidence.",
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    output = outputs / "acceptance"
    output.mkdir(parents=True, exist_ok=True)
    (output / "V1_7_RELEASE_ACCEPTANCE.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# v1.7 release acceptance", "", f"Status: **{report['status']}**.", "", "| Check | Status |", "|---|---|"]
    lines.extend(f"| {row['check']} | {row['status']} |" for row in checks)
    lines.extend(["", "The acceptance status freezes the saved experimental material and offline analysis boundary. It does not convert historical E2 projection timeouts into complete evidence, pool model tiers, or recover missing rejected-candidate time series."])
    (output / "V1_7_RELEASE_ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks), "failed": len(failures), "provider_calls": 0, "network_attempts": 0}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
