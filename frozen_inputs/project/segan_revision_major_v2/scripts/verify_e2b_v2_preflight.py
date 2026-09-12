#!/usr/bin/env python3
"""Verify the complete provider-free E2b-v2 preflight and frozen artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "reviews/e2b_protocol_v2"
MANIFEST = OUT / "E2B_V2_PRE_CALL_MANIFEST.json"

sys.path.insert(0, str(ROOT / "scripts"))

from analyze_e2b_v2 import analyze  # noqa: E402
from e2b_v2_common import canonical_json, sha256  # noqa: E402

BRANCHES = ("C1", "C2", "S1", "S2")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def synthetic_record(scenario: str, *, between: bool, parser_failure: bool = False) -> dict[str, Any]:
    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        action = [10.0, 10.0] if between and branch.startswith("S") else [0.0, 0.0]
        ok = not (parser_failure and branch == "C1")
        branches[branch] = {
            "parsed": {"ok": ok, "dense_action_kw": action if ok else [], "diagnostics": [] if ok else ["invalid_json"]},
            "score": {"authoritative_replay": {"terminal_absolute_error_kwh": abs(sum(action))}} if ok else None,
        }
    return {
        "scenario_id": scenario,
        "model_condition": "deepseek_formal",
        "branches": branches,
        "same_information_mpc": {"action_kw": [0.0, 0.0]},
    }


def verify() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["manifest_execution_hold"] = manifest["status"] == "OFFLINE_GATES_PASS_EXECUTION_HOLD"
    checks["no_provider_calls"] = manifest["provider_calls_performed"] == 0
    checks["no_network_attempts"] = manifest["network_attempts"] == 0
    checks["explicit_live_authorization_absent"] = manifest["explicit_user_authorization_for_live_smoke"] is False
    artifact_results: dict[str, bool] = {}
    for label, item in manifest["artifacts"].items():
        path = PROJECT / item["path"]
        artifact_results[label] = path.is_file() and sha256(path) == item["sha256"]
    checks["all_frozen_hashes_match"] = all(artifact_results.values())

    f1 = load_jsonl(PROJECT / manifest["artifacts"]["F1_run_plan"]["path"])
    smoke = load_jsonl(PROJECT / manifest["artifacts"]["smoke_run_plan"]["path"])
    f0 = load_jsonl(PROJECT / manifest["artifacts"]["F0_optional_plan"]["path"])
    checks["F1_120_unique_blocks_480_calls"] = len(f1) == len({row["block_id"] for row in f1}) == 120 and sum(row["planned_provider_calls"] for row in f1) == 480
    checks["smoke_4_unique_blocks_16_calls"] = len(smoke) == len({row["block_id"] for row in smoke}) == 4 and sum(row["planned_provider_calls"] for row in smoke) == 16
    checks["F0_30_unique_blocks_120_calls"] = len(f0) == len({row["block_id"] for row in f0}) == 30 and sum(row["planned_provider_calls"] for row in f0) == 120
    checks["duplicate_request_hash_identity"] = all(
        row["expected_request_sha256"]["C1"] == row["expected_request_sha256"]["C2"]
        and row["expected_request_sha256"]["S1"] == row["expected_request_sha256"]["S2"]
        for row in f1
    )
    checks["balanced_branch_positions"] = all(
        all(
            sum(row["branch_order"][position] == branch for row in f1 if row["model_condition"] == model) == 15
            for position in range(4)
            for branch in BRANCHES
        )
        for model in ("deepseek_formal", "qwen_flash")
    )
    controls = json.loads((OUT / "E2B_V2_PROVIDER_FREE_CONTROL_REPORT.json").read_text(encoding="utf-8"))
    checks["provider_free_controls_pass"] = controls["status"] == "pass" and all(value == 60 for value in controls["counts"].values())
    old_inventory = load_jsonl(OUT / "E2B_V1_SOURCE_HASH_MANIFEST.jsonl")
    checks["old_evidence_immutable"] = all(
        (ROOT / row["path"]).is_file() and sha256(ROOT / row["path"]) == row["sha256"]
        for row in old_inventory
    )
    fixtures = json.loads((OUT / "E2B_V2_RENDERED_PROMPT_FIXTURES.json").read_text(encoding="utf-8"))
    checks["fixtures_have_no_global_t_rows"] = all(
        "t" not in row
        for key in ("canonical_contract", "stale_contract")
        for row in fixtures[key]["timeseries"]
    )
    synthetic = analyze(
        [
            synthetic_record("identical", between=False),
            synthetic_record("between", between=True),
            synthetic_record("parser_failure", between=False, parser_failure=True),
        ],
        bootstrap_resamples=100,
    )
    model = synthetic["model_reports"]["deepseek_formal"]
    checks["synthetic_analysis_complete_case_gate"] = model["planned_blocks"] == 3 and model["complete_blocks"] == 2
    checks["synthetic_analysis_parser_failure_not_zero_action"] = model["parsed_branches"] == 11
    report = {
        "schema_version": "e2b_v2_preflight_verification_v1",
        "status": "pass" if all(checks.values()) else "HOLD",
        "checks": checks,
        "artifact_hash_checks": artifact_results,
        "synthetic_analysis": synthetic,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "live_smoke_authorized": False,
        "next_gate": "separate explicit authorization for isolated 16-call live smoke",
    }
    return report


def main() -> None:
    report = verify()
    path = OUT / "E2B_V2_PREFLIGHT_VERIFICATION_REPORT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"status": report["status"], "provider_calls_performed": 0, "network_attempts": 0}))
    if report["status"] != "pass":
        raise SystemExit("E2b-v2 preflight verification HOLD")


if __name__ == "__main__":
    main()
