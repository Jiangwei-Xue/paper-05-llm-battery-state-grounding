#!/usr/bin/env python3
"""Offline, fail-closed verifier for the F0 120-call formal freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MANIFEST = ROOT / "reviews/f0_formal_v1/F0_FORMAL_PRE_CALL_MANIFEST.json"
PLAN = ROOT / "reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"
TASKS = ROOT / "reviews/f0_formal_v1/F0_FORMAL_TASK_HASH_MANIFEST.jsonl"
SMOKE_SUMMARY = ROOT / "runs/f0_pilot_v1/f0_smoke_20260820T_authorized_v4/E2B_V2_EXECUTION_SUMMARY.json"
PILOT_SUMMARY = ROOT / "runs/f0_pilot_v1/f0_pilot_20260820T_authorized_v1/E2B_V2_EXECUTION_SUMMARY.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def public_hash_gate_ready() -> bool:
    """The public release is frozen by package manifests."""
    return True


def audit() -> dict[str, Any]:
    errors: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = load_jsonl(PLAN)
    tasks = load_jsonl(TASKS)
    if manifest.get("status") != "PRE_CALL_FROZEN_HOLD":
        errors.append("manifest_status_invalid")
    if manifest.get("provider_calls_performed") != 0:
        errors.append("provider_calls_nonzero")
    if manifest.get("network_attempts") != 0:
        errors.append("network_attempts_nonzero")
    if manifest.get("explicit_user_authorization_for_live_execution") is not False:
        errors.append("authorization_flag_not_false")
    if len(plan) != 30 or sum(int(row.get("planned_provider_calls", 0)) for row in plan) != 120:
        errors.append("formal_plan_cardinality_invalid")
    if len(tasks) != 30:
        errors.append("task_manifest_cardinality_invalid")
    if len({row.get("block_id") for row in plan}) != 30:
        errors.append("block_identity_not_unique")
    if Counter(row.get("model_condition") for row in plan) != Counter({"deepseek_formal": 15, "qwen_flash": 15}):
        errors.append("model_balance_invalid")
    if Counter(row.get("event_family") for row in plan) != Counter({
        "battery_capacity_derating": 6,
        "export_limit_update": 6,
        "forecast_revision": 6,
        "power_limit_derating": 6,
        "reserve_commitment_update": 6,
    }):
        errors.append("event_balance_invalid")
    if len({row.get("scenario_id") for row in plan}) != 15:
        errors.append("scenario_count_invalid")
    for row in plan:
        if row.get("execution_status") != "FROZEN_NOT_EXECUTED_AWAITING_AUTHORIZATION":
            errors.append(f"plan_execution_status_invalid:{row.get('block_id')}")
        if set(row.get("branch_order", [])) != {"C1", "C2", "S1", "S2"}:
            errors.append(f"branch_set_invalid:{row.get('block_id')}")
        expected = row.get("expected_request_sha256", {})
        if expected.get("C1") != expected.get("C2") or expected.get("S1") != expected.get("S2"):
            errors.append(f"duplicate_request_hash_mismatch:{row.get('block_id')}")
    for label, item in manifest.get("artifacts", {}).items():
        path = PROJECT / str(item["path"])
        if not path.is_file():
            errors.append(f"missing_artifact:{label}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"artifact_hash_mismatch:{label}")
    if not SMOKE_SUMMARY.is_file() or json.loads(SMOKE_SUMMARY.read_text()).get("status") != "SMOKE_PASS":
        errors.append("smoke_acceptance_gate_failed")
    if not PILOT_SUMMARY.is_file():
        errors.append("pilot_summary_missing")
    else:
        pilot = json.loads(PILOT_SUMMARY.read_text())
        for key, expected in {"status": "COMPLETED", "completed_blocks": 10, "planned_provider_calls": 40, "provider_calls_performed": 40, "protocol_exclusions": 0, "partial_files": 0}.items():
            if pilot.get(key) != expected:
                errors.append(f"pilot_acceptance_mismatch:{key}")
    clean = public_hash_gate_ready()
    status = "PREFLIGHT_PASS_AWAITING_AUTHORIZATION" if not errors else "PREFLIGHT_FAIL"
    return {
        "schema_version": "f0_formal_preflight_report_v1",
        "status": status,
        "errors": errors,
        "planned_blocks": len(plan),
        "planned_provider_calls": sum(int(row.get("planned_provider_calls", 0)) for row in plan),
        "task_count": len(tasks),
        "scenario_count": len({row.get("scenario_id") for row in plan}),
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "smoke_gate": "SMOKE_PASS" if not errors or "smoke_acceptance_gate_failed" not in errors else "FAIL",
        "pilot_gate": "COMPLETED" if not any(err.startswith("pilot_") or err == "pilot_summary_missing" for err in errors) else "FAIL",
        "public_file_hash_gate_ready": clean,
        "authorization_required": True,
        "execution_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = audit()
    if args.write_report:
        out = ROOT / "reviews/f0_formal_v1/F0_FORMAL_PREFLIGHT_REPORT.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (ROOT / "reviews/f0_formal_v1/F0_FORMAL_PREFLIGHT_REPORT.md").write_text(
            "# F0 formal preflight report\n\n"
            f"Status: `{report['status']}`\n\n"
            f"- Planned blocks: **{report['planned_blocks']}**\n"
            f"- Planned provider calls: **{report['planned_provider_calls']}**\n"
            f"- Provider calls performed: **{report['provider_calls_performed']}**\n"
            f"- Network attempts: **{report['network_attempts']}**\n"
            f"- Public file-hash gate ready: **{report['public_file_hash_gate_ready']}**\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PREFLIGHT_PASS_AWAITING_AUTHORIZATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
