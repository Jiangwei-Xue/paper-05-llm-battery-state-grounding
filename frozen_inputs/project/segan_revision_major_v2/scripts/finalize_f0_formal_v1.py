#!/usr/bin/env python3
"""Finalize F0 formal output with an offline integrity report and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PLAN = ROOT / "reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    run_root = ROOT / "runs/f0_formal_v1" / args.run_label
    records_dir = run_root / "records"
    summary_path = run_root / "E2B_V2_EXECUTION_SUMMARY.json"
    if not summary_path.is_file() or not records_dir.is_dir():
        raise SystemExit("formal run output is incomplete or missing")
    plan = {row["block_id"]: row for row in load_jsonl(PLAN)}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(records_dir.glob("*.json"))]
    errors: list[str] = []
    response_status: Counter[str] = Counter()
    parse_ok = 0
    returned_model = 0
    branch_rows = 0
    request_hash_mismatches = 0
    duplicate_failures = 0
    timestamp_failures = 0
    protocol_exclusions = 0
    for record in records:
        block_id = str(record.get("block_id"))
        if block_id not in plan:
            errors.append(f"unexpected_block:{block_id}")
            continue
        expected = plan[block_id]["expected_request_sha256"]
        protocol_exclusions += len(record.get("protocol_exclusions", []))
        requests: dict[str, str] = {}
        for branch, item in record.get("branches", {}).items():
            branch_rows += 1
            response = item.get("response", {})
            requests[branch] = str(response.get("request_sha256"))
            response_status[str(response.get("status"))] += 1
            if response.get("request_sha256") != expected.get(branch):
                request_hash_mismatches += 1
            if response.get("returned_model"):
                returned_model += 1
            if item.get("parsed", {}).get("ok") is True:
                parse_ok += 1
            attempts = response.get("attempts", [])
            if not attempts or any(not attempt.get("request_sent_utc") or not attempt.get("response_received_utc") for attempt in attempts):
                timestamp_failures += 1
        if requests.get("C1") != requests.get("C2") or requests.get("S1") != requests.get("S2"):
            duplicate_failures += 1
    if len(records) != 30:
        errors.append(f"record_count:{len(records)}")
    if branch_rows != 120:
        errors.append(f"branch_count:{branch_rows}")
    if request_hash_mismatches or duplicate_failures or timestamp_failures or protocol_exclusions:
        errors.append("integrity_checks_failed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = "PASS" if not errors and summary.get("status") == "COMPLETED" else "HOLD"
    report = {
        "schema_version": "f0_formal_execution_report_v1",
        "status": status,
        "run_label": args.run_label,
        "planned_blocks": 30,
        "completed_blocks": len(records),
        "planned_provider_calls": 120,
        "provider_calls_performed": summary.get("provider_calls_performed"),
        "network_attempts": summary.get("network_attempts"),
        "record_count": len(records),
        "branch_rows": branch_rows,
        "response_status": dict(response_status),
        "parsed_ok": parse_ok,
        "returned_model_metadata": returned_model,
        "request_hash_mismatches": request_hash_mismatches,
        "duplicate_payload_failures": duplicate_failures,
        "timestamp_failures": timestamp_failures,
        "protocol_exclusions": protocol_exclusions,
        "partial_files": len(list(run_root.rglob("*.partial"))),
        "selective_retry": summary.get("selective_retry"),
        "errors": errors,
        "provider_calls_after_completion": 0,
        "network_attempts_after_completion": 0,
    }
    report_path = run_root / "F0_FORMAL_EXECUTION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "F0_FORMAL_EXECUTION_REPORT.md").write_text(
        "# F0 formal execution report\n\n"
        f"Status: **{status}**\n\n"
        f"- Completed blocks: **{len(records)}/30**\n"
        f"- Provider calls: **{summary.get('provider_calls_performed')}/120**\n"
        f"- Network attempts: **{summary.get('network_attempts')}**\n"
        f"- Parsed branches: **{parse_ok}/120**\n"
        f"- Request hash mismatches: **{request_hash_mismatches}**\n"
        f"- Duplicate payload failures: **{duplicate_failures}**\n"
        f"- Protocol exclusions: **{protocol_exclusions}**\n"
        f"- Partial files: **{len(list(run_root.rglob('*.partial')))}**\n",
        encoding="utf-8",
    )
    manifest_path = run_root / "F0_FORMAL_OUTPUT_HASH_MANIFEST.jsonl"
    files = sorted(path for path in run_root.rglob("*") if path.is_file() and path.name != manifest_path.name)
    manifest_path.write_text(
        "".join(json.dumps({"path": str(path.relative_to(PROJECT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}, sort_keys=True) + "\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
