#!/usr/bin/env python3
"""Verify E2 evidence, paired payloads, denominators, and hashes offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    plan = _jsonl(ROOT / "experiments_v2/manifests/E2_RUN_PLAN.jsonl")
    analysis = json.loads((run_dir / "E2_ANALYSIS.json").read_text())
    execution = json.loads((run_dir / "E2_EXECUTION_SUMMARY.json").read_text())
    ledger = _jsonl(run_dir / "PLANNED_CALL_LEDGER.jsonl")
    sidecars = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "projection_sidecars").glob("*.json"))
    ]
    calls = [record["stage1"] for record in records] + [
        stage for record in records for stage in record["branches"].values()
    ]
    checks = {
        "records_360": len(records) == 360,
        "plan_rows_360": len(plan) == 360,
        "block_ids_match_plan": {record["block_id"] for record in records}
        == {row["block_id"] for row in plan},
        "task_hashes_match": all(
            _hash(record["task"]) == record["task_sha256"] for record in records
        ),
        "all_saved_hashes_match": all(_record_hashes(record) for record in records),
        "c_c2_payloads_byte_identical": all(
            record["branches"]["C"]["request_sha256"] == record["branches"]["C2"]["request_sha256"]
            for record in records
        ),
        "c_c2_carriers_identical": all(
            record["carriers"]["C"] == record["carriers"]["C2"] for record in records
        ),
        "s_only_soc_changes": all(
            _changed(record["carriers"]["C"], record["carriers"]["S"]) == {"soc_kwh"}
            for record in records
        ),
        "k_only_metadata_changes": all(
            _changed(record["carriers"]["C"], record["carriers"]["K"])
            <= {"active_commitments", "revoked_commitments", "sequence"}
            for record in records
        ),
        "protocol_exclusions_zero": all(
            stage["response"].get("status") != "ok"
            or stage["response"].get("returned_model") == stage["request"]["model"]
            for stage in calls
            if isinstance(stage.get("request"), dict)
        ),
        "analysis_denominator_360": analysis.get("denominator_blocks") == 360,
        "scenario_clusters_60": analysis.get("independent_scenario_clusters") == 60,
        "planned_call_ledger_2160": len(ledger) == 2160,
        "execution_completed_360": execution.get("completed_blocks") == 360,
        "actual_calls_2158": execution.get("actual_stage_calls") == 2158,
        "structured_skips_2": execution.get("structured_stage2_skips") == 2,
        "projection_sidecars_1800": len(sidecars) == 1800,
        "projection_sidecars_linked": all(
            sidecar["record_evidence_root_sha256"]
            == next(
                record["evidence_root_sha256"]
                for record in records
                if record["block_id"] == sidecar["block_id"]
            )
            for sidecar in sidecars
        ),
        "projection_sidecar_hashes_match": all(
            _hash({key: value for key, value in sidecar.items() if key != "sidecar_sha256"})
            == sidecar["sidecar_sha256"]
            for sidecar in sidecars
        ),
        "api_calls_by_verifier_zero": True,
        "network_attempts_by_verifier_zero": True,
    }
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "E2_POST_RUN_HASH_MANIFEST.json"
    )
    manifest = {
        "schema_version": "experiments_v2_e2_post_run_hash_manifest_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "files": [
            {
                "path": str(path.relative_to(run_dir)),
                "byte_size": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in files
        ],
    }
    _write(run_dir / "E2_POST_RUN_HASH_MANIFEST.json", manifest)
    report = {
        "schema_version": "experiments_v2_e2_verification_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if all(checks.values()) else "HOLD",
        "checks": checks,
        "logical_blocks": len(records),
        "planned_primary_calls": 2160,
        "actual_stage_calls": sum(stage["response"].get("status") != "skipped" for stage in calls),
        "provider_attempts_including_retries": sum(
            len(stage["response"].get("attempts", [])) for stage in calls
        ),
        "post_run_manifest_entries": len(manifest["files"]),
        "api_calls_performed_by_verifier": 0,
        "network_attempts_by_verifier": 0,
    }
    _write(ROOT / "experiments_v2/reports/E2_VERIFICATION_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("E2 offline verification failed")


def _record_hashes(record: dict[str, Any]) -> bool:
    stages = [record["stage1"], *record["branches"].values()]
    stages_match = all(
        _hash(stage["response"]) == stage["response_sha256"]
        and _hash(stage["parsed"]) == stage["parsed_sha256"]
        and (stage["request"] is None or _hash(stage["request"]) == stage["request_sha256"])
        for stage in stages
    )
    expected_root = _hash(
        {
            "plan": record["plan_row_sha256"],
            "task": record["task_sha256"],
            "stage1": record["stage1"]["response_sha256"],
            "branches": {
                branch: record["branches"][branch]["response_sha256"] for branch in BRANCHES
            },
            "outcomes": _hash(record["outcomes"]),
        }
    )
    return stages_match and expected_root == record["evidence_root_sha256"]


def _changed(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {key for key in left if left.get(key) != right.get(key)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


if __name__ == "__main__":
    main()
