#!/usr/bin/env python3
"""Verify E1 row evidence, projection sidecars, and hashes offline."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    sidecars = {
        path.stem: json.loads(path.read_text())
        for path in sorted((run_dir / "projection_sidecars").glob("*.json"))
    }
    plan = _jsonl(ROOT / "experiments_v2/manifests/E1_RUN_PLAN.jsonl")
    analysis = json.loads((run_dir / "E1_ANALYSIS.json").read_text())
    row_checks = [_verify_record(record) for record in records]
    sidecar_checks = [
        sidecars.get(str(record["episode_id"]), {}).get("record_evidence_root_sha256")
        == record["evidence_root_sha256"]
        for record in records
    ]
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "E1_POST_RUN_HASH_MANIFEST.json"
    )
    manifest = {
        "schema_version": "experiments_v2_e1_post_run_hash_manifest_v1",
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
    _write(run_dir / "E1_POST_RUN_HASH_MANIFEST.json", manifest)
    checks = {
        "records_240": len(records) == 240,
        "plan_rows_240": len(plan) == 240,
        "episode_ids_match_plan": {record["episode_id"] for record in records}
        == {row["episode_id"] for row in plan},
        "all_row_hashes_match": all(all(row.values()) for row in row_checks),
        "task_hashes_match": all(bool(record["task_hash_matches"]) for record in records),
        "projection_sidecars_240": len(sidecars) == 240,
        "projection_sidecars_linked": all(sidecar_checks),
        "protocol_exclusions_zero": all(
            record[stage]["response"].get("status") != "ok"
            or record[stage]["response"].get("returned_model")
            == record[stage]["request"]["model"]
            for record in records
            for stage in ("stage1", "stage2")
        ),
        "analysis_denominator_240": analysis.get("denominator_logical_episodes") == 240,
        "claim_bearing_tier": all(
            not record["excluded_from_claim_bearing_analysis"] for record in records
        ),
        "api_calls_by_verifier_zero": True,
    }
    report = {
        "schema_version": "experiments_v2_e1_verification_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if all(checks.values()) else "HOLD",
        "checks": checks,
        "logical_episodes": len(records),
        "planned_stage_calls": 480,
        "provider_attempts_including_retries": sum(
            len(record[stage]["response"].get("attempts", []))
            for record in records
            for stage in ("stage1", "stage2")
        ),
        "successful_provider_responses": sum(
            record[stage]["response"].get("status") == "ok"
            for record in records
            for stage in ("stage1", "stage2")
        ),
        "projection_computed": sum(
            sidecar.get("status") == "computed" for sidecar in sidecars.values()
        ),
        "post_run_manifest_entries": len(manifest["files"]),
        "api_calls_performed_by_verifier": 0,
        "network_attempts_by_verifier": 0,
    }
    _write(ROOT / "experiments_v2/reports/E1_VERIFICATION_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("E1 offline verification failed")


def _verify_record(record: dict[str, Any]) -> dict[str, bool]:
    return {
        "task": _json_hash(record["task"]) == record["task_sha256"],
        "stage1_request": _json_hash(record["stage1"]["request"])
        == record["stage1"]["request_sha256"],
        "stage1_response": _json_hash(record["stage1"]["response"])
        == record["stage1"]["response_sha256"],
        "stage1_parsed": _json_hash(record["stage1"]["parsed"])
        == record["stage1"]["parsed_sha256"],
        "stage2_request": _json_hash(record["stage2"]["request"])
        == record["stage2"]["request_sha256"],
        "stage2_response": _json_hash(record["stage2"]["response"])
        == record["stage2"]["response_sha256"],
        "stage2_parsed": _json_hash(record["stage2"]["parsed"])
        == record["stage2"]["parsed_sha256"],
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    main()
