#!/usr/bin/env python3
"""Create the E2 completion report from saved analysis and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    run_dir = ROOT / "runs/experiments_v2/e2" / args.run_label
    execution = _read(run_dir / "E2_EXECUTION_SUMMARY.json")
    analysis = _read(run_dir / "E2_ANALYSIS.json")
    verification = _read(ROOT / "experiments_v2/reports/E2_VERIFICATION_REPORT.json")
    repair = _read(ROOT / "experiments_v2/reports/E2_PROJECTION_RUNNER_REPAIR_REPORT.json")
    sidecars = [_read(path) for path in sorted((run_dir / "projection_sidecars").glob("*.json"))]
    repair["completed_sidecars_total"] = len(sidecars)
    repair["computed_sidecars"] = sum(row["status"] == "computed" for row in sidecars)
    repair["parser_not_applicable_sidecars"] = sum(
        row.get("reason") == "not_applicable_parser_failure" for row in sidecars
    )
    repair["hard_timeout_sidecars"] = sum(
        row.get("reason") == "hard_timeout_120s" for row in sidecars
    )
    repair["repaired_script_sha256"] = _file_hash(ROOT / "scripts/project_experiments_v2_e2.py")
    repair["provider_evidence_tree_sha256"] = _tree_hash(
        [
            *sorted((run_dir / "records").glob("*.json")),
            *sorted((run_dir / "checkpoints").glob("*/*.json")),
        ]
    )
    _write(ROOT / "experiments_v2/reports/E2_PROJECTION_RUNNER_REPAIR_REPORT.json", repair)
    report = {
        "schema_version": "experiments_v2_e2_completion_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "E2_COMPLETED_AND_VERIFIED"
        if execution["status"] == "E2_COMPLETED" and verification["status"] == "pass"
        else "HOLD",
        "run_label": args.run_label,
        "planned_blocks": 360,
        "completed_blocks": execution["completed_blocks"],
        "planned_primary_calls": 2160,
        "actual_stage_calls": execution["actual_stage_calls"],
        "structured_stage2_skips": execution["structured_stage2_skips"],
        "provider_attempts_including_retries": execution["provider_attempts_including_retries"],
        "successful_provider_responses": execution["successful_provider_responses"],
        "protocol_exclusions": execution["protocol_exclusions"],
        "co_primary": analysis["co_primary"],
        "co_primary_by_model": analysis["co_primary_by_model"],
        "paired_null": analysis["paired_null"],
        "branch_results": analysis["branch_results"],
        "paired_parseability": analysis["paired_parseability"],
        "projection_results": analysis["projection_results"],
        "projection_sidecar_status": {
            "computed": repair["computed_sidecars"],
            "parser_not_applicable": repair["parser_not_applicable_sidecars"],
            "hard_timeout_120s": repair["hard_timeout_sidecars"],
        },
        "provider_cost": analysis["provider_cost"],
        "verification_status": verification["status"],
        "offline_projection_runner_repair_disclosed": True,
        "artifact_hashes": {
            "pre_run_manifest": _file_hash(
                ROOT / "experiments_v2/manifests/E2_PRE_RUN_MANIFEST.json"
            ),
            "run_plan": _file_hash(ROOT / "experiments_v2/manifests/E2_RUN_PLAN.jsonl"),
            "planned_call_ledger": _file_hash(run_dir / "PLANNED_CALL_LEDGER.jsonl"),
            "execution_summary": _file_hash(run_dir / "E2_EXECUTION_SUMMARY.json"),
            "analysis": _file_hash(run_dir / "E2_ANALYSIS.json"),
            "provider_evidence_tree": repair["provider_evidence_tree_sha256"],
            "projection_sidecar_tree": _tree_hash(
                sorted((run_dir / "projection_sidecars").glob("*.json"))
            ),
            "post_run_hash_manifest": _file_hash(run_dir / "E2_POST_RUN_HASH_MANIFEST.json"),
        },
        "provider_calls_performed_by_finalizer": 0,
        "network_attempts_by_finalizer": 0,
    }
    _write(ROOT / "experiments_v2/reports/E2_COMPLETION_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "E2_COMPLETED_AND_VERIFIED":
        raise SystemExit("E2 completion gate failed")


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


if __name__ == "__main__":
    main()
