#!/usr/bin/env python3
"""Verify the F0 formal VCR evidence package without network or provider access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from e2b_v2_common import canonical_json, digest, score_branch

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="reviews/f0_formal_v1/artifact_release")
    args = parser.parse_args()
    package = ROOT / args.package
    errors: list[str] = []
    manifest_path = package / "F0_FORMAL_HASH_MANIFEST.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in manifest_rows:
        path = package / str(row["path"])
        if not path.is_file():
            errors.append(f"missing_manifest_file:{row['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['path']}")
    evidence_rows = [json.loads(line) for line in (package / "row_evidence.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(evidence_rows) != 120:
        errors.append(f"evidence_row_count:{len(evidence_rows)}")
    seen: set[str] = set()
    replay_ok = 0
    timestamp_ok = 0
    byte_identity_ok = 0
    for row in evidence_rows:
        evidence_id = str(row["evidence_id"])
        if evidence_id in seen:
            errors.append(f"duplicate_evidence_id:{evidence_id}")
        seen.add(evidence_id)
        block = load_json(package / str(row["row_path"]))
        if block["task_sha256"] != digest(block["task"]):
            errors.append(f"task_hash_mismatch:{evidence_id}")
        for branch in ("C1", "C2", "S1", "S2"):
            item = block["branches"][branch]
            hashes = item["evidence_hashes"]
            if hashes["prompt_sha256"] != item["prompt_sha256"]:
                errors.append(f"prompt_hash_mismatch:{evidence_id}")
            if hashes["parsed_output_sha256"] != digest(item["parsed"]):
                errors.append(f"parsed_hash_mismatch:{evidence_id}")
            if hashes["raw_response_sha256"] != digest(item["response"].get("raw_response")):
                errors.append(f"raw_hash_mismatch:{evidence_id}")
            if item["parsed"].get("ok") is True:
                recomputed = score_branch(block["task"], branch, [float(value) for value in item["parsed"]["dense_action_kw"]])
                if canonical_json(recomputed) != canonical_json(item["score"]):
                    errors.append(f"replay_mismatch:{evidence_id}")
                else:
                    replay_ok += 1
            attempts = item["response"].get("attempts", [])
            if not attempts or any(not attempt.get("request_sent_utc") or not attempt.get("response_received_utc") for attempt in attempts):
                errors.append(f"timestamp_missing:{evidence_id}")
            else:
                timestamp_ok += 1
            if item["response"].get("status") != "ok":
                errors.append(f"unexpected_response_status:{evidence_id}")
        comparison = block["branch_comparison"]
        if not comparison["C1_C2_request_byte_identity"] or not comparison["S1_S2_request_byte_identity"]:
            errors.append(f"duplicate_payload_identity_failure:{evidence_id}")
        else:
            byte_identity_ok += 1
        if block.get("provider_calls_performed") != 0 or block.get("network_attempts") != 0:
            errors.append(f"package_build_network_counter_nonzero:{evidence_id}")
    private_markers = ("NREL_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "Bearer ", ".env")
    for path in package.rglob("*"):
        if path.is_file() and path.name != manifest_path.name:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in private_markers:
                if marker in text:
                    errors.append(f"private_marker:{marker}:{path.relative_to(package)}")
    report = {"schema_version": "f0_formal_vcr_verification_v1", "status": "PASS" if not errors else "HOLD", "package": str(package), "evidence_rows": len(evidence_rows), "replay_exact_rows": replay_ok, "timestamp_complete_branch_rows": timestamp_ok, "duplicate_payload_identity_blocks": byte_identity_ok, "manifest_entries": len(manifest_rows), "provider_calls_performed_by_verifier": 0, "network_attempts_by_verifier": 0, "errors": errors}
    (ROOT / "reviews/f0_formal_v1/F0_FORMAL_VCR_VERIFICATION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
