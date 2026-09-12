#!/usr/bin/env python3
"""Fail-closed offline verifier for Qwen3.7-Plus formal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan() -> dict[str, dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "reviews/qwen37plus_f1f0_v1/QWEN37PLUS_FORMAL_RUN_PLAN.jsonl"
    return {str(row["block_id"]): dict(row["expected_request_sha256"]) for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = load(args.run_dir / "QWEN37PLUS_EXECUTION_SUMMARY.json")
    records = sorted((args.run_dir / "records").glob("*.json"))
    expected_by_block = load_plan()
    errors: list[str] = []
    if len(records) != 75:
        errors.append(f"records={len(records)} expected=75")
    if summary.get("provider_calls_performed") != 300:
        errors.append("provider call count mismatch")
    if summary.get("protocol_exclusions") != 0:
        errors.append("protocol exclusions present")
    if summary.get("partial_files") != 0 or list(args.run_dir.rglob("*.partial")):
        errors.append("partial files present")
    seen: set[str] = set()
    branches = 0
    for path in records:
        record = load(path)
        key = f"{record.get('tier')}::{record.get('scenario_id')}::{record.get('model_condition')}"
        if key in seen:
            errors.append(f"duplicate block identity: {key}")
        seen.add(key)
        if record.get("protocol_id") != "energybench-qwen37plus-e2b-sensitivity-v1":
            errors.append(f"protocol id mismatch: {path.name}")
        for name in ("C1", "C2", "S1", "S2"):
            branch = record.get("branches", {}).get(name)
            if not isinstance(branch, dict):
                errors.append(f"missing branch {name}: {path.name}")
                continue
            branches += 1
            response = branch.get("response", {})
            if response.get("status") == "ok" and response.get("returned_model") != "qwen3.7-plus":
                errors.append(f"returned model mismatch: {path.name}/{name}")
            if branch.get("request_payload_bytes_sha256") != expected_by_block.get(str(record.get("block_id")), {}).get(name):
                errors.append(f"request hash mismatch: {path.name}/{name}")
    if branches != 300:
        errors.append(f"branches={branches} expected=300")
    result = {"status": "PASS" if not errors else "HOLD", "records": len(records), "branches": branches, "errors": errors, "provider_calls_performed": summary.get("provider_calls_performed", 0), "network_attempts": summary.get("network_attempts", 0), "hashes_checked": len(records)}
    (args.run_dir / "QWEN37PLUS_FORMAL_VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
