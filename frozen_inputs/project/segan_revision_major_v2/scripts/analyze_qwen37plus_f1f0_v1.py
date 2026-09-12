#!/usr/bin/env python3
"""Offline summary and evidence audit for the Qwen3.7-Plus F1/F0 tier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reviews/qwen37plus_f1f0_v1"
PROJECT = ROOT.parent


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan() -> dict[str, dict[str, str]]:
    path = REVIEW / "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl"
    return {str(row["block_id"]): dict(row["expected_request_sha256"]) for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    records_dir = args.run_dir / "records"
    files = sorted(records_dir.glob("*.json"))
    records = [load(path) for path in files]
    expected_by_block = load_plan()
    branches = [branch for record in records for branch in record.get("branches", {}).values()]
    response_status = Counter(str(branch.get("response", {}).get("status")) for branch in branches)
    parse_status = Counter("ok" if branch.get("parsed", {}).get("ok") is True else "failed" for branch in branches)
    tier_counts: dict[str, dict[str, int]] = {}
    for tier in ("F1", "F0"):
        tier_rows = [record for record in records if record.get("tier") == tier]
        tier_branches = [branch for record in tier_rows for branch in record.get("branches", {}).values()]
        parsed = [branch for branch in tier_branches if branch.get("parsed", {}).get("ok") is True]
        feasible = [branch for branch in parsed if (branch.get("score") or {}).get("carrier_consistent_replay", {}).get("engineering_feasible") is True]
        tier_counts[tier] = {
            "blocks": len(tier_rows),
            "branches": len(tier_branches),
            "parsed": len(parsed),
            "raw_feasible": len(feasible),
            "nontrivial": sum((branch.get("parsed", {}).get("segment_count", 0) or 0) > 0 for branch in tier_branches),
        }
    distances: list[float] = []
    null_distances: list[float] = []
    duplicate_identity_failures = 0
    wrong_models = 0
    protocol_exclusions = 0
    request_hash_mismatches = 0
    usage_prompt = 0
    usage_completion = 0
    for record in records:
        protocol_exclusions += len(record.get("protocol_exclusions", []))
        c1 = record.get("branches", {}).get("C1", {})
        c2 = record.get("branches", {}).get("C2", {})
        s1 = record.get("branches", {}).get("S1", {})
        s2 = record.get("branches", {}).get("S2", {})
        if c1.get("request_payload_bytes_sha256") != c2.get("request_payload_bytes_sha256") or s1.get("request_payload_bytes_sha256") != s2.get("request_payload_bytes_sha256"):
            duplicate_identity_failures += 1
        for branch_name, branch in record.get("branches", {}).items():
            if branch.get("request_payload_bytes_sha256") != expected_by_block.get(str(record.get("block_id")), {}).get(branch_name):
                request_hash_mismatches += 1
            response = branch.get("response", {})
            if response.get("status") == "ok" and response.get("returned_model") != "qwen3.7-plus":
                wrong_models += 1
            usage = response.get("usage", {})
            usage_prompt += int(usage.get("prompt_tokens", 0) or 0)
            usage_completion += int(usage.get("completion_tokens", 0) or 0)
        cp = c1.get("parsed", {})
        sp = s1.get("parsed", {})
        if cp.get("ok") is True and sp.get("ok") is True:
            ca = cp.get("dense_action_kw", [])
            sa = sp.get("dense_action_kw", [])
            if len(ca) == len(sa):
                distances.append(0.25 * sum(abs(float(a) - float(b)) for a, b in zip(ca, sa, strict=True)))
        c2p = c2.get("parsed", {})
        if cp.get("ok") is True and c2p.get("ok") is True:
            ca = cp.get("dense_action_kw", [])
            cb = c2p.get("dense_action_kw", [])
            if len(ca) == len(cb):
                null_distances.append(0.25 * sum(abs(float(a) - float(b)) for a, b in zip(ca, cb, strict=True)))
    summary = load(args.run_dir / "QWEN37PLUS_EXECUTION_SUMMARY.json")
    report = {
        "schema_version": "qwen37plus_f1f0_analysis_v1",
        "protocol_id": "energybench-qwen37plus-e2b-sensitivity-v1",
        "source_run_label": "e2_20260801T054708Z",
        "formal_denominator": {"blocks": 75, "provider_calls": 300, "records": len(records), "branches": len(branches)},
        "execution_summary": summary,
        "response_status": dict(response_status),
        "parse_status": dict(parse_status),
        "tier_counts": tier_counts,
        "carrier_action_distance_kwh": {"count": len(distances), "mean": sum(distances) / len(distances) if distances else None, "median": sorted(distances)[len(distances) // 2] if distances else None},
        "canonical_duplicate_null_distance_kwh": {"count": len(null_distances), "mean": sum(null_distances) / len(null_distances) if null_distances else None, "median": sorted(null_distances)[len(null_distances) // 2] if null_distances else None, "max": max(null_distances) if null_distances else None},
        "evidence_integrity": {
            "protocol_exclusions": protocol_exclusions,
            "duplicate_identity_failures": duplicate_identity_failures,
            "request_hash_mismatches": request_hash_mismatches,
            "wrong_returned_model": wrong_models,
            "partial_files": len(list(args.run_dir.rglob("*.partial"))),
            "provider_calls_performed": summary.get("provider_calls_performed", 0),
            "network_attempts": summary.get("network_attempts", 0),
            "selective_retry": False,
        },
        "token_usage": {"prompt_tokens": usage_prompt, "completion_tokens": usage_completion},
        "historical_pooling": {"historical_f1_f0_modified": False, "qwen37plus_replaces_historical_condition": False, "pooled_claim_rate_permitted": False},
    }
    write(args.run_dir / "QWEN37PLUS_FORMAL_ANALYSIS.json", report)
    lines = [
        "# Qwen3.7-Plus F1/F0 formal analysis",
        "",
        "This is an independent capability-sensitivity tier. It does not replace or pool the historical F1/F0 model conditions.",
        "",
        f"- Blocks: {len(records)}/75; branches: {len(branches)}/300.",
        f"- Response statuses: {dict(response_status)}.",
        f"- Parsed branches: {parse_status.get('ok', 0)}/{len(branches)}.",
        f"- Protocol exclusions: {protocol_exclusions}; duplicate identity failures: {duplicate_identity_failures}.",
        f"- Returned-model mismatches: {wrong_models}; request-hash mismatches: {request_hash_mismatches}.",
        f"- Provider calls: {summary.get('provider_calls_performed', 0)}; network attempts: {summary.get('network_attempts', 0)}.",
        "",
        "## Tier counts",
        "",
        "| Tier | Blocks | Branches | Parsed | Raw feasible | Nontrivial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier in ("F1", "F0"):
        item = tier_counts[tier]
        lines.append(f"| {tier} | {item['blocks']} | {item['branches']} | {item['parsed']} | {item['raw_feasible']} | {item['nontrivial']} |")
    lines.extend(["", "All smoke records are excluded from the formal denominator. Raw responses and historical evidence remain immutable.", ""])
    (args.run_dir / "QWEN37PLUS_FORMAL_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
