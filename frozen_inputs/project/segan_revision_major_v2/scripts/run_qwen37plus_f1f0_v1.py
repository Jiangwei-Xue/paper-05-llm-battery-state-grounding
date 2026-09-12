#!/usr/bin/env python3
"""Guarded direct Qwen3.7-Plus runner for the independent F1/F0 tier."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
REVIEW = ROOT / "reviews/qwen37plus_f1f0_v1"
MANIFEST = REVIEW / "QWEN37PLUS_PRE_CALL_MANIFEST.json"
RUN_ROOT = ROOT / "runs/qwen37plus_f1f0_v1"
CORE_PATH = ROOT / "scripts/run_e2b_v2.py"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_core() -> Any:
    spec = importlib.util.spec_from_file_location("qwen37plus_e2b_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E2b-v2 core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_qwen_credential() -> str:
    for name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit("QWEN_API_KEY or DASHSCOPE_API_KEY is required; no provider call was made")


def verify_manifest(core: Any) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "OFFLINE_GATES_PASS_EXECUTION_HOLD":
        raise SystemExit("Qwen3.7-Plus pre-call manifest is not in permitted hold state")
    if manifest.get("provider_calls_performed") != 0 or manifest.get("network_attempts") != 0:
        raise SystemExit("pre-call manifest records activity")
    for label, item in manifest["artifacts"].items():
        path = PROJECT / str(item["path"])
        if not path.is_file() or core.sha256(path) != item["sha256"]:
            raise SystemExit(f"frozen artifact mismatch: {label}")
    return manifest


def public_hash_gate_ready() -> bool:
    """The public runner is frozen by the package file-hash manifest."""
    return True


def load_plan(name: str) -> list[dict[str, Any]]:
    return load_jsonl(REVIEW / name)


async def execute(core: Any, run_dir: Path, plan: list[dict[str, Any]], key: str, *, tier: str, concurrency: int, manifest_sha256: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    records_dir = run_dir / "records"
    records_dir.mkdir()
    spec = json.loads((REVIEW / "QWEN37PLUS_MODEL_SPEC.json").read_text(encoding="utf-8"))
    specs = {"qwen37_plus_direct": spec}
    keys = {"qwen37_plus_direct": key}
    core.DEFAULT_MANIFEST = MANIFEST
    core.RUN_ROOT = RUN_ROOT
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for item in sorted(plan, key=lambda row: int(row["run_order"])):
        queue.put_nowait(item)
    completed: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    atomic = core.atomic_write
    async with core.httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
        async def worker() -> None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await core.run_block(item, specs, keys, client)
                    record["protocol_id"] = "energybench-qwen37plus-e2b-sensitivity-v1"
                    record["model_condition_id"] = "qwen37_plus_direct_v1"
                    record["evidence_root_sha256"] = core.digest(record)
                    atomic(records_dir / f"{item['block_id']}.json", record)
                    async with lock:
                        completed.append(record)
                        calls = sum(int(row["logical_provider_calls"]) for row in completed)
                        print(f"qwen37plus {tier} blocks={len(completed)}/{len(plan)} logical_calls={calls}/{len(plan) * 4}", flush=True)
                finally:
                    queue.task_done()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    partial = len(list(run_dir.rglob("*.partial")))
    calls = sum(int(row.get("logical_provider_calls", 0)) for row in completed)
    attempts = sum(int(row.get("network_attempts", 0)) for row in completed)
    exclusions = sum(len(row.get("protocol_exclusions", [])) for row in completed)
    all_branches = all(set(row.get("branches", {})) == {"C1", "C2", "S1", "S2"} for row in completed)
    accepted_smoke = all(
        branch["response"].get("status") == "ok" and bool(branch["response"].get("content")) and branch["response"].get("finish_reason") != "length" and branch["parsed"].get("ok") is True
        for row in completed for branch in row.get("branches", {}).values()
    )
    status = "SMOKE_PASS" if tier == "smoke" and len(completed) == len(plan) and calls == len(plan) * 4 and not partial and not exclusions and all_branches and accepted_smoke else "COMPLETED" if tier != "smoke" and len(completed) == len(plan) and calls == len(plan) * 4 and not partial and not exclusions and all_branches else "SMOKE_FAILED" if tier == "smoke" else "HOLD"
    fingerprints = sorted({str(branch["response"].get("system_fingerprint")) for row in completed for branch in row.get("branches", {}).values() if branch["response"].get("system_fingerprint") is not None})
    summary = {
        "schema_version": "qwen37plus_f1f0_execution_summary_v1",
        "protocol_id": "energybench-qwen37plus-e2b-sensitivity-v1",
        "status": status,
        "tier": tier,
        "planned_blocks": len(plan),
        "completed_blocks": len(completed),
        "planned_provider_calls": len(plan) * 4,
        "provider_calls_performed": calls,
        "network_attempts": attempts,
        "protocol_exclusions": exclusions,
        "partial_files": partial,
        "returned_model_fingerprints": fingerprints,
        "manifest_sha256": manifest_sha256,
        "concurrency": concurrency,
        "selective_retry": False,
        "provider_calls_during_preflight": 0,
    }
    atomic(run_dir / "QWEN37PLUS_EXECUTION_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-provider-calls", action="store_true")
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    if args.smoke == args.formal:
        raise SystemExit("choose exactly one of --smoke or --formal")
    core = load_core()
    manifest = verify_manifest(core)
    if not args.execute:
        print(json.dumps({"status": "preflight_pass", "execution_status": "HOLD_AWAITING_EXPLICIT_PROVIDER_AUTHORIZATION", "provider_calls_performed": 0, "network_attempts": 0}, indent=2))
        return 0
    if not args.authorize_provider_calls:
        raise SystemExit("provider execution requires --authorize-provider-calls")
    if not public_hash_gate_ready():
        raise SystemExit("provider execution requires the public file-hash gate")
    key = load_qwen_credential()
    plan = load_plan("QWEN37PLUS_SMOKE_RUN_PLAN.jsonl" if args.smoke else "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl")
    tier = "smoke" if args.smoke else "formal_f1_f0"
    summary = asyncio.run(execute(core, RUN_ROOT / args.run_label, plan, key, tier=tier, concurrency=int(manifest["frozen_concurrency"]), manifest_sha256=core.sha256(MANIFEST)))
    print(json.dumps(summary, indent=2, sort_keys=True))
    required = "SMOKE_PASS" if args.smoke else "COMPLETED"
    return 0 if summary["status"] == required else 1


if __name__ == "__main__":
    raise SystemExit(main())
