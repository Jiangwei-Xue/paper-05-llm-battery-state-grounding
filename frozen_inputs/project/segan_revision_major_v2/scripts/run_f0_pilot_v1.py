#!/usr/bin/env python3
"""F0 pilot entry point with an offline preflight and guarded execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts/verify_f0_pilot_preflight.py"
MANIFEST = ROOT / "reviews/f0_protocol_v1/F0_PILOT_PRE_CALL_MANIFEST.json"
PILOT_PLAN = ROOT / "reviews/f0_protocol_v1/F0_PILOT_RUN_PLAN.jsonl"
SMOKE_PLAN = ROOT / "reviews/f0_protocol_v1/F0_PILOT_SMOKE_RUN_PLAN.jsonl"
CORE_PATH = ROOT / "scripts/run_e2b_v2.py"


def load_core() -> Any:
    spec = importlib.util.spec_from_file_location("f0_e2b_v2_runtime", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E2b-v2 runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def credentials() -> dict[str, str]:
    values: dict[str, str] = {}
    for key in ("DEEPSEEK_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        if os.environ.get(key):
            values[key] = str(os.environ[key])
    if not values.get("DEEPSEEK_API_KEY"):
        raise SystemExit("missing DeepSeek credential; key value was not printed")
    if not (values.get("QWEN_API_KEY") or values.get("DASHSCOPE_API_KEY")):
        raise SystemExit("missing Qwen/DashScope credential; no provider call was made")
    return values


def adapt_plan_for_core(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the core's stable block-id alias without changing frozen plan rows."""
    adapted: list[dict[str, Any]] = []
    for item in plan:
        row = dict(item)
        if "block_id" not in row:
            source_block = row.get("source_f0_block_id")
            if not isinstance(source_block, str) or not source_block:
                raise SystemExit("F0 plan row missing stable source_f0_block_id")
            row["block_id"] = source_block
        adapted.append(row)
    return adapted


def execute_run(plan_path: Path, run_label: str, tier: str) -> int:
    preflight = subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=False)
    if preflight.returncode != 0:
        return preflight.returncode
    runtime = load_core()
    runtime.DEFAULT_MANIFEST = MANIFEST
    runtime.RUN_ROOT = ROOT / "runs/f0_pilot_v1"
    manifest = runtime.verify_manifest(MANIFEST)
    if not runtime.public_hash_gate_ready():
        raise SystemExit("F0 execution requires the public file-hash gate")
    # The F0 freeze deliberately records the archived source block under
    # ``source_f0_block_id``.  The reused E2b core requires the equivalent
    # stable identifier under ``block_id``.  This adapter is in-memory only;
    # frozen plan bytes and their hashes remain unchanged.
    plan = adapt_plan_for_core(runtime.load_jsonl(plan_path))
    expected = 2 if tier == "smoke" else 10
    if len(plan) != expected:
        raise SystemExit(f"F0 {tier} plan cardinality mismatch")
    specs = runtime.load_model_specs()
    available = credentials()
    keys = {name: available["DEEPSEEK_API_KEY"] if name == "deepseek_formal" else (available.get("QWEN_API_KEY") or available["DASHSCOPE_API_KEY"]) for name in specs}
    summary = runtime.asyncio.run(
        runtime.execute(
            runtime.RUN_ROOT / run_label,
            plan,
            specs,
            keys,
            tier="smoke" if tier == "smoke" else "F0_PILOT",
            concurrency=int(manifest["frozen_concurrency"]),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"SMOKE_PASS", "COMPLETED"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run the separate 8-call smoke")
    parser.add_argument("--pilot", action="store_true", help="run the frozen 40-call pilot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-provider-calls", action="store_true")
    parser.add_argument("--run-label")
    args = parser.parse_args()
    check = subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=False)
    if check.returncode != 0:
        return check.returncode
    if args.execute:
        if not args.authorize_provider_calls:
            print("F0 provider execution requires a separate explicit authorization", file=sys.stderr)
            return 2
        if args.smoke == args.pilot:
            print("choose exactly one of --smoke or --pilot", file=sys.stderr)
            return 2
        if not args.run_label:
            print("--run-label is required for provider execution", file=sys.stderr)
            return 2
        plan_path = SMOKE_PLAN if args.smoke else PILOT_PLAN
        return execute_run(plan_path, args.run_label, "smoke" if args.smoke else "pilot")
    print(json.dumps({
        "status": "HOLD_AWAITING_EXPLICIT_PROVIDER_AUTHORIZATION",
        "execution_started": False,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
