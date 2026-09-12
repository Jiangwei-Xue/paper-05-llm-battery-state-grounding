#!/usr/bin/env python3
"""Guarded F0 120-call formal runner; default mode is offline preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reviews/f0_formal_v1/F0_FORMAL_PRE_CALL_MANIFEST.json"
PLAN = ROOT / "reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"
VERIFY = ROOT / "scripts/verify_f0_formal_preflight.py"
CORE_PATH = ROOT / "scripts/run_e2b_v2.py"
PILOT_HELPERS = ROOT / "scripts/run_f0_pilot_v1.py"
RUN_ROOT = ROOT / "runs/f0_formal_v1"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-provider-calls", action="store_true")
    parser.add_argument("--run-label")
    args = parser.parse_args()
    check = subprocess.run([sys.executable, str(VERIFY)], cwd=ROOT, check=False)
    if check.returncode != 0:
        return check.returncode
    if not args.execute:
        print(json.dumps({
            "status": "PREFLIGHT_PASS_AWAITING_AUTHORIZATION",
            "execution_started": False,
            "planned_blocks": 30,
            "planned_provider_calls": 120,
            "provider_calls_performed": 0,
            "network_attempts": 0,
        }, indent=2))
        return 0
    if not args.authorize_provider_calls:
        print("F0 formal execution requires a separate explicit authorization", file=sys.stderr)
        return 2
    if not args.run_label:
        print("--run-label is required for formal execution", file=sys.stderr)
        return 2
    runtime = load_module(CORE_PATH, "f0_formal_e2b_v2_runtime")
    runtime.DEFAULT_MANIFEST = MANIFEST
    runtime.RUN_ROOT = RUN_ROOT
    manifest = runtime.verify_manifest(MANIFEST)
    if not runtime.public_hash_gate_ready():
        raise SystemExit("F0 formal execution requires the public file-hash gate")
    helper = load_module(PILOT_HELPERS, "f0_pilot_helpers_for_formal")
    plan = helper.adapt_plan_for_core(runtime.load_jsonl(PLAN))
    if len(plan) != 30:
        raise SystemExit("F0 formal plan cardinality mismatch")
    specs = runtime.load_model_specs()
    available = helper.credentials()
    keys = {
        name: available["DEEPSEEK_API_KEY"]
        if name == "deepseek_formal"
        else (available.get("QWEN_API_KEY") or available["DASHSCOPE_API_KEY"])
        for name in specs
    }
    summary = runtime.asyncio.run(
        runtime.execute(
            RUN_ROOT / args.run_label,
            plan,
            specs,
            keys,
            tier="F0_FORMAL",
            concurrency=int(manifest["frozen_concurrency"]),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
