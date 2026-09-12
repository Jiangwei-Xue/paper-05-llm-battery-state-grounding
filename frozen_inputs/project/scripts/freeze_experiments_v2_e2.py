#!/usr/bin/env python3
"""Freeze the E2 block plan, prompt invariants, and file hashes."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.e2 import BRANCHES, INTERFACE, build_e2_plan, e2_states  # noqa: E402
from experiments_v2.hashing import sha256_json, sha256_text  # noqa: E402
from experiments_v2.p0 import build_payload, compile_prompt  # noqa: E402

from energy_agent_reliability.provenance import read_jsonl, sha256_file, write_jsonl  # noqa: E402

TASKS = ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl"
PLAN = ROOT / "experiments_v2/manifests/E2_RUN_PLAN.jsonl"
MANIFEST = ROOT / "experiments_v2/manifests/E2_PRE_RUN_MANIFEST.json"


def main() -> None:
    tasks = list(read_jsonl(TASKS))
    plan = build_e2_plan(tasks)
    write_jsonl(PLAN, plan)
    models = yaml.safe_load((ROOT / "experiments_v2/configs/models.yaml").read_text())["conditions"]
    prompt_hashes: dict[str, dict[str, str]] = {}
    for task in tasks:
        states = e2_states(task)
        task_hashes: dict[str, str] = {}
        for model in ("deepseek_formal", "qwen_flash"):
            row = models[model]
            for label in ("stage1", "C", "S", "K", "C2"):
                stage = "stage1" if label == "stage1" else "stage2"
                state = states["initial"] if label == "stage1" else states[label]
                prompt = compile_prompt(task, INTERFACE, stage, state)
                payload = build_payload(
                    model_id=str(row["configured_model_id"]),
                    provider=str(row["provider"]),
                    interface=INTERFACE,
                    prompt=prompt,
                    max_output_tokens=int(row["max_output_tokens"]),
                )
                task_hashes[f"{model}|{label}|prompt"] = sha256_text(prompt)
                task_hashes[f"{model}|{label}|payload"] = sha256_json(payload)
            if task_hashes[f"{model}|C|payload"] != task_hashes[f"{model}|C2|payload"]:
                raise SystemExit("C/C2 frozen payload mismatch")
        prompt_hashes[str(task["scenario_id"])] = task_hashes
    frozen = [
        TASKS,
        PLAN,
        ROOT / "experiments_v2/manifests/E2_ADMISSION_RESULTS.jsonl",
        ROOT / "experiments_v2/reports/E2_PREPARATION_REPORT.json",
        ROOT / "experiments_v2/reports/E2_VARIABLE_CONTROL_AUDIT.json",
        ROOT / "experiments_v2/reports/E2_VARIABLE_CONTROL_AUDIT.md",
        ROOT / "experiments_v2/configs/experiment_e2.yaml",
        ROOT / "experiments_v2/configs/models.yaml",
        ROOT / "experiments_v2/configs/pricing.yaml",
        ROOT / "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml",
        ROOT / "experiments_v2/src/experiments_v2/p0.py",
        ROOT / "experiments_v2/src/experiments_v2/carriers.py",
        ROOT / "experiments_v2/src/experiments_v2/e2.py",
        ROOT / "scripts/prepare_experiments_v2_e2.py",
        ROOT / "scripts/audit_experiments_v2_e2_controls.py",
        ROOT / "scripts/run_experiments_v2_e2.py",
        ROOT / "scripts/project_experiments_v2_e2.py",
        ROOT / "scripts/analyze_experiments_v2_e2.py",
        ROOT / "scripts/verify_experiments_v2_e2.py",
        ROOT / "experiments_v2/tests/test_e2.py",
        Path(__file__),
    ]
    counts = Counter(str(row["model_condition"]) for row in plan)
    manifest = {
        "schema_version": "experiments_v2_e2_pre_run_manifest_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_no_provider_calls",
        "task_count": len(tasks),
        "planned_blocks": len(plan),
        "planned_primary_calls": sum(int(row["planned_stage_calls"]) for row in plan),
        "model_block_counts": dict(counts),
        "stage2_branches": list(BRANCHES),
        "interface": INTERFACE,
        "stage1_shared": True,
        "c2_null_control": "byte_identical_payload_to_C",
        "m_payload_rule": "constructed_only_from_saved_unrepaired_stage1_state",
        "run_order_seed": 20260803,
        "bootstrap_seed": 20260716,
        "prompt_and_payload_hashes": prompt_hashes,
        "file_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen},
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    if (
        len(plan) != 360
        or manifest["planned_primary_calls"] != 2160
        or counts != {"deepseek_formal": 180, "qwen_flash": 180}
    ):
        raise SystemExit("E2 cardinality or balance gate failed")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "task_count",
                    "planned_blocks",
                    "planned_primary_calls",
                    "model_block_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
