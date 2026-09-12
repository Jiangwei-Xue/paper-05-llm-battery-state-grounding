#!/usr/bin/env python3
"""Freeze the E1 run plan and pre-run hashes without provider access."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.e1 import build_e1_plan  # noqa: E402
from experiments_v2.hashing import sha256_text  # noqa: E402
from experiments_v2.p0 import canonical_initial_state, compile_prompt  # noqa: E402

from energy_agent_reliability.provenance import read_jsonl, sha256_file, write_jsonl  # noqa: E402

TASKS = ROOT / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl"
PLAN = ROOT / "experiments_v2/manifests/E1_RUN_PLAN.jsonl"
MANIFEST = ROOT / "experiments_v2/manifests/E1_PRE_RUN_MANIFEST.json"


def main() -> None:
    tasks = list(read_jsonl(TASKS))
    plan = build_e1_plan(tasks)
    write_jsonl(PLAN, plan)
    frozen_paths = [
        TASKS,
        PLAN,
        ROOT / "experiments_v2/manifests/E1_ADMISSION_RESULTS.jsonl",
        ROOT / "experiments_v2/reports/E1_PREPARATION_REPORT.json",
        ROOT / "experiments_v2/reports/P0_I_BEST_DECISION.json",
        ROOT / "experiments_v2/configs/experiment_e1.yaml",
        ROOT / "experiments_v2/configs/models.yaml",
        ROOT / "experiments_v2/configs/pricing.yaml",
        ROOT / "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml",
        ROOT / "experiments_v2/src/experiments_v2/p0.py",
        ROOT / "experiments_v2/src/experiments_v2/e1.py",
        ROOT / "scripts/prepare_experiments_v2_e1.py",
        ROOT / "scripts/run_experiments_v2_p0.py",
        ROOT / "scripts/run_experiments_v2_e1.py",
        ROOT / "scripts/project_experiments_v2_e1.py",
        ROOT / "scripts/analyze_experiments_v2_e1.py",
        ROOT / "scripts/verify_experiments_v2_e1.py",
        ROOT / "experiments_v2/tests/test_e1.py",
        Path(__file__),
    ]
    prompt_hashes = {
        f"{task['scenario_id']}|{interface}|stage1": sha256_text(
            compile_prompt(task, interface, "stage1", canonical_initial_state(task))
        )
        for task in tasks
        for interface in ("I0", "I3")
    }
    manifest = {
        "schema_version": "experiments_v2_e1_pre_run_manifest_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_no_provider_calls",
        "planned_logical_episodes": len(plan),
        "planned_primary_calls": sum(int(row["planned_stage_calls"]) for row in plan),
        "models": sorted({str(row["model_condition"]) for row in plan}),
        "interfaces": sorted({str(row["interface"]) for row in plan}),
        "task_count": len(tasks),
        "prompt_hashes": prompt_hashes,
        "file_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_paths},
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "p0_rows_in_denominator": False,
    }
    if len(plan) != 240 or manifest["planned_primary_calls"] != 480:
        raise SystemExit("E1 cardinality gate failed")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
