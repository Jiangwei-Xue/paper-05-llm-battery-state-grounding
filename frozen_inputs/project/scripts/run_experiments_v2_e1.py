#!/usr/bin/env python3
"""Run the 480-call E1 interface-confirmation experiment."""

from __future__ import annotations

from pathlib import Path

import run_experiments_v2_p0 as runner

ROOT = Path(__file__).resolve().parents[1]

runner.TASKS = ROOT / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl"
runner.PLAN = ROOT / "experiments_v2/manifests/E1_RUN_PLAN.jsonl"
runner.MANIFEST = ROOT / "experiments_v2/manifests/E1_PRE_RUN_MANIFEST.json"
runner.RUN_ROOT = ROOT / "runs/experiments_v2/e1"
runner.PRE_RUN_TAG = "energybench-v2-e1-pre-run"
runner.EXPERIMENT_LABEL = "E1"
runner.EXPECTED_EPISODES = 240
runner.EXPECTED_CALLS = 480
runner.IDENTITY_SCHEMA = "experiments_v2_e1_run_identity_v1"
runner.RECORD_SCHEMA = "experiments_v2_e1_episode_v1"
runner.EVIDENCE_TIER = "confirmatory_interface"
runner.EXCLUDED_FROM_CLAIMS = False
runner.EXECUTION_SCHEMA = "experiments_v2_e1_execution_summary_v1"
runner.COMPLETION_STATUS = "E1_COMPLETED"
runner.HOLD_FILENAME = "E1_HOLD.json"
runner.SUMMARY_FILENAME = "E1_EXECUTION_SUMMARY.json"
runner.HASH_FILENAME = "E1_HASH_MANIFEST.json"
runner.HASH_SCHEMA = "experiments_v2_e1_hash_manifest_v1"
runner.ENABLE_INLINE_PROJECTION = False


if __name__ == "__main__":
    runner.main()
