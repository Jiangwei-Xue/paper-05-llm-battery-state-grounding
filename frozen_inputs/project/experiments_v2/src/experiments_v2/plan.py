"""Deterministic V2 call-plan construction."""

from __future__ import annotations

import random
from typing import Any

from .e3_plan import build_e3_v22_plan
from .hashing import sha256_json

PROTOCOL_VERSION = "energybench-v2.1-two-condition-20260801"


def _scenario_ids(prefix: str, count: int) -> list[str]:
    return [f"dryrun_{prefix}_{index:03d}" for index in range(count)]


def build_call_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        experiment: str,
        tier: str,
        scenario_count: int,
        models: list[str],
        repetitions: int,
        branches: list[str],
        stages: list[str],
        prefix: str,
    ) -> None:
        for model in models:
            for scenario in _scenario_ids(prefix, scenario_count):
                for repetition in range(repetitions):
                    for branch in branches:
                        for stage in stages:
                            rows.append({
                                "experiment_id": experiment,
                                "tier": tier,
                                "scenario_id": scenario,
                                "model_condition": model,
                                "repetition": repetition,
                                "branch": branch,
                                "stage": stage,
                                "interface": branch if experiment in {"v2_p0_interface_pilot", "v2_e1_interface_confirmation"} else "I_best",
                                "mock_only": True,
                            })

    add("v2_p0_interface_pilot", "pilot", 8, ["deepseek_formal", "qwen_flash"], 1, ["I0", "I1", "I2", "I3"], ["stage1", "stage2"], "p0")
    # E1 has two interfaces, two stages and three repetitions.
    add("v2_e1_interface_confirmation", "confirmatory_interface", 20, ["deepseek_formal", "qwen_flash"], 3, ["I0", "I_best"], ["stage1", "stage2"], "e1")
    # E2 has one shared stage1 and five stage2 branches. The stage1 rows are represented once per
    # block; stage2 rows carry the branch. This preserves the planned six calls per block.
    for model in ["deepseek_formal", "qwen_flash"]:
        for scenario in _scenario_ids("e2", 60):
            for repetition in range(3):
                rows.append({"experiment_id": "v2_e2_physical_carrier", "tier": "confirmatory_physical_carrier", "scenario_id": scenario, "model_condition": model, "repetition": repetition, "branch": "shared", "stage": "stage1", "interface": "I_best", "mock_only": True})
                for branch in ["C", "M", "S", "K", "C2"]:
                    rows.append({"experiment_id": "v2_e2_physical_carrier", "tier": "confirmatory_physical_carrier", "scenario_id": scenario, "model_condition": model, "repetition": repetition, "branch": branch, "stage": "stage2", "interface": "I_best", "mock_only": True})
    # E3 v2.2: provider-free plan construction uses the exact 20-scenario,
    # four-branch, five-event schedule and its balanced event-level shadows.
    e3_tasks = [{"scenario_id": scenario} for scenario in _scenario_ids("e3_v22", 20)]
    for row in build_e3_v22_plan(e3_tasks):
        row.update({
            "tier": "exploratory_longitudinal_state_authority",
            "interface": "I_best",
            "mock_only": True,
            "repetition": 0,
            "stage": "stage1" if row["branch"] == "shared_initial" else ("shadow" if row["request_role"] == "shadow_duplicate" else "replan"),
        })
        rows.append(row)
    # Bridge: DeepSeek only, shared Stage1 plus two Stage2 branches.
    add("v2_deepseek_preview_formal_bridge", "version_bridge", 20, ["deepseek_formal"], 3, ["shared", "model_carrier", "canonical_carrier"], ["stage1"], "bridge")
    rows = [row for row in rows if row["experiment_id"] != "v2_deepseek_preview_formal_bridge"]
    for scenario in _scenario_ids("bridge", 20):
        for repetition in range(3):
            for branch, stage in [("shared", "stage1"), ("model_carrier", "stage2"), ("canonical_carrier", "stage2")]:
                rows.append({"experiment_id": "v2_deepseek_preview_formal_bridge", "tier": "version_bridge", "scenario_id": scenario, "model_condition": "deepseek_formal", "repetition": repetition, "branch": branch, "stage": stage, "interface": "I0", "mock_only": True})
    for row in rows:
        if row["experiment_id"] != "v2_e3_longitudinal_state_authority":
            row["protocol_version"] = PROTOCOL_VERSION
    rows.sort(key=lambda row: (row["experiment_id"], row["model_condition"], row["scenario_id"], row["repetition"], row["branch"], row["stage"], row.get("event_index", -1)))
    rng = random.Random(20260803)
    rng.shuffle(rows)
    for order, row in enumerate(rows):
        row["call_order"] = order
        row["plan_row_hash"] = sha256_json(row)
    return rows
