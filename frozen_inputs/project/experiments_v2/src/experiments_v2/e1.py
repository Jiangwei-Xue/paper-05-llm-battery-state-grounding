"""Frozen plan construction for the E1 interface-confirmation tier."""

from __future__ import annotations

import random
from typing import Any

from .hashing import sha256_json

PROTOCOL_VERSION = "energybench-v2.1-two-condition-20260801"
EXPERIMENT_ID = "v2_e1_interface_confirmation"
RUN_ORDER_SEED = 20260803
MODEL_CONDITIONS = ("deepseek_formal", "qwen_flash")
INTERFACES = ("I0", "I3")


def build_e1_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the balanced 240-episode E1 plan in frozen pseudorandom order."""
    if len(tasks) != 20 or len({str(task["scenario_id"]) for task in tasks}) != 20:
        raise ValueError("E1 requires twenty unique frozen tasks")
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        for model in MODEL_CONDITIONS:
            for interface in INTERFACES:
                for repetition in range(3):
                    identity = {
                        "protocol_version": PROTOCOL_VERSION,
                        "experiment_id": EXPERIMENT_ID,
                        "scenario_id": task["scenario_id"],
                        "model_condition": model,
                        "interface": interface,
                        "repetition": repetition,
                    }
                    rows.append(
                        {
                            **identity,
                            "episode_id": f"e1_{sha256_json(identity)[:20]}",
                            "task_sha256": sha256_json(task),
                            "planned_stage_calls": 2,
                            "excluded_from_claim_bearing_analysis": False,
                        }
                    )
    random.Random(RUN_ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
        row["plan_row_sha256"] = sha256_json(row)
    return rows
