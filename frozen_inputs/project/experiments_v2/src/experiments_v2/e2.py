"""Frozen E2 plan and carrier construction contracts."""

from __future__ import annotations

import copy
import random
from typing import Any

from .carriers import assert_mutation_scope, mutate_carrier
from .hashing import sha256_json
from .p0 import PROTOCOL_VERSION, canonical_event_state, canonical_initial_state

EXPERIMENT_ID = "v2_e2_physical_carrier"
RUN_ORDER_SEED = 20260803
MODEL_CONDITIONS = ("deepseek_formal", "qwen_flash")
BRANCHES = ("C", "M", "S", "K", "C2")
INTERFACE = "I3"


def build_e2_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build 360 shared-Stage-1 blocks and their 2,160-call budget."""
    if len(tasks) != 60 or len({str(task["scenario_id"]) for task in tasks}) != 60:
        raise ValueError("E2 requires sixty unique frozen tasks")
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        for model in MODEL_CONDITIONS:
            for repetition in range(3):
                identity = {
                    "protocol_version": PROTOCOL_VERSION,
                    "experiment_id": EXPERIMENT_ID,
                    "scenario_id": task["scenario_id"],
                    "model_condition": model,
                    "interface": INTERFACE,
                    "repetition": repetition,
                }
                rows.append(
                    {
                        **identity,
                        "block_id": f"e2_{sha256_json(identity)[:20]}",
                        "task_sha256": sha256_json(task),
                        "stage1_shared": True,
                        "stage2_branches": list(BRANCHES),
                        "planned_stage_calls": 6,
                        "excluded_from_claim_bearing_analysis": False,
                    }
                )
    random.Random(RUN_ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
        row["plan_row_sha256"] = sha256_json(row)
    return rows


def e2_states(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return frozen deterministic carriers; M is supplied from Stage 1."""
    intervention = task["v2_prefix_intervention"]
    canonical = canonical_event_state(task, float(intervention["canonical_event_soc_kwh"]))
    initial = canonical_initial_state(task)
    stale_soc = mutate_carrier(
        canonical,
        "S",
        stale_values={"physical_field": "soc_kwh", "soc_kwh": intervention["stale_event_soc_kwh"]},
    )
    stale_metadata = mutate_carrier(
        canonical,
        "K",
        stale_values={
            "active_commitments": copy.deepcopy(initial["active_commitments"]),
            "revoked_commitments": copy.deepcopy(initial["revoked_commitments"]),
            "sequence": initial["sequence"],
        },
    )
    canonical2 = mutate_carrier(canonical, "C2")
    assert_mutation_scope(canonical, canonical2, "C2")
    assert_mutation_scope(canonical, stale_soc, "S")
    assert_mutation_scope(canonical, stale_metadata, "K")
    return {
        "initial": initial,
        "C": canonical,
        "S": stale_soc,
        "K": stale_metadata,
        "C2": canonical2,
    }


def model_carrier(stage1_state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Preserve the parsed model carrier exactly; do not repair missing fields."""
    if stage1_state is None:
        return None
    return copy.deepcopy(stage1_state)
