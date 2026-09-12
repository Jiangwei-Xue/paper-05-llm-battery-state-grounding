"""Provider-free E3 v2.2 plan and balanced event-level null schedule."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .hashing import sha256_json

E3_PROTOCOL_VERSION = "energybench-v2.2-e3-20260801"
E3_EXPERIMENT_ID = "v2_e3_longitudinal_state_authority"
E3_RUN_ORDER_SEED = 20260803
E3_SHADOW_SEED = 20260804
E3_BRANCHES = ("P_SYS__C_SYS", "P_SYS__C_MDL", "P_MDL__C_SYS", "P_MDL__C_MDL")
E3_EVENT_COUNT = 5


def _identity(task: dict[str, Any], model: str, event_index: int | None, branch: str, role: str) -> dict[str, Any]:
    return {
        "protocol_version": E3_PROTOCOL_VERSION,
        "experiment_id": E3_EXPERIMENT_ID,
        "scenario_id": str(task["scenario_id"]),
        "model_condition": model,
        "event_index": event_index,
        "branch": branch,
        "request_role": role,
    }


def _pair_id(task: dict[str, Any], model: str, event_index: int, branch: str) -> str:
    """Stable identifier shared by the executed and non-executed pair members."""
    return f"null_{sha256_json({'protocol_version': E3_PROTOCOL_VERSION, 'experiment_id': E3_EXPERIMENT_ID, 'scenario_id': str(task['scenario_id']), 'model_condition': model, 'event_index': event_index, 'branch': branch})[:20]}"


def _shadow_schedule(tasks: list[dict[str, Any]], models: tuple[str, ...]) -> dict[tuple[str, str, int, str], str]:
    """Assign exactly five scenarios per branch at every event for each model."""
    ordered = sorted(tasks, key=lambda item: str(item["scenario_id"]))
    schedule: dict[tuple[str, str, int, str], str] = {}
    for model in models:
        for event_index in range(E3_EVENT_COUNT):
            shuffled = ordered.copy()
            model_offset = int(sha256_json(model)[:8], 16) % 997
            random.Random(E3_SHADOW_SEED + event_index + model_offset).shuffle(shuffled)
            for branch, selected in zip(E3_BRANCHES, [shuffled[i * 5 : (i + 1) * 5] for i in range(4)], strict=True):
                for task in selected:
                    schedule[(model, str(task["scenario_id"]), event_index, branch)] = "duplicate"
    return schedule


def build_e3_v22_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build 40 blocks, 800 branch calls, and 200 balanced shadow calls."""
    if len(tasks) != 20 or len({str(task["scenario_id"]) for task in tasks}) != 20:
        raise ValueError("E3 v2.2 requires exactly twenty unique scenarios")
    models = ("deepseek_formal", "qwen_flash")
    schedule = _shadow_schedule(tasks, models)
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        for model in models:
            initial = _identity(task, model, None, "shared_initial", "primary")
            initial["call_id"] = f"e3_{sha256_json(initial)[:20]}"
            initial["task_sha256"] = sha256_json(task)
            initial["planned_execution"] = "shared_initial"
            rows.append(initial)
            for event_index in range(E3_EVENT_COUNT):
                for branch in E3_BRANCHES:
                    identity = _identity(task, model, event_index, branch, "primary")
                    identity["call_id"] = f"e3_{sha256_json(identity)[:20]}"
                    identity["task_sha256"] = sha256_json(task)
                    identity["planned_execution"] = "executed_candidate"
                    pair_key = (model, str(task["scenario_id"]), event_index, branch)
                    if schedule.get(pair_key) == "duplicate":
                        identity["shadow_pair_id"] = _pair_id(task, model, event_index, branch)
                    rows.append(identity)
                duplicate_branch = next(
                    branch for branch in E3_BRANCHES if schedule.get((model, str(task["scenario_id"]), event_index, branch)) == "duplicate"
                )
                shadow = _identity(task, model, event_index, duplicate_branch, "shadow_duplicate")
                shadow["call_id"] = f"e3_{sha256_json(shadow)[:20]}"
                shadow["task_sha256"] = sha256_json(task)
                shadow["duplicate_of_branch"] = duplicate_branch
                shadow["shadow_pair_id"] = _pair_id(task, model, event_index, duplicate_branch)
                rows.append(shadow)

    # Balance which member is executed within each model without using outcomes.
    pairs = sorted(
        [row for row in rows if row["request_role"] == "shadow_duplicate"],
        key=lambda row: (str(row["model_condition"]), str(row["shadow_pair_id"])),
    )
    for model in models:
        model_pairs = [row for row in pairs if row["model_condition"] == model]
        shuffled = model_pairs.copy()
        random.Random(E3_SHADOW_SEED + len(model)).shuffle(shuffled)
        duplicate_executed = {str(row["shadow_pair_id"]): index % 2 == 1 for index, row in enumerate(shuffled)}
        for row in rows:
            pair_id = row.get("shadow_pair_id")
            if pair_id is not None and row["model_condition"] == model:
                row["executed_member"] = "duplicate" if duplicate_executed[str(pair_id)] else "primary"

    rng = random.Random(E3_RUN_ORDER_SEED)
    rng.shuffle(rows)
    for order, row in enumerate(rows):
        row["run_order"] = order
        row["plan_row_sha256"] = sha256_json(row)
    return rows


def validate_e3_v22_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["request_role"] == "primary"]
    shadow = [row for row in rows if row["request_role"] == "shadow_duplicate"]
    initial = [row for row in rows if row["request_role"] == "primary" and row["branch"] == "shared_initial"]
    counts = Counter((row["model_condition"], row["request_role"]) for row in rows)
    branch_counts = Counter((row.get("model_condition"), row.get("event_index"), row.get("duplicate_of_branch")) for row in shadow)
    return {
        "status": "pass" if len(rows) == 1040 and len(primary) == 840 and len(shadow) == 200 and len(initial) == 40 else "fail",
        "planned_calls": len(rows),
        "initial_calls": len(initial),
        "primary_calls": len(primary),
        "shadow_calls": len(shadow),
        "counts_by_model_role": {f"{model}:{role}": count for (model, role), count in sorted(counts.items())},
        "shadow_balance_by_model_event_branch": {"|".join(map(str, key)): value for key, value in sorted(branch_counts.items())},
        "unique_call_ids": len({row["call_id"] for row in rows}) == len(rows),
        "provider_calls_performed": 0,
    }
