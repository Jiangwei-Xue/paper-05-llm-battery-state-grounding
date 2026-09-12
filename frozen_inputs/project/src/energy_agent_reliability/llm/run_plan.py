"""Pilot run-plan generation from frozen pilot scenarios only."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from energy_agent_reliability.provenance import read_jsonl, sha256_file, write_jsonl
from energy_agent_reliability.state_methods import STATE_METHODS

from .models import ModelCondition, PilotRunPlanRow, StateMethodName
from .request import compile_stage1_prompt, compile_stage2_prompt_template, stable_id

PILOT_RUN_ORDER_SEED = 20260717
PLACEHOLDER_CONDITION_IDS = ("model_condition_A", "model_condition_B", "model_condition_C")


def load_model_conditions(path: str | Path) -> list[ModelCondition]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    conditions: list[ModelCondition] = []
    for item in raw["model_conditions"]:
        conditions.append(
            ModelCondition(
                condition_id=str(item["condition_id"]),
                provider=str(item["provider"]),
                api_route=str(item["api_route"]),
                requested_model=str(item["requested_model"]),
                expected_returned_model=str(item["expected_returned_model"]),
                model_family=str(item["model_family"]),
                route_type=str(item["route_type"]),
                structured_output_support=str(item["structured_output_support"]),
                temperature_control=str(item["temperature_control"]),
                reasoning_control=str(item["reasoning_control"]),
                tool_and_web_control=str(item["tool_and_web_control"]),
                fallback_control=str(item["fallback_control"]),
                timeout_seconds=int(item["timeout_seconds"]),
                retry_policy=str(item["retry_policy"]),
                max_input_tokens=_optional_int(item["max_input_tokens"]),
                max_output_tokens=_optional_int(item["max_output_tokens"]),
                pricing_snapshot_source=str(item["pricing_snapshot_source"]),
                pricing_retrieval_utc=item["pricing_retrieval_utc"],
                data_use_boundary=str(item["data_use_boundary"]),
                admission_status=str(item["admission_status"]),
                exclusion_reason=item["exclusion_reason"],
            )
        )
    return conditions


def build_pilot_run_plan(
    pilot_path: str | Path = "scenarios/frozen_pilot_tasks.jsonl",
    conditions_path: str | Path = "protocol/MODEL_CONDITIONS_DRAFT_V1.yaml",
) -> list[dict[str, Any]]:
    pilots = list(read_jsonl(pilot_path))
    conditions = load_model_conditions(conditions_path)
    if len(pilots) != 10:
        raise ValueError("Pilot plan requires exactly 10 frozen pilot scenarios.")
    if [condition.condition_id for condition in conditions] != list(PLACEHOLDER_CONDITION_IDS):
        raise ValueError("Pilot plan expects exactly three placeholder model conditions.")
    rows: list[dict[str, Any]] = []
    for scenario_index, pilot in enumerate(sorted(pilots, key=lambda item: item["scenario_id"])):
        for method_index, method in enumerate(STATE_METHODS):
            for condition_index, condition in enumerate(conditions):
                typed_method = _as_state_method(method)
                stage1 = compile_stage1_prompt(pilot, typed_method)
                stage2 = compile_stage2_prompt_template(pilot, typed_method)
                identity = {
                    "scenario_id": pilot["scenario_id"],
                    "condition_id": condition.condition_id,
                    "state_method": method,
                    "repetition": 0,
                }
                episode_id = stable_id("episode", identity)
                run_id = stable_id("run", {**identity, "pilot_run_order_seed": PILOT_RUN_ORDER_SEED})
                plan_row = PilotRunPlanRow(
                    scenario_id=str(pilot["scenario_id"]),
                    episode_id=episode_id,
                    run_id=run_id,
                    model_condition_id=condition.condition_id,
                    state_method=typed_method,
                    repetition=0,
                    frozen_task_hash=_task_hash(pilot),
                    stage1_prompt_sha256=stage1.prompt_sha256,
                    stage2_prompt_template_sha256=stage2.prompt_sha256,
                    planned_primary_calls=2,
                    run_order=scenario_index * len(STATE_METHODS) * len(conditions)
                    + method_index * len(conditions)
                    + condition_index,
                )
                rows.append(plan_row.__dict__)
    rng = random.Random(PILOT_RUN_ORDER_SEED)
    grouped = _balanced_shuffle(rows, rng)
    for order, row in enumerate(grouped):
        row["run_order"] = order
    return grouped


def write_pilot_run_plan(plan: list[dict[str, Any]], destination: str | Path) -> None:
    write_jsonl(destination, plan)


def pilot_plan_report(plan: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_ids = {str(row["scenario_id"]) for row in plan}
    condition_ids = {str(row["model_condition_id"]) for row in plan}
    methods = {str(row["state_method"]) for row in plan}
    return {
        "status": "pass",
        "pilot_run_order_seed": PILOT_RUN_ORDER_SEED,
        "episodes": len(plan),
        "planned_primary_calls": sum(int(row["planned_primary_calls"]) for row in plan),
        "pilot_scenarios_covered": len(scenario_ids),
        "model_condition_placeholders": sorted(condition_ids),
        "state_methods_covered": sorted(methods),
        "main_tasks_exposed": 0,
        "llm_calls": 0,
    }


def _balanced_shuffle(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["state_method"]), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result: list[dict[str, Any]] = []
    while any(buckets.values()):
        for method in STATE_METHODS:
            bucket = buckets[str(method)]
            if bucket:
                result.append(bucket.pop(0))
    return result


def _task_hash(task: dict[str, Any]) -> str:
    return stable_id("taskhash", task).removeprefix("taskhash_")


def _optional_int(value: Any) -> int | None:
    if value in (None, "unknown"):
        return None
    return int(value)


def _as_state_method(value: str) -> StateMethodName:
    if value not in STATE_METHODS:
        raise ValueError(f"Unknown method: {value}")
    return value


def write_pilot_plan_hash(plan_path: str | Path, destination: str | Path) -> None:
    digest = sha256_file(plan_path)
    Path(destination).write_text(f"{digest}  {Path(plan_path).as_posix()}\n", encoding="utf-8")
