"""V2 prompt compiler with compact JSON and exact model-visible schemas."""

from __future__ import annotations

import json
from typing import Any

from .models import PromptBundle, StageName, StateMethodName
from .request import canonical_json, oracle_leak_scan, sha256_text
from .response_v2 import STATE_KEYS, materialize_response_schema_v2

OUTPUT_RULES = [
    "Return exactly one compact JSON object and nothing else.",
    "Do not output Markdown, code fences, reasoning, planning explanation, recommendations, or prose outside JSON.",
    "Do not rename fields, add fields, copy input time series, or attach reason, description, or confidence to dispatch rows.",
    "Use JSON numbers, never numeric strings, NaN, or Infinity. Round battery_action_kw and typed numeric state fields to at most three decimals.",
]


def compile_stage1_prompt_v2(candidate: dict[str, Any], state_method: StateMethodName) -> PromptBundle:
    surface = candidate["model_visible_episode"]
    task_surface = {
        key: surface[key]
        for key in (
            "episode_schema_version",
            "scenario_id",
            "objective",
            "resolution_minutes",
            "horizon_steps",
            "action_schema",
            "stage_1",
        )
    }
    return _build_bundle(
        stage="stage1",
        candidate=candidate,
        state_method=state_method,
        required_length=96,
        task_surface=task_surface,
    )


def compile_stage2_prompt_v2(
    candidate: dict[str, Any],
    state_method: StateMethodName,
    state_carrier_value: dict[str, Any] | str | None,
) -> PromptBundle:
    surface = candidate["model_visible_episode"]
    activation = int(candidate["visible_update"]["activation_step"])
    task_surface = {
        "current_time": surface["stage_2"]["current_time_utc"],
        "visible_update": surface["stage_2"]["visible_update"],
        "state_carrier": {
            "carrier_type": _carrier_type(state_method),
            "read_only": True,
            "value": state_carrier_value,
        },
        "remaining_visible_information": surface["stage_2"]["remaining_timeseries"],
    }
    return _build_bundle(
        stage="stage2",
        candidate=candidate,
        state_method=state_method,
        required_length=96 - activation,
        task_surface=task_surface,
    )


def compile_stage2_prompt_template_v2(
    candidate: dict[str, Any], state_method: StateMethodName
) -> PromptBundle:
    return compile_stage2_prompt_v2(candidate, state_method, "FILLED_AT_RUNTIME")


def _build_bundle(
    *,
    stage: StageName,
    candidate: dict[str, Any],
    state_method: StateMethodName,
    required_length: int,
    task_surface: dict[str, Any],
) -> PromptBundle:
    schema = materialize_response_schema_v2(stage, state_method, required_length)
    prompt = {
        "prompt_schema": f"{stage}_prompt_v2",
        "scenario_id": candidate["scenario_id"],
        "stage": stage,
        "state_method": state_method,
        "task_surface": task_surface,
        "state_output_contract": _state_contract(state_method),
        "dispatch_contract": {
            "required_items": required_length,
            "battery_action_kw_sign": "positive_charge_negative_discharge",
            "maximum_decimal_places": 3,
        },
        "output_rules": OUTPUT_RULES,
        "response_json_schema": schema,
        "forbidden_information_policy": "no_oracle_hidden_future_answer_tool_or_feedback_material",
    }
    rendered = "\n".join([*OUTPUT_RULES, canonical_json(prompt)])
    return PromptBundle(
        stage=stage,
        scenario_id=str(candidate["scenario_id"]),
        state_method=state_method,
        canonical_prompt=prompt,
        rendered_prompt=rendered,
        prompt_sha256=sha256_text(canonical_json(prompt)),
        model_visible_field_manifest=_field_manifest(prompt),
        oracle_leak_scan=oracle_leak_scan(prompt),
    )


def _state_contract(method: StateMethodName) -> dict[str, Any]:
    key = STATE_KEYS[method]
    if key == "typed_state":
        return {
            "output_field": key,
            "schema": "typed_state_v2",
            "free_text_extensions": False,
            "copy_full_canonical_history": False,
        }
    return {
        "output_field": key,
        "type": "string",
        "maximum_utf8_characters": 1200,
        "maximum_lines": 8,
        "allowed_content": "current_effective_operational_state_and_explicitly_revoked_items_only",
        "copy_task_price_load_pv_history_or_dispatch": False,
    }


def _carrier_type(method: StateMethodName) -> str:
    return {
        "rolling_summary": "model_generated_rolling_summary",
        "visible_carry": "deterministic_visible_carry",
        "typed_state": "model_generated_typed_state",
        "canonical_typed_carry": "canonical_typed_carry",
    }[method]


def _field_manifest(value: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            fields.append(path)
            fields.extend(_field_manifest(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            fields.extend(_field_manifest(item, f"{prefix}[{index}]"))
    return fields


def compact_json_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
