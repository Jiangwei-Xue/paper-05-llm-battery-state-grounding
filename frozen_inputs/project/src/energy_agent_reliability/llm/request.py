"""Prompt compiler and request payload construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from energy_agent_reliability.state_methods import STATE_METHODS

from .models import ModelCondition, PromptBundle, ProviderRequest, StageName, StateMethodName

FORBIDDEN_PROMPT_TOKENS = {
    "scorer_oracle",
    "future_actual_pv",
    "perfect_information_solution",
    "mpc_future_optimal_actions",
    "correct_dispatch_answer",
    "stage_score_feedback",
    "other_model_outputs",
    "pre_event_capacity_kwh",
    "pre_event_max_charge_kw",
    "pre_event_max_discharge_kw",
    "pre_event_export_limit_kw",
    "pre_event_reserve_soc_kwh",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(prefix: str, payload: dict[str, Any], length: int = 20) -> str:
    return f"{prefix}_{sha256_text(canonical_json(payload))[:length]}"


def compile_stage1_prompt(candidate: dict[str, Any], state_method: StateMethodName) -> PromptBundle:
    _validate_method(state_method)
    surface = candidate["model_visible_episode"]
    stage1_surface = {
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
    canonical_prompt = {
        "prompt_schema": "stage1_prompt_v1",
        "scenario_id": candidate["scenario_id"],
        "stage": "stage1",
        "state_method": state_method,
        "task_surface": stage1_surface,
        "state_output_requirement": _state_requirement(state_method, "stage1"),
        "response_contract": _response_contract("stage1", required_length=96, first_t=0),
        "dispatch_output_requirement": {
            "schema": "dispatch_plan_v1",
            "required_length": 96,
            "battery_action_kw_sign": "positive_charge_negative_discharge",
        },
        "forbidden_information_policy": "no_oracle_hidden_future_answer_tool_or_feedback_material",
    }
    return _bundle("stage1", candidate["scenario_id"], state_method, canonical_prompt)


def compile_stage2_prompt_template(candidate: dict[str, Any], state_method: StateMethodName) -> PromptBundle:
    _validate_method(state_method)
    surface = candidate["model_visible_episode"]
    activation = int(candidate["visible_update"]["activation_step"])
    carrier = _stage2_carrier_template(state_method)
    canonical_prompt = {
        "prompt_schema": "stage2_prompt_template_v1",
        "scenario_id": candidate["scenario_id"],
        "stage": "stage2",
        "state_method": state_method,
        "current_time": surface["stage_2"]["current_time_utc"],
        "visible_update": surface["stage_2"]["visible_update"],
        "state_carrier": carrier,
        "remaining_visible_information": surface["stage_2"]["remaining_timeseries"],
        "state_output_requirement": _state_requirement(state_method, "stage2"),
        "response_contract": _response_contract(
            "stage2",
            required_length=96 - activation,
            first_t=activation,
        ),
        "dispatch_output_requirement": {
            "schema": "dispatch_plan_v1",
            "required_length": 96 - activation,
            "battery_action_kw_sign": "positive_charge_negative_discharge",
        },
        "forbidden_information_policy": "no_oracle_hidden_future_answer_tool_or_feedback_material",
    }
    return _bundle("stage2", candidate["scenario_id"], state_method, canonical_prompt)


def build_provider_request(
    *,
    bundle: PromptBundle,
    run_id: str,
    condition: ModelCondition,
) -> ProviderRequest:
    payload = {
        "model": condition.requested_model,
        "stream": False,
        "tools": [],
        "tool_choice": "none",
        "temperature": 0,
        "messages": [{"role": "user", "content": bundle.rendered_prompt}],
        "response_format": {"type": "json_object"},
        "metadata": {
            "run_id": run_id,
            "condition_id": condition.condition_id,
            "prompt_sha256": bundle.prompt_sha256,
        },
    }
    encoded = canonical_json(payload)
    return ProviderRequest(
        stage=bundle.stage,
        run_id=run_id,
        condition_id=condition.condition_id,
        requested_model=condition.requested_model,
        payload=payload,
        payload_sha256=sha256_text(encoded),
        timeout_seconds=condition.timeout_seconds,
    )


def oracle_leak_scan(value: Any) -> dict[str, Any]:
    payload = canonical_json(value).lower()
    leaked = sorted(token for token in FORBIDDEN_PROMPT_TOKENS if token in payload)
    policy_mentions = {"forbidden"}
    filtered = [token for token in leaked if token not in policy_mentions]
    return {"passed": not filtered, "matches": filtered}


def _bundle(
    stage: StageName, scenario_id: str, state_method: StateMethodName, canonical_prompt: dict[str, Any]
) -> PromptBundle:
    rendered = _render_prompt(canonical_prompt)
    prompt_hash = sha256_text(canonical_json(canonical_prompt))
    manifest = _field_manifest(canonical_prompt)
    scan_target = {key: value for key, value in canonical_prompt.items() if key != "forbidden"}
    return PromptBundle(
        stage=stage,
        scenario_id=scenario_id,
        state_method=state_method,
        canonical_prompt=canonical_prompt,
        rendered_prompt=rendered,
        prompt_sha256=prompt_hash,
        model_visible_field_manifest=manifest,
        oracle_leak_scan=oracle_leak_scan(scan_target),
    )


def _render_prompt(canonical_prompt: dict[str, Any]) -> str:
    return (
        "Return only JSON matching the frozen schema.\n"
        "Do not use tools, web, code execution, hidden information, oracle material, answer keys, or feedback.\n"
        f"{json.dumps(canonical_prompt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)}"
    )


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


def _state_requirement(state_method: StateMethodName, stage: StageName) -> dict[str, Any]:
    output_field = "carried_state" if stage == "stage1" else "updated_state"
    if state_method in {"typed_state", "canonical_typed_carry"}:
        return {
            "stage": stage,
            "state_method": state_method,
            "output_field": output_field,
            "schema": "typed_state_v1",
            "type": "object",
            "required_fields": [
                "current_time",
                "soc_kwh",
                "usable_capacity_kwh",
                "max_charge_kw",
                "max_discharge_kw",
                "reserve_soc_kwh",
                "export_limit_kw",
                "active_forecast_version",
                "active_commitments",
                "revoked_or_superseded_items",
            ],
            "carrier_wrapper_forbidden": True,
            "instruction": (
                "Place the typed-state fields directly inside the output field. "
                "Do not return a carrier_type/value wrapper."
            ),
        }
    return {
        "stage": stage,
        "state_method": state_method,
        "output_field": output_field,
        "schema": "non_empty_natural_language_state",
        "type": "string",
        "carrier_wrapper_forbidden": True,
        "instruction": "Return one non-empty natural-language state string directly in the output field.",
    }


def _response_contract(stage: StageName, *, required_length: int, first_t: int) -> dict[str, Any]:
    state_field = "carried_state" if stage == "stage1" else "updated_state"
    dispatch_field = "dispatch_plan" if stage == "stage1" else "revised_dispatch_plan"
    return {
        "format": "JSON",
        "top_level_type": "object",
        "top_level_required_fields": [state_field, dispatch_field],
        "top_level_additional_properties": False,
        "dispatch_field": dispatch_field,
        "dispatch_length": required_length,
        "dispatch_t_range": [first_t, first_t + required_length - 1],
        "dispatch_item_required_fields": ["t", "timestamp_utc", "battery_action_kw"],
        "dispatch_item_additional_properties": False,
        "dispatch_t_must_match_visible_row": True,
        "dispatch_timestamp_must_match_visible_row": True,
        "prose_outside_json_forbidden": True,
    }


def _stage2_carrier_template(state_method: StateMethodName) -> dict[str, Any]:
    if state_method == "rolling_summary":
        return {"carrier_type": "model_generated_rolling_summary", "value": "FILLED_FROM_STAGE1_OUTPUT_AT_RUNTIME"}
    if state_method == "visible_carry":
        return {"carrier_type": "deterministic_visible_carry", "value": "FILLED_FROM_VISIBLE_HISTORY_AT_RUNTIME"}
    if state_method == "typed_state":
        return {"carrier_type": "model_generated_typed_state", "value": "FILLED_FROM_STAGE1_OUTPUT_AT_RUNTIME"}
    return {"carrier_type": "canonical_typed_carry", "value": "FILLED_FROM_DETERMINISTIC_ENVIRONMENT_AT_RUNTIME"}


def _validate_method(state_method: StateMethodName) -> None:
    if state_method not in STATE_METHODS:
        raise ValueError(f"Unknown state method: {state_method}")
