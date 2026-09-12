"""Strict V2 JSON-schema parsing with method-specific state boundaries."""

from __future__ import annotations

import copy
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .models import ParserResult, StageName, StateMethodName

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_NATURAL_STATE_CHARS = 1200
MAX_NATURAL_STATE_LINES = 8
STATE_KEYS: dict[StateMethodName, str] = {
    "rolling_summary": "state_summary",
    "visible_carry": "state_update",
    "typed_state": "typed_state",
    "canonical_typed_carry": "typed_state",
}
COPY_MARKERS = {
    "visible_timeseries",
    "remaining_timeseries",
    "dispatch_plan",
    "revised_dispatch_plan",
    "price_usd_per_mwh",
    "load_kw",
    "pv_ac_kw",
}


def materialize_response_schema_v2(
    stage: StageName,
    state_method: StateMethodName,
    required_dispatch_length: int,
) -> dict[str, Any]:
    """Build the exact model-visible schema for one stage and method."""
    if not 1 <= required_dispatch_length <= 96:
        raise ValueError("V2 dispatch length must be between 1 and 96.")
    typed_state = _inline_typed_state_schema(_load_schema("typed_state_v2.json"))
    dispatch_plan = _inline_dispatch_schema(_load_schema("dispatch_plan_v2.json"))
    state_key = STATE_KEYS[state_method]
    state_schema: dict[str, Any]
    if state_key == "typed_state":
        state_schema = _without_schema_metadata(typed_state)
    else:
        state_schema = {"type": "string", "minLength": 1, "maxLength": MAX_NATURAL_STATE_CHARS}
    dispatch_schema = _without_schema_metadata(dispatch_plan)
    dispatch_schema["minItems"] = required_dispatch_length
    dispatch_schema["maxItems"] = required_dispatch_length
    dispatch_key = "dispatch_plan" if stage == "stage1" else "revised_dispatch_plan"
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [state_key, dispatch_key],
        "properties": {state_key: state_schema, dispatch_key: dispatch_schema},
    }


def parse_stage_response_v2(
    raw_text: str,
    stage: StageName,
    state_method: StateMethodName,
    required_dispatch_length: int,
    *,
    expected_dispatch_rows: list[dict[str, Any]],
    finish_reason: str | None = None,
) -> ParserResult:
    if finish_reason == "length":
        return ParserResult(False, None, [], ["finish_reason_length"])
    try:
        payload = json.loads(raw_text, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as exc:
        message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        return ParserResult(False, None, [], [f"malformed_json:{message}"])
    if not isinstance(payload, dict):
        return ParserResult(False, None, [], ["top_level_not_object"])
    schema = materialize_response_schema_v2(stage, state_method, required_dispatch_length)
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        return ParserResult(False, _state_value(payload, state_method), [], [_schema_diagnostic(errors[0])])
    state = _state_value(payload, state_method)
    state_diagnostic = _validate_state_surface(state, state_method)
    if state_diagnostic is not None:
        return ParserResult(False, state, [], [state_diagnostic])
    dispatch_key = "dispatch_plan" if stage == "stage1" else "revised_dispatch_plan"
    dispatch = payload[dispatch_key]
    if len(expected_dispatch_rows) != required_dispatch_length:
        raise ValueError("Expected dispatch rows must match the exact V2 dispatch length.")
    actions: list[float] = []
    for index, (item, expected) in enumerate(zip(dispatch, expected_dispatch_rows, strict=True)):
        if item["t"] != expected.get("t"):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_wrong_t"])
        if item["timestamp_utc"] != expected.get("timestamp_utc"):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_wrong_timestamp"])
        if not _at_most_three_decimals(item["battery_action_kw"]):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_excess_precision"])
        actions.append(float(item["battery_action_kw"]))
    if isinstance(state, dict):
        for key in (
            "soc_kwh",
            "usable_capacity_kwh",
            "max_charge_kw",
            "max_discharge_kw",
            "reserve_soc_kwh",
            "export_limit_kw",
        ):
            if not _at_most_three_decimals(state[key]):
                return ParserResult(False, state, [], [f"typed_state_{key}_excess_precision"])
    return ParserResult(True, state, actions, [])


def _validate_state_surface(state: dict[str, Any] | str | None, method: StateMethodName) -> str | None:
    if method in {"typed_state", "canonical_typed_carry"}:
        return None
    if not isinstance(state, str):
        return "natural_state_not_string"
    if len(state) > MAX_NATURAL_STATE_CHARS:
        return "natural_state_too_long"
    if len(state.splitlines()) > MAX_NATURAL_STATE_LINES:
        return "natural_state_too_many_lines"
    lowered = state.lower()
    if any(marker in lowered for marker in COPY_MARKERS):
        return "natural_state_copies_input_or_dispatch_surface"
    if len(re.findall(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}", lowered)) > 2:
        return "natural_state_copies_timestamp_sequence"
    return None


def _schema_diagnostic(error: Any) -> str:
    path = ".".join(str(item) for item in error.absolute_path) or "root"
    validator = str(error.validator or "schema")
    return f"schema_validation:{path}:{validator}"


def _state_value(payload: dict[str, Any], state_method: StateMethodName) -> dict[str, Any] | str | None:
    value = payload.get(STATE_KEYS[state_method])
    return value if isinstance(value, dict | str) else None


def _at_most_three_decimals(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    decimal_value = Decimal(str(value))
    return decimal_value == decimal_value.quantize(Decimal("0.001"))


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite_number:{value}")


def _load_schema(name: str) -> dict[str, Any]:
    value = json.loads((PROJECT_ROOT / "schemas" / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object schema: {name}")
    return value


def _without_schema_metadata(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in ("$schema", "$id", "title", "description"):
        result.pop(key, None)
    return result


def _inline_typed_state_schema(value: dict[str, Any]) -> dict[str, Any]:
    """Inline local definitions so nested response schemas remain self-contained."""
    result = _without_schema_metadata(value)
    definitions = result.pop("$defs")
    bounded_value = copy.deepcopy(definitions["bounded_value"])
    active_commitment = copy.deepcopy(definitions["active_commitment"])
    revoked_item = copy.deepcopy(definitions["revoked_item"])
    active_commitment["properties"]["value"] = copy.deepcopy(bounded_value)
    revoked_item["properties"]["previous_value"] = copy.deepcopy(bounded_value)
    result["properties"]["active_commitments"]["items"] = active_commitment
    result["properties"]["revoked_or_superseded_items"]["items"] = revoked_item
    return result


def _inline_dispatch_schema(value: dict[str, Any]) -> dict[str, Any]:
    """Inline the dispatch item because the plan is nested under a response key."""
    result = _without_schema_metadata(value)
    definitions = result.pop("$defs")
    result["items"] = copy.deepcopy(definitions["dispatch_item"])
    return result
