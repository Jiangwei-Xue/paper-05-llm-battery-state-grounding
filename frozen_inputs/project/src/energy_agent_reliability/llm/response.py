"""Strict response parsing for frozen stage schemas."""

from __future__ import annotations

import json
import math
from typing import Any

from .models import ParserResult, StageName


def parse_stage_response(
    raw_text: str,
    stage: StageName,
    required_dispatch_length: int,
    *,
    expected_dispatch_rows: list[dict[str, Any]] | None = None,
) -> ParserResult:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ParserResult(False, None, [], [f"malformed_json:{exc.msg}"])
    if not isinstance(payload, dict):
        return ParserResult(False, None, [], ["top_level_not_object"])
    if _contains_nonfinite(payload):
        return ParserResult(False, None, [], ["nonfinite_number"])
    state_key = "carried_state" if stage == "stage1" else "updated_state"
    dispatch_key = "dispatch_plan" if stage == "stage1" else "revised_dispatch_plan"
    if state_key not in payload or dispatch_key not in payload:
        return ParserResult(False, payload.get(state_key), [], ["missing_required_field"])
    if set(payload) != {state_key, dispatch_key}:
        return ParserResult(False, payload.get(state_key), [], ["top_level_bad_fields"])
    state = payload[state_key]
    if state in ("", {}, [], None):
        return ParserResult(False, state, [], ["empty_state"])
    dispatch = payload[dispatch_key]
    if not isinstance(dispatch, list):
        return ParserResult(False, state, [], ["dispatch_not_array"])
    if len(dispatch) != required_dispatch_length:
        return ParserResult(False, state, [], ["invalid_dispatch_length"])
    actions: list[float] = []
    for index, item in enumerate(dispatch):
        if not isinstance(item, dict):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_not_object"])
        if set(item) != {"t", "timestamp_utc", "battery_action_kw"}:
            return ParserResult(False, state, [], [f"dispatch_item_{index}_bad_fields"])
        if not isinstance(item["t"], int) or isinstance(item["t"], bool):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_bad_t"])
        if not isinstance(item["timestamp_utc"], str) or not item["timestamp_utc"]:
            return ParserResult(False, state, [], [f"dispatch_item_{index}_bad_timestamp"])
        if expected_dispatch_rows is not None:
            if len(expected_dispatch_rows) != required_dispatch_length:
                raise ValueError("Expected dispatch rows must match the required dispatch length.")
            expected = expected_dispatch_rows[index]
            if item["t"] != expected.get("t"):
                return ParserResult(False, state, [], [f"dispatch_item_{index}_wrong_t"])
            if item["timestamp_utc"] != expected.get("timestamp_utc"):
                return ParserResult(False, state, [], [f"dispatch_item_{index}_wrong_timestamp"])
        action = item["battery_action_kw"]
        if not isinstance(action, int | float) or not math.isfinite(float(action)):
            return ParserResult(False, state, [], [f"dispatch_item_{index}_bad_action"])
        actions.append(float(action))
    return ParserResult(True, state, actions, [])


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False
