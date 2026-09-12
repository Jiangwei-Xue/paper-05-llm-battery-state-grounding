"""Returned-model, fallback, reasoning, and tool-use validation."""

from __future__ import annotations

from typing import Any

from .models import ModelCondition, ProviderResponse


def validate_route_response(condition: ModelCondition, response: ProviderResponse) -> dict[str, Any]:
    metadata = response.metadata
    checks = {
        "returned_model_matches": response.returned_model == condition.expected_returned_model
        or condition.expected_returned_model == "TBD",
        "fallback_not_detected": not bool(metadata.get("fallback_detected")),
        "reasoning_not_requested_or_disabled": not bool(metadata.get("reasoning_enabled")),
        "tool_call_not_detected": not bool(metadata.get("tool_calls")),
    }
    protocol_exclusion = None
    if not checks["returned_model_matches"]:
        protocol_exclusion = "wrong_model_route"
    elif not checks["fallback_not_detected"]:
        protocol_exclusion = "fallback"
    elif not checks["tool_call_not_detected"]:
        protocol_exclusion = "disabled_tool_used"
    return {"passed": all(checks.values()), "checks": checks, "protocol_exclusion": protocol_exclusion}

