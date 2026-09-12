"""V3 route identity semantics.

Route identity is evaluated only for provider requests that were actually
attempted. A Stage 2 skip after a Stage 1 model-output contract failure is a
stage execution outcome, not a provider routing failure.
"""

from __future__ import annotations

from typing import Any, Literal

from .models import ParserResult, ProviderRequest
from .protocol_v2 import ProviderSpecV2

ProviderRouteStatus = Literal[
    "pass",
    "fail",
    "unavailable_metadata",
    "no_response_transport_failure",
    "not_applicable",
]
StageExecutionStatus = Literal[
    "attempted_completed",
    "attempted_transport_failed",
    "skipped_due_to_stage1_contract_failure",
    "skipped_due_to_protocol_exclusion",
    "not_applicable",
]


def provider_route_identity_v3(
    spec: ProviderSpecV2,
    response: dict[str, Any],
    request: ProviderRequest | None,
) -> dict[str, Any]:
    """Validate route identity for one actual provider request."""
    if request is None or response.get("status") == "skipped":
        return {
            "attempted": False,
            "status": "not_applicable",
            "passed": True,
            "protocol_exclusion": None,
            "checks": {},
            "reason": response.get("error_type") or "not_attempted",
        }

    attempts = list(response.get("attempts", []))
    payload_hash_matches = bool(attempts) and all(
        attempt.get("payload_sha256") == request.payload_sha256 for attempt in attempts
    )
    provider_status_ok = response.get("status") == "ok"
    returned_model = response.get("returned_model")
    checks = {
        "provider_status_ok": provider_status_ok,
        "returned_model_metadata_present": bool(returned_model),
        "returned_model_matches": returned_model == spec.expected_returned_model,
        "endpoint_matches": True,
        "payload_hash_matches": payload_hash_matches,
        "reasoning_content_absent": not bool(response.get("reasoning_content_present")),
        "tool_calls_absent": not bool(response.get("tool_calls_present")),
        "fallback_absent": not bool(response.get("fallback_detected")),
    }
    status: ProviderRouteStatus = "pass"
    exclusion = None
    if not provider_status_ok:
        status = "no_response_transport_failure"
        exclusion = "provider_route_no_response"
    elif not checks["returned_model_metadata_present"]:
        status = "unavailable_metadata"
        exclusion = "route_metadata_unavailable"
    elif not checks["returned_model_matches"]:
        status = "fail"
        exclusion = "wrong_model_route"
    elif not checks["endpoint_matches"]:
        status = "fail"
        exclusion = "wrong_endpoint"
    elif not checks["payload_hash_matches"]:
        status = "fail"
        exclusion = "request_payload_hash_mismatch"
    elif not checks["reasoning_content_absent"]:
        status = "fail"
        exclusion = "reasoning_control_violation"
    elif not checks["tool_calls_absent"]:
        status = "fail"
        exclusion = "disabled_tool_used"
    elif not checks["fallback_absent"]:
        status = "fail"
        exclusion = "fallback"
    return {
        "attempted": True,
        "status": status,
        "passed": status == "pass",
        "protocol_exclusion": exclusion,
        "requested_model": spec.requested_model,
        "expected_returned_model": spec.expected_returned_model,
        "returned_model": returned_model,
        "endpoint": spec.endpoint,
        "payload_sha256": request.payload_sha256,
        "attempt_count": len(attempts),
        "checks": checks,
    }


def stage_execution_status_v3(
    response: dict[str, Any],
    parse: ParserResult,
    *,
    skipped_due_to_stage1_failure: bool = False,
    skipped_due_to_protocol_exclusion: bool = False,
) -> StageExecutionStatus:
    if response.get("status") == "skipped":
        if skipped_due_to_stage1_failure:
            return "skipped_due_to_stage1_contract_failure"
        if skipped_due_to_protocol_exclusion:
            return "skipped_due_to_protocol_exclusion"
        return "not_applicable"
    if response.get("status") == "ok":
        return "attempted_completed"
    if parse.protocol_exclusion is not None or skipped_due_to_protocol_exclusion:
        return "skipped_due_to_protocol_exclusion"
    return "attempted_transport_failed"


def episode_route_integrity_v3(stage_routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only attempted route identity checks."""
    attempted = [route for route in stage_routes if route.get("attempted") is True]
    failures = [route for route in attempted if route.get("passed") is not True]
    return {
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "attempted_stage_count": len(attempted),
        "failed_stage_count": len(failures),
        "failure_protocol_exclusions": [
            route.get("protocol_exclusion") for route in failures if route.get("protocol_exclusion")
        ],
    }
