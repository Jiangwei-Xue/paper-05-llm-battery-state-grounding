"""V4 route and transport semantics.

V4 separates two facts that V3 conflated:

* observed route identity for actual provider responses;
* missing/no-response transport outcomes after a frozen request attempt.

A no-response transport outcome remains in the experimental denominator as
failure-as-zero. It is not a wrong-model/fallback/route-mismatch exclusion.
"""

from __future__ import annotations

from typing import Any, Literal

from .models import ParserResult, ProviderRequest
from .protocol_v2 import ProviderSpecV2

ProviderRouteStatusV4 = Literal[
    "pass",
    "fail",
    "unavailable_metadata",
    "transport_no_response",
    "not_applicable",
]
StageExecutionStatusV4 = Literal[
    "attempted_completed",
    "attempted_transport_failed",
    "skipped_due_to_stage1_contract_failure",
    "skipped_due_to_protocol_exclusion",
    "not_applicable",
]


def provider_route_identity_v4(
    spec: ProviderSpecV2,
    response: dict[str, Any],
    request: ProviderRequest | None,
) -> dict[str, Any]:
    """Validate route identity when observable; classify transport otherwise."""
    if request is None or response.get("status") == "skipped":
        return {
            "attempted": False,
            "status": "not_applicable",
            "passed": True,
            "protocol_exclusion": None,
            "transport_status": "not_applicable",
            "route_observation": "not_attempted",
            "route_identity_evaluated": False,
            "checks": {},
            "reason": response.get("error_type") or "not_attempted",
        }

    attempts = list(response.get("attempts", []))
    payload_hash_matches = bool(attempts) and all(
        attempt.get("payload_sha256") == request.payload_sha256 for attempt in attempts
    )
    provider_status_ok = response.get("status") == "ok"
    if not provider_status_ok:
        transport_evidence = _transport_evidence(spec, response, request)
        return {
            "attempted": True,
            "status": "transport_no_response",
            "passed": True,
            "protocol_exclusion": None,
            "transport_status": "no_response_after_retries",
            "route_observation": "unobserved_due_to_transport_failure",
            "route_identity_evaluated": False,
            "requested_model": spec.requested_model,
            "expected_returned_model": spec.expected_returned_model,
            "returned_model": response.get("returned_model"),
            "endpoint": spec.endpoint,
            "payload_sha256": request.payload_sha256,
            "attempt_count": len(attempts),
            "final_status": response.get("status"),
            "final_reason": response.get("error_type") or response.get("http_status"),
            "checks": {
                "provider_status_ok": False,
                "request_payload_hash_recorded": bool(request.payload_sha256),
                "payload_hash_matches_all_attempts": payload_hash_matches,
                "transport_evidence_complete": transport_evidence["complete"],
            },
            "transport_evidence": transport_evidence,
        }

    returned_model = response.get("returned_model")
    checks = {
        "provider_status_ok": True,
        "returned_model_metadata_present": bool(returned_model),
        "returned_model_matches": returned_model == spec.expected_returned_model,
        "endpoint_matches": True,
        "payload_hash_matches": payload_hash_matches,
        "reasoning_content_absent": not bool(response.get("reasoning_content_present")),
        "tool_calls_absent": not bool(response.get("tool_calls_present")),
        "fallback_absent": not bool(response.get("fallback_detected")),
    }
    status: ProviderRouteStatusV4 = "pass"
    exclusion = None
    if not checks["returned_model_metadata_present"]:
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
        "transport_status": "response_observed",
        "route_observation": "observed",
        "route_identity_evaluated": True,
        "requested_model": spec.requested_model,
        "expected_returned_model": spec.expected_returned_model,
        "returned_model": returned_model,
        "endpoint": spec.endpoint,
        "payload_sha256": request.payload_sha256,
        "attempt_count": len(attempts),
        "checks": checks,
    }


def stage_execution_status_v4(
    response: dict[str, Any],
    parse: ParserResult,
    *,
    skipped_due_to_stage1_failure: bool = False,
    skipped_due_to_protocol_exclusion: bool = False,
) -> StageExecutionStatusV4:
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


def episode_route_integrity_v4(stage_routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate route identity only where a provider response is observable."""
    attempted = [route for route in stage_routes if route.get("attempted") is True]
    evaluated = [route for route in attempted if route.get("route_identity_evaluated") is True]
    failures = [route for route in evaluated if route.get("passed") is not True]
    no_response = [
        route
        for route in attempted
        if route.get("transport_status") == "no_response_after_retries"
    ]
    return {
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "attempted_stage_count": len(attempted),
        "evaluated_route_stage_count": len(evaluated),
        "transport_no_response_stage_count": len(no_response),
        "failed_stage_count": len(failures),
        "failure_protocol_exclusions": [
            route.get("protocol_exclusion") for route in failures if route.get("protocol_exclusion")
        ],
    }


def _transport_evidence(
    spec: ProviderSpecV2,
    response: dict[str, Any],
    request: ProviderRequest,
) -> dict[str, Any]:
    attempts = list(response.get("attempts", []))
    raw_stage_timing = response.get("stage_timing")
    stage_timing: dict[str, Any] = raw_stage_timing if isinstance(raw_stage_timing, dict) else {}
    attempts_with_required_timing = sum(_attempt_has_required_timing(attempt) for attempt in attempts)
    payload_hashes_match = bool(attempts) and all(
        attempt.get("payload_sha256") == request.payload_sha256 for attempt in attempts
    )
    has_final_reason = response.get("error_type") is not None or response.get("http_status") is not None
    stage_timing_complete = (
        stage_timing.get("clock") == "time.monotonic"
        and stage_timing.get("stage_wall_elapsed_seconds") is not None
        and stage_timing.get("retry_backoff_seconds") is not None
    )
    complete = all(
        (
            bool(request.payload_sha256),
            bool(request.requested_model),
            bool(spec.endpoint),
            bool(attempts),
            attempts_with_required_timing == len(attempts),
            payload_hashes_match,
            stage_timing_complete,
            has_final_reason,
        )
    )
    return {
        "complete": complete,
        "payload_sha256": request.payload_sha256,
        "requested_model": request.requested_model,
        "endpoint": spec.endpoint,
        "attempt_count": len(attempts),
        "attempts_with_required_timing": attempts_with_required_timing,
        "payload_hashes_match": payload_hashes_match,
        "stage_timing_complete": stage_timing_complete,
        "monotonic_stage_duration_seconds": stage_timing.get("stage_wall_elapsed_seconds"),
        "retry_backoff_seconds": stage_timing.get("retry_backoff_seconds"),
        "final_status": response.get("status"),
        "final_error_type": response.get("error_type"),
        "http_status": response.get("http_status"),
        "final_reason": response.get("error_type") or response.get("http_status"),
    }


def _attempt_has_required_timing(attempt: dict[str, Any]) -> bool:
    required = {
        "clock",
        "stage_offset_before_queue_seconds",
        "stage_remaining_before_queue_seconds",
        "queue_wait_seconds",
        "attempt_send_offset_seconds",
        "attempt_timeout_seconds",
        "attempt_elapsed_seconds",
        "attempt_end_offset_seconds",
        "payload_sha256",
    }
    return attempt.get("clock") == "time.monotonic" and required.issubset(attempt)
