"""Provider-neutral controls for excluded pre-main protocol validation."""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from .models import PromptBundle, ProviderRequest, StateMethodName
from .request import canonical_json, sha256_text, stable_id

EXCLUDED_VALIDATION_SEED = 20260717
RETRYABLE_HTTP_STATUS = {429, 502, 503, 504}
BACKOFF_SECONDS = (1.0, 2.0)
EXCLUDED_STATE_METHODS: tuple[StateMethodName, ...] = (
    "rolling_summary",
    "visible_carry",
    "typed_state",
    "canonical_typed_carry",
)


@dataclass(frozen=True)
class DirectProviderSpec:
    condition_id: str
    provider: str
    endpoint: str
    requested_model: str
    expected_returned_model: str
    api_key_envs: tuple[str, ...]
    reasoning_control: dict[str, Any]
    temperature: float = 0.0
    max_output_tokens: int = 8000
    timeout_seconds: int = 240
    max_transport_attempts: int = 3


def excluded_provider_specs() -> tuple[DirectProviderSpec, ...]:
    return (
        DirectProviderSpec(
            condition_id="excluded_deepseek_v4_flash_candidate",
            provider="deepseek",
            endpoint="https://api.deepseek.com/chat/completions",
            requested_model="deepseek-v4-flash",
            expected_returned_model="deepseek-v4-flash",
            api_key_envs=("DEEPSEEK_API_KEY",),
            reasoning_control={"thinking": {"type": "disabled"}},
        ),
        DirectProviderSpec(
            condition_id="excluded_qwen36_flash_candidate",
            provider="qwen",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            requested_model="qwen3.6-flash",
            expected_returned_model="qwen3.6-flash",
            api_key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            reasoning_control={"enable_thinking": False},
        ),
    )


def build_excluded_validation_plan(
    pilots: list[dict[str, Any]],
    specs: tuple[DirectProviderSpec, ...] | None = None,
    *,
    run_order_seed: int = EXCLUDED_VALIDATION_SEED,
) -> list[dict[str, Any]]:
    providers = specs or excluded_provider_specs()
    rows: list[dict[str, Any]] = []
    for candidate in sorted(pilots, key=lambda item: item["scenario_id"]):
        task_hash = sha256_text(canonical_json(candidate))
        for spec in providers:
            for method in EXCLUDED_STATE_METHODS:
                identity = {
                    "scenario_id": candidate["scenario_id"],
                    "condition_id": spec.condition_id,
                    "state_method": method,
                    "repetition": 0,
                    "run_kind": "excluded_protocol_validation",
                }
                rows.append(
                    {
                        **identity,
                        "run_id": stable_id("xpv", identity),
                        "task_sha256": task_hash,
                        "planned_primary_calls": 2,
                        "excluded_from_formal_analysis": True,
                    }
                )
    random.Random(run_order_seed).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
    return rows


def build_direct_provider_request(
    bundle: PromptBundle,
    run_id: str,
    spec: DirectProviderSpec,
) -> ProviderRequest:
    payload: dict[str, Any] = {
        "model": spec.requested_model,
        "stream": False,
        "tools": [],
        "tool_choice": "none",
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
        "messages": [{"role": "user", "content": bundle.rendered_prompt}],
        "response_format": {"type": "json_object"},
        **spec.reasoning_control,
    }
    return ProviderRequest(
        stage=bundle.stage,
        run_id=run_id,
        condition_id=spec.condition_id,
        requested_model=spec.requested_model,
        payload=payload,
        payload_sha256=sha256_text(canonical_json(payload)),
        timeout_seconds=spec.timeout_seconds,
    )


def validate_direct_response(spec: DirectProviderSpec, response: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "provider_status_ok": response.get("status") == "ok",
        "returned_model_matches": response.get("returned_model") == spec.expected_returned_model,
        "reasoning_content_absent": not bool(response.get("reasoning_content_present")),
        "tool_calls_absent": not bool(response.get("tool_calls_present")),
        "fallback_absent": not bool(response.get("fallback_detected")),
    }
    exclusion = None
    if not checks["provider_status_ok"]:
        exclusion = "provider_failure"
    elif not checks["returned_model_matches"]:
        exclusion = "wrong_model_route"
    elif not checks["reasoning_content_absent"]:
        exclusion = "reasoning_control_violation"
    elif not checks["tool_calls_absent"]:
        exclusion = "disabled_tool_used"
    elif not checks["fallback_absent"]:
        exclusion = "fallback"
    return {"passed": all(checks.values()), "checks": checks, "protocol_exclusion": exclusion}


async def complete_with_retry(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    spec: DirectProviderSpec,
    api_key: str,
    request: ProviderRequest,
) -> dict[str, Any]:
    """Execute one direct request with bounded transport-only retries."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] = _skipped_response("no_attempt")
    for attempt_index in range(spec.max_transport_attempts):
        queued_at = time.perf_counter()
        queue_wait_seconds = 0.0
        request_started_at: float | None = None
        try:
            async with semaphore:
                request_started_at = time.perf_counter()
                queue_wait_seconds = request_started_at - queued_at
                async with asyncio.timeout(request.timeout_seconds):
                    response = await client.post(
                        spec.endpoint,
                        headers=headers,
                        json=request.payload,
                        timeout=request.timeout_seconds,
                    )
            raw: Any
            try:
                raw = response.json()
            except ValueError:
                raw = None
            request_elapsed = time.perf_counter() - request_started_at
            final = _provider_response(response.status_code, raw, request_elapsed)
        except (TimeoutError, httpx.TimeoutException) as exc:
            request_elapsed = (
                time.perf_counter() - request_started_at if request_started_at is not None else 0.0
            )
            final = _transport_response("timeout", exc.__class__.__name__, request_elapsed)
        except httpx.HTTPError as exc:
            request_elapsed = (
                time.perf_counter() - request_started_at if request_started_at is not None else 0.0
            )
            final = _transport_response(
                "transport_error",
                exc.__class__.__name__,
                request_elapsed,
            )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "status": final["status"],
                "http_status": final.get("http_status"),
                "error_type": final.get("error_type"),
                "payload_sha256": request.payload_sha256,
                "elapsed_seconds": final.get("elapsed_seconds"),
                "queue_wait_seconds": round(queue_wait_seconds, 6),
                "total_attempt_seconds": round(time.perf_counter() - queued_at, 6),
            }
        )
        retryable = final["status"] in {"timeout", "transport_error"} or final.get(
            "http_status"
        ) in RETRYABLE_HTTP_STATUS
        if final["status"] == "ok" or not retryable or attempt_index == spec.max_transport_attempts - 1:
            break
        await asyncio.sleep(BACKOFF_SECONDS[min(attempt_index, len(BACKOFF_SECONDS) - 1)])
    final["attempts"] = attempts
    return final


def _provider_response(status_code: int, raw: Any, elapsed: float) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "status": "provider_error" if status_code >= 400 else "transport_error",
            "content": "",
            "raw_response": None,
            "usage": {},
            "returned_model": None,
            "http_status": status_code,
            "elapsed_seconds": round(elapsed, 6),
            "error_type": "non_json_provider_response",
            "reasoning_content_present": False,
            "tool_calls_present": False,
            "fallback_detected": False,
        }
    choices = raw.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    return {
        "status": "provider_error" if status_code >= 400 else "ok",
        "content": message.get("content") or "",
        "raw_response": raw,
        "usage": raw.get("usage") or {},
        "returned_model": raw.get("model"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "http_status": status_code,
        "elapsed_seconds": round(elapsed, 6),
        "error_type": "http_error" if status_code >= 400 else None,
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "tool_calls_present": bool(message.get("tool_calls")),
        "fallback_detected": False,
    }


def _transport_response(status: str, error_type: str, elapsed: float) -> dict[str, Any]:
    return {
        "status": status,
        "content": "",
        "raw_response": None,
        "usage": {},
        "returned_model": None,
        "http_status": None,
        "elapsed_seconds": round(elapsed, 6),
        "error_type": error_type,
        "reasoning_content_present": False,
        "tool_calls_present": False,
        "fallback_detected": False,
    }


def _skipped_response(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "content": "",
        "raw_response": None,
        "usage": {},
        "returned_model": None,
        "http_status": None,
        "elapsed_seconds": 0.0,
        "error_type": reason,
        "reasoning_content_present": False,
        "tool_calls_present": False,
        "fallback_detected": False,
        "attempts": [],
    }


def failure_as_zero_outcome(failure_codes: list[str]) -> dict[str, Any]:
    return {
        "parser_success": False,
        "G_state_governance_success": False,
        "A_governed_action_success": False,
        "D_dispatch_execution_success": False,
        "C_feasible_operational_feasibility_success": False,
        "C_economic_operational_economic_success": False,
        "J_feasible": False,
        "J_economic": False,
        "joint_quadrant": "G=0,C=0",
        "normalized_regret_to_mpc": None,
        "failure_accounting": "failure_as_zero",
        "failure_codes": sorted(set(failure_codes)),
    }


def protocol_validation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count_true(path: tuple[str, ...]) -> int:
        total = 0
        for record in records:
            value: Any = record
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            total += value is True
        return total

    provider_rows: list[dict[str, Any]] = []
    for provider in sorted({str(record["provider"]) for record in records}):
        subset = [record for record in records if record["provider"] == provider]
        provider_rows.append(
            {
                "provider": provider,
                "episodes": len(subset),
                "parser_successes": sum(item["outcome"].get("parser_success") is True for item in subset),
                "route_passes": sum(item["route_validation"].get("passed") is True for item in subset),
                "failure_as_zero_records": sum(
                    item["outcome"].get("failure_accounting") == "failure_as_zero" for item in subset
                ),
            }
        )
    method_rows: list[dict[str, Any]] = []
    for method in EXCLUDED_STATE_METHODS:
        subset = [record for record in records if record["state_method"] == method]
        method_rows.append(
            {
                "state_method": method,
                "episodes": len(subset),
                "parser_successes": sum(item["outcome"].get("parser_success") is True for item in subset),
                "state_governance_successes": sum(
                    item["outcome"].get("G_state_governance_success") is True for item in subset
                ),
            }
        )
    failure_counts = Counter(
        code for record in records for code in record.get("outcome", {}).get("failure_codes", [])
    )
    return {
        "episodes": len(records),
        "parser_successes": count_true(("outcome", "parser_success")),
        "route_passes": count_true(("route_validation", "passed")),
        "prompt_leakage_passes": count_true(("information_boundary", "prompt_leakage_passed")),
        "canonical_carry_leakage_passes": count_true(
            ("information_boundary", "canonical_carry_leakage_passed")
        ),
        "baseline_passes": count_true(("baseline", "passed")),
        "retry_payload_hash_constant_passes": count_true(("retry_validation", "payload_hash_constant")),
        "provider_summary": provider_rows,
        "state_method_summary": method_rows,
        "failure_code_counts": [
            {"code": code, "count": count} for code, count in sorted(failure_counts.items())
        ],
    }


def provider_spec_manifest(specs: tuple[DirectProviderSpec, ...] | None = None) -> list[dict[str, Any]]:
    return [
        {
            **asdict(spec),
            "api_key_envs": list(spec.api_key_envs),
            "api_key_value_recorded": False,
        }
        for spec in (specs or excluded_provider_specs())
    ]
