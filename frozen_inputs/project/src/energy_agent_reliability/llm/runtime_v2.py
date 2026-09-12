"""Monotonic hard-deadline transport runtime for protocol V2."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .models import ProviderRequest
from .protocol_v2 import (
    V2_BACKOFF_SECONDS,
    V2_RETRYABLE_HTTP_STATUS,
    ProviderSpecV2,
)

MonotonicClock = Callable[[], float]
AsyncSleep = Callable[[float], Awaitable[None]]


async def complete_stage_v2(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    spec: ProviderSpecV2,
    api_key: str,
    request: ProviderRequest,
    *,
    stage_wall_timeout_seconds: float | None = None,
    absolute_episode_deadline: float | None = None,
    clock: MonotonicClock = time.monotonic,
    sleep: AsyncSleep = asyncio.sleep,
) -> dict[str, Any]:
    stage_limit = float(stage_wall_timeout_seconds or spec.stage_wall_timeout_seconds)
    stage_started = clock()
    stage_deadline = stage_started + stage_limit
    if absolute_episode_deadline is not None:
        stage_deadline = min(stage_deadline, absolute_episode_deadline)
    attempts: list[dict[str, Any]] = []
    backoff_total = 0.0
    final = _empty_response("protocol_deadline", "no_attempt")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt_index in range(spec.max_transport_attempts):
        if attempt_index:
            requested_backoff = V2_BACKOFF_SECONDS[min(attempt_index - 1, len(V2_BACKOFF_SECONDS) - 1)]
            remaining = stage_deadline - clock()
            if remaining <= requested_backoff:
                final = _empty_response("protocol_deadline", "stage_deadline_exhausted_in_backoff")
                break
            before_backoff = clock()
            await sleep(requested_backoff)
            actual_backoff = max(clock() - before_backoff, 0.0)
            backoff_total += actual_backoff

        remaining_before_queue = stage_deadline - clock()
        if remaining_before_queue <= 0:
            final = _empty_response("protocol_deadline", "stage_deadline_exhausted_before_attempt")
            break
        queued_at = clock()
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            "payload_sha256": request.payload_sha256,
            "clock": "time.monotonic",
            "stage_offset_before_queue_seconds": round(queued_at - stage_started, 6),
            "stage_remaining_before_queue_seconds": round(remaining_before_queue, 6),
        }
        try:
            async with semaphore:
                send_started = clock()
                queue_wait = max(send_started - queued_at, 0.0)
                remaining_before_send = stage_deadline - send_started
                attempt_timeout = min(float(spec.attempt_timeout_seconds), remaining_before_send)
                attempt.update(
                    {
                        "queue_wait_seconds": round(queue_wait, 6),
                        "attempt_send_offset_seconds": round(send_started - stage_started, 6),
                        "attempt_timeout_seconds": round(max(attempt_timeout, 0.0), 6),
                    }
                )
                if attempt_timeout <= 0:
                    final = _empty_response("protocol_deadline", "stage_deadline_exhausted_in_queue")
                    attempt.update(_attempt_end(final, clock(), stage_started, send_started))
                    attempts.append(attempt)
                    break
                try:
                    async with asyncio.timeout(attempt_timeout):
                        response = await client.post(
                            spec.endpoint,
                            headers=headers,
                            json=request.payload,
                            timeout=attempt_timeout,
                        )
                    raw: Any
                    try:
                        raw = response.json()
                    except ValueError:
                        raw = None
                    final = _provider_response(response.status_code, raw)
                except (TimeoutError, httpx.TimeoutException) as exc:
                    status = "protocol_deadline" if clock() >= stage_deadline else "provider_timeout"
                    final = _empty_response(status, exc.__class__.__name__)
                except httpx.HTTPError as exc:
                    final = _empty_response("transport_error", exc.__class__.__name__)
                attempt.update(_attempt_end(final, clock(), stage_started, send_started))
        except asyncio.CancelledError:
            raise
        attempts.append(attempt)
        retryable = final["status"] in {"provider_timeout", "transport_error"} or final.get(
            "http_status"
        ) in V2_RETRYABLE_HTTP_STATUS
        if final["status"] == "ok" or not retryable:
            break

    final["attempts"] = attempts
    final["stage_timing"] = {
        "clock": "time.monotonic",
        "stage_wall_timeout_seconds": stage_limit,
        "effective_deadline_offset_seconds": round(max(stage_deadline - stage_started, 0.0), 6),
        "episode_deadline_applied": absolute_episode_deadline is not None,
        "stage_wall_elapsed_seconds": round(max(clock() - stage_started, 0.0), 6),
        "retry_backoff_seconds": round(backoff_total, 6),
        "deadline_exhausted": final["status"] == "protocol_deadline",
    }
    return final


def _attempt_end(
    result: dict[str, Any], ended: float, stage_started: float, send_started: float
) -> dict[str, Any]:
    return {
        "status": result["status"],
        "http_status": result.get("http_status"),
        "error_type": result.get("error_type"),
        "attempt_elapsed_seconds": round(max(ended - send_started, 0.0), 6),
        "attempt_end_offset_seconds": round(max(ended - stage_started, 0.0), 6),
    }


def _provider_response(status_code: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _empty_response(
            "provider_error" if status_code >= 400 else "transport_error",
            "non_json_provider_response",
            http_status=status_code,
        )
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
        "error_type": "http_error" if status_code >= 400 else None,
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "tool_calls_present": bool(message.get("tool_calls")),
        "fallback_detected": False,
    }


def _empty_response(status: str, error_type: str, *, http_status: int | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "content": "",
        "raw_response": None,
        "usage": {},
        "returned_model": None,
        "finish_reason": None,
        "http_status": http_status,
        "error_type": error_type,
        "reasoning_content_present": False,
        "tool_calls_present": False,
        "fallback_detected": False,
    }
