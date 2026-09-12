"""Bounded retry logic with immutable payload hashes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters.base import ProviderAdapter
from .models import ProviderRequest, ProviderResponse

RETRYABLE_ERROR_TYPES = {"timeout", "transport_exception"}
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    max_transport_attempts: int = 3


@dataclass(frozen=True)
class RetryResult:
    response: ProviderResponse
    attempts: list[dict[str, Any]]
    payload_hash_constant: bool


def should_retry(response: ProviderResponse) -> bool:
    status_code = response.metadata.get("status_code")
    return bool(
        response.status in {"timeout", "transport_error"}
        or response.error_type in RETRYABLE_ERROR_TYPES
        or status_code in RETRYABLE_STATUS_CODES
    )


def execute_with_retry(
    adapter: ProviderAdapter, request: ProviderRequest, policy: RetryPolicy
) -> RetryResult:
    attempts: list[dict[str, Any]] = []
    payload_hashes: list[str] = []
    latest: ProviderResponse | None = None
    for attempt_index in range(policy.max_transport_attempts):
        payload_hashes.append(request.payload_sha256)
        response = adapter.complete(request)
        latest = response
        attempts.append(
            {
                "attempt_index": attempt_index,
                "status": response.status,
                "error_type": response.error_type,
                "payload_sha256": request.payload_sha256,
            }
        )
        if response.status == "ok" or not should_retry(response):
            break
    if latest is None:
        raise RuntimeError("Retry policy made no attempts.")
    return RetryResult(latest, attempts, len(set(payload_hashes)) == 1)
