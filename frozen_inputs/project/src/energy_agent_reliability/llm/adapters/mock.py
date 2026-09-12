"""Offline mock provider adapter for tests and evidence plumbing."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from energy_agent_reliability.llm.models import ProviderRequest, ProviderResponse


class MockProviderAdapter:
    def __init__(self, responses: Iterable[ProviderResponse]) -> None:
        self._responses: deque[ProviderResponse] = deque(responses)
        self.seen_payload_hashes: list[str] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.seen_payload_hashes.append(request.payload_sha256)
        if not self._responses:
            return ProviderResponse(
                status="provider_error",
                raw_text="",
                returned_model=request.requested_model,
                usage={},
                metadata={},
                error_type="mock_exhausted",
            )
        return self._responses.popleft()


def mock_text_response(raw_text: str, returned_model: str = "mock-model") -> ProviderResponse:
    return ProviderResponse(
        status="ok",
        raw_text=raw_text,
        returned_model=returned_model,
        usage={"input_tokens": 1, "output_tokens": 1},
        metadata={"mock": True},
    )

