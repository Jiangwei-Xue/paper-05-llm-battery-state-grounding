"""Adapter interface for LLM providers."""

from __future__ import annotations

from typing import Protocol

from energy_agent_reliability.llm.models import ProviderRequest, ProviderResponse


class ProviderAdapter(Protocol):
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Return one provider response for an already constructed payload."""

