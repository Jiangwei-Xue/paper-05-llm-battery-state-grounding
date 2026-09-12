"""OpenAI-compatible adapter shell with network disabled by default."""

from __future__ import annotations

import os

from energy_agent_reliability.llm.models import ProviderRequest, ProviderResponse


class OpenAICompatibleAdapter:
    def __init__(self, *, api_key_env: str, allow_network: bool = False) -> None:
        self.api_key_env = api_key_env
        self.allow_network = allow_network

    def api_key_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if not self.allow_network:
            return ProviderResponse(
                status="provider_error",
                raw_text="",
                returned_model=None,
                usage={},
                metadata={
                    "network_disabled": True,
                    "requested_model": request.requested_model,
                    "payload_sha256": request.payload_sha256,
                },
                error_type="network_disabled_current_stage",
            )
        raise RuntimeError("Provider calls require an explicit future implementation and user authorization.")

