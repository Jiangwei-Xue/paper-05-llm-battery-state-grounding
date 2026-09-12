"""Provider boundary. The default adapter is mock-only and cannot make network calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .actions import HORIZON
from .hashing import canonical_json, sha256_json


@dataclass(frozen=True)
class MockResponse:
    status: str
    raw_text: str
    returned_model: str
    usage: dict[str, int]
    request_id: str


class ProviderAdapter(Protocol):
    def complete(self, payload: dict[str, Any], *, call_id: str) -> MockResponse:
        ...


class NetworkProviderDisabled(RuntimeError):
    pass


class MockProvider:
    """Deterministic fixture provider used by dry-run only."""

    network_attempts = 0

    def __init__(self, model_id: str = "mock-v2") -> None:
        self.model_id = model_id

    def complete(self, payload: dict[str, Any], *, call_id: str) -> MockResponse:
        digest = sha256_json(payload)
        interface = str(payload.get("interface", "I0"))
        branch = str(payload.get("branch", ""))
        # V2 dry-run intentionally exercises both zero and nonzero parser paths. It is not a
        # scientific result and is labelled mock in every evidence row.
        nonzero = interface in {"I2", "I3", "I_best"} or branch in {"S", "SYS_ALL"}
        if interface in {"I2", "I3", "I_best"}:
            start, end = (0, 4) if branch == "S" else (8, 12)
            body: dict[str, Any] = {
                "state": payload.get("carrier", {}),
                "actions": [{"start_step": start, "end_step_exclusive": end, "power_kw": 20.0}],
            }
        else:
            body = {"state": payload.get("carrier", {}), "actions_kw": [0.0] * HORIZON}
        if not nonzero:
            body["actions_kw"] = [0.0] * HORIZON
        raw = canonical_json(body)
        return MockResponse(
            status="ok",
            raw_text=raw,
            returned_model=self.model_id,
            usage={"input_tokens": max(1, len(canonical_json(payload)) // 4), "output_tokens": max(1, len(raw) // 4), "reasoning_tokens": 0},
            request_id=f"mock-{call_id}-{digest[:12]}",
        )


class LiveProviderPlaceholder:
    """Explicit placeholder; live network adapters are not enabled in V2 design phase."""

    def complete(self, payload: dict[str, Any], *, call_id: str) -> MockResponse:
        raise NetworkProviderDisabled(
            "live provider adapter is disabled; complete V2 dry-run and freeze before authorization"
        )
