"""Usage recording and named cost contracts; no budget hard stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class Pricing:
    input_price_per_million: float | None
    cached_input_price_per_million: float | None
    output_price_per_million: float | None
    reasoning_price_per_million: float | None = None


def estimate_cost_cny(usage: Usage, pricing: Pricing) -> float | None:
    input_price = pricing.input_price_per_million
    cached_input_price = pricing.cached_input_price_per_million
    output_price = pricing.output_price_per_million
    if input_price is None or cached_input_price is None or output_price is None:
        return None
    assert input_price is not None
    assert cached_input_price is not None
    assert output_price is not None
    result = (
        usage.input_tokens * input_price / 1_000_000
        + usage.cached_input_tokens * cached_input_price / 1_000_000
        + usage.output_tokens * output_price / 1_000_000
    )
    if pricing.reasoning_price_per_million is not None:
        result += usage.reasoning_tokens * pricing.reasoning_price_per_million / 1_000_000
    return result


class CostLedger:
    """Accumulates estimates and actual usage while never stopping on cost."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        call_id: str,
        usage: Usage | None,
        pricing: Pricing,
        actual_cost_cny: float | None = None,
    ) -> dict[str, Any]:
        estimate = estimate_cost_cny(usage, pricing) if usage is not None else None
        row = {
            "call_id": call_id,
            "usage_present": usage is not None,
            "estimated_cost_cny": estimate,
            "actual_cost_cny": actual_cost_cny,
            "budget_hard_stop": False,
            "cost_status": "recorded" if estimate is not None or actual_cost_cny is not None else "pending_pricing_or_usage",
        }
        self.rows.append(row)
        return row
