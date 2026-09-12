"""V4 parser surface.

V4 deliberately keeps the V3 Decimal parser unchanged. The V4 measurement
change is in transport/route attribution and protocol gates, not in the
model-visible schema or numeric parser.
"""

from __future__ import annotations

from typing import Any

from .models import ParserResult, StageName, StateMethodName
from .response_v3 import (
    canonical_parsed_representation_v3,
    parse_stage_response_v3,
)
from .response_v3 import (
    decimal_validator_report_cases as decimal_validator_report_cases_v3,
)


def parse_stage_response_v4(
    raw_text: str,
    stage: StageName,
    state_method: StateMethodName,
    required_dispatch_length: int,
    *,
    expected_dispatch_rows: list[dict[str, Any]],
    finish_reason: str | None = None,
) -> ParserResult:
    return parse_stage_response_v3(
        raw_text,
        stage,
        state_method,
        required_dispatch_length,
        expected_dispatch_rows=expected_dispatch_rows,
        finish_reason=finish_reason,
    )


def canonical_parsed_representation_v4(raw_text: str) -> dict[str, Any] | list[Any] | None:
    return canonical_parsed_representation_v3(raw_text)


def decimal_validator_report_cases() -> list[dict[str, Any]]:
    return decimal_validator_report_cases_v3()
