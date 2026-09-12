#!/usr/bin/env python3
"""Strict parser for the E2b-v2 local-offset action contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

TOP_LEVEL_FIELDS = {"actions"}
ACTION_FIELDS = {"start_offset", "end_offset_exclusive", "power_kw"}
DEFAULT_HARD_POWER_LIMIT_KW = 250.0


@dataclass(frozen=True)
class ParsedE2BV2Output:
    ok: bool
    raw_actions: Any
    dense_action_kw: list[float]
    diagnostics: list[str]
    horizon: int
    segment_count: int


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_output(
    raw_text: str,
    horizon: int,
    *,
    hard_power_limit_kw: float = DEFAULT_HARD_POWER_LIMIT_KW,
) -> ParsedE2BV2Output:
    """Parse one response without aliases, repair, clipping, or index inference."""

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    try:
        parsed = json.loads(raw_text, parse_constant=_reject_nonfinite_constant)
    except (json.JSONDecodeError, ValueError):
        return ParsedE2BV2Output(False, None, [], ["invalid_json"], horizon, 0)
    if not isinstance(parsed, dict):
        return ParsedE2BV2Output(False, None, [], ["top_level_not_object"], horizon, 0)
    if set(parsed) != TOP_LEVEL_FIELDS:
        return ParsedE2BV2Output(
            False,
            parsed.get("actions"),
            [],
            ["top_level_fields_invalid"],
            horizon,
            len(parsed.get("actions", [])) if isinstance(parsed.get("actions"), list) else 0,
        )
    actions = parsed["actions"]
    if not isinstance(actions, list):
        return ParsedE2BV2Output(False, actions, [], ["actions_not_array"], horizon, 0)

    diagnostics: list[str] = []
    if len(actions) > horizon:
        diagnostics.append("segment_count_exceeds_horizon")
    dense = [0.0] * horizon
    previous_end = 0
    occupied = [False] * horizon
    for index, segment in enumerate(actions):
        prefix = f"segment_{index}"
        if not isinstance(segment, dict):
            diagnostics.append(f"{prefix}_not_object")
            continue
        if set(segment) != ACTION_FIELDS:
            diagnostics.append(f"{prefix}_fields_invalid")
            continue
        start = segment["start_offset"]
        end = segment["end_offset_exclusive"]
        power = segment["power_kw"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            diagnostics.append(f"{prefix}_offset_not_integer")
            continue
        if start < 0 or end > horizon or start >= end:
            diagnostics.append(f"{prefix}_range_invalid")
            continue
        if index > 0 and start < previous_end:
            diagnostics.append(f"{prefix}_unsorted_or_overlap")
        previous_end = max(previous_end, end)
        if (
            isinstance(power, bool)
            or not isinstance(power, (int, float))
            or not math.isfinite(float(power))
        ):
            diagnostics.append(f"{prefix}_power_nonfinite_or_nonnumeric")
            continue
        if abs(float(power)) > hard_power_limit_kw:
            diagnostics.append(f"{prefix}_power_outside_hard_envelope")
            continue
        for offset in range(start, end):
            if occupied[offset]:
                if f"{prefix}_unsorted_or_overlap" not in diagnostics:
                    diagnostics.append(f"{prefix}_unsorted_or_overlap")
                continue
            occupied[offset] = True
            dense[offset] = float(power)
    return ParsedE2BV2Output(
        not diagnostics,
        actions,
        dense if not diagnostics else [],
        diagnostics,
        horizon,
        len(actions),
    )
