"""Strict dense and sparse action contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

HORIZON = 96
INTERVAL_HOURS = 0.25
MAX_SEGMENTS = 24


class ActionContractError(ValueError):
    """Raised when an action violates the frozen output contract."""


def expand_sparse_segments(
    segments: Iterable[dict[str, Any]],
    *,
    horizon: int = HORIZON,
    max_segments: int = MAX_SEGMENTS,
) -> list[float]:
    items = list(segments)
    if len(items) > max_segments:
        raise ActionContractError(f"segment_count>{max_segments}")
    dense = [0.0] * horizon
    occupied = [False] * horizon
    for index, segment in enumerate(items):
        if not isinstance(segment, dict):
            raise ActionContractError(f"segment_{index}_not_object")
        required = {"start_step", "end_step_exclusive", "power_kw"}
        if set(segment) != required:
            raise ActionContractError(f"segment_{index}_fields_invalid")
        start = segment["start_step"]
        end = segment["end_step_exclusive"]
        power = segment["power_kw"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ActionContractError(f"segment_{index}_start_not_integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ActionContractError(f"segment_{index}_end_not_integer")
        if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(power):
            raise ActionContractError(f"segment_{index}_power_not_finite")
        if start < 0 or end > horizon or start >= end:
            raise ActionContractError(f"segment_{index}_range_invalid")
        if abs(float(power)) > 250.0:
            raise ActionContractError(f"segment_{index}_power_out_of_range")
        for step in range(start, end):
            if occupied[step]:
                raise ActionContractError(f"segment_{index}_overlap")
            occupied[step] = True
            dense[step] = float(power)
    return dense


def validate_dense_action(values: Iterable[Any], *, horizon: int = HORIZON) -> list[float]:
    dense = list(values)
    if len(dense) != horizon:
        raise ActionContractError(f"dense_length_{len(dense)}_expected_{horizon}")
    result: list[float] = []
    for index, value in enumerate(dense):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ActionContractError(f"dense_{index}_not_finite")
        if abs(float(value)) > 250.0:
            raise ActionContractError(f"dense_{index}_out_of_range")
        result.append(float(value))
    return result


def action_energy_kwh(values: Iterable[float]) -> float:
    return INTERVAL_HOURS * sum(abs(float(value)) for value in values)


def is_nontrivial(values: Iterable[float], *, threshold_kwh: float = 5.0) -> bool:
    return action_energy_kwh(values) >= threshold_kwh
