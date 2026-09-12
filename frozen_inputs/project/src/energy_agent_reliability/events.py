"""Dynamic-event definitions with an explicit visible/oracle information boundary."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .battery import BatteryOverrides
from .config import BatteryConfig


@dataclass(frozen=True)
class DynamicEvent:
    visible_update: dict[str, Any]
    scorer_oracle: dict[str, Any]


_SEVERITY = {
    "easy": (0.08, 0.10),
    "medium": (0.20, 0.25),
    "hard": (0.38, 0.40),
}


def generate_event(family: str, difficulty: str, seed: int, battery: BatteryConfig) -> DynamicEvent:
    """Create one reproducible event without placing hidden values in visible text."""
    if difficulty not in _SEVERITY:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    magnitude, oracle_gap = _SEVERITY[difficulty]
    rng = random.Random(seed)
    activation_step = rng.randint(4, 15)
    common = {"event_family": family, "difficulty": difficulty, "activation_step": activation_step}
    if family == "forecast_revision":
        return DynamicEvent(
            visible_update={
                **common,
                "forecast_update_method": "latest_observation_clearsky_index_persistence",
                "forecast_revision_blend_weight": round(magnitude / 0.40, 4),
            },
            scorer_oracle={
                "pre_revision_forecast_error_reference": round(oracle_gap, 4),
                "oracle_version": 1,
            },
        )
    if family == "battery_capacity_derating":
        capacity = battery.energy_capacity_kwh * (1 - magnitude)
        return DynamicEvent(
            visible_update={**common, "usable_capacity_kwh": round(capacity, 3)},
            scorer_oracle={"pre_event_capacity_kwh": battery.energy_capacity_kwh, "oracle_version": 1},
        )
    if family == "power_limit_derating":
        charge = battery.max_charge_kw * (1 - magnitude)
        discharge = battery.max_discharge_kw * (1 - magnitude)
        return DynamicEvent(
            visible_update={**common, "max_charge_kw": round(charge, 3), "max_discharge_kw": round(discharge, 3)},
            scorer_oracle={"pre_event_max_charge_kw": battery.max_charge_kw, "pre_event_max_discharge_kw": battery.max_discharge_kw, "oracle_version": 1},
        )
    if family == "export_limit_update":
        export = battery.export_limit_kw * (1 - magnitude)
        return DynamicEvent(
            visible_update={**common, "export_limit_kw": round(export, 3)},
            scorer_oracle={"pre_event_export_limit_kw": battery.export_limit_kw, "oracle_version": 1},
        )
    if family == "reserve_commitment_update":
        reserve = battery.reserve_soc_kwh + (battery.energy_capacity_kwh - battery.reserve_soc_kwh) * magnitude
        return DynamicEvent(
            visible_update={**common, "reserve_soc_kwh": round(reserve, 3)},
            scorer_oracle={"pre_event_reserve_soc_kwh": battery.reserve_soc_kwh, "oracle_version": 1},
        )
    raise ValueError(f"Unsupported event family: {family}")


def battery_overrides(visible_update: dict[str, Any]) -> BatteryOverrides:
    """Map public event fields to simulator constraints; unknown fields cannot affect state."""
    return BatteryOverrides(
        usable_capacity_kwh=_as_float(visible_update.get("usable_capacity_kwh")),
        max_charge_kw=_as_float(visible_update.get("max_charge_kw")),
        max_discharge_kw=_as_float(visible_update.get("max_discharge_kw")),
        export_limit_kw=_as_float(visible_update.get("export_limit_kw")),
        reserve_soc_kwh=_as_float(visible_update.get("reserve_soc_kwh")),
    )


def dynamic_override_schedule(visible_update: dict[str, Any], horizon_steps: int) -> list[BatteryOverrides]:
    """Keep nominal constraints until the visible event's declared activation step."""
    activation = int(visible_update["activation_step"])
    if not 0 <= activation < horizon_steps:
        raise ValueError("Event activation must fall inside the scenario horizon.")
    event_override = battery_overrides(visible_update)
    return [BatteryOverrides()] * activation + [event_override] * (horizon_steps - activation)


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)
