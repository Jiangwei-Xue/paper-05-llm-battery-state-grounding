"""Deterministic, constraint-explicit battery state transition model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BatteryConfig


@dataclass(frozen=True)
class BatteryOverrides:
    usable_capacity_kwh: float | None = None
    max_charge_kw: float | None = None
    max_discharge_kw: float | None = None
    export_limit_kw: float | None = None
    reserve_soc_kwh: float | None = None


@dataclass(frozen=True)
class SimulationResult:
    trace: pd.DataFrame
    total_cost: float
    feasible: bool
    terminal_soc_gap_kwh: float


def _effective(config: BatteryConfig, overrides: BatteryOverrides) -> BatteryOverrides:
    capacity = overrides.usable_capacity_kwh or config.energy_capacity_kwh
    reserve = overrides.reserve_soc_kwh if overrides.reserve_soc_kwh is not None else config.reserve_soc_kwh
    if capacity <= 0 or reserve < 0 or reserve > capacity:
        raise ValueError("Battery capacity and reserve must define a non-empty feasible SOC interval.")
    return BatteryOverrides(
        usable_capacity_kwh=capacity,
        max_charge_kw=overrides.max_charge_kw or config.max_charge_kw,
        max_discharge_kw=overrides.max_discharge_kw or config.max_discharge_kw,
        export_limit_kw=overrides.export_limit_kw or config.export_limit_kw,
        reserve_soc_kwh=reserve,
    )


def simulate_dispatch(
    frame: pd.DataFrame,
    action_kw: np.ndarray | list[float],
    config: BatteryConfig,
    overrides: BatteryOverrides = BatteryOverrides(),
    overrides_by_step: list[BatteryOverrides] | None = None,
    initial_soc_kwh: float | None = None,
    require_terminal_soc: bool = True,
    step_hours: float | None = None,
) -> SimulationResult:
    """Apply a grid-side action vector without silently violating any battery bound.

    Positive actions charge from the grid; negative actions discharge to serve load or
    export. Each requested action is deterministically clipped to the instantaneous
    SOC and power constraints. Costs use $/MWh prices and a throughput degradation term.
    """
    required = {"load_kw"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing simulation columns: {sorted(missing)}")
    pv_column = _pv_column(frame)
    price_column = _price_column(frame)
    actions = np.asarray(action_kw, dtype=float)
    if len(actions) != len(frame):
        raise ValueError("Dispatch action length must equal the data frame length.")
    if not np.isfinite(actions).all():
        raise ValueError("Dispatch actions must be finite.")
    schedule = overrides_by_step or [overrides] * len(frame)
    if len(schedule) != len(frame):
        raise ValueError("Dynamic override schedule length must equal the data frame length.")
    dt_hours = _infer_step_hours(frame.index, step_hours)
    initial = config.initial_soc_kwh if initial_soc_kwh is None else initial_soc_kwh
    first_limits = _effective(config, schedule[0])
    first_reserve = float(first_limits.reserve_soc_kwh or config.reserve_soc_kwh)
    first_capacity = float(first_limits.usable_capacity_kwh or config.energy_capacity_kwh)
    soc = min(max(initial, first_reserve), first_capacity)
    loads = frame["load_kw"].to_numpy(dtype=float)
    photovoltaics = frame[pv_column].to_numpy(dtype=float)
    prices = frame[price_column].to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for load_kw, pv_kw, price, requested, requested_limits in zip(
        loads, photovoltaics, prices, actions, schedule, strict=True
    ):
        limits = _effective(config, requested_limits)
        capacity = float(limits.usable_capacity_kwh or config.energy_capacity_kwh)
        reserve = float(limits.reserve_soc_kwh or config.reserve_soc_kwh)
        unavailable_energy_kwh = max(soc - capacity, 0.0)
        soc = min(soc, capacity)
        max_charge = float(limits.max_charge_kw or config.max_charge_kw)
        max_discharge = float(limits.max_discharge_kw or config.max_discharge_kw)
        export_limit = float(limits.export_limit_kw or config.export_limit_kw)
        charge = min(
            max(requested, 0.0), max_charge, (capacity - soc) / (config.charge_efficiency * dt_hours)
        )
        discharge = min(
            max(-requested, 0.0),
            max_discharge,
            max((soc - reserve) * config.discharge_efficiency / dt_hours, 0.0),
        )
        soc += charge * config.charge_efficiency * dt_hours
        soc -= discharge / config.discharge_efficiency * dt_hours
        residual_kw = float(load_kw) - float(pv_kw) + charge - discharge
        import_kw = max(residual_kw, 0.0)
        uncapped_export_kw = max(-residual_kw, 0.0)
        export_kw = min(uncapped_export_kw, export_limit)
        curtailed_kw = uncapped_export_kw - export_kw
        energy_mwh = dt_hours / 1000.0
        energy_cost = (import_kw - export_kw) * float(price) * energy_mwh
        degradation_cost = (charge + discharge) * dt_hours * config.degradation_cost_per_kwh
        clipping_kwh = max(requested - charge, 0.0) * dt_hours
        clipping_kwh += max(-requested - discharge, 0.0) * dt_hours
        violation_cost = clipping_kwh * config.violation_penalty_per_kwh
        rows.append(
            {
                "requested_action_kw": requested,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "soc_kwh": soc,
                "available_capacity_kwh": capacity,
                "reserve_soc_kwh": reserve,
                "capacity_loss_kwh": unavailable_energy_kwh,
                "import_kw": import_kw,
                "export_kw": export_kw,
                "curtailed_kw": curtailed_kw,
                "energy_cost": energy_cost,
                "degradation_cost": degradation_cost,
                "violation_cost": violation_cost,
            }
        )
    trace = pd.DataFrame(rows, index=frame.index)
    terminal_gap = abs(soc - config.terminal_soc_kwh)
    feasible = bool(
        (trace["soc_kwh"] >= trace["reserve_soc_kwh"] - 1e-8).all()
        and (trace["soc_kwh"] <= trace["available_capacity_kwh"] + 1e-8).all()
        and (not require_terminal_soc or terminal_gap <= 1e-6)
    )
    return SimulationResult(
        trace=trace,
        total_cost=float(trace[["energy_cost", "degradation_cost", "violation_cost"]].sum().sum()),
        feasible=feasible,
        terminal_soc_gap_kwh=terminal_gap,
    )


def _infer_step_hours(index: pd.Index, explicit_step_hours: float | None = None) -> float:
    if explicit_step_hours is not None:
        step = float(explicit_step_hours)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Explicit simulation step_hours must be finite and positive.")
        return step
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Cannot infer simulation interval from fewer than two timestamps; pass step_hours explicitly.")
    deltas = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    step = float(np.median(deltas) / 3600.0)
    if step <= 0:
        raise ValueError("Datetime index must be strictly increasing.")
    return step


def _pv_column(frame: pd.DataFrame) -> str:
    for column in ("pv_ac_kw", "pv_kw"):
        if column in frame.columns:
            return column
    raise ValueError("Missing simulation PV column: expected pv_ac_kw or pv_kw.")


def _price_column(frame: pd.DataFrame) -> str:
    for column in ("day_ahead_price_usd_mwh", "price_per_mwh"):
        if column in frame.columns:
            return column
    raise ValueError("Missing simulation price column: expected day_ahead_price_usd_mwh or price_per_mwh.")
