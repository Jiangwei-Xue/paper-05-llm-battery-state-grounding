"""Load and cross-check frozen non-data contracts before processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from .config import FrozenProtocol


@dataclass(frozen=True)
class SystemSizing:
    ac_capacity_kw: float
    dc_ac_ratio: float
    peak_load_kw: float
    scaling_policy: str


def load_system_sizing(path: str | Path) -> SystemSizing:
    """Read the pre-candidate sizing contract and reject malformed values."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("sizing_contract_id") != "SYSTEM_SIZING_CONTRACT_V1":
        raise ValueError("Expected SYSTEM_SIZING_CONTRACT_V1.")
    pv = _mapping(raw, "pv")
    load = _mapping(raw, "load")
    sizing = SystemSizing(
        ac_capacity_kw=float(pv["ac_capacity_kw"]),
        dc_ac_ratio=float(pv["dc_ac_ratio"]),
        peak_load_kw=float(load["peak_load_kw"]),
        scaling_policy=str(load["scaling_policy"]),
    )
    if sizing.ac_capacity_kw <= 0 or sizing.dc_ac_ratio < 1 or sizing.peak_load_kw <= 0:
        raise ValueError("Sizing capacities must be positive and DC/AC ratio must be at least one.")
    if sizing.scaling_policy != "scale_representative_oedi_profile_to_fixed_peak_kw":
        raise ValueError("Unexpected OEDI scaling policy.")
    return sizing


def validate_sizing_matches_protocol(sizing_path: str | Path, protocol: FrozenProtocol) -> SystemSizing:
    """Require simulator economics and battery state to match the frozen sizing file."""
    raw = yaml.safe_load(Path(sizing_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Sizing contract is not a mapping.")
    sizing = load_system_sizing(sizing_path)
    battery = _mapping(raw, "battery")
    grid = _mapping(raw, "grid")
    economics = _mapping(raw, "economics")
    penalties = _mapping(economics, "violation_penalties")
    expected = {
        "usable_capacity_kwh": protocol.battery.energy_capacity_kwh,
        "max_charge_kw": protocol.battery.max_charge_kw,
        "max_discharge_kw": protocol.battery.max_discharge_kw,
        "charge_efficiency": protocol.battery.charge_efficiency,
        "discharge_efficiency": protocol.battery.discharge_efficiency,
        "initial_soc_kwh": protocol.battery.initial_soc_kwh,
        "minimum_soc_kwh": protocol.battery.reserve_soc_kwh,
        "terminal_soc_kwh": protocol.battery.terminal_soc_kwh,
    }
    for key, value in expected.items():
        if float(battery[key]) != value:
            raise ValueError(f"Sizing contract and frozen protocol disagree on {key}.")
    if float(economics["degradation_cost_usd_per_kwh_throughput"]) != protocol.battery.degradation_cost_per_kwh:
        raise ValueError("Sizing contract and frozen protocol disagree on degradation cost.")
    if float(penalties["clipped_action_usd_per_kwh"]) != protocol.battery.violation_penalty_per_kwh:
        raise ValueError("Sizing contract and frozen protocol disagree on clipping penalty.")
    if float(grid["base_export_limit_kw"]) != protocol.battery.export_limit_kw:
        raise ValueError("Sizing contract and frozen protocol disagree on export limit.")
    return sizing


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Sizing contract lacks mapping {key}.")
    return cast(dict[str, Any], value)
