"""Typed, frozen protocol configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, PositiveFloat, PositiveInt


class Location(BaseModel, frozen=True):
    location_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float
    tilt_deg: float = Field(ge=0, le=90)
    azimuth_deg: float = Field(ge=0, le=360)


class NSRDBConfig(BaseModel, frozen=True):
    endpoint: HttpUrl
    dataset: str
    api_key_env: str
    attributes: list[str]
    interval_minutes: PositiveInt
    leap_day: bool
    utc: bool
    source_note: str


class OEDIConfig(BaseModel, frozen=True):
    url: HttpUrl
    building_type: str
    source_note: str
    max_bytes: PositiveInt


class CAISOConfig(BaseModel, frozen=True):
    endpoint: HttpUrl
    node: str
    market_run_id: str
    source_note: str
    chunk_mode: Literal["monthly", "weekly", "daily"]
    max_retries: PositiveInt
    initial_backoff_seconds: PositiveFloat
    max_backoff_seconds: PositiveFloat


class DataConfig(BaseModel, frozen=True):
    nsrdb: NSRDBConfig
    oedi: OEDIConfig
    caiso: CAISOConfig


class BatteryConfig(BaseModel, frozen=True):
    energy_capacity_kwh: PositiveFloat
    max_charge_kw: PositiveFloat
    max_discharge_kw: PositiveFloat
    charge_efficiency: PositiveFloat = Field(le=1)
    discharge_efficiency: PositiveFloat = Field(le=1)
    initial_soc_kwh: float = Field(ge=0)
    reserve_soc_kwh: float = Field(ge=0)
    terminal_soc_kwh: float = Field(ge=0)
    export_limit_kw: PositiveFloat
    degradation_cost_per_kwh: float = Field(ge=0)
    violation_penalty_per_kwh: float = Field(default=100.0, ge=0)


class ScenarioConfig(BaseModel, frozen=True):
    horizon_hours: PositiveInt
    candidates_per_cell: PositiveInt
    event_families: list[str]
    difficulties: list[str]
    seasons: dict[str, list[int]]


class SelectionConfig(BaseModel, frozen=True):
    selection_seed: int
    event_generation_seed: int
    bootstrap_seed: int
    main_pool_size: PositiveInt
    pilot_pool_size: PositiveInt
    strata_fields: list[str]
    pilot_strata_fields: list[str]


class FrozenProtocol(BaseModel, frozen=True):
    protocol_version: str
    frozen_year: PositiveInt
    timezone: str
    storage_timezone: str
    analysis_resolution_minutes: PositiveInt
    locations: list[Location]
    data: DataConfig
    battery: BatteryConfig
    scenario: ScenarioConfig
    selection: SelectionConfig


def load_protocol(path: str | Path) -> FrozenProtocol:
    """Load one YAML protocol and reject structurally invalid scope changes."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    protocol = FrozenProtocol.model_validate(raw)
    if len(protocol.locations) != 4:
        raise ValueError("The frozen protocol must contain exactly four California locations.")
    if protocol.selection.main_pool_size != 40 or protocol.selection.pilot_pool_size != 10:
        raise ValueError("The frozen protocol fixes a 40-task main pool and a 10-task pilot pool.")
    if set(protocol.scenario.difficulties) != {"easy", "medium", "hard"}:
        raise ValueError("Difficulty levels must be exactly easy, medium, hard.")
    if protocol.analysis_resolution_minutes != 15 or protocol.storage_timezone != "UTC":
        raise ValueError("The pre-execution protocol fixes 15-minute UTC internal storage.")
    expected_nsrdb_path = "/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
    if protocol.data.nsrdb.endpoint.host != "developer.nlr.gov" or protocol.data.nsrdb.endpoint.path != expected_nsrdb_path:
        raise ValueError("NSRDB must use the official developer.nlr.gov GOES CONUS v4.0.0 endpoint.")
    required_nsrdb_attributes = {
        "ghi",
        "dni",
        "dhi",
        "air_temperature",
        "wind_speed",
        "solar_zenith_angle",
        "clearsky_ghi",
    }
    if (
        protocol.data.nsrdb.dataset != "nsrdb-GOES-conus-v4-0-0"
        or set(protocol.data.nsrdb.attributes) != required_nsrdb_attributes
        or protocol.data.nsrdb.interval_minutes != 15
        or protocol.data.nsrdb.leap_day
        or not protocol.data.nsrdb.utc
    ):
        raise ValueError("NSRDB request parameters are not the frozen 2024 15-minute UTC non-leap specification.")
    return protocol
