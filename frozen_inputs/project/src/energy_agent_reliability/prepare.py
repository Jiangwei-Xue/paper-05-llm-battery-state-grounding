"""Build the contract-aligned, real-data operational table without substitutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib  # type: ignore[import-untyped]

from .caiso_bulk import resolve_caiso_source
from .calendar_policy import claim_index, drop_excluded_dates, has_excluded_date, source_index
from .config import FrozenProtocol, Location
from .contracts import SystemSizing, validate_sizing_matches_protocol
from .forecasting import day_ahead_pv_forecast, forecast_is_causal
from .provenance import sha256_file, utc_now


def prepare_all(
    protocol: FrozenProtocol,
    raw_dir: str | Path,
    processed_dir: str | Path,
    sizing_path: str | Path = "protocol/SYSTEM_SIZING_CONTRACT_V1.yaml",
) -> list[Path]:
    """Create all site tables and one unified table only after every input is complete."""
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    sizing = validate_sizing_matches_protocol(sizing_path, protocol)
    load_kw = _prepare_oedi_load(
        raw / "oedi" / f"ca_{protocol.data.oedi.building_type.lower()}.csv", protocol, sizing
    )
    prices = _prepare_caiso_prices(raw / "caiso", protocol)
    raw_hashes = _raw_hashes(raw, protocol)
    site_tables: dict[str, pd.DataFrame] = {}
    for location in protocol.locations:
        weather = _prepare_nsrdb_weather(
            raw / "nsrdb" / f"{location.location_id}_{protocol.frozen_year}.csv", protocol
        )
        pv = pv_ac_power(weather, location, sizing.ac_capacity_kw, sizing.dc_ac_ratio)
        weather_index = pd.DatetimeIndex(weather.index)
        clear_sky_ghi = weather["clearsky_ghi"].rename("clear_sky_ghi")
        forecast = day_ahead_pv_forecast(
            weather_index, pv["pv_ac_kw"], clear_sky_ghi, sizing.ac_capacity_kw
        )
        aligned = pd.concat(
            [weather.rename(columns={"air_temperature": "temperature"}), pv, load_kw, prices, clear_sky_ghi, forecast],
            axis=1,
        ).reindex(claim_index(protocol.frozen_year, protocol.analysis_resolution_minutes))
        if aligned.isna().any().any():
            missing = aligned.isna().sum().to_dict()
            raise ValueError(f"Data alignment incomplete for {location.location_id}: {missing}")
        if has_excluded_date(pd.DatetimeIndex(aligned.index)):
            raise ValueError(f"Excluded date remained in {location.location_id} processed table.")
        aligned.insert(0, "site_id", location.location_id)
        aligned.insert(0, "timestamp_utc", pd.DatetimeIndex(aligned.index))
        aligned["real_time_price_usd_mwh"] = np.nan
        aligned["source_row_hashes"] = json.dumps(raw_hashes[location.location_id], sort_keys=True)
        ordered = aligned[
            [
                "timestamp_utc",
                "site_id",
                "ghi",
                "dni",
                "dhi",
                "temperature",
                "wind_speed",
                "pv_ac_kw",
                "load_kw",
                "day_ahead_price_usd_mwh",
                "real_time_price_usd_mwh",
                "clear_sky_ghi",
                "forecast_pv_kw",
                "forecast_issue_time",
                "forecast_method",
                "source_row_hashes",
            ]
        ].copy()
        site_tables[location.location_id] = ordered
    processed.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for location in protocol.locations:
        target = processed / f"{location.location_id}_{protocol.frozen_year}_15min.parquet"
        _write_parquet_atomic(site_tables[location.location_id], target)
        _write_metadata(target, protocol, location.location_id, raw_hashes[location.location_id])
        outputs.append(target)
    unified = pd.concat([site_tables[location.location_id] for location in protocol.locations], ignore_index=True)
    unified_target = processed / "operational_timeseries_v1.parquet"
    _write_parquet_atomic(unified, unified_target)
    _write_metadata(unified_target, protocol, "all_sites", raw_hashes)
    audit_processed_data(protocol, processed, Path(raw).parent.parent)
    return outputs


def pv_ac_power(
    weather: pd.DataFrame, location: Location, ac_capacity_kw: float = 400.0, dc_ac_ratio: float = 1.2
) -> pd.DataFrame:
    """Use pvlib to derive AC PV and clip it to the fixed AC nameplate."""
    required = {"ghi", "dhi", "dni", "air_temperature", "wind_speed"}
    if missing := required.difference(weather.columns):
        raise ValueError(f"NSRDB weather lacks required fields: {sorted(missing)}")
    solar_position = pvlib.solarposition.get_solarposition(
        weather.index, location.latitude, location.longitude, altitude=location.altitude_m
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=location.tilt_deg,
        surface_azimuth=location.azimuth_deg,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"].clip(lower=0)
    cell_temperature = pvlib.temperature.faiman(
        poa_global=poa, temp_air=weather["air_temperature"], wind_speed=weather["wind_speed"]
    )
    dc_w = pvlib.pvsystem.pvwatts_dc(
        poa, temp_cell=cell_temperature, pdc0=ac_capacity_kw * dc_ac_ratio * 1000, gamma_pdc=-0.0037
    )
    inverter_pdc0 = ac_capacity_kw * 1000 / 0.96
    ac_w = pvlib.inverter.pvwatts(dc_w, pdc0=inverter_pdc0).clip(lower=0, upper=ac_capacity_kw * 1000)
    return pd.DataFrame({"pv_ac_kw": ac_w / 1000.0}, index=weather.index)


def audit_processed_data(
    protocol: FrozenProtocol, processed_dir: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Audit the unified table; write a blocked report rather than invent incomplete data."""
    processed = Path(processed_dir)
    root = Path(project_root)
    target = processed / "operational_timeseries_v1.parquet"
    report: dict[str, Any] = {
        "report_type": "PROCESSED_DATA_AUDIT",
        "created_utc": utc_now(),
        "protocol_id": "DATA_CONTRACT_V1_2",
        "status": "blocked",
        "passed": False,
        "reason": "Unified processed data are absent.",
    }
    if target.exists():
        table = pd.read_parquet(target)
        expected = claim_index(protocol.frozen_year, protocol.analysis_resolution_minutes)
        site_reports: dict[str, Any] = {}
        all_passed = True
        for location in protocol.locations:
            site = table[table["site_id"] == location.location_id].copy()
            index = pd.DatetimeIndex(pd.to_datetime(site["timestamp_utc"], utc=True))
            expected_match = index.equals(expected)
            solar = pvlib.solarposition.get_solarposition(
                index, location.latitude, location.longitude, altitude=location.altitude_m
            )
            night_pv = site.loc[solar["apparent_zenith"].to_numpy() >= 90, "pv_ac_kw"]
            valid = bool(
                expected_match
                and not has_excluded_date(index)
                and np.isfinite(site[["ghi", "dni", "dhi", "temperature", "wind_speed", "pv_ac_kw", "load_kw", "day_ahead_price_usd_mwh", "forecast_pv_kw"]].to_numpy(dtype=float)).all()
                and (site["pv_ac_kw"] >= 0).all()
                and (site["pv_ac_kw"] <= 400.0 + 1e-6).all()
                and (site["load_kw"] >= 0).all()
                and forecast_is_causal(index, site["forecast_issue_time"])
                and (night_pv.abs() <= 0.5).all()
            )
            site_reports[location.location_id] = {
                "rows": len(site),
                "expected_rows": len(expected),
                "expected_grid_match": expected_match,
                "contains_excluded_date": has_excluded_date(index),
                "night_pv_max_kw": None if night_pv.empty else float(night_pv.abs().max()),
                "pv_max_kw": None if site.empty else float(site["pv_ac_kw"].max()),
                "load_range_kw": [float(site["load_kw"].min()), float(site["load_kw"].max())] if not site.empty else None,
                "price_range_usd_mwh": [float(site["day_ahead_price_usd_mwh"].min()), float(site["day_ahead_price_usd_mwh"].max())] if not site.empty else None,
                "forecast_causal": forecast_is_causal(index, site["forecast_issue_time"]),
                "passed": valid,
            }
            all_passed = all_passed and valid
        report = {
            "report_type": "PROCESSED_DATA_AUDIT",
            "created_utc": utc_now(),
            "protocol_id": "DATA_CONTRACT_V1_2",
            "status": "completed",
            "rows": len(table),
            "expected_rows": len(expected) * len(protocol.locations),
            "processed_sha256": sha256_file(target),
            "site_reports": site_reports,
            "passed": all_passed and len(table) == len(expected) * len(protocol.locations),
        }
        report["status"] = "pass" if report["passed"] else "fail"
    destination = root / "reports" / "PROCESSED_DATA_AUDIT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _prepare_nsrdb_weather(path: Path, protocol: FrozenProtocol) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NSRDB raw file: {path}")
    raw = pd.read_csv(path, skiprows=2)
    raw.columns = [str(column).strip().lower().replace(" ", "_") for column in raw.columns]
    aliases = {"temperature": "air_temperature", "air_temperature": "air_temperature"}
    raw = raw.rename(columns=aliases)
    required = set(protocol.data.nsrdb.attributes)
    if missing := required.difference(raw.columns):
        raise ValueError(f"NSRDB file does not contain {sorted(missing)}")
    timestamp = pd.DatetimeIndex(pd.to_datetime(raw[["year", "month", "day", "hour", "minute"]])).tz_localize("UTC")
    output = raw[list(required)].apply(pd.to_numeric, errors="coerce")
    output.index = timestamp
    native = _full_claim_year(output, protocol.frozen_year, "NSRDB", protocol.data.nsrdb.interval_minutes)
    return _resample_nsrdb_to_analysis_resolution(native, protocol)


def _prepare_oedi_load(path: Path, protocol: FrozenProtocol, sizing: SystemSizing) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing OEDI raw file: {path}")
    field = "out.electricity.total.energy_consumption"
    raw = pd.read_csv(path, usecols=["timestamp", field])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    source = pd.DataFrame({"load_kwh_interval": raw.set_index("timestamp")[field]})
    source = _map_nonleap_calendar(source, protocol.frozen_year, "OEDI")
    # This is a representative profile, not an observed colocated time series. Its
    # AMY clock labels are assigned to the nominal UTC study grid after validation.
    source.index = pd.DatetimeIndex(source.index).tz_localize("UTC")
    aligned = _full_claim_year(source, protocol.frozen_year, "OEDI", 15)
    unscaled_kw = aligned["load_kwh_interval"] / 0.25
    peak = float(unscaled_kw.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("OEDI profile has no positive scalable peak.")
    return pd.DataFrame({"load_kw": unscaled_kw * (sizing.peak_load_kw / peak)}, index=aligned.index)


def _prepare_caiso_prices(directory: Path, protocol: FrozenProtocol) -> pd.DataFrame:
    prices, _ = resolve_caiso_source(protocol, directory.parent)
    hourly = _full_source_year(prices, protocol.frozen_year, "CAISO", 60)
    claim_hourly = drop_excluded_dates(hourly)
    return _repeat_hourly_dam_to_15min(claim_hourly, protocol.frozen_year)


def _map_nonleap_calendar(frame: pd.DataFrame, target_year: int, source: str) -> pd.DataFrame:
    del source  # Retained for call-site diagnostics and compatibility.
    index = pd.DatetimeIndex(frame.index)
    remapped = frame.loc[~((index.month == 2) & (index.day == 29))].copy()
    source_index = pd.DatetimeIndex(remapped.index)
    remapped.index = pd.DatetimeIndex([timestamp.replace(year=target_year) for timestamp in source_index])
    return remapped.sort_index()


def _resample_nsrdb_to_analysis_resolution(frame: pd.DataFrame, protocol: FrozenProtocol) -> pd.DataFrame:
    source_grid = claim_index(protocol.frozen_year, protocol.analysis_resolution_minutes)
    if protocol.data.nsrdb.interval_minutes == protocol.analysis_resolution_minutes:
        normalized = frame.reindex(source_grid)
    else:
        normalized = frame.reindex(source_grid).interpolate(method="time", limit_area="inside")
        normalized = normalized.ffill(
            limit=(protocol.data.nsrdb.interval_minutes // protocol.analysis_resolution_minutes) - 1
        )
    if normalized.isna().any().any():
        raise ValueError("NSRDB resolution normalization left missing realized-resource values.")
    return normalized


def _repeat_hourly_dam_to_15min(hourly: pd.DataFrame, year: int) -> pd.DataFrame:
    expected_hourly = claim_index(year, 60)
    if not hourly.index.equals(expected_hourly) or hourly.isna().any().any():
        raise ValueError("CAISO DAM price has a missing claim-bearing hour; interpolation is forbidden.")
    target = claim_index(year, 15)
    values = hourly.reindex(target.floor("h"))
    values.index = target
    if values.isna().any().any():
        raise ValueError("CAISO hourly repeat left a missing fifteen-minute price.")
    return values


def _full_source_year(frame: pd.DataFrame, year: int, source: str, minutes: int) -> pd.DataFrame:
    if frame.index.has_duplicates:
        raise ValueError(f"{source} has duplicate timestamps.")
    expected = source_index(year, minutes)
    normalized = frame.sort_index().reindex(expected)
    if normalized.isna().any().any():
        missing = int(normalized.isna().any(axis=1).sum())
        raise ValueError(f"{source} has {missing} missing raw {minutes}-minute timestamps in {year}.")
    return normalized


def _full_claim_year(frame: pd.DataFrame, year: int, source: str, minutes: int) -> pd.DataFrame:
    if frame.index.has_duplicates:
        raise ValueError(f"{source} has duplicate timestamps.")
    expected = claim_index(year, minutes)
    normalized = frame.sort_index().reindex(expected)
    if normalized.isna().any().any():
        missing = int(normalized.isna().any(axis=1).sum())
        raise ValueError(f"{source} has {missing} missing claim-bearing {minutes}-minute timestamps in {year}.")
    return normalized


def _raw_hashes(raw: Path, protocol: FrozenProtocol) -> dict[str, dict[str, Any]]:
    oedi = raw / "oedi" / f"ca_{protocol.data.oedi.building_type.lower()}.csv"
    caiso = {path.name: sha256_file(path) for path in sorted((raw / "caiso").glob("*.zip"))}
    hashes: dict[str, dict[str, Any]] = {}
    for location in protocol.locations:
        nsrdb = raw / "nsrdb" / f"{location.location_id}_{protocol.frozen_year}.csv"
        hashes[location.location_id] = {
            "nsrdb": sha256_file(nsrdb),
            "oedi": sha256_file(oedi),
            "caiso_archives": caiso,
        }
    return hashes


def _write_parquet_atomic(table: pd.DataFrame, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".partial")
    table.to_parquet(partial, index=False, engine="pyarrow")
    partial.replace(target)


def _write_metadata(
    target: Path, protocol: FrozenProtocol, location_id: str, raw_hashes: dict[str, Any]
) -> None:
    metadata = {
        "contract_id": "DATA_CONTRACT_V1_2",
        "protocol_version": protocol.protocol_version,
        "location_id": location_id,
        "rows": int(len(pd.read_parquet(target, columns=["timestamp_utc"]))),
        "timezone": "UTC",
        "resolution_minutes": protocol.analysis_resolution_minutes,
        "processed_sha256": sha256_file(target),
        "raw_hashes": raw_hashes,
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
