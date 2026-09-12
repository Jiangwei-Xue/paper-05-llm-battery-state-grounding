"""Validation of one-point, one-year NSRDB GOES CONUS API responses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .calendar_policy import claim_index, has_excluded_date
from .config import Location, NSRDBConfig
from .provenance import sha256_file


def validate_nsrdb_response(path: Path, config: NSRDBConfig, location: Location, year: int) -> dict[str, Any]:
    """Verify response metadata and all 35,040 non-leap UTC quarter-hour rows."""
    metadata = _read_metadata(path)
    table = pd.read_csv(path, skiprows=2)
    table.columns = [str(column).strip().lower().replace(" ", "_") for column in table.columns]
    table = table.rename(columns={"temperature": "air_temperature"})
    required = set(config.attributes)
    missing = sorted(required.difference(table.columns))
    timestamp = pd.to_datetime(table[["year", "month", "day", "hour", "minute"]], errors="coerce")
    index = pd.DatetimeIndex(timestamp).tz_localize("UTC")
    expected = claim_index(year, config.interval_minutes)
    returned_latitude = _metadata_value(metadata, "latitude")
    returned_longitude = _metadata_value(metadata, "longitude")
    returned_coordinates_match = _coordinates_match(returned_latitude, returned_longitude, location)
    interval_metadata = _metadata_value(metadata, "interval")
    timezone_metadata = _metadata_value(metadata, "time zone", "timezone", "time_zone")
    dataset_metadata = _metadata_value(metadata, "dataset", "source", "version", "data set")
    checks: dict[str, Any] = {
        "dataset_version": {
            "requested": config.dataset,
            "response_metadata": dataset_metadata,
            "endpoint": str(config.endpoint),
            "passed": (config.endpoint.host or "") == "developer.nlr.gov"
            and (config.endpoint.path or "").endswith("nsrdb-GOES-conus-v4-0-0-download.csv"),
        },
        "requested_coordinates": {"latitude": location.latitude, "longitude": location.longitude},
        "returned_coordinates": {
            "latitude": returned_latitude,
            "longitude": returned_longitude,
            "matches_requested": returned_coordinates_match,
        },
        "temporal_interval_minutes": {
            "requested": config.interval_minutes,
            "response_metadata": interval_metadata,
            "observed": _observed_interval_minutes(index),
            "passed": _observed_interval_minutes(index) == config.interval_minutes,
        },
        "utc": {
            "requested": config.utc,
            "response_metadata": timezone_metadata,
            "stored_timezone": "UTC",
            "passed": config.utc and _utc_metadata_compatible(timezone_metadata),
        },
        "year": {"requested": year, "observed": sorted(set(index.year)), "passed": set(index.year) == {year}},
        "leap_day_policy": {
            "requested": config.leap_day,
            "contains_february_29": has_excluded_date(index),
            "passed": not config.leap_day and not has_excluded_date(index),
        },
        "attributes": {"requested": list(config.attributes), "missing": missing, "passed": not missing},
        "row_count": {"expected": len(expected), "observed": len(index), "passed": index.equals(expected)},
        "sha256": sha256_file(path),
    }
    passed = bool(
        checks["dataset_version"]["passed"]
        and returned_coordinates_match
        and checks["temporal_interval_minutes"]["passed"]
        and checks["utc"]["passed"]
        and checks["year"]["passed"]
        and checks["leap_day_policy"]["passed"]
        and checks["attributes"]["passed"]
        and checks["row_count"]["passed"]
    )
    return {"validation_status": "passed" if passed else "failed", "checks": checks}


def _read_metadata(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
        values = next(reader, [])
    return {header.strip().lower(): value.strip() for header, value in zip(headers, values, strict=False)}


def _metadata_value(metadata: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _coordinates_match(latitude: str | None, longitude: str | None, location: Location) -> bool:
    if latitude is None or longitude is None:
        return False
    try:
        return abs(float(latitude) - location.latitude) <= 0.02 and abs(float(longitude) - location.longitude) <= 0.02
    except (TypeError, ValueError):
        return False


def _observed_interval_minutes(index: pd.DatetimeIndex) -> int | None:
    if len(index) < 2 or bool(pd.isna(index).any()):
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    regular = deltas[deltas <= pd.Timedelta(hours=1).total_seconds()]
    if regular.empty or regular.nunique() != 1:
        return None
    return int(regular.iloc[0] // 60)


def _utc_metadata_compatible(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().upper()
    return normalized in {"0", "0.0", "UTC", "GMT"}
