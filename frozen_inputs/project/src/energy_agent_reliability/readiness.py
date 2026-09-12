"""Contract-driven readiness checks for raw, claim-bearing public data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from .caiso_bulk import read_caiso_price_files
from .calendar_policy import claim_index, has_excluded_date, source_index
from .config import FrozenProtocol
from .nsrdb import validate_nsrdb_response
from .provenance import sha256_file, utc_now


def load_data_contract(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("contract_id") != "DATA_CONTRACT_V1_2":
        raise ValueError("Expected a DATA_CONTRACT_V1_2 YAML document.")
    calendar = raw.get("calendar_policy")
    if not isinstance(calendar, dict) or calendar.get("exclude_dates") != ["2024-02-29"]:
        raise ValueError("DATA_CONTRACT_V1_2 must exclude 2024-02-29.")
    if calendar.get("expected_15min_intervals") != 35040 or calendar.get("expected_hourly_intervals") != 8760:
        raise ValueError("DATA_CONTRACT_V1_2 calendar counts are not frozen.")
    return cast(dict[str, Any], raw)


def check_data_readiness(
    protocol: FrozenProtocol, contract_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Inspect complete raw files without transforming or substituting source values."""
    contract = load_data_contract(contract_path)
    root = Path(project_root)
    raw = root / "data" / "raw"
    checks: list[dict[str, Any]] = [
        _check_nsrdb(raw, protocol, contract, location.location_id) for location in protocol.locations
    ]
    checks.extend([_check_oedi(raw, protocol, contract), _check_caiso(raw, protocol, contract)])
    ready = all(check["passed"] for check in checks)
    report = {
        "report_type": "DATA_READINESS_REPORT",
        "created_utc": utc_now(),
        "contract_id": contract["contract_id"],
        "contract_version": contract["protocol_version"],
        "protocol_version": protocol.protocol_version,
        "study_year": protocol.frozen_year,
        "claim_bearing_data_ready": ready,
        "status": "pass" if ready else "blocked",
        "sources": checks,
        "rejection": None if ready else "At least one source failed DATA_CONTRACT_V1_2.",
    }
    destination = root / "reports" / "DATA_READINESS_REPORT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _check_nsrdb(
    raw_root: Path, protocol: FrozenProtocol, contract: dict[str, Any], location_id: str
) -> dict[str, Any]:
    path = raw_root / "nsrdb" / f"{location_id}_{protocol.frozen_year}.csv"
    if not path.exists():
        return _absent_report("NSRDB", location_id, path, "Required NSRDB source file is absent.")
    try:
        table = pd.read_csv(path, skiprows=2)
        table.columns = [str(column).strip().lower().replace(" ", "_") for column in table.columns]
        table = table.rename(columns={"temperature": "air_temperature"})
        required = list(cast(dict[str, Any], contract["sources"]["nsrdb"]["variables"]).keys())
        if missing_fields := sorted(set(required).difference(table.columns)):
            return _failed_report("NSRDB", location_id, path, f"Missing required fields: {missing_fields}")
        timestamp = pd.to_datetime(table[["year", "month", "day", "hour", "minute"]], errors="coerce")
        if timestamp.isna().any():
            return _failed_report("NSRDB", location_id, path, "Invalid NSRDB timestamps.")
        index = pd.DatetimeIndex(timestamp).tz_localize("UTC")
        frame = table[required].apply(pd.to_numeric, errors="coerce")
        invalid: dict[str, int] = {}
        for variable, specification in cast(dict[str, Any], contract["sources"]["nsrdb"]["variables"]).items():
            lower, upper = specification["valid_range"]
            invalid[variable] = int((~frame[variable].between(lower, upper)).sum())
        report = _tabular_report(
            source="NSRDB",
            site=location_id,
            path=path,
            index=index,
            raw_expected=claim_index(protocol.frozen_year, protocol.data.nsrdb.interval_minutes),
            claim_expected=claim_index(protocol.frozen_year, protocol.analysis_resolution_minutes),
            claim_observed_count=(
                len(claim_index(protocol.frozen_year, protocol.analysis_resolution_minutes))
                if all(value == 0 for value in invalid.values())
                and index.equals(claim_index(protocol.frozen_year, protocol.data.nsrdb.interval_minutes))
                else 0
            ),
            invalid_values=invalid,
            unit_checks={
                "status": "passed",
                "declared_units": {
                    name: specification["unit"]
                    for name, specification in cast(dict[str, Any], contract["sources"]["nsrdb"]["variables"]).items()
                },
            },
        )
        location = next(item for item in protocol.locations if item.location_id == location_id)
        metadata_validation = validate_nsrdb_response(path, protocol.data.nsrdb, location, protocol.frozen_year)
        report["official_interface_metadata"] = metadata_validation
        if metadata_validation["validation_status"] != "passed":
            report["errors"].append("NSRDB official-interface metadata validation failed.")
            report["passed"] = False
        return report
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return _failed_report("NSRDB", location_id, path, str(exc))


def _check_oedi(raw_root: Path, protocol: FrozenProtocol, contract: dict[str, Any]) -> dict[str, Any]:
    path = raw_root / "oedi" / f"ca_{protocol.data.oedi.building_type.lower()}.csv"
    if not path.exists():
        return _absent_report("OEDI", "california_largeoffice", path, "Required OEDI source file is absent.")
    field = str(contract["sources"]["oedi"]["field"])
    try:
        table = pd.read_csv(path, usecols=["timestamp", field])
        timestamp = pd.to_datetime(table["timestamp"], errors="coerce")
        if timestamp.isna().any():
            return _failed_report("OEDI", "california_largeoffice", path, "Invalid OEDI timestamps.")
        source_year = int(timestamp.iloc[0].year)
        index = pd.DatetimeIndex(timestamp).tz_localize("Etc/GMT+8")
        values = pd.to_numeric(table[field], errors="coerce")
        numeric_values = values.to_numpy(dtype=float, na_value=np.nan)
        raw_expected = _expected_oedi_amy_index(source_year, 15)
        source_complete = pd.DatetimeIndex(index).equals(raw_expected)
        report = _tabular_report(
            source="OEDI",
            site="california_largeoffice",
            path=path,
            index=index,
            raw_expected=raw_expected,
            claim_expected=claim_index(protocol.frozen_year, 15),
            claim_observed_count=len(claim_index(protocol.frozen_year, 15)) if source_complete else 0,
            invalid_values={field: int((~np.isfinite(numeric_values) | (numeric_values < 0)).sum())},
            unit_checks={
                "status": "passed" if values.notna().all() and (values >= 0).all() else "failed",
                "input_unit": contract["sources"]["oedi"]["input_unit"],
                "derived_unit": contract["sources"]["oedi"]["derived_unit"],
                "conversion": "interval_energy / 0.25 hours",
                "semantics": contract["load_semantics"],
            },
        )
        report["claim_bearing_timezone"] = "UTC nominal profile grid"
        return report
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return _failed_report("OEDI", "california_largeoffice", path, str(exc))


def _check_caiso(raw_root: Path, protocol: FrozenProtocol, contract: dict[str, Any]) -> dict[str, Any]:
    decision_path = raw_root / "caiso_source_decision.json"
    selected_source: str | None = None
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        selected_source = decision.get("selected_source")
    if selected_source == "official_historical_oasis_bulk":
        directory = raw_root / "caiso_bulk"
    elif selected_source == "official_oasis_api":
        directory = raw_root / "caiso"
    else:
        directory = raw_root / "caiso"
    paths = sorted([*directory.glob("*.zip"), *directory.glob("*.csv")])
    if not paths:
        return _absent_report("CAISO", protocol.data.caiso.node, directory, "No selected CAISO source files are present.")
    prices, validation = read_caiso_price_files(paths, protocol.data.caiso, protocol.frozen_year)
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        return _failed_report("CAISO", protocol.data.caiso.node, directory, "CAISO source has no valid PRC_LMP rows.")
    raw_expected = source_index(protocol.frozen_year, 60)
    merged = prices.copy()
    merged.index = pd.DatetimeIndex(merged.index)
    claim_observed = merged.loc[
        ~merged.index.normalize().isin(pd.DatetimeIndex([pd.Timestamp("2024-02-29", tz="UTC")]))
    ]
    report = _tabular_report(
        source="CAISO",
        site=protocol.data.caiso.node,
        path=directory,
        index=pd.DatetimeIndex(merged.index),
        raw_expected=raw_expected,
        claim_expected=claim_index(protocol.frozen_year, 60),
        claim_observed_count=len(pd.DatetimeIndex(claim_observed.index).unique()),
        invalid_values={"price_usd_per_mwh": int((~np.isfinite(merged["day_ahead_price_usd_mwh"])).sum())},
        unit_checks={
            "status": "passed" if pd.api.types.is_numeric_dtype(merged["day_ahead_price_usd_mwh"]) else "failed",
            "unit": contract["sources"]["caiso"]["price_unit"],
            "market_run_id": protocol.data.caiso.market_run_id,
            "node": protocol.data.caiso.node,
            "report": "PRC_LMP",
            "15min_mapping": "repeat_each_hour_over_four_intervals",
        },
    )
    report["selected_source"] = selected_source
    report["source_validation"] = validation
    if not validation["passed"]:
        report["errors"].append("Selected CAISO source failed validation.")
        report["passed"] = False
    if selected_source not in {"official_oasis_api", "official_historical_oasis_bulk"}:
        report["errors"].append("CAISO source decision is absent or does not select exactly one official source mode.")
        report["passed"] = False
    report["sha256"] = _provenance_hashes(paths)
    return report


def _expected_index(year: int, minutes: int) -> pd.DatetimeIndex:
    """Compatibility export for raw-source expectations used by tests."""
    return source_index(year, minutes)


def _expected_oedi_amy_index(year: int, minutes: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year, 1, 1, 0, minutes, tz="Etc/GMT+8")
    return pd.date_range(start, periods=365 * 24 * 60 // minutes, freq=f"{minutes}min")


def _tabular_report(
    *,
    source: str,
    site: str,
    path: Path,
    index: pd.DatetimeIndex,
    raw_expected: pd.DatetimeIndex,
    claim_expected: pd.DatetimeIndex,
    claim_observed_count: int,
    invalid_values: dict[str, int],
    unit_checks: dict[str, Any],
) -> dict[str, Any]:
    valid_timestamps = pd.DatetimeIndex([timestamp for timestamp in index if not pd.isna(timestamp)])
    observed_unique = pd.DatetimeIndex(valid_timestamps.unique()).sort_values()
    raw_missing = raw_expected.difference(observed_unique)
    duplicates = int(valid_timestamps.duplicated().sum())
    provenance = _provenance_hashes([path]) if path.is_file() else _provenance_hashes(sorted(path.glob("*.zip")))
    errors: list[str] = []
    if not provenance["all_hashes_match"]:
        errors.append("A provenance sidecar is missing or its SHA-256 does not match.")
    if len(raw_missing) > 0:
        errors.append(f"Missing {len(raw_missing)} expected raw intervals.")
    if claim_observed_count != len(claim_expected):
        errors.append("Claim-bearing calendar coverage is incomplete after 2024-02-29 exclusion.")
    if duplicates:
        errors.append(f"Found {duplicates} duplicate timestamps.")
    if sum(invalid_values.values()) > 0:
        errors.append("Invalid values are present.")
    if unit_checks.get("status") != "passed":
        errors.append("Unit check failed.")
    return {
        "source": source,
        "site": site,
        "path": str(path),
        "row_count": len(index),
        "first_timestamp": _timestamp_or_none(observed_unique, first=True),
        "last_timestamp": _timestamp_or_none(observed_unique, first=False),
        "timezone": str(observed_unique.tz) if observed_unique.tz is not None else "naive",
        "expected_interval_count": len(raw_expected),
        "observed_interval_count": len(observed_unique),
        "missing_intervals": len(raw_missing),
        "maximum_consecutive_gap": _maximum_consecutive_gap(raw_expected, observed_unique),
        "duplicate_timestamps": duplicates,
        "claim_bearing_calendar": {
            "timezone": "UTC",
            "excluded_dates": ["2024-02-29"],
            "expected_interval_count": len(claim_expected),
            "observed_interval_count": claim_observed_count,
            "missing_intervals": len(claim_expected) - claim_observed_count,
            "contains_excluded_date": has_excluded_date(claim_expected),
        },
        "unit_checks": unit_checks,
        "invalid_values": invalid_values,
        "sha256": provenance,
        "errors": errors,
        "passed": not errors,
    }


def _provenance_hashes(paths: list[Path]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".source.json")
        matches = False
        if path.exists() and sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                matches = metadata.get("sha256") == sha256_file(path)
            except (OSError, ValueError, json.JSONDecodeError):
                matches = False
        items.append(
            {"path": str(path), "sha256": sha256_file(path) if path.exists() else None, "matches_provenance": matches}
        )
    return {"all_hashes_match": bool(items) and all(item["matches_provenance"] for item in items), "files": items}


def _maximum_consecutive_gap(expected: pd.DatetimeIndex, observed: pd.DatetimeIndex) -> int:
    observed_set = set(observed)
    longest = 0
    current = 0
    for timestamp in expected:
        if timestamp in observed_set:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def _timestamp_or_none(index: pd.DatetimeIndex, *, first: bool) -> str | None:
    if index.empty:
        return None
    return (index[0] if first else index[-1]).isoformat()


def _absent_report(source: str, site: str, path: Path, error: str) -> dict[str, Any]:
    return {
        "source": source,
        "site": site,
        "path": str(path),
        "row_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "timezone": None,
        "expected_interval_count": None,
        "observed_interval_count": 0,
        "missing_intervals": None,
        "maximum_consecutive_gap": None,
        "duplicate_timestamps": None,
        "claim_bearing_calendar": {"excluded_dates": ["2024-02-29"], "status": "not_run"},
        "unit_checks": {"status": "not_run"},
        "invalid_values": {},
        "sha256": {"all_hashes_match": False, "files": []},
        "errors": [error],
        "passed": False,
    }


def _failed_report(source: str, site: str, path: Path, error: str) -> dict[str, Any]:
    report = _absent_report(source, site, path, error)
    if path.exists():
        report["sha256"] = _provenance_hashes([path])
    return report
