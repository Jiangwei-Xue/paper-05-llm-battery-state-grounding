"""Report fixed-calendar alignment before and after processed-table construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calendar_policy import claim_index, has_excluded_date
from .config import FrozenProtocol
from .provenance import sha256_file, utc_now
from .readiness import check_data_readiness, load_data_contract


def check_calendar_alignment(
    protocol: FrozenProtocol, contract_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Write a transparent alignment report without manufacturing missing data."""
    root = Path(project_root)
    contract = load_data_contract(contract_path)
    readiness = check_data_readiness(protocol, contract_path, root)
    expected_15 = claim_index(protocol.frozen_year, 15)
    expected_hourly = claim_index(protocol.frozen_year, 60)
    source_checks = {
        f"{item['source']}:{item['site']}": {
            "raw_passed": item["passed"],
            "raw_timezone": item["timezone"],
            "raw_row_count": item["row_count"],
            "claim_bearing_calendar": item.get("claim_bearing_calendar"),
            "duplicate_timestamps": item["duplicate_timestamps"],
            "maximum_consecutive_gap": item["maximum_consecutive_gap"],
        }
        for item in readiness["sources"]
    }
    processed = _processed_alignment(protocol, root / "data" / "processed", expected_15)
    report = {
        "report_type": "CALENDAR_ALIGNMENT_REPORT",
        "created_utc": utc_now(),
        "contract_id": contract["contract_id"],
        "calendar_policy": contract["calendar_policy"],
        "canonical_reporting_timezone": contract["canonical_timezone"],
        "internal_storage_timezone": contract["internal_storage_timezone"],
        "expected_15min_intervals": len(expected_15),
        "expected_hourly_intervals": len(expected_hourly),
        "cross_frequency_mapping": contract["resampling"]["hourly_dam_price_to_15min"],
        "source_checks": source_checks,
        "calendar_sources_ready": readiness["claim_bearing_data_ready"],
        "processed_alignment": processed,
        "processed_alignment_complete": processed["complete"],
        "passed": readiness["claim_bearing_data_ready"] and processed["complete"],
    }
    report["status"] = "pass" if report["passed"] else "blocked"
    destination = root / "reports" / "CALENDAR_ALIGNMENT_REPORT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _processed_alignment(
    protocol: FrozenProtocol, processed: Path, expected: pd.DatetimeIndex
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for location in protocol.locations:
        path = processed / f"{location.location_id}_{protocol.frozen_year}_15min.parquet"
        files.append(_processed_file_report(path, expected))
    unified = processed / "operational_timeseries_v1.parquet"
    unified_report = _unified_report(unified, expected, len(protocol.locations))
    complete = all(item["passed"] for item in files) and unified_report["passed"]
    return {"files": files, "unified": unified_report, "complete": complete}


def _processed_file_report(path: Path, expected: pd.DatetimeIndex) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "absent", "passed": False, "sha256": None}
    table = pd.read_parquet(path, columns=["timestamp_utc"])
    index = pd.DatetimeIndex(pd.to_datetime(table["timestamp_utc"], utc=True))
    passed = bool(index.equals(expected) and not has_excluded_date(index) and index.is_unique)
    return {
        "path": str(path),
        "status": "checked",
        "rows": len(index),
        "timezone": str(index.tz),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "contains_excluded_date": has_excluded_date(index),
        "expected_grid_match": index.equals(expected),
        "sha256": sha256_file(path),
        "passed": passed,
    }


def _unified_report(path: Path, expected: pd.DatetimeIndex, site_count: int) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "absent", "passed": False, "sha256": None}
    table = pd.read_parquet(path, columns=["timestamp_utc", "site_id"])
    per_site = [
        pd.DatetimeIndex(pd.to_datetime(group["timestamp_utc"], utc=True)).equals(expected)
        for _, group in table.groupby("site_id", sort=True)
    ]
    passed = len(per_site) == site_count and all(per_site)
    return {
        "path": str(path),
        "status": "checked",
        "rows": len(table),
        "expected_rows": len(expected) * site_count,
        "sha256": sha256_file(path),
        "passed": passed and len(table) == len(expected) * site_count,
    }
