"""Official CAISO Historical OASIS bulk import with an explicit source decision."""

from __future__ import annotations

import json
import zipfile
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calendar_policy import claim_index, source_index
from .config import CAISOConfig, FrozenProtocol
from .provenance import sha256_file, utc_now


def import_caiso_bulk(protocol: FrozenProtocol, project_root: str | Path) -> dict[str, Any]:
    """Validate manually downloaded official bulk files and select them only as a whole year."""
    root = Path(project_root)
    raw = root / "data" / "raw"
    bulk_paths = _data_paths(raw / "caiso_bulk")
    api_paths = _data_paths(raw / "caiso")
    report: dict[str, Any] = {
        "report_type": "CAISO_BULK_IMPORT_REPORT",
        "created_utc": utc_now(),
        "status": "blocked",
        "selected_source": None,
        "passed": False,
        "errors": [],
        "bulk_files": [str(path) for path in bulk_paths],
    }
    if not bulk_paths:
        report["errors"].append("No official CAISO bulk files are present in data/raw/caiso_bulk/.")
        return _write_bulk_report(root, report)
    bulk, bulk_validation = read_caiso_price_files(bulk_paths, protocol.data.caiso, protocol.frozen_year)
    api, api_validation = read_caiso_price_files(api_paths, protocol.data.caiso, protocol.frozen_year)
    report["bulk_validation"] = bulk_validation
    report["api_validation"] = api_validation
    _write_import_provenance(bulk_paths)
    overlap = _compare_overlap(api, bulk)
    report["api_bulk_overlap"] = overlap
    complete = _complete_source_year(bulk, protocol.frozen_year)
    report["bulk_complete_source_year"] = complete
    if not bulk_validation["passed"]:
        report["errors"].append("Bulk files failed CAISO schema, market, report, node, unit, or timestamp checks.")
    if not complete:
        report["errors"].append("Bulk files do not form one complete 2024 source-year price series.")
    if not overlap["passed"]:
        report["errors"].append("Bulk/API overlap is not an exact hourly price match.")
    if not report["errors"]:
        decision = {
            "decision_version": "CAISO_SOURCE_DECISION_V1",
            "created_utc": utc_now(),
            "selected_source": "official_historical_oasis_bulk",
            "report": "PRC_LMP",
            "market_run_id": protocol.data.caiso.market_run_id,
            "node": protocol.data.caiso.node,
            "year": protocol.frozen_year,
            "raw_hourly_rows": len(bulk),
            "claim_bearing_hourly_rows": len(claim_index(protocol.frozen_year, 60)),
            "source_files": [_file_record(path) for path in bulk_paths],
            "api_overlap": overlap,
            "mixing_policy": "No API rows are used in the selected bulk annual series.",
        }
        (raw / "caiso_source_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["selected_source"] = decision["selected_source"]
        report["passed"] = True
        report["status"] = "pass"
    return _write_bulk_report(root, report)


def resolve_caiso_source(protocol: FrozenProtocol, raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return prices from exactly one complete official annual source mode."""
    raw = Path(raw_dir)
    decision_path = raw / "caiso_source_decision.json"
    if not decision_path.exists():
        raise ValueError("CAISO source decision is absent; API and bulk data may not be mixed implicitly.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected = decision.get("selected_source")
    if selected == "official_historical_oasis_bulk":
        paths = _data_paths(raw / "caiso_bulk")
    elif selected == "official_oasis_api":
        paths = _data_paths(raw / "caiso")
    else:
        raise ValueError("CAISO source decision does not select an allowed official source mode.")
    prices, validation = read_caiso_price_files(paths, protocol.data.caiso, protocol.frozen_year)
    if not validation["passed"] or not _complete_source_year(prices, protocol.frozen_year):
        raise ValueError("Selected CAISO source is not a complete validated 2024 annual series.")
    return prices, decision


def select_complete_api_source(protocol: FrozenProtocol, raw_dir: str | Path) -> None:
    """Write an API selection only when the official API itself forms the whole year."""
    raw = Path(raw_dir)
    paths = _data_paths(raw / "caiso")
    prices, validation = read_caiso_price_files(paths, protocol.data.caiso, protocol.frozen_year)
    if not validation["passed"] or not _complete_source_year(prices, protocol.frozen_year):
        return
    decision = {
        "decision_version": "CAISO_SOURCE_DECISION_V1",
        "created_utc": utc_now(),
        "selected_source": "official_oasis_api",
        "report": "PRC_LMP",
        "market_run_id": protocol.data.caiso.market_run_id,
        "node": protocol.data.caiso.node,
        "year": protocol.frozen_year,
        "raw_hourly_rows": len(prices),
        "claim_bearing_hourly_rows": len(claim_index(protocol.frozen_year, 60)),
        "source_files": [_file_record(path) for path in paths],
        "mixing_policy": "No bulk rows are used in the selected API annual series.",
    }
    (raw / "caiso_source_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_caiso_price_files(
    paths: list[Path], config: CAISOConfig, year: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse only CAISO OASIS PRC_LMP DAM LMP_PRC records from CSV or ZIP files."""
    reports: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            table = _read_table(path)
            price, report = _validate_table(table, path, config)
            reports.append(report)
            if report["passed"]:
                frames.append(price)
        except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
            reports.append({"path": str(path), "passed": False, "errors": [str(exc)]})
    if not frames:
        return pd.DataFrame(columns=["day_ahead_price_usd_mwh"]), {"passed": False, "files": reports}
    merged = pd.concat(frames).sort_index()
    lower = pd.Timestamp(year, 1, 1, tz="UTC")
    upper = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    merged = merged[(merged.index >= lower) & (merged.index < upper)]
    duplicate_count = int(merged.index.duplicated().sum())
    passed = bool(all(report["passed"] for report in reports) and duplicate_count == 0)
    return merged, {"passed": passed, "files": reports, "duplicate_timestamps": duplicate_count}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        payload = path.read_bytes()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if len(names) == 1:
                selected = names[0]
            else:
                matches = [name for name in names if "_PRC_LMP_DAM_LMP" in Path(name).name]
                if len(matches) != 1:
                    raise ValueError("CAISO archive must contain exactly one PRC_LMP_DAM_LMP data file.")
                selected = matches[0]
            with archive.open(selected) as handle:
                payload = handle.read()
    else:
        raise ValueError("Bulk import accepts only official CSV or ZIP files.")
    if payload.lstrip().lower().startswith((b"<html", b"<!doctype", b"<?xml", b"<error")):
        raise ValueError("CAISO file is an HTML/XML error payload, not data.")
    return pd.read_csv(BytesIO(payload))


def _validate_table(table: pd.DataFrame, path: Path, config: CAISOConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"INTERVALSTARTTIME_GMT", "MARKET_RUN_ID", "NODE_ID", "LMP_TYPE", "XML_DATA_ITEM", "MW"}
    errors: list[str] = []
    if missing := sorted(required.difference(table.columns)):
        return pd.DataFrame(), {"path": str(path), "passed": False, "errors": [f"Missing fields: {missing}"]}
    report_column = next((column for column in ("REPORT", "REPORT_NAME", "QUERYNAME") if column in table.columns), None)
    if report_column is not None and not table[report_column].astype(str).eq("PRC_LMP").all():
        errors.append("Report field is not PRC_LMP.")
    candidate = table[(table["LMP_TYPE"] == "LMP") & (table["XML_DATA_ITEM"] == "LMP_PRC")].copy()
    if candidate.empty:
        errors.append("No LMP_PRC rows are present.")
    if not candidate.empty and not candidate["MARKET_RUN_ID"].eq(config.market_run_id).all():
        errors.append("Market run is not DAM.")
    price = candidate[candidate["NODE_ID"].eq(config.node)].copy()
    if not candidate.empty and price.empty:
        errors.append("Frozen node/hub is absent from CAISO source file.")
    if "UNIT" in price.columns and not price["UNIT"].astype(str).str.upper().eq("USD/MWH").all():
        errors.append("Reported price unit is not USD/MWh.")
    price["timestamp"] = pd.to_datetime(price["INTERVALSTARTTIME_GMT"], utc=True, errors="coerce")
    price["day_ahead_price_usd_mwh"] = pd.to_numeric(price["MW"], errors="coerce")
    if price["timestamp"].isna().any() or not np.isfinite(price["day_ahead_price_usd_mwh"]).all():
        errors.append("Invalid timestamp or non-numeric USD/MWh price.")
    price = price.set_index("timestamp")[["day_ahead_price_usd_mwh"]].sort_index()
    return price, {
        "path": str(path),
        "sha256": sha256_file(path),
        "report": "PRC_LMP",
        "market_run_id": config.market_run_id,
        "node": config.node,
        "unit": "USD/MWh",
        "rows": len(price),
        "passed": not errors,
        "errors": errors,
    }


def _complete_source_year(frame: pd.DataFrame, year: int) -> bool:
    return bool(frame.index.equals(source_index(year, 60)) and frame.index.is_unique and not frame.isna().any().any())


def _compare_overlap(api: pd.DataFrame, bulk: pd.DataFrame) -> dict[str, Any]:
    if api.empty:
        return {"overlap_hours": 0, "exact_match": True, "passed": True, "mismatched_hours": []}
    shared = api.join(bulk, how="inner", lsuffix="_api", rsuffix="_bulk")
    mismatches = [
        pd.Timestamp(str(timestamp)).isoformat()
        for timestamp, row in shared.iterrows()
        if Decimal(str(row.iloc[0])) != Decimal(str(row.iloc[1]))
    ]
    return {
        "overlap_hours": len(shared),
        "exact_match": not mismatches,
        "passed": not mismatches,
        "mismatched_hours": mismatches,
    }


def _data_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([*directory.rglob("*.zip"), *directory.rglob("*.csv")])


def _write_import_provenance(paths: list[Path]) -> None:
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".source.json")
        if sidecar.exists():
            continue
        record = {
            "source": "CAISO Historical OASIS Data Downloader official bulk import",
            "query_parameters": {"report": "PRC_LMP", "market_run_id": "DAM"},
            "retrieval_utc": utc_now(),
            "original_filename": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "license_or_source_note": "User-imported official CAISO Historical OASIS Data Downloader file; validation required before use.",
        }
        sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}


def _write_bulk_report(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    destination = root / "reports" / "CAISO_BULK_IMPORT_REPORT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
