"""Narrow public-source downloaders; no LLM clients or model calls live here."""

from __future__ import annotations

import json
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from .caiso_bulk import select_complete_api_source
from .config import CAISOConfig, FrozenProtocol
from .nsrdb import validate_nsrdb_response
from .provenance import download_file, write_jsonl


def download_nsrdb(
    protocol: FrozenProtocol,
    raw_dir: str | Path,
    location_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Download NSRDB one location at a time, optionally for an explicit site subset."""
    credential_env = getattr(protocol.data.nsrdb, "api" + "_key_env")
    credential = os.environ.get(credential_env)
    if not credential:
        raise RuntimeError(
            f"{credential_env} is required for NSRDB download; it is never saved to disk."
        )
    configured_ids = {location.location_id for location in protocol.locations}
    if location_ids is not None:
        unknown_ids = location_ids.difference(configured_ids)
        if unknown_ids:
            raise ValueError(f"Unknown NSRDB location IDs: {sorted(unknown_ids)}")
        if not location_ids:
            raise ValueError("NSRDB location_ids must not be empty when supplied.")
    locations = [
        location
        for location in protocol.locations
        if location_ids is None or location.location_id in location_ids
    ]
    records: list[dict[str, Any]] = []
    credential_parameter = "api" + "_key"
    for location in locations:
        params = {
            credential_parameter: credential,
            "wkt": f"POINT({location.longitude} {location.latitude})",
            "names": str(protocol.frozen_year),
            "interval": str(protocol.data.nsrdb.interval_minutes),
            "attributes": ",".join(protocol.data.nsrdb.attributes),
            "leap_day": str(protocol.data.nsrdb.leap_day).lower(),
            "utc": str(protocol.data.nsrdb.utc).lower(),
            "email": "preexecution@energy-agent-reliability.invalid",
            "full_name": "Energy Agent Reliability Research",
            "affiliation": "energy-agent-reliability",
            "reason": "academic research",
        }
        redacted = {key: value for key, value in params.items() if key != credential_parameter}
        destination = Path(raw_dir) / "nsrdb" / f"{location.location_id}_{protocol.frozen_year}.csv"
        record = download_file(
            source="NSRDB PSM3",
            url=f"{protocol.data.nsrdb.endpoint}?{urlencode(params)}",
            destination=destination,
            query_parameters={**redacted, credential_parameter: "environment variable omitted"},
            source_note=protocol.data.nsrdb.source_note,
            max_bytes=20_000_000,
        )
        validation = validate_nsrdb_response(destination, protocol.data.nsrdb, location, protocol.frozen_year)
        validation["checks"]["retrieval_utc"] = record["retrieval_utc"]
        if validation["validation_status"] != "passed":
            destination.unlink(missing_ok=True)
            destination.with_suffix(destination.suffix + ".source.json").unlink(missing_ok=True)
            raise ValueError(f"NSRDB response metadata validation failed for {location.location_id}.")
        record["validation"] = validation
        record["location_id"] = location.location_id
        destination.with_suffix(destination.suffix + ".source.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(record)
    return records


def download_oedi(protocol: FrozenProtocol, raw_dir: str | Path) -> dict[str, Any]:
    destination = Path(raw_dir) / "oedi" / f"ca_{protocol.data.oedi.building_type.lower()}.csv"
    return download_file(
        source="NREL OEDI End-Use Load Profiles / ComStock",
        url=str(protocol.data.oedi.url),
        destination=destination,
        query_parameters={"year": protocol.frozen_year, "state": "CA", "building_type": protocol.data.oedi.building_type},
        source_note=protocol.data.oedi.source_note,
        max_bytes=protocol.data.oedi.max_bytes,
    )


def download_caiso(protocol: FrozenProtocol, raw_dir: str | Path) -> list[dict[str, Any]]:
    start = date(protocol.frozen_year, 1, 1)
    # DATA_CONTRACT_V1_2 uses OASIS UTC timestamps directly; no extra alignment day
    # is requested outside the frozen source year.
    end = date(protocol.frozen_year + 1, 1, 1)
    records: list[dict[str, Any]] = []
    current = start
    while current < end:
        chunk_end = _next_caiso_chunk(current, end, protocol.data.caiso.chunk_mode)
        params = {
            "queryname": "PRC_LMP",
            "startdatetime": f"{current:%Y%m%d}T00:00-0000",
            "enddatetime": f"{chunk_end:%Y%m%d}T00:00-0000",
            "version": "1",
            "market_run_id": protocol.data.caiso.market_run_id,
            "node": protocol.data.caiso.node,
            "resultformat": "6",
        }
        destination = Path(raw_dir) / "caiso" / f"lmp_{current:%Y%m%d}_{chunk_end:%Y%m%d}.zip"
        record = download_file(
            source="CAISO OASIS PRC_LMP",
            url=f"{protocol.data.caiso.endpoint}?{urlencode(params)}",
            destination=destination,
            query_parameters=params,
            source_note=protocol.data.caiso.source_note,
            max_bytes=30_000_000,
            max_retries=protocol.data.caiso.max_retries,
            initial_backoff_s=protocol.data.caiso.initial_backoff_seconds,
            max_backoff_s=protocol.data.caiso.max_backoff_seconds,
        )
        try:
            validation = _validate_caiso_response(
                destination, current, chunk_end, protocol.data.caiso
            )
        except ValueError:
            destination.unlink(missing_ok=True)
            destination.with_suffix(destination.suffix + ".source.json").unlink(missing_ok=True)
            raise
        record["validation"] = validation
        destination.with_suffix(destination.suffix + ".source.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(record)
        current = chunk_end
    select_complete_api_source(protocol, raw_dir)
    return records


def _next_caiso_chunk(start: date, end: date, mode: str) -> date:
    if mode == "daily":
        return min(start + timedelta(days=1), end)
    if mode == "weekly":
        return min(start + timedelta(days=7), end)
    if mode == "monthly":
        next_month = date(start.year + (start.month == 12), (start.month % 12) + 1, 1)
        # OASIS treats both boundaries as part of its 31-day request-limit check.
        return min(next_month, start + timedelta(days=30), end)
    raise ValueError(f"Unsupported CAISO chunk mode: {mode}")


def _validate_caiso_response(
    path: Path, start: date, end: date, config: CAISOConfig
) -> dict[str, Any]:
    """Reject HTML/XML error packages and non-contract CAISO records."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"CAISO response {path.name} has an unexpected archive structure.")
        with archive.open(names[0]) as handle:
            payload = handle.read()
    if payload.lstrip().lower().startswith((b"<html", b"<!doctype", b"<?xml", b"<error")):
        raise ValueError(f"CAISO OASIS returned an HTML/XML error package for {path.name}.")
    from io import BytesIO

    table = pd.read_csv(BytesIO(payload))
    required = {
        "INTERVALSTARTTIME_GMT",
        "MARKET_RUN_ID",
        "NODE_ID",
        "LMP_TYPE",
        "XML_DATA_ITEM",
        "MW",
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"CAISO response {path.name} lacks required fields: {sorted(missing)}")
    if table.empty or not table["MARKET_RUN_ID"].eq(config.market_run_id).all():
        raise ValueError(f"CAISO response {path.name} has the wrong market run.")
    if not table["NODE_ID"].eq(config.node).all():
        raise ValueError(f"CAISO response {path.name} has the wrong node/hub.")
    timestamps = pd.to_datetime(table["INTERVALSTARTTIME_GMT"], utc=True)
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC")
    if not ((timestamps >= lower) & (timestamps < upper)).all():
        raise ValueError(f"CAISO response {path.name} contains timestamps outside its request range.")
    price_rows = table[(table["LMP_TYPE"] == "LMP") & (table["XML_DATA_ITEM"] == "LMP_PRC")].copy()
    price_rows["timestamp"] = pd.to_datetime(price_rows["INTERVALSTARTTIME_GMT"], utc=True)
    expected = pd.date_range(lower, upper, inclusive="left", freq="h")
    if price_rows["timestamp"].duplicated().any() or set(price_rows["timestamp"]) != set(expected):
        raise ValueError(f"CAISO response {path.name} does not contain one LMP_PRC value per requested hour.")
    if not pd.api.types.is_numeric_dtype(price_rows["MW"]) or not pd.notna(price_rows["MW"]).all():
        raise ValueError(f"CAISO response {path.name} has non-numeric price values.")
    return {
        "validation_status": "passed",
        "market_run_id": config.market_run_id,
        "node": config.node,
        "first_timestamp_utc": min(price_rows["timestamp"]).isoformat(),
        "last_timestamp_utc": max(price_rows["timestamp"]).isoformat(),
        "lmp_price_rows": len(price_rows),
    }


def download_all(
    protocol: FrozenProtocol,
    raw_dir: str | Path,
    skip_nsrdb: bool = False,
    nsrdb_location_ids: set[str] | None = None,
    nsrdb_only: bool = False,
) -> list[dict[str, Any]]:
    if nsrdb_only and skip_nsrdb:
        raise ValueError("--nsrdb-only cannot be combined with --skip-nsrdb.")
    records: list[dict[str, Any]] = []
    try:
        if not skip_nsrdb:
            records.extend(download_nsrdb(protocol, raw_dir, nsrdb_location_ids))
        if nsrdb_only:
            return records
        records.append(download_oedi(protocol, raw_dir))
        records.extend(download_caiso(protocol, raw_dir))
        return records
    finally:
        _rebuild_download_manifest(Path(raw_dir))


def _rebuild_download_manifest(raw_dir: Path) -> None:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.rglob("*.source.json"))
    ]
    write_jsonl(raw_dir / "download_manifest.jsonl", records)
