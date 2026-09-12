"""Audit active NSRDB request configuration without reading secret values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import FrozenProtocol
from .provenance import utc_now

_RETIRED_HOST = "developer." + "nrel.gov"


def audit_nsrdb_endpoints(protocol: FrozenProtocol, project_root: str | Path) -> dict[str, Any]:
    """Reject retired active endpoints while allowing historical changelog references."""
    root = Path(project_root)
    active_files = [
        path
        for relative in ("configs", "src", "scripts")
        for path in (root / relative).rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    ]
    retired_references = [
        str(path.relative_to(root))
        for path in active_files
        if _RETIRED_HOST in path.read_text(encoding="utf-8")
    ]
    endpoint = protocol.data.nsrdb.endpoint
    expected_attributes = [
        "ghi",
        "dni",
        "dhi",
        "air_temperature",
        "wind_speed",
        "solar_zenith_angle",
        "clearsky_ghi",
    ]
    passed = bool(
        not retired_references
        and endpoint.host == "developer.nlr.gov"
        and endpoint.path == "/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
        and protocol.data.nsrdb.dataset == "nsrdb-GOES-conus-v4-0-0"
        and protocol.data.nsrdb.attributes == expected_attributes
        and protocol.data.nsrdb.interval_minutes == 15
        and protocol.data.nsrdb.utc
        and not protocol.data.nsrdb.leap_day
    )
    report: dict[str, Any] = {
        "report_type": "NSRDB_ENDPOINT_AUDIT",
        "created_utc": utc_now(),
        "status": "pass" if passed else "fail",
        "active_retired_endpoint_references": retired_references,
        "endpoint": str(endpoint),
        "dataset": protocol.data.nsrdb.dataset,
        "request_parameters": {
            "names": str(protocol.frozen_year),
            "interval": protocol.data.nsrdb.interval_minutes,
            "utc": protocol.data.nsrdb.utc,
            "leap_day": protocol.data.nsrdb.leap_day,
            "request_cardinality": "one POINT and one YEAR per request",
            "attributes": protocol.data.nsrdb.attributes,
            "api_key": "redacted; process environment only",
        },
        "passed": passed,
    }
    destination = root / "reports" / "NSRDB_ENDPOINT_AUDIT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
