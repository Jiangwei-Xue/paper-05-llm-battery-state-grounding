"""Fail-closed progression gates for the pre-execution protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def require_formal_freeze_gate(project_root: str | Path) -> None:
    """Permit freeze writing only after real-data readiness and smoke success."""
    root = Path(project_root)
    readiness = _load_report(root / "reports" / "DATA_READINESS_REPORT.json")
    processed = _load_report(root / "reports" / "PROCESSED_DATA_AUDIT.json")
    smoke = _load_report(root / "reports" / "DETERMINISTIC_SMOKE_REPORT.json")
    if (
        readiness.get("contract_id") != "DATA_CONTRACT_V1_2"
        or readiness.get("status") != "pass"
        or readiness.get("claim_bearing_data_ready") is not True
    ):
        raise RuntimeError("Formal freeze blocked: DATA_CONTRACT_V1_2 readiness has not passed.")
    if processed.get("status") != "pass" or processed.get("passed") is not True:
        raise RuntimeError("Formal freeze blocked: processed-data audit has not passed.")
    if smoke.get("status") != "pass" or smoke.get("passed") is not True:
        raise RuntimeError("Formal freeze blocked: deterministic real-data smoke has not passed.")


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
