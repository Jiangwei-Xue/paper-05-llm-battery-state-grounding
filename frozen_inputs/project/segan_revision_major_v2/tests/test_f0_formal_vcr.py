"""Offline integrity tests for the completed F0 formal evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "reviews/f0_formal_v1/artifact_release"


def test_f0_package_has_complete_denominator_and_interlock() -> None:
    report = json.loads((PACKAGE / "F0_FORMAL_H5_E5_INTERLOCK_REPORT.json").read_text())
    verification = json.loads((ROOT / "reviews/f0_formal_v1/F0_FORMAL_VCR_VERIFICATION.json").read_text())
    rows = [json.loads(line) for line in (PACKAGE / "row_evidence.jsonl").read_text().splitlines() if line.strip()]
    assert report["status"] == "PASS"
    assert report["h5"]["status"] == "PASS"
    assert report["e5"]["status"] == "PASS"
    assert verification["status"] == "PASS"
    assert len(rows) == 120
    assert len({row["evidence_id"] for row in rows}) == 120


def test_f0_package_manifest_matches_all_listed_files() -> None:
    manifest = [json.loads(line) for line in (PACKAGE / "F0_FORMAL_HASH_MANIFEST.jsonl").read_text().splitlines() if line.strip()]
    assert len(manifest) > 0
    for row in manifest:
        path = PACKAGE / row["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_f0_package_contains_no_credential_markers() -> None:
    markers = ("NREL_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "Bearer ", ".env")
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path.name != "F0_FORMAL_HASH_MANIFEST.jsonl":
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(marker in text for marker in markers), path
