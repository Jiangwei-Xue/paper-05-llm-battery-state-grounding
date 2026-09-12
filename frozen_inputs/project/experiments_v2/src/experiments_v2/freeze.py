"""Offline manifest and run-plan freeze helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json
from .plan import build_call_plan


def freeze_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    base = Path(root)
    destination = Path(output).resolve()
    paths = sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.resolve() != destination
    )
    entries = [{"path": str(path.relative_to(base)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in paths]
    plan = build_call_plan()
    manifest = {
        "schema_version": "experiments_v2_offline_manifest_v1",
        "status": "offline_dry_run_only",
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "artifact_entries": entries,
        "plan_sha256": sha256_json(plan),
        "manifest_sha256": sha256_json(entries),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
