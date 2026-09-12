"""Credential presence and leak checks that never disclose secret values."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .provenance import utc_now

_SENSITIVE_NAME = re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)")
_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)\s*=\s*[^\s#]{8,}"
)
_REDACTED = {"environment variable omitted", "redacted", "<redacted>", ""}


def credential_hygiene_report(project_root: str | Path) -> dict[str, Any]:
    """Report only booleans, filenames, and counts; never return a credential value."""
    root = Path(project_root)
    active_key = os.environ.get("NREL_API_KEY")
    files = _tracked_files(root) + _operational_files(root)
    unique_files = sorted({path for path in files if path.is_file() and path.name != ".env"})
    leaked_files = [str(path.relative_to(root)) for path in unique_files if _contains_secret(path, active_key)]
    provenance_violations = _provenance_violations(root / "data" / "raw")
    return {
        "report_type": "CREDENTIAL_HYGIENE_REPORT",
        "created_utc": utc_now(),
        "nrel_api_key_present": active_key is not None and bool(active_key.strip()),
        "searched_env_files": False,
        "tracked_file_check": "performed" if _is_git_repository(root) else "not_a_git_repository",
        "files_scanned": len(unique_files),
        "secret_leak_files": leaked_files,
        "provenance_redaction_violations": provenance_violations,
        "request_record_redaction_passed": not provenance_violations,
        "passed": not leaked_files and not provenance_violations,
    }


def _tracked_files(root: Path) -> list[Path]:
    if not _is_git_repository(root):
        return []
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=False
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _is_git_repository(root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=root, check=False, capture_output=True, text=True
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _operational_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, pattern in ((root / "logs", "*"), (root / "reports", "*"), (root / "data" / "raw", "*.source.json")):
        if directory.exists():
            paths.extend(path for path in directory.rglob(pattern) if path.is_file())
    return paths


def _contains_secret(path: Path, active_key: str | None) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    if active_key and active_key in content:
        return True
    return bool(_ASSIGNMENT.search(content))


def _provenance_violations(raw_root: Path) -> list[str]:
    violations: list[str] = []
    for path in raw_root.rglob("*.source.json") if raw_root.exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            violations.append(str(path))
            continue
        if _contains_unredacted_sensitive_value(value):
            violations.append(str(path))
    return violations


def _contains_unredacted_sensitive_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _SENSITIVE_NAME.search(str(key)) and str(nested).strip().lower() not in _REDACTED:
                return True
            if _contains_unredacted_sensitive_value(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_unredacted_sensitive_value(item) for item in value)
    return False
