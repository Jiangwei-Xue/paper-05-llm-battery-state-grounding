#!/usr/bin/env python3
"""Canonical, fail-closed release/archive packaging implementation.

The module uses only the Python standard library.  External archive tools are
detected at runtime and are used for independent integrity/extraction checks.
"""

from __future__ import annotations

import ast
import base64
import csv
import fnmatch
import gzip
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Iterator, Sequence


TOOL_VERSION = "1.2.1"
SCHEMA_VERSION = 1
METADATA_DIR = "RELEASE_METADATA"
MANIFEST_PATH = f"{METADATA_DIR}/RELEASE_MANIFEST.jsonl"
CHECKSUM_PATH = f"{METADATA_DIR}/SHA256SUMS.txt"
PUBLIC_CONFIG_PATH = f"{METADATA_DIR}/RELEASE_CONFIG.public.json"
SOURCE_MAP_PATH = f"{METADATA_DIR}/SOURCE_TO_PUBLIC_HASH_MAP.jsonl"
TRANSFORM_PATH = f"{METADATA_DIR}/SANITIZATION_TRANSFORM_LEDGER.jsonl"
PREFLIGHT_PATH = f"{METADATA_DIR}/PREFLIGHT_AUDIT.json"
BUILD_ATTESTATION_PATH = f"{METADATA_DIR}/BUILD_ATTESTATION.json"
DOCUMENTATION_ATTESTATION_PATH = f"{METADATA_DIR}/DOCUMENTATION_ATTESTATION.json"
TOOL_PROVENANCE_PATH = f"{METADATA_DIR}/TOOL_PROVENANCE.json"

MANIFEST_SELF_EXCLUSIONS = {MANIFEST_PATH, CHECKSUM_PATH}
TEXT_SUFFIXES = {
    ".bash",
    ".cfg",
    ".conf",
    ".csv",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mjs",
    ".py",
    ".r",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

FORBIDDEN_FILE_NAMES = {
    ".DS_Store",
    ".coverage",
}
FORBIDDEN_DIR_NAMES = {
    ".cache",
    ".DocumentRevisions-V100",
    ".fseventsd",
    ".history",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".Spotlight-V100",
    ".TemporaryItems",
    ".tox",
    ".Trashes",
    "__MACOSX",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".swp", ".swo", ".tmp"}
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip", ".7z")
NESTED_ARCHIVE_MAX_DEPTH = 3
NESTED_ARCHIVE_MAX_MEMBERS = 100_000
NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024
SECRET_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "service-account*.json",
)
NON_WAIVABLE_RULE_PREFIXES = (
    "archive.",
    "documentation.",
    "integrity.",
    "manifest.",
    "path.",
    "permission.",
    "secret.",
    "self_hygiene.",
    "symlink.",
)

_HOST_FORCED_XATTR_CACHE: set[str] | None = None


class ReleaseError(RuntimeError):
    """Expected fail-closed build or verification error."""


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "path": self.path, "detail": self.detail}


@dataclass
class Gate:
    name: str
    status: str = "NOT_VERIFIED"
    findings: list[Finding] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str | None = None) -> "Gate":
        if status is not None:
            self.status = status
        elif self.findings:
            self.status = "FAIL"
        else:
            self.status = "PASS"
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "finding_count": len(self.findings),
            "findings": [item.as_dict() for item in self.findings],
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AllowItem:
    source: str
    target: str
    role: str


@dataclass
class ReleaseConfig:
    config_path: Path
    source_root: Path
    output_directory: Path
    release_name: str
    release_class: str
    public_disclosure_profile: str
    private_audit_directory: Path | None
    formats: list[str]
    source_date_epoch: int
    allowlist: list[AllowItem]
    sanitize_machine_paths: bool
    text_suffixes: set[str]
    allow_safe_symlinks: bool
    documentation: dict[str, Any]
    references: dict[str, Any]
    statistical_claims: list[dict[str, Any]]
    exceptions: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def public_view(self) -> dict[str, Any]:
        """Configuration safe to embed in a public archive."""
        result = {
            "schema_version": SCHEMA_VERSION,
            "release_name": self.release_name,
            "release_class": self.release_class,
            "public_disclosure_profile": self.public_disclosure_profile,
            "source_date_epoch": self.source_date_epoch,
            "allow_safe_symlinks": self.allow_safe_symlinks,
            "documentation": self.documentation,
            "references": self.references,
            "statistical_claims": self.statistical_claims,
            "tool_contract": {
                "name": "experiment-release",
                "version": TOOL_VERSION,
                "mandatory_public_checks": [
                    "package_integrity",
                    "documentation",
                    "archive_integrity",
                    "reproducibility",
                ],
            },
        }
        if self.public_disclosure_profile != "minimal":
            result["exceptions"] = self.exceptions
        return result


def utc_from_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def current_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_mode(path: Path, executable_hint: bool = False) -> int:
    if path.is_dir():
        return 0o755
    source_exec = bool(path.stat(follow_symlinks=False).st_mode & 0o111) if path.exists() and not path.is_symlink() else False
    return 0o755 if executable_hint or source_exec else 0o644


def write_bytes(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
    path.chmod(mode)


def write_text(path: Path, text: str, mode: int = 0o644) -> None:
    write_bytes(path, text.encode("utf-8"), mode=mode)


def safe_relative_string(value: str) -> bool:
    if not value or "\x00" in value:
        return False
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return False
    return ".." not in posix.parts and ".." not in windows.parts


def safe_archive_member(value: str, expected_root: str | None = None) -> bool:
    if not safe_relative_string(value):
        return False
    normalized = PurePosixPath(value.replace("\\", "/"))
    if expected_root is not None and normalized.parts and normalized.parts[0] != expected_root:
        return False
    return bool(normalized.parts)


def sanitize_runtime_text(text: str) -> str:
    """Redact machine-local values from logs and errors."""
    replacements: list[tuple[str, str]] = []
    try:
        replacements.append((str(Path.home()), "<HOME>"))
    except RuntimeError:
        pass
    try:
        replacements.append((str(Path.cwd()), "<WORKSPACE>"))
    except OSError:
        pass
    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if old:
            text = text.replace(old, new)
    text = re.sub(r"/Users/[A-Za-z0-9._-]+(?:/[^\s\"']*)?", "<HOME>", text)
    text = re.sub(r"/home/[A-Za-z0-9._-]+(?:/[^\s\"']*)?", "<HOME>", text)
    text = re.sub(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._ -]+(?:[\\/][^\s\"']*)?", "<HOME>", text)
    text = re.sub(r"/var/f[o]lders/[^\s\"']+", "<TEMP_ROOT>", text)
    text = re.sub(r"/private/v[a]r/[^\s\"']+", "<TEMP_ROOT>", text)
    text = re.sub(r"/t[m]p/[^\s\"']+", "<TEMP_ROOT>", text)
    return text


def logical_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return "<OUTSIDE_ROOT>"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read JSON configuration: {sanitize_runtime_text(str(exc))}") from exc


def _resolve_config_path(config_dir: Path, raw: Any, field_name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReleaseError(f"{field_name} must be a non-empty string")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_dir / candidate).resolve()


def _parse_allowlist(value: Any) -> list[AllowItem]:
    if not isinstance(value, list) or not value:
        raise ReleaseError("allowlist must contain at least one explicit selection")
    result: list[AllowItem] = []
    targets: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, str):
            source = target = item
            role = "payload"
        elif isinstance(item, dict):
            source = item.get("source")
            target = item.get("target", source)
            role = item.get("role", "payload")
        else:
            raise ReleaseError(f"allowlist[{index}] must be a string or object")
        if not isinstance(source, str) or not safe_relative_string(source):
            raise ReleaseError(f"allowlist[{index}].source is not a safe relative path")
        if not isinstance(target, str) or not safe_relative_string(target):
            raise ReleaseError(f"allowlist[{index}].target is not a safe relative path")
        target = PurePosixPath(target.replace("\\", "/")).as_posix()
        source = PurePosixPath(source.replace("\\", "/")).as_posix()
        if target in targets:
            raise ReleaseError(f"duplicate allowlist target: {target}")
        if not isinstance(role, str) or not role or any(ord(ch) < 32 for ch in role):
            raise ReleaseError(f"allowlist[{index}].role is invalid")
        targets.add(target)
        result.append(AllowItem(source=source, target=target, role=role))
    return result


def _validate_exceptions(value: Any, release_class: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("exceptions file must be an object with schema_version 1")
    raw_items = value.get("exceptions", [])
    if not isinstance(raw_items, list):
        raise ReleaseError("exceptions must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ReleaseError(f"exceptions[{index}] must be an object")
        required = ("rule", "path", "reason", "public", "approval", "approved_at_utc")
        missing = [name for name in required if name not in item]
        if missing:
            raise ReleaseError(f"exceptions[{index}] missing fields: {', '.join(missing)}")
        rule = item["rule"]
        path = item["path"]
        if not isinstance(rule, str) or not rule:
            raise ReleaseError(f"exceptions[{index}].rule is invalid")
        if rule.startswith(NON_WAIVABLE_RULE_PREFIXES):
            raise ReleaseError(f"exceptions[{index}] attempts to waive non-waivable rule {rule}")
        if not isinstance(path, str) or not safe_relative_string(path) or path in {"*", "**", "**/*"}:
            raise ReleaseError(f"exceptions[{index}].path must be a scoped safe relative glob")
        if not isinstance(item["reason"], str) or len(item["reason"].strip()) < 12:
            raise ReleaseError(f"exceptions[{index}].reason is too short")
        if not isinstance(item["approval"], str) or len(item["approval"].strip()) < 3:
            raise ReleaseError(f"exceptions[{index}].approval is invalid")
        if not isinstance(item["approved_at_utc"], str) or not item["approved_at_utc"].endswith("Z"):
            raise ReleaseError(f"exceptions[{index}].approved_at_utc must be an explicit UTC timestamp")
        if not isinstance(item["public"], bool):
            raise ReleaseError(f"exceptions[{index}].public must be boolean")
        if release_class == "public" and not item["public"]:
            raise ReleaseError(f"exceptions[{index}] is not approved for a public release")
        result.append({name: item[name] for name in required})
    return result


def load_config(config_path: Path) -> ReleaseConfig:
    config_path = config_path.resolve()
    raw = load_json(config_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("release config must be an object with schema_version 1")
    config_dir = config_path.parent
    release_name = raw.get("release_name")
    if not isinstance(release_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_name):
        raise ReleaseError("release_name must be a portable single path component")
    release_class = raw.get("release_class", "public")
    if release_class not in {"public", "internal"}:
        raise ReleaseError("release_class must be public or internal")
    public_disclosure_profile = raw.get("public_disclosure_profile", "standard")
    if public_disclosure_profile not in {"standard", "minimal"}:
        raise ReleaseError("public_disclosure_profile must be standard or minimal")
    source_root = _resolve_config_path(config_dir, raw.get("source_root", "."), "source_root")
    output_directory = _resolve_config_path(config_dir, raw.get("output_directory", "dist"), "output_directory")
    if not source_root.is_dir():
        raise ReleaseError("configured source_root is not a directory")
    formats = raw.get("formats", ["tar.gz"])
    if not isinstance(formats, list) or not formats or any(item not in {"tar", "tar.gz", "tgz", "zip", "7z"} for item in formats):
        raise ReleaseError("formats must be a non-empty list containing tar, tar.gz, tgz, zip, or 7z")
    if len(formats) != len(set(formats)):
        raise ReleaseError("formats contains duplicates")
    source_date_epoch = raw.get("source_date_epoch", 315532800)
    if not isinstance(source_date_epoch, int) or source_date_epoch < 0:
        raise ReleaseError("source_date_epoch must be a non-negative UTC epoch integer")
    sanitization = raw.get("sanitization", {})
    if not isinstance(sanitization, dict):
        raise ReleaseError("sanitization must be an object")
    sanitize_machine_paths = sanitization.get("generic_machine_paths", release_class == "public")
    if not isinstance(sanitize_machine_paths, bool):
        raise ReleaseError("sanitization.generic_machine_paths must be boolean")
    if release_class == "public" and not sanitize_machine_paths:
        raise ReleaseError("public releases must enable generic machine-path sanitization")
    raw_suffixes = sanitization.get("text_suffixes", sorted(TEXT_SUFFIXES))
    if not isinstance(raw_suffixes, list) or any(not isinstance(item, str) or not item.startswith(".") for item in raw_suffixes):
        raise ReleaseError("sanitization.text_suffixes must be a list of dot-prefixed strings")
    documentation = raw.get("documentation", {})
    if not isinstance(documentation, dict) or not isinstance(documentation.get("commands"), list) or not documentation["commands"]:
        raise ReleaseError("documentation.commands must contain at least the canonical verifier command")
    command_ids: set[str] = set()
    for index, command in enumerate(documentation["commands"]):
        if not isinstance(command, dict):
            raise ReleaseError(f"documentation.commands[{index}] must be an object")
        if not all(isinstance(command.get(key), str) and command.get(key) for key in ("id", "file", "command")):
            raise ReleaseError(f"documentation.commands[{index}] requires id, file, and command")
        if command["id"] in command_ids:
            raise ReleaseError(f"duplicate documentation command id: {command['id']}")
        if not safe_relative_string(command["file"]):
            raise ReleaseError(f"documentation command {command['id']} has unsafe file path")
        if "<ARCHIVE_PATH>" not in command["command"]:
            raise ReleaseError(f"documentation command {command['id']} must contain <ARCHIVE_PATH>")
        if any(token in command["command"] for token in ("|", ";", "&&", "||", "`", "$(")):
            raise ReleaseError(f"documentation command {command['id']} contains unsupported shell syntax")
        command_ids.add(command["id"])
    verify_commands = [item for item in documentation["commands"] if item["id"] == "verify"]
    if len(verify_commands) != 1 or "VERIFY_ARCHIVE.py" not in verify_commands[0]["command"] or " verify " not in f" {verify_commands[0]['command']} ":
        raise ReleaseError("documentation.commands must define id=verify using tools/VERIFY_ARCHIVE.py verify")
    references = raw.get("references", {})
    if not isinstance(references, dict):
        raise ReleaseError("references must be an object")
    claims = raw.get("statistical_claims", [])
    if not isinstance(claims, list):
        raise ReleaseError("statistical_claims must be a list")
    exception_path_raw = raw.get("exceptions_file")
    if exception_path_raw:
        exception_path = _resolve_config_path(config_dir, exception_path_raw, "exceptions_file")
        exception_data = load_json(exception_path)
    else:
        exception_data = {"schema_version": SCHEMA_VERSION, "exceptions": []}
    exceptions = _validate_exceptions(exception_data, release_class)
    if public_disclosure_profile == "minimal" and exceptions:
        raise ReleaseError("minimal public disclosure requires an empty exception set")
    private_audit_raw = raw.get("private_audit_directory")
    private_audit_directory = None
    if private_audit_raw is not None:
        private_audit_directory = _resolve_config_path(
            config_dir, private_audit_raw, "private_audit_directory"
        )
    if public_disclosure_profile == "minimal" and private_audit_directory is None:
        raise ReleaseError("minimal public disclosure requires private_audit_directory")
    if private_audit_directory is not None:
        for forbidden_root, label in (
            (source_root, "source_root"),
            (output_directory, "output_directory"),
        ):
            try:
                private_audit_directory.relative_to(forbidden_root)
            except ValueError:
                pass
            else:
                raise ReleaseError(f"private_audit_directory must be outside {label}")
    return ReleaseConfig(
        config_path=config_path,
        source_root=source_root,
        output_directory=output_directory,
        release_name=release_name,
        release_class=release_class,
        public_disclosure_profile=public_disclosure_profile,
        private_audit_directory=private_audit_directory,
        formats=list(formats),
        source_date_epoch=source_date_epoch,
        allowlist=_parse_allowlist(raw.get("allowlist")),
        sanitize_machine_paths=sanitize_machine_paths,
        text_suffixes={item.lower() for item in raw_suffixes},
        allow_safe_symlinks=bool(raw.get("allow_safe_symlinks", False)),
        documentation=documentation,
        references=references,
        statistical_claims=claims,
        exceptions=exceptions,
        raw=raw,
    )


def load_public_config(root: Path) -> dict[str, Any]:
    value = load_json(root / PUBLIC_CONFIG_PATH)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("embedded public release config is invalid")
    return value


def iter_tree(root: Path) -> Iterator[Path]:
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for directory in directories:
            yield current_path / directory
        for filename in files:
            yield current_path / filename


def iter_files(root: Path) -> Iterator[Path]:
    for path in iter_tree(root):
        if path.is_file() and not path.is_symlink():
            yield path


def role_for_path(relative: str, allowlist: Sequence[AllowItem]) -> str:
    best: tuple[int, str] | None = None
    for item in allowlist:
        target = item.target.rstrip("/")
        if relative == target or relative.startswith(target + "/"):
            candidate = (len(target), item.role)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1] if best else "generated-metadata"


def junk_finding(relative: str) -> Finding | None:
    pure = PurePosixPath(relative)
    for part in pure.parts:
        if part in FORBIDDEN_DIR_NAMES:
            return Finding("archive.junk_file", relative, f"forbidden directory component: {part}")
    name = pure.name
    if name in FORBIDDEN_FILE_NAMES or name.startswith("._") or name.endswith("~"):
        return Finding("archive.junk_file", relative, "forbidden Finder/editor artifact")
    if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        return Finding("archive.junk_file", relative, "forbidden temporary-file suffix")
    return None


def secret_filename_finding(relative: str) -> Finding | None:
    name = PurePosixPath(relative).name
    for pattern in SECRET_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return Finding("secret.filename", relative, f"sensitive filename class: {pattern}")
    return None


def source_entries(source: Path) -> Iterator[Path]:
    if source.is_symlink() or source.is_file():
        yield source
        return
    for current, directories, files in os.walk(source, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for directory in list(directories):
            path = current_path / directory
            if path.is_symlink():
                yield path
                directories.remove(directory)
        for filename in files:
            yield current_path / filename


def _privacy_patterns() -> dict[str, re.Pattern[bytes]]:
    return {
        "privacy.macos_home": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
        "privacy.linux_home": re.compile(rb"/home/[A-Za-z0-9._-]+/"),
        "privacy.root_home": re.compile(rb"/r[o]ot/(?:[A-Za-z0-9]|\.[A-Za-z0-9])"),
        "privacy.macos_volume": re.compile(rb"/Volumes/[A-Za-z0-9._ -]+/"),
        "privacy.macos_private_var": re.compile(rb"/private/v[a]r/(?:[A-Za-z0-9]|\.[A-Za-z0-9])"),
        "privacy.macos_var_folders": re.compile(rb"/var/f[o]lders/(?:[A-Za-z0-9]|\.[A-Za-z0-9])"),
        "privacy.posix_tmp": re.compile(rb"/t[m]p/(?:[A-Za-z0-9]|\.[A-Za-z0-9])"),
        "privacy.windows_home_backslash": re.compile(rb"[A-Za-z]:\\Users\\[A-Za-z0-9._ -]+\\"),
        "privacy.windows_home_slash": re.compile(rb"[A-Za-z]:/Users/[A-Za-z0-9._ -]+/"),
        "privacy.windows_drive": re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:\\(?!\\)(?!Users\\<)"),
        "privacy.unc_path": re.compile(rb"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._ -]+\\"),
        "privacy.ci_workspace": re.compile(rb"/(?:__w|github/workspace|builds|workspace)/[A-Za-z0-9._/-]+"),
    }


def _secret_patterns() -> dict[str, re.Pattern[bytes]]:
    return {
        "secret.private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        "secret.bearer": re.compile(rb"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/-]{16,}={0,2}"),
        "secret.api_key": re.compile(rb"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token)[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9._~+/-]{12,}"),
        "secret.password": re.compile(rb"(?i)\b(?:password|passwd|pwd)[ \t]*[:=][ \t]*['\"]?[^\s'\"]{8,}"),
        "secret.authorization_header": re.compile(rb"(?i)\bAuthorization[ \t]*:[ \t]*(?:Basic|Bearer)[ \t]+[^\s]{8,}"),
        "secret.database_url": re.compile(rb"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:/]+:[^\s@/]+@"),
        "secret.aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "secret.github_token": re.compile(rb"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),
        "secret.provider_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "secret.session_cookie": re.compile(rb"(?i)\b(?:session|sessionid|cookie)[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9._~+/-]{16,}"),
        "secret.service_account": re.compile(rb"\"(?:private_key|private_key_id|client_secret)\"[ \t]*:[ \t]*\"[^\"]{8,}\""),
    }


def scan_bytes_patterns(path: Path, patterns: dict[str, re.Pattern[bytes]], relative: str) -> list[Finding]:
    findings: list[Finding] = []
    counts: Counter[str] = Counter()
    overlap = 8192
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            final = not chunk
            data = tail + chunk
            boundary = len(data) if final else max(0, len(data) - overlap)
            for rule, pattern in patterns.items():
                for match in pattern.finditer(data):
                    if final or match.start() < boundary:
                        counts[rule] += 1
            if final:
                break
            tail = data[-overlap:]
    for rule, count in sorted(counts.items()):
        findings.append(Finding(rule, relative, f"{count} match(es)"))
    return findings


def contextual_runtime_identity_pattern(token: str, *, identity_kind: str) -> re.Pattern[bytes]:
    """Match an ephemeral runtime identity only in machine-identity contexts.

    A login name can also be a legitimate author surname, place name, acronym,
    or scientific token.  Treating the bare token as private identity therefore
    corrupts attribution and creates host-dependent verification results.  The
    generic path detectors remain authoritative for home/workspace paths; this
    supplemental detector covers explicit environment and metadata contexts.
    """
    encoded = re.escape(token.encode("utf-8", errors="ignore"))
    right_boundary = rb"(?![A-Za-z0-9_.-])"
    if identity_kind == "user":
        alternatives = (
            rb"\b(?:USER|USERNAME|LOGNAME|LOGIN|OWNER|ACCOUNT)\b[ \t]*[:=][ \t]*['\"]?"
            + encoded
            + right_boundary,
            rb"/(?:Users|home)/" + encoded + rb"(?=[/\\])",
            rb"[A-Za-z]:\\Users\\" + encoded + rb"(?=\\)",
        )
    elif identity_kind == "hostname":
        alternatives = (
            rb"\b(?:HOST|HOSTNAME|MACHINE|NODE|NODENAME)\b[ \t]*[:=][ \t]*['\"]?"
            + encoded
            + right_boundary,
        )
    else:
        raise ValueError(f"unsupported runtime identity kind: {identity_kind}")
    return re.compile(rb"(?:" + rb"|".join(alternatives) + rb")", re.IGNORECASE)


def dynamic_identity_patterns(source_root: Path | None = None) -> dict[str, re.Pattern[bytes]]:
    patterns: dict[str, re.Pattern[bytes]] = {}
    try:
        user = Path.home().name
        if user and len(user) >= 3:
            patterns["privacy.runtime_user"] = contextual_runtime_identity_pattern(
                user,
                identity_kind="user",
            )
    except RuntimeError:
        pass
    try:
        host = socket.gethostname().split(".")[0]
        if host and len(host) >= 3:
            patterns["privacy.runtime_hostname"] = contextual_runtime_identity_pattern(
                host,
                identity_kind="hostname",
            )
    except OSError:
        pass
    if source_root is not None:
        workspace = str(source_root.resolve())
        if workspace:
            patterns["privacy.runtime_workspace"] = re.compile(re.escape(workspace.encode("utf-8")))
    return patterns


def sanitization_patterns(source_root: Path) -> list[tuple[str, re.Pattern[bytes], bytes]]:
    patterns: list[tuple[str, re.Pattern[bytes], bytes]] = [
        ("macos_home", re.compile(rb"/Users/[A-Za-z0-9._-]+"), b"<HOME>"),
        ("linux_home", re.compile(rb"/home/[A-Za-z0-9._-]+"), b"<HOME>"),
        ("root_home", re.compile(rb"/r[o]ot(?=/(?:[A-Za-z0-9]|\.[A-Za-z0-9]))"), b"<HOME>"),
        ("macos_volume", re.compile(rb"/Volumes/[A-Za-z0-9._ -]+"), b"<VOLUME>"),
        ("private_var", re.compile(rb"/private/v[a]r(?=/(?:[A-Za-z0-9]|\.[A-Za-z0-9]))"), b"<PRIVATE_VAR>"),
        ("var_folders", re.compile(rb"/var/f[o]lders(?=/(?:[A-Za-z0-9]|\.[A-Za-z0-9]))"), b"<TEMP_ROOT>"),
        ("posix_tmp", re.compile(rb"/t[m]p(?=/(?:[A-Za-z0-9]|\.[A-Za-z0-9]))"), b"<TEMP_ROOT>"),
        ("windows_home_backslash", re.compile(rb"[A-Za-z]:\\Users\\[A-Za-z0-9._ -]+"), b"<HOME>"),
        ("windows_home_slash", re.compile(rb"[A-Za-z]:/Users/[A-Za-z0-9._ -]+"), b"<HOME>"),
        ("unc_root", re.compile(rb"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._ -]+"), b"<UNC_ROOT>"),
    ]
    root_bytes = str(source_root.resolve()).encode("utf-8")
    if root_bytes:
        patterns.insert(0, ("project_root", re.compile(re.escape(root_bytes)), b"<PROJECT_ROOT>"))
    # Do not rewrite bare runtime user or host tokens.  They may be legitimate
    # scientific or attribution text.  Explicit paths are handled above, while
    # contextual runtime-identity residue is rejected by the privacy gate so it
    # can be corrected deliberately rather than silently altering content.
    return patterns


def sanitize_text_file(source: Path, target: Path, source_root: Path) -> tuple[str, str, int, dict[str, int]]:
    original_digest = hashlib.sha256()
    public_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    patterns = sanitization_patterns(source_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlap = 8192
    pending = b""
    with source.open("rb") as reader, target.open("wb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if chunk:
                original_digest.update(chunk)
                pending += chunk
            final = not chunk
            if not final and len(pending) <= overlap * 2:
                continue
            boundary = len(pending) if final else len(pending) - overlap
            piece = pending[:boundary]
            pending = pending[boundary:]
            for rule, pattern, replacement in patterns:
                piece, count = pattern.subn(replacement, piece)
                if count:
                    counts[rule] += count
            writer.write(piece)
            public_digest.update(piece)
            if final:
                break
    target.chmod(normalized_mode(source))
    return original_digest.hexdigest(), public_digest.hexdigest(), target.stat().st_size, dict(sorted(counts.items()))


def copy_binary_file(source: Path, target: Path) -> tuple[str, str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
            writer.write(chunk)
    target.chmod(normalized_mode(source))
    value = digest.hexdigest()
    return value, value, target.stat().st_size


def ensure_safe_symlink(source: Path, source_root: Path) -> str:
    target = os.readlink(source)
    if os.path.isabs(target) or not safe_relative_string(target):
        raise ReleaseError(f"unsafe source symlink: {logical_path(source, source_root)}")
    resolved = (source.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ReleaseError(f"source symlink escapes source root: {logical_path(source, source_root)}") from exc
    return target.replace("\\", "/")


def copy_allowlisted(config: ReleaseConfig, staging: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_map: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    occupied: set[str] = set()
    for item in config.allowlist:
        source_base = config.source_root / PurePosixPath(item.source)
        if not source_base.exists() and not source_base.is_symlink():
            raise ReleaseError(f"allowlisted source does not exist: {item.source}")
        for source in source_entries(source_base):
            if source_base.is_dir() and not source_base.is_symlink():
                suffix = source.relative_to(source_base).as_posix()
                target_rel = PurePosixPath(item.target, suffix).as_posix()
            else:
                target_rel = item.target
            if target_rel in occupied:
                raise ReleaseError(f"allowlist target collision: {target_rel}")
            occupied.add(target_rel)
            junk = junk_finding(target_rel)
            if junk:
                raise ReleaseError(f"{junk.rule}: {junk.path}")
            secret_name = secret_filename_finding(target_rel)
            if secret_name:
                raise ReleaseError(f"{secret_name.rule}: {secret_name.path}")
            target = staging / PurePosixPath(target_rel)
            if source.is_symlink():
                if not config.allow_safe_symlinks:
                    raise ReleaseError(f"symlink selected while allow_safe_symlinks=false: {item.source}")
                link_target = ensure_safe_symlink(source, config.source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link_target)
                source_map.append({
                    "source_relative_path": logical_path(source, config.source_root),
                    "public_relative_path": target_rel,
                    "status": "included_safe_symlink",
                    "symlink_target": link_target,
                    "logical_role": item.role,
                })
                continue
            if not source.is_file():
                raise ReleaseError(f"unsupported allowlist entry type: {item.source}")
            if config.sanitize_machine_paths and source.suffix.lower() in config.text_suffixes:
                original_hash, public_hash, size, replacement_counts = sanitize_text_file(source, target, config.source_root)
            else:
                original_hash, public_hash, size = copy_binary_file(source, target)
                replacement_counts = {}
            changed = original_hash != public_hash
            source_map.append({
                "source_relative_path": logical_path(source, config.source_root),
                "original_sha256": original_hash,
                "public_relative_path": target_rel,
                "public_sha256": public_hash,
                "size_bytes": size,
                "status": "included_sanitized" if changed else "included_byte_identical",
                "logical_role": item.role,
            })
            if changed:
                transforms.append({
                    "source_relative_path": logical_path(source, config.source_root),
                    "original_sha256": original_hash,
                    "public_relative_path": target_rel,
                    "public_sha256": public_hash,
                    "replacement_counts": replacement_counts,
                    "replacement_count": sum(replacement_counts.values()),
                })
    return source_map, transforms


def list_xattrs(path: Path) -> list[str]:
    if hasattr(os, "listxattr"):
        try:
            return sorted(os.listxattr(path, follow_symlinks=False))
        except OSError:
            return []
    tool = shutil.which("xattr")
    if not tool:
        return []
    try:
        result = subprocess.run(
            [tool, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def host_forced_xattrs() -> set[str]:
    """Detect attributes that the host immediately restores after deletion.

    This is a host-behavior classification, not an archive exception.  Parsed
    archive metadata remains fail-closed for every xattr, including attributes
    returned here.
    """
    global _HOST_FORCED_XATTR_CACHE
    if _HOST_FORCED_XATTR_CACHE is not None:
        return set(_HOST_FORCED_XATTR_CACHE)
    forced: set[str] = set()
    tool = shutil.which("xattr")
    if sys.platform == "darwin" and tool:
        with tempfile.TemporaryDirectory(prefix="experiment-release-xattr-probe-") as raw:
            probe = Path(raw) / "probe"
            try:
                probe.write_bytes(b"")
            except OSError:
                probe = Path(raw)
            for name in list_xattrs(probe):
                try:
                    result = subprocess.run(
                        [tool, "-d", name, str(probe)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if result.returncode == 0 and name in list_xattrs(probe):
                    forced.add(name)
    _HOST_FORCED_XATTR_CACHE = forced
    return set(forced)


def xattr_inventory(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in [root, *iter_tree(root)]:
        counts.update(list_xattrs(path))
    return counts


def strip_xattrs(root: Path) -> dict[str, int]:
    seen = 0
    removed = 0
    failures = 0
    for path in [root, *iter_tree(root)]:
        for name in list_xattrs(path):
            seen += 1
            if hasattr(os, "removexattr"):
                try:
                    os.removexattr(path, name, follow_symlinks=False)
                    removed += 1
                    continue
                except OSError:
                    pass
            tool = shutil.which("xattr")
            if tool:
                try:
                    result = subprocess.run(
                        [tool, "-d", name, str(path)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    result = subprocess.CompletedProcess([], 1, "", "xattr removal failed")
                if result.returncode == 0:
                    removed += 1
                    continue
            failures += 1
    remaining = xattr_inventory(root)
    forced = host_forced_xattrs()
    forced_remaining = sum(count for name, count in remaining.items() if name in forced)
    disallowed_remaining = sum(count for name, count in remaining.items() if name not in forced)
    return {
        "seen": seen,
        "removed": removed,
        "remove_failures": failures,
        "host_forced_remaining": forced_remaining,
        "disallowed_remaining": disallowed_remaining,
    }


def xattr_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    forced = host_forced_xattrs()
    for path in [root, *iter_tree(root)]:
        relative = "." if path == root else logical_path(path, root)
        for name in list_xattrs(path):
            if name in forced:
                continue
            findings.append(Finding(f"xattr.{name}", relative, "extended attribute remains"))
    return findings


def tree_static_findings(root: Path, source_root: Path | None = None) -> dict[str, list[Finding]]:
    result: dict[str, list[Finding]] = {
        "privacy": [],
        "secret": [],
        "archive_hygiene": [],
    }
    privacy_patterns = _privacy_patterns()
    privacy_patterns.update(dynamic_identity_patterns(source_root))
    secret_patterns = _secret_patterns()
    for path in iter_tree(root):
        relative = logical_path(path, root)
        if not safe_relative_string(relative):
            result["archive_hygiene"].append(Finding("path.unsafe_tree_path", relative, "unsafe relative path"))
        junk = junk_finding(relative)
        if junk:
            result["archive_hygiene"].append(junk)
        secret_name = secret_filename_finding(relative)
        if secret_name:
            result["secret"].append(secret_name)
        mode = path.lstat().st_mode
        if mode & (stat.S_ISUID | stat.S_ISGID):
            result["archive_hygiene"].append(Finding("permission.special_bits", relative, "setuid/setgid bit is forbidden"))
        if mode & stat.S_IWOTH:
            result["archive_hygiene"].append(Finding("permission.world_writable", relative, "world-writable entry"))
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not safe_relative_string(target):
                result["archive_hygiene"].append(Finding("symlink.unsafe_target", relative, "absolute or traversal symlink target"))
            else:
                resolved = (path.parent / target).resolve(strict=False)
                try:
                    resolved.relative_to(root.resolve())
                except ValueError:
                    result["archive_hygiene"].append(Finding("symlink.escape", relative, "symlink chain escapes release root"))
            continue
        if path.is_file():
            result["privacy"].extend(scan_bytes_patterns(path, privacy_patterns, relative))
            result["secret"].extend(scan_bytes_patterns(path, secret_patterns, relative))
    result["archive_hygiene"].extend(xattr_findings(root))
    return result


def _decoded_literal_candidates(value: Any) -> Iterator[tuple[str, bytes]]:
    if isinstance(value, bytes):
        yield "bytes_literal", value
        return
    if not isinstance(value, str):
        return
    raw = value.encode("utf-8", errors="ignore")
    yield "string_literal", raw
    compact = value.strip()
    if len(compact) >= 8 and len(compact) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", compact):
        try:
            yield "hex_literal", bytes.fromhex(compact)
        except ValueError:
            pass
    if len(compact) >= 8 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact):
        padded = compact + "=" * ((4 - len(compact) % 4) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                decoded = decoder(padded.encode("ascii"))
            except (ValueError, UnicodeEncodeError):
                continue
            if decoded:
                yield "base64_literal", decoded


def _fold_literal_concat(node: ast.AST) -> str | bytes | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_literal_concat(node.left)
        right = _fold_literal_concat(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, bytes) and isinstance(right, bytes):
            return left + right
    return None


def verifier_self_hygiene(paths: Sequence[Path], root: Path) -> Gate:
    gate = Gate("verifier_self_hygiene")
    privacy_patterns = _privacy_patterns()
    for path in paths:
        relative = logical_path(path, root)
        if not path.is_file():
            gate.findings.append(Finding("self_hygiene.missing_verifier", relative, "verifier component is missing"))
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            gate.findings.append(Finding("self_hygiene.parse_error", relative, sanitize_runtime_text(str(exc))))
            continue
        for node in ast.walk(tree):
            values: list[tuple[str, bytes]] = []
            if isinstance(node, ast.Constant):
                values.extend(_decoded_literal_candidates(node.value))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                folded = _fold_literal_concat(node)
                if folded is not None:
                    values.extend(("concatenated_" + kind, data) for kind, data in _decoded_literal_candidates(folded))
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 3 and all(
                isinstance(child, ast.Constant) and isinstance(child.value, int) and 0 <= child.value <= 255
                for child in node.elts
            ):
                decoded = bytes(child.value for child in node.elts if isinstance(child, ast.Constant))
                printable = sum(32 <= value < 127 for value in decoded)
                if decoded and printable / len(decoded) >= 0.8:
                    gate.findings.append(Finding(
                        "self_hygiene.encoded_identity_bytes_array",
                        relative,
                        f"printable integer-array encoding at line {getattr(node, 'lineno', 0)}",
                    ))
                    values.append(("integer_array", decoded))
            for kind, data in values:
                if kind.endswith(("hex_literal", "base64_literal")):
                    printable = sum(32 <= value < 127 for value in data)
                    if (
                        3 <= len(data) <= 128
                        and printable / len(data) >= 0.9
                        and re.fullmatch(rb"[A-Za-z0-9._@-]+", data)
                    ):
                        gate.findings.append(Finding(
                            "self_hygiene.encoded_identity_literal",
                            relative,
                            f"printable {kind} at line {getattr(node, 'lineno', 0)}",
                        ))
                for rule, pattern in privacy_patterns.items():
                    if pattern.search(data):
                        gate.findings.append(Finding(
                            "self_hygiene.concrete_machine_literal",
                            relative,
                            f"{kind} matches {rule} at line {getattr(node, 'lineno', 0)}",
                        ))
        raw_findings = scan_bytes_patterns(path, privacy_patterns, relative)
        for finding in raw_findings:
            gate.findings.append(Finding("self_hygiene.raw_machine_path", relative, finding.detail + f" ({finding.rule})"))
    gate.evidence = {"files_checked": len(paths), "detector_basis": "generic path classes and AST literal decoding"}
    return gate.finish()


def apply_exceptions(findings: list[Finding], exceptions: Sequence[dict[str, Any]]) -> tuple[list[Finding], list[dict[str, Any]], list[dict[str, Any]]]:
    remaining: list[Finding] = []
    used_indices: set[int] = set()
    waived: list[dict[str, Any]] = []
    for finding in findings:
        matched = False
        for index, exception in enumerate(exceptions):
            if finding.rule == exception["rule"] and fnmatch.fnmatch(finding.path, exception["path"]):
                used_indices.add(index)
                waived.append({
                    "finding": finding.as_dict(),
                    "exception": {
                        "rule": exception["rule"],
                        "path": exception["path"],
                        "reason": exception["reason"],
                        "public": exception["public"],
                        "approval": exception["approval"],
                        "approved_at_utc": exception["approved_at_utc"],
                    },
                })
                matched = True
                break
        if not matched:
            remaining.append(finding)
    unused = [dict(item) for index, item in enumerate(exceptions) if index not in used_indices]
    return remaining, waived, unused


def extract_release_command(markdown: str, command_id: str) -> list[str]:
    pattern = re.compile(
        rf"<!--\s*RELEASE_COMMAND:{re.escape(command_id)}\s*-->\s*"
        rf"```(?:sh|shell|bash|zsh)?\s*\n(?P<body>.*?)\n```\s*"
        rf"<!--\s*END_RELEASE_COMMAND:{re.escape(command_id)}\s*-->",
        re.DOTALL | re.IGNORECASE,
    )
    commands = []
    for match in pattern.finditer(markdown):
        body = match.group("body").strip()
        if body:
            commands.append(body)
    return commands


def documentation_consistency_gate(root: Path, documentation: dict[str, Any], attestation_required: bool = False) -> Gate:
    gate = Gate("documentation")
    checked = 0
    for item in documentation.get("commands", []):
        path = root / PurePosixPath(item["file"])
        relative = item["file"]
        if not path.is_file():
            gate.findings.append(Finding("documentation.missing_file", relative, f"command marker {item['id']} has no document"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            gate.findings.append(Finding("documentation.unreadable", relative, sanitize_runtime_text(str(exc))))
            continue
        commands = extract_release_command(text, item["id"])
        if len(commands) != 1:
            gate.findings.append(Finding("documentation.command_marker_count", relative, f"{item['id']} found {len(commands)} time(s)"))
            continue
        if commands[0] != item["command"].strip():
            gate.findings.append(Finding("documentation.command_drift", relative, f"{item['id']} differs from canonical config"))
        checked += 1
    if attestation_required:
        path = root / DOCUMENTATION_ATTESTATION_PATH
        if not path.is_file():
            gate.findings.append(Finding("documentation.missing_attestation", DOCUMENTATION_ATTESTATION_PATH, "build-time command execution attestation missing"))
        else:
            value = load_json(path)
            expected_ids = sorted(item["id"] for item in documentation.get("commands", []))
            if value.get("status") != "PASS" or sorted(value.get("command_ids", [])) != expected_ids:
                gate.findings.append(Finding("documentation.invalid_attestation", DOCUMENTATION_ATTESTATION_PATH, "attestation does not cover canonical commands"))
    gate.evidence = {"commands_checked": checked, "attestation_required": attestation_required}
    return gate.finish()


def markdown_reference_findings(root: Path, files: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for relative in files:
        path = root / PurePosixPath(relative)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in link_pattern.finditer(text):
            raw = match.group(1).strip().strip("<>").split("#", 1)[0]
            if not raw or raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not safe_relative_string(raw):
                findings.append(Finding("reference.unsafe_markdown_link", relative, raw))
                continue
            target = (path.parent / PurePosixPath(raw)).resolve(strict=False)
            try:
                target.relative_to(root.resolve())
            except ValueError:
                findings.append(Finding("reference.escape", relative, raw))
                continue
            if not target.exists():
                findings.append(Finding("reference.missing_markdown_target", relative, raw))
    return findings


def _json_reference_values(node: Any, parent_key: str = "") -> Iterator[str]:
    keys = {"ref", "reference", "file", "path", "manifest", "readme", "target"}
    if isinstance(node, dict):
        for key, value in node.items():
            if key.lower() in keys and isinstance(value, str):
                yield value
            yield from _json_reference_values(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _json_reference_values(value, parent_key)


def json_reference_findings(root: Path, globs: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_files(root):
        relative = logical_path(path, root)
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in globs):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for raw in _json_reference_values(value):
            if raw.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
                continue
            if not safe_relative_string(raw):
                findings.append(Finding("reference.unsafe_json_path", relative, raw))
                continue
            target = root / PurePosixPath(raw)
            if not target.exists():
                findings.append(Finding("reference.missing_json_target", relative, raw))
    return findings


def internal_reference_gate(root: Path, references: dict[str, Any]) -> Gate:
    gate = Gate("internal_references")
    markdown_files = references.get("markdown_files", [])
    if not isinstance(markdown_files, list) or any(not isinstance(item, str) or not safe_relative_string(item) for item in markdown_files):
        gate.findings.append(Finding("reference.invalid_config", PUBLIC_CONFIG_PATH, "markdown_files must be safe relative paths"))
        return gate.finish()
    gate.findings.extend(markdown_reference_findings(root, markdown_files))
    json_globs = references.get("json_globs", [])
    if not isinstance(json_globs, list) or any(not isinstance(item, str) for item in json_globs):
        gate.findings.append(Finding("reference.invalid_config", PUBLIC_CONFIG_PATH, "json_globs must be strings"))
        return gate.finish()
    gate.findings.extend(json_reference_findings(root, json_globs))
    stale_patterns = references.get("stale_patterns", [])
    if not isinstance(stale_patterns, list) or any(not isinstance(item, str) or not item for item in stale_patterns):
        gate.findings.append(Finding("reference.invalid_config", PUBLIC_CONFIG_PATH, "stale_patterns must be non-empty strings"))
        return gate.finish()
    scan_files = set(markdown_files)
    for path in iter_files(root):
        relative = logical_path(path, root)
        if any(fnmatch.fnmatch(relative, pattern) for pattern in json_globs):
            scan_files.add(relative)
    for relative in sorted(scan_files):
        path = root / PurePosixPath(relative)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in stale_patterns:
            count = text.count(pattern)
            if count:
                gate.findings.append(Finding("reference.stale_pattern", relative, f"{pattern!r}: {count} occurrence(s)"))
    gate.evidence = {
        "markdown_files_checked": len(markdown_files),
        "json_globs": list(json_globs),
        "stale_patterns": len(stale_patterns),
    }
    return gate.finish()


def measure_statistic(root: Path, claim: dict[str, Any]) -> int:
    source = claim.get("source")
    measure = claim.get("measure")
    if not isinstance(source, str) or not safe_relative_string(source):
        raise ReleaseError(f"statistical claim {claim.get('id', '<unknown>')} has unsafe source")
    path = root / PurePosixPath(source)
    if measure == "file_count":
        if not path.is_dir():
            raise ReleaseError(f"statistical claim source is not a directory: {source}")
        pattern = claim.get("glob", "**/*")
        return sum(1 for item in path.glob(pattern) if item.is_file())
    if not path.is_file():
        raise ReleaseError(f"statistical claim source is not a file: {source}")
    if measure in {"nonempty_lines", "jsonl_records"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if measure == "csv_data_rows":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return max(0, sum(1 for _ in reader) - 1)
    if measure == "json_array_length":
        value = load_json(path)
        if not isinstance(value, list):
            raise ReleaseError(f"statistical claim source is not a JSON array: {source}")
        return len(value)
    raise ReleaseError(f"unsupported statistical measure: {measure}")


def statistical_claim_gate(root: Path, claims: Sequence[dict[str, Any]], documentation: dict[str, Any]) -> Gate:
    gate = Gate("statistical_claims")
    covered: dict[str, list[tuple[int, int]]] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or not all(isinstance(claim.get(key), str) and claim.get(key) for key in ("id", "file", "claim_regex", "source", "measure")):
            gate.findings.append(Finding("statistics.invalid_config", PUBLIC_CONFIG_PATH, f"claim {index} is incomplete"))
            continue
        relative = claim["file"]
        path = root / PurePosixPath(relative)
        if not path.is_file():
            gate.findings.append(Finding("statistics.missing_document", relative, claim["id"]))
            continue
        try:
            text = path.read_text(encoding="utf-8")
            pattern = re.compile(claim["claim_regex"], re.IGNORECASE)
        except (OSError, UnicodeDecodeError, re.error) as exc:
            gate.findings.append(Finding("statistics.invalid_claim_regex", relative, sanitize_runtime_text(str(exc))))
            continue
        matches = list(pattern.finditer(text))
        if len(matches) != 1 or len(matches[0].groups()) != 1:
            gate.findings.append(Finding("statistics.claim_match_count", relative, f"{claim['id']} matched {len(matches)} time(s); exactly one capture is required"))
            continue
        raw_value = matches[0].group(1).replace(",", "")
        if not raw_value.isdigit():
            gate.findings.append(Finding("statistics.claim_value", relative, f"{claim['id']} capture is not an integer"))
            continue
        documented = int(raw_value)
        try:
            actual = measure_statistic(root, claim)
        except ReleaseError as exc:
            gate.findings.append(Finding("statistics.measurement_error", claim["source"], str(exc)))
            continue
        expected = claim.get("expected")
        if expected is not None and (not isinstance(expected, int) or expected != actual):
            gate.findings.append(Finding("statistics.expected_mismatch", claim["source"], f"expected {expected}, measured {actual}"))
        if documented != actual:
            gate.findings.append(Finding("statistics.documentation_mismatch", relative, f"{claim['id']}: documented {documented}, measured {actual}"))
        covered.setdefault(relative, []).append(matches[0].span())
    summary_pattern = re.compile(r"\b\d[\d,]*\s+(?:rows?|runs?|trials?|samples?|files?|entries|records?)\b", re.IGNORECASE)
    declared_files = {item.get("file") for item in claims if isinstance(item, dict)}
    command_files = {item.get("file") for item in documentation.get("commands", []) if isinstance(item, dict)}
    for relative in sorted(item for item in declared_files | command_files if isinstance(item, str) and safe_relative_string(item)):
        path = root / PurePosixPath(relative)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        spans = covered.get(relative, [])
        for match in summary_pattern.finditer(text):
            if not any(start <= match.start() and match.end() <= end for start, end in spans):
                gate.findings.append(Finding("statistics.undeclared_summary_claim", relative, match.group(0)))
    gate.evidence = {"declared_claims": len(claims)}
    return gate.finish()


def audit_tree(
    root: Path,
    public_config: dict[str, Any],
    *,
    source_root: Path | None = None,
    attestation_required: bool = False,
) -> dict[str, Gate]:
    """Run all non-manifest tree gates and apply scoped exceptions."""
    gates: dict[str, Gate] = {}
    static = tree_static_findings(root, source_root=source_root)
    if not bool(public_config.get("allow_safe_symlinks", False)):
        for path in iter_tree(root):
            if path.is_symlink():
                static["archive_hygiene"].append(Finding(
                    "symlink.forbidden",
                    logical_path(path, root),
                    "release config disables symlinks",
                ))
    exceptions = public_config.get("exceptions", [])
    privacy_remaining, privacy_waived, _privacy_unused = apply_exceptions(static["privacy"], exceptions)
    privacy_gate = Gate("privacy", findings=privacy_remaining)
    privacy_gate.evidence = {
        "waived_findings": privacy_waived,
        "detector_basis": "generic path classes plus ephemeral runtime identity checks",
    }
    gates["privacy"] = privacy_gate.finish()

    secret_gate = Gate("secret", findings=static["secret"])
    secret_gate.evidence = {"exceptions_allowed": False}
    gates["secret"] = secret_gate.finish()

    hygiene_remaining, hygiene_waived, _hygiene_unused = apply_exceptions(static["archive_hygiene"], exceptions)
    hygiene_gate = Gate("tree_hygiene", findings=hygiene_remaining)
    inventory = xattr_inventory(root)
    forced = host_forced_xattrs()
    hygiene_gate.evidence = {
        "waived_findings": hygiene_waived,
        "xattr_counts": dict(sorted(inventory.items())),
        "host_forced_xattrs": sorted(forced),
        "host_forced_xattr_count": sum(count for name, count in inventory.items() if name in forced),
        "host_forced_scope": "ignored only in filesystem-tree checks after deletion/reappearance probe; never ignored in archive metadata",
    }
    gates["tree_hygiene"] = hygiene_gate.finish()
    gates["nested_archives"] = nested_archive_gate(root)

    verifier_paths = [root / "tools/VERIFY_ARCHIVE.py", root / "tools/release_core.py"]
    gates["verifier_self_hygiene"] = verifier_self_hygiene(verifier_paths, root)
    gates["documentation"] = documentation_consistency_gate(
        root,
        public_config.get("documentation", {}),
        attestation_required=attestation_required,
    )
    gates["internal_references"] = internal_reference_gate(root, public_config.get("references", {}))
    gates["statistical_claims"] = statistical_claim_gate(
        root,
        public_config.get("statistical_claims", []),
        public_config.get("documentation", {}),
    )
    used_exception_records = privacy_waived + hygiene_waived
    used_keys = {
        (record["exception"]["rule"], record["exception"]["path"])
        for record in used_exception_records
    }
    unused = [
        item for item in exceptions
        if (item.get("rule"), item.get("path")) not in used_keys
    ]
    exception_gate = Gate("exceptions")
    for item in unused:
        exception_gate.findings.append(Finding(
            "exception.unused",
            str(item.get("path", "<invalid>")),
            f"exception for {item.get('rule', '<invalid>')} matched no finding",
        ))
    exception_gate.evidence = {
        "configured": len(exceptions),
        "used": len(used_keys),
        "unused": len(unused),
    }
    gates["exceptions"] = exception_gate.finish()
    return gates


def manifest_records(root: Path, allowlist: Sequence[AllowItem] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_tree(root):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = logical_path(path, root)
        if relative in MANIFEST_SELF_EXCLUSIONS:
            continue
        if path.is_symlink():
            records.append({
                "path": relative,
                "entry_type": "symlink",
                "symlink_target": os.readlink(path),
                "mode": "0777",
                "logical_role": role_for_path(relative, allowlist or []),
            })
            continue
        if not path.is_file():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        records.append({
            "path": relative,
            "entry_type": "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "mode": f"{mode:04o}",
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "logical_role": role_for_path(relative, allowlist or []),
        })
    return sorted(records, key=lambda item: item["path"])


def write_manifest(root: Path, allowlist: Sequence[AllowItem]) -> None:
    manifest = root / MANIFEST_PATH
    checksum = root / CHECKSUM_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_text(manifest, jsonl_text(manifest_records(root, allowlist)))
    checksum_lines = []
    for path in iter_files(root):
        relative = logical_path(path, root)
        if relative == CHECKSUM_PATH:
            continue
        checksum_lines.append(f"{sha256_file(path)}  {relative}\n")
    write_text(checksum, "".join(checksum_lines))


def integrity_gate(root: Path) -> Gate:
    gate = Gate("integrity")
    manifest = root / MANIFEST_PATH
    checksum = root / CHECKSUM_PATH
    if not manifest.is_file():
        gate.findings.append(Finding("manifest.missing", MANIFEST_PATH, "manifest file is missing"))
        return gate.finish()
    if not checksum.is_file():
        gate.findings.append(Finding("manifest.missing", CHECKSUM_PATH, "checksum file is missing"))
        return gate.finish()
    expected: dict[str, dict[str, Any]] = {}
    try:
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            relative = record.get("path")
            if not isinstance(relative, str) or not safe_relative_string(relative):
                gate.findings.append(Finding("manifest.unsafe_path", MANIFEST_PATH, f"line {line_number}"))
                continue
            if relative in expected:
                gate.findings.append(Finding("manifest.duplicate_path", MANIFEST_PATH, relative))
                continue
            expected[relative] = record
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        gate.findings.append(Finding("manifest.parse_error", MANIFEST_PATH, sanitize_runtime_text(str(exc))))
        return gate.finish()
    for relative, record in sorted(expected.items()):
        path = root / PurePosixPath(relative)
        entry_type = record.get("entry_type", "file")
        if entry_type == "symlink":
            if not path.is_symlink():
                gate.findings.append(Finding("integrity.missing", relative, "manifest symlink is missing"))
            elif os.readlink(path) != record.get("symlink_target"):
                gate.findings.append(Finding("integrity.symlink_target_mismatch", relative, "symlink target mismatch"))
            continue
        if entry_type != "file":
            gate.findings.append(Finding("manifest.entry_type", relative, str(entry_type)))
            continue
        if not path.is_file() or path.is_symlink():
            gate.findings.append(Finding("integrity.missing", relative, "manifest entry is not a regular file"))
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        actual_mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        if actual_hash != record.get("sha256"):
            gate.findings.append(Finding("integrity.hash_mismatch", relative, "SHA-256 mismatch"))
        if actual_size != record.get("size_bytes"):
            gate.findings.append(Finding("integrity.size_mismatch", relative, "byte-size mismatch"))
        if actual_mode != record.get("mode"):
            gate.findings.append(Finding("integrity.mode_mismatch", relative, f"expected {record.get('mode')}, observed {actual_mode}"))
    observed = {
        logical_path(path, root)
        for path in iter_tree(root)
        if path.is_symlink() or path.is_file()
    }
    expected_all = set(expected) | MANIFEST_SELF_EXCLUSIONS
    for relative in sorted(observed - expected_all):
        gate.findings.append(Finding("integrity.unexpected", relative, "file is absent from manifest contract"))
    for relative in sorted(expected_all - observed):
        gate.findings.append(Finding("integrity.missing", relative, "required manifest/control file is missing"))

    checksum_expected: dict[str, str] = {}
    try:
        for line_number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match or not safe_relative_string(match.group(2)):
                gate.findings.append(Finding("manifest.checksum_format", CHECKSUM_PATH, f"line {line_number}"))
                continue
            checksum_expected[match.group(2)] = match.group(1)
    except (OSError, UnicodeDecodeError) as exc:
        gate.findings.append(Finding("manifest.checksum_parse_error", CHECKSUM_PATH, sanitize_runtime_text(str(exc))))
    checksum_observed = {
        logical_path(path, root)
        for path in iter_files(root)
        if logical_path(path, root) != CHECKSUM_PATH
    }
    if set(checksum_expected) != checksum_observed:
        gate.findings.append(Finding("manifest.checksum_coverage", CHECKSUM_PATH, "checksum paths do not exactly cover release files"))
    for relative, expected_hash in checksum_expected.items():
        path = root / PurePosixPath(relative)
        if path.is_file() and sha256_file(path) != expected_hash:
            gate.findings.append(Finding("integrity.checksum_mismatch", relative, "SHA256SUMS mismatch"))
    gate.evidence = {
        "manifest_records": len(expected),
        "observed_regular_files": len(observed),
        "checksum_records": len(checksum_expected),
    }
    return gate.finish()


def tree_hash_map(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in iter_files(root):
        relative = logical_path(path, root)
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        }
    for path in iter_tree(root):
        if path.is_symlink():
            relative = logical_path(path, root)
            result[relative] = {"symlink_target": os.readlink(path)}
    return result


def _tar_info_for_directory(name: str, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = epoch
    info.pax_headers = {}
    return info


def _tar_info_for_file(name: str, path: Path, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = path.stat().st_size
    info.mode = stat.S_IMODE(path.stat().st_mode)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = epoch
    info.pax_headers = {}
    return info


def create_tar_archive(staging: Path, archive: Path, epoch: int, gzip_enabled: bool) -> None:
    def write_tar(fileobj: Any) -> None:
        with tarfile.open(fileobj=fileobj, mode="w", format=tarfile.PAX_FORMAT, pax_headers={}) as handle:
            handle.addfile(_tar_info_for_directory(staging.name, epoch))
            for path in iter_tree(staging):
                relative = logical_path(path, staging)
                name = f"{staging.name}/{relative}"
                if path.is_symlink():
                    info = tarfile.TarInfo(name=name)
                    info.type = tarfile.SYMTYPE
                    info.linkname = os.readlink(path)
                    info.mode = 0o777
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    info.pax_headers = {}
                    handle.addfile(info)
                elif path.is_dir():
                    handle.addfile(_tar_info_for_directory(name, epoch))
                elif path.is_file():
                    info = _tar_info_for_file(name, path, epoch)
                    with path.open("rb") as source:
                        handle.addfile(info, source)
                else:
                    raise ReleaseError(f"unsupported staging entry type: {relative}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if gzip_enabled:
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=epoch) as compressed:
                write_tar(compressed)
    else:
        with archive.open("wb") as raw:
            write_tar(raw)
    archive.chmod(0o644)


def zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    value = datetime.fromtimestamp(max(epoch, 315532800), timezone.utc)
    second = value.second - value.second % 2
    return (value.year, value.month, value.day, value.hour, value.minute, second)


def create_zip_archive(staging: Path, archive: Path, epoch: int) -> None:
    timestamp = zip_datetime(epoch)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as handle:
        entries = [staging, *iter_tree(staging)]
        for path in entries:
            relative = "." if path == staging else logical_path(path, staging)
            name = staging.name + "/" if relative == "." else f"{staging.name}/{relative}"
            is_directory = path == staging or path.is_dir()
            if is_directory and not name.endswith("/"):
                name += "/"
            info = zipfile.ZipInfo(filename=name, date_time=timestamp)
            info.create_system = 3
            info.extra = b""
            info.comment = b""
            if is_directory:
                mode = stat.S_IFDIR | 0o755
                info.external_attr = (mode << 16) | 0x10
                handle.writestr(info, b"")
            elif path.is_symlink():
                mode = stat.S_IFLNK | 0o777
                info.external_attr = mode << 16
                handle.writestr(info, os.readlink(path).encode("utf-8"))
            elif path.is_file():
                mode = stat.S_IFREG | stat.S_IMODE(path.stat().st_mode)
                info.external_attr = mode << 16
                with path.open("rb") as source, handle.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            else:
                raise ReleaseError(f"unsupported staging entry type: {relative}")
    archive.chmod(0o644)


def find_7z() -> str | None:
    for name in ("7zz", "7z", "7za"):
        value = shutil.which(name)
        if value:
            return value
    return None


def run_command(args: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(list(args), 124, "", sanitize_runtime_text(str(exc)))


def create_7z_archive(staging: Path, archive: Path) -> dict[str, Any]:
    tool = find_7z()
    if not tool:
        raise ReleaseError("7z format requested but no 7zz, 7z, or 7za implementation is available")
    version = run_command([tool, "i"], timeout=30)
    if version.returncode != 0:
        raise ReleaseError("7z implementation detection failed")
    archive.parent.mkdir(parents=True, exist_ok=True)
    args = [
        tool,
        "a",
        "-t7z",
        "-mx=9",
        "-mmt=off",
        "-mtc=off",
        "-mta=off",
        "-mtm=off",
        "-mtr=off",
        "-bd",
        "-y",
        str(archive),
        staging.name,
    ]
    build_env = dict(os.environ)
    build_env["EXPERIMENT_RELEASE_AUTHORIZED"] = "1"
    build_env["COPYFILE_DISABLE"] = "1"
    result = run_command(args, cwd=staging.parent, env=build_env, timeout=3600)
    if result.returncode != 0:
        raise ReleaseError(f"7z creation failed: {sanitize_runtime_text(result.stderr or result.stdout)[:500]}")
    archive.chmod(0o644)
    first_line = next((line.strip() for line in version.stdout.splitlines() if line.strip()), "unknown")
    return {"tool": Path(tool).name, "version_banner": sanitize_runtime_text(first_line)}


def archive_suffix(format_name: str) -> str:
    return {
        "tar": ".tar",
        "tar.gz": ".tar.gz",
        "tgz": ".tgz",
        "zip": ".zip",
        "7z": ".7z",
    }[format_name]


def create_archive(staging: Path, archive: Path, format_name: str, epoch: int) -> dict[str, Any]:
    if format_name == "tar":
        create_tar_archive(staging, archive, epoch, gzip_enabled=False)
        return {"writer": "python.tarfile", "format": "pax-without-xattrs"}
    if format_name in {"tar.gz", "tgz"}:
        create_tar_archive(staging, archive, epoch, gzip_enabled=True)
        return {"writer": "python.tarfile+gzip", "format": "pax-without-xattrs"}
    if format_name == "zip":
        create_zip_archive(staging, archive, epoch)
        return {"writer": "python.zipfile", "format": "zip"}
    if format_name == "7z":
        return {"writer": "external.7z", **create_7z_archive(staging, archive)}
    raise ReleaseError(f"unsupported archive format: {format_name}")


def archive_format_from_path(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".tar.gz"):
        return "tar.gz"
    if lower.endswith(".tgz"):
        return "tgz"
    if lower.endswith(".tar"):
        return "tar"
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".7z"):
        return "7z"
    raise ReleaseError("archive extension is unsupported")


def _member_path_findings(name: str, expected_root: str) -> list[Finding]:
    findings: list[Finding] = []
    normalized = name.replace("\\", "/")
    if not safe_relative_string(normalized):
        findings.append(Finding("archive.unsafe_member_path", normalized, "absolute, traversal, drive, or NUL member"))
    else:
        pure = PurePosixPath(normalized)
        if not pure.parts or pure.parts[0] != expected_root:
            findings.append(Finding("archive.wrong_root", normalized, f"member is outside expected root {expected_root}"))
    junk = junk_finding(normalized)
    if junk:
        findings.append(junk)
    privacy_data = normalized.encode("utf-8", errors="ignore")
    for rule, pattern in _privacy_patterns().items():
        if pattern.search(privacy_data):
            findings.append(Finding(rule, normalized, "privacy class found in archive member name"))
    return findings


def inspect_tar(path: Path, expected_root: str, epoch: int, allow_safe_symlinks: bool = False) -> Gate:
    gate = Gate("archive_inspection")
    member_count = 0
    try:
        with tarfile.open(path, "r:*") as handle:
            for member in handle.getmembers():
                member_count += 1
                gate.findings.extend(_member_path_findings(member.name, expected_root))
                if member.isdev() or member.isfifo():
                    gate.findings.append(Finding("archive.special_member", member.name, "device/FIFO member is forbidden"))
                if member.issym() or member.islnk():
                    if not allow_safe_symlinks:
                        gate.findings.append(Finding("archive.symlink_forbidden", member.name, "release config disables symlinks/hardlinks"))
                    target = member.linkname
                    if not safe_relative_string(target):
                        gate.findings.append(Finding("archive.unsafe_symlink", member.name, target))
                for key in member.pax_headers:
                    if key.startswith(("LIBARCHIVE.xattr.", "SCHILY.xattr.")):
                        gate.findings.append(Finding("archive.xattr_header", member.name, key))
                    value = str(member.pax_headers[key]).encode("utf-8", errors="ignore")
                    for rule, pattern in _privacy_patterns().items():
                        if pattern.search(value):
                            gate.findings.append(Finding(rule, member.name, f"privacy class in pax field {key}"))
                    for rule, pattern in _secret_patterns().items():
                        if pattern.search(value):
                            gate.findings.append(Finding(rule, member.name, f"secret class in pax field {key}"))
                if member.uid != 0 or member.gid != 0 or member.uname not in {"", None} or member.gname not in {"", None}:
                    gate.findings.append(Finding("archive.owner_metadata", member.name, "owner/group metadata is not normalized"))
                if int(member.mtime) != epoch:
                    gate.findings.append(Finding("archive.mtime_metadata", member.name, f"mtime {member.mtime} differs from epoch"))
                mode = member.mode & 0o7777
                expected_mode = 0o755 if member.isdir() else 0o777 if member.issym() else mode
                if not (member.issym() or member.islnk()) and (mode & (stat.S_ISUID | stat.S_ISGID) or mode & stat.S_IWOTH):
                    gate.findings.append(Finding("archive.permission_metadata", member.name, f"unsafe mode {mode:04o}"))
                if member.isdir() and mode != 0o755:
                    gate.findings.append(Finding("archive.permission_metadata", member.name, f"directory mode {mode:04o}"))
    except (OSError, tarfile.TarError) as exc:
        gate.findings.append(Finding("archive.integrity", path.name, sanitize_runtime_text(str(exc))))
    # Payload files may legitimately contain the marker names as detector source
    # code.  Parsed pax-header keys are the authoritative archive-metadata gate.
    gate.evidence = {"member_count": member_count, "xattr_check": "parsed pax-header keys"}
    return gate.finish()


def inspect_zip(path: Path, expected_root: str, epoch: int, allow_safe_symlinks: bool = False) -> Gate:
    gate = Gate("archive_inspection")
    member_count = 0
    expected_time = zip_datetime(epoch)
    try:
        with zipfile.ZipFile(path, "r") as handle:
            for rule, pattern in {**_privacy_patterns(), **_secret_patterns()}.items():
                if pattern.search(handle.comment):
                    gate.findings.append(Finding(rule, path.name, "match in ZIP archive comment"))
            bad = handle.testzip()
            if bad:
                gate.findings.append(Finding("archive.integrity", bad, "ZIP CRC failure"))
            for info in handle.infolist():
                member_count += 1
                gate.findings.extend(_member_path_findings(info.filename, expected_root))
                if info.flag_bits & 0x1:
                    gate.findings.append(Finding("archive.encrypted_member", info.filename, "encrypted ZIP member"))
                if info.extra:
                    gate.findings.append(Finding("archive.zip_extra_field", info.filename, f"{len(info.extra)} extra-field bytes"))
                if info.create_system != 3:
                    gate.findings.append(Finding("archive.zip_host_metadata", info.filename, f"create_system={info.create_system}"))
                if info.date_time != expected_time:
                    gate.findings.append(Finding("archive.mtime_metadata", info.filename, f"timestamp {info.date_time}"))
                mode = (info.external_attr >> 16) & 0o177777
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    if not allow_safe_symlinks:
                        gate.findings.append(Finding("archive.symlink_forbidden", info.filename, "release config disables symlinks"))
                    target = handle.read(info).decode("utf-8", errors="replace")
                    if not safe_relative_string(target):
                        gate.findings.append(Finding("archive.unsafe_symlink", info.filename, target))
                permissions = mode & 0o7777
                if file_type != stat.S_IFLNK and (permissions & (stat.S_ISUID | stat.S_ISGID) or permissions & stat.S_IWOTH):
                    gate.findings.append(Finding("archive.permission_metadata", info.filename, f"unsafe mode {permissions:04o}"))
                for rule, pattern in {**_privacy_patterns(), **_secret_patterns()}.items():
                    if pattern.search(info.comment):
                        gate.findings.append(Finding(rule, info.filename, "match in ZIP member comment"))
    except (OSError, zipfile.BadZipFile) as exc:
        gate.findings.append(Finding("archive.integrity", path.name, sanitize_runtime_text(str(exc))))
    gate.evidence = {"member_count": member_count}
    return gate.finish()


def parse_7z_slt(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_entries = False
    for line in output.splitlines():
        if line.startswith("----------"):
            in_entries = True
            current = {}
            continue
        if not in_entries:
            continue
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if current:
        records.append(current)
    return records


def inspect_7z(path: Path, expected_root: str, allow_safe_symlinks: bool = False) -> Gate:
    gate = Gate("archive_inspection")
    tool = find_7z()
    if not tool:
        gate.findings.append(Finding("archive.tool_missing", path.name, "7z implementation unavailable"))
        return gate.finish()
    listing = run_command([tool, "l", "-slt", "-bd", str(path)], timeout=300)
    if listing.returncode != 0:
        gate.findings.append(Finding("archive.integrity", path.name, "7z listing failed"))
        return gate.finish()
    records = parse_7z_slt(listing.stdout)
    for record in records:
        name = record.get("Path", "")
        gate.findings.extend(_member_path_findings(name, expected_root))
        if record.get("Symbolic Link"):
            if not allow_safe_symlinks:
                gate.findings.append(Finding("archive.symlink_forbidden", name, "release config disables symlinks"))
            target = record["Symbolic Link"]
            if not safe_relative_string(target):
                gate.findings.append(Finding("archive.unsafe_symlink", name, target))
        for field_name in ("Created", "Accessed", "Modified"):
            if record.get(field_name):
                gate.findings.append(Finding("archive.mtime_metadata", name, f"7z field {field_name} is present"))
        attributes = record.get("Attributes", "")
        if "D" not in attributes and any(token in attributes.lower() for token in ("s", "u", "g")):
            gate.findings.append(Finding("archive.permission_metadata", name, f"attributes={attributes}"))
    marker_text = listing.stdout + listing.stderr
    for marker in ("LIBARCHIVE.xattr.", "SCHILY.xattr.", "com.apple.provenance", "com.apple.quarantine"):
        if marker in marker_text:
            gate.findings.append(Finding("archive.xattr_header", path.name, marker))
    gate.evidence = {
        "member_count": len(records),
        "tool": Path(tool).name,
        "listing_returncode": listing.returncode,
    }
    return gate.finish()


def inspect_archive(path: Path, expected_root: str, epoch: int, allow_safe_symlinks: bool = False) -> Gate:
    format_name = archive_format_from_path(path)
    if format_name in {"tar", "tar.gz", "tgz"}:
        return inspect_tar(path, expected_root, epoch, allow_safe_symlinks=allow_safe_symlinks)
    if format_name == "zip":
        return inspect_zip(path, expected_root, epoch, allow_safe_symlinks=allow_safe_symlinks)
    if format_name == "7z":
        return inspect_7z(path, expected_root, allow_safe_symlinks=allow_safe_symlinks)
    raise ReleaseError(f"unsupported archive format: {format_name}")


def _nested_archive_inventory(path: Path) -> dict[str, Any]:
    """Read bounded member metadata without extracting a nested archive."""
    format_name = archive_format_from_path(path)
    roots: set[str] = set()
    member_count = 0
    uncompressed_bytes = 0
    epoch: int | None = None
    if format_name in {"tar", "tar.gz", "tgz"}:
        with tarfile.open(path, "r:*") as handle:
            for member in handle:
                member_count += 1
                if safe_relative_string(member.name):
                    parts = PurePosixPath(member.name.replace("\\", "/")).parts
                    if parts:
                        roots.add(parts[0])
                if member.isfile():
                    uncompressed_bytes += max(0, member.size)
                if epoch is None:
                    epoch = int(member.mtime)
                if member_count > NESTED_ARCHIVE_MAX_MEMBERS or uncompressed_bytes > NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                    break
    elif format_name == "zip":
        with zipfile.ZipFile(path, "r") as handle:
            for info in handle.infolist():
                member_count += 1
                if safe_relative_string(info.filename):
                    parts = PurePosixPath(info.filename.replace("\\", "/")).parts
                    if parts:
                        roots.add(parts[0])
                uncompressed_bytes += max(0, info.file_size)
                if epoch is None:
                    epoch = int(datetime(*info.date_time, tzinfo=timezone.utc).timestamp())
                if member_count > NESTED_ARCHIVE_MAX_MEMBERS or uncompressed_bytes > NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                    break
    else:
        tool = find_7z()
        if not tool:
            raise ReleaseError("7z implementation unavailable for nested archive audit")
        listing = run_command([tool, "l", "-slt", "-bd", str(path)], timeout=300)
        if listing.returncode != 0:
            raise ReleaseError("7z listing failed during nested archive audit")
        for record in parse_7z_slt(listing.stdout):
            name = record.get("Path", "")
            member_count += 1
            if safe_relative_string(name):
                parts = PurePosixPath(name.replace("\\", "/")).parts
                if parts:
                    roots.add(parts[0])
            try:
                uncompressed_bytes += max(0, int(record.get("Size", "0") or 0))
            except ValueError:
                raise ReleaseError("7z member size is not an integer")
            if member_count > NESTED_ARCHIVE_MAX_MEMBERS or uncompressed_bytes > NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                break
    return {
        "format": format_name,
        "roots": sorted(roots),
        "member_count": member_count,
        "uncompressed_bytes": uncompressed_bytes,
        "epoch": epoch if epoch is not None else 315532800,
    }


def _nested_findings(findings: Sequence[Finding], archive_relative: str) -> list[Finding]:
    return [
        Finding(item.rule, f"{archive_relative}!/{item.path}", item.detail)
        for item in findings
    ]


def nested_archive_gate(root: Path, *, max_depth: int = NESTED_ARCHIVE_MAX_DEPTH) -> Gate:
    """Fail closed on dirty, unsafe, private, or uninspectable nested archives."""
    gate = Gate("nested_archives")
    audited = 0

    def audit_container(container_root: Path, label_prefix: str, depth: int) -> None:
        nonlocal audited
        for nested in iter_files(container_root):
            if not nested.name.lower().endswith(ARCHIVE_SUFFIXES):
                continue
            relative = logical_path(nested, container_root)
            label = f"{label_prefix}!/{relative}" if label_prefix else relative
            if depth > max_depth:
                gate.findings.append(Finding(
                    "archive.nested_depth_limit",
                    label,
                    f"nested archive depth exceeds {max_depth}",
                ))
                continue
            audited += 1
            try:
                inventory = _nested_archive_inventory(nested)
            except (OSError, tarfile.TarError, zipfile.BadZipFile, ReleaseError) as exc:
                gate.findings.append(Finding("archive.nested_integrity", label, sanitize_runtime_text(str(exc))))
                continue
            if inventory["member_count"] > NESTED_ARCHIVE_MAX_MEMBERS:
                gate.findings.append(Finding(
                    "archive.nested_member_limit",
                    label,
                    f"more than {NESTED_ARCHIVE_MAX_MEMBERS} members",
                ))
                continue
            if inventory["uncompressed_bytes"] > NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                gate.findings.append(Finding(
                    "archive.nested_size_limit",
                    label,
                    f"uncompressed bytes exceed {NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES}",
                ))
                continue
            roots = inventory["roots"]
            if len(roots) != 1:
                gate.findings.append(Finding(
                    "archive.nested_root_layout",
                    label,
                    f"expected one top-level root, observed {len(roots)}",
                ))
                continue
            archive_gate = inspect_archive(nested, roots[0], int(inventory["epoch"]))
            gate.findings.extend(_nested_findings(archive_gate.findings, label))
            if archive_gate.status != "PASS":
                continue
            with tempfile.TemporaryDirectory(prefix="experiment-release-nested-") as raw_temp:
                destination = Path(raw_temp) / "extract"
                destination.mkdir()
                extraction = system_integrity_and_extract(nested, destination)
                if extraction.get("status") != "PASS":
                    gate.findings.append(Finding(
                        "archive.nested_system_extraction",
                        label,
                        str(extraction.get("reason", "integrity/extraction tool failed or warned")),
                    ))
                    continue
                extracted_roots = sorted(destination.iterdir())
                if len(extracted_roots) != 1 or not extracted_roots[0].is_dir():
                    gate.findings.append(Finding(
                        "archive.nested_root_layout",
                        label,
                        f"extraction produced {len(extracted_roots)} top-level entries",
                    ))
                    continue
                static = tree_static_findings(extracted_roots[0])
                for category in ("privacy", "secret", "archive_hygiene"):
                    gate.findings.extend(_nested_findings(static[category], label))
                audit_container(extracted_roots[0], label, depth + 1)

    audit_container(root, "", 1)
    gate.evidence = {
        "archives_audited": audited,
        "maximum_depth": max_depth,
        "maximum_members_per_archive": NESTED_ARCHIVE_MAX_MEMBERS,
        "maximum_uncompressed_bytes_per_archive": NESTED_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
        "scope": "member metadata, clean extraction, privacy, secrets, junk, xattrs, and recursive nested archives",
    }
    return gate.finish()


def minimal_subprocess_env(temp_root: Path | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "COMSPEC"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COPYFILE_DISABLE"] = "1"
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    if temp_root is not None:
        env["TMPDIR"] = str(temp_root)
        env["TEMP"] = str(temp_root)
        env["TMP"] = str(temp_root)
    return env


def system_integrity_and_extract(archive: Path, destination: Path) -> dict[str, Any]:
    format_name = archive_format_from_path(archive)
    env = minimal_subprocess_env(destination.parent)
    if format_name in {"tar", "tar.gz", "tgz"}:
        tool = shutil.which("tar")
        if not tool:
            return {"status": "NOT_VERIFIED", "reason": "tar implementation unavailable"}
        listing = run_command([tool, "-tf", str(archive)], env=env, timeout=300)
        extraction = run_command([tool, "-xf", str(archive), "-C", str(destination)], env=env, timeout=1800)
        version = run_command([tool, "--version"], env=env, timeout=30)
    elif format_name == "zip":
        tool = shutil.which("unzip")
        if not tool:
            return {"status": "NOT_VERIFIED", "reason": "unzip implementation unavailable"}
        listing = run_command([tool, "-tqq", str(archive)], env=env, timeout=300)
        extraction = run_command([tool, "-qq", str(archive), "-d", str(destination)], env=env, timeout=1800)
        version = run_command([tool, "-v"], env=env, timeout=30)
    else:
        tool = find_7z()
        if not tool:
            return {"status": "NOT_VERIFIED", "reason": "7z implementation unavailable"}
        listing = run_command([tool, "t", "-bd", "-y", str(archive)], env=env, timeout=1800)
        extraction = run_command([tool, "x", "-bd", "-y", f"-o{destination}", str(archive)], env=env, timeout=1800)
        version = run_command([tool, "i"], env=env, timeout=30)
    warning_lines = [
        sanitize_runtime_text(line.strip())
        for line in (listing.stderr + "\n" + extraction.stderr).splitlines()
        if line.strip()
    ]
    status = "PASS" if listing.returncode == 0 and extraction.returncode == 0 and not warning_lines else "FAIL"
    banner = next((sanitize_runtime_text(line.strip()) for line in version.stdout.splitlines() if line.strip()), Path(tool).name)
    return {
        "status": status,
        "tool": Path(tool).name,
        "version_banner": banner[:200],
        "integrity_returncode": listing.returncode,
        "extraction_returncode": extraction.returncode,
        "warning_count": len(warning_lines),
        "warnings": warning_lines[:20],
    }


def execute_documentation_commands(root: Path, archive: Path, documentation: dict[str, Any]) -> Gate:
    gate = Gate("documentation_execution")
    records: list[dict[str, Any]] = []
    env = minimal_subprocess_env(root.parent)
    for item in documentation.get("commands", []):
        command_text = item["command"]
        try:
            args = shlex.split(command_text, posix=os.name != "nt")
        except ValueError as exc:
            gate.findings.append(Finding("documentation.command_parse", item["file"], sanitize_runtime_text(str(exc))))
            continue
        substituted = []
        for arg in args:
            arg = arg.replace("<ARCHIVE_PATH>", str(archive))
            arg = arg.replace("<RELEASE_ROOT>", str(root))
            arg = arg.replace("<PYTHON>", sys.executable)
            substituted.append(arg)
        cwd_raw = item.get("cwd", ".")
        if not isinstance(cwd_raw, str) or not safe_relative_string(cwd_raw):
            gate.findings.append(Finding("documentation.unsafe_cwd", item["file"], str(cwd_raw)))
            continue
        cwd = root / PurePosixPath(cwd_raw)
        if not cwd.is_dir():
            gate.findings.append(Finding("documentation.missing_cwd", item["file"], cwd_raw))
            continue
        result = run_command(substituted, cwd=cwd, env=env, timeout=int(item.get("timeout_seconds", 300)))
        stdout = sanitize_runtime_text(result.stdout)[-4000:]
        stderr = sanitize_runtime_text(result.stderr)[-4000:]
        records.append({
            "id": item["id"],
            "returncode": result.returncode,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
        })
        if result.returncode != 0:
            gate.findings.append(Finding("documentation.command_failed", item["file"], f"{item['id']} returncode={result.returncode}"))
    gate.evidence = {"commands": records}
    return gate.finish()


def summarized_gates(gates: dict[str, Gate]) -> dict[str, str]:
    integrity_components = ["integrity"]
    privacy_components = ["privacy", "secret", "verifier_self_hygiene"]
    documentation_components = ["documentation", "internal_references", "statistical_claims"]
    hygiene_components = ["tree_hygiene", "nested_archives", "exceptions", "archive_inspection", "system_extraction", "roundtrip"]
    reproduction_components = ["reproducibility"]

    def aggregate(names: Sequence[str]) -> str:
        statuses = [gates[name].status for name in names if name in gates]
        if not statuses or any(status == "NOT_VERIFIED" for status in statuses):
            return "NOT_VERIFIED"
        if any(status != "PASS" for status in statuses):
            return "FAIL"
        return "PASS"

    result = {
        "INTEGRITY_GATE": aggregate(integrity_components),
        "PRIVACY_GATE": aggregate(privacy_components),
        "DOCUMENTATION_GATE": aggregate(documentation_components),
        "ARCHIVE_HYGIENE_GATE": aggregate(hygiene_components),
        "REPRODUCIBILITY_GATE": aggregate(reproduction_components),
    }
    result["OVERALL_RELEASE_STATUS"] = "PASS" if all(value == "PASS" for value in result.values()) else "FAIL"
    return result


def verify_archive(archive: Path, *, expected_staging: Path | None = None) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise ReleaseError("archive does not exist")
    with tempfile.TemporaryDirectory(prefix="experiment-release-verify-") as temp_raw:
        temp = Path(temp_raw)
        probe = temp / "probe"
        probe.mkdir()
        # The member inspection must pass before any extractor sees the archive.
        provisional_root = archive.name
        format_name = archive_format_from_path(archive)
        if format_name == "tar.gz":
            provisional_root = archive.name[: -len(".tar.gz")]
        elif format_name == "tgz":
            provisional_root = archive.name[: -len(".tgz")]
        else:
            provisional_root = archive.stem
        # Read the actual first root without extracting.  The expected root is the
        # portable archive basename contract used by the builder.
        archive_gate = inspect_archive(archive, provisional_root, 315532800)
        # If the default epoch/root guess differs, inspect again after reading the
        # embedded config.  Unsafe paths still stop extraction here.
        unsafe_now = [item for item in archive_gate.findings if item.rule in {"archive.unsafe_member_path", "archive.unsafe_symlink"}]
        if unsafe_now:
            gates = {"archive_inspection": archive_gate}
            summary = summarized_gates(gates)
            return {
                "tool_version": TOOL_VERSION,
                "archive": archive.name,
                "gates": {name: gate.as_dict() for name, gate in gates.items()},
                **summary,
            }
        extraction_info = system_integrity_and_extract(archive, probe)
        roots = sorted(item for item in probe.iterdir()) if probe.exists() else []
        if len(roots) != 1 or not roots[0].is_dir():
            root_gate = Gate("system_extraction")
            root_gate.findings.append(Finding("archive.root_layout", archive.name, f"observed {len(roots)} top-level entries"))
            root_gate.evidence = extraction_info
            root_gate.finish()
            gates = {"archive_inspection": archive_gate, "system_extraction": root_gate}
            summary = summarized_gates(gates)
            return {
                "tool_version": TOOL_VERSION,
                "archive": archive.name,
                "gates": {name: gate.as_dict() for name, gate in gates.items()},
                **summary,
            }
        root = roots[0]
        public_config = load_public_config(root)
        release_name = public_config.get("release_name")
        epoch = public_config.get("source_date_epoch")
        if not isinstance(release_name, str) or not isinstance(epoch, int):
            raise ReleaseError("embedded config lacks release_name/source_date_epoch")
        archive_gate = inspect_archive(
            archive,
            release_name,
            epoch,
            allow_safe_symlinks=bool(public_config.get("allow_safe_symlinks", False)),
        )
        extraction_gate = Gate("system_extraction")
        extraction_gate.evidence = extraction_info
        if extraction_info.get("status") != "PASS":
            extraction_gate.findings.append(Finding("archive.system_extraction", archive.name, str(extraction_info.get("reason", "integrity/extraction tool failed or warned"))))
        extraction_gate.finish()
        gates = audit_tree(root, public_config, attestation_required=True)
        gates["archive_inspection"] = archive_gate
        gates["system_extraction"] = extraction_gate
        gates["integrity"] = integrity_gate(root)
        roundtrip = Gate("roundtrip")
        if expected_staging is not None:
            expected = tree_hash_map(expected_staging)
            observed = tree_hash_map(root)
            if expected != observed:
                missing = sorted(set(expected) - set(observed))
                extra = sorted(set(observed) - set(expected))
                mismatched = sorted(key for key in set(expected) & set(observed) if expected[key] != observed[key])
                roundtrip.findings.append(Finding(
                    "integrity.roundtrip_mismatch",
                    release_name,
                    f"missing={len(missing)} extra={len(extra)} mismatched={len(mismatched)}",
                ))
                roundtrip.evidence = {"missing": missing[:50], "extra": extra[:50], "mismatched": mismatched[:50]}
            else:
                roundtrip.evidence = {"entries_compared": len(expected)}
        else:
            roundtrip.evidence = {"scope": "manifest-closed extracted-tree comparison"}
        roundtrip.finish()
        gates["roundtrip"] = roundtrip
        reproduction = Gate("reproducibility")
        attestation_path = root / BUILD_ATTESTATION_PATH
        if not attestation_path.is_file():
            reproduction.findings.append(Finding("reproducibility.missing_attestation", BUILD_ATTESTATION_PATH, "duplicate-build attestation missing"))
        else:
            value = load_json(attestation_path)
            if value.get("status") != "PASS" or value.get("method") != "byte-for-byte duplicate archive build":
                reproduction.findings.append(Finding("reproducibility.invalid_attestation", BUILD_ATTESTATION_PATH, "duplicate-build contract not satisfied"))
        reproduction.evidence = {"scope": "normalized metadata plus embedded duplicate-build attestation"}
        gates["reproducibility"] = reproduction.finish()
        summary = summarized_gates(gates)
        return {
            "tool_version": TOOL_VERSION,
            "archive": archive.name,
            "archive_format": format_name,
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "release_root": release_name,
            "gates": {name: gate.as_dict() for name, gate in sorted(gates.items())},
            **summary,
        }


def preflight_report(config: ReleaseConfig, staging: Path) -> tuple[dict[str, Gate], dict[str, str]]:
    public_config = config.public_view
    gates = audit_tree(staging, public_config, source_root=config.source_root, attestation_required=False)
    gates["integrity"] = Gate("integrity", status="NOT_VERIFIED", evidence={"reason": "manifest is written after preflight"})
    gates["archive_inspection"] = Gate("archive_inspection", status="NOT_VERIFIED")
    gates["system_extraction"] = Gate("system_extraction", status="NOT_VERIFIED")
    gates["roundtrip"] = Gate("roundtrip", status="NOT_VERIFIED")
    gates["reproducibility"] = Gate("reproducibility", status="NOT_VERIFIED")
    return gates, summarized_gates(gates)


def all_pass(gates: dict[str, Gate], names: Sequence[str]) -> bool:
    return all(name in gates and gates[name].status == "PASS" for name in names)


def build_release(config_path: Path, *, audit_only: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    final_paths = {
        format_name: config.output_directory / f"{config.release_name}{archive_suffix(format_name)}"
        for format_name in config.formats
    }
    existing = [path.name for path in final_paths.values() if path.exists()]
    if existing and not audit_only:
        raise ReleaseError(f"refusing to overwrite existing artifact(s): {', '.join(sorted(existing))}")
    with tempfile.TemporaryDirectory(prefix=".experiment-release-build-", dir=config.output_directory) as temp_raw:
        temp = Path(temp_raw)
        staging = temp / config.release_name
        staging.mkdir(mode=0o755)
        source_map, transforms = copy_allowlisted(config, staging)
        metadata = staging / METADATA_DIR
        metadata.mkdir(mode=0o755)

        canonical_dir = Path(__file__).resolve().parent
        core_source = canonical_dir / "release_core.py"
        cli_source = canonical_dir / "release_tool.py"
        if not core_source.is_file() or not cli_source.is_file():
            raise ReleaseError("canonical verifier components are missing")
        copy_binary_file(core_source, staging / "tools/release_core.py")
        copy_binary_file(cli_source, staging / "tools/VERIFY_ARCHIVE.py")
        # The documented interface invokes the verifier through Python, so an
        # executable bit is unnecessary and would be lost by some 7z tools.
        (staging / "tools/VERIFY_ARCHIVE.py").chmod(0o644)

        write_text(metadata / "RELEASE_CONFIG.public.json", json_text(config.public_view))
        if config.public_disclosure_profile == "minimal":
            assert config.private_audit_directory is not None
            private_metadata = config.private_audit_directory / config.release_name
            private_metadata.mkdir(parents=True, exist_ok=True)
            write_text(private_metadata / "SOURCE_TO_PUBLIC_HASH_MAP.jsonl", jsonl_text(source_map))
            write_text(private_metadata / "SANITIZATION_TRANSFORM_LEDGER.jsonl", jsonl_text(transforms))
        else:
            write_text(metadata / "SOURCE_TO_PUBLIC_HASH_MAP.jsonl", jsonl_text(source_map))
            write_text(metadata / "SANITIZATION_TRANSFORM_LEDGER.jsonl", jsonl_text(transforms))
        write_text(metadata / "TOOL_PROVENANCE.json", json_text({
            "tool": "experiment-release",
            "version": TOOL_VERSION,
            "release_core_sha256": sha256_file(staging / "tools/release_core.py"),
            "verifier_cli_sha256": sha256_file(staging / "tools/VERIFY_ARCHIVE.py"),
            "source_path_recorded": False,
        }))
        write_text(metadata / "BUILD_ATTESTATION.json", json_text({
            "status": "PASS",
            "method": "byte-for-byte duplicate archive build",
            "source_date_epoch": config.source_date_epoch,
            "created_utc": utc_from_epoch(config.source_date_epoch),
            "publication_rule": "Builder aborts before atomic publication if duplicate bytes differ.",
        }))
        write_text(metadata / "DOCUMENTATION_ATTESTATION.json", json_text({
            "status": "PASS",
            "command_ids": sorted(item["id"] for item in config.documentation["commands"]),
            "created_utc": utc_from_epoch(config.source_date_epoch),
            "publication_rule": "Builder aborts before atomic publication if any canonical command returns non-zero.",
        }))
        strip_result = strip_xattrs(staging)
        pre_gates, _pre_summary = preflight_report(config, staging)
        preflight_payload = {
            "tool_version": TOOL_VERSION,
            "stage": "preflight",
            "preflight_status": "PASS" if all_pass(pre_gates, [
                "privacy",
                "secret",
                "tree_hygiene",
                "verifier_self_hygiene",
                "documentation",
                "internal_references",
                "statistical_claims",
                "exceptions",
            ]) else "FAIL",
            "release_name": config.release_name,
            "release_class": config.release_class,
            "created_utc": utc_from_epoch(config.source_date_epoch),
            "source_selection_records": len(source_map),
            "sanitized_files": len(transforms),
            "xattr_cleanup": strip_result,
            "gates": {name: gate.as_dict() for name, gate in sorted(pre_gates.items())},
            "not_yet_performed": [
                "manifest closure",
                "archive inspection",
                "independent integrity test",
                "clean extraction",
                "documentation command execution",
                "duplicate-build comparison",
            ],
        }
        if config.public_disclosure_profile == "minimal":
            assert config.private_audit_directory is not None
            private_metadata = config.private_audit_directory / config.release_name
            private_metadata.mkdir(parents=True, exist_ok=True)
            write_text(private_metadata / "PREFLIGHT_AUDIT.json", json_text(preflight_payload))
        else:
            write_text(metadata / "PREFLIGHT_AUDIT.json", json_text(preflight_payload))
        strip_result_2 = strip_xattrs(staging)
        final_pre_gates = audit_tree(staging, config.public_view, source_root=config.source_root, attestation_required=True)
        mandatory_pre = [
            "privacy",
            "secret",
            "tree_hygiene",
            "verifier_self_hygiene",
            "documentation",
            "internal_references",
            "statistical_claims",
            "exceptions",
        ]
        if not all_pass(final_pre_gates, mandatory_pre):
            summary = summarized_gates(final_pre_gates)
            return {
                "tool_version": TOOL_VERSION,
                "mode": "audit-only" if audit_only else "build",
                "release_name": config.release_name,
                "source_selection_records": len(source_map),
                "sanitized_files": len(transforms),
                "xattr_cleanup": [strip_result, strip_result_2],
                "gates": {name: gate.as_dict() for name, gate in sorted(final_pre_gates.items())},
                **summary,
            }
        if audit_only:
            gates = dict(final_pre_gates)
            for name in ("integrity", "archive_inspection", "system_extraction", "roundtrip", "reproducibility"):
                gates[name] = Gate(name, status="NOT_VERIFIED", evidence={"reason": "audit-only mode does not create an archive"})
            summary = summarized_gates(gates)
            return {
                "tool_version": TOOL_VERSION,
                "mode": "audit-only",
                "release_name": config.release_name,
                "source_selection_records": len(source_map),
                "sanitized_files": len(transforms),
                "gates": {name: gate.as_dict() for name, gate in sorted(gates.items())},
                **summary,
            }

        write_manifest(staging, config.allowlist)
        strip_result_3 = strip_xattrs(staging)
        post_manifest_gates = audit_tree(staging, config.public_view, source_root=config.source_root, attestation_required=True)
        post_manifest_gates["integrity"] = integrity_gate(staging)
        if not all_pass(post_manifest_gates, mandatory_pre + ["integrity"]):
            summary = summarized_gates(post_manifest_gates)
            return {
                "tool_version": TOOL_VERSION,
                "mode": "build",
                "release_name": config.release_name,
                "xattr_cleanup": [strip_result, strip_result_2, strip_result_3],
                "gates": {name: gate.as_dict() for name, gate in sorted(post_manifest_gates.items())},
                **summary,
            }

        archive_results: list[dict[str, Any]] = []
        temporary_outputs: list[tuple[Path, Path]] = []
        overall_ok = True
        for format_name, final_path in final_paths.items():
            temp_a = temp / f"first{archive_suffix(format_name)}"
            temp_b = temp / f"second{archive_suffix(format_name)}"
            writer = create_archive(staging, temp_a, format_name, config.source_date_epoch)
            create_archive(staging, temp_b, format_name, config.source_date_epoch)
            reproducibility_gate = Gate("reproducibility")
            first_hash = sha256_file(temp_a)
            second_hash = sha256_file(temp_b)
            if first_hash != second_hash:
                reproducibility_gate.findings.append(Finding("reproducibility.byte_mismatch", final_path.name, "duplicate archive builds differ"))
            reproducibility_gate.evidence = {
                "method": "byte-for-byte duplicate archive build",
                "first_sha256": first_hash,
                "second_sha256": second_hash,
            }
            reproducibility_gate.finish()
            verification = verify_archive(temp_a, expected_staging=staging)
            doc_execution = Gate("documentation_execution", status="NOT_VERIFIED")
            with tempfile.TemporaryDirectory(prefix=".experiment-release-doc-", dir=config.output_directory) as doc_temp_raw:
                doc_temp = Path(doc_temp_raw)
                extraction = system_integrity_and_extract(temp_a, doc_temp)
                roots = sorted(item for item in doc_temp.iterdir() if item.is_dir())
                if extraction.get("status") == "PASS" and len(roots) == 1:
                    doc_execution = execute_documentation_commands(roots[0], temp_a, config.documentation)
                    after_commands = integrity_gate(roots[0])
                    if after_commands.status != "PASS":
                        doc_execution.findings.extend(after_commands.findings)
                        doc_execution.finish("FAIL")
                else:
                    doc_execution = Gate("documentation_execution")
                    doc_execution.findings.append(Finding("documentation.extraction_failed", final_path.name, "could not prepare extracted tree"))
                    doc_execution.evidence = extraction
                    doc_execution.finish()
            verification["gates"]["documentation_execution"] = doc_execution.as_dict()
            verification["gates"]["reproducibility"] = reproducibility_gate.as_dict()
            verification["REPRODUCIBILITY_GATE"] = reproducibility_gate.status
            verification["DOCUMENTATION_GATE"] = "PASS" if verification.get("DOCUMENTATION_GATE") == "PASS" and doc_execution.status == "PASS" else "FAIL"
            verification["OVERALL_RELEASE_STATUS"] = "PASS" if all(
                verification.get(name) == "PASS"
                for name in (
                    "INTEGRITY_GATE",
                    "PRIVACY_GATE",
                    "DOCUMENTATION_GATE",
                    "ARCHIVE_HYGIENE_GATE",
                    "REPRODUCIBILITY_GATE",
                )
            ) else "FAIL"
            verification["writer"] = writer
            archive_results.append(verification)
            if verification["OVERALL_RELEASE_STATUS"] != "PASS":
                overall_ok = False
            temporary_outputs.append((temp_a, final_path))
        if not overall_ok:
            return {
                "tool_version": TOOL_VERSION,
                "mode": "build",
                "release_name": config.release_name,
                "archives": archive_results,
                "OVERALL_RELEASE_STATUS": "FAIL",
                "publication": "ABORTED_BEFORE_ATOMIC_RENAME",
            }

        published: list[dict[str, Any]] = []
        for temporary, final_path in temporary_outputs:
            os.replace(temporary, final_path)
            report_path = final_path.with_name(final_path.name + ".release-report.json")
            archive_result = next(item for item in archive_results if item["archive_format"] == archive_format_from_path(final_path))
            report_payload = {
                **archive_result,
                "archive": final_path.name,
                "archive_sha256": sha256_file(final_path),
                "published_utc": current_utc(),
            }
            temp_report = temp / (report_path.name + ".tmp")
            write_text(temp_report, json_text(report_payload))
            os.replace(temp_report, report_path)
            published.append({
                "filename": final_path.name,
                "format": archive_format_from_path(final_path),
                "size_bytes": final_path.stat().st_size,
                "sha256": sha256_file(final_path),
                "file_count": len(manifest_records(staging, config.allowlist)) + len(MANIFEST_SELF_EXCLUSIONS),
                "audit_status": "PASS",
                "report": report_path.name,
            })
        return {
            "tool_version": TOOL_VERSION,
            "mode": "build",
            "release_name": config.release_name,
            "release_class": config.release_class,
            "source_selection_records": len(source_map),
            "sanitized_files": len(transforms),
            "published": published,
            "INTEGRITY_GATE": "PASS",
            "PRIVACY_GATE": "PASS",
            "DOCUMENTATION_GATE": "PASS",
            "ARCHIVE_HYGIENE_GATE": "PASS",
            "REPRODUCIBILITY_GATE": "PASS",
            "OVERALL_RELEASE_STATUS": "PASS",
            "publication": "ATOMIC_RENAME_COMPLETE",
        }


def project_launcher_text() -> str:
    return '''#!/usr/bin/env python3
"""Thin launcher for the canonical experiment-release implementation."""
from __future__ import annotations
import os
import runpy
import sys
from pathlib import Path

explicit = os.environ.get("EXPERIMENT_RELEASE_TOOL")
candidates = [Path(explicit)] if explicit else []
for parent in Path(__file__).resolve().parents:
    candidates.append(parent / "shared_tools" / "release" / "release_tool.py")
for candidate in candidates:
    if candidate.is_file():
        sys.path.insert(0, str(candidate.parent))
        runpy.run_path(str(candidate), run_name="__main__")
        raise SystemExit(0)
raise SystemExit("Canonical experiment-release tool not found; set EXPERIMENT_RELEASE_TOOL")
'''


def init_project(project_root: Path, template_dir: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ReleaseError("project root does not exist")
    files = {
        "release_config.json": template_dir / "release_config.json",
        "release_exceptions.json": template_dir / "release_exceptions.json",
        "Makefile.release": template_dir / "Makefile.release",
        "RELEASE_README.md": template_dir / "RELEASE_README.md",
    }
    created: list[str] = []
    skipped: list[str] = []
    for relative, source in files.items():
        target = project_root / relative
        if target.exists():
            skipped.append(relative)
            continue
        if not source.is_file():
            raise ReleaseError(f"initializer template missing: {relative}")
        write_bytes(target, source.read_bytes())
        created.append(relative)
    launcher = project_root / "tools/release.py"
    if launcher.exists():
        skipped.append("tools/release.py")
    else:
        write_text(launcher, project_launcher_text(), mode=0o755)
        created.append("tools/release.py")
    return {
        "status": "PASS",
        "created": sorted(created),
        "skipped_existing": sorted(skipped),
        "next_command": "python3 tools/release.py build --config release_config.json",
    }
