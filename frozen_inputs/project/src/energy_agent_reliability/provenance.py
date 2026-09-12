"""Immutable-style records for raw-source provenance and freeze hashes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_record(
    *,
    source: str,
    request_url: str,
    query_parameters: dict[str, Any],
    path: Path,
    original_filename: str,
    source_note: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "request_url": request_url,
        "query_parameters": query_parameters,
        "retrieval_utc": utc_now(),
        "original_filename": original_filename,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "license_or_source_note": source_note,
    }


def _filename_from_response(response: httpx.Response, url: str) -> str:
    disposition = str(response.headers.get("content-disposition", ""))
    if "filename=" in disposition:
        return disposition.rsplit("filename=", 1)[1].strip('"; ')
    parsed = urlparse(url)
    return Path(parsed.path).name or "download.bin"


def download_file(
    *,
    source: str,
    url: str,
    destination: str | Path,
    query_parameters: dict[str, Any],
    source_note: str,
    max_bytes: int | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 5,
    initial_backoff_s: float = 1.0,
    max_backoff_s: float = 16.0,
) -> dict[str, Any]:
    """Download to an atomic partial file, then write a provenance sidecar.

    The URL's secret-like parameters are not persisted; callers must pass a redacted
    `query_parameters` dictionary for the provenance sidecar.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix(target.suffix + ".source.json")
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() and sidecar.exists():
        existing_record = cast(dict[str, Any], json.loads(sidecar.read_text(encoding="utf-8")))
        if existing_record.get("sha256") == sha256_file(target):
            existing_record.update(
                {
                    "source": source,
                    "request_url": url.split("?", 1)[0],
                    "query_parameters": query_parameters,
                    "original_filename": existing_record.get("original_filename", target.name),
                    "byte_size": target.stat().st_size,
                    "license_or_source_note": source_note,
                }
            )
            sidecar.write_text(json.dumps(existing_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return existing_record
    if target.exists() and not partial.exists():
        target.replace(partial)
    original_filename = "download.bin"
    for attempt in range(max_retries):
        # This workstation's direct CAISO/OEDI routes are more reliable than its
        # SOCKS proxy; retain an environment-proxy retry for networks that need it.
        use_environment_proxy = attempt % 2 == 1
        existing = partial.stat().st_size if partial.exists() else 0
        headers: dict[str, str] = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with (
                httpx.Client(
                    follow_redirects=True, timeout=timeout_s, trust_env=use_environment_proxy
                ) as client,
                client.stream("GET", url, headers=headers) as response,
            ):
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if not append:
                    existing = 0
                content_length = response.headers.get("content-length")
                if max_bytes is not None and content_length is not None and existing + int(content_length) > max_bytes:
                    raise ValueError(f"Refusing {target.name}: exceeds configured {max_bytes} byte limit.")
                bytes_written = existing
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_bytes():
                        bytes_written += len(chunk)
                        if max_bytes is not None and bytes_written > max_bytes:
                            raise ValueError(
                                f"Refusing {target.name}: streaming body exceeds byte limit."
                            )
                        handle.write(chunk)
                original_filename = _filename_from_response(response, url)
            partial.replace(target)
            break
        except httpx.HTTPError:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(initial_backoff_s * (2**attempt), max_backoff_s))
    record = source_record(
        source=source,
        request_url=url.split("?", 1)[0],
        query_parameters=query_parameters,
        path=target,
        original_filename=original_filename,
        source_note=source_note,
    )
    sidecar.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def freeze_manifest(paths: list[Path], destination: str | Path) -> dict[str, Any]:
    root = Path(destination).parent
    files = [
        {"path": str(path.relative_to(root)), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    manifest = {"created_utc": utc_now(), "algorithm": "sha256", "files": files}
    Path(destination).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_sha256sum_manifest(paths: list[Path], destination: str | Path, root: str | Path) -> None:
    base = Path(root)
    rows = [
        f"{sha256_file(path)}  {path.relative_to(base).as_posix()}"
        for path in sorted(paths)
        if path.is_file()
    ]
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_sha256sum_manifest(path: str | Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, relative = raw.split(maxsplit=1)
        manifest[relative.strip()] = digest
    return manifest
