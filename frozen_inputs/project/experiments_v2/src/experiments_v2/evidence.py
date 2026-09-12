"""Hash-linked V2 call evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_json, sha256_text


def make_evidence_row(
    *,
    plan_row: dict[str, Any],
    payload: dict[str, Any],
    raw_response: str,
    parsed: dict[str, Any],
    mock: bool,
) -> dict[str, Any]:
    payload_hash = sha256_json(payload)
    raw_hash = sha256_text(raw_response)
    parsed_hash = sha256_json(parsed)
    row = {
        **plan_row,
        "mock_only": mock,
        "request_hash": payload_hash,
        "raw_response_hash": raw_hash,
        "parsed_output_hash": parsed_hash,
        "evidence_root_hash": sha256_json({"request": payload_hash, "raw": raw_hash, "parsed": parsed_hash}),
        "raw_response": raw_response,
        "parsed": parsed,
        "parser_success": True,
        "protocol_exclusion": None,
    }
    return row


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
