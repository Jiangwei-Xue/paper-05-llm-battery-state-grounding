"""Hash-linked V2 episode evidence and provider-free replay."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, cast

from .evidence import evidence_contains_secret, hash_payload, redact_secrets
from .models import StageName, StateMethodName
from .request import canonical_json, sha256_text
from .response_v2 import parse_stage_response_v2

EVIDENCE_ORDER = (
    "task",
    "stage1_prompt",
    "stage1_request",
    "stage1_raw_response",
    "stage1_parsed_response",
    "state_carrier",
    "stage2_prompt",
    "stage2_request",
    "stage2_raw_response",
    "stage2_parsed_response",
    "dispatch",
    "simulator_trace",
    "metrics",
)


def build_evidence_chain_v2(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    """Build a named parent-linked chain over every claim-bearing episode artifact."""
    parent = ""
    rows: list[dict[str, str]] = []
    for label in EVIDENCE_ORDER:
        payload_sha256 = hash_payload(artifacts[label])
        node_sha256 = sha256_text(f"{label}:{parent}:{payload_sha256}")
        rows.append(
            {
                "label": label,
                "parent_sha256": parent,
                "payload_sha256": payload_sha256,
                "node_sha256": node_sha256,
            }
        )
        parent = node_sha256
    return rows


def finalize_evidence_record_v2(record: dict[str, Any]) -> dict[str, Any]:
    """Redact, attach chain closure, and hash a complete V2 record."""
    redacted = redact_secrets(copy.deepcopy(record))
    if not isinstance(redacted, dict):
        raise TypeError("Evidence record must remain a JSON object after redaction.")
    clean: dict[str, Any] = redacted
    clean["evidence_chain"] = build_evidence_chain_v2(clean["artifacts"])
    clean["evidence_root_sha256"] = clean["evidence_chain"][-1]["node_sha256"]
    clean["verifier"] = {
        "version": "offline_replay_v2",
        "provider_access_required": False,
        "expected_evidence_root_sha256": clean["evidence_root_sha256"],
    }
    clean["record_sha256"] = sha256_text(canonical_json(clean))
    return clean


def replay_evidence_record_v2(record: dict[str, Any]) -> dict[str, Any]:
    """Reparse raw text and verify the complete evidence chain without network access."""
    saved_record_hash = record.get("record_sha256")
    unhashed = copy.deepcopy(record)
    unhashed.pop("record_sha256", None)
    record_hash_match = saved_record_hash == sha256_text(canonical_json(unhashed))
    artifacts = record["artifacts"]
    rebuilt_chain = build_evidence_chain_v2(artifacts)
    chain_match = rebuilt_chain == record.get("evidence_chain")
    root_match = bool(
        rebuilt_chain
        and rebuilt_chain[-1]["node_sha256"] == record.get("evidence_root_sha256")
        and record.get("verifier", {}).get("expected_evidence_root_sha256")
        == record.get("evidence_root_sha256")
    )
    method = state_method(str(record["state_method"]))
    stage1 = artifacts["stage1_raw_response"]
    stage2 = artifacts["stage2_raw_response"]
    stage1_expected = artifacts["task"]["model_visible_episode"]["stage_1"]["visible_timeseries"]
    stage2_expected = artifacts["task"]["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    stage1_parse = _replay_stage_parse(
        stage1,
        "stage1",
        method,
        96,
        stage1_expected,
    )
    activation = int(artifacts["task"]["visible_update"]["activation_step"])
    stage2_parse = _replay_stage_parse(
        stage2,
        "stage2",
        method,
        96 - activation,
        stage2_expected,
    )
    stage1_parse_match = stage1_parse == artifacts["stage1_parsed_response"]
    stage2_parse_match = stage2_parse == artifacts["stage2_parsed_response"]
    secret_free = not evidence_contains_secret(record)
    task_hash_match = record.get("task_sha256") == hash_payload(artifacts["task"])
    passed = all(
        (
            record_hash_match,
            chain_match,
            root_match,
            stage1_parse_match,
            stage2_parse_match,
            secret_free,
            task_hash_match,
        )
    )
    return {
        "passed": passed,
        "record_hash_match": record_hash_match,
        "evidence_chain_match": chain_match,
        "evidence_root_match": root_match,
        "stage1_parse_match": stage1_parse_match,
        "stage2_parse_match": stage2_parse_match,
        "secret_free": secret_free,
        "task_hash_match": task_hash_match,
    }


def state_method(value: str) -> StateMethodName:
    """Narrow a persisted method name after explicit membership validation."""
    allowed = {
        "rolling_summary",
        "visible_carry",
        "typed_state",
        "canonical_typed_carry",
    }
    if value not in allowed:
        raise ValueError(f"Unknown state method: {value}")
    return value  # type: ignore[return-value]


def _replay_stage_parse(
    raw_response: dict[str, Any],
    stage: str,
    method: StateMethodName,
    required_dispatch_length: int,
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay parser output without treating fail-closed skips as JSON parse attempts."""
    status = raw_response.get("status", "ok" if raw_response.get("content") else None)
    diagnostic = raw_response.get("parse_replay_diagnostic")
    control_plane_failure = diagnostic in {
        "provider_response_not_ok",
        "route_validation_failed",
        "stage2_not_run",
    }
    if status != "ok" or control_plane_failure:
        return {
            "ok": False,
            "parsed_state": None,
            "dispatch_kw": [],
            "diagnostics": [str(raw_response.get("parse_replay_diagnostic", "provider_response_not_ok"))],
            "protocol_exclusion": None,
        }
    return asdict(
        parse_stage_response_v2(
            str(raw_response.get("content", "")),
            cast(StageName, stage),
            method,
            required_dispatch_length,
            expected_dispatch_rows=expected_rows,
            finish_reason=raw_response.get("finish_reason"),
        )
    )
