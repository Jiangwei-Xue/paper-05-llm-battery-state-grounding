"""Evidence redaction and row-level hash-chain helpers."""

from __future__ import annotations

from typing import Any

from .request import canonical_json, sha256_text

SECRET_HEADER_NAMES = {"authorization", "x-api-key", "api-key", "openai-api-key"}
SECRET_FIELD_NAMES = {"api_key", "access_token", "secret", "password"}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in SECRET_HEADER_NAMES or lowered in SECRET_FIELD_NAMES or "authorization" in lowered:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def hash_payload(value: Any) -> str:
    return sha256_text(canonical_json(value))


def build_hash_chain(items: list[tuple[str, Any]]) -> list[dict[str, str]]:
    previous = ""
    rows: list[dict[str, str]] = []
    for label, payload in items:
        payload_hash = hash_payload(payload)
        chained = sha256_text(previous + payload_hash)
        rows.append({"label": label, "payload_sha256": payload_hash, "chain_sha256": chained})
        previous = chained
    return rows


def evidence_contains_secret(value: Any) -> bool:
    encoded = canonical_json(value).lower()
    return any(token in encoded for token in ("bearer ", "sk-", "api_key", "access_token"))


def build_evidence_record(
    *,
    scenario_id: str,
    episode_id: str,
    run_id: str,
    model_condition: dict[str, Any],
    state_method: str,
    stage1_prompt: dict[str, Any],
    stage1_request: dict[str, Any],
    stage1_raw_response: dict[str, Any],
    stage2_prompt: dict[str, Any],
    stage2_request: dict[str, Any],
    stage2_raw_response: dict[str, Any],
    scorer_oracle_hash: str,
    outcome_vector: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "scenario_id": scenario_id,
        "episode_id": episode_id,
        "run_id": run_id,
        "model_condition": model_condition,
        "state_method": state_method,
        "stage1_prompt": stage1_prompt,
        "stage1_request_payload": redact_secrets(stage1_request),
        "stage1_raw_response": stage1_raw_response,
        "stage2_prompt": stage2_prompt,
        "stage2_request_payload": redact_secrets(stage2_request),
        "stage2_raw_response": stage2_raw_response,
        "scorer_oracle_hash": scorer_oracle_hash,
        "outcome_vector": outcome_vector,
    }
    record["row_level_sha256_chain"] = build_hash_chain(
        [
            ("stage1_prompt", stage1_prompt),
            ("stage1_request", record["stage1_request_payload"]),
            ("stage1_raw_response", stage1_raw_response),
            ("stage2_prompt", stage2_prompt),
            ("stage2_request", record["stage2_request_payload"]),
            ("stage2_raw_response", stage2_raw_response),
            ("outcome_vector", outcome_vector),
        ]
    )
    return record

