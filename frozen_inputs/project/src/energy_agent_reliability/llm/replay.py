"""Offline replay from saved evidence without provider access."""

from __future__ import annotations

from typing import Any

from .evidence import build_hash_chain
from .response import parse_stage_response


def replay_evidence_record(record: dict[str, Any]) -> dict[str, Any]:
    stage1_text = str(record["stage1_raw_response"]["raw_text"])
    stage2_text = str(record["stage2_raw_response"]["raw_text"])
    activation = int(record.get("activation_step", 0))
    stage1 = parse_stage_response(stage1_text, "stage1", 96)
    stage2 = parse_stage_response(stage2_text, "stage2", 96 - activation)
    rebuilt_chain = build_hash_chain(
        [
            ("stage1_prompt", record["stage1_prompt"]),
            ("stage1_request", record["stage1_request_payload"]),
            ("stage1_raw_response", record["stage1_raw_response"]),
            ("stage2_prompt", record["stage2_prompt"]),
            ("stage2_request", record["stage2_request_payload"]),
            ("stage2_raw_response", record["stage2_raw_response"]),
            ("outcome_vector", record["outcome_vector"]),
        ]
    )
    saved_chain = record.get("row_level_sha256_chain")
    return {
        "passed": stage1.ok and stage2.ok and saved_chain == rebuilt_chain,
        "stage1_parse_ok": stage1.ok,
        "stage2_parse_ok": stage2.ok,
        "hash_chain_match": saved_chain == rebuilt_chain,
    }

