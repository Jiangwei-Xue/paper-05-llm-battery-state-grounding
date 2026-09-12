from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reviews/qwen37plus_f1f0_v1"
sys.path.insert(0, str(ROOT / "scripts"))

from e2b_v2_common import request_payload  # noqa: E402


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_protocol_is_pre_call_frozen_and_direct() -> None:
    protocol = yaml.safe_load((ROOT / "protocol/qwen37plus_f1f0_v1/QWEN37PLUS_F1F0_PROTOCOL_V1.yaml").read_text(encoding="utf-8"))
    assert protocol["status"] == "PRE_CALL_FROZEN"
    assert protocol["controls"]["concurrency"] == 8
    assert protocol["controls"]["direct_route_only"] is True
    assert protocol["controls"]["fallback"] == "disabled"
    assert protocol["matrix"]["planned_calls"] == 300


def test_model_payload_is_qwen37_plus_with_thinking_disabled() -> None:
    spec = json.loads((REVIEW / "QWEN37PLUS_MODEL_SPEC.json").read_text(encoding="utf-8"))
    payload = request_payload(spec, "fixed prompt")
    assert payload["model"] == "qwen3.7-plus"
    assert payload["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert "api" not in json.dumps(payload).lower()


def test_formal_plans_have_expected_cardinality_and_unique_keys() -> None:
    f1 = rows(REVIEW / "QWEN37PLUS_F1_RUN_PLAN.jsonl")
    f0 = rows(REVIEW / "QWEN37PLUS_F0_RUN_PLAN.jsonl")
    formal = rows(REVIEW / "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl")
    assert len(f1) == 60
    assert len(f0) == 15
    assert len(formal) == 75
    assert len({str(row["block_id"]) for row in formal}) == 75
    assert all(str(row["model_condition"]) == "qwen37_plus_direct" for row in formal)
    assert all(int(row["planned_provider_calls"]) == 4 for row in formal)


def test_duplicate_branch_payload_hashes_are_frozen() -> None:
    for row in rows(REVIEW / "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl"):
        expected = row["expected_request_sha256"]
        assert expected["C1"] == expected["C2"]
        assert expected["S1"] == expected["S2"]


def test_task_source_hashes_are_inherited() -> None:
    task_rows = rows(REVIEW / "QWEN37PLUS_TASK_HASH_MANIFEST.jsonl")
    assert len(task_rows) == 60
    assert len({str(row["scenario_id"]) for row in task_rows}) == 60
    for row in task_rows:
        source = ROOT.parent / str(row["source_record_path"])
        assert source.is_file()
