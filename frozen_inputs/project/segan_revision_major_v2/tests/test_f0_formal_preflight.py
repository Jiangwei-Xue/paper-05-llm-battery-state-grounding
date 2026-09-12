"""Offline checks for the separately frozen F0 formal package."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "reviews/f0_formal_v1/F0_FORMAL_RUN_PLAN.jsonl"
MANIFEST = ROOT / "reviews/f0_formal_v1/F0_FORMAL_PRE_CALL_MANIFEST.json"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_formal_plan_has_30_blocks_and_120_calls() -> None:
    rows = load_jsonl(PLAN)
    assert len(rows) == 30
    assert sum(int(row["planned_provider_calls"]) for row in rows) == 120
    assert len({str(row["block_id"]) for row in rows}) == 30


def test_formal_plan_balances_models_events_and_scenarios() -> None:
    rows = load_jsonl(PLAN)
    assert Counter(str(row["model_condition"]) for row in rows) == Counter({"deepseek_formal": 15, "qwen_flash": 15})
    assert Counter(str(row["event_family"]) for row in rows) == Counter({
        "battery_capacity_derating": 6,
        "export_limit_update": 6,
        "forecast_revision": 6,
        "power_limit_derating": 6,
        "reserve_commitment_update": 6,
    })
    assert len({str(row["scenario_id"]) for row in rows}) == 15


def test_formal_duplicate_payload_hashes_and_hold_state() -> None:
    rows = load_jsonl(PLAN)
    for row in rows:
        expected = row["expected_request_sha256"]
        assert expected["C1"] == expected["C2"]
        assert expected["S1"] == expected["S2"]
        assert row["execution_status"] == "FROZEN_NOT_EXECUTED_AWAITING_AUTHORIZATION"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "PRE_CALL_FROZEN_HOLD"
    assert manifest["planned_provider_calls"] == 120
    assert manifest["provider_calls_performed"] == 0
    assert manifest["network_attempts"] == 0
    assert manifest["explicit_user_authorization_for_live_execution"] is False
    assert "api_key" not in json.dumps(manifest).lower()
