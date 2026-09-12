"""Provider-free contract tests for the unexecuted F0 bridge.

These tests exercise the timestamped parser and the F0 plan shape.  They do
not construct provider requests and do not treat an absent response as a
parseable model result.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PARSER_PATH = ROOT / "scripts/e2b_parser_v1.py"
PLAN_PATH = ROOT / "reviews/e2b_protocol_v2/E2B_V2_F0_OPTIONAL_BRIDGE_PLAN.jsonl"


def _load_parser():
    spec = importlib.util.spec_from_file_location("f0_e2b_parser_v1", PARSER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state() -> dict[str, object]:
    return {
        "current_step": 0,
        "soc_kwh": 250.0,
        "usable_capacity_kwh": 500.0,
        "charge_limit_kw": 250.0,
        "discharge_limit_kw": 250.0,
        "export_limit_kw": 150.0,
        "reserve_kwh": 75.0,
        "forecast_version": "f0-offline-fixture",
        "active_commitments": [],
        "revoked_commitments": [],
        "sequence": 0,
    }


def _timestamps(n: int = 4) -> list[str]:
    return [f"2024-01-01T00:{15 * i:02d}:00Z" for i in range(n)]


def _payload(timestamps: list[str], **segment: object) -> str:
    default = {
        "start_step": 0,
        "end_step_exclusive": 2,
        "start_timestamp_utc": timestamps[0],
        "end_timestamp_utc": timestamps[2],
        "power_kw": 40.0,
    }
    default.update(segment)
    return json.dumps({"state": _state(), "actions": [default]})


def test_f0_plan_is_30_blocks_and_120_planned_calls() -> None:
    rows = [json.loads(line) for line in PLAN_PATH.read_text().splitlines() if line.strip()]
    assert len(rows) == 30
    assert {row["planned_provider_calls"] for row in rows} == {4}
    assert sum(row["planned_provider_calls"] for row in rows) == 120
    assert {row["execution_status"] for row in rows} == {
        "OPTIONAL_NOT_ADOPTED_REQUIRES_SEPARATE_CONTRACT_FREEZE"
    }


def test_valid_timestamped_action_is_accepted() -> None:
    parser = _load_parser()
    timestamps = _timestamps()
    parsed = parser.parse_output(_payload(timestamps), timestamps)
    assert parsed.ok is True
    assert parsed.dense_action_kw == [40.0, 40.0, 0.0, 0.0]
    assert parsed.diagnostics == []


def test_local_index_and_utc_timestamp_must_agree() -> None:
    parser = _load_parser()
    timestamps = _timestamps()
    parsed = parser.parse_output(
        _payload(timestamps, start_timestamp_utc=timestamps[1]), timestamps
    )
    assert parsed.ok is False
    assert "segment_0_timestamp_index_mismatch" in parsed.diagnostics


def test_non_utc_timestamp_is_rejected() -> None:
    parser = _load_parser()
    timestamps = _timestamps()
    parsed = parser.parse_output(
        _payload(timestamps, start_timestamp_utc="2024-01-01T00:00:00-05:00"), timestamps
    )
    assert parsed.ok is False
    assert "segment_0_timestamp_invalid" in parsed.diagnostics


def test_overlap_and_power_bound_are_rejected() -> None:
    parser = _load_parser()
    timestamps = _timestamps()
    payload = {
        "state": _state(),
        "actions": [
            {
                "start_step": 0,
                "end_step_exclusive": 2,
                "start_timestamp_utc": timestamps[0],
                "end_timestamp_utc": timestamps[2],
                "power_kw": 251.0,
            },
            {
                "start_step": 1,
                "end_step_exclusive": 3,
                "start_timestamp_utc": timestamps[1],
                "end_timestamp_utc": timestamps[3],
                "power_kw": 1.0,
            },
        ],
    }
    parsed = parser.parse_output(json.dumps(payload), timestamps)
    assert parsed.ok is False
    assert "segment_0_power_invalid" in parsed.diagnostics
    assert "segment_1_overlap" in parsed.diagnostics


def test_invalid_json_is_failure_without_dense_action() -> None:
    parser = _load_parser()
    parsed = parser.parse_output("not-json", _timestamps())
    assert parsed.ok is False
    assert parsed.raw_action is None
    assert parsed.dense_action_kw == []
    assert parsed.diagnostics == ["invalid_json"]


def test_offline_test_has_no_provider_or_network_surface() -> None:
    source = Path(__file__).read_text()
    assert "request" + "s." not in source
    assert "htt" + "px" not in source
    assert "open" + "ai" not in source.lower()
    assert "provider" + "_calls_performed" not in source
