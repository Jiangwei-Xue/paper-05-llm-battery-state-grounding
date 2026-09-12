"""Provider-free regression tests for the prospective E2b-v2 protocol."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_e2b_v2 as analysis  # noqa: E402
import run_e2b_v2 as runner  # noqa: E402
from e2b_parser_v2 import parse_output  # noqa: E402
from e2b_v2_common import (  # noqa: E402
    canonical_bytes,
    load_unique_tasks,
    model_visible_contract,
    raw_replay,
    render_prompt,
    request_payload,
    visible_suffix_rows,
)


def payload(actions: list[dict[str, Any]]) -> str:
    return json.dumps({"actions": actions}, separators=(",", ":"))


def test_parser_local_offset_and_dynamic_segment_capacity() -> None:
    parsed = parse_output(payload([{"start_offset": 0, "end_offset_exclusive": 1, "power_kw": 10.0}]), 3)
    assert parsed.ok
    assert parsed.dense_action_kw == [10.0, 0.0, 0.0]
    h_segments = [
        {"start_offset": index, "end_offset_exclusive": index + 1, "power_kw": (-1.0) ** index}
        for index in range(5)
    ]
    assert parse_output(payload(h_segments), 5).ok
    too_many = h_segments + [{"start_offset": 0, "end_offset_exclusive": 1, "power_kw": 0.0}]
    result = parse_output(payload(too_many), 5)
    assert not result.ok
    assert "segment_count_exceeds_horizon" in result.diagnostics


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ('{"actions":[{"start_step":0,"end_step_exclusive":1,"power_kw":1}]}', "segment_0_fields_invalid"),
        ('{"actions":[{"start_offset":3,"end_offset_exclusive":4,"power_kw":1}]}', "segment_0_range_invalid"),
        ('{"actions":[{"start_offset":true,"end_offset_exclusive":1,"power_kw":1}]}', "segment_0_offset_not_integer"),
        ('{"actions":[{"start_offset":0,"end_offset_exclusive":1,"power_kw":NaN}]}', "invalid_json"),
        ('{"actions":[{"start_offset":0,"end_offset_exclusive":1,"power_kw":Infinity}]}', "invalid_json"),
        ('{"actions":[{"start_offset":0,"end_offset_exclusive":1,"power_kw":251}]}', "segment_0_power_outside_hard_envelope"),
        ('{"actions":[],"state":{}}', "top_level_fields_invalid"),
        ('```json\n{"actions":[]}\n```', "invalid_json"),
    ],
)
def test_parser_rejects_invalid_contracts(raw: str, code: str) -> None:
    result = parse_output(raw, 3)
    assert not result.ok
    assert code in result.diagnostics


def test_parser_rejects_overlap_reverse_and_extra_fields() -> None:
    overlap = payload(
        [
            {"start_offset": 0, "end_offset_exclusive": 2, "power_kw": 1.0},
            {"start_offset": 1, "end_offset_exclusive": 3, "power_kw": 2.0},
        ]
    )
    assert "segment_1_unsorted_or_overlap" in parse_output(overlap, 3).diagnostics
    reverse = payload([{"start_offset": 2, "end_offset_exclusive": 1, "power_kw": 1.0}])
    assert "segment_0_range_invalid" in parse_output(reverse, 3).diagnostics
    extra = payload([{"start_offset": 0, "end_offset_exclusive": 1, "power_kw": 1.0, "timestamp": "x"}])
    assert "segment_0_fields_invalid" in parse_output(extra, 3).diagnostics


def test_zero_action_is_syntactically_valid_but_terminal_negative_control() -> None:
    task = load_unique_tasks()[0]["task"]
    horizon = len(visible_suffix_rows(task))
    parsed = parse_output('{"actions":[]}', horizon)
    assert parsed.ok
    initial = float(task["v2_prefix_intervention"]["canonical_event_soc_kwh"])
    score = raw_replay(task, parsed.dense_action_kw, initial_soc_kwh=initial)
    assert not score["engineering_feasible"]
    assert score["terminal_absolute_error_kwh"] > 1.0


def test_prompt_exposes_exact_schema_physics_and_no_global_t() -> None:
    task = load_unique_tasks()[0]["task"]
    prompt = render_prompt(task, "C1")
    for field in ("actions", "start_offset", "end_offset_exclusive", "power_kw"):
        assert f"`{field}`" in prompt or f'"{field}"' in prompt
    assert '{"actions":[{"start_offset":0,"end_offset_exclusive":4,"power_kw":10.0}]}' in prompt
    assert "soc[t+1] = soc[t] + 0.95 * power_kw[t] * 0.25" in prompt
    contract = model_visible_contract(task, "C1")
    assert all("t" not in row for row in contract["timeseries"])
    assert all(row["offset"] == index for index, row in enumerate(contract["timeseries"]))


def test_duplicate_request_bytes_and_treatment_scope() -> None:
    task = load_unique_tasks()[0]["task"]
    specs = yaml.safe_load((PROJECT / "experiments_v2/configs/models.yaml").read_text())["conditions"]
    for model in ("deepseek_formal", "qwen_flash"):
        c1 = request_payload(specs[model], render_prompt(task, "C1"))
        c2 = request_payload(specs[model], render_prompt(task, "C2"))
        s1 = request_payload(specs[model], render_prompt(task, "S1"))
        s2 = request_payload(specs[model], render_prompt(task, "S2"))
        assert canonical_bytes(c1) == canonical_bytes(c2)
        assert canonical_bytes(s1) == canonical_bytes(s2)
    c = model_visible_contract(task, "C1")
    s = model_visible_contract(task, "S1")
    assert c["carrier_state"]["soc_kwh"] != s["carrier_state"]["soc_kwh"]
    c["carrier_state"].pop("soc_kwh")
    s["carrier_state"].pop("soc_kwh")
    assert c == s


def test_provider_payloads_explicitly_disable_thinking() -> None:
    task = load_unique_tasks()[0]["task"]
    specs = yaml.safe_load((PROJECT / "experiments_v2/configs/models.yaml").read_text())["conditions"]
    deepseek = request_payload(specs["deepseek_formal"], render_prompt(task, "C1"))
    qwen = request_payload(specs["qwen_flash"], render_prompt(task, "C1"))
    assert deepseek["thinking"] == {"type": "disabled"}
    assert qwen["enable_thinking"] is False
    assert deepseek["response_format"] == qwen["response_format"] == {"type": "json_object"}


def synthetic_record(
    *,
    scenario: str,
    model: str,
    actions: dict[str, list[float] | None],
) -> dict[str, Any]:
    branches: dict[str, Any] = {}
    for branch, action in actions.items():
        ok = action is not None
        terminal = abs(sum(action or []))
        branches[branch] = {
            "parsed": {
                "ok": ok,
                "dense_action_kw": action or [],
                "diagnostics": [] if ok else ["invalid_json"],
            },
            "score": {"authoritative_replay": {"terminal_absolute_error_kwh": terminal}} if ok else None,
        }
    return {
        "scenario_id": scenario,
        "model_condition": model,
        "branches": branches,
        "same_information_mpc": {"action_kw": [0.0, 0.0]},
    }


def test_locked_analysis_identical_between_within_and_parser_failure() -> None:
    identical = synthetic_record(
        scenario="s0",
        model="deepseek_formal",
        actions={branch: [0.0, 0.0] for branch in analysis.BRANCHES},
    )
    metrics = analysis.block_metrics(identical)
    assert metrics is not None and metrics["energy_distance_ED_kwh"] == 0.0
    between = synthetic_record(
        scenario="s1",
        model="deepseek_formal",
        actions={"C1": [0.0, 0.0], "C2": [0.0, 0.0], "S1": [10.0, 10.0], "S2": [10.0, 10.0]},
    )
    metrics = analysis.block_metrics(between)
    assert metrics is not None and metrics["energy_distance_ED_kwh"] > 0.0
    within = synthetic_record(
        scenario="s2",
        model="deepseek_formal",
        actions={"C1": [0.0, 0.0], "C2": [10.0, 10.0], "S1": [0.0, 0.0], "S2": [10.0, 10.0]},
    )
    metrics = analysis.block_metrics(within)
    assert metrics is not None and metrics["energy_distance_ED_kwh"] < 0.0
    failed = synthetic_record(
        scenario="s3",
        model="deepseek_formal",
        actions={"C1": None, "C2": [0.0, 0.0], "S1": [0.0, 0.0], "S2": [0.0, 0.0]},
    )
    assert analysis.block_metrics(failed) is None


def test_analysis_rejects_unequal_horizon_and_bootstrap_is_reproducible() -> None:
    unequal = synthetic_record(
        scenario="s0",
        model="deepseek_formal",
        actions={"C1": [0.0], "C2": [0.0, 0.0], "S1": [0.0], "S2": [0.0]},
    )
    with pytest.raises(ValueError, match="unequal horizons"):
        analysis.block_metrics(unequal)
    rows = [
        synthetic_record(
            scenario=f"s{index}",
            model="deepseek_formal",
            actions={"C1": [0.0, 0.0], "C2": [0.0, 0.0], "S1": [float(index), 0.0], "S2": [float(index), 0.0]},
        )
        for index in range(4)
    ]
    first = analysis.analyze(rows, bootstrap_resamples=100)
    second = analysis.analyze(rows, bootstrap_resamples=100)
    assert first == second


def test_completion_gate_catches_v1_failure_shape() -> None:
    old_shape = [{"logical_provider_calls": 1, "branches": {}, "protocol_exclusions": []} for _ in range(360)]
    assert runner._completion_status(
        tier="F1", records=old_shape, planned_blocks=360, planned_calls=1800, partial_files=0
    ) == "HOLD"
    no_stage2 = [{"logical_provider_calls": 0, "branches": {}, "protocol_exclusions": []} for _ in range(120)]
    assert runner._completion_status(
        tier="F1", records=no_stage2, planned_blocks=120, planned_calls=480, partial_files=0
    ) == "HOLD"


def test_branch_failures_do_not_skip_sibling_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    item = load_unique_tasks()[0]
    plan = {
        "tier": "F1",
        "block_id": "test",
        "scenario_id": item["scenario_id"],
        "model_condition": "deepseek_formal",
        "branch_order": ["C1", "S1", "C2", "S2"],
        "source_record_path": item["source_record_path"],
        "source_record_sha256": item["source_record_sha256"],
        "task_sha256": item["task_sha256"],
    }
    calls: list[str] = []
    spec = yaml.safe_load((PROJECT / "experiments_v2/configs/models.yaml").read_text())["conditions"]["deepseek_formal"]
    plan["expected_request_sha256"] = {
        branch: runner.digest(request_payload(spec, render_prompt(item["task"], branch)))
        for branch in runner.BRANCHES
    }

    async def fake_call(*_args: Any, call_id: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(call_id)
        return {
            "status": "ok",
            "content": "not-json",
            "returned_model": "deepseek-v4-flash",
            "system_fingerprint": "fixed",
            "finish_reason": "stop",
            "attempt_count": 1,
        }

    monkeypatch.setattr(runner, "provider_call", fake_call)
    monkeypatch.setattr(runner, "same_information_mpc", lambda _task: {"action_kw": [0.0] * len(visible_suffix_rows(_task))})
    result = asyncio.run(runner.run_block(plan, {"deepseek_formal": spec}, {"deepseek_formal": "test"}, object()))
    assert len(calls) == 4
    assert set(result["branches"]) == set(runner.BRANCHES)
    assert all(not branch["parsed"]["ok"] for branch in result["branches"].values())


def test_frozen_report_links_schema_parser_and_analysis_if_present() -> None:
    manifest_path = ROOT / "reviews/e2b_protocol_v2/E2B_V2_PRE_CALL_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("pre-call manifest is generated by offline preparation")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"]["schema"]["sha256"]
    assert manifest["artifacts"]["parser"]["sha256"]
    assert manifest["artifacts"]["analysis"]["sha256"]
    assert manifest["provider_calls_performed"] == 0
    assert manifest["network_attempts"] == 0
