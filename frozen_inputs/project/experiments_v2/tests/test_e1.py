from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiments_v2.e1 import build_e1_plan

ROOT = Path(__file__).resolve().parents[2]


def _tasks() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl")
        .read_text()
        .splitlines()
        if line
    ]


def test_e1_plan_cardinality_and_balance() -> None:
    plan = build_e1_plan(_tasks())
    assert len(plan) == 240
    assert len({row["episode_id"] for row in plan}) == 240
    assert Counter(row["model_condition"] for row in plan) == {
        "deepseek_formal": 120,
        "qwen_flash": 120,
    }
    assert Counter(row["interface"] for row in plan) == {"I0": 120, "I3": 120}
    assert all(row["planned_stage_calls"] == 2 for row in plan)


def test_e1_tasks_preserve_non_degenerate_admission() -> None:
    tasks = _tasks()
    report = json.loads(
        (ROOT / "experiments_v2/reports/E1_PREPARATION_REPORT.json").read_text()
    )
    assert report["status"] == "pass"
    assert len(tasks) == 20
    assert Counter(task["event_family"] for task in tasks) == {
        "forecast_revision": 4,
        "battery_capacity_derating": 4,
        "power_limit_derating": 4,
        "export_limit_update": 4,
        "reserve_commitment_update": 4,
    }
    assert all(task["terminal_soc_target_kwh"] == 325.0 for task in tasks)


def test_forecast_revision_is_materialized_and_causal() -> None:
    forecast_tasks = [task for task in _tasks() if task["event_family"] == "forecast_revision"]
    assert len(forecast_tasks) == 4
    for task in forecast_tasks:
        metadata = task["v2_forecast_revision_materialization"]
        assert metadata["future_realized_pv_used"] is False
        activation = task["visible_update"]["activation_step"]
        initial = task["model_visible_episode"]["stage_1"]["visible_timeseries"][activation:]
        revised = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
        assert any(
            left["pv_forecast_kw"] != right["pv_forecast_kw"]
            for left, right in zip(initial, revised, strict=True)
        )
