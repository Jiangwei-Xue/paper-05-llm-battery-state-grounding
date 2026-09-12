from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_e2_frozen_task_balance_and_disjointness() -> None:
    tasks = _jsonl(ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl")
    assert len(tasks) == len({str(task["scenario_id"]) for task in tasks}) == 60
    assert Counter(task["event_family"] for task in tasks) == {
        "forecast_revision": 12,
        "battery_capacity_derating": 12,
        "power_limit_derating": 12,
        "export_limit_update": 12,
        "reserve_commitment_update": 12,
    }
    assert Counter(
        float(task["v2_prefix_intervention"]["divergence_kwh"]) for task in tasks
    ) == {50.0: 20, 75.0: 20, 100.0: 20}
    prior = []
    for name in ("P0_FROZEN_TASKS.jsonl", "E1_FROZEN_TASKS.jsonl"):
        prior.extend(_jsonl(ROOT / "experiments_v2/manifests" / name))
    prior_sources = {
        str(task.get("source_scenario_id", task["scenario_id"])) for task in prior
    }
    assert not {str(task["source_scenario_id"]) for task in tasks}.intersection(prior_sources)


def test_e2_selected_rows_pass_frozen_admission() -> None:
    tasks = _jsonl(ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl")
    selected = {str(task["scenario_id"]) for task in tasks}
    rows = _jsonl(ROOT / "experiments_v2/manifests/E2_ADMISSION_RESULTS.jsonl")
    selected_rows = [row for row in rows if str(row["scenario_id"]) in selected]
    assert len(selected_rows) == 60
    assert all(bool(row["admitted"]) for row in selected_rows)
    assert all(float(row["deterministic_action_l1_kwh"]) >= 25.0 for row in selected_rows)
    assert all(float(row["binding_slack_fraction"]) <= 0.05 for row in selected_rows)
