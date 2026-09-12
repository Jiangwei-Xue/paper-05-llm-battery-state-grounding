from __future__ import annotations

import json
from pathlib import Path

from experiments_v2.e2 import BRANCHES, build_e2_plan, e2_states
from experiments_v2.hashing import sha256_json
from experiments_v2.p0 import build_payload, compile_prompt

ROOT = Path(__file__).resolve().parents[2]


def _tasks() -> list[dict]:
    path = ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_e2_plan_cardinality_and_balance() -> None:
    plan = build_e2_plan(_tasks())
    assert len(plan) == 360
    assert sum(row["planned_stage_calls"] for row in plan) == 2160
    assert len({row["block_id"] for row in plan}) == 360
    assert {tuple(row["stage2_branches"]) for row in plan} == {BRANCHES}
    assert sum(row["model_condition"] == "deepseek_formal" for row in plan) == 180
    assert sum(row["model_condition"] == "qwen_flash" for row in plan) == 180


def test_e2_carrier_mutations_and_null_payload() -> None:
    for task in _tasks():
        states = e2_states(task)
        assert states["C"] == states["C2"]
        assert states["S"]["soc_kwh"] == task["v2_prefix_intervention"]["stale_event_soc_kwh"]
        assert states["S"]["soc_kwh"] != states["C"]["soc_kwh"]
        changed_k = {key for key in states["C"] if states["C"][key] != states["K"][key]}
        assert changed_k <= {"active_commitments", "revoked_commitments", "sequence"}
        c = compile_prompt(task, "I3", "stage2", states["C"])
        c2 = compile_prompt(task, "I3", "stage2", states["C2"])
        payload_c = build_payload(
            model_id="model", provider="deepseek", interface="I3", prompt=c, max_output_tokens=16000
        )
        payload_c2 = build_payload(
            model_id="model",
            provider="deepseek",
            interface="I3",
            prompt=c2,
            max_output_tokens=16000,
        )
        assert sha256_json(payload_c) == sha256_json(payload_c2)
