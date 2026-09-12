"""Frozen plan and prompt compiler for the V6 online-gate evidence tier."""

from __future__ import annotations

import copy
import random
from typing import Any, Final

from .models import PromptBundle, StateMethodName
from .request import canonical_json, oracle_leak_scan, sha256_text, stable_id
from .request_v2 import compile_stage1_prompt_v2, compile_stage2_prompt_v2

PROTOCOL_ID = "energybench_online_gate_v6"
RUN_ORDER_SEED = 20260721
STATE_METHOD: Final[StateMethodName] = "canonical_typed_carry"
REPETITIONS = 3
TERMINAL_SOC_KWH = 325.0


def build_run_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build 60 stable rows and balance the shuffled order across repetitions."""
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        task_hash = sha256_text(canonical_json(task))
        for repetition in range(REPETITIONS):
            identity = {
                "protocol_id": PROTOCOL_ID,
                "scenario_id": task["scenario_id"],
                "condition_id": "v2_deepseek_v4_flash_direct",
                "state_method": STATE_METHOD,
                "repetition": repetition,
            }
            rows.append(
                {
                    **identity,
                    "run_id": stable_id("v6og", identity),
                    "task_sha256": task_hash,
                    "planned_primary_calls": 2,
                    "evidence_tier": "v6_online_gate",
                    "formal_v4_analysis_eligible": False,
                }
            )
    random.Random(RUN_ORDER_SEED).shuffle(rows)
    for run_order, row in enumerate(rows):
        row["run_order"] = run_order
    return rows


def compile_stage1_prompt(task: dict[str, Any]) -> PromptBundle:
    base = compile_stage1_prompt_v2(task, STATE_METHOD)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "online_gate_stage1_v6"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    prompt["task_surface"]["execution_architecture"] = {
        "proposal_role": "candidate_dispatch_for_deterministic_feasibility_projection",
        "projection_objective_order": [
            "minimum_L1_distance_to_your_actions",
            "minimum_battery_throughput",
            "minimum_operating_cost_tiebreak",
        ],
        "event_is_not_visible_at_stage1": True,
    }
    return _bundle(base, prompt)


def compile_stage2_prompt(
    task: dict[str, Any], canonical_carrier: dict[str, Any]
) -> PromptBundle:
    base = compile_stage2_prompt_v2(task, STATE_METHOD, canonical_carrier)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "online_gate_stage2_v6"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    prompt["task_surface"]["execution_architecture"] = {
        "prefix_execution": "deterministically_projected_then_executed",
        "state_carrier_soc_source": "executed_projected_prefix",
        "proposal_role": "candidate_suffix_for_deterministic_feasibility_projection",
        "projection_objective_order": [
            "minimum_L1_distance_to_your_actions",
            "minimum_battery_throughput",
            "minimum_operating_cost_tiebreak",
        ],
    }
    return _bundle(base, prompt)


def _bundle(base: PromptBundle, prompt: dict[str, Any]) -> PromptBundle:
    rendered = "\n".join(
        [
            "Return exactly one compact JSON object and nothing else.",
            "Do not output Markdown, code fences, reasoning, or prose outside JSON.",
            canonical_json(prompt),
        ]
    )
    return PromptBundle(
        stage=base.stage,
        scenario_id=base.scenario_id,
        state_method=base.state_method,
        canonical_prompt=prompt,
        rendered_prompt=rendered,
        prompt_sha256=sha256_text(canonical_json(prompt)),
        model_visible_field_manifest=_field_manifest(prompt),
        oracle_leak_scan=oracle_leak_scan(prompt),
    )


def _field_manifest(value: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            fields.append(path)
            fields.extend(_field_manifest(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            fields.extend(_field_manifest(item, f"{prefix}[{index}]"))
    return fields
