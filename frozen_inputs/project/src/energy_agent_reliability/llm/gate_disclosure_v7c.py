"""Frozen prompt and run-plan contracts for the V7C disclosure ablation."""

from __future__ import annotations

import copy
import random
from typing import Any, Final, Literal

from .models import PromptBundle, StateMethodName
from .request import canonical_json, oracle_leak_scan, sha256_text, stable_id
from .request_v2 import compile_stage1_prompt_v2, compile_stage2_prompt_v2

PROTOCOL_ID = "energybench_v7c_gate_disclosure_ablation"
RUN_ORDER_SEED = 2026072203
STATE_METHOD: Final[StateMethodName] = "canonical_typed_carry"
REPETITIONS = 3
TERMINAL_SOC_KWH = 325.0

ConditionName = Literal[
    "gate_hidden_raw_score_hidden",
    "gate_hidden_raw_score_disclosed",
    "gate_disclosed_raw_score_hidden",
    "gate_disclosed_raw_score_disclosed",
]

CONDITIONS: Final[tuple[ConditionName, ...]] = (
    "gate_hidden_raw_score_hidden",
    "gate_hidden_raw_score_disclosed",
    "gate_disclosed_raw_score_hidden",
    "gate_disclosed_raw_score_disclosed",
)


def build_run_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build balanced four-row blocks and randomize block and within-block order."""
    blocks: list[list[dict[str, Any]]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        task_sha256 = sha256_text(canonical_json(task))
        for repetition in range(REPETITIONS):
            block: list[dict[str, Any]] = []
            block_id = stable_id(
                "v7cb",
                {
                    "protocol_id": PROTOCOL_ID,
                    "scenario_id": task["scenario_id"],
                    "repetition": repetition,
                },
            )
            for condition in CONDITIONS:
                gate_disclosed, raw_score_disclosed = factors(condition)
                identity = {
                    "protocol_id": PROTOCOL_ID,
                    "scenario_id": task["scenario_id"],
                    "condition": condition,
                    "repetition": repetition,
                }
                block.append(
                    {
                        **identity,
                        "run_id": stable_id("v7c", identity),
                        "block_id": block_id,
                        "task_sha256": task_sha256,
                        "gate_disclosed": gate_disclosed,
                        "raw_score_disclosed": raw_score_disclosed,
                        "planned_provider_calls": 2,
                        "evidence_tier": "v7c_gate_disclosure_ablation",
                        "excluded_from_prior_claim_bearing_tiers": True,
                    }
                )
            blocks.append(block)
    rng = random.Random(RUN_ORDER_SEED)
    rng.shuffle(blocks)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        rng.shuffle(block)
        rows.extend(block)
    for run_order, row in enumerate(rows):
        row["run_order"] = run_order
    return rows


def factors(condition: str) -> tuple[bool, bool]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown V7C condition: {condition}")
    return condition.startswith("gate_disclosed"), condition.endswith(
        "raw_score_disclosed"
    )


def compile_stage1_prompt(task: dict[str, Any], condition: str) -> PromptBundle:
    base = compile_stage1_prompt_v2(task, STATE_METHOD)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "gate_disclosure_stage1_v7c"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    _apply_disclosures(prompt, condition, stage="stage1")
    return _bundle(base, prompt)


def compile_stage2_prompt(
    task: dict[str, Any], canonical_carrier: dict[str, Any], condition: str
) -> PromptBundle:
    base = compile_stage2_prompt_v2(task, STATE_METHOD, canonical_carrier)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "gate_disclosure_stage2_v7c"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    _apply_disclosures(prompt, condition, stage="stage2")
    return _bundle(base, prompt)


def _apply_disclosures(
    prompt: dict[str, Any], condition: str, *, stage: str
) -> None:
    gate_disclosed, raw_score_disclosed = factors(condition)
    if gate_disclosed:
        prompt["task_surface"]["execution_architecture"] = {
            "proposal_role": (
                "candidate_dispatch_for_deterministic_feasibility_projection"
                if stage == "stage1"
                else "candidate_suffix_for_deterministic_feasibility_projection"
            ),
            "projection_objective_order": [
                "minimum_L1_distance_to_your_actions",
                "minimum_battery_throughput",
                "minimum_physical_operating_cost_tiebreak",
                "deterministic_time_weighted_tiebreak",
            ],
            "projection_occurs_after_raw_proposal": True,
        }
        if stage == "stage1":
            prompt["task_surface"]["execution_architecture"][
                "event_is_not_visible_at_stage1"
            ] = True
        else:
            prompt["task_surface"]["execution_architecture"].update(
                {
                    "prefix_execution": "deterministically_projected_then_executed",
                    "state_carrier_soc_source": "executed_projected_prefix",
                }
            )
    if raw_score_disclosed:
        prompt["task_surface"]["evaluation_contract"] = {
            "raw_proposal_scored_before_projection": True,
            "raw_feasibility_reported_separately": True,
            "raw_cost_reported_only_when_raw_proposal_is_feasible": True,
            "projection_cannot_improve_raw_proposal_score": True,
        }


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


__all__ = [
    "CONDITIONS",
    "PROTOCOL_ID",
    "REPETITIONS",
    "RUN_ORDER_SEED",
    "STATE_METHOD",
    "TERMINAL_SOC_KWH",
    "build_run_plan",
    "compile_stage1_prompt",
    "compile_stage2_prompt",
    "factors",
]
