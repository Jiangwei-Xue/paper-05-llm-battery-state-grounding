"""Prompt and run-plan contracts for the V7B state-consequential challenge."""

from __future__ import annotations

import copy
import random
from typing import Any

from .models import PromptBundle
from .paired_authority_v1 import (
    Owner,
    stage2_noncarrier_sha256,
)
from .paired_authority_v1 import (
    compile_branched_stage2_prompt as compile_paired_stage2,
)
from .paired_authority_v1 import (
    compile_shared_stage1_prompt as compile_paired_stage1,
)
from .protocol_v2 import ProviderSpecV2
from .request import canonical_json, oracle_leak_scan, sha256_text, stable_id

PROTOCOL_ID = "energybench_v7b_state_consequential_carrier"
TERMINAL_SOC_KWH = 325.0
RUN_ORDER_SEED = 2026072202
REPETITIONS = 3


def build_run_plan(
    tasks: list[dict[str, Any]], spec: ProviderSpecV2
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        task_sha256 = sha256_text(canonical_json(task))
        for repetition in range(REPETITIONS):
            identity = {
                "protocol_id": PROTOCOL_ID,
                "scenario_id": task["scenario_id"],
                "condition_id": spec.condition_id,
                "representation": "typed_json",
                "repetition": repetition,
            }
            rows.append(
                {
                    **identity,
                    "paired_run_id": stable_id("v7b", identity),
                    "task_sha256": task_sha256,
                    "planned_provider_calls": 3,
                    "planned_branch_outcomes": 2,
                    "excluded_from_primary_and_extension": True,
                    "corrective_evidence_tier": "v7b_state_consequential",
                }
            )
    rng = random.Random(RUN_ORDER_SEED)
    rng.shuffle(rows)
    for run_order, row in enumerate(rows):
        order = ["model", "deterministic_system"]
        rng.shuffle(order)
        row["run_order"] = run_order
        row["stage2_branch_order"] = order
    return rows


def compile_shared_stage1_prompt(task: dict[str, Any], representation: str) -> PromptBundle:
    if representation != "typed_json":
        raise ValueError("V7B freezes typed_json representation")
    base = compile_paired_stage1(task, "typed_json")
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "state_consequential_stage1_v7b"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    prompt["task_surface"]["evaluation_contract"] = {
        "raw_dispatch_scored": True,
        "deterministic_action_gate_disclosed": False,
        "action_projection_applied_to_primary_endpoint": False,
    }
    prompt["paired_authority_contract"]["evidence_tier"] = (
        "post_review_state_consequential_carrier_challenge"
    )
    return _bundle(base, prompt)


def compile_branched_stage2_prompt(
    task: dict[str, Any],
    representation: str,
    owner: Owner,
    carrier_value: dict[str, Any] | str,
) -> PromptBundle:
    if representation != "typed_json":
        raise ValueError("V7B freezes typed_json representation")
    base = compile_paired_stage2(task, "typed_json", owner, carrier_value)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "state_consequential_stage2_v7b"
    prompt["task_surface"]["terminal_soc_target_kwh"] = TERMINAL_SOC_KWH
    prompt["task_surface"]["evaluation_contract"] = {
        "raw_dispatch_scored": True,
        "deterministic_action_gate_disclosed": False,
        "action_projection_applied_to_primary_endpoint": False,
    }
    prompt["paired_authority_contract"]["evidence_tier"] = (
        "post_review_state_consequential_carrier_challenge"
    )
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


__all__ = [
    "PROTOCOL_ID",
    "REPETITIONS",
    "RUN_ORDER_SEED",
    "TERMINAL_SOC_KWH",
    "build_run_plan",
    "compile_branched_stage2_prompt",
    "compile_shared_stage1_prompt",
    "stage2_noncarrier_sha256",
]
