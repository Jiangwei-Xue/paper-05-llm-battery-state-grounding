"""Offline pilot token and cost estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from energy_agent_reliability.provenance import read_jsonl

from .run_plan import load_model_conditions


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def estimate_pilot_cost(
    plan_path: str | Path = "runs/pilot_run_plan_v1.jsonl",
    conditions_path: str | Path = "protocol/MODEL_CONDITIONS_DRAFT_V1.yaml",
) -> dict[str, Any]:
    plan = list(read_jsonl(plan_path))
    conditions = load_model_conditions(conditions_path)
    episodes_by_condition = {
        condition.condition_id: sum(1 for row in plan if row["model_condition_id"] == condition.condition_id)
        for condition in conditions
    }
    per_condition: dict[str, Any] = {}
    for condition in conditions:
        prompt_tokens = episodes_by_condition[condition.condition_id] * 2 * 2000
        max_output = condition.max_output_tokens or 2500
        output_tokens = episodes_by_condition[condition.condition_id] * 2 * max_output
        per_condition[condition.condition_id] = {
            "prompt_tokens_estimate": prompt_tokens,
            "output_tokens_upper_bound": output_tokens,
            "pricing_snapshot_source": condition.pricing_snapshot_source,
            "pricing_retrieval_utc": condition.pricing_retrieval_utc,
            "cost_estimate_status": "unknown_pricing_not_guessed",
            "normal_primary_calls": episodes_by_condition[condition.condition_id] * 2,
            "max_transport_attempt_calls": episodes_by_condition[condition.condition_id] * 2 * 3,
        }
    return {
        "status": "pass",
        "episodes": len(plan),
        "planned_primary_calls": sum(int(row["planned_primary_calls"]) for row in plan),
        "max_retry_boundary_calls": sum(int(row["planned_primary_calls"]) for row in plan) * 3,
        "per_condition": per_condition,
        "llm_calls": 0,
    }
