"""Prospective controls for the excluded paired-authority correction pilot."""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Literal

from .models import PromptBundle, StateMethodName
from .protocol_v2 import ProviderSpecV2
from .request import canonical_json, oracle_leak_scan, sha256_text, stable_id
from .request_v2 import compile_stage1_prompt_v2, compile_stage2_prompt_v2

PAIRED_AUTHORITY_PROTOCOL_ID = "energybench_paired_authority_pilot_v1"
PAIRED_AUTHORITY_RUN_ORDER_SEED = 20260719
REPRESENTATIONS = ("natural_language", "typed_json")
OWNERS = ("model", "deterministic_system")
STATE_FIELDS = (
    "current_time",
    "soc_kwh",
    "usable_capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "reserve_soc_kwh",
    "export_limit_kw",
    "active_forecast_version",
    "active_commitments",
    "revoked_or_superseded_items",
)

Representation = Literal["natural_language", "typed_json"]
Owner = Literal["model", "deterministic_system"]


def build_paired_plan(
    tasks: list[dict[str, Any]],
    specs: tuple[ProviderSpecV2, ...],
    *,
    run_order_seed: int = PAIRED_AUTHORITY_RUN_ORDER_SEED,
    repetitions: int = 1,
    protocol_id: str = PAIRED_AUTHORITY_PROTOCOL_ID,
    run_id_prefix: str = "papv1",
    excluded_from_formal_analysis: bool = True,
) -> list[dict[str, Any]]:
    """Build stable paired rows across tasks, providers, representations, and repeats."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        task_sha256 = sha256_text(canonical_json(task))
        for spec in specs:
            for representation in REPRESENTATIONS:
                for repetition in range(repetitions):
                    identity = {
                        "protocol_id": protocol_id,
                        "scenario_id": task["scenario_id"],
                        "condition_id": spec.condition_id,
                        "representation": representation,
                        "repetition": repetition,
                    }
                    row = {
                        **identity,
                        "paired_run_id": stable_id(run_id_prefix, identity),
                        "task_sha256": task_sha256,
                        "planned_provider_calls": 3,
                        "planned_branch_outcomes": 2,
                        "excluded_from_formal_analysis": (
                            excluded_from_formal_analysis
                        ),
                    }
                    if "corrective_evidence_tier" in task:
                        row["corrective_evidence_tier"] = task[
                            "corrective_evidence_tier"
                        ]
                    rows.append(row)
    rng = random.Random(run_order_seed)
    rng.shuffle(rows)
    for run_order, row in enumerate(rows):
        branch_order = list(OWNERS)
        rng.shuffle(branch_order)
        row["run_order"] = run_order
        row["stage2_branch_order"] = branch_order
    return rows


def representation_method(representation: Representation) -> StateMethodName:
    """Return the parser/output schema used by both authority branches."""
    if representation == "natural_language":
        return "rolling_summary"
    if representation == "typed_json":
        return "typed_state"
    raise ValueError(f"Unknown paired representation: {representation}")


def scoring_method(representation: Representation, owner: Owner) -> StateMethodName:
    """Map a paired branch to the existing common simulator/state validator."""
    if representation == "natural_language":
        return "rolling_summary" if owner == "model" else "visible_carry"
    if representation == "typed_json":
        return "typed_state" if owner == "model" else "canonical_typed_carry"
    raise ValueError(f"Unknown paired representation: {representation}")


def compile_shared_stage1_prompt(
    candidate: dict[str, Any], representation: Representation
) -> PromptBundle:
    """Compile one Stage-1 prompt shared byte-for-byte by both owner branches."""
    method = representation_method(representation)
    base = compile_stage1_prompt_v2(candidate, method)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "paired_authority_stage1_v1"
    prompt["state_method"] = f"content_matched_{representation}"
    prompt["paired_authority_contract"] = {
        "stage1_shared_across_owner_branches": True,
        "owner_not_assigned_until_stage2": True,
        "required_operational_fields": list(STATE_FIELDS),
        "cross_representation_claim_status": "excluded_pilot_measurement_check_only",
    }
    prompt["state_output_contract"] = _state_output_contract(representation)
    return _bundle(base, prompt)


def compile_branched_stage2_prompt(
    candidate: dict[str, Any],
    representation: Representation,
    owner: Owner,
    carrier_value: dict[str, Any] | str,
) -> PromptBundle:
    """Compile a Stage-2 prompt whose only owner-specific surface is state_carrier."""
    method = representation_method(representation)
    base = compile_stage2_prompt_v2(candidate, method, carrier_value)
    prompt = copy.deepcopy(base.canonical_prompt)
    prompt["prompt_schema"] = "paired_authority_stage2_v1"
    prompt["state_method"] = f"content_matched_{representation}"
    prompt["paired_authority_contract"] = {
        "shared_stage1_required": True,
        "stage2_difference_restricted_to_state_carrier": True,
        "required_operational_fields": list(STATE_FIELDS),
        "cross_representation_claim_status": "excluded_pilot_measurement_check_only",
    }
    prompt["state_output_contract"] = _state_output_contract(representation)
    prompt["task_surface"]["state_carrier"] = {
        "read_only": True,
        "source": owner,
        "representation": representation,
        "value": carrier_value,
    }
    return _bundle(base, prompt)


def stage2_noncarrier_sha256(bundle: PromptBundle) -> str:
    """Hash a Stage-2 prompt after replacing the complete state-carrier surface."""
    prompt = copy.deepcopy(bundle.canonical_prompt)
    prompt["task_surface"]["state_carrier"] = "<PAIRED_STATE_CARRIER>"
    return sha256_text(canonical_json(prompt))


def render_canonical_natural_state(state: dict[str, Any]) -> str:
    """Render the ten typed fields as one controlled-text record in identical order."""
    missing = [field for field in STATE_FIELDS if field not in state]
    if missing:
        raise ValueError(f"Canonical state missing fields: {missing}")
    lines = []
    for field in STATE_FIELDS:
        value = state[field]
        encoded = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if isinstance(value, list | dict)
            else str(value)
        )
        lines.append(f"{field}={encoded}")
    rendered = "; ".join(lines)
    if len(rendered) > 1200:
        raise ValueError("Controlled natural state exceeds the frozen 1200-character limit.")
    return rendered


def natural_state_content_audit(value: Any) -> dict[str, Any]:
    """Check field presence and order without silently repairing model text."""
    if not isinstance(value, str):
        return {
            "passed": False,
            "required_fields": list(STATE_FIELDS),
            "observed_fields": [],
            "reason": "not_string",
        }
    observed = []
    for item in value.split(";"):
        if "=" in item:
            observed.append(item.split("=", 1)[0].strip())
    return {
        "passed": observed == list(STATE_FIELDS),
        "required_fields": list(STATE_FIELDS),
        "observed_fields": observed,
        "reason": None if observed == list(STATE_FIELDS) else "field_set_or_order_mismatch",
    }


def _state_output_contract(representation: Representation) -> dict[str, Any]:
    if representation == "typed_json":
        return {
            "output_field": "typed_state",
            "representation": "typed_json",
            "schema": "typed_state_v2",
            "required_field_order": list(STATE_FIELDS),
            "free_text_extensions": False,
        }
    return {
        "output_field": "state_summary",
        "representation": "controlled_natural_language",
        "required_field_order": list(STATE_FIELDS),
        "record_contract": "<field>=<value>; <field>=<value>; ...",
        "exactly_one_record_line": True,
        "maximum_utf8_characters": 1200,
        "maximum_lines": 1,
        "free_text_extensions": False,
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
