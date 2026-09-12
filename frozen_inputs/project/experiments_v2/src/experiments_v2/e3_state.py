"""E3 v2.2 state blocks, parsing, and fail-closed recursion rules.

The provider runner is intentionally out of this module.  These functions define
the deterministic contract used by the provider-facing runner and by offline tests.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

PLANT_FIELDS = (
    "soc_kwh",
    "usable_capacity_kwh",
    "charge_limit_kw",
    "discharge_limit_kw",
    "export_limit_kw",
)
PLANNING_FIELDS = (
    "forecast_version",
    "forecast_bundle_id",
    "reserve_schedule_kwh",
    "terminal_soc_target_kwh",
    "active_commitments",
    "revoked_commitments",
    "commitment_lineage",
    "operational_sequence_id",
)
BLOCKS = ("plant_state", "planning_context")


@dataclass(frozen=True)
class ParsedE3Response:
    json_valid: bool
    plant_state: dict[str, Any] | None
    planning_context: dict[str, Any] | None
    actions_kw: list[float] | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class RecursiveUpdate:
    plant_state: dict[str, Any]
    planning_context: dict[str, Any]
    plant_block_updated: bool
    planning_block_updated: bool
    state_carried_forward_after_failure: bool


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_plant_block(block: Any, *, tolerance: float = 1e-6) -> tuple[bool, list[str]]:
    """Validate the physical state atomically; no field-level patching is allowed."""
    failures: list[str] = []
    if not isinstance(block, dict):
        return False, ["plant_state_not_object"]
    missing = sorted(set(PLANT_FIELDS).difference(block))
    if missing:
        failures.append("plant_state_missing:" + ",".join(missing))
        return False, failures
    for field in PLANT_FIELDS:
        if not _finite_number(block[field]):
            failures.append(f"plant_state_nonfinite:{field}")
    if failures:
        return False, failures
    soc = float(block["soc_kwh"])
    capacity = float(block["usable_capacity_kwh"])
    charge = float(block["charge_limit_kw"])
    discharge = float(block["discharge_limit_kw"])
    export = float(block["export_limit_kw"])
    if capacity <= 0:
        failures.append("plant_state_capacity_nonpositive")
    if min(charge, discharge, export) < 0:
        failures.append("plant_state_negative_limit")
    if soc < -tolerance or soc > capacity + tolerance:
        failures.append("plant_state_soc_outside_capacity")
    return not failures, failures


def validate_planning_context(
    block: Any,
    *,
    bundle_catalog: dict[str, dict[str, Any]],
    expected_sequence_min: int,
) -> tuple[bool, list[str]]:
    """Validate the planning-context record as one atomic record."""
    failures: list[str] = []
    if not isinstance(block, dict):
        return False, ["planning_context_not_object"]
    missing = sorted(set(PLANNING_FIELDS).difference(block))
    if missing:
        failures.append("planning_context_missing:" + ",".join(missing))
        return False, failures
    version = block.get("forecast_version")
    bundle_id = block.get("forecast_bundle_id")
    if not isinstance(version, str) or not version:
        failures.append("planning_context_invalid_forecast_version")
    if not isinstance(bundle_id, str) or bundle_id not in bundle_catalog:
        failures.append("planning_context_unknown_bundle")
    elif bundle_catalog[bundle_id].get("forecast_version") != version:
        failures.append("planning_context_version_bundle_mismatch")
    reserve = block.get("reserve_schedule_kwh")
    if not isinstance(reserve, list) or not reserve or not all(_finite_number(item) for item in reserve):
        failures.append("planning_context_invalid_reserve_schedule")
    elif any(float(item) < 0 for item in reserve):
        failures.append("planning_context_negative_reserve")
    target = block.get("terminal_soc_target_kwh")
    target_value = (
        float(target)
        if isinstance(target, (int, float)) and not isinstance(target, bool) and math.isfinite(float(target))
        else None
    )
    if target_value is None or target_value < 0:
        failures.append("planning_context_invalid_terminal_target")
    for field in ("active_commitments", "revoked_commitments", "commitment_lineage"):
        if not isinstance(block.get(field), list):
            failures.append(f"planning_context_invalid:{field}")
    sequence = block.get("operational_sequence_id")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < expected_sequence_min:
        failures.append("planning_context_invalid_sequence")
    active_ids = {item.get("id") for item in block.get("active_commitments", []) if isinstance(item, dict)}
    revoked_ids = {item.get("id") for item in block.get("revoked_commitments", []) if isinstance(item, dict)}
    if active_ids.intersection(revoked_ids):
        failures.append("planning_context_active_revoked_overlap")
    return not failures, failures


def validate_actions(actions: Any, *, expected_length: int) -> tuple[bool, list[float] | None, list[str]]:
    failures: list[str] = []
    if not isinstance(actions, list) or len(actions) != expected_length:
        return False, None, ["action_length"]
    values: list[float] = []
    for value in actions:
        if not _finite_number(value) or float(value) < -250.0 or float(value) > 250.0:
            failures.append("action_value")
            continue
        values.append(float(value))
    return not failures, values if not failures else None, failures


def parse_e3_response(
    raw_text: str,
    *,
    expected_action_length: int,
    bundle_catalog: dict[str, dict[str, Any]],
    expected_sequence_min: int,
) -> ParsedE3Response:
    diagnostics: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return ParsedE3Response(False, None, None, None, ("invalid_json",))
    if not isinstance(parsed, dict):
        return ParsedE3Response(True, None, None, None, ("top_level_not_object",))
    plant_ok, plant_failures = validate_plant_block(parsed.get("plant_state"))
    context_ok, context_failures = validate_planning_context(
        parsed.get("planning_context"),
        bundle_catalog=bundle_catalog,
        expected_sequence_min=expected_sequence_min,
    )
    action_ok, actions, action_failures = validate_actions(
        parsed.get("actions_kw"), expected_length=expected_action_length
    )
    diagnostics.extend(plant_failures)
    diagnostics.extend(context_failures)
    diagnostics.extend(action_failures)
    return ParsedE3Response(
        True,
        deepcopy(parsed["plant_state"]) if plant_ok else None,
        deepcopy(parsed["planning_context"]) if context_ok else None,
        actions,
        tuple(diagnostics),
    )


def recursive_update(
    previous_plant: dict[str, Any],
    previous_context: dict[str, Any],
    parsed: ParsedE3Response,
    *,
    plant_owned_by_system: bool,
    context_owned_by_system: bool,
    canonical_plant: dict[str, Any],
    canonical_context: dict[str, Any],
) -> RecursiveUpdate:
    """Apply blocks atomically and preserve the pre-call input on failure."""
    next_plant = deepcopy(canonical_plant if plant_owned_by_system else previous_plant)
    next_context = deepcopy(canonical_context if context_owned_by_system else previous_context)
    plant_updated = False
    context_updated = False
    if not plant_owned_by_system and parsed.plant_state is not None:
        next_plant = deepcopy(parsed.plant_state)
        plant_updated = True
    if not context_owned_by_system and parsed.planning_context is not None:
        next_context = deepcopy(parsed.planning_context)
        context_updated = True
    carried = (not plant_owned_by_system and parsed.plant_state is None) or (
        not context_owned_by_system and parsed.planning_context is None
    )
    return RecursiveUpdate(next_plant, next_context, plant_updated, context_updated, carried)


def canonical_block_pair(
    *,
    plant_state: dict[str, Any],
    planning_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copies used by system-owned conditions; never mutate model output."""
    return deepcopy(plant_state), deepcopy(planning_context)
