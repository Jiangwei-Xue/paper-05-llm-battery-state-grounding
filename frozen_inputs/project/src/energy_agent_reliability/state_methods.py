"""State-management methods and fail-closed state-boundary checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

StateMethod = Literal["rolling_summary", "visible_carry", "typed_state", "canonical_typed_carry"]

STATE_METHODS: tuple[StateMethod, ...] = (
    "rolling_summary",
    "visible_carry",
    "typed_state",
    "canonical_typed_carry",
)

OUTPUT_BEARING_FIELDS = {
    "current_time",
    "soc_kwh",
    "usable_capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "reserve_soc_kwh",
    "export_limit_kw",
    "active_forecast_version",
    "active_commitments",
}

ANNOTATION_FIELDS = {"notes", "confidence", "rationale"}
EXCLUSION_FIELDS = {"revoked_or_superseded_items"}

FORBIDDEN_SURFACE_TOKENS = {
    "scorer_oracle",
    "future_actual_pv",
    "realized_future",
    "perfect_information",
    "mpc_future_action",
    "correct_answer",
    "hidden_event_parameter",
    "pre_event_capacity_kwh",
    "pre_event_max_charge_kw",
    "pre_event_max_discharge_kw",
    "pre_event_export_limit_kw",
    "pre_event_reserve_soc_kwh",
}


@dataclass(frozen=True)
class StateValidation:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str]


def method_surface_definitions() -> dict[str, dict[str, Any]]:
    """Return the frozen method axis without model-specific material."""
    return {
        "rolling_summary": {
            "state_owner": "model",
            "state_format": "natural_language_summary",
            "stage2_carry_source": "model_generated_stage1_summary",
            "system_correction": False,
        },
        "visible_carry": {
            "state_owner": "deterministic_environment",
            "state_format": "natural_language_visible_history",
            "stage2_carry_source": "all_prior_model_visible_information_only",
            "system_correction": "deterministic_visible_information_only",
        },
        "typed_state": {
            "state_owner": "model",
            "state_format": "json_typed_state_v1",
            "stage2_carry_source": "model_generated_stage1_json",
            "system_correction": False,
        },
        "canonical_typed_carry": {
            "state_owner": "deterministic_environment",
            "state_format": "json_typed_state_v1",
            "stage2_carry_source": "deterministic_current_observable_state",
            "system_correction": "canonical_state_maintenance",
        },
    }


def typed_state_schema() -> dict[str, Any]:
    return {
        "schema_version": "typed_state_v1",
        "output_bearing_fields": sorted(OUTPUT_BEARING_FIELDS),
        "exclusion_fields": sorted(EXCLUSION_FIELDS),
        "annotation_fields": sorted(ANNOTATION_FIELDS),
        "required_fields": sorted(OUTPUT_BEARING_FIELDS | EXCLUSION_FIELDS),
        "forbidden_tokens": sorted(FORBIDDEN_SURFACE_TOKENS),
    }


def deterministic_visible_carry(visible_history: dict[str, Any]) -> str:
    """Summarize only information already visible to the model."""
    visible_text = json.dumps(_strip_forbidden(visible_history), sort_keys=True, ensure_ascii=True)
    return f"Visible carry generated deterministically from prior visible inputs: {visible_text}"


def validate_explicit_state(
    state: dict[str, Any] | str | None,
    expected_state: dict[str, Any],
    *,
    method: StateMethod,
    tolerance: float = 1e-3,
) -> StateValidation:
    """Check Stage 2 explicit state for governance success without using scorer-only data."""
    checks: dict[str, bool] = {
        "non_empty": state not in (None, "", {}, []),
        "no_forbidden_tokens": not _contains_forbidden(state),
        "state_method_known": method in STATE_METHODS,
    }
    if not checks["non_empty"] or not checks["no_forbidden_tokens"] or not checks["state_method_known"]:
        return _validation(checks)
    if method in {"rolling_summary", "visible_carry"}:
        checks["natural_language_state"] = isinstance(state, str)
        return _validation(checks)
    checks["json_object"] = isinstance(state, dict)
    if not isinstance(state, dict):
        return _validation(checks)
    checks["required_fields_present"] = (OUTPUT_BEARING_FIELDS | EXCLUSION_FIELDS).issubset(state)
    checks["no_exclusion_as_active_constraint"] = _no_exclusion_as_active_constraint(state)
    for field in (
        "soc_kwh",
        "usable_capacity_kwh",
        "max_charge_kw",
        "max_discharge_kw",
        "reserve_soc_kwh",
        "export_limit_kw",
    ):
        checks[f"{field}_correct"] = _numeric_close(state.get(field), expected_state.get(field), tolerance)
    checks["current_time_correct"] = state.get("current_time") == expected_state.get("current_time")
    checks["forecast_version_correct"] = state.get("active_forecast_version") == expected_state.get(
        "active_forecast_version"
    )
    checks["active_commitments_correct"] = state.get("active_commitments") == expected_state.get(
        "active_commitments"
    )
    checks["revoked_items_correct"] = state.get("revoked_or_superseded_items") == expected_state.get(
        "revoked_or_superseded_items"
    )
    return _validation(checks)


def stale_state_negative_control(expected_state: dict[str, Any], stale_state: dict[str, Any]) -> bool:
    """Return true only when the validator rejects the stale state."""
    return not validate_explicit_state(stale_state, expected_state, method="typed_state").passed


def empty_state_rejected(expected_state: dict[str, Any]) -> bool:
    return not validate_explicit_state({}, expected_state, method="typed_state").passed


def carry_everything_rejected(expected_state: dict[str, Any]) -> bool:
    polluted = {
        **expected_state,
        "scorer_oracle": {"perfect_information": "not visible"},
        "future_actual_pv": [1.0, 2.0],
    }
    return not validate_explicit_state(polluted, expected_state, method="typed_state").passed


def _validation(checks: dict[str, bool]) -> StateValidation:
    reasons = [name for name, passed in checks.items() if not passed]
    return StateValidation(passed=not reasons, checks=checks, reasons=reasons)


def _numeric_close(left: Any, right: Any, tolerance: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _contains_forbidden(state: dict[str, Any] | str | None) -> bool:
    payload = json.dumps(state, sort_keys=True, ensure_ascii=True) if not isinstance(state, str) else state
    lowered = payload.lower()
    return any(token in lowered for token in FORBIDDEN_SURFACE_TOKENS)


def _strip_forbidden(value: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: item
        for key, item in value.items()
        if not any(token in key.lower() for token in FORBIDDEN_SURFACE_TOKENS)
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    if any(token in encoded.lower() for token in FORBIDDEN_SURFACE_TOKENS):
        return {}
    return payload


def _no_exclusion_as_active_constraint(state: dict[str, Any]) -> bool:
    revoked = state.get("revoked_or_superseded_items")
    commitments = state.get("active_commitments")
    if not isinstance(revoked, list) or not isinstance(commitments, list):
        return False
    revoked_pairs = {(item.get("field"), item.get("previous_value")) for item in revoked if isinstance(item, dict)}
    active_pairs = {(item.get("field"), item.get("value")) for item in commitments if isinstance(item, dict)}
    return revoked_pairs.isdisjoint(active_pairs)
