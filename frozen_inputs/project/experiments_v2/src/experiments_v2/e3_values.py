"""Provider-free E3 value definitions and accounting identities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def one_segment_intervention_value(
    *,
    immediate_cost: Callable[[Sequence[float]], float],
    next_state: Callable[[Sequence[float]], dict[str, Any]],
    canonical_continuation: Callable[[dict[str, Any]], float],
    zero_action: Sequence[float],
    candidate_action: Sequence[float],
) -> dict[str, Any]:
    """Compare two first-segment actions with one common canonical continuation.

    The returned value is local to the event.  It must not be summed over events,
    because every continuation includes the overlapping suffix to the terminal time.
    """
    zero_state = next_state(zero_action)
    candidate_state = next_state(candidate_action)
    zero_q = float(immediate_cost(zero_action)) + float(canonical_continuation(zero_state))
    candidate_q = float(immediate_cost(candidate_action)) + float(canonical_continuation(candidate_state))
    return {
        "q_zero_usd": zero_q,
        "q_candidate_usd": candidate_q,
        "one_segment_intervention_value_usd": zero_q - candidate_q,
    }


def additive_accounting_decomposition(
    *,
    q_llm: float,
    q_carrier_mpc: float,
    q_canonical_mpc: float,
) -> dict[str, Any]:
    """Return a closed accounting identity under one shared Q functional."""
    carrier_contrast = q_carrier_mpc - q_canonical_mpc
    planner_contrast = q_llm - q_carrier_mpc
    total_contrast = q_llm - q_canonical_mpc
    if abs(total_contrast - (carrier_contrast + planner_contrast)) > 1e-8:
        raise AssertionError("Q-function accounting identity failed")
    return {
        "q_llm_usd": q_llm,
        "q_carrier_mpc_usd": q_carrier_mpc,
        "q_canonical_mpc_usd": q_canonical_mpc,
        "carrier_information_contrast_usd": carrier_contrast,
        "planner_interface_contrast_usd": planner_contrast,
        "total_contrast_usd": total_contrast,
        "decomposition_type": "additive_accounting",
    }


def summarize_local_values(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize event-local values without creating a cumulative pseudo-endpoint."""
    values = [float(row["one_segment_intervention_value_usd"]) for row in rows]
    return {
        "event_rows": len(rows),
        "finite_values": all(value == value and abs(value) != float("inf") for value in values),
        "mean_usd": sum(values) / len(values) if values else None,
        "min_usd": min(values) if values else None,
        "max_usd": max(values) if values else None,
        "cumulative_sum_forbidden": True,
    }


def fallback_workload_summary(rows: Sequence[dict[str, Any]], *, planned_replans: int) -> dict[str, Any]:
    """Report ITT fallback counts and conditional gate distances separately."""
    distances = [
        float(row["gate_distance_kwh"])
        for row in rows
        if row.get("gate_distance_kwh") is not None and row.get("raw_action_computable") is True
    ]
    fallback = sum(bool(row.get("fallback_used")) for row in rows)
    return {
        "planned_replans": planned_replans,
        "fallback_count": fallback,
        "fallback_rate": fallback / planned_replans if planned_replans else None,
        "computable_replans": len(distances),
        "conditional_gate_distance_sum_kwh": sum(distances),
        "conditional_gate_distance_mean_kwh": sum(distances) / len(distances) if distances else None,
        "conditional_gate_distance_median_kwh": sorted(distances)[len(distances) // 2] if distances else None,
        "complete_case_not_used": True,
    }
