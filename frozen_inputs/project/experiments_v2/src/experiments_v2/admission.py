"""Provider-free scenario admission predicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdmissionThresholds:
    canonical_feasible: bool = True
    zero_action_infeasible: bool = True
    mpc_action_energy_min_kwh: float = 50.0
    event_stale_action_l1_min_kwh: float = 25.0
    zero_minus_mpc_min_usd: float = 5.0
    zero_minus_mpc_relative_min: float = 0.02
    binding_slack_fraction_max: float = 0.05


def admit_e1(metrics: dict[str, Any], thresholds: AdmissionThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if bool(metrics.get("canonical_feasible")) is not thresholds.canonical_feasible:
        failures.append("canonical_feasibility")
    if bool(metrics.get("zero_action_feasible")) is thresholds.zero_action_infeasible:
        failures.append("zero_action_not_infeasible")
    if float(metrics.get("mpc_action_energy_kwh", 0.0)) < thresholds.mpc_action_energy_min_kwh:
        failures.append("mpc_action_energy")
    if float(metrics.get("event_stale_action_l1_kwh", 0.0)) < thresholds.event_stale_action_l1_min_kwh:
        failures.append("event_stale_action_difference")
    gap = float(metrics.get("zero_minus_mpc_usd", 0.0))
    mpc = abs(float(metrics.get("mpc_cost_usd", 0.0)))
    if gap < thresholds.zero_minus_mpc_min_usd and gap < thresholds.zero_minus_mpc_relative_min * mpc:
        failures.append("economic_space")
    if metrics.get("event_family") == "forecast_revision" and not bool(metrics.get("forecast_plan_changed")):
        failures.append("forecast_plan_unchanged")
    return not failures, failures


def admit_e2(metrics: dict[str, Any], thresholds: AdmissionThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not bool(metrics.get("canonical_suffix_feasible")):
        failures.append("canonical_suffix_infeasible")
    if bool(metrics.get("zero_suffix_feasible")):
        failures.append("zero_suffix_feasible")
    if not bool(metrics.get("stale_infeasible_or_cost_worse")):
        failures.append("stale_not_consequential")
    if float(metrics.get("deterministic_action_l1_kwh", 0.0)) < thresholds.event_stale_action_l1_min_kwh:
        failures.append("deterministic_action_difference")
    if not bool(metrics.get("strong_consequence")):
        failures.append("strong_consequence_missing")
    if float(metrics.get("binding_slack_fraction", 1.0)) > thresholds.binding_slack_fraction_max:
        failures.append("constraint_not_binding")
    if metrics.get("event_family") == "forecast_revision" and not bool(metrics.get("forecast_plan_changed")):
        failures.append("forecast_plan_unchanged")
    return not failures, failures
