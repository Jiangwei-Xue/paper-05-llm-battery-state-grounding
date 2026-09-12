"""Model-independent candidate admission checks conducted before any model run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import perfect_information_baseline
from .battery import simulate_dispatch
from .config import FrozenProtocol
from .episode import canonical_typed_state, episode_split
from .events import dynamic_override_schedule
from .provenance import write_jsonl
from .scenarios import scenario_frame
from .state_methods import (
    carry_everything_rejected,
    empty_state_rejected,
    stale_state_negative_control,
    validate_explicit_state,
)


def admit_candidates(
    protocol: FrozenProtocol,
    candidates: list[dict[str, Any]],
    oracle_root: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run candidate-level fail-closed checks without accessing any model."""
    results = [admit_one(protocol, candidate, Path(oracle_root)) for candidate in candidates]
    destination = Path(output_dir) if output_dir is not None else Path(oracle_root)
    write_jsonl(destination / "admission_results.jsonl", results)
    return results


def admit_one(protocol: FrozenProtocol, candidate: dict[str, Any], oracle_root: Path) -> dict[str, Any]:
    frame = scenario_frame(candidate)
    oracle = json.loads((oracle_root / candidate["oracle_ref"]).read_text(encoding="utf-8"))
    schedule = dynamic_override_schedule(candidate["visible_update"], len(frame))
    no_action = simulate_dispatch(frame, np.zeros(len(frame)), protocol.battery, overrides_by_step=schedule)
    nominal_plan_error: str | None = None
    nominal_plan = None
    event_plan = None
    baseline_error: str | None = None
    baseline_cost: float | None = None
    baseline_feasible = False
    try:
        nominal_plan = perfect_information_baseline(frame, protocol.battery)
        event_plan = perfect_information_baseline(frame, protocol.battery, overrides_by_step=schedule)
        baseline_feasible = event_plan.simulation.feasible
        baseline_cost = event_plan.simulation.total_cost
    except (RuntimeError, ValueError) as exc:
        baseline_error = str(exc)
    try:
        if nominal_plan is None:
            nominal_plan = perfect_information_baseline(frame, protocol.battery)
    except (RuntimeError, ValueError) as exc:
        nominal_plan_error = str(exc)
    expected_state = _expected_event_state(protocol, candidate, frame)
    stale_state = _stale_state(protocol, expected_state)
    governance = validate_explicit_state(expected_state, expected_state, method="canonical_typed_carry")
    checks = {
        "data_completeness": _data_complete(frame),
        "timestamp_alignment": _timestamps_aligned(frame, int(candidate["horizon_hours"])),
        "simulator_validity": no_action.feasible and _valid_trace(no_action.trace),
        "baseline_feasibility": baseline_feasible,
        "post_event_feasibility": baseline_feasible and event_plan is not None,
        "event_materiality": _event_material(candidate, protocol, nominal_plan, event_plan),
        "energy_conservation": _energy_conserved(no_action.trace, protocol),
        "oracle_consistency": _oracle_consistent(candidate, oracle),
        "scorer_ceiling": baseline_cost is not None and baseline_cost <= no_action.total_cost + 1e-5,
        "stale_state_negative_control_rejection": stale_state_negative_control(expected_state, stale_state),
        "empty_state_rejection": empty_state_rejected(expected_state),
        "carry_everything_rejection": carry_everything_rejected(expected_state),
        "oracle_separation": _no_visible_oracle_leakage(candidate, oracle),
        "no_answer_leakage": _no_answer_leakage(candidate),
        "no_future_data_leakage": _no_future_data_leakage(candidate),
        "state_governance_oracle_free": governance.passed,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "scenario_id": candidate["scenario_id"],
        "checks": checks,
        "admitted": not failed,
        "all_passed": not failed,
        "failed_checks": failed,
        "failure_reasons": {name: f"{name} failed" for name in failed},
        "perfect_information_cost": baseline_cost,
        "no_action_cost": no_action.total_cost,
        "baseline_error": baseline_error,
        "nominal_plan_error": nominal_plan_error,
        "admission_protocol_version": protocol.protocol_version,
    }


def _data_complete(frame: pd.DataFrame) -> bool:
    pv_column = "pv_ac_kw" if "pv_ac_kw" in frame.columns else "pv_kw"
    price_column = "day_ahead_price_usd_mwh" if "day_ahead_price_usd_mwh" in frame.columns else "price_per_mwh"
    required = ["load_kw", pv_column, price_column]
    return bool(set(required).issubset(frame.columns) and np.isfinite(frame[required].to_numpy(dtype=float)).all())


def _timestamps_aligned(frame: pd.DataFrame, horizon: int) -> bool:
    expected_steps = horizon * 4
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) != expected_steps or not frame.index.is_monotonic_increasing:
        return False
    return bool((frame.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all())


def _valid_trace(trace: pd.DataFrame) -> bool:
    return bool(
        np.isfinite(trace.to_numpy(dtype=float)).all()
        and (trace["soc_kwh"] >= trace["reserve_soc_kwh"] - 1e-8).all()
        and (trace["soc_kwh"] <= trace["available_capacity_kwh"] + 1e-8).all()
        and (trace[["import_kw", "export_kw", "curtailed_kw"]] >= -1e-8).all().all()
    )


def _event_relevant(update: dict[str, Any], protocol: FrozenProtocol) -> bool:
    family = update["event_family"]
    if family == "forecast_revision":
        return (
            update.get("forecast_update_method") == "latest_observation_clearsky_index_persistence"
            and float(update["forecast_revision_blend_weight"]) >= 0.05
        )
    if family == "battery_capacity_derating":
        return float(update["usable_capacity_kwh"]) < protocol.battery.energy_capacity_kwh
    if family == "power_limit_derating":
        return float(update["max_charge_kw"]) < protocol.battery.max_charge_kw and float(update["max_discharge_kw"]) < protocol.battery.max_discharge_kw
    if family == "export_limit_update":
        return float(update["export_limit_kw"]) < protocol.battery.export_limit_kw
    if family == "reserve_commitment_update":
        return float(update["reserve_soc_kwh"]) > protocol.battery.reserve_soc_kwh
    return False


def _event_material(
    candidate: dict[str, Any],
    protocol: FrozenProtocol,
    nominal_plan: Any,
    event_plan: Any,
) -> bool:
    if not _event_relevant(candidate["visible_update"], protocol):
        return False
    if candidate["event_family"] == "forecast_revision":
        return True
    if nominal_plan is None or event_plan is None:
        return False
    action_delta = float(np.max(np.abs(nominal_plan.action_kw - event_plan.action_kw)))
    cost_delta = abs(float(nominal_plan.simulation.total_cost - event_plan.simulation.total_cost))
    return bool(action_delta > 1e-6 or cost_delta > 1e-6)


def _expected_event_state(
    protocol: FrozenProtocol, candidate: dict[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    split = episode_split(len(frame), int(candidate["visible_update"]["activation_step"]))
    current_time = frame.index[split.activation_step]
    return canonical_typed_state(
        config=protocol.battery,
        current_time=current_time,
        visible_update=candidate["visible_update"],
        prefix=None,
    )


def _stale_state(protocol: FrozenProtocol, expected_state: dict[str, Any]) -> dict[str, Any]:
    stale = dict(expected_state)
    stale.update(
        {
            "usable_capacity_kwh": protocol.battery.energy_capacity_kwh,
            "max_charge_kw": protocol.battery.max_charge_kw,
            "max_discharge_kw": protocol.battery.max_discharge_kw,
            "reserve_soc_kwh": protocol.battery.reserve_soc_kwh,
            "export_limit_kw": protocol.battery.export_limit_kw,
            "active_commitments": [],
            "revoked_or_superseded_items": [],
        }
    )
    if stale == expected_state:
        stale["active_forecast_version"] = "stale_forecast_v0"
    return stale


def _energy_conserved(trace: pd.DataFrame, protocol: FrozenProtocol, tolerance: float = 1e-6) -> bool:
    previous = protocol.battery.initial_soc_kwh
    charges = trace["charge_kw"].to_numpy(dtype=float)
    discharges = trace["discharge_kw"].to_numpy(dtype=float)
    socs = trace["soc_kwh"].to_numpy(dtype=float)
    for charge_kw, discharge_kw, soc_kwh in zip(charges, discharges, socs, strict=True):
        expected = previous + charge_kw * protocol.battery.charge_efficiency * 0.25
        expected -= discharge_kw / protocol.battery.discharge_efficiency * 0.25
        if abs(expected - soc_kwh) > tolerance:
            return False
        previous = soc_kwh
    return True


def _oracle_consistent(candidate: dict[str, Any], oracle: dict[str, Any]) -> bool:
    hidden = oracle.get("scorer_oracle")
    if oracle.get("scenario_id") != candidate["scenario_id"] or not isinstance(hidden, dict):
        return False
    if hidden.get("oracle_version") != 1:
        return False
    numeric_values = [value for key, value in hidden.items() if key != "oracle_version"]
    return bool(numeric_values and all(isinstance(value, (int, float)) and np.isfinite(value) for value in numeric_values))


def _no_visible_oracle_leakage(candidate: dict[str, Any], oracle: dict[str, Any]) -> bool:
    public_payload = json.dumps(_scrub_policy_fields(candidate), sort_keys=True)
    banned = {
        "scorer_oracle",
        "realized_future",
        "pre_event_capacity_kwh",
        "pre_event_max_charge_kw",
        "pre_event_max_discharge_kw",
        "pre_event_export_limit_kw",
        "pre_event_reserve_soc_kwh",
    }
    if any(token in public_payload for token in banned):
        return False
    for value in oracle["scorer_oracle"].values():
        if isinstance(value, str) and value in public_payload:
            return False
    return True


def _no_answer_leakage(candidate: dict[str, Any]) -> bool:
    payload = json.dumps(_scrub_policy_fields(candidate), sort_keys=True).lower()
    banned = ["perfect_information", "optimal_action", "correct_answer", "model_score"]
    return not any(token in payload for token in banned)


def _no_future_data_leakage(candidate: dict[str, Any]) -> bool:
    payload = json.dumps(_scrub_policy_fields(candidate), sort_keys=True).lower()
    banned = ["future_actual_pv", "realized_future", "scorer_oracle", "mpc_future_action"]
    return not any(token in payload for token in banned)


def _scrub_policy_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_policy_fields(item)
            for key, item in value.items()
            if key
            not in {
                "oracle_ref",
                "forbidden_information",
                "forbidden_tokens",
                "required_outputs",
                "required_dispatch_plan_steps",
            }
        }
    if isinstance(value, list):
        return [_scrub_policy_fields(item) for item in value]
    return value
