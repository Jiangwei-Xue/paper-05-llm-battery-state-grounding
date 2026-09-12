"""Two-stage dynamic episode surfaces and deterministic execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .battery import BatteryOverrides, SimulationResult, simulate_dispatch
from .config import BatteryConfig, FrozenProtocol
from .events import dynamic_override_schedule

ACTION_SCHEMA_VERSION = "dispatch_action_v1"


@dataclass(frozen=True)
class EpisodeSplit:
    total_steps: int
    activation_step: int
    stage1_plan_steps: int
    stage1_executed_steps: int
    stage2_plan_steps: int


@dataclass(frozen=True)
class ExecutedEpisode:
    split: EpisodeSplit
    trace: pd.DataFrame
    total_cost: float
    feasible: bool
    terminal_soc_gap_kwh: float
    state_at_event: dict[str, Any]


def episode_split(total_steps: int, activation_step: int) -> EpisodeSplit:
    """Return the frozen two-stage cut point for one episode."""
    if total_steps <= 0:
        raise ValueError("Episode must contain at least one time step.")
    if not 0 <= activation_step < total_steps:
        raise ValueError("Event activation step must fall inside the episode horizon.")
    return EpisodeSplit(
        total_steps=total_steps,
        activation_step=activation_step,
        stage1_plan_steps=total_steps,
        stage1_executed_steps=activation_step,
        stage2_plan_steps=total_steps - activation_step,
    )


def activation_step_from_candidate(candidate: dict[str, Any]) -> int:
    return int(candidate["visible_update"]["activation_step"])


def build_episode_surface(
    protocol: FrozenProtocol, candidate: dict[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    """Build the model-visible, scorer-oracle-free episode description."""
    split = episode_split(len(frame), activation_step_from_candidate(candidate))
    timestamps = [_format_timestamp(ts) for ts in frame.index]
    pv_column = "pv_ac_kw" if "pv_ac_kw" in frame.columns else "pv_kw"
    price_column = (
        "day_ahead_price_usd_mwh" if "day_ahead_price_usd_mwh" in frame.columns else "price_per_mwh"
    )
    series = [
        {
            "t": idx,
            "timestamp_utc": timestamps[idx],
            "pv_forecast_kw": _round_float(frame.iloc[idx][pv_column]),
            "load_kw": _round_float(frame.iloc[idx]["load_kw"]),
            "dam_price_usd_mwh": _round_float(frame.iloc[idx][price_column]),
        }
        for idx in range(len(frame))
    ]
    visible_update = dict(candidate["visible_update"])
    return {
        "episode_schema_version": "agent_episode_v1",
        "scenario_id": candidate["scenario_id"],
        "horizon_steps": len(frame),
        "resolution_minutes": protocol.analysis_resolution_minutes,
        "action_schema": canonical_action_schema(),
        "objective": "minimize total operating cost while satisfying physical battery, reserve, export, and terminal SOC constraints",
        "stage_1": {
            "name": "pre_event_plan",
            "visible_initial_state": initial_visible_state(protocol.battery),
            "visible_timeseries": series,
            "required_dispatch_plan_steps": split.stage1_plan_steps,
            "executed_prefix_steps_before_event": split.stage1_executed_steps,
            "required_outputs": ["carried_state", "dispatch_plan"],
        },
        "stage_2": {
            "name": "post_event_replan",
            "current_step": split.activation_step,
            "current_time_utc": timestamps[split.activation_step],
            "visible_update": visible_update,
            "remaining_timeseries": series[split.activation_step :],
            "required_dispatch_plan_steps": split.stage2_plan_steps,
            "forbidden_information": [
                "scorer_oracle",
                "future_actual_pv",
                "perfect_information_solution",
                "rolling_horizon_future_actions",
                "hidden_event_parameters",
                "model_result_feedback",
            ],
            "required_outputs": ["updated_state", "revised_dispatch_plan"],
        },
    }


def canonical_action_schema() -> dict[str, Any]:
    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "dispatch_plan": {
            "type": "array",
            "items": {
                "timestamp_utc": "RFC3339 UTC timestamp",
                "battery_action_kw": "signed float; positive charges, negative discharges",
            },
        },
    }


def initial_visible_state(config: BatteryConfig) -> dict[str, Any]:
    return {
        "soc_kwh": config.initial_soc_kwh,
        "usable_capacity_kwh": config.energy_capacity_kwh,
        "max_charge_kw": config.max_charge_kw,
        "max_discharge_kw": config.max_discharge_kw,
        "reserve_soc_kwh": config.reserve_soc_kwh,
        "export_limit_kw": config.export_limit_kw,
        "active_forecast_version": "initial_forecast_v1",
        "active_commitments": [],
        "revoked_or_superseded_items": [],
    }


def execute_two_stage_episode(
    frame: pd.DataFrame,
    stage1_full_plan_kw: list[float] | np.ndarray,
    stage2_plan_kw: list[float] | np.ndarray,
    config: BatteryConfig,
    visible_update: dict[str, Any],
) -> ExecutedEpisode:
    """Execute Stage 1 until the event and Stage 2 over the remaining horizon."""
    split = episode_split(len(frame), int(visible_update["activation_step"]))
    stage1_actions = np.asarray(stage1_full_plan_kw, dtype=float)
    stage2_actions = np.asarray(stage2_plan_kw, dtype=float)
    if len(stage1_actions) != split.stage1_plan_steps:
        raise ValueError("Stage 1 dispatch plan length must equal the full horizon.")
    if len(stage2_actions) != split.stage2_plan_steps:
        raise ValueError("Stage 2 dispatch plan length must equal the remaining horizon.")
    schedule = dynamic_override_schedule(visible_update, len(frame))
    step_hours = _episode_step_hours(frame.index)
    prefix = _execute_prefix(frame, stage1_actions, config, split, step_hours)
    state_at_event = canonical_typed_state(
        config=config,
        current_time=frame.index[split.activation_step],
        visible_update=visible_update,
        prefix=prefix,
    )
    suffix = simulate_dispatch(
        frame.iloc[split.activation_step :],
        stage2_actions,
        config,
        overrides_by_step=schedule[split.activation_step :],
        initial_soc_kwh=float(state_at_event["soc_kwh"]),
        require_terminal_soc=True,
        step_hours=step_hours,
    )
    trace = pd.concat([prefix.trace, suffix.trace])
    total_cost = float(prefix.total_cost + suffix.total_cost)
    feasible = bool(prefix.feasible and suffix.feasible)
    return ExecutedEpisode(
        split=split,
        trace=trace,
        total_cost=total_cost,
        feasible=feasible,
        terminal_soc_gap_kwh=suffix.terminal_soc_gap_kwh,
        state_at_event=state_at_event,
    )


def canonical_typed_state(
    *,
    config: BatteryConfig,
    current_time: pd.Timestamp,
    visible_update: dict[str, Any],
    prefix: SimulationResult | None = None,
) -> dict[str, Any]:
    """Create the deterministic canonical typed carry allowed at Stage 2."""
    prior_state = initial_visible_state(config)
    soc = config.initial_soc_kwh
    if prefix is not None and len(prefix.trace) > 0:
        soc = float(prefix.trace.iloc[-1]["soc_kwh"])
    effective = _event_effective_limits(config, visible_update)
    active_commitments = _active_commitments(visible_update)
    return {
        "current_time": _format_timestamp(current_time),
        "soc_kwh": _round_float(soc),
        "usable_capacity_kwh": effective["usable_capacity_kwh"],
        "max_charge_kw": effective["max_charge_kw"],
        "max_discharge_kw": effective["max_discharge_kw"],
        "reserve_soc_kwh": effective["reserve_soc_kwh"],
        "export_limit_kw": effective["export_limit_kw"],
        "active_forecast_version": (
            "forecast_revision_v2"
            if visible_update.get("event_family") == "forecast_revision"
            else prior_state["active_forecast_version"]
        ),
        "active_commitments": active_commitments,
        "revoked_or_superseded_items": _revoked_items(config, visible_update),
    }


def _execute_prefix(
    frame: pd.DataFrame,
    stage1_actions: np.ndarray,
    config: BatteryConfig,
    split: EpisodeSplit,
    step_hours: float,
) -> SimulationResult:
    if split.stage1_executed_steps == 0:
        empty = pd.DataFrame(
            columns=[
                "requested_action_kw",
                "charge_kw",
                "discharge_kw",
                "soc_kwh",
                "available_capacity_kwh",
                "reserve_soc_kwh",
                "capacity_loss_kwh",
                "import_kw",
                "export_kw",
                "curtailed_kw",
                "energy_cost",
                "degradation_cost",
                "violation_cost",
            ],
            index=frame.iloc[:0].index,
        )
        return SimulationResult(empty, 0.0, True, 0.0)
    return simulate_dispatch(
        frame.iloc[: split.stage1_executed_steps],
        stage1_actions[: split.stage1_executed_steps],
        config,
        overrides=BatteryOverrides(),
        require_terminal_soc=False,
        step_hours=step_hours,
    )


def _episode_step_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Episode execution requires at least two timestamps to determine its interval.")
    deltas = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) == 0 or not np.isfinite(deltas).all():
        raise ValueError("Episode timestamps do not define a finite interval.")
    step_hours = float(np.median(deltas) / 3600.0)
    if step_hours <= 0:
        raise ValueError("Episode interval must be positive.")
    return step_hours


def _event_effective_limits(config: BatteryConfig, visible_update: dict[str, Any]) -> dict[str, float]:
    return {
        "usable_capacity_kwh": _round_float(
            float(visible_update.get("usable_capacity_kwh", config.energy_capacity_kwh))
        ),
        "max_charge_kw": _round_float(float(visible_update.get("max_charge_kw", config.max_charge_kw))),
        "max_discharge_kw": _round_float(
            float(visible_update.get("max_discharge_kw", config.max_discharge_kw))
        ),
        "reserve_soc_kwh": _round_float(
            float(visible_update.get("reserve_soc_kwh", config.reserve_soc_kwh))
        ),
        "export_limit_kw": _round_float(float(visible_update.get("export_limit_kw", config.export_limit_kw))),
    }


def _active_commitments(visible_update: dict[str, Any]) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    for field in (
        "usable_capacity_kwh",
        "max_charge_kw",
        "max_discharge_kw",
        "reserve_soc_kwh",
        "export_limit_kw",
        "forecast_revision_blend_weight",
    ):
        if field in visible_update:
            commitments.append({"field": field, "value": visible_update[field]})
    return commitments


def _revoked_items(config: BatteryConfig, visible_update: dict[str, Any]) -> list[dict[str, Any]]:
    old_values = {
        "usable_capacity_kwh": config.energy_capacity_kwh,
        "max_charge_kw": config.max_charge_kw,
        "max_discharge_kw": config.max_discharge_kw,
        "reserve_soc_kwh": config.reserve_soc_kwh,
        "export_limit_kw": config.export_limit_kw,
    }
    return [
        {"field": field, "previous_value": _round_float(value)}
        for field, value in old_values.items()
        if field in visible_update and float(visible_update[field]) != value
    ]


def _format_timestamp(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _round_float(value: Any) -> float:
    return round(float(value), 6)
