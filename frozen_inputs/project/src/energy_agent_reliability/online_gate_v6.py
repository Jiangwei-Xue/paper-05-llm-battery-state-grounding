"""Deterministic optimization and replay primitives for the V6 online-gate study.

The V6 study is a new evidence tier.  These functions do not alter the frozen V4
scorer or any saved primary, extension, or paired-authority record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import BatteryConfig

STEP_HOURS = 0.25
POWER_TOLERANCE_KW = 1e-4
ENERGY_TOLERANCE_KWH = 1e-5
LEXICOGRAPHIC_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DispatchSolution:
    action_kw: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    curtailment_kw: np.ndarray
    objective_usd: float
    solver_status: str
    simultaneous_intervals: int
    maximum_simultaneous_kw: float


@dataclass(frozen=True)
class ProjectionSolution:
    action_kw: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    curtailment_kw: np.ndarray
    distance_kwh: float
    throughput_kwh: float
    physical_cost_usd: float
    solver_status: str
    simultaneous_intervals: int
    maximum_simultaneous_kw: float


@dataclass(frozen=True)
class ReplayResult:
    trace: pd.DataFrame
    feasible: bool
    total_cost_usd: float
    terminal_soc_gap_kwh: float
    failure_codes: tuple[str, ...]
    event_soc_kwh: float


def task_frame(task: dict[str, Any]) -> pd.DataFrame:
    """Return the information available after the event, with the revised suffix."""
    activation = int(task["visible_update"]["activation_step"])
    stage1 = task["model_visible_episode"]["stage_1"]["visible_timeseries"]
    stage2 = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    rows = [*stage1[:activation], *stage2]
    return pd.DataFrame(
        {
            "load_kw": [float(row["load_kw"]) for row in rows],
            "pv_kw": [float(row["pv_forecast_kw"]) for row in rows],
            "price_usd_mwh": [float(row["dam_price_usd_mwh"]) for row in rows],
        },
        index=pd.to_datetime([row["timestamp_utc"] for row in rows], utc=True),
    )


def stage1_information_frame(task: dict[str, Any]) -> pd.DataFrame:
    """Return the pre-event forecast available to Stage 1 for all 96 steps."""
    rows = task["model_visible_episode"]["stage_1"]["visible_timeseries"]
    return pd.DataFrame(
        {
            "load_kw": [float(row["load_kw"]) for row in rows],
            "pv_kw": [float(row["pv_forecast_kw"]) for row in rows],
            "price_usd_mwh": [float(row["dam_price_usd_mwh"]) for row in rows],
        },
        index=pd.to_datetime([row["timestamp_utc"] for row in rows], utc=True),
    )


def limits_at(
    task: dict[str, Any],
    step: int,
    battery: BatteryConfig,
    *,
    apply_event: bool,
) -> dict[str, float]:
    update = task["visible_update"]
    active = apply_event and step >= int(update["activation_step"])
    return {
        "usable_capacity_kwh": float(
            update.get("usable_capacity_kwh", battery.energy_capacity_kwh)
            if active
            else battery.energy_capacity_kwh
        ),
        "max_charge_kw": float(
            update.get("max_charge_kw", battery.max_charge_kw)
            if active
            else battery.max_charge_kw
        ),
        "max_discharge_kw": float(
            update.get("max_discharge_kw", battery.max_discharge_kw)
            if active
            else battery.max_discharge_kw
        ),
        "reserve_soc_kwh": float(
            update.get("reserve_soc_kwh", battery.reserve_soc_kwh)
            if active
            else battery.reserve_soc_kwh
        ),
        "export_limit_kw": float(
            update.get("export_limit_kw", battery.export_limit_kw)
            if active
            else battery.export_limit_kw
        ),
    }


def economic_dispatch(
    task: dict[str, Any],
    battery: BatteryConfig,
    *,
    initial_soc_kwh: float,
    start_step: int,
    apply_event: bool,
    enforce_mutual_exclusion: bool = True,
    allow_curtailment: bool = True,
    frame_override: pd.DataFrame | None = None,
) -> DispatchSolution:
    """Solve one full-information segment and retain charge/discharge components."""
    frame = (
        task_frame(task)
        if frame_override is None
        else _normalized_frame(frame_override)
    ).iloc[start_step:]
    n = len(frame)
    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    mode = cp.Variable(n, boolean=True) if enforce_mutual_exclusion else None
    curtailment = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    load = frame["load_kw"].to_numpy(dtype=float)
    pv = frame["pv_kw"].to_numpy(dtype=float)
    price = frame["price_usd_mwh"].to_numpy(dtype=float)
    constraints: list[Any] = [soc[0] == initial_soc_kwh, curtailment <= pv]
    if not allow_curtailment:
        constraints.append(curtailment == 0)
    signed_grid_terms: list[Any] = []
    for local_step in range(n):
        step = start_step + local_step
        limits = limits_at(task, step, battery, apply_event=apply_event)
        signed_grid = (
            load[local_step]
            - pv[local_step]
            + curtailment[local_step]
            + charge[local_step]
            - discharge[local_step]
        )
        signed_grid_terms.append(signed_grid)
        constraints.extend(
            [
                signed_grid >= -limits["export_limit_kw"],
                soc[local_step + 1] >= limits["reserve_soc_kwh"],
                soc[local_step + 1] <= limits["usable_capacity_kwh"],
                soc[local_step + 1]
                == soc[local_step]
                + charge[local_step] * battery.charge_efficiency * STEP_HOURS
                - discharge[local_step]
                / battery.discharge_efficiency
                * STEP_HOURS,
            ]
        )
        if mode is None:
            constraints.extend(
                [
                    charge[local_step] <= limits["max_charge_kw"],
                    discharge[local_step] <= limits["max_discharge_kw"],
                ]
            )
        else:
            constraints.extend(
                [
                    charge[local_step]
                    <= limits["max_charge_kw"] * mode[local_step],
                    discharge[local_step]
                    <= limits["max_discharge_kw"] * (1 - mode[local_step]),
                ]
            )
    constraints.append(soc[-1] == battery.terminal_soc_kwh)
    signed_grid_vector = cp.hstack(signed_grid_terms)
    objective = STEP_HOURS / 1000.0 * cp.sum(
        cp.multiply(price, signed_grid_vector)
    )
    objective += (
        battery.degradation_cost_per_kwh
        * STEP_HOURS
        * cp.sum(charge + discharge)
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    _solve(problem, "economic_dispatch")
    return _dispatch_solution(problem, charge, discharge, curtailment)


def lexicographic_projection(
    task: dict[str, Any],
    reference_actions: np.ndarray | list[float],
    battery: BatteryConfig,
    *,
    initial_soc_kwh: float,
    start_step: int,
    apply_event: bool,
    require_terminal: bool,
    frame_override: pd.DataFrame | None = None,
) -> ProjectionSolution:
    """Project actions by distance, throughput, then operating cost."""
    reference = np.asarray(reference_actions, dtype=float)
    frame = (
        task_frame(task)
        if frame_override is None
        else _normalized_frame(frame_override)
    ).iloc[start_step:]
    if len(reference) != len(frame) or not np.isfinite(reference).all():
        raise ValueError("Projection reference does not match the finite segment.")
    n = len(frame)
    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    mode = cp.Variable(n, boolean=True)
    curtailment = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    absolute_delta = cp.Variable(n, nonneg=True)
    action = charge - discharge
    load = frame["load_kw"].to_numpy(dtype=float)
    pv = frame["pv_kw"].to_numpy(dtype=float)
    price = frame["price_usd_mwh"].to_numpy(dtype=float)
    constraints: list[Any] = [
        soc[0] == initial_soc_kwh,
        absolute_delta >= action - reference,
        absolute_delta >= reference - action,
        curtailment <= pv,
    ]
    signed_grid_terms: list[Any] = []
    for local_step in range(n):
        step = start_step + local_step
        limits = limits_at(task, step, battery, apply_event=apply_event)
        signed_grid = (
            load[local_step]
            - pv[local_step]
            + curtailment[local_step]
            + charge[local_step]
            - discharge[local_step]
        )
        signed_grid_terms.append(signed_grid)
        constraints.extend(
            [
                charge[local_step]
                <= limits["max_charge_kw"] * mode[local_step],
                discharge[local_step]
                <= limits["max_discharge_kw"] * (1 - mode[local_step]),
                signed_grid >= -limits["export_limit_kw"],
                soc[local_step + 1] >= limits["reserve_soc_kwh"],
                soc[local_step + 1] <= limits["usable_capacity_kwh"],
                soc[local_step + 1]
                == soc[local_step]
                + charge[local_step] * battery.charge_efficiency * STEP_HOURS
                - discharge[local_step]
                / battery.discharge_efficiency
                * STEP_HOURS,
            ]
        )
    if require_terminal:
        constraints.append(soc[-1] == battery.terminal_soc_kwh)

    distance = STEP_HOURS * cp.sum(absolute_delta)
    first = cp.Problem(cp.Minimize(distance), constraints)
    _solve(first, "projection_distance")
    distance_optimum = _finite(distance.value, "projection distance")

    throughput = STEP_HOURS * cp.sum(charge + discharge)
    second_constraints = [
        *constraints,
        distance <= distance_optimum + LEXICOGRAPHIC_TOLERANCE,
    ]
    second = cp.Problem(cp.Minimize(throughput), second_constraints)
    _solve(second, "projection_throughput")
    throughput_optimum = _finite(throughput.value, "projection throughput")

    signed_grid_vector = cp.hstack(signed_grid_terms)
    physical_cost = STEP_HOURS / 1000.0 * cp.sum(
        cp.multiply(price, signed_grid_vector)
    )
    physical_cost += (
        battery.degradation_cost_per_kwh
        * STEP_HOURS
        * cp.sum(charge + discharge)
    )
    third = cp.Problem(
        cp.Minimize(physical_cost),
        [
            *second_constraints,
            throughput <= throughput_optimum + ENERGY_TOLERANCE_KWH,
        ],
    )
    _solve(third, "projection_cost_tiebreak")
    if charge.value is None or discharge.value is None or curtailment.value is None:
        raise RuntimeError("Projection returned no primal values.")
    charge_value = np.maximum(np.asarray(charge.value, dtype=float), 0.0)
    discharge_value = np.maximum(np.asarray(discharge.value, dtype=float), 0.0)
    curtailment_value = np.maximum(np.asarray(curtailment.value, dtype=float), 0.0)
    overlap = np.minimum(charge_value, discharge_value)
    return ProjectionSolution(
        action_kw=charge_value - discharge_value,
        charge_kw=charge_value,
        discharge_kw=discharge_value,
        curtailment_kw=curtailment_value,
        distance_kwh=float(
            STEP_HOURS
            * np.abs(charge_value - discharge_value - reference).sum()
        ),
        throughput_kwh=float(
            STEP_HOURS * (charge_value + discharge_value).sum()
        ),
        physical_cost_usd=float(third.value),
        solver_status=str(third.status),
        simultaneous_intervals=int(
            np.count_nonzero(overlap > POWER_TOLERANCE_KW)
        ),
        maximum_simultaneous_kw=float(overlap.max(initial=0.0)),
    )


def replay_two_stage(
    task: dict[str, Any],
    stage1_actions: np.ndarray | list[float],
    stage2_actions: np.ndarray | list[float],
    battery: BatteryConfig,
    *,
    frame_override: pd.DataFrame | None = None,
) -> ReplayResult:
    activation = int(task["visible_update"]["activation_step"])
    stage1 = np.asarray(stage1_actions, dtype=float)
    stage2 = np.asarray(stage2_actions, dtype=float)
    horizon = len(task["model_visible_episode"]["stage_1"]["visible_timeseries"])
    if len(stage1) != horizon or len(stage2) != horizon - activation:
        raise ValueError("Two-stage dispatch lengths do not match the task horizon.")
    actions = np.concatenate([stage1[:activation], stage2])
    return replay_actions(task, actions, battery, frame_override=frame_override)


def replay_actions(
    task: dict[str, Any],
    actions: np.ndarray | list[float],
    battery: BatteryConfig,
    *,
    frame_override: pd.DataFrame | None = None,
) -> ReplayResult:
    frame = (
        task_frame(task)
        if frame_override is None
        else _normalized_frame(frame_override)
    )
    requested = np.asarray(actions, dtype=float)
    if len(requested) != len(frame) or not np.isfinite(requested).all():
        raise ValueError("Replay actions do not match the finite task horizon.")
    activation = int(task["visible_update"]["activation_step"])
    soc = float(battery.initial_soc_kwh)
    event_soc = soc
    rows: list[dict[str, float]] = []
    failure_codes: set[str] = set()
    for step, (_, row) in enumerate(frame.iterrows()):
        if step == activation:
            event_soc = soc
        limits = limits_at(task, step, battery, apply_event=True)
        capacity = limits["usable_capacity_kwh"]
        reserve = limits["reserve_soc_kwh"]
        if soc > capacity + ENERGY_TOLERANCE_KWH:
            failure_codes.add("EVENT_CAPACITY_DROP_BELOW_SOC")
        soc = min(soc, capacity)
        action = float(requested[step])
        requested_charge = max(action, 0.0)
        requested_discharge = max(-action, 0.0)
        charge = min(
            requested_charge,
            limits["max_charge_kw"],
            max(capacity - soc, 0.0) / (battery.charge_efficiency * STEP_HOURS),
        )
        discharge = min(
            requested_discharge,
            limits["max_discharge_kw"],
            max(soc - reserve, 0.0)
            * battery.discharge_efficiency
            / STEP_HOURS,
        )
        if abs(action - (charge - discharge)) > POWER_TOLERANCE_KW:
            failure_codes.add("ACTION_CLIPPED")
        soc += charge * battery.charge_efficiency * STEP_HOURS
        soc -= discharge / battery.discharge_efficiency * STEP_HOURS
        residual = float(row["load_kw"]) - float(row["pv_kw"]) + charge - discharge
        export = min(max(-residual, 0.0), limits["export_limit_kw"])
        curtailment = max(-residual, 0.0) - export
        grid_import = max(residual, 0.0)
        energy_cost = (
            (grid_import - export)
            * float(row["price_usd_mwh"])
            * STEP_HOURS
            / 1000.0
        )
        degradation = (
            (charge + discharge)
            * STEP_HOURS
            * battery.degradation_cost_per_kwh
        )
        rows.append(
            {
                "requested_action_kw": action,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "soc_kwh": soc,
                "capacity_kwh": capacity,
                "reserve_kwh": reserve,
                "grid_import_kw": grid_import,
                "grid_export_kw": export,
                "curtailment_kw": curtailment,
                "energy_cost_usd": energy_cost,
                "degradation_cost_usd": degradation,
            }
        )
    terminal_gap = abs(soc - battery.terminal_soc_kwh)
    if terminal_gap > ENERGY_TOLERANCE_KWH:
        failure_codes.add("TERMINAL_SOC_MISMATCH")
    trace = pd.DataFrame(rows, index=frame.index)
    total_cost = float(
        trace[["energy_cost_usd", "degradation_cost_usd"]].sum().sum()
    )
    return ReplayResult(
        trace=trace,
        feasible=not failure_codes,
        total_cost_usd=total_cost,
        terminal_soc_gap_kwh=terminal_gap,
        failure_codes=tuple(sorted(failure_codes)),
        event_soc_kwh=float(event_soc),
    )


def _dispatch_solution(
    problem: cp.Problem,
    charge: Any,
    discharge: Any,
    curtailment: Any,
) -> DispatchSolution:
    if charge.value is None or discharge.value is None or curtailment.value is None:
        raise RuntimeError("Economic dispatch returned no primal values.")
    charge_value = np.maximum(np.asarray(charge.value, dtype=float), 0.0)
    discharge_value = np.maximum(np.asarray(discharge.value, dtype=float), 0.0)
    curtailment_value = np.maximum(np.asarray(curtailment.value, dtype=float), 0.0)
    overlap = np.minimum(charge_value, discharge_value)
    return DispatchSolution(
        action_kw=charge_value - discharge_value,
        charge_kw=charge_value,
        discharge_kw=discharge_value,
        curtailment_kw=curtailment_value,
        objective_usd=float(problem.value),
        solver_status=str(problem.status),
        simultaneous_intervals=int(
            np.count_nonzero(overlap > POWER_TOLERANCE_KW)
        ),
        maximum_simultaneous_kw=float(overlap.max(initial=0.0)),
    )


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    pv_column = "pv_ac_kw" if "pv_ac_kw" in frame.columns else "pv_kw"
    price_column = (
        "day_ahead_price_usd_mwh"
        if "day_ahead_price_usd_mwh" in frame.columns
        else "price_usd_mwh"
    )
    return pd.DataFrame(
        {
            "load_kw": frame["load_kw"].to_numpy(dtype=float),
            "pv_kw": frame[pv_column].to_numpy(dtype=float),
            "price_usd_mwh": frame[price_column].to_numpy(dtype=float),
        },
        index=frame.index,
    )


def _solve(problem: cp.Problem, label: str) -> None:
    problem.solve(solver=cp.HIGHS, warm_start=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"{label} failed with status {problem.status}")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result
