"""CVXPY + HiGHS perfect-information and rolling-horizon dispatch baselines."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .battery import BatteryOverrides, SimulationResult, simulate_dispatch
from .config import BatteryConfig


@dataclass(frozen=True)
class BaselineResult:
    action_kw: np.ndarray
    simulation: SimulationResult
    objective_value: float
    solver_status: str


class BaselineError(RuntimeError):
    """Fail-closed baseline error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, step: int | None = None) -> None:
        self.code = code
        self.step = step
        suffix = f" at rolling step {step}" if step is not None else ""
        super().__init__(f"{code}{suffix}: {message}")


def perfect_information_baseline(
    frame: pd.DataFrame,
    config: BatteryConfig,
    overrides: BatteryOverrides = BatteryOverrides(),
    overrides_by_step: list[BatteryOverrides] | None = None,
    initial_soc_kwh: float | None = None,
    enforce_terminal: bool = True,
    step_hours: float | None = None,
) -> BaselineResult:
    """Optimize a deterministic full-horizon dispatch against realized inputs."""
    _validate_frame(frame)
    n = len(frame)
    dt = _step_hours(frame.index, step_hours)
    schedule = overrides_by_step or [overrides] * n
    if len(schedule) != n:
        raise ValueError("Dynamic override schedule length must equal optimization horizon.")
    cap = np.asarray([item.usable_capacity_kwh or config.energy_capacity_kwh for item in schedule])
    reserve = np.asarray(
        [item.reserve_soc_kwh if item.reserve_soc_kwh is not None else config.reserve_soc_kwh for item in schedule]
    )
    max_charge = np.asarray([item.max_charge_kw or config.max_charge_kw for item in schedule])
    max_discharge = np.asarray([item.max_discharge_kw or config.max_discharge_kw for item in schedule])
    export_limit = np.asarray([item.export_limit_kw or config.export_limit_kw for item in schedule])
    initial = config.initial_soc_kwh if initial_soc_kwh is None else initial_soc_kwh
    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    grid_import = cp.Variable(n, nonneg=True)
    grid_export = cp.Variable(n, nonneg=True)
    load = frame["load_kw"].to_numpy(dtype=float)
    pv = frame[_pv_column(frame)].to_numpy(dtype=float)
    price = frame[_price_column(frame)].to_numpy(dtype=float)
    constraints: list[cp.Constraint] = [
        soc[0] == initial,
        soc[1:] >= reserve,
        soc[1:] <= cap,
        charge <= max_charge,
        discharge <= max_discharge,
        grid_export <= export_limit,
        grid_import - grid_export == load - pv + charge - discharge,
    ]
    for step in range(n):
        constraints.append(
            soc[step + 1]
            == soc[step] + charge[step] * config.charge_efficiency * dt - discharge[step] / config.discharge_efficiency * dt
        )
    if enforce_terminal:
        constraints.append(soc[-1] == config.terminal_soc_kwh)
    cost = cp.sum(cp.multiply(price * dt / 1000.0, grid_import - grid_export))
    cost += config.degradation_cost_per_kwh * dt * cp.sum(charge + discharge)
    problem = cp.Problem(cp.Minimize(cost), constraints)
    value = problem.solve(solver=cp.HIGHS, warm_start=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise BaselineError(
            f"PI_SOLVER_{str(problem.status).upper()}",
            f"HiGHS baseline did not solve: {problem.status}",
        )
    if charge.value is None or discharge.value is None:
        raise BaselineError(
            "PI_MISSING_DISPATCH_VALUES",
            "HiGHS reported a solution without dispatch variable values.",
        )
    actions = np.asarray(charge.value - discharge.value, dtype=float)
    simulation = simulate_dispatch(
        frame,
        actions,
        config,
        overrides,
        overrides_by_step=schedule,
        initial_soc_kwh=initial,
        require_terminal_soc=enforce_terminal,
        step_hours=dt,
    )
    if not simulation.feasible:
        raise BaselineError(
            "PI_SIMULATION_INFEASIBLE",
            f"optimized dispatch failed deterministic replay; terminal gap={simulation.terminal_soc_gap_kwh}",
        )
    return BaselineResult(actions, simulation, float(value), str(problem.status))


def rolling_horizon_baseline(
    frame: pd.DataFrame,
    config: BatteryConfig,
    overrides: BatteryOverrides = BatteryOverrides(),
    overrides_by_step: list[BatteryOverrides] | None = None,
    lookahead_steps: int = 6,
) -> BaselineResult:
    """Re-solve at each step using only the current rolling lookahead slice."""
    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive.")
    actions: list[float] = []
    state = config.initial_soc_kwh
    dt = _step_hours(frame.index)
    for start in range(len(frame)):
        stop = min(start + lookahead_steps, len(frame))
        is_final_window = stop == len(frame)
        window = frame.iloc[start:stop]
        try:
            result = perfect_information_baseline(
                window,
                config,
                overrides,
                overrides_by_step=(overrides_by_step or [overrides] * len(frame))[start:stop],
                initial_soc_kwh=state,
                enforce_terminal=is_final_window,
                step_hours=dt,
            )
        except BaselineError as exc:
            raise BaselineError(
                f"MPC_WINDOW_{exc.code}",
                str(exc),
                step=start,
            ) from exc
        action = float(result.action_kw[0])
        actions.append(action)
        one_step = simulate_dispatch(
            frame.iloc[start : start + 1],
            [action],
            config,
            overrides,
            overrides_by_step=(overrides_by_step or [overrides] * len(frame))[start : start + 1],
            initial_soc_kwh=state,
            require_terminal_soc=False,
            step_hours=dt,
        )
        state = float(one_step.trace.iloc[-1]["soc_kwh"])
    simulation = simulate_dispatch(
        frame,
        actions,
        config,
        overrides,
        overrides_by_step=overrides_by_step,
        require_terminal_soc=True,
        step_hours=dt,
    )
    if not simulation.feasible:
        raise BaselineError(
            "MPC_SIMULATION_INFEASIBLE",
            f"rolling dispatch failed deterministic replay; terminal gap={simulation.terminal_soc_gap_kwh}",
        )
    return BaselineResult(np.asarray(actions), simulation, simulation.total_cost, "rolling_highs")


def _validate_frame(frame: pd.DataFrame) -> None:
    if len(frame) == 0:
        raise ValueError("Cannot optimize an empty horizon.")
    required = {"load_kw"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing baseline columns: {sorted(missing)}")
    columns = ["load_kw", _pv_column(frame), _price_column(frame)]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Baseline inputs must be finite.")


def _step_hours(index: pd.Index, explicit_step_hours: float | None = None) -> float:
    if explicit_step_hours is not None:
        step = float(explicit_step_hours)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Explicit baseline step_hours must be finite and positive.")
        return step
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Cannot infer baseline interval from fewer than two timestamps; pass step_hours explicitly.")
    deltas = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    return float(np.median(deltas) / 3600.0)


def _pv_column(frame: pd.DataFrame) -> str:
    return "pv_ac_kw" if "pv_ac_kw" in frame.columns else "pv_kw"


def _price_column(frame: pd.DataFrame) -> str:
    return "day_ahead_price_usd_mwh" if "day_ahead_price_usd_mwh" in frame.columns else "price_per_mwh"
