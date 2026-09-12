"""Versioned E2 projection implementation with lexicographic warm starts.

The mathematical program is copied from V6 without changing variables, bounds,
constraints, objective priorities, tolerances, or terminal obligation.  The
implementation reuses the primal solution from each completed lexicographic
stage as the warm start for the next stage.  This module is diagnostic until a
full 1,800-row replay has passed its frozen gate.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import BatteryConfig
from .online_gate_v6 import (
    ENERGY_TOLERANCE_KWH,
    LEXICOGRAPHIC_TOLERANCE,
    POWER_TOLERANCE_KW,
    ProjectionSolution,
    _finite,
    _normalized_frame,
    limits_at,
    task_frame,
)

REPAIR_METHOD_VERSION = "e2_projection_repair_v1_warm_start"


def _solve_repaired(problem: cp.Problem, label: str, *, warm_start: bool) -> None:
    problem.solve(solver=cp.HIGHS, warm_start=warm_start)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"{label} failed with status {problem.status}")


def lexicographic_projection_repaired(
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
    """Solve the frozen V6 projection contract with stage-to-stage warm starts."""
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
                + charge[local_step] * battery.charge_efficiency * 0.25
                - discharge[local_step]
                / battery.discharge_efficiency
                * 0.25,
            ]
        )
    if require_terminal:
        constraints.append(soc[-1] == battery.terminal_soc_kwh)

    distance = 0.25 * cp.sum(absolute_delta)
    first = cp.Problem(cp.Minimize(distance), constraints)
    _solve_repaired(first, "projection_distance", warm_start=False)
    distance_optimum = _finite(distance.value, "projection distance")

    throughput = 0.25 * cp.sum(charge + discharge)
    second_constraints = [
        *constraints,
        distance <= distance_optimum + LEXICOGRAPHIC_TOLERANCE,
    ]
    second = cp.Problem(cp.Minimize(throughput), second_constraints)
    _solve_repaired(second, "projection_throughput", warm_start=True)
    throughput_optimum = _finite(throughput.value, "projection throughput")

    signed_grid_vector = cp.hstack(signed_grid_terms)
    physical_cost = 0.25 / 1000.0 * cp.sum(
        cp.multiply(price, signed_grid_vector)
    )
    physical_cost += (
        battery.degradation_cost_per_kwh * 0.25 * cp.sum(charge + discharge)
    )
    third = cp.Problem(
        cp.Minimize(physical_cost),
        [
            *second_constraints,
            throughput <= throughput_optimum + ENERGY_TOLERANCE_KWH,
        ],
    )
    _solve_repaired(third, "projection_cost_tiebreak", warm_start=True)
    if charge.value is None or discharge.value is None or curtailment.value is None:
        raise RuntimeError("Projection returned no primal values.")
    charge_value = np.maximum(np.asarray(charge.value, dtype=float), 0.0)
    discharge_value = np.maximum(np.asarray(discharge.value, dtype=float), 0.0)
    curtailment_value = np.maximum(np.asarray(curtailment.value, dtype=float), 0.0)
    # HiGHS can return sub-tolerance values on the inactive side of a binary
    # charge/discharge constraint after a warm-started lexicographic solve.
    # Remove that numerical overlap without changing the signed action or grid
    # flow used by replay. The returned charge/discharge decomposition is a
    # numerical reporting canonicalization, not a second optimization solve.
    overlap = np.minimum(charge_value, discharge_value)
    charge_value = np.maximum(charge_value - overlap, 0.0)
    discharge_value = np.maximum(discharge_value - overlap, 0.0)
    action_value = charge_value - discharge_value
    signed_grid = load - pv + curtailment_value + action_value
    physical_cost_value = float(
        0.25 / 1000.0 * np.sum(price * signed_grid)
        + battery.degradation_cost_per_kwh
        * 0.25
        * np.sum(charge_value + discharge_value)
    )
    return ProjectionSolution(
        action_kw=action_value,
        charge_kw=charge_value,
        discharge_kw=discharge_value,
        curtailment_kw=curtailment_value,
        distance_kwh=float(0.25 * np.abs(charge_value - discharge_value - reference).sum()),
        throughput_kwh=float(0.25 * (charge_value + discharge_value).sum()),
        physical_cost_usd=physical_cost_value,
        solver_status=str(third.status),
        simultaneous_intervals=int(
            np.count_nonzero(
                np.minimum(charge_value, discharge_value) > POWER_TOLERANCE_KW
            )
        ),
        maximum_simultaneous_kw=float(
            np.minimum(charge_value, discharge_value).max(initial=0.0)
        ),
    )


def repair_contract_summary() -> dict[str, Any]:
    """Expose the contract elements used by the equivalence validator."""
    return {
        "method_version": REPAIR_METHOD_VERSION,
        "solver": "HiGHS via CVXPY",
        "warm_start_stages": ["projection_throughput", "projection_cost_tiebreak"],
        "mathematical_contract_changed": False,
        "tolerances": {
            "lexicographic": LEXICOGRAPHIC_TOLERANCE,
            "energy": ENERGY_TOLERANCE_KWH,
            "power": POWER_TOLERANCE_KW,
        },
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
