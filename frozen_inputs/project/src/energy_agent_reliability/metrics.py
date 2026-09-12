"""Pre-registered state, action, execution, and operating-success metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .battery import SimulationResult

MAIN_NORMALIZED_REGRET_THRESHOLD = 0.10
SENSITIVITY_THRESHOLDS = (0.05, 0.20)


def compute_outcome_vector(
    *,
    state_governance_success: bool,
    simulation: SimulationResult,
    mpc_cost: float,
    perfect_information_cost: float,
    parser_success: bool,
    plan_length_correct: bool,
    raw_action_success: bool | None = None,
    normalized_regret_threshold: float = MAIN_NORMALIZED_REGRET_THRESHOLD,
) -> dict[str, Any]:
    """Compute the formal G/A/D/C/R/J outcome vector for one executed episode."""
    action_success = action_success_from_simulation(simulation) if raw_action_success is None else raw_action_success
    dispatch_execution_success = bool(parser_success and plan_length_correct)
    feasible_success = operational_feasibility_success(simulation, action_success)
    regret_to_mpc = normalized_regret(simulation.total_cost, mpc_cost)
    economic_success = bool(feasible_success and regret_to_mpc <= normalized_regret_threshold)
    continuous = continuous_operational_outcome(simulation, mpc_cost, perfect_information_cost)
    return {
        "G_state_governance_success": bool(state_governance_success),
        "A_governed_action_success": bool(action_success),
        "D_dispatch_execution_success": dispatch_execution_success,
        "C_feasible_operational_feasibility_success": feasible_success,
        "C_economic_operational_economic_success": economic_success,
        "R_continuous_operational_outcome": continuous,
        "J_feasible": bool(state_governance_success and feasible_success),
        "J_economic": bool(state_governance_success and economic_success),
        "joint_quadrant": _joint_quadrant(bool(state_governance_success), feasible_success),
        "normalized_regret_to_mpc": regret_to_mpc,
        "main_normalized_regret_threshold": normalized_regret_threshold,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
    }


def action_success_from_simulation(simulation: SimulationResult, tolerance: float = 1e-8) -> bool:
    """Raw action success fails if the simulator had to clip the requested action."""
    trace = simulation.trace
    if len(trace) == 0:
        return False
    requested = trace["requested_action_kw"].to_numpy(dtype=float)
    realized = trace["charge_kw"].to_numpy(dtype=float) - trace["discharge_kw"].to_numpy(dtype=float)
    violation = trace["violation_cost"].to_numpy(dtype=float)
    return bool(np.max(np.abs(requested - realized)) <= tolerance and np.max(violation) <= tolerance)


def operational_feasibility_success(simulation: SimulationResult, action_success: bool) -> bool:
    """Feasibility cannot be rescued by post-hoc clipping."""
    return bool(
        simulation.feasible
        and action_success
        and simulation.terminal_soc_gap_kwh <= 1e-6
        and float(simulation.trace["violation_cost"].sum()) <= 1e-8
    )


def normalized_regret(cost: float, baseline_cost: float) -> float:
    denominator = max(abs(float(baseline_cost)), 1.0)
    return float((float(cost) - float(baseline_cost)) / denominator)


def continuous_operational_outcome(
    simulation: SimulationResult, mpc_cost: float, perfect_information_cost: float
) -> dict[str, Any]:
    trace = simulation.trace
    step_hours = _step_hours_from_trace(simulation)
    clipped_kwh = float(
        np.abs(
            trace["requested_action_kw"].to_numpy(dtype=float)
            - (trace["charge_kw"].to_numpy(dtype=float) - trace["discharge_kw"].to_numpy(dtype=float))
        ).sum()
        * step_hours
    )
    grid_purchase_cost = float(trace.loc[trace["energy_cost"] > 0, "energy_cost"].sum())
    export_revenue = float(-trace.loc[trace["energy_cost"] < 0, "energy_cost"].sum())
    return {
        "total_operating_cost": float(simulation.total_cost),
        "relative_cost_to_mpc": normalized_regret(simulation.total_cost, mpc_cost),
        "relative_cost_to_perfect_information": normalized_regret(
            simulation.total_cost, perfect_information_cost
        ),
        "battery_degradation_cost": float(trace["degradation_cost"].sum()),
        "grid_purchase_cost": grid_purchase_cost,
        "export_revenue": export_revenue,
        "curtailment_kwh": float(trace["curtailed_kw"].sum() * step_hours),
        "clipped_action_magnitude_kwh": clipped_kwh,
        "penalty_components": {
            "clipped_action_penalty": float(trace["violation_cost"].sum()),
        },
    }


def _joint_quadrant(state_success: bool, feasible_success: bool) -> str:
    return f"G={int(state_success)},C={int(feasible_success)}"


def _step_hours_from_trace(simulation: SimulationResult) -> float:
    index = simulation.trace.index
    if len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    return float(np.median(deltas) / 3600.0)
