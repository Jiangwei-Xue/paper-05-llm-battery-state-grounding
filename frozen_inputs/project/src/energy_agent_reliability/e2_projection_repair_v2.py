"""Incremental native-HiGHS implementation of the frozen E2 projection.

One model is built for each row.  The three lexicographic solves reuse that
model: only the objective changes and one bound row is appended after each
completed stage.  The formulation and reporting tolerances are intentionally
the V6 contract; this module does not define a new physical problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import cvxpy as cp
import highspy
import numpy as np
import pandas as pd

from .config import BatteryConfig
from .online_gate_v6 import (
    ENERGY_TOLERANCE_KWH,
    LEXICOGRAPHIC_TOLERANCE,
    POWER_TOLERANCE_KW,
    ProjectionSolution,
    _normalized_frame,
    limits_at,
    task_frame,
)

REPAIR_METHOD_VERSION = "e2_projection_repair_v2_incremental_highs"
STEP_HOURS = 0.25


@dataclass(frozen=True)
class StageTrace:
    stage: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    model_status: str
    primal_solution_status: str
    objective_value: float | None
    variable_count: int
    row_count: int
    binary_variable_count: int
    node_count: int | None
    simplex_iterations: int | None
    mip_gap: float | None
    maximum_rss_kb: int | None
    solver_options: dict[str, Any]
    previous_stage_incumbent_supplied: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rss_kb() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        return None


def _safe_info(info: Any, name: str) -> int | float | None:
    value = getattr(info, name, None)
    if value is None:
        return None
    try:
        return int(value) if name in {"mip_node_count", "simplex_iteration_count"} else float(value)
    except (TypeError, ValueError):
        return None


def _safe_int_info(info: Any, name: str) -> int | None:
    value = _safe_info(info, name)
    return None if value is None else int(value)


class _NativeProjection:
    """Single-row HiGHS model with reusable lexicographic stages."""

    def __init__(
        self,
        task: dict[str, Any],
        reference: np.ndarray,
        battery: BatteryConfig,
        *,
        initial_soc_kwh: float,
        start_step: int,
        apply_event: bool,
        require_terminal: bool,
        frame: pd.DataFrame,
    ) -> None:
        self.task = task
        self.reference = reference
        self.battery = battery
        self.start_step = start_step
        self.apply_event = apply_event
        self.require_terminal = require_terminal
        self.frame = frame.iloc[start_step:]
        self.n = len(self.frame)
        self.load = self.frame["load_kw"].to_numpy(dtype=float)
        self.pv = self.frame["pv_kw"].to_numpy(dtype=float)
        self.price = self.frame["price_usd_mwh"].to_numpy(dtype=float)
        self.highs = highspy.Highs()  # type: ignore[no-untyped-call]
        self.options: dict[str, Any] = {
            "output_flag": False,
            "threads": 1,
            "mip_rel_gap": 0.0,
            "mip_abs_gap": 0.0,
            "mip_feasibility_tolerance": 1e-6,
        }
        for key, value in self.options.items():
            status = self.highs.setOptionValue(key, value)
            if status != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS option rejected: {key}={value!r}")

        self._build_canonical_model(initial_soc_kwh)

    def _build_canonical_model(self, initial_soc_kwh: float) -> None:
        """Canonicalize the V6 program once, then load its matrix natively."""
        charge = cp.Variable(self.n, nonneg=True)
        discharge = cp.Variable(self.n, nonneg=True)
        mode = cp.Variable(self.n, boolean=True)
        curtailment = cp.Variable(self.n, nonneg=True)
        soc = cp.Variable(self.n + 1)
        absolute_delta = cp.Variable(self.n, nonneg=True)
        constraints: list[Any] = [
            soc[0] == float(initial_soc_kwh),
            absolute_delta >= charge - discharge - self.reference,
            absolute_delta >= self.reference - charge + discharge,
            curtailment <= self.pv,
        ]
        signed_grid_terms: list[Any] = []
        for t in range(self.n):
            limits = limits_at(self.task, self.start_step + t, self.battery, apply_event=self.apply_event)
            signed_grid = (
                float(self.load[t]) - float(self.pv[t]) + curtailment[t] + charge[t] - discharge[t]
            )
            signed_grid_terms.append(signed_grid)
            constraints.extend(
                [
                    charge[t] <= limits["max_charge_kw"] * mode[t],
                    discharge[t] <= limits["max_discharge_kw"] * (1 - mode[t]),
                    signed_grid >= -limits["export_limit_kw"],
                    soc[t + 1] >= limits["reserve_soc_kwh"],
                    soc[t + 1] <= limits["usable_capacity_kwh"],
                    soc[t + 1]
                    == soc[t]
                    + charge[t] * self.battery.charge_efficiency * STEP_HOURS
                    - discharge[t] / self.battery.discharge_efficiency * STEP_HOURS,
                ]
            )
        if self.require_terminal:
            constraints.append(soc[-1] == self.battery.terminal_soc_kwh)
        distance = STEP_HOURS * cp.sum(absolute_delta)
        problem = cp.Problem(cp.Minimize(distance), constraints)
        data, _, inverse = problem.get_problem_data(cp.HIGHS)
        matrix = data["A"].tocsr()
        n_rows, n_cols = matrix.shape
        inf = highspy.kHighsInf
        lower = (
            np.asarray(data["lower_bounds"], dtype=float)
            if data["lower_bounds"] is not None
            else np.full(n_cols, -inf)
        )
        upper = (
            np.asarray(data["upper_bounds"], dtype=float)
            if data["upper_bounds"] is not None
            else np.full(n_cols, inf)
        )
        self.highs.setOptionValue("output_flag", False)
        self.highs.setOptionValue("threads", 1)
        self.highs.setOptionValue("mip_rel_gap", 0.0)
        self.highs.setOptionValue("mip_abs_gap", 0.0)
        self.highs.setOptionValue("mip_feasibility_tolerance", 1e-6)
        lp = highspy.HighsLp()
        lp.num_col_ = n_cols
        lp.num_row_ = n_rows
        lp.col_cost_ = np.asarray(data["c"], dtype=float)
        lp.col_lower_ = lower
        lp.col_upper_ = upper
        row_upper = np.asarray(data["b"], dtype=float)
        row_lower = np.full(n_rows, -inf)
        equalities = int(data["dims"].zero)
        row_lower[:equalities] = row_upper[:equalities]
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        integrality = [highspy.HighsVarType.kContinuous] * n_cols
        for index in data.get("bool_vars_idx", []):
            integrality[int(index)] = highspy.HighsVarType.kInteger
        lp.integrality_ = integrality
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = n_cols
        lp.a_matrix_.num_row_ = n_rows
        lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
        lp.a_matrix_.value_ = matrix.data.astype(float)
        status = self.highs.passModel(lp)
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"native HiGHS canonical model load failed: {status}")
        offsets = inverse[0].var_offsets
        self.offsets = {int(key): (int(start), int(size)) for key, start in offsets.items() for size in [inverse[0].var_shapes[key][0] if len(inverse[0].var_shapes[key]) == 1 else int(np.prod(inverse[0].var_shapes[key]))]}
        self.var_ids = {
            "charge": int(charge.id),
            "discharge": int(discharge.id),
            "curtailment": int(curtailment.id),
            "soc": int(soc.id),
            "absolute_delta": int(absolute_delta.id),
            "mode": int(mode.id),
        }
        self.distance_expr = np.zeros(n_cols, dtype=float)
        self.distance_expr[self.offsets[self.var_ids["absolute_delta"]][0] : self.offsets[self.var_ids["absolute_delta"]][0] + self.n] = STEP_HOURS
        self.throughput_expr = np.zeros(n_cols, dtype=float)
        for name in ("charge", "discharge"):
            start, _ = self.offsets[self.var_ids[name]]
            self.throughput_expr[start : start + self.n] = STEP_HOURS
        self.cost_expr = np.zeros(n_cols, dtype=float)
        price_coeff = STEP_HOURS / 1000.0 * self.price
        for name, coeff in (("charge", price_coeff), ("discharge", -price_coeff), ("curtailment", price_coeff)):
            start, _ = self.offsets[self.var_ids[name]]
            self.cost_expr[start : start + self.n] = coeff
        self.cost_expr += self.battery.degradation_cost_per_kwh * self.throughput_expr
        self.variable_count = n_cols
        self.base_row_count = n_rows

    def _extract(self) -> np.ndarray:
        solution = self.highs.getSolution()
        if not solution.value_valid or len(solution.col_value) != self.highs.numVariables:
            raise RuntimeError("HiGHS returned an incomplete primal solution")
        return np.asarray(solution.col_value, dtype=float)

    def _stage(self, name: str, objective: Any, incumbent: np.ndarray | None) -> tuple[float, np.ndarray, StageTrace]:
        # Clear solver internals after an upstream CVXPY/HiGHS call while
        # retaining the loaded matrix and lexicographic rows.
        self.highs.clearSolver()
        if incumbent is not None:
            supplied = False
            try:
                status = self.highs.setSolution(
                    len(incumbent),
                    list(range(len(incumbent))),
                    incumbent.tolist(),
                )
                supplied = status == highspy.HighsStatus.kOk
            except (TypeError, ValueError, RuntimeError):
                supplied = False
        else:
            supplied = False
        for index, coefficient in enumerate(objective):
            self.highs.changeColCost(index, float(coefficient))
        self.highs.changeObjectiveOffset(0.0)
        started = _utc_now()
        start_clock = __import__("time").monotonic()
        run_status = self.highs.run()
        elapsed = __import__("time").monotonic() - start_clock
        finished = _utc_now()
        model_status = self.highs.getModelStatus()
        info = self.highs.getInfo()
        if run_status != highspy.HighsStatus.kOk or model_status != highspy.HighsModelStatus.kOptimal:
            trace = StageTrace(
                stage=name,
                started_utc=started,
                finished_utc=finished,
                elapsed_seconds=elapsed,
                model_status=str(model_status),
                primal_solution_status=str(getattr(info, "primal_solution_status", None)),
                objective_value=None,
                variable_count=self.highs.numVariables,
                row_count=self.highs.numConstrs,
                binary_variable_count=self.n,
                node_count=_safe_int_info(info, "mip_node_count"),
                simplex_iterations=_safe_int_info(info, "simplex_iteration_count"),
                mip_gap=_safe_info(info, "mip_gap"),
                maximum_rss_kb=_rss_kb(),
                solver_options=self.options,
                previous_stage_incumbent_supplied=supplied,
            )
            raise RuntimeError(f"{name} did not reach HiGHS optimal: {asdict(trace)}")
        values = self._extract()
        objective_value = float(self.highs.getObjectiveValue())
        trace = StageTrace(
            stage=name,
            started_utc=started,
            finished_utc=finished,
            elapsed_seconds=elapsed,
            model_status=str(model_status),
            primal_solution_status=str(getattr(info, "primal_solution_status", None)),
            objective_value=objective_value,
            variable_count=self.highs.numVariables,
            row_count=self.highs.numConstrs,
            binary_variable_count=self.n,
            node_count=_safe_int_info(info, "mip_node_count"),
            simplex_iterations=_safe_int_info(info, "simplex_iteration_count"),
            mip_gap=_safe_info(info, "mip_gap"),
            maximum_rss_kb=_rss_kb(),
            solver_options=self.options,
            previous_stage_incumbent_supplied=supplied,
        )
        return objective_value, values, trace

    def solve(self) -> tuple[ProjectionSolution, list[dict[str, Any]]]:
        distance_opt, values, trace1 = self._stage("projection_distance", self.distance_expr, None)
        self._add_bound(self.distance_expr, distance_opt + LEXICOGRAPHIC_TOLERANCE)
        throughput_opt, values, trace2 = self._stage("projection_throughput", self.throughput_expr, values)
        self._add_bound(self.throughput_expr, throughput_opt + ENERGY_TOLERANCE_KWH)
        _, values, trace3 = self._stage("projection_cost_tiebreak", self.cost_expr, values)
        charge_start, _ = self.offsets[self.var_ids["charge"]]
        discharge_start, _ = self.offsets[self.var_ids["discharge"]]
        curtailment_start, _ = self.offsets[self.var_ids["curtailment"]]
        charge_value = np.maximum(values[charge_start : charge_start + self.n], 0.0)
        discharge_value = np.maximum(values[discharge_start : discharge_start + self.n], 0.0)
        curtailment_value = np.maximum(values[curtailment_start : curtailment_start + self.n], 0.0)
        overlap = np.minimum(charge_value, discharge_value)
        charge_value = np.maximum(charge_value - overlap, 0.0)
        discharge_value = np.maximum(discharge_value - overlap, 0.0)
        reporting_overlap = np.minimum(charge_value, discharge_value)
        action_value = charge_value - discharge_value
        signed_grid = self.load - self.pv + curtailment_value + action_value
        physical_cost = float(
            STEP_HOURS / 1000.0 * np.sum(self.price * signed_grid)
            + self.battery.degradation_cost_per_kwh
            * STEP_HOURS
            * np.sum(charge_value + discharge_value)
        )
        solution = ProjectionSolution(
            action_kw=action_value,
            charge_kw=charge_value,
            discharge_kw=discharge_value,
            curtailment_kw=curtailment_value,
            distance_kwh=float(STEP_HOURS * np.abs(action_value - self.reference).sum()),
            throughput_kwh=float(STEP_HOURS * np.sum(charge_value + discharge_value)),
            physical_cost_usd=physical_cost,
            solver_status="optimal",
            simultaneous_intervals=int(np.count_nonzero(reporting_overlap > POWER_TOLERANCE_KW)),
            maximum_simultaneous_kw=float(reporting_overlap.max(initial=0.0)),
        )
        return solution, [asdict(t) for t in (trace1, trace2, trace3)]

    def _add_bound(self, coefficients: np.ndarray, upper: float) -> None:
        indices = np.flatnonzero(np.abs(coefficients) > 0.0).astype(np.int32)
        values = coefficients[indices].astype(float)
        status = self.highs.addRow(
            -highspy.kHighsInf,
            float(upper),
            len(indices),
            indices,
            values,
        )
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"failed to append lexicographic bound row: {status}")


def lexicographic_projection_repair_v2(
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
    """Return the V6 lexicographic projection using one native HiGHS model."""
    reference = np.asarray(reference_actions, dtype=float)
    frame = task_frame(task) if frame_override is None else _normalized_frame(frame_override)
    segment = frame.iloc[start_step:]
    if len(reference) != len(segment) or not np.isfinite(reference).all():
        raise ValueError("Projection reference does not match the finite segment.")
    solver = _NativeProjection(
        task,
        reference,
        battery,
        initial_soc_kwh=initial_soc_kwh,
        start_step=start_step,
        apply_event=apply_event,
        require_terminal=require_terminal,
        frame=frame,
    )
    solution, _ = solver.solve()
    return solution


def lexicographic_projection_repair_v2_with_trace(
    task: dict[str, Any],
    reference_actions: np.ndarray | list[float],
    battery: BatteryConfig,
    **kwargs: Any,
) -> tuple[ProjectionSolution, list[dict[str, Any]]]:
    """Internal runner hook returning the solution and auditable stage trace."""
    reference = np.asarray(reference_actions, dtype=float)
    frame_override = kwargs.pop("frame_override", None)
    frame = task_frame(task) if frame_override is None else _normalized_frame(frame_override)
    segment = frame.iloc[int(kwargs["start_step"]):]
    if len(reference) != len(segment) or not np.isfinite(reference).all():
        raise ValueError("Projection reference does not match the finite segment.")
    solver = _NativeProjection(task, reference, battery, frame=frame, **kwargs)
    return solver.solve()


def repair_contract_summary() -> dict[str, Any]:
    return {
        "method_version": REPAIR_METHOD_VERSION,
        "solver": "native HiGHS via highspy",
        "model_reuse": "one native MIP per row; objective changes and lexicographic bound rows",
        "solver_options": {
            "output_flag": False,
            "threads": 1,
            "mip_rel_gap": 0.0,
            "mip_abs_gap": 0.0,
            "mip_feasibility_tolerance": 1e-6,
        },
        "mathematical_contract_changed": False,
        "tolerances": {
            "lexicographic": LEXICOGRAPHIC_TOLERANCE,
            "energy": ENERGY_TOLERANCE_KWH,
            "power": POWER_TOLERANCE_KW,
        },
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
