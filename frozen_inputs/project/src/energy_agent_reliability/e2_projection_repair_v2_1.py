"""V2.1 native-HiGHS projection with the audited final action tie-break.

The V6/V1 E2 runner has a three-stage contract.  The fourth stage below is
the deterministic weighted-action objective already present in the frozen V7
optimal-face audit.  It is kept in a separate version so that the old E2
evidence is never rewritten or silently reinterpreted.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import BatteryConfig
from .e2_projection_repair_v2 import (
    _NativeProjection,
)
from .online_gate_v6 import ProjectionSolution, _normalized_frame, task_frame

REPAIR_METHOD_VERSION = "e2_projection_repair_v2_1_exact_final_tiebreak"
FINAL_TIEBREAK_COST_TOLERANCE_USD = 1e-6


def _weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.linspace(1.0, 2.0, n, dtype=float)
    coefficients = np.concatenate((weights, weights[::-1], weights**2))
    return weights, coefficients


def _objective_hash(coefficients: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(coefficients, dtype=np.float64).tobytes()
    ).hexdigest()


class _NativeProjectionV21(_NativeProjection):
    """Reuse V2's canonical model and append the audited fourth stage."""

    def solve(self) -> tuple[ProjectionSolution, list[dict[str, Any]]]:
        distance_opt, values, trace1 = self._stage(
            "projection_distance", self.distance_expr, None
        )
        self._add_bound(
            self.distance_expr, distance_opt + self._lexicographic_tolerance
        )
        throughput_opt, values, trace2 = self._stage(
            "projection_throughput", self.throughput_expr, values
        )
        self._add_bound(
            self.throughput_expr, throughput_opt + self._energy_tolerance
        )
        physical_cost_opt, values, trace3 = self._stage(
            "projection_cost_tiebreak", self.cost_expr, values
        )
        self._add_bound(
            self.cost_expr,
            physical_cost_opt + FINAL_TIEBREAK_COST_TOLERANCE_USD,
        )
        weights, weighted_coefficients = _weights(self.n)
        weighted_objective = np.zeros(self.variable_count, dtype=float)
        for name, coeff in (
            ("charge", weights),
            ("discharge", weights[::-1]),
            ("curtailment", weights**2),
        ):
            start, _ = self.offsets[self.var_ids[name]]
            weighted_objective[start : start + self.n] = coeff
        weighted_opt, values, trace4 = self._stage(
            "projection_final_weighted_tiebreak", weighted_objective, values
        )
        trace4_dict = asdict(trace4)
        trace4_dict.update(
            {
                "physical_cost_optimum": float(physical_cost_opt),
                "physical_cost_face_tolerance": FINAL_TIEBREAK_COST_TOLERANCE_USD,
                "weighted_objective_value": float(weighted_opt),
                "weighted_objective_coefficient_hash": _objective_hash(
                    weighted_coefficients
                ),
                "incumbent_supplied": bool(
                    trace4.previous_stage_incumbent_supplied
                ),
            }
        )

        charge_start, _ = self.offsets[self.var_ids["charge"]]
        discharge_start, _ = self.offsets[self.var_ids["discharge"]]
        curtailment_start, _ = self.offsets[self.var_ids["curtailment"]]
        charge_value = np.maximum(values[charge_start : charge_start + self.n], 0.0)
        discharge_value = np.maximum(
            values[discharge_start : discharge_start + self.n], 0.0
        )
        curtailment_value = np.maximum(
            values[curtailment_start : curtailment_start + self.n], 0.0
        )
        overlap = np.minimum(charge_value, discharge_value)
        charge_value = np.maximum(charge_value - overlap, 0.0)
        discharge_value = np.maximum(discharge_value - overlap, 0.0)
        reporting_overlap = np.minimum(charge_value, discharge_value)
        action_value = charge_value - discharge_value
        signed_grid = self.load - self.pv + curtailment_value + action_value
        physical_cost = float(
            0.25 / 1000.0 * np.sum(self.price * signed_grid)
            + self.battery.degradation_cost_per_kwh
            * 0.25
            * np.sum(charge_value + discharge_value)
        )
        solution = ProjectionSolution(
            action_kw=action_value,
            charge_kw=charge_value,
            discharge_kw=discharge_value,
            curtailment_kw=curtailment_value,
            distance_kwh=float(
                0.25 * np.abs(action_value - self.reference).sum()
            ),
            throughput_kwh=float(
                0.25 * np.sum(charge_value + discharge_value)
            ),
            physical_cost_usd=physical_cost,
            solver_status="optimal",
            simultaneous_intervals=int(
                np.count_nonzero(reporting_overlap > self._power_tolerance)
            ),
            maximum_simultaneous_kw=float(reporting_overlap.max(initial=0.0)),
        )
        return solution, [asdict(trace1), asdict(trace2), asdict(trace3), trace4_dict]

    @property
    def _lexicographic_tolerance(self) -> float:
        from .online_gate_v6 import LEXICOGRAPHIC_TOLERANCE

        return LEXICOGRAPHIC_TOLERANCE

    @property
    def _energy_tolerance(self) -> float:
        from .online_gate_v6 import ENERGY_TOLERANCE_KWH

        return ENERGY_TOLERANCE_KWH

    @property
    def _power_tolerance(self) -> float:
        from .online_gate_v6 import POWER_TOLERANCE_KW

        return POWER_TOLERANCE_KW


def lexicographic_projection_repair_v2_1_with_trace(
    task: dict[str, Any],
    reference_actions: np.ndarray | list[float],
    battery: BatteryConfig,
    **kwargs: Any,
) -> tuple[ProjectionSolution, list[dict[str, Any]]]:
    reference = np.asarray(reference_actions, dtype=float)
    frame_override = kwargs.pop("frame_override", None)
    frame = task_frame(task) if frame_override is None else _normalized_frame(frame_override)
    segment = frame.iloc[int(kwargs["start_step"]):]
    if len(reference) != len(segment) or not np.isfinite(reference).all():
        raise ValueError("Projection reference does not match the finite segment.")
    solver = _NativeProjectionV21(
        task, reference, battery, frame=frame, **kwargs
    )
    return solver.solve()


def lexicographic_projection_repair_v2_1(
    task: dict[str, Any],
    reference_actions: np.ndarray | list[float],
    battery: BatteryConfig,
    **kwargs: Any,
) -> ProjectionSolution:
    solution, _ = lexicographic_projection_repair_v2_1_with_trace(
        task, reference_actions, battery, **kwargs
    )
    return solution


def repair_contract_summary() -> dict[str, Any]:
    return {
        "method_version": REPAIR_METHOD_VERSION,
        "solver": "native HiGHS via highspy",
        "model_reuse": "one native MIP per row; four objectives and bound rows",
        "objective_order": [
            "distance",
            "throughput",
            "physical_cost",
            "weighted_action_deterministic_tiebreak",
        ],
        "final_tiebreak_formula": "sum(w_t*charge_t + w_{n-1-t}*discharge_t + w_t^2*curtailment_t), w=linspace(1,2,n)",
        "physical_cost_face_tolerance_usd": FINAL_TIEBREAK_COST_TOLERANCE_USD,
        "mathematical_contract_changed": False,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
