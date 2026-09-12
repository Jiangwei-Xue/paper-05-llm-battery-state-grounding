#!/usr/bin/env python3
"""Prepare the 20-cluster E1 non-degenerate interface experiment."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.hashing import sha256_json  # noqa: E402

from energy_agent_reliability.config import BatteryConfig, load_protocol  # noqa: E402
from energy_agent_reliability.forecasting import revised_pv_forecast  # noqa: E402
from energy_agent_reliability.online_gate_v6 import (  # noqa: E402
    economic_dispatch,
    replay_actions,
    replay_two_stage,
    stage1_information_frame,
)
from energy_agent_reliability.provenance import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_jsonl,
)

RESERVE = ROOT / "scenarios/reserve_order.jsonl"
LEGACY_CANDIDATES = ROOT / "scenarios/candidates.jsonl"
LEGACY_ADMISSION = ROOT / "scenarios/admission_results.jsonl"
P0_TASKS = ROOT / "experiments_v2/manifests/P0_FROZEN_TASKS.jsonl"
TASKS = ROOT / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl"
ADMISSION = ROOT / "experiments_v2/manifests/E1_ADMISSION_RESULTS.jsonl"
ADMISSION_PARTIAL = ROOT / "experiments_v2/manifests/E1_ADMISSION_RESULTS.partial.jsonl"
TASKS_PARTIAL = ROOT / "experiments_v2/manifests/E1_ELIGIBLE_TASKS.partial.jsonl"
REPORT = ROOT / "experiments_v2/reports/E1_PREPARATION_REPORT.json"
EVENTS = (
    "forecast_revision",
    "battery_capacity_derating",
    "power_limit_derating",
    "export_limit_update",
    "reserve_commitment_update",
)
SITES = ("fresno", "los_angeles", "sacramento", "san_diego")


def main() -> None:
    protocol = load_protocol(ROOT / "configs/frozen_protocol.yaml")
    battery = protocol.battery.model_copy(update={"terminal_soc_kwh": 325.0})
    exposed = _provider_exposed_ids()
    p0_sources = {
        str(task["source_scenario_id"])
        for task in read_jsonl(P0_TASKS)
        if task.get("v2_task_kind") == "public_data"
    }
    reserve = list(read_jsonl(RESERVE))
    supplemental = _supplemental_sources(reserve)
    source_pool = [*reserve, *supplemental]
    admissions = list(read_jsonl(ADMISSION_PARTIAL)) if ADMISSION_PARTIAL.exists() else []
    admitted_tasks = list(read_jsonl(TASKS_PARTIAL)) if TASKS_PARTIAL.exists() else []
    checked_sources = {str(row["source_scenario_id"]) for row in admissions}
    selected: list[dict[str, Any]] | None = None
    for reserve_rank, source in enumerate(source_pool):
        source_id = str(source["scenario_id"])
        if source_id in exposed or source_id in p0_sources or source_id in checked_sources:
            continue
        task = _transform(source)
        row = _admit(task, battery, reserve_rank)
        admissions.append(row)
        if row["admitted"]:
            admitted_tasks.append(task)
        if len(admissions) % 20 == 0:
            write_jsonl(ADMISSION_PARTIAL, admissions)
            write_jsonl(TASKS_PARTIAL, admitted_tasks)
            print(
                f"E1 admission progress: {len(admissions)} checked, "
                f"{len(admitted_tasks)} admitted",
                flush=True,
            )
            selected = _select_balanced(admitted_tasks, admissions, required=False)
            if selected is not None:
                break
    if selected is None:
        selected = _select_balanced(admitted_tasks, admissions, required=True)
    assert selected is not None
    selected.sort(key=lambda task: str(task["scenario_id"]))
    write_jsonl(TASKS, selected)
    write_jsonl(ADMISSION, admissions)
    ADMISSION_PARTIAL.unlink(missing_ok=True)
    TASKS_PARTIAL.unlink(missing_ok=True)
    event_counts = Counter(str(task["event_family"]) for task in selected)
    site_counts = Counter(str(task["location_id"]) for task in selected)
    checks = {
        "twenty_unique_tasks": len(selected) == 20
        and len({task["scenario_id"] for task in selected}) == 20,
        "five_events_four_each": set(event_counts) == set(EVENTS)
        and all(count == 4 for count in event_counts.values()),
        "four_sites_near_balanced": set(site_counts) == set(SITES)
        and all(4 <= count <= 6 for count in site_counts.values()),
        "event_site_concentration_at_most_two": max(
            Counter(
                (str(task["event_family"]), str(task["location_id"]))
                for task in selected
            ).values()
        )
        <= 2,
        "four_seasons_near_balanced": all(
            4 <= count <= 6
            for count in Counter(str(task["season"]) for task in selected).values()
        )
        and len({str(task["season"]) for task in selected}) == 4,
        "difficulty_counts_six_or_seven": all(
            count in {6, 7}
            for count in Counter(str(task["difficulty"]) for task in selected).values()
        )
        and len({str(task["difficulty"]) for task in selected}) == 3,
        "source_disjoint_from_p0": not {
            str(task["source_scenario_id"]) for task in selected
        }.intersection(p0_sources),
        "source_provider_unexposed": not {
            str(task["source_scenario_id"]) for task in selected
        }.intersection(exposed),
    }
    report = {
        "schema_version": "experiments_v2_e1_preparation_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if all(checks.values()) else "HOLD",
        "checks": checks,
        "task_count": len(selected),
        "event_counts": dict(event_counts),
        "site_counts": dict(site_counts),
        "season_counts": dict(Counter(str(task["season"]) for task in selected)),
        "difficulty_counts": dict(Counter(str(task["difficulty"]) for task in selected)),
        "activation_step_range": [
            min(int(task["visible_update"]["activation_step"]) for task in selected),
            max(int(task["visible_update"]["activation_step"]) for task in selected),
        ],
        "selected_source_scenario_ids": [task["source_scenario_id"] for task in selected],
        "admission_candidates_checked": len(admissions),
        "admission_candidates_passed": len(admitted_tasks),
        "provider_calls_performed": 0,
        "provider_outputs_read": False,
        "reserve_order_sha256": sha256_file(RESERVE),
        "legacy_candidates_sha256": sha256_file(LEGACY_CANDIDATES),
        "legacy_admission_sha256": sha256_file(LEGACY_ADMISSION),
        "supplemental_source_count": len(supplemental),
        "tasks_sha256": sha256_file(TASKS),
        "admission_sha256": sha256_file(ADMISSION),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("E1 preparation failed")


def _select_balanced(
    tasks: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
    *,
    required: bool,
) -> list[dict[str, Any]] | None:
    """Select a deterministic marginally balanced set without outcome data."""
    admitted_rows = {
        str(row["scenario_id"]): row for row in admissions if row["admitted"]
    }
    tasks = sorted(tasks, key=lambda task: str(task["scenario_id"]))
    n = len(tasks)
    if n < 20:
        if required:
            raise SystemExit(f"E1 admission produced only {n} eligible tasks")
        return None
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_exact(predicate: Any, target: int) -> None:
        rows.append(np.asarray([1.0 if predicate(task) else 0.0 for task in tasks]))
        lower.append(float(target))
        upper.append(float(target))

    for event in EVENTS:
        add_exact(lambda task, event=event: task["event_family"] == event, 4)
    for site in SITES:
        vector = np.asarray(
            [1.0 if task["location_id"] == site else 0.0 for task in tasks]
        )
        rows.append(vector)
        lower.append(4.0)
        upper.append(6.0)
    for season in ("winter", "spring", "summer", "autumn"):
        vector = np.asarray([1.0 if task["season"] == season else 0.0 for task in tasks])
        rows.append(vector)
        lower.append(4.0)
        upper.append(6.0)
    for difficulty in ("easy", "medium", "hard"):
        vector = np.asarray(
            [1.0 if task["difficulty"] == difficulty else 0.0 for task in tasks]
        )
        rows.append(vector)
        lower.append(6.0)
        upper.append(7.0)
    for event in EVENTS:
        for site in SITES:
            vector = np.asarray(
                [
                    1.0
                    if task["event_family"] == event and task["location_id"] == site
                    else 0.0
                    for task in tasks
                ]
            )
            rows.append(vector)
            lower.append(0.0)
            upper.append(2.0)

    objective = np.asarray(
        [
            float(admitted_rows[str(task["scenario_id"])]["reserve_rank"])
            + index * 1e-6
            for index, task in enumerate(tasks)
        ]
    )
    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        if required:
            raise SystemExit(f"E1 balanced selection infeasible: {result.message}")
        return None
    selected = [task for task, value in zip(tasks, result.x, strict=True) if value > 0.5]
    if len(selected) != 20:
        raise SystemExit(f"E1 balanced selection returned {len(selected)} tasks")
    return selected


def _transform(source: dict[str, Any]) -> dict[str, Any]:
    task = copy.deepcopy(source)
    source_id = str(source["scenario_id"])
    task["source_scenario_id"] = source_id
    task["scenario_id"] = f"v2e1_{sha256_json({'source': source_id, 'experiment': 'E1'})[:20]}"
    task["v2_task_kind"] = "e1_public_data"
    task["v2_source_eligibility"] = source.get(
        "v2_source_eligibility", "legacy_admitted_reserve"
    )
    task["terminal_soc_target_kwh"] = 325.0
    task["model_visible_episode"]["objective"] = (
        "minimize operating cost subject to the visible battery envelope and a 325 kWh terminal SOC"
    )
    if task["event_family"] == "forecast_revision":
        _materialize_forecast_revision(task)
    return task


def _supplemental_sources(reserve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unselected legacy candidates whose only old failure was materiality.

    E1 changes the terminal obligation from 250 to 325 kWh and therefore must
    recompute event materiality under its own frozen objective.  All invariant
    legacy checks still have to pass; candidates with any other old failure are
    ineligible.
    """
    reserve_ids = {str(task["scenario_id"]) for task in reserve}
    candidates = {str(task["scenario_id"]): task for task in read_jsonl(LEGACY_CANDIDATES)}
    admission = {str(row["scenario_id"]): row for row in read_jsonl(LEGACY_ADMISSION)}
    supplemental: list[dict[str, Any]] = []
    for scenario_id in sorted(candidates):
        if scenario_id in reserve_ids:
            continue
        row = admission[scenario_id]
        failed = set(row.get("failed_checks", []))
        if failed != {"event_materiality"}:
            continue
        checks = dict(row.get("checks", {}))
        if any(not bool(value) for key, value in checks.items() if key != "event_materiality"):
            continue
        task = copy.deepcopy(candidates[scenario_id])
        task["v2_source_eligibility"] = (
            "legacy_all_invariant_checks_passed_materiality_recomputed_for_325kwh"
        )
        supplemental.append(task)
    return supplemental


def _materialize_forecast_revision(task: dict[str, Any]) -> None:
    """Materialize the frozen causal revision omitted by the legacy task surface.

    The original candidate generator stored the revision method and weight but
    copied the Stage 1 forecast into Stage 2.  V2 derives the revised suffix
    directly from the immutable processed table using the already-frozen
    ``latest_observation_clearsky_index_persistence`` rule.  No future realized
    PV value enters the forecast calculation.
    """
    source_path = ROOT / str(task["source_data"])
    source_sha256 = sha256_file(source_path)
    frame = pd.read_parquet(source_path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    start = pd.Timestamp(task["start_timestamp"])
    day = frame.loc[
        (frame["timestamp_utc"] >= start)
        & (frame["timestamp_utc"] < start + pd.Timedelta(hours=24))
    ].copy()
    if len(day) != 96:
        raise ValueError(f"Expected 96 source rows for {task['scenario_id']}; found {len(day)}")
    day = day.set_index("timestamp_utc")
    activation = int(task["visible_update"]["activation_step"])
    weight = float(task["visible_update"]["forecast_revision_blend_weight"])
    revised = revised_pv_forecast(day, activation, weight, ac_capacity_kw=400.0)
    suffix = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    if len(suffix) != 96 - activation:
        raise ValueError(f"Unexpected Stage 2 suffix length for {task['scenario_id']}")
    for row, value in zip(suffix, revised.iloc[activation:], strict=True):
        row["pv_forecast_kw"] = round(float(value), 6)
    task["v2_forecast_revision_materialization"] = {
        "method": "latest_observation_clearsky_index_persistence",
        "input_source": str(task["source_data"]),
        "input_sha256": source_sha256,
        "activation_step": activation,
        "blend_weight": weight,
        "future_realized_pv_used": False,
    }


def _admit(task: dict[str, Any], battery: BatteryConfig, reserve_rank: int) -> dict[str, Any]:
    base = {
        "scenario_id": task["scenario_id"],
        "source_scenario_id": task["source_scenario_id"],
        "event_family": task["event_family"],
        "location_id": task["location_id"],
        "season": task["season"],
        "difficulty": task["difficulty"],
        "reserve_rank": reserve_rank,
        "task_sha256": sha256_json(task),
        "provider_calls_performed": 0,
        "provider_outputs_read": False,
    }
    activation = int(task["visible_update"]["activation_step"])
    try:
        initial_frame = stage1_information_frame(task)
        stage1 = economic_dispatch(
            task,
            battery,
            initial_soc_kwh=battery.initial_soc_kwh,
            start_step=0,
            apply_event=False,
            frame_override=initial_frame,
        )
        event_soc = _event_soc(battery, stage1, activation)
        event_plan = economic_dispatch(
            task,
            battery,
            initial_soc_kwh=event_soc,
            start_step=activation,
            apply_event=True,
        )
        stale_plan = economic_dispatch(
            task,
            battery,
            initial_soc_kwh=event_soc,
            start_step=activation,
            apply_event=False,
            frame_override=initial_frame,
        )
        canonical = replay_two_stage(task, stage1.action_kw, event_plan.action_kw, battery)
        stale = replay_two_stage(task, stage1.action_kw, stale_plan.action_kw, battery)
        zero = replay_actions(task, np.zeros(96), battery)
    except Exception as exc:
        return {**base, "admitted": False, "rejection_codes": [f"solver:{type(exc).__name__}"]}
    combined = np.concatenate([stage1.action_kw[:activation], event_plan.action_kw])
    action_energy = float(0.25 * np.abs(combined).sum())
    event_stale_l1 = float(0.25 * np.abs(event_plan.action_kw - stale_plan.action_kw).sum())
    zero_penalized = zero.total_cost_usd + battery.violation_penalty_per_kwh * zero.terminal_soc_gap_kwh
    economic_space = zero_penalized - canonical.total_cost_usd
    economic_threshold = max(5.0, 0.02 * abs(canonical.total_cost_usd))
    checks = {
        "canonical_feasible": bool(canonical.feasible),
        "zero_action_infeasible": not bool(zero.feasible),
        "terminal_target_325": battery.terminal_soc_kwh == 325.0,
        "canonical_action_energy_ge_50": action_energy >= 50.0,
        "event_stale_l1_ge_25": event_stale_l1 >= 25.0,
        "economic_space_identifiable": economic_space >= economic_threshold,
        "forecast_plan_changes_if_applicable": task["event_family"] != "forecast_revision"
        or event_stale_l1 >= 25.0,
    }
    return {
        **base,
        **checks,
        "admitted": all(checks.values()),
        "rejection_codes": [key for key, passed in checks.items() if not passed],
        "activation_step": activation,
        "event_soc_kwh": event_soc,
        "canonical_action_energy_kwh": action_energy,
        "event_stale_l1_kwh": event_stale_l1,
        "stale_replay_feasible": stale.feasible,
        "zero_penalized_cost_usd": zero_penalized,
        "canonical_cost_usd": canonical.total_cost_usd,
        "zero_minus_canonical_cost_usd": economic_space,
        "economic_space_threshold_usd": economic_threshold,
    }


def _event_soc(battery: BatteryConfig, solution: Any, activation: int) -> float:
    return float(
        battery.initial_soc_kwh
        + 0.25
        * (
            solution.charge_kw[:activation].sum() * battery.charge_efficiency
            - solution.discharge_kw[:activation].sum() / battery.discharge_efficiency
        )
    )


def _provider_exposed_ids() -> set[str]:
    result = subprocess.run(
        ["rg", "-o", "--no-ignore", "--no-filename", r'"scenario_id"\s*:\s*"[^"]+"', "runs", "-g", "*.json", "-g", "*.jsonl"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {str(json.loads(line.split(":", 1)[1].strip())) for line in result.stdout.splitlines()}


if __name__ == "__main__":
    main()
