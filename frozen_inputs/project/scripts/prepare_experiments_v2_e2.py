#!/usr/bin/env python3
"""Prepare 60 disjoint state-consequential E2 tasks without provider data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiments_v2.hashing import sha256_json  # noqa: E402

import prepare_experiments_v2_e1 as e1prep  # noqa: E402
from energy_agent_reliability.battery import (  # noqa: E402
    BatteryOverrides,
    simulate_dispatch,
)
from energy_agent_reliability.config import BatteryConfig, load_protocol  # noqa: E402
from energy_agent_reliability.online_gate_v6 import (  # noqa: E402
    economic_dispatch,
    limits_at,
    stage1_information_frame,
    task_frame,
)
from energy_agent_reliability.provenance import read_jsonl, sha256_file, write_jsonl  # noqa: E402

TASKS = ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl"
ADMISSION = ROOT / "experiments_v2/manifests/E2_ADMISSION_RESULTS.jsonl"
PARTIAL_TASKS = ROOT / "experiments_v2/manifests/E2_ELIGIBLE_TASKS.partial.jsonl"
PARTIAL_ADMISSION = ROOT / "experiments_v2/manifests/E2_ADMISSION_RESULTS.partial.jsonl"
REPORT = ROOT / "experiments_v2/reports/E2_PREPARATION_REPORT.json"
EVENTS = e1prep.EVENTS
SITES = e1prep.SITES
DIVERGENCES = (50.0, 75.0, 100.0)


def main() -> None:
    protocol = load_protocol(ROOT / "configs/frozen_protocol.yaml")
    battery = protocol.battery.model_copy(update={"terminal_soc_kwh": 325.0})
    sources = _source_pool()
    excluded_sources = _used_source_ids()
    rows = list(read_jsonl(PARTIAL_ADMISSION)) if PARTIAL_ADMISSION.exists() else []
    eligible = list(read_jsonl(PARTIAL_TASKS)) if PARTIAL_TASKS.exists() else []
    checked = {str(row["source_scenario_id"]) for row in rows}
    selected: list[dict[str, Any]] | None = None
    for rank, source in enumerate(sources):
        source_id = str(source["scenario_id"])
        if source_id in excluded_sources or source_id in checked:
            continue
        task = _transform(source, rank)
        row = _admit(task, battery, rank)
        rows.append(row)
        if row["admitted"]:
            eligible.append(task)
        if len(rows) % 20 == 0:
            write_jsonl(PARTIAL_ADMISSION, rows)
            write_jsonl(PARTIAL_TASKS, eligible)
            print(f"E2 admission: {len(rows)} checked, {len(eligible)} admitted", flush=True)
            selected = _select(eligible, rows, required=False)
            if selected is not None:
                break
    if selected is None:
        selected = _select(eligible, rows, required=True)
    assert selected is not None
    selected.sort(key=lambda task: str(task["scenario_id"]))
    write_jsonl(TASKS, selected)
    write_jsonl(ADMISSION, rows)
    PARTIAL_TASKS.unlink(missing_ok=True)
    PARTIAL_ADMISSION.unlink(missing_ok=True)
    report = _report(selected, rows, excluded_sources)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("E2 preparation failed")


def _source_pool() -> list[dict[str, Any]]:
    reserve = list(read_jsonl(e1prep.RESERVE))
    supplemental = e1prep._supplemental_sources(reserve)
    by_id = {str(task["scenario_id"]): task for task in [*reserve, *supplemental]}
    return [by_id[key] for key in sorted(by_id)]


def _used_source_ids() -> set[str]:
    paths = [
        ROOT / "scenarios/frozen_main_tasks.jsonl",
        ROOT / "scenarios/frozen_pilot_tasks.jsonl",
        ROOT / "scenarios/frozen_extension_tasks.jsonl",
        ROOT / "protocol/v7_revision/V7B_FROZEN_TASKS.jsonl",
        ROOT / "experiments_v2/manifests/P0_FROZEN_TASKS.jsonl",
        ROOT / "experiments_v2/manifests/E1_FROZEN_TASKS.jsonl",
    ]
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for task in read_jsonl(path):
            result.add(str(task.get("source_scenario_id", task["scenario_id"])))
    return result


def _transform(source: dict[str, Any], rank: int) -> dict[str, Any]:
    task = e1prep._transform(source)
    source_id = str(source["scenario_id"])
    divergence = DIVERGENCES[int(sha256_json({"source": source_id, "seed": 20260802})[:8], 16) % 3]
    task["scenario_id"] = f"v2e2_{sha256_json({'source': source_id, 'd': divergence})[:20]}"
    task["v2_task_kind"] = "e2_state_consequential"
    task["source_scenario_id"] = source_id
    task["v2_prefix_intervention"] = {
        "divergence_kwh": divergence,
        "assignment_seed": 20260802,
        "source_rank": rank,
        "future_oracle_used": False,
    }
    return task


def _admit(task: dict[str, Any], battery: BatteryConfig, rank: int) -> dict[str, Any]:
    base = {
        "scenario_id": task["scenario_id"],
        "source_scenario_id": task["source_scenario_id"],
        "event_family": task["event_family"],
        "location_id": task["location_id"],
        "season": task["season"],
        "difficulty": task["difficulty"],
        "reserve_rank": rank,
        "task_sha256": sha256_json(task),
        "provider_calls_performed": 0,
        "provider_outputs_read": False,
    }
    activation = int(task["visible_update"]["activation_step"])
    divergence = float(task["v2_prefix_intervention"]["divergence_kwh"])
    try:
        canonical_soc, prefix_action = _prefix(task, battery, activation, divergence)
        task["v2_prefix_intervention"].update(
            {
                "canonical_event_soc_kwh": canonical_soc,
                "stale_event_soc_kwh": battery.initial_soc_kwh,
                "prefix_action_kw": prefix_action,
            }
        )
        canonical = economic_dispatch(
            task, battery, initial_soc_kwh=canonical_soc, start_step=activation, apply_event=True
        )
        stale = economic_dispatch(
            task,
            battery,
            initial_soc_kwh=battery.initial_soc_kwh,
            start_step=activation,
            apply_event=True,
        )
        canonical_replay = _replay_suffix(task, canonical.action_kw, battery, canonical_soc)
        stale_replay = _replay_suffix(task, stale.action_kw, battery, canonical_soc)
        zero_replay = _replay_suffix(task, np.zeros(96 - activation), battery, canonical_soc)
        l1 = float(0.25 * np.abs(canonical.action_kw - stale.action_kw).sum())
        first_two_hour = float(np.max(np.abs(canonical.action_kw[:8] - stale.action_kw[:8])))
        stale_cost_worse = stale_replay["cost_usd"] - canonical_replay["cost_usd"] > 1.0
        strong = (
            not stale_replay["feasible"]
            or stale_replay["terminal_gap_kwh"] >= 10.0
            or first_two_hour >= 20.0
        )
        binding = _binding_slack(task, battery, canonical, canonical_replay["trace"], activation)
        forecast_changed = True
        if task["event_family"] == "forecast_revision":
            pre = economic_dispatch(
                task,
                battery,
                initial_soc_kwh=canonical_soc,
                start_step=activation,
                apply_event=True,
                frame_override=stage1_information_frame(task),
            )
            forecast_changed = float(0.25 * np.abs(canonical.action_kw - pre.action_kw).sum()) >= 25.0
        checks = {
            "canonical_suffix_feasible": canonical_replay["feasible"],
            "zero_suffix_infeasible": not zero_replay["feasible"],
            "stale_plan_infeasible_or_cost_worse": (not stale_replay["feasible"]) or stale_cost_worse,
            "deterministic_action_l1_ge_25": l1 >= 25.0,
            "strong_consequence": strong,
            "binding_slack_fraction_le_005": binding <= 0.05,
            "forecast_plan_changes_if_applicable": forecast_changed,
        }
    except Exception as exc:
        return {**base, "admitted": False, "rejection_codes": [f"solver:{type(exc).__name__}"]}
    return {
        **base,
        **checks,
        "admitted": all(checks.values()),
        "rejection_codes": [key for key, passed in checks.items() if not passed],
        "activation_step": activation,
        "divergence_kwh": divergence,
        "canonical_event_soc_kwh": canonical_soc,
        "prefix_action_kw": prefix_action,
        "canonical_cost_usd": canonical_replay["cost_usd"],
        "stale_cost_usd": stale_replay["cost_usd"],
        "stale_terminal_gap_kwh": stale_replay["terminal_gap_kwh"],
        "deterministic_action_l1_kwh": l1,
        "first_two_hours_action_delta_kw": first_two_hour,
        "binding_slack_fraction": binding,
    }


def _prefix(
    task: dict[str, Any], battery: BatteryConfig, activation: int, divergence: float
) -> tuple[float, float]:
    limits = limits_at(task, activation, battery, apply_event=True)
    candidates = [battery.initial_soc_kwh - divergence, battery.initial_soc_kwh + divergence]
    candidates.sort(key=lambda value: sha256_json({"id": task["scenario_id"], "soc": value}))
    for soc in candidates:
        if (
            limits["reserve_soc_kwh"] + 1e-6 <= soc <= limits["usable_capacity_kwh"] - 1e-6
            and abs(soc - battery.terminal_soc_kwh) >= 10.0
        ):
            delta = soc - battery.initial_soc_kwh
            if delta >= 0:
                action = delta / (battery.charge_efficiency * 0.25 * activation)
            else:
                action = delta * battery.discharge_efficiency / (0.25 * activation)
            if -battery.max_discharge_kw <= action <= battery.max_charge_kw:
                return float(soc), float(action)
    raise ValueError("no feasible frozen prefix intervention")


def _replay_suffix(
    task: dict[str, Any], actions: Any, battery: BatteryConfig, initial_soc: float
) -> dict[str, Any]:
    activation = int(task["visible_update"]["activation_step"])
    frame = task_frame(task).iloc[activation:].rename(columns={"price_usd_mwh": "price_per_mwh"})
    schedule = []
    for step in range(activation, 96):
        limits = limits_at(task, step, battery, apply_event=True)
        schedule.append(BatteryOverrides(**limits))
    result = simulate_dispatch(
        frame,
        np.asarray(actions, dtype=float),
        battery,
        overrides_by_step=schedule,
        initial_soc_kwh=initial_soc,
        require_terminal_soc=True,
        step_hours=0.25,
    )
    clipped = float(result.trace["violation_cost"].sum()) > 1e-8
    return {
        "feasible": bool(result.feasible and not clipped),
        "terminal_gap_kwh": result.terminal_soc_gap_kwh,
        "cost_usd": result.total_cost,
        "trace": result.trace,
    }


def _binding_slack(
    task: dict[str, Any], battery: BatteryConfig, solution: Any, trace: Any, activation: int
) -> float:
    values: list[float] = []
    for local, (_, row) in enumerate(trace.iterrows()):
        limits = limits_at(task, activation + local, battery, apply_event=True)
        values.extend(
            [
                max(limits["max_charge_kw"] - float(solution.charge_kw[local]), 0.0)
                / max(limits["max_charge_kw"], 1.0),
                max(limits["max_discharge_kw"] - float(solution.discharge_kw[local]), 0.0)
                / max(limits["max_discharge_kw"], 1.0),
                max(float(row["soc_kwh"]) - limits["reserve_soc_kwh"], 0.0)
                / max(limits["usable_capacity_kwh"], 1.0),
                max(limits["usable_capacity_kwh"] - float(row["soc_kwh"]), 0.0)
                / max(limits["usable_capacity_kwh"], 1.0),
            ]
        )
    return min(values)


def _select(
    tasks: list[dict[str, Any]], rows: list[dict[str, Any]], *, required: bool
) -> list[dict[str, Any]] | None:
    tasks = sorted(tasks, key=lambda task: str(task["scenario_id"]))
    if len(tasks) < 60:
        if required:
            raise SystemExit(f"E2 only admitted {len(tasks)} tasks")
        return None
    admitted = {str(row["scenario_id"]): row for row in rows if row["admitted"]}
    n = len(tasks)
    matrix: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def bounded(field: str, value: Any, low: int, high: int) -> None:
        matrix.append(np.asarray([1.0 if task[field] == value else 0.0 for task in tasks]))
        lower.append(float(low))
        upper.append(float(high))

    for event in EVENTS:
        bounded("event_family", event, 12, 12)
    for site in SITES:
        bounded("location_id", site, 12, 18)
    for season in ("winter", "spring", "summer", "autumn"):
        bounded("season", season, 12, 18)
    for difficulty in ("easy", "medium", "hard"):
        bounded("difficulty", difficulty, 18, 22)
    for divergence in DIVERGENCES:
        matrix.append(np.asarray([1.0 if task["v2_prefix_intervention"]["divergence_kwh"] == divergence else 0.0 for task in tasks]))
        lower.append(20.0)
        upper.append(20.0)
    objective = np.asarray([float(admitted[str(task["scenario_id"])]["reserve_rank"]) + i * 1e-6 for i, task in enumerate(tasks)])
    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(matrix), np.asarray(lower), np.asarray(upper)),
    )
    if not result.success or result.x is None:
        if required:
            raise SystemExit(f"E2 balanced selection infeasible: {result.message}")
        return None
    return [task for task, value in zip(tasks, result.x, strict=True) if value > 0.5]


def _report(
    tasks: list[dict[str, Any]], rows: list[dict[str, Any]], excluded: set[str]
) -> dict[str, Any]:
    event = Counter(str(task["event_family"]) for task in tasks)
    site = Counter(str(task["location_id"]) for task in tasks)
    season = Counter(str(task["season"]) for task in tasks)
    difficulty = Counter(str(task["difficulty"]) for task in tasks)
    divergence = Counter(float(task["v2_prefix_intervention"]["divergence_kwh"]) for task in tasks)
    checks = {
        "sixty_unique_tasks": len(tasks) == 60 and len({task["scenario_id"] for task in tasks}) == 60,
        "events_twelve_each": set(event) == set(EVENTS) and all(value == 12 for value in event.values()),
        "sites_near_balanced": set(site) == set(SITES) and all(12 <= value <= 18 for value in site.values()),
        "seasons_near_balanced": len(season) == 4 and all(12 <= value <= 18 for value in season.values()),
        "difficulty_near_balanced": len(difficulty) == 3 and all(18 <= value <= 22 for value in difficulty.values()),
        "divergence_twenty_each": all(divergence[value] == 20 for value in DIVERGENCES),
        "source_disjoint": not {str(task["source_scenario_id"]) for task in tasks}.intersection(excluded),
    }
    return {
        "schema_version": "experiments_v2_e2_preparation_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if all(checks.values()) else "HOLD",
        "checks": checks,
        "task_count": len(tasks),
        "event_counts": dict(event),
        "site_counts": dict(site),
        "season_counts": dict(season),
        "difficulty_counts": dict(difficulty),
        "divergence_counts": {str(key): value for key, value in divergence.items()},
        "admission_checked": len(rows),
        "admission_passed": sum(bool(row["admitted"]) for row in rows),
        "tasks_sha256": sha256_file(TASKS),
        "admission_sha256": sha256_file(ADMISSION),
        "provider_calls_performed": 0,
        "provider_outputs_read": False,
    }


if __name__ == "__main__":
    main()
