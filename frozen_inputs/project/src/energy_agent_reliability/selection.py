"""Seeded, stratified selection that preserves a full audit trace."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from .config import FrozenProtocol
from .provenance import write_jsonl


def select_and_freeze(
    protocol: FrozenProtocol,
    candidates: list[dict[str, Any]],
    admissions: list[dict[str, Any]],
    frozen_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    admitted_ids = {item["scenario_id"] for item in admissions if item.get("admitted", item.get("all_passed"))}
    eligible = [item for item in candidates if item["scenario_id"] in admitted_ids]
    _pool_level_quota_gate(protocol, eligible)
    buckets, trace = _stratified_shuffle(eligible, protocol)
    ordered = [candidate for key in sorted(buckets) for candidate in buckets[key]]
    main, remaining = _quota_select(ordered, _main_quotas(protocol), protocol.selection.main_pool_size, "main", trace)
    pilot, reserve = _quota_select(
        remaining, _pilot_quotas(protocol), protocol.selection.pilot_pool_size, "pilot", trace
    )
    _validate_pool_quotas(protocol, main, pilot, reserve)
    if len(main) != protocol.selection.main_pool_size or len(pilot) != protocol.selection.pilot_pool_size:
        raise ValueError("Selection did not satisfy frozen pool sizes.")
    destination = Path(frozen_dir)
    write_jsonl(destination / "selection_trace.jsonl", trace)
    write_jsonl(destination / "frozen_main_tasks.jsonl", main)
    write_jsonl(destination / "frozen_pilot_tasks.jsonl", pilot)
    write_jsonl(destination / "reserve_order.jsonl", reserve)
    return main, pilot, reserve


def _stratified_shuffle(
    candidates: list[dict[str, Any]], protocol: FrozenProtocol
) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: item["scenario_id"]):
        key = tuple(str(candidate[field]) for field in protocol.selection.strata_fields)
        grouped[key].append(candidate)
    rng = random.Random(protocol.selection.selection_seed)
    trace: list[dict[str, Any]] = []
    for key in sorted(grouped):
        # Required ordering: stable ID sort happens above; only then does this stratum shuffle occur.
        rng.shuffle(grouped[key])
        for rank, candidate in enumerate(grouped[key]):
            trace.append(
                {
                    "scenario_id": candidate["scenario_id"],
                    "stratum": dict(zip(protocol.selection.strata_fields, key, strict=True)),
                    "sort_then_shuffle_rank": rank,
                    "selection_seed": protocol.selection.selection_seed,
                    "event_generation_seed": protocol.selection.event_generation_seed,
                    "bootstrap_seed": protocol.selection.bootstrap_seed,
                    "selection_status": "unassigned",
                }
            )
    return dict(grouped), trace


def _quota_select(
    ordered: list[dict[str, Any]],
    quotas: dict[str, Counter[str]],
    size: int,
    pool: str,
    trace: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(ordered) < size:
        raise ValueError(f"Not enough admitted candidates remain for {pool}.")
    x = cp.Variable(len(ordered), boolean=True)
    constraints: list[Any] = [cp.sum(x) == size]
    for field, counter in quotas.items():
        for value, needed in counter.items():
            mask = np.asarray([1.0 if str(candidate[field]) == value else 0.0 for candidate in ordered])
            constraints.append(mask @ x == needed)
    objective_weights = np.arange(1, len(ordered) + 1, dtype=float)
    problem = cp.Problem(cp.Minimize(objective_weights @ x), constraints)
    problem.solve(solver=cp.HIGHS, warm_start=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        raise ValueError(f"No admitted candidate can satisfy remaining {pool} quotas.")
    selected_indices = [index for index, value in enumerate(np.asarray(x.value).ravel()) if value > 0.5]
    if len(selected_indices) != size:
        raise ValueError(f"{pool} quota solver returned {len(selected_indices)} candidates, expected {size}.")
    selected_index_set = set(selected_indices)
    selected = [candidate for index, candidate in enumerate(ordered) if index in selected_index_set]
    remaining = [candidate for index, candidate in enumerate(ordered) if index not in selected_index_set]
    for order, candidate in enumerate(selected):
        _mark_trace(trace, candidate["scenario_id"], pool, order)
    return selected, remaining


def _main_quotas(protocol: FrozenProtocol) -> dict[str, Counter[str]]:
    return {
        "event_family": Counter({family: 8 for family in protocol.scenario.event_families}),
        "location_id": Counter({location.location_id: 10 for location in protocol.locations}),
        "season": Counter({season: 10 for season in protocol.scenario.seasons}),
        "difficulty": Counter({"easy": 10, "medium": 15, "hard": 15}),
    }


def _pilot_quotas(protocol: FrozenProtocol) -> dict[str, Counter[str]]:
    locations = [location.location_id for location in protocol.locations]
    seasons = list(protocol.scenario.seasons)
    return {
        "event_family": Counter({family: 2 for family in protocol.scenario.event_families}),
        "location_id": Counter({locations[0]: 2, locations[1]: 3, locations[2]: 3, locations[3]: 2}),
        "season": Counter({seasons[0]: 2, seasons[1]: 2, seasons[2]: 3, seasons[3]: 3}),
        "difficulty": Counter({"easy": 4, "medium": 3, "hard": 3}),
    }


def _pool_level_quota_gate(protocol: FrozenProtocol, eligible: list[dict[str, Any]]) -> None:
    if len(eligible) < protocol.selection.main_pool_size + protocol.selection.pilot_pool_size:
        raise ValueError("Eligible candidate pool is smaller than pilot plus main.")
    _require_dimension(protocol, eligible, "event_family", protocol.scenario.event_families)
    _require_dimension(protocol, eligible, "location_id", [location.location_id for location in protocol.locations])
    _require_dimension(protocol, eligible, "season", list(protocol.scenario.seasons))
    _require_dimension(protocol, eligible, "difficulty", protocol.scenario.difficulties)
    reserve_minimum = max(1, len(eligible) - protocol.selection.main_pool_size - protocol.selection.pilot_pool_size)
    if reserve_minimum < 1:
        raise ValueError("Reserve order would be empty.")


def _require_dimension(
    protocol: FrozenProtocol, eligible: list[dict[str, Any]], field: str, values: list[str]
) -> None:
    counts = Counter(str(item[field]) for item in eligible)
    if field == "difficulty":
        required_by_value = {"easy": 10, "medium": 15, "hard": 15}
    else:
        required = {
            "event_family": protocol.selection.main_pool_size // len(protocol.scenario.event_families),
            "location_id": protocol.selection.main_pool_size // len(protocol.locations),
            "season": protocol.selection.main_pool_size // len(protocol.scenario.seasons),
        }[field]
        required_by_value = {value: required for value in values}
    for value in values:
        if counts[value] < required_by_value[value]:
            raise ValueError(f"Eligible candidate pool cannot satisfy main quota for {field}={value}.")


def _validate_pool_quotas(
    protocol: FrozenProtocol,
    main: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
    reserve: list[dict[str, Any]],
) -> None:
    main_ids = {item["scenario_id"] for item in main}
    pilot_ids = {item["scenario_id"] for item in pilot}
    reserve_ids = {item["scenario_id"] for item in reserve}
    if not main_ids.isdisjoint(pilot_ids) or not main_ids.isdisjoint(reserve_ids) or not pilot_ids.isdisjoint(reserve_ids):
        raise ValueError("Pilot, main, and reserve must be mutually disjoint.")
    if Counter(item["event_family"] for item in main) != {family: 8 for family in protocol.scenario.event_families}:
        raise ValueError("Main event-family quota is not satisfied.")
    if Counter(item["location_id"] for item in main) != {location.location_id: 10 for location in protocol.locations}:
        raise ValueError("Main site quota is not satisfied.")
    if Counter(item["season"] for item in main) != {season: 10 for season in protocol.scenario.seasons}:
        raise ValueError("Main season quota is not satisfied.")
    if Counter(item["difficulty"] for item in main) != {"easy": 10, "medium": 15, "hard": 15}:
        raise ValueError("Main difficulty quota is not satisfied.")
    if Counter(item["event_family"] for item in pilot) != {family: 2 for family in protocol.scenario.event_families}:
        raise ValueError("Pilot event-family quota is not satisfied.")
    if len({item["location_id"] for item in pilot}) < 3:
        raise ValueError("Pilot must cover at least three sites.")
    if len({item["season"] for item in pilot}) != len(protocol.scenario.seasons):
        raise ValueError("Pilot must cover all seasons.")
    if set(Counter(item["difficulty"] for item in pilot)) != {"easy", "medium", "hard"}:
        raise ValueError("Pilot must include every difficulty.")


def distribution(records: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, int]]:
    return {field: dict(Counter(str(item[field]) for item in records)) for field in fields}


def _mark_trace(trace: list[dict[str, Any]], scenario_id: str, pool: str, order: int) -> None:
    for row in trace:
        if row["scenario_id"] == scenario_id:
            row["selection_status"] = pool
            row["pool_order"] = order
            break
