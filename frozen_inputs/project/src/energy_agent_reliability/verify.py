"""Independent checks for the frozen pre-execution artifact set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import FrozenProtocol
from .provenance import read_jsonl, read_sha256sum_manifest, sha256_file
from .state_methods import STATE_METHODS


def verify_preexecution_freeze(protocol: FrozenProtocol, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    scenarios = root / "scenarios"
    frozen = root / "data" / "frozen"
    candidates = list(read_jsonl(scenarios / "candidates.jsonl"))
    admissions = list(read_jsonl(scenarios / "admission_results.jsonl"))
    main = list(read_jsonl(scenarios / "frozen_main_tasks.jsonl"))
    pilot = list(read_jsonl(scenarios / "frozen_pilot_tasks.jsonl"))
    reserve = list(read_jsonl(scenarios / "reserve_order.jsonl"))
    checks = {
        "stable_candidate_ids": _stable_candidate_ids(candidates),
        "admission_records_complete": _admission_records_complete(candidates, admissions),
        "selected_tasks_admitted": _selected_tasks_admitted([*main, *pilot, *reserve], admissions),
        "pilot_main_reserve_mutually_disjoint": _mutually_disjoint(main, pilot, reserve),
        "main_quota_satisfied": _main_quota(protocol, main),
        "pilot_quota_satisfied": _pilot_quota(protocol, pilot),
        "reserve_order_complete": _reserve_order_complete(main, pilot, reserve, admissions),
        "seeds_match_protocol": _seeds_match(protocol, scenarios / "selection_trace.jsonl"),
        "candidate_manifest_not_modified_after_selection": _candidate_hash_recorded(root),
        "scorer_oracle_not_model_visible": _no_model_visible_oracle(candidates),
        "two_stage_task_surfaces_complete": _two_stage_surfaces_complete(candidates),
        "state_methods_defined": _state_methods_defined(candidates),
        "typed_state_schema_fixed": (root / "protocol" / "STATE_METHODS_SPEC_V1.yaml").exists(),
        "hashes_match": _hashes_match(root),
        "git_worktree_status_recorded": _git_status_recorded(root),
        "no_model_call_records": _no_model_call_records(root, frozen),
        "no_model_outputs_or_scores": _no_model_outputs_or_scores(root),
        "pool_sizes_match": len(main) == protocol.selection.main_pool_size
        and len(pilot) == protocol.selection.pilot_pool_size,
    }
    admitted = [item for item in admissions if item.get("admitted", item.get("all_passed"))]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "candidates": len(candidates),
            "admitted": len(admitted),
            "rejected": len(admissions) - len(admitted),
            "main": len(main),
            "pilot": len(pilot),
            "reserve": len(reserve),
        },
    }


def _ids(records: list[dict[str, Any]]) -> set[str]:
    return {record["scenario_id"] for record in records}


def _stable_candidate_ids(candidates: list[dict[str, Any]]) -> bool:
    ids = [item.get("scenario_id") for item in candidates]
    if not all(isinstance(item, str) for item in ids):
        return False
    scenario_ids = [str(item) for item in ids]
    return bool(
        scenario_ids
        and len(scenario_ids) == len(set(scenario_ids))
        and all(item.startswith("scn_") for item in scenario_ids)
        and scenario_ids == sorted(scenario_ids)
    )


def _admission_records_complete(candidates: list[dict[str, Any]], admissions: list[dict[str, Any]]) -> bool:
    candidate_ids = _ids(candidates)
    if candidate_ids != _ids(admissions):
        return False
    required_checks = {
        "data_completeness",
        "timestamp_alignment",
        "simulator_validity",
        "baseline_feasibility",
        "post_event_feasibility",
        "event_materiality",
        "energy_conservation",
        "oracle_consistency",
        "scorer_ceiling",
        "stale_state_negative_control_rejection",
        "empty_state_rejection",
        "carry_everything_rejection",
        "oracle_separation",
        "no_answer_leakage",
        "no_future_data_leakage",
    }
    return all(required_checks.issubset(set(item.get("checks", {}))) for item in admissions)


def _selected_tasks_admitted(records: list[dict[str, Any]], admissions: list[dict[str, Any]]) -> bool:
    admitted_ids = {item["scenario_id"] for item in admissions if item.get("admitted", item.get("all_passed"))}
    return _ids(records).issubset(admitted_ids)


def _mutually_disjoint(
    main: list[dict[str, Any]], pilot: list[dict[str, Any]], reserve: list[dict[str, Any]]
) -> bool:
    main_ids = _ids(main)
    pilot_ids = _ids(pilot)
    reserve_ids = _ids(reserve)
    return main_ids.isdisjoint(pilot_ids) and main_ids.isdisjoint(reserve_ids) and pilot_ids.isdisjoint(reserve_ids)


def _main_quota(protocol: FrozenProtocol, main: list[dict[str, Any]]) -> bool:
    return bool(
        len(main) == protocol.selection.main_pool_size
        and Counter(task["event_family"] for task in main)
        == {family: 8 for family in protocol.scenario.event_families}
        and Counter(task["location_id"] for task in main)
        == {location.location_id: 10 for location in protocol.locations}
        and Counter(task["season"] for task in main) == {season: 10 for season in protocol.scenario.seasons}
        and Counter(task["difficulty"] for task in main) == {"easy": 10, "medium": 15, "hard": 15}
    )


def _pilot_quota(protocol: FrozenProtocol, pilot: list[dict[str, Any]]) -> bool:
    return bool(
        len(pilot) == protocol.selection.pilot_pool_size
        and Counter(task["event_family"] for task in pilot)
        == {family: 2 for family in protocol.scenario.event_families}
        and len({task["location_id"] for task in pilot}) >= 3
        and len({task["season"] for task in pilot}) == len(protocol.scenario.seasons)
        and {task["difficulty"] for task in pilot} == {"easy", "medium", "hard"}
    )


def _reserve_order_complete(
    main: list[dict[str, Any]], pilot: list[dict[str, Any]], reserve: list[dict[str, Any]], admissions: list[dict[str, Any]]
) -> bool:
    admitted_ids = {item["scenario_id"] for item in admissions if item.get("admitted", item.get("all_passed"))}
    return _ids([*main, *pilot, *reserve]) == admitted_ids


def _seeds_match(protocol: FrozenProtocol, trace_path: Path) -> bool:
    trace = list(read_jsonl(trace_path))
    return bool(
        trace
        and all(item.get("selection_seed") == protocol.selection.selection_seed for item in trace)
        and all(item.get("event_generation_seed") == protocol.selection.event_generation_seed for item in trace)
        and all(item.get("bootstrap_seed") == protocol.selection.bootstrap_seed for item in trace)
    )


def _candidate_hash_recorded(root: Path) -> bool:
    report_path = root / "reports" / "FREEZE_REPORT.json"
    candidate_path = root / "scenarios" / "candidates.jsonl"
    if not report_path.exists() or not candidate_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(report.get("candidate_manifest_sha256") == sha256_file(candidate_path))


def _no_model_visible_oracle(candidates: list[dict[str, Any]]) -> bool:
    banned = {
        "scorer_oracle",
        "pre_event_capacity_kwh",
        "pre_event_max_charge_kw",
        "pre_event_max_discharge_kw",
        "pre_event_export_limit_kw",
        "pre_event_reserve_soc_kwh",
    }
    for candidate in candidates:
        payload = json.dumps(_drop_policy_fields(candidate), sort_keys=True).lower()
        if any(token in payload for token in banned):
            return False
    return True


def _two_stage_surfaces_complete(candidates: list[dict[str, Any]]) -> bool:
    for candidate in candidates:
        surface = candidate.get("model_visible_episode")
        if not isinstance(surface, dict):
            return False
        stage1 = surface.get("stage_1")
        stage2 = surface.get("stage_2")
        if not isinstance(stage1, dict) or not isinstance(stage2, dict):
            return False
        if stage1.get("required_dispatch_plan_steps") != 96:
            return False
        activation = int(candidate["visible_update"]["activation_step"])
        if stage2.get("required_dispatch_plan_steps") != 96 - activation:
            return False
    return True


def _state_methods_defined(candidates: list[dict[str, Any]]) -> bool:
    expected = set(STATE_METHODS)
    return all(set(item.get("state_methods", {})) == expected for item in candidates)


def _drop_policy_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_policy_fields(item)
            for key, item in value.items()
            if key not in {"forbidden_information", "forbidden_tokens", "oracle_ref"}
        }
    if isinstance(value, list):
        return [_drop_policy_fields(item) for item in value]
    return value


def _strata_match(protocol: FrozenProtocol, candidates: list[dict[str, Any]], main: list[dict[str, Any]]) -> bool:
    fields = protocol.selection.strata_fields
    candidate_counts = Counter(tuple(record[field] for field in fields) for record in candidates)
    expected_strata = (
        len(protocol.scenario.event_families)
        * len(protocol.locations)
        * len(protocol.scenario.seasons)
        * len(protocol.scenario.difficulties)
    )
    if len(candidate_counts) != expected_strata or set(candidate_counts.values()) != {protocol.scenario.candidates_per_cell}:
        return False
    event_counts = Counter(task["event_family"] for task in main)
    location_counts = Counter(task["location_id"] for task in main)
    season_counts = Counter(task["season"] for task in main)
    difficulty_counts = Counter(task["difficulty"] for task in main)
    return bool(
        set(event_counts.values()) == {8}
        and set(location_counts.values()) == {10}
        and set(season_counts.values()) == {10}
        and max(difficulty_counts.values()) - min(difficulty_counts.values()) <= 1
    )


def _hashes_match(root: Path) -> bool:
    manifest_path = root / "hashes" / "preexecution.sha256"
    if not manifest_path.exists():
        return False
    manifest = read_sha256sum_manifest(manifest_path)
    for relative, expected in manifest.items():
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            return False
    return "hashes/preexecution.sha256" not in manifest


def _git_status_recorded(root: Path) -> bool:
    report_path = root / "reports" / "FREEZE_REPORT.json"
    if not report_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return isinstance(report.get("git_commit"), str) and "git_status_short" in report


def _no_model_call_records(root: Path, frozen: Path) -> bool:
    attestation_path = frozen / "preexecution_attestation.json"
    if not attestation_path.exists():
        return False
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("model_calls") != 0 or attestation.get("llm_api_calls") != 0:
        return False
    log_candidates = list((root / "logs").glob("*model*call*.jsonl")) + list((root / "logs").glob("*llm*.jsonl"))
    return all(path.stat().st_size == 0 for path in log_candidates)


def _no_model_outputs_or_scores(root: Path) -> bool:
    forbidden_roots = [root / "artifacts", root / "reports", root / "scenarios"]
    patterns = ["*model_output*", "*model-result*", "*model_score*", "*llm_output*", "*raw_response*"]
    for base in forbidden_roots:
        for pattern in patterns:
            if any(path.is_file() for path in base.rglob(pattern)):
                return False
    return True
