#!/usr/bin/env python3
"""Run the excluded 128-call P0 interface pilot with direct provider routes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.actions import action_energy_kwh, is_nontrivial  # noqa: E402
from experiments_v2.hashing import sha256_json, sha256_text  # noqa: E402
from experiments_v2.p0 import (  # noqa: E402
    build_payload,
    canonical_event_state,
    canonical_initial_state,
    compile_prompt,
    parse_output,
    parsed_payload,
)

from energy_agent_reliability.config import BatteryConfig, load_protocol  # noqa: E402
from energy_agent_reliability.llm.models import ProviderRequest  # noqa: E402
from energy_agent_reliability.llm.protocol_v2 import ProviderSpecV2  # noqa: E402
from energy_agent_reliability.llm.runtime_v2 import complete_stage_v2  # noqa: E402
from energy_agent_reliability.online_gate_v6 import (  # noqa: E402
    economic_dispatch,
    lexicographic_projection,
    replay_two_stage,
    stage1_information_frame,
)
from energy_agent_reliability.provenance import read_jsonl, sha256_file  # noqa: E402

TASKS = ROOT / "experiments_v2/manifests/P0_FROZEN_TASKS.jsonl"
PLAN = ROOT / "experiments_v2/manifests/P0_RUN_PLAN.jsonl"
MANIFEST = ROOT / "experiments_v2/manifests/P0_PRE_RUN_MANIFEST.json"
MODELS = ROOT / "experiments_v2/configs/models.yaml"
RUN_ROOT = ROOT / "runs/experiments_v2/p0"
PRE_RUN_TAG = "energybench-v2-p0-pre-run"
EXPERIMENT_LABEL = "P0"
EXPECTED_EPISODES = 64
EXPECTED_CALLS = 128
IDENTITY_SCHEMA = "experiments_v2_p0_run_identity_v1"
RECORD_SCHEMA = "experiments_v2_p0_episode_v1"
EVIDENCE_TIER = "excluded_interface_engineering_pilot"
EXCLUDED_FROM_CLAIMS = True
EXECUTION_SCHEMA = "experiments_v2_p0_execution_summary_v1"
COMPLETION_STATUS = "P0_COMPLETED"
HOLD_FILENAME = "P0_HOLD.json"
SUMMARY_FILENAME = "P0_EXECUTION_SUMMARY.json"
HASH_FILENAME = "P0_HASH_MANIFEST.json"
HASH_SCHEMA = "experiments_v2_p0_hash_manifest_v1"
ENABLE_INLINE_PROJECTION = True
ALLOWED_DIRTY = {
    "paper/DRAFT_BUILD_REPORT.json",
    "paper/figures/fig1_architecture.pdf",
    "paper/figures/fig2_selection_flow.pdf",
    "paper/figures/fig3_state_economic_crosstab.pdf",
    "paper/figures/fig4_factorial_heatmaps.pdf",
    "paper/figures/fig5_factorial_forest.pdf",
    "paper/figures/fig6_repeatability.pdf",
    "paper/figures/fig7_shield_repair.pdf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency != 5:
        raise SystemExit(f"{EXPERIMENT_LABEL} freezes concurrency at five")
    _verify_manifest()
    protocol = load_protocol(ROOT / "configs/frozen_protocol.yaml")
    battery = protocol.battery.model_copy(update={"terminal_soc_kwh": 325.0})
    tasks = list(read_jsonl(TASKS))
    plan = list(read_jsonl(PLAN))
    configs = _provider_specs()
    contexts = _contexts(tasks, battery)
    prepared = _prepare_requests(plan, tasks, contexts, configs)
    run_dir = RUN_ROOT / args.run_label
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "records").mkdir(exist_ok=True)
    _write_json_atomic(
        run_dir / "PRE_RUN_IDENTITY.json",
        {
            "schema_version": IDENTITY_SCHEMA,
            "created_utc": _utc_now(),
            "source_release_identity": "public-scientific-file-manifest",
            "plan_sha256": sha256_file(PLAN),
            "frozen_manifest_sha256": sha256_file(MANIFEST),
            "planned_logical_episodes": EXPECTED_EPISODES,
            "planned_primary_calls": EXPECTED_CALLS,
            "httpx_trust_env": False,
            "concurrency": 5,
            "api_key_values_recorded": False,
        },
    )
    _write_jsonl_atomic(run_dir / "PLANNED_CALL_LEDGER.jsonl", _call_ledger(prepared))
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "plan_locked",
                    "logical_episodes": len(prepared),
                    "planned_primary_calls": 2 * len(prepared),
                    "plan_sha256": sha256_file(PLAN),
                },
                sort_keys=True,
            )
        )
        return
    keys = _load_keys(configs)
    status = asyncio.run(
        _run(prepared, battery, configs, keys, run_dir, concurrency=args.concurrency)
    )
    _write_reports(run_dir, plan, status)
    if status["status"] != "completed":
        raise SystemExit(f"{EXPERIMENT_LABEL} entered HOLD")


def _provider_specs() -> dict[str, ProviderSpecV2]:
    raw = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["conditions"]
    result: dict[str, ProviderSpecV2] = {}
    for name in ("deepseek_formal", "qwen_flash"):
        row = raw[name]
        envs = (
            ("DEEPSEEK_API_KEY",)
            if name == "deepseek_formal"
            else ("QWEN_API_KEY", "DASHSCOPE_API_KEY")
        )
        result[name] = ProviderSpecV2(
            condition_id=str(row["condition_id"]),
            provider=str(row["provider"]),
            endpoint=str(row["endpoint"]),
            requested_model=str(row["configured_model_id"]),
            expected_returned_model=str(row["expected_returned_model"]),
            api_key_envs=envs,
            reasoning_control={},
            verified_max_output_tokens=int(row["max_output_tokens"]),
            official_model_documentation=str(row["official_source"]),
            official_json_documentation=str(row["official_source"]),
            temperature=float(row["temperature"]),
            max_output_tokens=int(row["max_output_tokens"]),
            attempt_timeout_seconds=int(row["attempt_timeout_seconds"]),
            stage_wall_timeout_seconds=int(row["stage_timeout_seconds"]),
            episode_wall_timeout_seconds=int(row["episode_timeout_seconds"]),
        )
    return result


def _contexts(
    tasks: list[dict[str, Any]], battery: BatteryConfig
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in tasks:
        activation = int(task["visible_update"]["activation_step"])
        stage1 = economic_dispatch(
            task,
            battery,
            initial_soc_kwh=battery.initial_soc_kwh,
            start_step=0,
            apply_event=False,
            frame_override=stage1_information_frame(task),
        )
        event_soc = float(
            battery.initial_soc_kwh
            + 0.25
            * (
                stage1.charge_kw[:activation].sum() * battery.charge_efficiency
                - stage1.discharge_kw[:activation].sum() / battery.discharge_efficiency
            )
        )
        result[str(task["scenario_id"])] = {
            "activation": activation,
            "deterministic_stage1_kw": stage1.action_kw.tolist(),
            "event_soc_kwh": event_soc,
            "initial_state": canonical_initial_state(task),
            "event_state": canonical_event_state(task, event_soc),
        }
    return result


def _prepare_requests(
    plan: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    specs: dict[str, ProviderSpecV2],
) -> list[dict[str, Any]]:
    task_map = {str(task["scenario_id"]): task for task in tasks}
    prepared: list[dict[str, Any]] = []
    for row in sorted(plan, key=lambda item: int(item["run_order"])):
        task = task_map[str(row["scenario_id"])]
        context = contexts[str(row["scenario_id"])]
        spec = specs[str(row["model_condition"])]
        stages: dict[str, dict[str, Any]] = {}
        for stage, state in (
            ("stage1", context["initial_state"]),
            ("stage2", context["event_state"]),
        ):
            prompt = compile_prompt(task, str(row["interface"]), stage, state)
            payload = build_payload(
                model_id=spec.requested_model,
                provider=spec.provider,
                interface=str(row["interface"]),
                prompt=prompt,
                max_output_tokens=spec.max_output_tokens,
            )
            stages[stage] = {
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "request": ProviderRequest(
                    stage=cast(Any, stage),
                    run_id=f"{row['episode_id']}_{stage}",
                    condition_id=spec.condition_id,
                    requested_model=spec.requested_model,
                    payload=payload,
                    payload_sha256=sha256_json(payload),
                    timeout_seconds=spec.attempt_timeout_seconds,
                ),
            }
        prepared.append({"plan": row, "task": task, "context": context, "stages": stages})
    if len(prepared) != EXPECTED_EPISODES:
        raise SystemExit(f"{EXPERIMENT_LABEL} prepared episode count mismatch")
    return prepared


async def _run(
    prepared: list[dict[str, Any]],
    battery: BatteryConfig,
    specs: dict[str, ProviderSpecV2],
    keys: dict[str, str],
    run_dir: Path,
    *,
    concurrency: int,
) -> dict[str, Any]:
    pending = [
        item
        for item in prepared
        if not (run_dir / "records" / f"{item['plan']['episode_id']}.json").exists()
    ]
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    fingerprints: dict[str, set[str]] = {name: set() for name in specs}
    hold_reasons: list[str] = []
    completed = 0
    calls_observed = 0

    async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
        async def worker() -> None:
            nonlocal completed, calls_observed
            while not stop.is_set():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await _run_episode(
                        item,
                        battery,
                        specs[str(item["plan"]["model_condition"])],
                        keys[str(item["plan"]["model_condition"])],
                        client,
                        semaphore,
                    )
                    _write_json_atomic(
                        run_dir / "records" / f"{item['plan']['episode_id']}.json",
                        record,
                    )
                    async with lock:
                        completed += 1
                        for stage in ("stage1", "stage2"):
                            response = record[stage]["response"]
                            if response.get("status") == "ok":
                                calls_observed += 1
                            returned = response.get("returned_model")
                            expected = specs[str(item["plan"]["model_condition"])].expected_returned_model
                            if response.get("status") == "ok" and returned != expected:
                                hold_reasons.append(
                                    f"wrong_returned_model:{item['plan']['model_condition']}:{returned}"
                                )
                            raw = response.get("raw_response")
                            fingerprint = raw.get("system_fingerprint") if isinstance(raw, dict) else None
                            if fingerprint:
                                fingerprints[str(item["plan"]["model_condition"])].add(str(fingerprint))
                        if len(fingerprints["deepseek_formal"]) > 1:
                            hold_reasons.append("deepseek_system_fingerprint_drift")
                        if hold_reasons:
                            stop.set()
                        print(
                            f"{EXPERIMENT_LABEL.lower()}_completed={completed}/{len(pending)} "
                            f"provider_responses={calls_observed}/{EXPECTED_CALLS} hold={stop.is_set()}",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    status = "hold" if hold_reasons else "completed"
    result = {
        "status": status,
        "completed_this_run": completed,
        "unissued_episodes": queue.qsize(),
        "provider_responses_observed": calls_observed,
        "hold_reasons": sorted(set(hold_reasons)),
        "returned_system_fingerprints": {
            name: sorted(values) for name, values in fingerprints.items()
        },
    }
    if status == "hold":
        _write_json_atomic(run_dir / HOLD_FILENAME, result)
    return result


async def _run_episode(
    item: dict[str, Any],
    battery: BatteryConfig,
    spec: ProviderSpecV2,
    api_key: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + spec.episode_wall_timeout_seconds
    stage_records: dict[str, Any] = {}
    for stage in ("stage1", "stage2"):
        request: ProviderRequest = item["stages"][stage]["request"]
        sent_utc = _utc_now()
        response = await complete_stage_v2(
            client,
            semaphore,
            spec,
            api_key,
            request,
            absolute_episode_deadline=deadline,
        )
        received_utc = _utc_now()
        horizon = 96 if stage == "stage1" else 96 - int(item["context"]["activation"])
        parsed = parse_output(str(response.get("content", "")), str(item["plan"]["interface"]), horizon)
        stage_records[stage] = {
            "prompt": item["stages"][stage]["prompt"],
            "prompt_sha256": item["stages"][stage]["prompt_sha256"],
            "request": request.payload,
            "request_sha256": request.payload_sha256,
            "request_sent_utc": sent_utc,
            "response_received_utc": received_utc,
            "response": response,
            "response_sha256": sha256_json(response),
            "parsed": parsed_payload(parsed),
            "parsed_sha256": sha256_json(parsed_payload(parsed)),
            "estimated_provider_cost": _usage_cost(
                str(item["plan"]["model_condition"]), response.get("usage", {})
            ),
        }
    outcome = _score(item, battery, stage_records)
    record = {
        "schema_version": RECORD_SCHEMA,
        "evidence_tier": EVIDENCE_TIER,
        "excluded_from_claim_bearing_analysis": EXCLUDED_FROM_CLAIMS,
        **item["plan"],
        "source_scenario_id": item["task"]["source_scenario_id"],
        "task": item["task"],
        "task_hash_matches": sha256_json(item["task"]) == item["plan"]["task_sha256"],
        "canonical_context": item["context"],
        "stage1": stage_records["stage1"],
        "stage2": stage_records["stage2"],
        "outcome": outcome,
        "episode_elapsed_seconds": round(time.monotonic() - started, 6),
    }
    record["evidence_root_sha256"] = sha256_json(
        {
            "plan": record["plan_row_sha256"],
            "task": record["task_sha256"],
            "stage1_request": record["stage1"]["request_sha256"],
            "stage1_response": record["stage1"]["response_sha256"],
            "stage2_request": record["stage2"]["request_sha256"],
            "stage2_response": record["stage2"]["response_sha256"],
            "outcome": sha256_json(outcome),
        }
    )
    return record


def _score(
    item: dict[str, Any], battery: BatteryConfig, stages: dict[str, Any]
) -> dict[str, Any]:
    p1 = stages["stage1"]["parsed"]
    p2 = stages["stage2"]["parsed"]
    parser_success = bool(p1["ok"] and p2["ok"])
    base = {
        "parser_success": parser_success,
        "schema_success": parser_success,
        "stage1_parser_success": bool(p1["ok"]),
        "stage2_parser_success": bool(p2["ok"]),
        "failure_as_zero": not parser_success,
        "diagnostics": [*p1["diagnostics"], *p2["diagnostics"]],
    }
    if not parser_success:
        return {
            **base,
            "exact_zero": False,
            "nontrivial_action": False,
            "raw_feasible": False,
            "terminal_soc_error_kwh": None,
            "action_energy_kwh": 0.0,
            "raw_cost_usd": None,
            "projection_computed": False,
        }
    stage1 = [float(value) for value in p1["dense_action_kw"]]
    stage2 = [float(value) for value in p2["dense_action_kw"]]
    replay = replay_two_stage(item["task"], stage1, stage2, battery)
    activation = int(item["context"]["activation"])
    energy = action_energy_kwh(stage1[:activation]) + action_energy_kwh(stage2)
    if not ENABLE_INLINE_PROJECTION:
        return {
            **base,
            "exact_zero": energy == 0.0,
            "nontrivial_action": is_nontrivial([*stage1[:activation], *stage2]),
            "raw_feasible": bool(replay.feasible),
            "raw_failure_codes": list(replay.failure_codes),
            "terminal_soc_error_kwh": replay.terminal_soc_gap_kwh,
            "action_energy_kwh": energy,
            "raw_cost_usd": replay.total_cost_usd,
            "projection_computed": False,
            "projection_pending_offline": True,
            "simulator_trace": json.loads(
                replay.trace.reset_index(names="timestamp_utc").to_json(
                    orient="records", date_format="iso"
                )
            ),
        }
    projection1 = lexicographic_projection(
        item["task"],
        stage1,
        battery,
        initial_soc_kwh=battery.initial_soc_kwh,
        start_step=0,
        apply_event=False,
        require_terminal=True,
        frame_override=stage1_information_frame(item["task"]),
    )
    projection2 = lexicographic_projection(
        item["task"],
        stage2,
        battery,
        initial_soc_kwh=float(item["context"]["event_soc_kwh"]),
        start_step=activation,
        apply_event=True,
        require_terminal=True,
    )
    return {
        **base,
        "exact_zero": energy == 0.0,
        "nontrivial_action": is_nontrivial([*stage1[:activation], *stage2]),
        "raw_feasible": bool(replay.feasible),
        "raw_failure_codes": list(replay.failure_codes),
        "terminal_soc_error_kwh": replay.terminal_soc_gap_kwh,
        "action_energy_kwh": energy,
        "raw_cost_usd": replay.total_cost_usd,
        "projection_computed": True,
        "projection_distance_kwh": projection1.distance_kwh + projection2.distance_kwh,
        "projected_stage1_kw": projection1.action_kw.tolist(),
        "projected_stage2_kw": projection2.action_kw.tolist(),
        "simulator_trace": json.loads(replay.trace.reset_index(names="timestamp_utc").to_json(orient="records", date_format="iso")),
    }


def _write_reports(run_dir: Path, plan: list[dict[str, Any]], run_status: dict[str, Any]) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "records").glob("*.json"))]
    attempts = sum(
        len(record[stage]["response"].get("attempts", []))
        for record in records
        for stage in ("stage1", "stage2")
    )
    actual_calls = sum(
        record[stage]["response"].get("status") != "skipped"
        for record in records
        for stage in ("stage1", "stage2")
    )
    summary = {
        "schema_version": EXECUTION_SCHEMA,
        "created_utc": _utc_now(),
        "status": COMPLETION_STATUS
        if run_status["status"] == "completed" and len(records) == EXPECTED_EPISODES
        else "HOLD",
        "excluded_from_claim_bearing_analysis": EXCLUDED_FROM_CLAIMS,
        "planned_logical_episodes": len(plan),
        "completed_logical_episodes": len(records),
        "planned_primary_calls": EXPECTED_CALLS,
        "actual_stage_calls": actual_calls,
        "provider_attempts_including_retries": attempts,
        "protocol_exclusions": sum(
            record[stage]["response"].get("status") == "ok"
            and record[stage]["response"].get("returned_model")
            != record[stage]["request"]["model"]
            for record in records
            for stage in ("stage1", "stage2")
        ),
        "parser_success_episodes": sum(record["outcome"]["parser_success"] for record in records),
        "nontrivial_action_episodes": sum(record["outcome"]["nontrivial_action"] for record in records),
        "raw_feasible_episodes": sum(record["outcome"]["raw_feasible"] for record in records),
        "run_status": run_status,
    }
    _write_json_atomic(run_dir / SUMMARY_FILENAME, summary)
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != HASH_FILENAME)
    _write_json_atomic(
        run_dir / HASH_FILENAME,
        {
            "schema_version": HASH_SCHEMA,
            "files": [
                {
                    "path": str(path.relative_to(run_dir)),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _call_ledger(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prepared:
        for stage in ("stage1", "stage2"):
            request: ProviderRequest = item["stages"][stage]["request"]
            rows.append(
                {
                    "episode_id": item["plan"]["episode_id"],
                    "run_order": item["plan"]["run_order"],
                    "scenario_id": item["plan"]["scenario_id"],
                    "model_condition": item["plan"]["model_condition"],
                    "interface": item["plan"]["interface"],
                    "stage": stage,
                    "requested_model": request.requested_model,
                    "payload_sha256": request.payload_sha256,
                    "planned_before_first_provider_call": True,
                }
            )
    return rows


def _usage_cost(condition: str, usage: Any) -> dict[str, Any]:
    pricing = yaml.safe_load(
        (ROOT / "experiments_v2/configs/pricing.yaml").read_text(encoding="utf-8")
    )["conditions"][condition]
    data = usage if isinstance(usage, dict) else {}
    prompt = int(data.get("prompt_tokens", data.get("input_tokens", 0)) or 0)
    completion = int(data.get("completion_tokens", data.get("output_tokens", 0)) or 0)
    details = data.get("prompt_tokens_details")
    cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    uncached = max(prompt - cached, 0)
    cached_rate = pricing.get("cached_input_price_per_million")
    amount = (
        uncached * float(pricing["input_price_per_million"])
        + completion * float(pricing["output_price_per_million"])
        + cached * float(cached_rate if cached_rate is not None else pricing["input_price_per_million"])
    ) / 1_000_000.0
    return {
        "amount": amount,
        "currency": pricing["currency"],
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "pricing_source": pricing["source_note"],
    }


def _load_keys(specs: dict[str, ProviderSpecV2]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for name, spec in specs.items():
        for env_name in spec.api_key_envs:
            value = os.environ.get(env_name)
            if value:
                keys[name] = value
                break
        if name not in keys:
            raise SystemExit(f"credential unavailable for {name}; expected one of {spec.api_key_envs}")
    return keys


def _verify_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["file_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"{EXPERIMENT_LABEL} frozen file hash mismatch: {relative}")
    if manifest["planned_primary_calls"] != EXPECTED_CALLS:
        raise SystemExit(f"{EXPERIMENT_LABEL} frozen call count mismatch")


def _require_pre_run_tag() -> None:
    """Compatibility no-op: the public release is frozen by file hashes."""


def _validate_worktree() -> list[str]:
    """Compatibility helper retained for callers; public files use hashes."""
    return []


def _git(*args: str) -> str:
    """Compatibility helper retained without a repository dependency."""
    return "public-scientific-file-manifest"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    main()
