#!/usr/bin/env python3
"""Run E2 with one shared Stage 1 and five paired Stage 2 branches."""

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
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.actions import action_energy_kwh, is_nontrivial  # noqa: E402
from experiments_v2.e2 import BRANCHES, INTERFACE, e2_states, model_carrier  # noqa: E402
from experiments_v2.hashing import sha256_json, sha256_text  # noqa: E402
from experiments_v2.p0 import (  # noqa: E402
    build_payload,
    compile_prompt,
    parse_output,
    parsed_payload,
)

from energy_agent_reliability.battery import BatteryOverrides, simulate_dispatch  # noqa: E402
from energy_agent_reliability.config import BatteryConfig, load_protocol  # noqa: E402
from energy_agent_reliability.llm.models import ProviderRequest  # noqa: E402
from energy_agent_reliability.llm.protocol_v2 import ProviderSpecV2  # noqa: E402
from energy_agent_reliability.llm.runtime_v2 import complete_stage_v2  # noqa: E402
from energy_agent_reliability.online_gate_v6 import limits_at, task_frame  # noqa: E402
from energy_agent_reliability.provenance import read_jsonl, sha256_file  # noqa: E402

TASKS = ROOT / "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl"
PLAN = ROOT / "experiments_v2/manifests/E2_RUN_PLAN.jsonl"
MANIFEST = ROOT / "experiments_v2/manifests/E2_PRE_RUN_MANIFEST.json"
MODELS = ROOT / "experiments_v2/configs/models.yaml"
RUN_ROOT = ROOT / "runs/experiments_v2/e2"
PRE_RUN_TAG = "energybench-v2-e2-pre-run"
EXPECTED_BLOCKS = 360
EXPECTED_CALLS = 2160
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
        raise SystemExit("E2 freezes provider concurrency at five")
    _verify_manifest()
    tasks = list(read_jsonl(TASKS))
    plan = list(read_jsonl(PLAN))
    specs = _provider_specs()
    prepared = _prepare(plan, tasks, specs)
    run_dir = RUN_ROOT / args.run_label
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory exists: {run_dir}")
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    _write_json(
        run_dir / "PRE_RUN_IDENTITY.json",
        {
            "schema_version": "experiments_v2_e2_run_identity_v1",
            "created_utc": _utc_now(),
            "source_release_identity": "public-scientific-file-manifest",
            "plan_sha256": sha256_file(PLAN),
            "frozen_manifest_sha256": sha256_file(MANIFEST),
            "planned_blocks": EXPECTED_BLOCKS,
            "planned_primary_calls": EXPECTED_CALLS,
            "concurrency": 5,
            "httpx_trust_env": False,
            "api_key_values_recorded": False,
        },
    )
    _write_jsonl(run_dir / "PLANNED_CALL_LEDGER.jsonl", _ledger(prepared))
    if args.plan_only:
        print(
            json.dumps({"status": "plan_locked", "blocks": len(prepared), "calls": EXPECTED_CALLS})
        )
        return
    keys = _load_keys(specs)
    battery = load_protocol(ROOT / "configs/frozen_protocol.yaml").battery.model_copy(
        update={"terminal_soc_kwh": 325.0}
    )
    status = asyncio.run(_run(prepared, specs, keys, battery, run_dir))
    _write_summary(run_dir, status)
    if status["status"] != "completed":
        raise SystemExit("E2 entered HOLD")


def _provider_specs() -> dict[str, ProviderSpecV2]:
    rows = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["conditions"]
    result: dict[str, ProviderSpecV2] = {}
    for name in ("deepseek_formal", "qwen_flash"):
        row = rows[name]
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


def _prepare(
    plan: list[dict[str, Any]], tasks: list[dict[str, Any]], specs: dict[str, ProviderSpecV2]
) -> list[dict[str, Any]]:
    task_map = {str(task["scenario_id"]): task for task in tasks}
    result: list[dict[str, Any]] = []
    for row in sorted(plan, key=lambda value: int(value["run_order"])):
        task = task_map[str(row["scenario_id"])]
        states = e2_states(task)
        spec = specs[str(row["model_condition"])]
        requests: dict[str, dict[str, Any]] = {}
        requests["stage1"] = _request(row, task, spec, "stage1", states["initial"], None)
        for branch in ("C", "S", "K", "C2"):
            requests[branch] = _request(row, task, spec, "stage2", states[branch], branch)
        if requests["C"]["request"].payload_sha256 != requests["C2"]["request"].payload_sha256:
            raise SystemExit(f"C/C2 payload mismatch before run: {row['block_id']}")
        result.append({"plan": row, "task": task, "states": states, "requests": requests})
    if len(result) != EXPECTED_BLOCKS:
        raise SystemExit(f"E2 block count mismatch: {len(result)}")
    return result


def _request(
    row: dict[str, Any],
    task: dict[str, Any],
    spec: ProviderSpecV2,
    stage: str,
    state: dict[str, Any],
    branch: str | None,
) -> dict[str, Any]:
    prompt = compile_prompt(task, INTERFACE, stage, state)
    payload = build_payload(
        model_id=spec.requested_model,
        provider=spec.provider,
        interface=INTERFACE,
        prompt=prompt,
        max_output_tokens=spec.max_output_tokens,
    )
    label = stage if branch is None else f"stage2_{branch}"
    request = ProviderRequest(
        stage=cast(Any, stage),
        run_id=f"{row['block_id']}_{label}",
        condition_id=spec.condition_id,
        requested_model=spec.requested_model,
        payload=payload,
        payload_sha256=sha256_json(payload),
        timeout_seconds=spec.attempt_timeout_seconds,
    )
    return {"prompt": prompt, "prompt_sha256": sha256_text(prompt), "request": request}


async def _run(
    prepared: list[dict[str, Any]],
    specs: dict[str, ProviderSpecV2],
    keys: dict[str, str],
    battery: BatteryConfig,
    run_dir: Path,
) -> dict[str, Any]:
    pending = [
        item
        for item in prepared
        if not (run_dir / "records" / f"{item['plan']['block_id']}.json").exists()
    ]
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    semaphore = asyncio.Semaphore(5)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    hold: list[str] = []
    fingerprints: dict[str, set[str]] = {name: set() for name in specs}
    done = len(prepared) - len(pending)
    responses = _count_saved_calls(run_dir)
    async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:

        async def worker() -> None:
            nonlocal done, responses
            while not stop.is_set():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await _run_block(
                        item,
                        specs[str(item["plan"]["model_condition"])],
                        keys[str(item["plan"]["model_condition"])],
                        battery,
                        client,
                        semaphore,
                        run_dir,
                    )
                    _write_json(run_dir / "records" / f"{item['plan']['block_id']}.json", record)
                    async with lock:
                        done += 1
                        for stage in [record["stage1"], *record["branches"].values()]:
                            response = stage["response"]
                            if response.get("status") != "skipped":
                                responses += 1
                            if response.get("status") == "ok":
                                expected = specs[
                                    str(item["plan"]["model_condition"])
                                ].expected_returned_model
                                if response.get("returned_model") != expected:
                                    hold.append(f"wrong_returned_model:{item['plan']['block_id']}")
                                raw = response.get("raw_response")
                                fp = (
                                    raw.get("system_fingerprint") if isinstance(raw, dict) else None
                                )
                                if fp:
                                    fingerprints[str(item["plan"]["model_condition"])].add(str(fp))
                        if len(fingerprints["deepseek_formal"]) > 1:
                            hold.append("deepseek_system_fingerprint_drift")
                        if hold:
                            stop.set()
                        print(
                            f"e2_blocks={done}/{EXPECTED_BLOCKS} stage_calls={responses}/{EXPECTED_CALLS} hold={stop.is_set()}",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(5)))
    result = {
        "status": "hold" if hold else "completed",
        "completed_blocks": done,
        "unissued_blocks": queue.qsize(),
        "stage_calls_observed": responses,
        "hold_reasons": sorted(set(hold)),
        "returned_system_fingerprints": {key: sorted(value) for key, value in fingerprints.items()},
    }
    if hold:
        _write_json(run_dir / "E2_HOLD.json", result)
    return result


async def _run_block(
    item: dict[str, Any],
    spec: ProviderSpecV2,
    key: str,
    battery: BatteryConfig,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + spec.episode_wall_timeout_seconds
    checkpoint = run_dir / "checkpoints" / str(item["plan"]["block_id"])
    checkpoint.mkdir(exist_ok=True)
    stage1 = await _load_or_call(
        checkpoint / "stage1.json",
        item["requests"]["stage1"],
        spec,
        key,
        client,
        semaphore,
        deadline,
        96,
    )
    dynamic = model_carrier(stage1["parsed"].get("state"))
    requests = dict(item["requests"])
    if dynamic is not None:
        requests["M"] = _request(item["plan"], item["task"], spec, "stage2", dynamic, "M")

    async def branch_call(branch: str) -> tuple[str, dict[str, Any]]:
        if branch == "M" and dynamic is None:
            return branch, _skipped_stage("stage1_state_unavailable")
        request = requests[branch]
        horizon = 96 - int(item["task"]["visible_update"]["activation_step"])
        value = await _load_or_call(
            checkpoint / f"stage2_{branch}.json",
            request,
            spec,
            key,
            client,
            semaphore,
            deadline,
            horizon,
        )
        return branch, value

    branches = dict(await asyncio.gather(*(branch_call(branch) for branch in BRANCHES)))
    if branches["C"]["request_sha256"] != branches["C2"]["request_sha256"]:
        raise RuntimeError("C/C2 payload changed during execution")
    outcomes = {
        branch: _score_suffix(item["task"], value, battery) for branch, value in branches.items()
    }
    record: dict[str, Any] = {
        "schema_version": "experiments_v2_e2_block_v1",
        "evidence_tier": "confirmatory_physical_carrier",
        "excluded_from_claim_bearing_analysis": False,
        **item["plan"],
        "source_scenario_id": item["task"]["source_scenario_id"],
        "task": item["task"],
        "task_hash_matches": sha256_json(item["task"]) == item["plan"]["task_sha256"],
        "carriers": {**item["states"], "M": dynamic},
        "stage1": stage1,
        "branches": branches,
        "outcomes": outcomes,
        "block_elapsed_seconds": round(time.monotonic() - started, 6),
    }
    record["evidence_root_sha256"] = sha256_json(
        {
            "plan": record["plan_row_sha256"],
            "task": record["task_sha256"],
            "stage1": stage1["response_sha256"],
            "branches": {branch: value["response_sha256"] for branch, value in branches.items()},
            "outcomes": sha256_json(outcomes),
        }
    )
    return record


async def _load_or_call(
    path: Path,
    prepared: dict[str, Any],
    spec: ProviderSpecV2,
    key: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    deadline: float,
    horizon: int,
) -> dict[str, Any]:
    request: ProviderRequest = prepared["request"]
    if path.exists():
        saved: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if saved["request_sha256"] != request.payload_sha256:
            raise RuntimeError(f"resume payload mismatch: {path}")
        return saved
    sent = _utc_now()
    response = await complete_stage_v2(
        client, semaphore, spec, key, request, absolute_episode_deadline=deadline
    )
    parsed = parse_output(str(response.get("content", "")), INTERFACE, horizon)
    value = {
        "prompt": prepared["prompt"],
        "prompt_sha256": prepared["prompt_sha256"],
        "request": request.payload,
        "request_sha256": request.payload_sha256,
        "request_sent_utc": sent,
        "response_received_utc": _utc_now(),
        "response": response,
        "response_sha256": sha256_json(response),
        "parsed": parsed_payload(parsed),
        "parsed_sha256": sha256_json(parsed_payload(parsed)),
        "estimated_provider_cost": _usage_cost(spec.condition_id, response.get("usage", {})),
    }
    _write_json(path, value)
    return value


def _skipped_stage(reason: str) -> dict[str, Any]:
    response = {"status": "skipped", "reason": reason, "attempts": []}
    parsed = parsed_payload(parse_output("", INTERFACE, 0))
    return {
        "prompt": None,
        "prompt_sha256": None,
        "request": None,
        "request_sha256": None,
        "request_sent_utc": None,
        "response_received_utc": None,
        "response": response,
        "response_sha256": sha256_json(response),
        "parsed": parsed,
        "parsed_sha256": sha256_json(parsed),
        "estimated_provider_cost": {"amount": 0.0, "currency": None},
    }


def _score_suffix(
    task: dict[str, Any], stage: dict[str, Any], battery: BatteryConfig
) -> dict[str, Any]:
    parsed = stage["parsed"]
    if not parsed["ok"]:
        return {
            "parser_success": False,
            "failure_as_zero": True,
            "exact_zero": False,
            "nontrivial_action": False,
            "raw_feasible": False,
            "terminal_soc_error_kwh": None,
            "action_energy_kwh": 0.0,
            "raw_cost_usd": None,
            "diagnostics": parsed["diagnostics"],
        }
    activation = int(task["visible_update"]["activation_step"])
    actions = np.asarray(parsed["dense_action_kw"], dtype=float)
    frame = task_frame(task).iloc[activation:].rename(columns={"price_usd_mwh": "price_per_mwh"})
    overrides = [
        BatteryOverrides(**limits_at(task, step, battery, apply_event=True))
        for step in range(activation, 96)
    ]
    result = simulate_dispatch(
        frame,
        actions,
        battery,
        overrides_by_step=overrides,
        initial_soc_kwh=float(task["v2_prefix_intervention"]["canonical_event_soc_kwh"]),
        require_terminal_soc=True,
        step_hours=0.25,
    )
    violation = float(result.trace["violation_cost"].sum())
    feasible = bool(result.feasible and violation <= 1e-8)
    return {
        "parser_success": True,
        "failure_as_zero": False,
        "exact_zero": bool(np.all(np.abs(actions) <= 1e-9)),
        "nontrivial_action": is_nontrivial(actions.tolist()),
        "raw_feasible": feasible,
        "terminal_soc_error_kwh": float(result.terminal_soc_gap_kwh),
        "action_energy_kwh": action_energy_kwh(actions.tolist()),
        "raw_cost_usd": float(result.total_cost),
        "violation_cost_usd": violation,
        "diagnostics": [],
        "simulator_trace": json.loads(
            result.trace.reset_index(names="timestamp_utc").to_json(
                orient="records", date_format="iso"
            )
        ),
    }


def _ledger(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prepared:
        for label in ("stage1", *BRANCHES):
            request = item["requests"].get(label, {}).get("request")
            rows.append(
                {
                    "block_id": item["plan"]["block_id"],
                    "run_order": item["plan"]["run_order"],
                    "scenario_id": item["plan"]["scenario_id"],
                    "model_condition": item["plan"]["model_condition"],
                    "repetition": item["plan"]["repetition"],
                    "stage": "stage1" if label == "stage1" else "stage2",
                    "branch": None if label == "stage1" else label,
                    "requested_model": item["plan"]["model_condition"],
                    "payload_sha256": request.payload_sha256
                    if request is not None
                    else "dynamic_from_saved_stage1",
                    "planned_before_first_provider_call": True,
                }
            )
    if len(rows) != EXPECTED_CALLS:
        raise RuntimeError("E2 call ledger cardinality mismatch")
    return rows


def _write_summary(run_dir: Path, status: dict[str, Any]) -> None:
    records = [
        json.loads(path.read_text()) for path in sorted((run_dir / "records").glob("*.json"))
    ]
    calls = [record["stage1"] for record in records] + [
        stage for record in records for stage in record["branches"].values()
    ]
    summary = {
        "schema_version": "experiments_v2_e2_execution_summary_v1",
        "created_utc": _utc_now(),
        "status": "E2_COMPLETED"
        if status["status"] == "completed" and len(records) == EXPECTED_BLOCKS
        else "HOLD",
        "planned_blocks": EXPECTED_BLOCKS,
        "completed_blocks": len(records),
        "planned_primary_calls": EXPECTED_CALLS,
        "actual_stage_calls": sum(stage["response"].get("status") != "skipped" for stage in calls),
        "successful_provider_responses": sum(
            stage["response"].get("status") == "ok" for stage in calls
        ),
        "structured_stage2_skips": sum(
            stage["response"].get("status") == "skipped" for stage in calls
        ),
        "provider_attempts_including_retries": sum(
            len(stage["response"].get("attempts", [])) for stage in calls
        ),
        "protocol_exclusions": sum(
            stage["response"].get("status") == "ok"
            and stage["response"].get("returned_model") != stage["request"].get("model")
            for stage in calls
            if isinstance(stage.get("request"), dict)
        ),
        "run_status": status,
    }
    _write_json(run_dir / "E2_EXECUTION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _usage_cost(condition_id: str, usage: Any) -> dict[str, Any]:
    conditions = yaml.safe_load((ROOT / "experiments_v2/configs/pricing.yaml").read_text())[
        "conditions"
    ]
    condition = next(
        (
            key
            for key, value in yaml.safe_load(MODELS.read_text())["conditions"].items()
            if value["condition_id"] == condition_id
        ),
        condition_id,
    )
    pricing = conditions[condition]
    data = usage if isinstance(usage, dict) else {}
    prompt = int(data.get("prompt_tokens", data.get("input_tokens", 0)) or 0)
    completion = int(data.get("completion_tokens", data.get("output_tokens", 0)) or 0)
    details = data.get("prompt_tokens_details")
    cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    rate = float(
        pricing.get("cached_input_price_per_million") or pricing["input_price_per_million"]
    )
    amount = (
        (prompt - cached) * float(pricing["input_price_per_million"])
        + cached * rate
        + completion * float(pricing["output_price_per_million"])
    ) / 1_000_000
    return {
        "amount": amount,
        "currency": pricing["currency"],
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "pricing_source": pricing["source_note"],
    }


def _load_keys(specs: dict[str, ProviderSpecV2]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, spec in specs.items():
        for env_name in spec.api_key_envs:
            if os.environ.get(env_name):
                result[name] = os.environ[env_name]
                break
        if name not in result:
            raise SystemExit(f"credential unavailable for {name}")
    return result


def _verify_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["file_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"E2 frozen file hash mismatch: {relative}")
    if manifest["planned_primary_calls"] != EXPECTED_CALLS:
        raise SystemExit("E2 frozen call count mismatch")


def _require_pre_run_tag() -> None:
    """Compatibility no-op: the public release is frozen by file hashes."""


def _validate_worktree() -> list[str]:
    """Compatibility helper retained for callers; public files use hashes."""
    return []


def _count_saved_calls(run_dir: Path) -> int:
    return len(list((run_dir / "checkpoints").glob("*/*.json")))


def _git(*args: str) -> str:
    """Compatibility helper retained without a repository dependency."""
    return "public-scientific-file-manifest"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    partial.replace(path)


if __name__ == "__main__":
    main()
