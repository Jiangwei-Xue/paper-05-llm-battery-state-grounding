#!/usr/bin/env python3
"""Fail-closed Stage-2-only runner for E2b-v2.

Provider execution requires an explicit flag and a separately authorized turn.
The default command performs local manifest verification only.
"""

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
PROJECT = ROOT.parent
REVIEW_ROOT = ROOT / "reviews/e2b_protocol_v2"
DEFAULT_MANIFEST = REVIEW_ROOT / "E2B_V2_PRE_CALL_MANIFEST.json"
RUN_ROOT = ROOT / "runs/e2b_protocol_v2"
BRANCHES = ("C1", "C2", "S1", "S2")

sys.path.insert(0, str(ROOT / "scripts"))

from e2b_parser_v2 import parse_output  # noqa: E402
from e2b_v2_common import (  # noqa: E402
    canonical_bytes,
    digest,
    render_prompt,
    request_payload,
    same_information_mpc,
    score_branch,
    sha256,
    visible_suffix_rows,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"OFFLINE_GATES_PASS_EXECUTION_HOLD", "PRE_CALL_FROZEN_HOLD"}:
        raise SystemExit("E2b-v2 manifest is not in a permitted pre-call hold state")
    if manifest.get("provider_calls_performed") != 0 or manifest.get("network_attempts") != 0:
        raise SystemExit("pre-call manifest records network activity")
    for label, item in manifest["artifacts"].items():
        artifact = PROJECT / str(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            raise SystemExit(f"frozen artifact mismatch: {label}: {artifact}")
    return cast(dict[str, Any], manifest)


def public_hash_gate_ready() -> bool:
    """The public runner is frozen by the package file-hash manifest."""
    return True


def load_model_specs() -> dict[str, dict[str, Any]]:
    path = PROJECT / "experiments_v2/configs/models.yaml"
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: dict(content["conditions"][name]) for name in ("deepseek_formal", "qwen_flash")}


def credential(spec: dict[str, Any], model: str) -> str:
    names = [str(spec["credentials_env"])]
    if model == "qwen_flash":
        names.extend(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(f"missing credential for {model}; secret values are never printed or stored")


async def provider_call(
    client: httpx.AsyncClient,
    spec: dict[str, Any],
    key: str,
    payload: dict[str, Any],
    *,
    call_id: str,
) -> dict[str, Any]:
    retryable = {429, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511}
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for attempt_index in range(3):
        started = time.monotonic()
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            "request_sent_utc": utc_now(),
            "http_status": None,
            "status": "transport_error",
            "elapsed_seconds": None,
        }
        should_retry = False
        try:
            response = await client.post(
                str(spec["endpoint"]),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                content=canonical_bytes(payload),
                timeout=float(spec["attempt_timeout_seconds"]),
            )
            attempt["http_status"] = response.status_code
            try:
                raw = response.json()
            except ValueError:
                raw = None
            if response.status_code >= 400 or not isinstance(raw, dict):
                attempt["status"] = "provider_error"
                attempt["error_type"] = "http_or_non_json_response"
                should_retry = response.status_code in retryable
                final = {"raw_response": raw, "status": "provider_error"}
            else:
                choices = raw.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                choice = choices[0] if choices else {}
                attempt["status"] = "ok"
                final = {
                    "status": "ok",
                    "raw_response": raw,
                    "content": str(message.get("content") or ""),
                    "returned_model": raw.get("model"),
                    "system_fingerprint": raw.get("system_fingerprint"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": raw.get("usage") or {},
                    "reasoning_content_present": bool(message.get("reasoning_content")),
                }
        except (httpx.HTTPError, TimeoutError) as exc:
            attempt["error_type"] = type(exc).__name__
            should_retry = True
            final = {"status": "transport_error", "raw_response": None}
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 6)
        attempt["response_received_utc"] = utc_now()
        attempts.append(attempt)
        if final.get("status") == "ok" or not should_retry or attempt_index == 2:
            break
    return {
        "call_id": call_id,
        "requested_model": str(spec["configured_model_id"]),
        "provider_route": str(spec["route"]),
        "request_sha256": digest(payload),
        "attempt_count": len(attempts),
        "attempts": attempts,
        **final,
    }


def failed_parse(code: str, horizon: int) -> dict[str, Any]:
    return {
        "ok": False,
        "raw_actions": None,
        "dense_action_kw": [],
        "diagnostics": [code],
        "horizon": horizon,
        "segment_count": 0,
    }


async def run_block(
    item: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    keys: dict[str, str],
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    source = PROJECT / str(item["source_record_path"])
    task = json.loads(source.read_text(encoding="utf-8"))["task"]
    if digest(task) != item["task_sha256"] or sha256(source) != item["source_record_sha256"]:
        raise RuntimeError("source task hash mismatch")
    model = str(item["model_condition"])
    spec = specs[model]
    horizon = len(visible_suffix_rows(task))
    record: dict[str, Any] = {
        "schema_version": "e2b_v2_execution_block_v1",
        "protocol_id": "energybench-e2b-stage2-only-carrier-sensitivity-v2",
        "tier": item["tier"],
        "block_id": item["block_id"],
        "scenario_id": item["scenario_id"],
        "model_condition": model,
        "branch_order": item["branch_order"],
        "task_sha256": item["task_sha256"],
        "source_record_sha256": item["source_record_sha256"],
        "stage1_calls": 0,
        "logical_provider_calls": 0,
        "network_attempts": 0,
        "branches": {},
        "protocol_exclusions": [],
        "same_information_mpc": same_information_mpc(task),
    }
    for branch in item["branch_order"]:
        prompt = render_prompt(task, branch)
        payload = request_payload(spec, prompt)
        if digest(payload) != item["expected_request_sha256"][branch]:
            record["protocol_exclusions"].append(f"frozen_request_hash_mismatch:{branch}")
        response = await provider_call(
            client,
            spec,
            keys[model],
            payload,
            call_id=f"{item['block_id']}_{branch}",
        )
        record["logical_provider_calls"] += 1
        record["network_attempts"] += int(response["attempt_count"])
        if response.get("status") == "ok":
            parsed_obj = parse_output(str(response.get("content") or ""), horizon)
            parsed = {
                "ok": parsed_obj.ok,
                "raw_actions": parsed_obj.raw_actions,
                "dense_action_kw": parsed_obj.dense_action_kw,
                "diagnostics": parsed_obj.diagnostics,
                "horizon": parsed_obj.horizon,
                "segment_count": parsed_obj.segment_count,
            }
        else:
            parsed = failed_parse("provider_no_response", horizon)
        score = score_branch(task, branch, parsed["dense_action_kw"]) if parsed["ok"] else None
        if response.get("status") == "ok" and response.get("returned_model") != spec["expected_returned_model"]:
            record["protocol_exclusions"].append(f"wrong_returned_model:{branch}")
        record["branches"][branch] = {
            "carrier": "canonical" if branch.startswith("C") else "stale_soc",
            "prompt_text": prompt,
            "prompt_sha256": digest(prompt),
            "request_payload": payload,
            "request_payload_bytes_sha256": digest(payload),
            "response": response,
            "parsed": parsed,
            "score": score,
        }
    if record["branches"]["C1"]["request_payload_bytes_sha256"] != record["branches"]["C2"]["request_payload_bytes_sha256"]:
        record["protocol_exclusions"].append("canonical_duplicate_request_mismatch")
    if record["branches"]["S1"]["request_payload_bytes_sha256"] != record["branches"]["S2"]["request_payload_bytes_sha256"]:
        record["protocol_exclusions"].append("stale_duplicate_request_mismatch")
    record["evidence_root_sha256"] = digest(record)
    return record


def _completion_status(
    *,
    tier: str,
    records: list[dict[str, Any]],
    planned_blocks: int,
    planned_calls: int,
    partial_files: int,
) -> str:
    logical_calls = sum(int(item.get("logical_provider_calls", 0)) for item in records)
    all_branches = all(set(item.get("branches", {})) == set(BRANCHES) for item in records)
    exclusions = sum(len(item.get("protocol_exclusions", [])) for item in records)
    if len(records) != planned_blocks or logical_calls != planned_calls or not all_branches or partial_files or exclusions:
        return "HOLD"
    if tier == "smoke":
        accepted = all(
            branch["response"].get("status") == "ok"
            and branch["response"].get("content")
            and branch["response"].get("finish_reason") != "length"
            and branch["parsed"].get("ok") is True
            for item in records
            for branch in item["branches"].values()
        )
        return "SMOKE_PASS" if accepted else "SMOKE_FAILED"
    return "COMPLETED"


async def execute(
    run_dir: Path,
    plan: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    keys: dict[str, str],
    *,
    tier: str,
    concurrency: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    records_dir = run_dir / "records"
    records_dir.mkdir()
    atomic_write(
        run_dir / "RUN_IDENTITY.json",
        {
            "created_utc": utc_now(),
            "tier": tier,
            "manifest_sha256": sha256(DEFAULT_MANIFEST),
            "planned_blocks": len(plan),
            "planned_provider_calls": len(plan) * 4,
            "concurrency": concurrency,
            "provider_calls_performed": 0,
            "network_attempts": 0,
        },
    )
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for item in plan:
        queue.put_nowait(item)
    completed: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:

        async def worker() -> None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await run_block(item, specs, keys, client)
                    atomic_write(records_dir / f"{item['block_id']}.json", record)
                    async with lock:
                        completed.append(record)
                        print(
                            f"e2b-v2 {tier} blocks={len(completed)}/{len(plan)} "
                            f"logical_calls={sum(int(row['logical_provider_calls']) for row in completed)}/{len(plan) * 4}",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    partial_count = len(list(run_dir.rglob("*.partial")))
    status = _completion_status(
        tier=tier,
        records=completed,
        planned_blocks=len(plan),
        planned_calls=len(plan) * 4,
        partial_files=partial_count,
    )
    fingerprints: dict[str, set[str]] = {}
    for model in specs:
        fingerprints[model] = {
            str(branch["response"]["system_fingerprint"])
            for row in completed
            if row["model_condition"] == model
            for branch in row["branches"].values()
            if branch["response"].get("system_fingerprint") is not None
        }
    if any(len(values) > 1 for values in fingerprints.values()):
        status = "HOLD" if tier != "smoke" else "SMOKE_FAILED"
    summary = {
        "schema_version": "e2b_v2_execution_summary_v1",
        "status": status,
        "tier": tier,
        "completed_blocks": len(completed),
        "planned_blocks": len(plan),
        "provider_calls_performed": sum(int(item["logical_provider_calls"]) for item in completed),
        "planned_provider_calls": len(plan) * 4,
        "network_attempts": sum(int(item["network_attempts"]) for item in completed),
        "protocol_exclusions": sum(len(item["protocol_exclusions"]) for item in completed),
        "partial_files": partial_count,
        "fingerprints_by_model": {key: sorted(value) for key, value in fingerprints.items()},
        "selective_retry": False,
        "stage1_calls": 0,
    }
    atomic_write(run_dir / "E2B_V2_EXECUTION_SUMMARY.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tier", choices=("smoke", "F1"), default="F1")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-provider-calls", action="store_true")
    parser.add_argument("--run-label")
    args = parser.parse_args()
    manifest = verify_manifest(args.manifest)
    plan_key = "smoke_run_plan" if args.tier == "smoke" else "F1_run_plan"
    plan_path = PROJECT / manifest["artifacts"][plan_key]["path"]
    plan = load_jsonl(plan_path)
    expected_blocks = 4 if args.tier == "smoke" else 120
    if len(plan) != expected_blocks:
        raise SystemExit(f"{args.tier} plan cardinality mismatch")
    if args.preflight or not args.execute:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "execution_status": "HOLD_AWAITING_EXPLICIT_PROVIDER_AUTHORIZATION",
                    "tier": args.tier,
                    "planned_blocks": len(plan),
                    "planned_provider_calls": len(plan) * 4,
                    "provider_calls_performed": 0,
                    "network_attempts": 0,
                },
                indent=2,
            )
        )
        return
    if not args.authorize_provider_calls:
        raise SystemExit("provider execution requires --authorize-provider-calls in a separately authorized turn")
    if not args.run_label:
        raise SystemExit("--run-label is required")
    if not public_hash_gate_ready():
        raise SystemExit("provider execution requires the public file-hash gate")
    specs = load_model_specs()
    keys = {name: credential(spec, name) for name, spec in specs.items()}
    summary = asyncio.run(
        execute(
            RUN_ROOT / args.run_label,
            plan,
            specs,
            keys,
            tier=args.tier,
            concurrency=int(manifest["frozen_concurrency"]),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    required = "SMOKE_PASS" if args.tier == "smoke" else "COMPLETED"
    if summary["status"] != required:
        raise SystemExit(f"E2b-v2 {args.tier} entered {summary['status']}")


if __name__ == "__main__":
    main()
