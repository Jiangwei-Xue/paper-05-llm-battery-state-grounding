#!/usr/bin/env python3
"""Freeze a Qwen3.7-Plus-only E2b F1/F0 plan without network access."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "reviews/qwen37plus_f1f0_v1"
PROTOCOL = ROOT / "protocol/qwen37plus_f1f0_v1/QWEN37PLUS_F1F0_PROTOCOL_V1.yaml"
PROTOCOL_MD = ROOT / "protocol/qwen37plus_f1f0_v1/QWEN37PLUS_F1F0_PROTOCOL_V1.md"
PROMPT = ROOT / "protocol/E2B_PROMPT_TEMPLATE_V2.txt"
SCHEMA = ROOT / "protocol/E2B_SCHEMA_V2.json"
PARSER = ROOT / "scripts/e2b_parser_v2.py"
COMMON = ROOT / "scripts/e2b_v2_common.py"
RUNNER = ROOT / "scripts/run_qwen37plus_f1f0_v1.py"
TEST = ROOT / "tests/test_qwen37plus_f1f0_protocol.py"
TASK_MANIFEST = ROOT / "reviews/e2b_protocol_v2/E2B_V2_TASK_HASH_MANIFEST.jsonl"
F0_SOURCE = ROOT / "reviews/e2b_protocol_v2/E2B_V2_F0_OPTIONAL_BRIDGE_PLAN.jsonl"
SEED = 20260814

sys.path.insert(0, str(ROOT / "scripts"))
from e2b_v2_common import digest, render_prompt, request_payload, sha256, visible_suffix_rows  # noqa: E402, I001


MODEL_SPEC: dict[str, Any] = {
    "condition_id": "qwen37_plus_direct_v1",
    "provider": "qwen",
    "configured_model_id": "qwen3.7-plus",
    "expected_returned_model": "qwen3.7-plus",
    "route": "direct",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "temperature": 0.0,
    "max_output_tokens": 16000,
    "attempt_timeout_seconds": 240,
    "retryable_http_statuses": [429, 502, 503, 504],
    "credentials_env": "QWEN_API_KEY",
    "fallback": "disabled",
    "tools": "disabled",
    "web_search": "disabled",
    "cache": "disabled",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def balanced_order(position: int) -> list[str]:
    square = (("C1", "S1", "C2", "S2"), ("S1", "C2", "S2", "C1"), ("C2", "S2", "C1", "S1"), ("S2", "C1", "S1", "C2"))
    return list(square[position % len(square)])


def load_tasks() -> list[dict[str, Any]]:
    rows = load_jsonl(TASK_MANIFEST)
    if len(rows) != 60 or len({str(row["scenario_id"]) for row in rows}) != 60:
        raise RuntimeError("expected 60 unique inherited E2b tasks")
    for row in rows:
        source = PROJECT / str(row["source_record_path"])
        if not source.is_file() or sha256(source) != row["source_record_sha256"]:
            raise RuntimeError(f"source record hash mismatch: {row['scenario_id']}")
    return sorted(rows, key=lambda row: str(row["scenario_id"]))


def f0_scenarios() -> list[str]:
    rows = load_jsonl(F0_SOURCE)
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    if len(scenarios) != 15:
        raise RuntimeError("expected 15 inherited F0 scenarios")
    return scenarios


def make_row(task_row: dict[str, Any], tier: str, position: int) -> dict[str, Any]:
    task = json.loads((PROJECT / str(task_row["source_record_path"])).read_text(encoding="utf-8"))["task"]
    if digest(task) != task_row["task_sha256"]:
        raise RuntimeError(f"task hash mismatch: {task_row['scenario_id']}")
    c_payload = request_payload(MODEL_SPEC, render_prompt(task, "C1"))
    s_payload = request_payload(MODEL_SPEC, render_prompt(task, "S1"))
    identity = {"protocol_id": "energybench-qwen37plus-e2b-sensitivity-v1", "tier": tier, "scenario_id": task_row["scenario_id"], "model_condition": "qwen37_plus_direct"}
    row = {
        **identity,
        "block_id": ("qwen37f1_" if tier == "F1" else "qwen37f0_") + digest(identity)[:20],
        "branch_order": balanced_order(position),
        "source_record_path": task_row["source_record_path"],
        "source_record_sha256": task_row["source_record_sha256"],
        "task_sha256": task_row["task_sha256"],
        "horizon": len(visible_suffix_rows(task)),
        "expected_request_sha256": {"C1": digest(c_payload), "C2": digest(c_payload), "S1": digest(s_payload), "S2": digest(s_payload)},
        "planned_provider_calls": 4,
        "run_order_seed": SEED,
    }
    row["plan_row_sha256"] = digest(row)
    return row


def source_release_state() -> tuple[str, list[str]]:
    """Return the portable provenance identity used by the public package."""
    return "public-scientific-file-manifest", []


def main() -> int:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PRE_CALL_FROZEN":
        raise RuntimeError("Qwen3.7-Plus protocol is not pre-call frozen")
    tasks = load_tasks()
    f0_ids = set(f0_scenarios())
    f1 = [make_row(row, "F1", i) for i, row in enumerate(tasks)]
    f0 = [make_row(row, "F0", i) for i, row in enumerate(tasks) if str(row["scenario_id"]) in f0_ids]
    if len(f1) != 60 or len(f0) != 15:
        raise RuntimeError("plan cardinality failure")
    all_rows = f1 + f0
    random.Random(SEED).shuffle(all_rows)
    for i, row in enumerate(all_rows):
        row["run_order"] = i
        row["plan_row_sha256"] = digest(row)
    smoke_ids = sorted({str(row["scenario_id"]) for row in f1}, key=lambda sid: (next(x["horizon"] for x in f1 if x["scenario_id"] == sid), sid))[:2]
    smoke = [dict(row, tier="smoke", block_id="qwen37smoke_" + digest({"scenario_id": row["scenario_id"]})[:20]) for row in f1 if str(row["scenario_id"]) in smoke_ids]
    for i, row in enumerate(smoke):
        row["run_order"] = i
        row["plan_row_sha256"] = digest(row)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "QWEN37PLUS_F1_RUN_PLAN.jsonl", [row for row in all_rows if row["tier"] == "F1"])
    write_jsonl(OUT / "QWEN37PLUS_F0_RUN_PLAN.jsonl", [row for row in all_rows if row["tier"] == "F0"])
    write_jsonl(OUT / "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl", all_rows)
    write_jsonl(OUT / "QWEN37PLUS_SMOKE_RUN_PLAN.jsonl", smoke)
    write_jsonl(OUT / "QWEN37PLUS_TASK_HASH_MANIFEST.jsonl", [{key: row[key] for key in ("scenario_id", "source_record_path", "source_record_sha256", "task_sha256")} for row in tasks])
    artifact_paths = {
        "protocol_yaml": PROTOCOL,
        "protocol_markdown": PROTOCOL_MD,
        "prompt_template": PROMPT,
        "schema": SCHEMA,
        "parser": PARSER,
        "common": COMMON,
        "runner": RUNNER,
        "test": TEST,
        "task_manifest": OUT / "QWEN37PLUS_TASK_HASH_MANIFEST.jsonl",
        "formal_plan": OUT / "QWEN37PLUS_FORMAL_RUN_PLAN.jsonl",
        "f1_plan": OUT / "QWEN37PLUS_F1_RUN_PLAN.jsonl",
        "f0_plan": OUT / "QWEN37PLUS_F0_RUN_PLAN.jsonl",
        "smoke_plan": OUT / "QWEN37PLUS_SMOKE_RUN_PLAN.jsonl",
        "source_task_manifest": TASK_MANIFEST,
        "source_f0_plan": F0_SOURCE,
    }
    artifacts = {label: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path), "size_bytes": path.stat().st_size} for label, path in artifact_paths.items()}
    write_json(OUT / "QWEN37PLUS_MODEL_SPEC.json", MODEL_SPEC)
    artifacts["model_spec"] = {"path": str((OUT / "QWEN37PLUS_MODEL_SPEC.json").relative_to(PROJECT)), "sha256": sha256(OUT / "QWEN37PLUS_MODEL_SPEC.json"), "size_bytes": (OUT / "QWEN37PLUS_MODEL_SPEC.json").stat().st_size}
    release_identity, changed_files = source_release_state()
    manifest = {
        "schema_version": "qwen37plus_f1f0_pre_call_manifest_v1",
        "protocol_id": protocol["protocol_id"],
        "status": "OFFLINE_GATES_PASS_EXECUTION_HOLD",
        "source_run_label": "e2_20260801T054708Z",
        "planned_f1_blocks": 60,
        "planned_f0_blocks": 15,
        "planned_formal_blocks": 75,
        "planned_formal_provider_calls": 300,
        "planned_smoke_blocks": 2,
        "planned_smoke_provider_calls": 8,
        "frozen_concurrency": 8,
        "transport_retry_maximum": 2,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "fallback": "disabled",
        "selective_retry": False,
        "formal_requires_smoke_pass": True,
        "formal_run_requires_explicit_user_authorization": True,
        "model_condition": MODEL_SPEC,
        "source_release_identity": release_identity,
        "source_files_verified_at_freeze": not changed_files,
        "source_file_mismatches_at_freeze": changed_files,
        "artifacts": artifacts,
        "historical_evidence_immutable": True,
    }
    write_json(OUT / "QWEN37PLUS_PRE_CALL_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "planned_formal_provider_calls": 300, "planned_smoke_provider_calls": 8, "provider_calls_performed": 0, "network_attempts": 0, "source_files_verified_at_freeze": not changed_files}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
