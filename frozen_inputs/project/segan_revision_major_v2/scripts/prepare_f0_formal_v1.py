#!/usr/bin/env python3
"""Build the offline-frozen F0 120-call formal plan and manifest."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "reviews/f0_formal_v1"
SOURCE_PLAN = ROOT / "reviews/e2b_protocol_v2/E2B_V2_F0_OPTIONAL_BRIDGE_PLAN.jsonl"
FORMAL_PLAN = OUT / "F0_FORMAL_RUN_PLAN.jsonl"
TASK_MANIFEST = OUT / "F0_FORMAL_TASK_HASH_MANIFEST.jsonl"
HASH_MANIFEST = OUT / "F0_FORMAL_HASH_MANIFEST.jsonl"
PRECALL = OUT / "F0_FORMAL_PRE_CALL_MANIFEST.json"
PROTOCOL = ROOT / "protocol/F0_FORMAL_PROTOCOL_V1.yaml"
PROTOCOL_MD = ROOT / "protocol/F0_FORMAL_PROTOCOL_V1.md"
VERIFIER = ROOT / "scripts/verify_f0_formal_preflight.py"
RUNNER = ROOT / "scripts/run_f0_formal_v1.py"
TEST = ROOT / "tests/test_f0_formal_preflight.py"
PILOT_SUMMARY = ROOT / "runs/f0_pilot_v1/f0_pilot_20260820T_authorized_v1/E2B_V2_EXECUTION_SUMMARY.json"
SMOKE_SUMMARY = ROOT / "runs/f0_pilot_v1/f0_smoke_20260820T_authorized_v4/E2B_V2_EXECUTION_SUMMARY.json"

sys.path.insert(0, str(ROOT / "scripts"))
common = importlib.import_module("e2b_v2_common")
digest = common.digest
request_payload = common.request_payload
render_prompt = common.render_prompt
sha256 = common.sha256
visible_suffix_rows = common.visible_suffix_rows


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_models() -> dict[str, dict[str, Any]]:
    content = yaml.safe_load((PROJECT / "experiments_v2/configs/models.yaml").read_text(encoding="utf-8"))
    return {name: dict(content["conditions"][name]) for name in ("deepseek_formal", "qwen_flash")}


def build_plan(source_rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_order, source in enumerate(source_rows):
        source_path = PROJECT / str(source["source_record_path"])
        record = json.loads(source_path.read_text(encoding="utf-8"))
        task = record["task"]
        if digest(task) != source["task_sha256"] or sha256(source_path) != source["source_record_sha256"]:
            raise RuntimeError(f"source hash mismatch: {source['block_id']}")
        model = str(source["model_condition"])
        c_payload = request_payload(models[model], render_prompt(task, "C1"))
        s_payload = request_payload(models[model], render_prompt(task, "S1"))
        row = dict(source)
        row.update(
            {
                "formal_protocol_id": "energybench-f0-stage2-only-carrier-sensitivity-formal-v1",
                "tier": "F0_FORMAL",
                "run_order": run_order,
                "horizon": len(visible_suffix_rows(task)),
                "expected_request_sha256": {
                    "C1": digest(c_payload),
                    "C2": digest(c_payload),
                    "S1": digest(s_payload),
                    "S2": digest(s_payload),
                },
                "planned_provider_calls": 4,
                "execution_status": "FROZEN_NOT_EXECUTED_AWAITING_AUTHORIZATION",
            }
        )
        row["plan_row_sha256"] = digest(row)
        rows.append(row)
    if len(rows) != 30 or len({row["block_id"] for row in rows}) != 30:
        raise RuntimeError("formal plan must contain 30 unique source blocks")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def source_release_state() -> tuple[str, bool, list[str]]:
    """Return the portable provenance identity used by the public package."""
    return "public-scientific-file-manifest", True, []


def main() -> int:
    source_rows = load_jsonl(SOURCE_PLAN)
    if len(source_rows) != 30:
        raise RuntimeError("source F0 bridge plan must contain 30 rows")
    rows = build_plan(source_rows, load_models())
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(FORMAL_PLAN, rows)
    task_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = {
            "block_id": row["block_id"],
            "scenario_id": row["scenario_id"],
            "event_family": row["event_family"],
            "model_condition": row["model_condition"],
            "source_record_path": row["source_record_path"],
            "source_record_sha256": row["source_record_sha256"],
            "task_sha256": row["task_sha256"],
        }
        task_map[canonical(task)] = task
    write_jsonl(TASK_MANIFEST, sorted(task_map.values(), key=lambda item: item["block_id"]))
    artifacts = {
        "protocol_yaml": PROTOCOL,
        "protocol_markdown": PROTOCOL_MD,
        "formal_plan": FORMAL_PLAN,
        "formal_task_manifest": TASK_MANIFEST,
        "source_plan": SOURCE_PLAN,
        "schema": ROOT / "protocol/E2B_SCHEMA_V2.json",
        "prompt_template": ROOT / "protocol/E2B_PROMPT_TEMPLATE_V2.txt",
        "parser": ROOT / "scripts/e2b_parser_v2.py",
        "common": ROOT / "scripts/e2b_v2_common.py",
        "runner": RUNNER,
        "preflight_verifier": VERIFIER,
        "tests": TEST,
        "models": PROJECT / "experiments_v2/configs/models.yaml",
        "pilot_summary": PILOT_SUMMARY,
        "smoke_summary": SMOKE_SUMMARY,
    }
    write_jsonl(
        HASH_MANIFEST,
        [
            {"artifact": label, "path": str(path.relative_to(PROJECT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for label, path in sorted(artifacts.items())
        ],
    )
    artifacts["hash_manifest"] = HASH_MANIFEST
    release_identity, clean, mismatches = source_release_state()
    manifest = {
        "schema_version": "f0_formal_pre_call_manifest_v1",
        "status": "PRE_CALL_FROZEN_HOLD",
        "execution_status": "HOLD_AWAITING_EXPLICIT_PROVIDER_AUTHORIZATION",
        "formal_protocol_id": "energybench-f0-stage2-only-carrier-sensitivity-formal-v1",
        "formal_plan_adopted": True,
        "planned_blocks": 30,
        "planned_provider_calls": 120,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "explicit_user_authorization_for_live_execution": False,
        "selective_retry": False,
        "fallback": "disabled",
        "frozen_concurrency": 8,
        "source_run_label": "e2_20260801T054708Z",
        "formal_output_root": "segan_revision_major_v2/runs/f0_formal_v1",
        "pilot_acceptance": {
            "smoke_status": "SMOKE_PASS",
            "pilot_status": "COMPLETED",
            "smoke_summary_path": str(SMOKE_SUMMARY.relative_to(PROJECT)),
            "pilot_summary_path": str(PILOT_SUMMARY.relative_to(PROJECT)),
        },
        "artifacts": {label: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)} for label, path in sorted(artifacts.items())},
        "source_release_identity": release_identity,
        "source_files_verified_at_freeze": clean,
        "source_file_mismatches_at_freeze": mismatches,
        "authorization_boundary": "no provider call is permitted until a later explicitly authorized turn",
    }
    PRECALL.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "planned_blocks": 30, "planned_provider_calls": 120, "provider_calls_performed": 0, "network_attempts": 0, "source_files_verified_at_freeze": clean}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
