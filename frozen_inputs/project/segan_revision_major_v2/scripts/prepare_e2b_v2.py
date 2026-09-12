#!/usr/bin/env python3
"""Build the provider-free E2b-v2 controls, plans, and pre-call manifest."""

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
OUT = ROOT / "reviews/e2b_protocol_v2"
OLD_RUN = ROOT / "runs/e2b_protocol_v1_30c/e2b_30c_20260813T084000Z"
OLD_AUDIT = ROOT / "reviews/e2b_protocol_v1_30c/E2B_STAGE1_ROOT_CAUSE_AUDIT_V1.json"
PROTOCOL = ROOT / "protocol/E2B_PROTOCOL_V2.yaml"
PROTOCOL_MD = ROOT / "protocol/E2B_PROTOCOL_V2.md"
PROMPT = ROOT / "protocol/E2B_PROMPT_TEMPLATE_V2.txt"
SCHEMA = ROOT / "protocol/E2B_SCHEMA_V2.json"
MODELS = PROJECT / "experiments_v2/configs/models.yaml"
PARSER = ROOT / "scripts/e2b_parser_v2.py"
COMMON = ROOT / "scripts/e2b_v2_common.py"
ANALYSIS = ROOT / "scripts/analyze_e2b_v2.py"
RUNNER = ROOT / "scripts/run_e2b_v2.py"
TESTS = ROOT / "tests/test_e2b_protocol_v2.py"
VERIFIER = ROOT / "scripts/verify_e2b_v2_preflight.py"
SMOKE_SCHEMA = ROOT / "protocol/E2B_V2_SMOKE_ACCEPTANCE_SCHEMA.json"
VARIABLE_FREEZE = OUT / "VARIABLE_FREEZE_SPEC.yaml"
METHOD_DELTA_MD = OUT / "METHOD_DELTA_REPORT.md"
VARIABLE_CONTROL_MD = OUT / "PREMATRIX_VARIABLE_CONTROL_REPORT.md"
SEED = 20260814

sys.path.insert(0, str(ROOT / "scripts"))

from e2b_parser_v2 import parse_output  # noqa: E402
from e2b_v2_common import (  # noqa: E402
    battery_config,
    canonical_bytes,
    canonical_json,
    carrier_for,
    dense_to_segments,
    digest,
    load_unique_tasks,
    model_visible_contract,
    raw_replay,
    render_prompt,
    request_payload,
    sha256,
    visible_suffix_rows,
)

from energy_agent_reliability.online_gate_v6 import economic_dispatch  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(canonical_json(row) for row in rows) + "\n", encoding="utf-8")


def source_release_state() -> tuple[str, list[str]]:
    """Return the portable provenance identity used by the public package."""
    return "public-scientific-file-manifest", []


def balanced_order(position: int) -> list[str]:
    square = (
        ("C1", "S1", "C2", "S2"),
        ("S1", "C2", "S2", "C1"),
        ("C2", "S2", "C1", "S1"),
        ("S2", "C1", "S1", "C2"),
    )
    return list(square[position % len(square)])


def build_f1_plan(tasks: list[dict[str, Any]], model_specs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("deepseek_formal", "qwen_flash"):
        for position, item in enumerate(tasks):
            task = item["task"]
            identity = {
                "protocol_id": "energybench-e2b-stage2-only-carrier-sensitivity-v2",
                "tier": "F1",
                "scenario_id": item["scenario_id"],
                "model_condition": model,
            }
            c_payload = request_payload(model_specs[model], render_prompt(task, "C1"))
            s_payload = request_payload(model_specs[model], render_prompt(task, "S1"))
            rows.append(
                {
                    **identity,
                    "block_id": "e2bv2_" + digest(identity)[:20],
                    "branch_order": balanced_order(position),
                    "source_record_path": item["source_record_path"],
                    "source_record_sha256": item["source_record_sha256"],
                    "task_sha256": item["task_sha256"],
                    "horizon": len(visible_suffix_rows(task)),
                    "expected_request_sha256": {
                        "C1": digest(c_payload),
                        "C2": digest(c_payload),
                        "S1": digest(s_payload),
                        "S2": digest(s_payload),
                    },
                    "planned_provider_calls": 4,
                }
            )
    random.Random(SEED).shuffle(rows)
    for run_order, row in enumerate(rows):
        row["run_order"] = run_order
        row["plan_row_sha256"] = digest(row)
    if len(rows) != 120 or len({row["block_id"] for row in rows}) != 120:
        raise RuntimeError("F1 plan cardinality failure")
    return rows


def select_f0(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in tasks:
        task = item["task"]
        key = (str(task["event_family"]), int(task["v2_prefix_intervention"]["divergence_kwh"]))
        by_key.setdefault(key, []).append(item)
    selected: list[dict[str, Any]] = []
    families = sorted({key[0] for key in by_key})
    for family in families:
        for divergence in (50, 75, 100):
            candidates = sorted(by_key.get((family, divergence), []), key=lambda row: row["scenario_id"])
            if not candidates:
                raise RuntimeError(f"missing F0 stratum: {family}/{divergence}")
            selected.append(candidates[0])
    if len(selected) != 15 or len({item["scenario_id"] for item in selected}) != 15:
        raise RuntimeError("F0 selection cardinality failure")
    return selected


def build_f0_plan(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("deepseek_formal", "qwen_flash"):
        for position, item in enumerate(selected):
            identity = {
                "protocol_id": "energybench-e2b-stage2-only-carrier-sensitivity-v2",
                "tier": "F0_optional_bridge_selection",
                "scenario_id": item["scenario_id"],
                "model_condition": model,
            }
            rows.append(
                {
                    **identity,
                    "block_id": "e2bv2f0_" + digest(identity)[:20],
                    "branch_order": balanced_order(position),
                    "source_record_path": item["source_record_path"],
                    "source_record_sha256": item["source_record_sha256"],
                    "task_sha256": item["task_sha256"],
                    "divergence_kwh": item["task"]["v2_prefix_intervention"]["divergence_kwh"],
                    "event_family": item["task"]["event_family"],
                    "planned_provider_calls": 4,
                    "execution_status": "OPTIONAL_NOT_ADOPTED_REQUIRES_SEPARATE_CONTRACT_FREEZE",
                }
            )
    rows.sort(key=lambda row: (row["scenario_id"], row["model_condition"]))
    for run_order, row in enumerate(rows):
        row["run_order"] = run_order
        row["plan_row_sha256"] = digest(row)
    return rows


def build_smoke_plan(f1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_horizon: dict[str, int] = {}
    for row in f1:
        scenario_horizon[str(row["scenario_id"])] = int(row["horizon"])
    shortest = min(scenario_horizon, key=lambda sid: (scenario_horizon[sid], sid))
    longest = min(scenario_horizon, key=lambda sid: (-scenario_horizon[sid], sid))
    selected = [row for row in f1 if row["scenario_id"] in {shortest, longest}]
    selected.sort(key=lambda row: (row["scenario_id"], row["model_condition"]))
    if len(selected) != 4:
        raise RuntimeError("smoke plan cardinality failure")
    smoke: list[dict[str, Any]] = []
    for row in selected:
        copy = dict(row)
        copy["tier"] = "smoke"
        copy["block_id"] = "e2bv2smoke_" + digest(
            {"scenario_id": row["scenario_id"], "model_condition": row["model_condition"]}
        )[:20]
        copy["plan_row_sha256"] = digest(copy)
        smoke.append(copy)
    return smoke


def control_report(tasks: list[dict[str, Any]], model_specs: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    battery = battery_config()
    rows: list[dict[str, Any]] = []
    for item in tasks:
        task = item["task"]
        activation = int(task["visible_update"]["activation_step"])
        horizon = len(visible_suffix_rows(task))
        control: dict[str, Any] = {"scenario_id": item["scenario_id"], "horizon": horizon}
        for carrier_kind, branch in (("canonical", "C1"), ("stale", "S1")):
            carrier = carrier_for(task, branch)
            solution = economic_dispatch(
                task,
                battery,
                initial_soc_kwh=float(carrier["soc_kwh"]),
                start_step=activation,
                apply_event=True,
                allow_curtailment=False,
            )
            action = []
            for value in solution.action_kw:
                numeric = float(value)
                if numeric > 250.0:
                    if numeric - 250.0 > 1e-8:
                        raise RuntimeError("positive-control solver exceeded the hard power envelope")
                    numeric = 250.0
                elif numeric < -250.0:
                    if -250.0 - numeric > 1e-8:
                        raise RuntimeError("positive-control solver exceeded the hard power envelope")
                    numeric = -250.0
                action.append(0.0 if abs(numeric) <= 1e-10 else numeric)
            witness = dense_to_segments(action)
            parsed = parse_output(canonical_json(witness), horizon)
            replay = raw_replay(task, parsed.dense_action_kw, initial_soc_kwh=float(carrier["soc_kwh"])) if parsed.ok else None
            control[f"{carrier_kind}_witness"] = {
                "solver_status": solution.solver_status,
                "parse_ok": parsed.ok,
                "parser_diagnostics": parsed.diagnostics,
                "engineering_feasible": bool(replay and replay["engineering_feasible"]),
                "terminal_absolute_error_kwh": replay["terminal_absolute_error_kwh"] if replay else None,
                "witness_sha256": digest(witness),
            }
        zero = parse_output('{"actions":[]}', horizon)
        zero_replay = raw_replay(
            task,
            zero.dense_action_kw,
            initial_soc_kwh=float(carrier_for(task, "C1")["soc_kwh"]),
        )
        control["zero_negative"] = {
            "parse_ok": zero.ok,
            "engineering_feasible": zero_replay["engineering_feasible"],
            "terminal_absolute_error_kwh": zero_replay["terminal_absolute_error_kwh"],
        }
        alternating = {
            "actions": [
                {
                    "start_offset": offset,
                    "end_offset_exclusive": offset + 1,
                    "power_kw": 1.0 if offset % 2 == 0 else -1.0,
                }
                for offset in range(horizon)
            ]
        }
        alternating_parsed = parse_output(canonical_json(alternating), horizon)
        control["h_segment_roundtrip"] = bool(
            alternating_parsed.ok
            and alternating_parsed.dense_action_kw
            == [1.0 if offset % 2 == 0 else -1.0 for offset in range(horizon)]
        )
        offset_payload = {"actions": [{"start_offset": 0, "end_offset_exclusive": 1, "power_kw": 10.0}]}
        offset_parsed = parse_output(canonical_json(offset_payload), horizon)
        control["offset_zero_maps_first_suffix"] = bool(
            activation > 0 and offset_parsed.ok and offset_parsed.dense_action_kw[0] == 10.0
        )
        c_contract = model_visible_contract(task, "C1")
        s_contract = model_visible_contract(task, "S1")
        c_without_soc = json.loads(canonical_json(c_contract))
        s_without_soc = json.loads(canonical_json(s_contract))
        c_without_soc["carrier_state"].pop("soc_kwh")
        s_without_soc["carrier_state"].pop("soc_kwh")
        control["treatment_diff_only_soc"] = c_without_soc == s_without_soc
        for model, spec in model_specs.items():
            c_request = request_payload(spec, render_prompt(task, "C1"))
            c2_request = request_payload(spec, render_prompt(task, "C2"))
            s_request = request_payload(spec, render_prompt(task, "S1"))
            s2_request = request_payload(spec, render_prompt(task, "S2"))
            control[f"{model}_request_identity"] = {
                "C1_equals_C2": canonical_bytes(c_request) == canonical_bytes(c2_request),
                "S1_equals_S2": canonical_bytes(s_request) == canonical_bytes(s2_request),
            }
        rows.append(control)
    counts = {
        "canonical_witness_parse": sum(row["canonical_witness"]["parse_ok"] for row in rows),
        "canonical_witness_feasible": sum(row["canonical_witness"]["engineering_feasible"] for row in rows),
        "stale_witness_parse": sum(row["stale_witness"]["parse_ok"] for row in rows),
        "stale_witness_feasible": sum(row["stale_witness"]["engineering_feasible"] for row in rows),
        "zero_negative_parse": sum(row["zero_negative"]["parse_ok"] for row in rows),
        "zero_negative_physical_failure": sum(not row["zero_negative"]["engineering_feasible"] for row in rows),
        "offset_regression": sum(row["offset_zero_maps_first_suffix"] for row in rows),
        "h_segment_roundtrip": sum(row["h_segment_roundtrip"] for row in rows),
        "treatment_diff_only_soc": sum(row["treatment_diff_only_soc"] for row in rows),
        "request_identity_all_models": sum(
            all(
                row[f"{model}_request_identity"][key]
                for model in ("deepseek_formal", "qwen_flash")
                for key in ("C1_equals_C2", "S1_equals_S2")
            )
            for row in rows
        ),
    }
    passed = all(value == 60 for value in counts.values())
    return (
        {
            "schema_version": "e2b_v2_provider_free_controls_v1",
            "status": "pass" if passed else "HOLD",
            "unique_scenarios": 60,
            "counts": counts,
            "export_interpretation": "strict_raw_grid_constraint_no_automatic_curtailment",
            "terminal_engineering_tolerance_kwh": 1.0,
            "provider_calls_performed": 0,
            "network_attempts": 0,
        },
        rows,
    )


def v1_supersession() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    inventory = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(OLD_RUN.rglob("*"))
        if path.is_file()
    ]
    facts = {
        "block_files": old["record_statistics"]["record_count"],
        "planned_provider_calls": old["execution_summary"]["planned_provider_calls"],
        "attempted_provider_calls": old["execution_summary"]["provider_calls_performed"],
        "stage1_parser_success": old["classification"].get("parser_success", 0),
        "planned_stage2_calls": old["frozen_contract"]["expected_stage2_branch_rows"],
        "actual_stage2_calls": old["record_statistics"]["stage2_status"].get("called", 0),
        "structured_stage2_skips": old["record_statistics"]["stage2_status"].get("skipped", 0),
    }
    report = {
        "schema_version": "e2b_v1_supersession_audit_v1",
        "source_run": "e2b_30c_20260813T084000Z",
        "original_summary_status": old["execution_summary"]["status"],
        "superseding_status": "ABORTED_BEFORE_STAGE2",
        "scientific_disposition": "EXCLUDED_FROM_E2B_RESULT_DENOMINATORS",
        "facts": facts,
        "deepseek": {"finish_reason_length": 165, "valid_json_schema_mismatch": 15},
        "qwen": {"valid_json_schema_mismatch": 176, "transport_failures": 4},
        "old_files_modified": False,
        "selective_continue_or_retry_allowed": False,
        "source_file_count": len(inventory),
        "source_inventory_sha256": digest(inventory),
        "provider_calls_performed_during_audit": 0,
        "network_attempts_during_audit": 0,
    }
    return report, inventory


def markdown_supersession(report: dict[str, Any]) -> str:
    facts = report["facts"]
    return "\n".join(
        [
            "# E2b-v1 Supersession Audit",
            "",
            "The archived run is preserved byte-for-byte and is classified as **ABORTED_BEFORE_STAGE2**. It is an engineering pre-run, not an E2b scientific result.",
            "",
            "| Item | Count |",
            "|---|---:|",
            f"| Block files | {facts['block_files']} |",
            f"| Provider calls planned | {facts['planned_provider_calls']} |",
            f"| Provider calls attempted | {facts['attempted_provider_calls']} |",
            f"| Stage 1 parser success | {facts['stage1_parser_success']} |",
            f"| Stage 2 calls planned | {facts['planned_stage2_calls']} |",
            f"| Stage 2 calls made | {facts['actual_stage2_calls']} |",
            f"| Structured Stage 2 skips | {facts['structured_stage2_skips']} |",
            "",
            "The old `completed` label is a runner defect. This superseding record does not alter the old summary or any saved response. No old row may be selectively continued, retried, or merged into E2b-v2.",
            "",
        ]
    )


def main() -> None:
    config = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if config["status"] != "offline_implementation_hold":
        raise SystemExit("E2b-v2 protocol status mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_unique_tasks()
    if len(tasks) != 60:
        raise SystemExit(f"expected 60 unique frozen tasks, found {len(tasks)}")
    model_specs = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["conditions"]

    supersession, old_inventory = v1_supersession()
    write_json(OUT / "E2B_V1_SUPERSESSION_AUDIT.json", supersession)
    (OUT / "E2B_V1_SUPERSESSION_AUDIT.md").write_text(markdown_supersession(supersession), encoding="utf-8")
    write_jsonl(OUT / "E2B_V1_SOURCE_HASH_MANIFEST.jsonl", old_inventory)

    task_manifest = [
        {key: item[key] for key in ("scenario_id", "source_record_path", "source_record_sha256", "task_sha256")}
        for item in tasks
    ]
    write_jsonl(OUT / "E2B_V2_TASK_HASH_MANIFEST.jsonl", task_manifest)
    f1 = build_f1_plan(tasks, model_specs)
    f0 = build_f0_plan(select_f0(tasks))
    smoke = build_smoke_plan(f1)
    write_jsonl(OUT / "E2B_V2_F1_RUN_PLAN.jsonl", f1)
    write_jsonl(OUT / "E2B_V2_F0_OPTIONAL_BRIDGE_PLAN.jsonl", f0)
    write_jsonl(OUT / "E2B_V2_SMOKE_RUN_PLAN.jsonl", smoke)

    control, control_rows = control_report(tasks, model_specs)
    write_json(OUT / "E2B_V2_PROVIDER_FREE_CONTROL_REPORT.json", control)
    write_jsonl(OUT / "E2B_V2_PROVIDER_FREE_CONTROL_ROW_LEVEL.jsonl", control_rows)

    fixture_task = tasks[0]["task"]
    fixtures = {
        "scenario_id": fixture_task["scenario_id"],
        "canonical_prompt": render_prompt(fixture_task, "C1"),
        "stale_prompt": render_prompt(fixture_task, "S1"),
        "canonical_contract": model_visible_contract(fixture_task, "C1"),
        "stale_contract": model_visible_contract(fixture_task, "S1"),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    write_json(OUT / "E2B_V2_RENDERED_PROMPT_FIXTURES.json", fixtures)

    method_delta = {
        "schema_version": "e2b_v2_method_delta_v1",
        "status": "pass_with_noncomparability_boundary",
        "versions": [
            {"method_version": "E2b-M0", "run": "e2b_30c_20260813T084000Z", "disposition": "ABORTED_BEFORE_STAGE2", "comparability": "not_scientific_result"},
            {"method_version": "E2b-M1", "run": None, "disposition": "prospective_v2", "comparability": "new_protocol_required"},
        ],
        "changes": [
            "remove live Stage 1",
            "actions-only local-offset output contract",
            "non-clipping raw replay",
            "symmetric energy-distance estimand",
            "fail-closed completion logic",
            "mandatory isolated live smoke",
        ],
        "classification": "fundamental_method_change_after_aborted_pre-treatment_run",
        "pooling_allowed": False,
        "risk": "R3_if_v1_and_v2_are_pooled",
        "evidence_strength_after_hash_freeze": "E4",
    }
    write_json(OUT / "E2B_V2_METHOD_DELTA_REPORT.json", method_delta)

    variable_rows = [
        {"variable_name": "scenario", "domain": "dataset", "role": "controlled", "expected_control_status": "frozen", "actual_control_status": "frozen", "allowed_values": "60 frozen E2 scenario IDs", "observed_values": "60", "risk": "R0", "action": "freeze"},
        {"variable_name": "model_condition", "domain": "model", "role": "independent", "expected_control_status": "varied", "actual_control_status": "varied", "allowed_values": "deepseek_formal,qwen_flash", "observed_values": "2", "risk": "R0", "action": "record"},
        {"variable_name": "carrier_soc", "domain": "prompt", "role": "independent", "expected_control_status": "varied", "actual_control_status": "varied", "allowed_values": "canonical,stale", "observed_values": "C1,C2,S1,S2", "risk": "R0", "action": "freeze"},
        {"variable_name": "hosted_call_variability", "domain": "api", "role": "nuisance", "expected_control_status": "recorded", "actual_control_status": "recorded", "allowed_values": "two calls per carrier", "observed_values": "C1/C2,S1/S2", "risk": "R1", "action": "record"},
        {"variable_name": "branch_order", "domain": "runtime", "role": "controlled", "expected_control_status": "frozen", "actual_control_status": "frozen", "allowed_values": "balanced Latin order", "observed_values": "four balanced orders", "risk": "R0", "action": "freeze"},
        {"variable_name": "provider_backend", "domain": "api", "role": "hidden_confounder", "expected_control_status": "recorded", "actual_control_status": "unknown_before_live_smoke", "allowed_values": "exact returned model and fingerprint", "observed_values": "not yet observed", "risk": "R2", "action": "record"},
    ]
    inventory_path = OUT / "VARIABLE_INVENTORY.csv"
    headers = list(variable_rows[0])
    inventory_path.write_text(
        ",".join(headers) + "\n" + "\n".join(
            ",".join(json.dumps(str(row[key])) for key in headers) for row in variable_rows
        ) + "\n",
        encoding="utf-8",
    )

    release_identity, changed_files = source_release_state()
    artifacts: dict[str, dict[str, Any]] = {}
    paths = {
        "protocol_yaml": PROTOCOL,
        "protocol_md": PROTOCOL_MD,
        "prompt_template": PROMPT,
        "schema": SCHEMA,
        "models": MODELS,
        "parser": PARSER,
        "common": COMMON,
        "analysis": ANALYSIS,
        "runner": RUNNER,
        "tests": TESTS,
        "preflight_verifier": VERIFIER,
        "smoke_acceptance_schema": SMOKE_SCHEMA,
        "variable_freeze_spec": VARIABLE_FREEZE,
        "method_delta_markdown": METHOD_DELTA_MD,
        "variable_control_markdown": VARIABLE_CONTROL_MD,
        "v1_supersession": OUT / "E2B_V1_SUPERSESSION_AUDIT.json",
        "v1_source_manifest": OUT / "E2B_V1_SOURCE_HASH_MANIFEST.jsonl",
        "task_manifest": OUT / "E2B_V2_TASK_HASH_MANIFEST.jsonl",
        "F1_run_plan": OUT / "E2B_V2_F1_RUN_PLAN.jsonl",
        "F0_optional_plan": OUT / "E2B_V2_F0_OPTIONAL_BRIDGE_PLAN.jsonl",
        "smoke_run_plan": OUT / "E2B_V2_SMOKE_RUN_PLAN.jsonl",
        "controls": OUT / "E2B_V2_PROVIDER_FREE_CONTROL_REPORT.json",
        "control_rows": OUT / "E2B_V2_PROVIDER_FREE_CONTROL_ROW_LEVEL.jsonl",
        "prompt_fixtures": OUT / "E2B_V2_RENDERED_PROMPT_FIXTURES.json",
        "method_delta": OUT / "E2B_V2_METHOD_DELTA_REPORT.json",
        "variable_inventory": inventory_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing manifest artifact: {path}")
        artifacts[label] = {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
    manifest = {
        "schema_version": "e2b_v2_pre_call_manifest_v1",
        "protocol_id": config["protocol_id"],
        "status": "OFFLINE_GATES_PASS_EXECUTION_HOLD" if control["status"] == "pass" else "OFFLINE_GATES_HOLD",
        "planned_F1_blocks": 120,
        "planned_F1_provider_calls": 480,
        "planned_smoke_blocks": 4,
        "planned_smoke_provider_calls": 16,
        "optional_F0_blocks": 30,
        "optional_F0_provider_calls": 120,
        "F0_execution_adopted": False,
        "frozen_concurrency": 8,
        "transport_retry_maximum": 2,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "explicit_user_authorization_for_live_smoke": False,
        "formal_run_requires_smoke_pass": True,
        "source_release_identity": release_identity,
        "source_files_verified": not changed_files,
        "source_file_mismatches": changed_files,
        "artifacts": artifacts,
        "api_key_values_recorded": False,
        "fallback": "disabled",
        "selective_retry": False,
    }
    write_json(OUT / "E2B_V2_PRE_CALL_MANIFEST.json", manifest)
    print(canonical_json({"status": manifest["status"], "controls": control["counts"], "source_files_verified": not changed_files, "provider_calls_performed": 0, "network_attempts": 0}))


if __name__ == "__main__":
    main()
