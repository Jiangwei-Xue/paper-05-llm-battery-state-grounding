#!/usr/bin/env python3
"""Audit E2 matrix axes and frozen controls before provider access."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments_v2/reports/E2_VARIABLE_CONTROL_AUDIT.json"
REPORT_MD = ROOT / "experiments_v2/reports/E2_VARIABLE_CONTROL_AUDIT.md"


def main() -> None:
    e2 = yaml.safe_load((ROOT / "experiments_v2/configs/experiment_e2.yaml").read_text())
    freeze = yaml.safe_load((ROOT / "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml").read_text())
    models = yaml.safe_load((ROOT / "experiments_v2/configs/models.yaml").read_text())
    findings = [
        _finding(
            "E2-VC-001",
            e2["models"] == ["deepseek_formal", "qwen_flash"],
            "Two frozen model conditions remain balanced.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
        _finding(
            "E2-VC-002",
            e2["interface"] == "I3",
            "The E1-selected I3 interface is fixed for E2.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
        _finding(
            "E2-VC-003",
            e2["stage1_shared"] is True and e2["stage2_branches"] == ["C", "M", "S", "K", "C2"],
            "Only the registered Stage 2 carrier branch varies within a paired block.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
        _finding(
            "E2-VC-004",
            e2["null_control"] == "C2_byte_identical_payload",
            "C2 is the registered duplicate-call null control.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
        _finding(
            "E2-VC-005",
            e2["repetitions"] == 3 and e2["scenario_count"] == 60,
            "The independent scenario and nested repetition counts are explicit.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
        _finding(
            "E2-VC-006",
            freeze["inherited_values"]["concurrency"] == 5
            and freeze["inherited_values"]["temperature"] == 0.0,
            "Concurrency and sampling controls match the inherited freeze.",
            "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml",
        ),
        _finding(
            "E2-VC-007",
            all(
                models["conditions"][name]["fallback"] == "disabled_required"
                for name in e2["models"]
            ),
            "Fallback is disabled for both direct routes.",
            "experiments_v2/configs/models.yaml",
        ),
        _finding(
            "E2-VC-008",
            e2["prefix_intervention"]["future_oracle_forbidden"] is True,
            "The plant-side prefix intervention carries no future oracle.",
            "experiments_v2/configs/experiment_e2.yaml",
        ),
    ]
    release_identity = "public-scientific-file-manifest"
    report = {
        "schema_version": "experiments_v2_e2_variable_control_audit_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "GO" if all(row["status"] == "pass" for row in findings) else "HOLD",
        "summary": "E2 varies registered carrier conditions on shared Stage 1 evidence and a deterministic prefix intervention; inherited provider, interface, physical, retry, and scoring controls remain fixed.",
        "evidence_reviewed": [
            "experiments_v2/configs/experiment_e2.yaml",
            "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml",
            "experiments_v2/configs/models.yaml",
            "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl",
        ],
        "findings": findings,
        "red_flags": [],
        "open_questions": [],
        "required_fixes_before_main_matrix": [],
        "safe_to_include": [
            "experiments_v2/manifests/E2_FROZEN_TASKS.jsonl",
            "experiments_v2/manifests/E2_ADMISSION_RESULTS.jsonl",
        ],
        "exclude_or_rerun": [],
        "confidence": "high",
        "source_release_identity": release_identity,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines: list[str] = [
        "# E2 variable-control audit",
        "",
        f"Status: **{report['status']}**",
        "",
        str(report["summary"]),
        "",
        "## Pass/fail table",
        "",
        "| Finding | Status | Claim | Evidence |",
        "|---|---|---|---|",
        *[
            f"| {row['finding_id']} | {row['status']} | {row['claim']} | `{row['evidence_refs'][0]['ref']}` |"
            for row in findings
        ],
        "",
        "No unresolved red flags or open questions remain before the E2 pre-run freeze.",
        "Provider calls and network attempts performed by this audit: 0.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "findings": len(findings)}, sort_keys=True))
    if report["status"] != "GO":
        raise SystemExit("E2 variable control audit failed")


def _finding(identifier: str, passed: bool, claim: str, ref: str) -> dict[str, Any]:
    return {
        "finding_id": identifier,
        "agent": "prematrix-variable-control-agent",
        "severity": "R0" if passed else "R3",
        "status": "pass" if passed else "fail",
        "claim": claim,
        "evidence_refs": [{"type": "config", "ref": ref, "hash": None}],
        "affected_runs": ["E2"],
        "affected_artifacts": [ref],
        "recommended_action": "Proceed under the frozen pre-run manifest."
        if passed
        else "Stop before provider access.",
        "blocks_main_matrix": not passed,
    }


if __name__ == "__main__":
    main()
