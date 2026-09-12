#!/usr/bin/env python3
"""Compare V2 controls with frozen V1 controls without contacting any provider."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _equal(findings: list[dict[str, Any]], name: str, expected: Any, actual: Any, ref: str) -> None:
    findings.append({
        "variable": name,
        "expected": expected,
        "actual": actual,
        "status": "pass" if expected == actual else "fail",
        "evidence": ref,
    })


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "experiments_v2/configs/VARIABLE_FREEZE_SPEC.yaml"
    spec = _yaml(spec_path)
    findings: list[dict[str, Any]] = []
    source_hashes = []
    for source in spec["inherited_control_sources"]:
        path = root / str(source["path"])
        actual_hash = _sha256(path) if path.is_file() else None
        expected_hash = source["sha256"]
        source_hashes.append({"path": source["path"], "expected": expected_hash, "actual": actual_hash})
        findings.append({
            "variable": f"source_hash:{source['path']}",
            "expected": expected_hash,
            "actual": actual_hash,
            "status": "pass" if actual_hash == expected_hash else "fail",
            "evidence": str(spec_path.relative_to(root)),
        })

    base = _yaml(root / "configs/frozen_protocol.yaml")
    inherited = spec["inherited_values"]
    _equal(findings, "study_year", base["frozen_year"], inherited["study_year"], "configs/frozen_protocol.yaml")
    _equal(findings, "storage_timezone", base["storage_timezone"], inherited["internal_storage_timezone"], "configs/frozen_protocol.yaml")
    _equal(findings, "resolution_minutes", base["analysis_resolution_minutes"], inherited["resolution_minutes"], "configs/frozen_protocol.yaml")
    for key, expected in inherited["battery"].items():
        _equal(findings, f"battery.{key}", base["battery"].get(key), expected, "configs/frozen_protocol.yaml")
    _equal(findings, "selection_seed", base["selection"]["selection_seed"], inherited["selection_seed"], "configs/frozen_protocol.yaml")
    _equal(findings, "event_generation_seed", base["selection"]["event_generation_seed"], inherited["event_generation_seed"], "configs/frozen_protocol.yaml")
    _equal(findings, "bootstrap_seed", base["selection"]["bootstrap_seed"], inherited["bootstrap_seed"], "configs/frozen_protocol.yaml")
    _equal(findings, "event_families", base["scenario"]["event_families"], inherited["event_families"], "configs/frozen_protocol.yaml")

    models = _yaml(root / "experiments_v2/configs/models.yaml")
    final_models = _yaml(root / "protocol/MODEL_CONDITIONS_FINAL_V1.yaml")
    final_by_id = {row["condition_id"]: row for row in final_models["model_conditions"]}
    for condition_name in ("deepseek_formal", "qwen_flash"):
        condition = models["conditions"][condition_name]
        baseline = final_by_id.get(condition["condition_id"], {})
        for field in ("configured_model_id", "expected_returned_model", "endpoint", "temperature", "max_output_tokens"):
            baseline_field = "requested_model" if field == "configured_model_id" else field
            expected = baseline.get(baseline_field)
            actual = condition.get(field)
            if field in {"configured_model_id", "expected_returned_model"}:
                model_family_match = actual == expected or (
                    isinstance(actual, str)
                    and isinstance(expected, str)
                    and actual.startswith(f"{expected}-")
                )
                findings.append({
                    "variable": f"{condition_name}.{field}",
                    "expected": {"inherited_model_family": expected},
                    "actual": actual,
                    "status": "pass" if model_family_match else "fail",
                    "evidence": "protocol/MODEL_CONDITIONS_FINAL_V1.yaml; experiments_v2/configs/models.yaml",
                })
            else:
                _equal(
                    findings,
                    f"{condition_name}.{field}",
                    expected,
                    actual,
                    "protocol/MODEL_CONDITIONS_FINAL_V1.yaml",
                )

    offline = json.loads(
        (root / "experiments_v2/reports/OFFLINE_VERIFICATION.json").read_text(encoding="utf-8")
    )
    findings.append({
        "variable": "offline_network_boundary",
        "expected": {"provider_calls_performed": 0, "network_attempts": 0},
        "actual": {"provider_calls_performed": offline.get("provider_calls_performed"), "network_attempts": offline.get("network_attempts")},
        "status": "pass" if offline.get("provider_calls_performed") == 0 and offline.get("network_attempts") == 0 else "fail",
        "evidence": "experiments_v2/reports/OFFLINE_VERIFICATION.json",
    })
    model_conditions = models["conditions"]
    live_blockers = []
    for name, condition in model_conditions.items():
        if condition.get("snapshot_id"):
            continue
        policy = condition.get("run_scoped_identity_policy")
        if not isinstance(policy, dict) or condition.get("alias_status") != "rolling_hosted_interface_frozen_as_condition":
            live_blockers.append(f"{name}.identity_policy_missing")
    live_blockers.extend(
        f"{name}.endpoint_missing"
        for name, condition in model_conditions.items()
        if not condition.get("endpoint")
    )
    live_blockers.extend(
        f"{name}.model_id_missing"
        for name, condition in model_conditions.items()
        if not condition.get("configured_model_id")
    )
    result = {
        "status": "pass_with_live_hold" if not any(f["status"] == "fail" for f in findings) else "control_mismatch",
        "control_findings": findings,
        "source_hashes": source_hashes,
        "live_blockers": live_blockers,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "budget_hard_stop": False,
        "control_policy": "V2 additions are allowed only on declared axes; inherited controls must hash-match.",
    }
    output = root / "experiments_v2/reports/VARIABLE_CONTROL_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass_with_live_hold":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
