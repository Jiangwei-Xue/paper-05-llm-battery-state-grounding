#!/usr/bin/env python3
"""Check the V2 pre-call gate without contacting a provider or printing secrets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    models = _load_yaml(root / "experiments_v2/configs/models.yaml")
    blockers: list[str] = []
    conditions = models.get("conditions", {})
    if not isinstance(conditions, dict):
        blockers.append("models.conditions_missing")
        conditions = {}
    for name in ("deepseek_formal", "qwen_flash"):
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            blockers.append(f"{name}.condition_missing")
            continue
        snapshot_id = condition.get("snapshot_id")
        alias_status = condition.get("alias_status")
        endpoint = condition.get("endpoint")
        model_id = condition.get("configured_model_id")
        if not model_id:
            blockers.append(f"{name}.configured_model_id_missing")
        if not condition.get("expected_returned_model"):
            blockers.append(f"{name}.expected_returned_model_missing")
        if not snapshot_id:
            policy = condition.get("run_scoped_identity_policy")
            if alias_status != "rolling_hosted_interface_frozen_as_condition":
                blockers.append(f"{name}.rolling_alias_not_frozen_as_condition")
            if not isinstance(policy, dict):
                blockers.append(f"{name}.run_scoped_identity_policy_missing")
            else:
                required = {
                    "returned_model_must_equal": model_id,
                    "record_request_and_response_utc": True,
                    "record_system_fingerprint_when_available": True,
                    "stop_on_non_null_fingerprint_drift": True,
                    "future_live_bit_reproducibility_claim": "forbidden",
                    "pool_with_historical_preview": "forbidden",
                }
                for key, expected in required.items():
                    if policy.get(key) != expected:
                        blockers.append(f"{name}.identity_policy_{key}_invalid")
        elif alias_status != "immutable_provider_snapshot":
            blockers.append(f"{name}.snapshot_alias_status_invalid")
        if not endpoint:
            blockers.append(f"{name}.endpoint_missing")
        if condition.get("fallback") != "disabled_required":
            blockers.append(f"{name}.fallback_policy_unfrozen")
        if condition.get("tools") != "disabled_required":
            blockers.append(f"{name}.tools_policy_unfrozen")
        if condition.get("web_search") != "disabled_required":
            blockers.append(f"{name}.web_search_policy_unfrozen")
    env_presence = {
        name: bool(os.environ.get(str(condition.get("credentials_env", ""))))
        for name, condition in conditions.items()
        if isinstance(condition, dict)
    }
    offline_report = root / "experiments_v2/reports/OFFLINE_VERIFICATION.json"
    if not offline_report.is_file():
        blockers.append("offline_verification_missing")
    else:
        offline = json.loads(offline_report.read_text(encoding="utf-8"))
        if offline.get("status") != "pass":
            blockers.append("offline_verification_not_pass")
        if offline.get("provider_calls_performed") != 0 or offline.get("network_attempts") != 0:
            blockers.append("offline_report_records_network_activity")
    preparation_path = root / "experiments_v2/reports/P0_PREPARATION_REPORT.json"
    if not preparation_path.is_file():
        blockers.append("p0_preparation_report_missing")
    else:
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        if preparation.get("status") != "pass" or preparation.get("task_count") != 8:
            blockers.append("p0_preparation_not_pass")
        if not preparation.get("zero_action_infeasible_all"):
            blockers.append("p0_zero_action_gate_failed")
        if not preparation.get("canonical_two_stage_feasible_all"):
            blockers.append("p0_canonical_feasibility_gate_failed")
    manifest_path = root / "experiments_v2/manifests/P0_PRE_RUN_MANIFEST.json"
    if not manifest_path.is_file():
        blockers.append("p0_pre_run_manifest_missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("planned_logical_episodes") != 64 or manifest.get("planned_primary_calls") != 128:
            blockers.append("p0_run_plan_cardinality_mismatch")
        for relative, expected in manifest.get("file_hashes", {}).items():
            path = root / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual != expected:
                blockers.append(f"p0_frozen_hash_mismatch:{relative}")
    pricing = _load_yaml(root / "experiments_v2/configs/pricing.yaml")
    if pricing.get("status") != "official_snapshot_recorded":
        blockers.append("official_pricing_snapshot_missing")
    for name in ("deepseek_formal", "qwen_flash"):
        row = pricing.get("conditions", {}).get(name, {})
        if row.get("input_price_per_million") is None or row.get("output_price_per_million") is None:
            blockers.append(f"{name}.pricing_incomplete")
    result = {
        "status": "pass" if not blockers else "HOLD",
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "budget_hard_stop": False,
        "credential_presence_only": env_presence,
        "credential_values_emitted": False,
        "blockers": blockers,
        "next_action": (
            "freeze and verify the DeepSeek identity rule and pre-call manifest"
            if blockers
            else "await separate live-call authorization"
        ),
    }
    output = root / "experiments_v2/reports/PRECALL_GATE_REPORT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
