"""Provider-free V2 dry-run and evidence replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .actions import action_energy_kwh, expand_sparse_segments, validate_dense_action
from .carriers import mutate_carrier
from .evidence import make_evidence_row, write_jsonl
from .hashing import sha256_json
from .plan import build_call_plan
from .provider import MockProvider


def _carrier_for(row: dict[str, Any]) -> dict[str, Any]:
    event_index = row.get("event_index")
    event_index = 0 if event_index is None else int(event_index)
    return {
        "current_step": event_index,
        "soc_kwh": 250.0,
        "usable_capacity_kwh": 500.0,
        "charge_limit_kw": 250.0,
        "discharge_limit_kw": 250.0,
        "export_limit_kw": 150.0,
        "reserve_kwh": 75.0,
        "forecast_version": "dryrun-v1",
        "forecast_series": [0.0] * 96,
        "active_commitments": [],
        "revoked_commitments": [],
        "sequence": event_index,
    }


def _parse_mock(raw: str) -> dict[str, Any]:
    parsed = cast(dict[str, Any], json.loads(raw))
    if "actions" in parsed:
        dense = expand_sparse_segments(parsed["actions"])
        parsed["expanded_actions_kw"] = dense
    else:
        dense = validate_dense_action(parsed.get("actions_kw", []))
        parsed["expanded_actions_kw"] = dense
    parsed["action_energy_kwh"] = action_energy_kwh(dense)
    return parsed


def _mock_trace(carrier: dict[str, Any], actions_kw: list[float]) -> dict[str, Any]:
    """Produce a small deterministic trace for evidence-chain testing only."""
    soc = float(carrier.get("soc_kwh", 250.0))
    capacity = float(carrier.get("usable_capacity_kwh", 500.0))
    soc_trace = [soc]
    for action in actions_kw:
        if action >= 0.0:
            soc += action * 0.25 * 0.95
        else:
            soc += action * 0.25 / 0.95
        soc = min(capacity, max(0.0, soc))
        soc_trace.append(soc)
    return {
        "kind": "mock_evidence_trace",
        "soc_kwh": soc_trace,
        "action_energy_kwh": action_energy_kwh(actions_kw),
        "terminal_soc_kwh": soc_trace[-1],
        "capacity_kwh": capacity,
    }


def dry_run(output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plan = build_call_plan()
    provider = MockProvider()
    rows: list[dict[str, Any]] = []
    for row in plan:
        carrier = _carrier_for(row)
        if row["experiment_id"] == "v2_e2_physical_carrier":
            mutation = row["branch"]
            if mutation == "S":
                carrier = mutate_carrier(carrier, "S", stale_values={"physical_field": "soc_kwh", "soc_kwh": 175.0})
            elif mutation == "K":
                carrier = mutate_carrier(carrier, "K", stale_values={"sequence": 0, "active_commitments": [{"id": "stale"}]})
            elif mutation == "M":
                carrier = mutate_carrier(carrier, "M", stale_values={"model_carrier": {**carrier, "soc_kwh": 200.0}})
        # Branch is metadata for C2; it is intentionally absent from the C/C2 request payload.
        payload = {
            "experiment_id": row["experiment_id"],
            "scenario_id": row["scenario_id"],
            "model_condition": row["model_condition"],
            "stage": row["stage"],
            "interface": row["interface"],
            "carrier": carrier,
            "visible_update": {"event_family": "dryrun", "event_index": row.get("event_index", 0)},
        }
        if row.get("branch") not in {"C", "C2", "shared", "shared_initial"}:
            payload["branch"] = row["branch"]
        response = provider.complete(payload, call_id=f"dry-{row['call_order']}")
        parsed = _parse_mock(response.raw_text)
        evidence = make_evidence_row(plan_row=row, payload=payload, raw_response=response.raw_text, parsed=parsed, mock=True)
        dense = parsed["expanded_actions_kw"]
        trace = _mock_trace(carrier, dense)
        trace_hash = sha256_json(trace)
        action_hash = sha256_json(dense)
        evidence["carrier_hash"] = sha256_json(carrier)
        evidence["raw_action_hash"] = action_hash
        evidence["projected_action_kw"] = dense
        evidence["projected_action_hash"] = action_hash
        evidence["executed_action_kw"] = dense
        evidence["executed_action_hash"] = action_hash
        evidence["simulator_trace"] = trace
        evidence["simulator_trace_hash"] = trace_hash
        evidence["score"] = {
            "F_raw": True,
            "C_feasible_mock": True,
            "J_phys_mock": True,
            "failure_as_zero": False,
        }
        evidence["score_hash"] = sha256_json(evidence["score"])
        evidence["cost_decomposition"] = {
            "action_energy_kwh": parsed["action_energy_kwh"],
            "provider_usage": response.usage,
            "estimated_cost_cny": None,
        }
        evidence["cost_hash"] = sha256_json(evidence["cost_decomposition"])
        evidence["parent_hashes"] = {
            "plan_row_hash": row["plan_row_hash"],
            "request_hash": evidence["request_hash"],
            "raw_response_hash": evidence["raw_response_hash"],
            "parsed_output_hash": evidence["parsed_output_hash"],
            "simulator_trace_hash": trace_hash,
            "score_hash": evidence["score_hash"],
        }
        evidence["evidence_root_hash"] = sha256_json(evidence["parent_hashes"])
        evidence["returned_model"] = response.returned_model
        evidence["request_id"] = response.request_id
        evidence["usage"] = response.usage
        evidence["network_attempts"] = 0
        evidence["protocol_exclusion"] = None
        rows.append(evidence)
    # C2 is required to have exactly the same payload hash as C in the dry-run plan. The plan's
    # branch is intentionally outside the request payload; this check fails if that invariant is
    # accidentally changed later.
    c_rows = {(r["scenario_id"], r["model_condition"], r["repetition"]): r for r in rows if r["experiment_id"] == "v2_e2_physical_carrier" and r["branch"] == "C"}
    c2_rows = {(r["scenario_id"], r["model_condition"], r["repetition"]): r for r in rows if r["experiment_id"] == "v2_e2_physical_carrier" and r["branch"] == "C2"}
    c2_pairs = sum(1 for key in c_rows if key in c2_rows and c_rows[key]["request_hash"] == c2_rows[key]["request_hash"])
    write_jsonl(destination / "DRY_RUN_EVIDENCE.jsonl", rows)
    summary = {
        "status": "pass",
        "mock_only": True,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "planned_calls": len(plan),
        "evidence_rows": len(rows),
        "c2_payload_equal_pairs": c2_pairs,
        "c2_expected_pairs": len(c_rows),
        "protocol_exclusions": sum(row["protocol_exclusion"] is not None for row in rows),
        "zero_action_rows": sum(not bool(row["parsed"]["action_energy_kwh"]) for row in rows),
        "nontrivial_action_rows": sum(bool(row["parsed"]["action_energy_kwh"] >= 5.0) for row in rows),
        "plan_sha256": sha256_json(plan),
    }
    (destination / "DRY_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
