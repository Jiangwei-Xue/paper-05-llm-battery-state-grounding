"""Frozen P0 task, prompt, parser, and plan contracts."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from .actions import ActionContractError, action_energy_kwh, expand_sparse_segments
from .hashing import canonical_json, sha256_json

PROTOCOL_VERSION = "energybench-v2.1-two-condition-20260801"
P0_EXPERIMENT_ID = "v2_p0_interface_pilot"
P0_RUN_ORDER_SEED = 20260803
INTERFACES = ("I0", "I1", "I2", "I3")
MODEL_CONDITIONS = ("deepseek_formal", "qwen_flash")
STATE_FIELDS = (
    "current_step",
    "soc_kwh",
    "usable_capacity_kwh",
    "charge_limit_kw",
    "discharge_limit_kw",
    "export_limit_kw",
    "reserve_kwh",
    "forecast_version",
    "active_commitments",
    "revoked_commitments",
    "sequence",
)


@dataclass(frozen=True)
class ParsedP0Output:
    ok: bool
    state: dict[str, Any] | None
    raw_action: Any
    dense_action_kw: list[float]
    diagnostics: list[str]


def build_p0_plan(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tasks) != 8 or len({str(task["scenario_id"]) for task in tasks}) != 8:
        raise ValueError("P0 requires eight unique frozen tasks")
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: str(item["scenario_id"])):
        for model in MODEL_CONDITIONS:
            for interface in INTERFACES:
                identity = {
                    "protocol_version": PROTOCOL_VERSION,
                    "experiment_id": P0_EXPERIMENT_ID,
                    "scenario_id": task["scenario_id"],
                    "model_condition": model,
                    "interface": interface,
                    "repetition": 0,
                }
                rows.append(
                    {
                        **identity,
                        "episode_id": f"p0_{sha256_json(identity)[:20]}",
                        "task_sha256": sha256_json(task),
                        "planned_stage_calls": 2,
                        "excluded_from_claim_bearing_analysis": True,
                    }
                )
    random.Random(P0_RUN_ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
        row["plan_row_sha256"] = sha256_json(row)
    return rows


def canonical_initial_state(task: dict[str, Any]) -> dict[str, Any]:
    visible = task["model_visible_episode"]["stage_1"]["visible_initial_state"]
    return {
        "current_step": 0,
        "soc_kwh": float(visible["soc_kwh"]),
        "usable_capacity_kwh": float(visible["usable_capacity_kwh"]),
        "charge_limit_kw": float(visible["max_charge_kw"]),
        "discharge_limit_kw": float(visible["max_discharge_kw"]),
        "export_limit_kw": float(visible["export_limit_kw"]),
        "reserve_kwh": float(visible["reserve_soc_kwh"]),
        "forecast_version": str(visible["active_forecast_version"]),
        "active_commitments": copy.deepcopy(visible.get("active_commitments", [])),
        "revoked_commitments": copy.deepcopy(
            visible.get("revoked_or_superseded_items", [])
        ),
        "sequence": 0,
    }


def canonical_event_state(
    task: dict[str, Any], event_soc_kwh: float
) -> dict[str, Any]:
    initial = canonical_initial_state(task)
    update = task["visible_update"]
    field_map = {
        "usable_capacity_kwh": "usable_capacity_kwh",
        "max_charge_kw": "charge_limit_kw",
        "max_discharge_kw": "discharge_limit_kw",
        "export_limit_kw": "export_limit_kw",
        "reserve_soc_kwh": "reserve_kwh",
    }
    state = copy.deepcopy(initial)
    state["current_step"] = int(update["activation_step"])
    state["soc_kwh"] = round(float(event_soc_kwh), 6)
    for source, target in field_map.items():
        if source in update:
            state[target] = float(update[source])
    if task["event_family"] == "forecast_revision":
        state["forecast_version"] = "forecast_revision_v2"
    changed = [
        {"field": source, "value": copy.deepcopy(update[source])}
        for source in field_map
        if source in update
    ]
    state["active_commitments"] = changed
    state["revoked_commitments"] = [
        {"field": item["field"], "previous_value": initial[field_map[item["field"]]]}
        for item in changed
    ]
    state["sequence"] = 1
    return state


def compile_prompt(
    task: dict[str, Any],
    interface: str,
    stage: str,
    canonical_state: dict[str, Any],
) -> str:
    if interface not in INTERFACES:
        raise ValueError(f"unknown P0 interface: {interface}")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unknown P0 stage: {stage}")
    if stage == "stage1":
        timeseries = task["model_visible_episode"]["stage_1"]["visible_timeseries"]
        visible_update: dict[str, Any] | None = None
    else:
        timeseries = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
        visible_update = task["visible_update"]
    horizon = len(timeseries)
    sparse = interface in {"I2", "I3"}
    action_contract: dict[str, Any]
    if sparse:
        action_contract = {
            "field": "actions",
            "type": "array",
            "maximum_segments": 24,
            "segment_fields": ["start_step", "end_step_exclusive", "power_kw"],
            "index_domain": [0, horizon],
            "end_step_exclusive": True,
            "overlap": "forbidden",
            "uncovered_steps": "deterministically_zero",
        }
    else:
        action_contract = {
            "field": "actions_kw",
            "type": "array_of_numbers",
            "required_length": horizon,
        }
    contract = {
        "task": "PV-BESS scheduling",
        "stage": stage,
        "scenario_id": task["scenario_id"],
        "resolution_minutes": 15,
        "action_convention": "positive_charge_negative_discharge_kw",
        "terminal_soc_target_kwh": 325.0,
        "current_state": canonical_state,
        "visible_update": visible_update,
        "visible_timeseries": timeseries,
        "output_contract": {
            "top_level_fields_exactly": ["state", action_contract["field"]],
            "state_fields_exactly": list(STATE_FIELDS),
            "action": action_contract,
            "prose_or_markdown": "forbidden",
        },
        "evaluation_boundary": {
            "raw_action_scored": True,
            "projection_not_disclosed": True,
            "oracle_not_available": True,
        },
    }
    header = (
        "Return exactly one compact JSON object. Respect SOC, capacity, charge and "
        "discharge power, reserve, export, and terminal-SOC constraints."
    )
    if interface == "I0":
        header += " Use direct text JSON."
    else:
        header += " The provider JSON-output channel is active."
    if interface == "I3":
        header += (
            " Independent format example only: a 10 kW charge over local steps 0 through 3 "
            "is {\"start_step\":0,\"end_step_exclusive\":4,\"power_kw\":10}. "
            "Do not copy it as the task answer."
        )
    return f"{header}\n{canonical_json(contract)}"


def build_payload(
    *,
    model_id: str,
    provider: str,
    interface: str,
    prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if interface != "I0":
        payload["response_format"] = {"type": "json_object"}
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "qwen":
        payload["enable_thinking"] = False
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return payload


def parse_output(raw_text: str, interface: str, horizon: int) -> ParsedP0Output:
    diagnostics: list[str] = []
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return ParsedP0Output(False, None, None, [], ["invalid_json"])
    if not isinstance(value, dict):
        return ParsedP0Output(False, None, None, [], ["top_level_not_object"])
    action_field = "actions" if interface in {"I2", "I3"} else "actions_kw"
    if set(value) != {"state", action_field}:
        diagnostics.append("top_level_fields_invalid")
    state = value.get("state")
    if not isinstance(state, dict) or set(state) != set(STATE_FIELDS):
        diagnostics.append("state_fields_invalid")
        parsed_state = None
    else:
        parsed_state = state
        diagnostics.extend(_state_diagnostics(state))
    raw_action = value.get(action_field)
    dense: list[float] = []
    try:
        if interface in {"I2", "I3"}:
            if not isinstance(raw_action, list):
                raise ActionContractError("segments_not_array")
            dense = expand_sparse_segments(raw_action, horizon=horizon)
        else:
            dense = _validate_dense_dynamic(raw_action, horizon)
    except ActionContractError as exc:
        diagnostics.append(str(exc))
    return ParsedP0Output(not diagnostics, parsed_state, raw_action, dense, diagnostics)


def parsed_payload(parsed: ParsedP0Output) -> dict[str, Any]:
    return {**asdict(parsed), "action_energy_kwh": action_energy_kwh(parsed.dense_action_kw)}


def _validate_dense_dynamic(value: Any, horizon: int) -> list[float]:
    if not isinstance(value, list):
        raise ActionContractError("dense_not_array")
    if len(value) != horizon:
        raise ActionContractError(f"dense_length_{len(value)}_expected_{horizon}")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ActionContractError(f"dense_{index}_not_number")
        number = float(item)
        if not (-250.0 <= number <= 250.0):
            raise ActionContractError(f"dense_{index}_out_of_range")
        result.append(number)
    return result


def _state_diagnostics(state: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    numeric = (
        "soc_kwh",
        "usable_capacity_kwh",
        "charge_limit_kw",
        "discharge_limit_kw",
        "export_limit_kw",
        "reserve_kwh",
    )
    if isinstance(state["current_step"], bool) or not isinstance(state["current_step"], int):
        diagnostics.append("state_current_step_invalid")
    if isinstance(state["sequence"], bool) or not isinstance(state["sequence"], int):
        diagnostics.append("state_sequence_invalid")
    for field in numeric:
        if isinstance(state[field], bool) or not isinstance(state[field], (int, float)):
            diagnostics.append(f"state_{field}_invalid")
    if not isinstance(state["forecast_version"], str) or not state["forecast_version"]:
        diagnostics.append("state_forecast_version_invalid")
    for field in ("active_commitments", "revoked_commitments"):
        if not isinstance(state[field], list) or any(not isinstance(item, dict) for item in state[field]):
            diagnostics.append(f"state_{field}_invalid")
    return diagnostics
