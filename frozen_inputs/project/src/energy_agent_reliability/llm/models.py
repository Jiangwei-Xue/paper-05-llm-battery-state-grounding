"""Typed LLM experiment condition and run-plan records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

StageName = Literal["stage1", "stage2"]
StateMethodName = Literal["rolling_summary", "visible_carry", "typed_state", "canonical_typed_carry"]


@dataclass(frozen=True)
class ModelCondition:
    condition_id: str
    provider: str
    api_route: str
    requested_model: str
    expected_returned_model: str
    model_family: str
    route_type: str
    structured_output_support: str
    temperature_control: str
    reasoning_control: str
    tool_and_web_control: str
    fallback_control: str
    timeout_seconds: int
    retry_policy: str
    max_input_tokens: int | None
    max_output_tokens: int | None
    pricing_snapshot_source: str
    pricing_retrieval_utc: str | None
    data_use_boundary: str
    admission_status: str
    exclusion_reason: str | None


@dataclass(frozen=True)
class PromptBundle:
    stage: StageName
    scenario_id: str
    state_method: StateMethodName
    canonical_prompt: dict[str, Any]
    rendered_prompt: str
    prompt_sha256: str
    model_visible_field_manifest: list[str]
    oracle_leak_scan: dict[str, Any]


@dataclass(frozen=True)
class ProviderRequest:
    stage: StageName
    run_id: str
    condition_id: str
    requested_model: str
    payload: dict[str, Any]
    payload_sha256: str
    timeout_seconds: int


@dataclass(frozen=True)
class ProviderResponse:
    status: Literal["ok", "timeout", "transport_error", "provider_error"]
    raw_text: str
    returned_model: str | None
    usage: dict[str, int]
    metadata: dict[str, Any]
    error_type: str | None = None


@dataclass(frozen=True)
class ParserResult:
    ok: bool
    parsed_state: dict[str, Any] | str | None
    dispatch_kw: list[float]
    diagnostics: list[str]
    protocol_exclusion: str | None = None


@dataclass(frozen=True)
class PilotRunPlanRow:
    scenario_id: str
    episode_id: str
    run_id: str
    model_condition_id: str
    state_method: StateMethodName
    repetition: int
    frozen_task_hash: str
    stage1_prompt_sha256: str
    stage2_prompt_template_sha256: str
    planned_primary_calls: int
    run_order: int

