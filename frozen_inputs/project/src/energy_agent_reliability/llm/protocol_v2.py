"""Frozen candidate controls for excluded protocol validation V2."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from .models import PromptBundle, ProviderRequest, StateMethodName
from .request import canonical_json, sha256_text, stable_id

V2_PROTOCOL_ID = "energybench_llm_protocol_v2"
V2_RUN_ORDER_SEED = 20260717
V2_COMMON_OUTPUT_CAP = 16000
V2_STAGE_WALL_TIMEOUT_SECONDS = 600
V2_EPISODE_WALL_TIMEOUT_SECONDS = 1260
V2_ATTEMPT_TIMEOUT_SECONDS = 240
V2_MAX_TRANSPORT_ATTEMPTS = 3
V2_BACKOFF_SECONDS = (1.0, 2.0)
V2_RETRYABLE_HTTP_STATUS = {429, 502, 503, 504}
V2_STATE_METHODS: tuple[StateMethodName, ...] = (
    "rolling_summary",
    "visible_carry",
    "typed_state",
    "canonical_typed_carry",
)


@dataclass(frozen=True)
class ProviderSpecV2:
    condition_id: str
    provider: str
    endpoint: str
    requested_model: str
    expected_returned_model: str
    api_key_envs: tuple[str, ...]
    reasoning_control: dict[str, Any]
    verified_max_output_tokens: int
    official_model_documentation: str
    official_json_documentation: str
    temperature: float = 0.0
    max_output_tokens: int = V2_COMMON_OUTPUT_CAP
    attempt_timeout_seconds: int = V2_ATTEMPT_TIMEOUT_SECONDS
    stage_wall_timeout_seconds: int = V2_STAGE_WALL_TIMEOUT_SECONDS
    episode_wall_timeout_seconds: int = V2_EPISODE_WALL_TIMEOUT_SECONDS
    max_transport_attempts: int = V2_MAX_TRANSPORT_ATTEMPTS


def provider_specs_v2() -> tuple[ProviderSpecV2, ...]:
    return (
        ProviderSpecV2(
            condition_id="v2_deepseek_v4_flash_direct",
            provider="deepseek",
            endpoint="https://api.deepseek.com/chat/completions",
            requested_model="deepseek-v4-flash",
            expected_returned_model="deepseek-v4-flash",
            api_key_envs=("DEEPSEEK_API_KEY",),
            reasoning_control={"thinking": {"type": "disabled"}},
            verified_max_output_tokens=384000,
            official_model_documentation="https://api-docs.deepseek.com/quick_start/pricing/",
            official_json_documentation="https://api-docs.deepseek.com/guides/json_mode/",
        ),
        ProviderSpecV2(
            condition_id="v2_qwen36_flash_beijing_direct",
            provider="qwen",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            requested_model="qwen3.6-flash",
            expected_returned_model="qwen3.6-flash",
            api_key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            reasoning_control={"enable_thinking": False},
            verified_max_output_tokens=64000,
            official_model_documentation=(
                "https://www.alibabacloud.com/help/en/model-studio/text-generation-model"
            ),
            official_json_documentation=(
                "https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output"
            ),
        ),
    )


def build_validation_plan_v2(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(pilots, key=lambda item: item["scenario_id"]):
        task_hash = sha256_text(canonical_json(candidate))
        for spec in provider_specs_v2():
            for method in V2_STATE_METHODS:
                identity = {
                    "protocol_id": V2_PROTOCOL_ID,
                    "scenario_id": candidate["scenario_id"],
                    "condition_id": spec.condition_id,
                    "state_method": method,
                    "repetition": 0,
                }
                rows.append(
                    {
                        **identity,
                        "run_id": stable_id("xpv2", identity),
                        "task_sha256": task_hash,
                        "planned_primary_calls": 2,
                        "excluded_from_formal_analysis": True,
                    }
                )
    random.Random(V2_RUN_ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
    return rows


def build_provider_request_v2(
    bundle: PromptBundle, run_id: str, spec: ProviderSpecV2
) -> ProviderRequest:
    payload: dict[str, Any] = {
        "model": spec.requested_model,
        "stream": False,
        "tools": [],
        "tool_choice": "none",
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
        "messages": [{"role": "user", "content": bundle.rendered_prompt}],
        "response_format": {"type": "json_object"},
        **spec.reasoning_control,
    }
    return ProviderRequest(
        stage=bundle.stage,
        run_id=run_id,
        condition_id=spec.condition_id,
        requested_model=spec.requested_model,
        payload=payload,
        payload_sha256=sha256_text(canonical_json(payload)),
        timeout_seconds=spec.attempt_timeout_seconds,
    )


def provider_manifest_v2() -> list[dict[str, Any]]:
    return [
        {**asdict(spec), "api_key_envs": list(spec.api_key_envs), "api_key_value_recorded": False}
        for spec in provider_specs_v2()
    ]
