"""Excluded protocol validation V3 controls.

V3 keeps the V2 logical experiment matrix and provider controls fixed. The only
method changes are Decimal schema validation, route aggregation semantics, and
gate/model-outcome classification.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from .models import PromptBundle
from .protocol_v2 import (
    V2_ATTEMPT_TIMEOUT_SECONDS,
    V2_BACKOFF_SECONDS,
    V2_COMMON_OUTPUT_CAP,
    V2_EPISODE_WALL_TIMEOUT_SECONDS,
    V2_MAX_TRANSPORT_ATTEMPTS,
    V2_RETRYABLE_HTTP_STATUS,
    V2_RUN_ORDER_SEED,
    V2_STAGE_WALL_TIMEOUT_SECONDS,
    V2_STATE_METHODS,
    ProviderSpecV2,
    build_provider_request_v2,
    provider_specs_v2,
)
from .request import canonical_json, sha256_text, stable_id

V3_PROTOCOL_ID = "energybench_llm_protocol_v3"
V3_RUN_ORDER_SEED = V2_RUN_ORDER_SEED
V3_COMMON_OUTPUT_CAP = V2_COMMON_OUTPUT_CAP
V3_STAGE_WALL_TIMEOUT_SECONDS = V2_STAGE_WALL_TIMEOUT_SECONDS
V3_EPISODE_WALL_TIMEOUT_SECONDS = V2_EPISODE_WALL_TIMEOUT_SECONDS
V3_ATTEMPT_TIMEOUT_SECONDS = V2_ATTEMPT_TIMEOUT_SECONDS
V3_MAX_TRANSPORT_ATTEMPTS = V2_MAX_TRANSPORT_ATTEMPTS
V3_BACKOFF_SECONDS = V2_BACKOFF_SECONDS
V3_RETRYABLE_HTTP_STATUS = V2_RETRYABLE_HTTP_STATUS
V3_STATE_METHODS = V2_STATE_METHODS


def provider_specs_v3() -> tuple[ProviderSpecV2, ...]:
    """Return the same provider/model conditions used by V2."""
    return provider_specs_v2()


def build_validation_plan_v3(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the same logical matrix as V2 with V3 run IDs."""
    rows: list[dict[str, Any]] = []
    for candidate in sorted(pilots, key=lambda item: item["scenario_id"]):
        task_hash = sha256_text(canonical_json(candidate))
        for spec in provider_specs_v3():
            for method in V3_STATE_METHODS:
                identity = {
                    "protocol_id": V3_PROTOCOL_ID,
                    "scenario_id": candidate["scenario_id"],
                    "condition_id": spec.condition_id,
                    "state_method": method,
                    "repetition": 0,
                }
                rows.append(
                    {
                        **identity,
                        "run_id": stable_id("xpv3", identity),
                        "task_sha256": task_hash,
                        "planned_primary_calls": 2,
                        "excluded_from_formal_analysis": True,
                    }
                )
    random.Random(V3_RUN_ORDER_SEED).shuffle(rows)
    for index, row in enumerate(rows):
        row["run_order"] = index
    return rows


def build_provider_request_v3(bundle: PromptBundle, run_id: str, spec: ProviderSpecV2) -> Any:
    """Build the unchanged V2 provider payload for a V3 record."""
    return build_provider_request_v2(bundle, run_id, spec)


def provider_manifest_v3() -> list[dict[str, Any]]:
    return [
        {
            **asdict(spec),
            "api_key_envs": list(spec.api_key_envs),
            "api_key_value_recorded": False,
            "model_condition_unchanged_from_v2": True,
        }
        for spec in provider_specs_v3()
    ]
