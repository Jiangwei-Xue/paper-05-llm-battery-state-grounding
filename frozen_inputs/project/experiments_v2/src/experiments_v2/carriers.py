"""Carrier construction and mutation rules for C, M, S, K and C2."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import sha256_json

PHYSICAL_FIELDS = {
    "soc_kwh",
    "usable_capacity_kwh",
    "charge_limit_kw",
    "discharge_limit_kw",
    "export_limit_kw",
    "reserve_kwh",
    "forecast_version",
    "forecast_series",
}
METADATA_FIELDS = {"active_commitments", "revoked_commitments", "sequence"}


def carrier_payload(carrier: dict[str, Any], visible_update: dict[str, Any]) -> dict[str, Any]:
    return {"carrier": deepcopy(carrier), "visible_update": deepcopy(visible_update)}


def carrier_hash(carrier: dict[str, Any], visible_update: dict[str, Any]) -> str:
    return sha256_json(carrier_payload(carrier, visible_update))


def mutate_carrier(
    canonical: dict[str, Any],
    mutation: str,
    *,
    stale_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = deepcopy(canonical)
    stale_values = stale_values or {}
    if mutation == "C" or mutation == "C2":
        return result
    if mutation == "M":
        return deepcopy(stale_values.get("model_carrier", canonical))
    if mutation == "S":
        field = stale_values.get("physical_field")
        if not isinstance(field, str) or field not in PHYSICAL_FIELDS:
            raise ValueError("S requires one registered physical_field")
        if field not in stale_values:
            raise ValueError(f"S stale value missing for {field}")
        result[field] = deepcopy(stale_values[field])
        return result
    if mutation == "K":
        for field in METADATA_FIELDS:
            if field in stale_values:
                result[field] = deepcopy(stale_values[field])
        return result
    raise ValueError(f"unknown carrier mutation {mutation}")


def assert_mutation_scope(
    canonical: dict[str, Any], mutated: dict[str, Any], mutation: str
) -> None:
    changed = {key for key in canonical if canonical.get(key) != mutated.get(key)}
    if mutation in {"C", "C2"} and changed:
        raise AssertionError(f"canonical mutation changed fields: {sorted(changed)}")
    if mutation == "K" and not changed.issubset(METADATA_FIELDS):
        raise AssertionError(f"metadata mutation changed physical fields: {sorted(changed)}")
    if mutation == "S" and not changed.issubset(PHYSICAL_FIELDS):
        raise AssertionError(f"physical mutation changed unexpected fields: {sorted(changed)}")
