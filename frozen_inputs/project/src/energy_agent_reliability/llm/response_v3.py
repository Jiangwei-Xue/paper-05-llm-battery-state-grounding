"""Strict V3 parsing with exact Decimal JSON Schema numeric validation.

V3 intentionally keeps the V2 model-visible schema surface unchanged. The
measurement repair is local to the parser: JSON numbers are parsed as Decimal,
``multipleOf`` is evaluated by exact Decimal arithmetic, and only schema-valid
values are converted to simulator floats.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from jsonschema import Draft202012Validator, ValidationError, validators

from .models import ParserResult, StageName, StateMethodName
from .response_v2 import (
    _reject_nonfinite,
    _schema_diagnostic,
    _state_value,
    _validate_state_surface,
    materialize_response_schema_v2,
)

JSONNumber = int | Decimal


def parse_stage_response_v3(
    raw_text: str,
    stage: StageName,
    state_method: StateMethodName,
    required_dispatch_length: int,
    *,
    expected_dispatch_rows: list[dict[str, Any]],
    finish_reason: str | None = None,
) -> ParserResult:
    """Parse a stage response under the V3 measurement contract."""
    if finish_reason == "length":
        return ParserResult(False, None, [], ["finish_reason_length"])
    try:
        payload = json.loads(raw_text, parse_float=Decimal, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as exc:
        message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        return ParserResult(False, None, [], [f"malformed_json:{message}"])
    if not isinstance(payload, dict):
        return ParserResult(False, None, [], ["top_level_not_object"])
    schema = materialize_response_schema_v2(stage, state_method, required_dispatch_length)
    errors = sorted(_DECIMAL_VALIDATOR(schema).iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        return ParserResult(False, _to_simulator_value(_state_value(payload, state_method)), [], [_schema_diagnostic(errors[0])])
    state = _state_value(payload, state_method)
    state_for_simulator = _to_simulator_value(state)
    state_diagnostic = _validate_state_surface(
        cast(dict[str, Any] | str | None, state_for_simulator),
        state_method,
    )
    if state_diagnostic is not None:
        return ParserResult(False, cast(dict[str, Any] | str | None, state_for_simulator), [], [state_diagnostic])
    dispatch_key = "dispatch_plan" if stage == "stage1" else "revised_dispatch_plan"
    dispatch = payload[dispatch_key]
    if len(expected_dispatch_rows) != required_dispatch_length:
        raise ValueError("Expected dispatch rows must match the exact V3 dispatch length.")
    actions: list[float] = []
    for index, (item, expected) in enumerate(zip(dispatch, expected_dispatch_rows, strict=True)):
        if item["t"] != expected.get("t"):
            return ParserResult(False, cast(dict[str, Any] | str | None, state_for_simulator), [], [f"dispatch_item_{index}_wrong_t"])
        if item["timestamp_utc"] != expected.get("timestamp_utc"):
            return ParserResult(False, cast(dict[str, Any] | str | None, state_for_simulator), [], [f"dispatch_item_{index}_wrong_timestamp"])
        if not at_most_three_decimals_v3(item["battery_action_kw"]):
            return ParserResult(False, cast(dict[str, Any] | str | None, state_for_simulator), [], [f"dispatch_item_{index}_excess_precision"])
        actions.append(float(item["battery_action_kw"]))
    if isinstance(state, dict):
        for key in (
            "soc_kwh",
            "usable_capacity_kwh",
            "max_charge_kw",
            "max_discharge_kw",
            "reserve_soc_kwh",
            "export_limit_kw",
        ):
            if not at_most_three_decimals_v3(state[key]):
                return ParserResult(False, cast(dict[str, Any] | str | None, state_for_simulator), [], [f"typed_state_{key}_excess_precision"])
    return ParserResult(True, cast(dict[str, Any] | str | None, state_for_simulator), actions, [])


def canonical_parsed_representation_v3(raw_text: str) -> dict[str, Any] | list[Any] | None:
    """Return a stable Decimal-preserving JSON representation for evidence hashes."""
    try:
        payload = json.loads(raw_text, parse_float=Decimal, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError):
        return None
    converted = _to_canonical_decimal_value(payload)
    return cast(dict[str, Any] | list[Any] | None, converted)


def at_most_three_decimals_v3(value: Any) -> bool:
    """True only for JSON numbers that are exactly representable at 0.001 resolution."""
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        return False
    decimal_value = Decimal(value) if isinstance(value, int) else value
    try:
        return decimal_value == decimal_value.quantize(Decimal("0.001"))
    except InvalidOperation:
        return False


def decimal_validator_report_cases() -> list[dict[str, Any]]:
    """Small built-in regression table used by tests and V3 preflight."""
    schema = {"type": "number", "multipleOf": 0.001}
    cases: list[tuple[str, str, bool]] = [
        ("0.001", "number", True),
        ("0.002", "number", True),
        ("1.234", "number", True),
        ("-1.234", "number", True),
        ("1.2345", "number", False),
        ("0.0005", "number", False),
        ('"1.234"', "string", False),
        ("NaN", "nonfinite", False),
        ("Infinity", "nonfinite", False),
    ]
    rows: list[dict[str, Any]] = []
    validator = _DECIMAL_VALIDATOR(schema)
    for raw, kind, expected in cases:
        try:
            payload = json.loads(raw, parse_float=Decimal, parse_constant=_reject_nonfinite)
            diagnostics = [_schema_diagnostic(error) for error in validator.iter_errors(payload)]
            passed = not diagnostics
        except (json.JSONDecodeError, ValueError) as exc:
            diagnostics = [str(exc)]
            passed = False
        rows.append(
            {
                "raw": raw,
                "case_kind": kind,
                "expected_pass": expected,
                "v3_passed": passed,
                "diagnostics": diagnostics,
                "case_passed": passed == expected,
            }
        )
    return rows


def _is_number(_checker: Any, instance: Any) -> bool:
    return isinstance(instance, int | Decimal) and not isinstance(instance, bool)


def _is_integer(_checker: Any, instance: Any) -> bool:
    return isinstance(instance, int) and not isinstance(instance, bool)


def _decimal_multiple_of(validator: Any, divisor: Any, instance: Any, schema: Any) -> Any:
    del validator, schema
    if not _is_number(None, instance):
        return
    try:
        instance_decimal = Decimal(instance) if isinstance(instance, int) else instance
        divisor_decimal = Decimal(str(divisor))
    except (InvalidOperation, ValueError, TypeError) as exc:
        yield ValidationError(f"{instance!r} cannot be checked against multipleOf {divisor!r}: {exc}")
        return
    if divisor_decimal == 0:
        yield ValidationError("multipleOf divisor must be non-zero")
        return
    if instance_decimal % divisor_decimal != Decimal("0"):
        yield ValidationError(f"{instance_decimal} is not a multiple of {divisor_decimal}")


_TYPE_CHECKER = Draft202012Validator.TYPE_CHECKER.redefine_many(
    {
        "number": _is_number,
        "integer": _is_integer,
    }
)

_DECIMAL_VALIDATOR = cast(Any, validators.extend)(
    Draft202012Validator,
    type_checker=_TYPE_CHECKER,
    validators={"multipleOf": _decimal_multiple_of},
)


def _to_simulator_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_to_simulator_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_simulator_value(item) for key, item in value.items()}
    return value


def _to_canonical_decimal_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, list):
        return [_to_canonical_decimal_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_canonical_decimal_value(value[key]) for key in sorted(value)}
    return value
