#!/usr/bin/env python3
"""Strict JSON and Draft 2020-12 profile for compact-runtime wire schemas."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError, validators
from rfc3339_validator import validate_rfc3339


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite(value)
    return parsed


def loads_strict(text: str) -> Any:
    """Parse one JSON document, rejecting duplicate keys and non-I-JSON numbers."""

    return json.loads(
        text,
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_nonfinite,
        parse_float=_finite_float,
    )


def _exact_number_kind(
    validator: Any,
    expected: str,
    instance: Any,
    schema: Mapping[str, Any],
) -> Iterator[ValidationError]:
    del validator, schema
    exact_integer = type(instance) is int
    finite_number = type(instance) is int or (
        type(instance) is float and math.isfinite(instance)
    )
    if (expected == "integer" and not exact_integer) or (
        expected == "number" and not finite_number
    ):
        yield ValidationError(
            f"{instance!r} is not an exact finite JSON {expected}"
        )


def _sorted_unique(
    validator: Any,
    enabled: bool,
    instance: Any,
    schema: Mapping[str, Any],
) -> Iterator[ValidationError]:
    del validator, schema
    if enabled and isinstance(instance, list):
        if (
            not all(type(item) is str for item in instance)
            or len(instance) != len(set(instance))
            or instance != sorted(instance)
        ):
            yield ValidationError(f"{instance!r} is not sorted and unique")


CompactRuntimeValidator = validators.extend(
    Draft202012Validator,
    {
        "x-ds4-number-kind": _exact_number_kind,
        "x-ds4-sorted-unique": _sorted_unique,
    },
)

_FORMAT_CHECKER = FormatChecker()
_FORMAT_CHECKER.checks("date-time", raises=TypeError)(validate_rfc3339)


def validator_for(schema: Mapping[str, Any]) -> Any:
    """Return the normative DS4 validator for one independent wire schema."""

    CompactRuntimeValidator.check_schema(schema)
    return CompactRuntimeValidator(schema, format_checker=_FORMAT_CHECKER)


__all__ = ["CompactRuntimeValidator", "loads_strict", "validator_for"]
