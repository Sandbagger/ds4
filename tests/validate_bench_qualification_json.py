#!/usr/bin/env python3
"""Dependency-free strict checker for ds4.bench.qualification/v1 JSONL."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/ds4-bench-qualification-v1.schema.json"
RUNTIME_SCHEMA_PATH = ROOT / "schemas/ds4-runtime-v1.schema.json"
REQUEST_SCHEMA_PATH = ROOT / "schemas/ds4-runtime-request-v1.schema.json"
UINT64_RE = re.compile(r"^(?:0|[1-9][0-9]{0,18}|1[0-7][0-9]{18}|18[0-3][0-9]{17}|184[0-3][0-9]{16}|1844[0-5][0-9]{15}|18446[0-6][0-9]{14}|184467[0-3][0-9]{13}|1844674[0-3][0-9]{12}|184467440[0-6][0-9]{10}|1844674407[0-2][0-9]{9}|18446744073[0-6][0-9]{8}|1844674407370[0-8][0-9]{6}|18446744073709[0-4][0-9]{5}|184467440737095[0-4][0-9]{4}|18446744073709550[0-9]{3}|18446744073709551[0-5][0-9]{2}|1844674407370955160[0-9]|1844674407370955161[0-4]|18446744073709551615)(?![\s\S])")
UUID_RE = re.compile(r"^(?!00000000-0000-0000-0000-000000000000(?![\s\S]))[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![\s\S])")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def loads_strict(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON: {exc}") from exc


def _resolve_ref(
    ref: str, document: Any, document_path: Path
) -> tuple[Any, Any, Path]:
    if ref.startswith("#/"):
        value = document
        for part in ref[2:].split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
        return value, document, document_path
    name, _, fragment = ref.partition("#")
    path = (document_path.parent / name).resolve()
    with path.open("rb") as handle:
        external_document = loads_strict(handle.read())
    value = external_document
    if fragment:
        for part in fragment.lstrip("/").split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
    return value, external_document, path


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in (int, float) and type(value) is not bool,
        "boolean": type(value) is bool,
        "null": value is None,
    }.get(expected, True)


def _matches(value: Any, schema: Any, document: Any, document_path: Path) -> bool:
    try:
        _validate(value, schema, "$", document, document_path)
        return True
    except ValueError:
        return False


def _validate(
    value: Any,
    schema: Any,
    path: str,
    document: Any,
    document_path: Path,
) -> None:
    if schema is True:
        return
    if schema is False:
        raise ValueError(f"{path}: schema rejects value")
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: malformed schema")
    if "$ref" in schema:
        target, target_document, target_path = _resolve_ref(
            schema["$ref"], document, document_path
        )
        _validate(value, target, path, target_document, target_path)
        return
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in enum")
    if "type" in schema:
        expected = schema["type"]
        if isinstance(expected, list):
            if not any(_type_matches(value, item) for item in expected):
                raise ValueError(f"{path}: wrong JSON type")
        elif not _type_matches(value, expected):
            raise ValueError(f"{path}: expected {expected}")
    if "allOf" in schema:
        for child in schema["allOf"]:
            _validate(value, child, path, document, document_path)
    if "anyOf" in schema:
        if not any(_matches(value, child, document, document_path) for child in schema["anyOf"]):
            raise ValueError(f"{path}: no anyOf branch matched")
    if "oneOf" in schema:
        if sum(_matches(value, child, document, document_path) for child in schema["oneOf"]) != 1:
            raise ValueError(f"{path}: expected exactly one oneOf branch")
    if "not" in schema and _matches(value, schema["not"], document, document_path):
        raise ValueError(f"{path}: forbidden value")
    if "if" in schema:
        branch = schema.get("then") if _matches(value, schema["if"], document, document_path) else schema.get("else")
        if branch is not None:
            _validate(value, branch, path, document, document_path)
    if type(value) is str:
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path}: string is too short")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: string does not match pattern")
    if type(value) is int and type(value) is not bool:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: number is above maximum")
    if type(value) is float:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: number is above maximum")
    if type(value) is dict:
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}: missing key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f"{path}: unexpected key {extras[0]!r}")
        for key, child in properties.items():
            if key in value:
                _validate(value[key], child, f"{path}.{key}", document, document_path)
    if type(value) is list:
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path}: too many items")
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(prefix):
            if index < len(value):
                _validate(value[index], child, f"{path}[{index}]", document, document_path)
        if schema.get("items") is False and len(value) > len(prefix):
            raise ValueError(f"{path}: unexpected array item")
        if isinstance(schema.get("items"), dict):
            for index in range(len(prefix), len(value)):
                _validate(value[index], schema["items"], f"{path}[{index}]", document, document_path)


def _validate_wire_scalar_types(record: dict[str, Any]) -> None:
    for key in (
        "manifest_sha256", "input_sha256",
        "snapshot_seq", "monotonic_ns",
        "session_payload_bytes", "kv_allocated_bytes",
        "expert_cache_limit_bytes", "expert_cache_current_bytes",
        "expert_cache_peak_bytes", "qualification_total_current_bytes",
        "qualification_total_bound_bytes", "qualification_total_peak_bytes",
        "model_inode_resident_bytes",
    ):
        value = record[key]
        if not isinstance(value, str):
            raise ValueError(f"$.{key}: uint64-like values must be JSON strings")
    for key in ("configured_prefill_rows", "allocated_prefill_rows", "repetition_index", "prompt_order_index"):
        if type(record[key]) is not int:
            raise ValueError(f"$.{key}: row/index values must be JSON integers")
    for key in ("snapshot_seq", "monotonic_ns", "session_payload_bytes", "kv_allocated_bytes", "expert_cache_limit_bytes", "expert_cache_current_bytes", "expert_cache_peak_bytes", "qualification_total_current_bytes", "qualification_total_bound_bytes", "qualification_total_peak_bytes", "model_inode_resident_bytes"):
        if UINT64_RE.fullmatch(record[key]) is None:
            raise ValueError(f"$.{key}: non-canonical uint64 string")
    for key in ("request_id", "instance_id"):
        if UUID_RE.fullmatch(record[key]) is None:
            raise ValueError(f"$.{key}: invalid identity")


def validate_record(record: Any, *, complete: bool | None = None) -> None:
    schema = loads_strict(SCHEMA_PATH.read_bytes())
    _validate(record, schema, "$", schema, SCHEMA_PATH)
    if not isinstance(record, dict):
        raise ValueError("record root is not an object")
    _validate_wire_scalar_types(record)
    expected_external = {
        "model_source_resident", "host_library_unattributed",
        "cuda_library_unattributed", "unrelated_process_inventory_stable",
    }
    if set(record["external_attribution"]) != expected_external:
        raise ValueError("external_attribution is not closed")
    if complete is True and record["event"] != "request_complete":
        raise ValueError("canned fixture must be a request_complete record")


def main() -> int:
    raw = sys.stdin.buffer.read()
    if not raw:
        print("qualification-json: no JSONL input", file=sys.stderr)
        return 1
    records: list[Any] = []
    try:
        for index, line in enumerate(raw.splitlines(keepends=True)):
            if not line.endswith(b"\n"):
                raise ValueError(f"line {index + 1} is not LF terminated")
            if line[:-1].endswith(b"\r"):
                raise ValueError(f"line {index + 1} uses CRLF")
            records.append(loads_strict(line[:-1]))
        if not records:
            raise ValueError("no JSONL records")
        for index, record in enumerate(records):
            validate_record(record)
        if "--fixture" not in sys.argv:
            expected_events = ("request_accepted", "first_token", "request_complete")
            if len(records) != 12:
                raise ValueError("lifecycle JSONL must contain exactly 12 records")
            for index, record in enumerate(records):
                expected_repetition = index // len(expected_events)
                expected_event = expected_events[index % len(expected_events)]
                if record.get("repetition_index") != expected_repetition:
                    raise ValueError(
                        f"record {index + 1}: repetition_index is not {expected_repetition}"
                    )
                if record.get("event") != expected_event:
                    raise ValueError(
                        f"record {index + 1}: event is not {expected_event}"
                    )
    except ValueError as exc:
        print(f"qualification-json: {exc}", file=sys.stderr)
        return 1
    print(f"qualification-json: {len(records)} strict record(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
