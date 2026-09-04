#!/usr/bin/env python3
"""Dependency-free strict validator for ``ds4.eval.case/v1`` JSONL.

The evidence digest is deliberately reimplemented here rather than imported
from production.  Its preimage is the exact UTF-8 byte stream beginning with
``b"ds4.eval.case.evidence/v1\n"`` and followed, for each of the eight fields
in required order, by ``FIELD_NAME=DECIMAL_UTF8_BYTE_LENGTH:EXACT_UTF8_VALUE\n``.
Field names and length digits are ASCII; lengths count UTF-8 bytes, and the
final LF is included.  ``evidence_sha256`` and all unrecorded model output are
excluded.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/ds4-eval-case-v1.schema.json"
FIXTURE_PATH = ROOT / "tests/fixtures/ds4-eval-case-v1.jsonl"
SCHEMA_ID = "ds4.eval.case/v1"
CASE_KEYS = (
    "schema",
    "case_id",
    "answer",
    "grade",
    "terminal_status",
    "request_id",
    "instance_id",
    "snapshot_seq",
    "evidence_sha256",
)
EVIDENCE_FIELDS = CASE_KEYS[:-1]
CASE_IDS = (
    "recNu3MXkvWUzHZr9",
    "001b51d76b4d422988f2c11f104a2c6c",
    "aime2025-01",
    "compsec-076",
)
EXPECTED_ANSWERS = ("B", "C", "70", "17-20")
GRADES = ("passed", "failed", "not_graded")
TERMINAL_STATUSES = (
    "completed",
    "cancelled",
    "rejected",
    "recoverable_error",
    "unsafe_error",
)
UUID_RE = re.compile(
    r"^(?!00000000-0000-0000-0000-000000000000$)"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
NONZERO_DECIMAL_RE = re.compile(r"^[1-9][0-9]*$")
SHA256_RE = re.compile(r"^(?!0{64}$)[0-9a-f]{64}$")
UINT64_MAX = (1 << 64) - 1
COMPSEC_TOKEN_RE = re.compile(r"[0-9]+(?:-[0-9]*)?")


class StrictJSONError(ValueError):
    """Raised when JSON syntax or object-key uniqueness is not strict."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON value {value}")


def loads_strict(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValueError) as exc:
        raise StrictJSONError(f"invalid strict JSON: {exc}") from exc
    if _contains_nonfinite(value):
        raise StrictJSONError("non-finite JSON number")
    return value


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _compsec_line_set(spec: str) -> set[int]:
    """Parse the COMPSEC line-set syntax independently of production."""
    lines: set[int] = set()
    for token in COMPSEC_TOKEN_RE.findall(spec):
        bounds = token.split("-")
        lower = int(bounds[0])
        upper = lower if len(bounds) == 1 else int(bounds[1] or 0)
        lower, upper = sorted((lower, upper))
        lower = max(lower, 0)
        upper = min(upper, 255)
        lines.update(range(lower, upper + 1))
    return lines


def _compsec_answer_matches(expected_spec: str, got_spec: str) -> bool:
    expected = _compsec_line_set(expected_spec)
    got = _compsec_line_set(got_spec)
    return bool(got) and got <= expected


def _load_schema() -> dict[str, Any]:
    value = loads_strict(SCHEMA_PATH.read_bytes())
    if not isinstance(value, dict):
        raise StrictJSONError("case schema is not an object")
    return value


def _validate_schema_shape(schema: dict[str, Any]) -> None:
    if schema.get("$id") != SCHEMA_ID:
        raise StrictJSONError("case schema has the wrong $id")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise StrictJSONError("case schema is not a closed object")
    if tuple(schema.get("required", ())) != CASE_KEYS:
        raise StrictJSONError("case schema required order is not closed")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or tuple(properties) != CASE_KEYS:
        raise StrictJSONError("case schema property order is not closed")


def _require_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if type(value) is not str:
        raise StrictJSONError(f"$.{key}: expected a JSON string")
    return value


def _canonical_case_preimage(record: dict[str, Any]) -> bytes:
    preimage = bytearray(b"ds4.eval.case.evidence/v1\n")
    for key in EVIDENCE_FIELDS:
        value = _require_string(record, key)
        field = value.encode("utf-8")
        preimage.extend(key.encode("ascii"))
        preimage.extend(b"=")
        preimage.extend(str(len(field)).encode("ascii"))
        preimage.extend(b":")
        preimage.extend(field)
        preimage.extend(b"\n")
    return bytes(preimage)


def evidence_sha256(record: dict[str, Any]) -> str:
    """Return the independently specified digest for one record."""
    return hashlib.sha256(_canonical_case_preimage(record)).hexdigest()


def validate_record(record: Any, *, expected_case_index: int | None = None) -> None:
    if type(record) is not dict:
        raise StrictJSONError("record root must be a JSON object")
    if tuple(record) != CASE_KEYS:
        raise StrictJSONError("record keys must be exactly the nine fields in canonical order")
    if record["schema"] != SCHEMA_ID:
        raise StrictJSONError("$.schema: wrong schema literal")

    case_id = _require_string(record, "case_id")
    if case_id not in CASE_IDS:
        raise StrictJSONError("$.case_id: unknown case ID")
    if expected_case_index is not None and case_id != CASE_IDS[expected_case_index]:
        raise StrictJSONError("$.case_id: selection is not the required manifest order")

    answer = _require_string(record, "answer")
    if not answer:
        raise StrictJSONError("$.answer: blank answer")
    grade = _require_string(record, "grade")
    if grade not in GRADES:
        raise StrictJSONError("$.grade: invalid enum")
    terminal = _require_string(record, "terminal_status")
    if terminal not in TERMINAL_STATUSES:
        raise StrictJSONError("$.terminal_status: invalid enum")
    expected_answer = EXPECTED_ANSWERS[CASE_IDS.index(case_id)]
    if case_id == "compsec-076":
        answer_matches = _compsec_answer_matches(expected_answer, answer)
    else:
        answer_matches = answer == expected_answer
    expected_grade = (
        ("passed" if answer_matches else "failed")
        if terminal == "completed" else "not_graded"
    )
    if grade != expected_grade:
        raise StrictJSONError(
            "$.grade: must be " + expected_grade + " for this answer/terminal_status"
        )

    for key in ("request_id", "instance_id"):
        value = _require_string(record, key)
        if UUID_RE.fullmatch(value) is None:
            raise StrictJSONError(f"$.{key}: non-canonical UUID")

    snapshot = _require_string(record, "snapshot_seq")
    if NONZERO_DECIMAL_RE.fullmatch(snapshot) is None:
        raise StrictJSONError("$.snapshot_seq: non-canonical nonzero decimal")
    if int(snapshot) > UINT64_MAX:
        raise StrictJSONError("$.snapshot_seq: exceeds uint64")

    digest = _require_string(record, "evidence_sha256")
    if SHA256_RE.fullmatch(digest) is None:
        raise StrictJSONError("$.evidence_sha256: non-canonical SHA-256")
    expected = evidence_sha256(record)
    if digest != expected:
        raise StrictJSONError("$.evidence_sha256: digest does not match recorded fields")


def parse_jsonl(raw: bytes) -> list[Any]:
    if not raw:
        raise StrictJSONError("no JSONL input")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise StrictJSONError("no JSONL records")
    records: list[Any] = []
    for number, line in enumerate(lines, start=1):
        if not line.endswith(b"\n"):
            raise StrictJSONError(f"line {number} is missing LF termination")
        body = line[:-1]
        if body.endswith(b"\r"):
            raise StrictJSONError(f"line {number} uses CRLF instead of LF")
        if not body.strip():
            raise StrictJSONError(f"line {number} is blank")
        records.append(loads_strict(body))
    return records


def validate_jsonl(raw: bytes, *, fixture: bool = False) -> list[Any]:
    _validate_schema_shape(_load_schema())
    records = parse_jsonl(raw)
    if len(records) != len(CASE_IDS):
        raise StrictJSONError(f"JSONL must contain exactly {len(CASE_IDS)} records")
    seen: set[str] = set()
    for index, record in enumerate(records):
        validate_record(record, expected_case_index=index)
        case_id = record["case_id"]
        if case_id in seen:
            raise StrictJSONError("duplicate case selection")
        seen.add(case_id)
    if tuple(record["case_id"] for record in records) != CASE_IDS:
        raise StrictJSONError("case IDs are incomplete or reordered")
    if fixture:
        expected = parse_jsonl(FIXTURE_PATH.read_bytes())
        if records != expected:
            raise StrictJSONError("input does not equal the checked-in canonical fixture")
    return records


def main() -> int:
    try:
        validate_jsonl(sys.stdin.buffer.read(), fixture="--fixture" in sys.argv[1:])
    except (OSError, StrictJSONError, ValueError) as exc:
        print(f"eval-case-json: {exc}", file=sys.stderr)
        return 1
    print(f"eval-case-json: {len(CASE_IDS)} strict record(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
