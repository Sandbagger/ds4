#!/usr/bin/env python3
"""Strict qualification-record validation and bounded incremental JSONL parsing."""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas/ds4-bench-qualification-v1.schema.json"
RUNTIME_SCHEMA_PATH = ROOT / "schemas/ds4-runtime-v1.schema.json"
REQUEST_SCHEMA_PATH = ROOT / "schemas/ds4-runtime-request-v1.schema.json"
UINT64_RE = re.compile(r"^(?:0|[1-9][0-9]{0,18}|1[0-7][0-9]{18}|18[0-3][0-9]{17}|184[0-3][0-9]{16}|1844[0-5][0-9]{15}|18446[0-6][0-9]{14}|184467[0-3][0-9]{13}|1844674[0-3][0-9]{12}|184467440[0-6][0-9]{10}|1844674407[0-2][0-9]{9}|18446744073[0-6][0-9]{8}|1844674407370[0-8][0-9]{6}|18446744073709[0-4][0-9]{5}|184467440737095[0-4][0-9]{4}|18446744073709550[0-9]{3}|18446744073709551[0-5][0-9]{2}|1844674407370955160[0-9]|1844674407370955161[0-4]|18446744073709551615)(?![\s\S])")
UUID_RE = re.compile(r"^(?!00000000-0000-0000-0000-000000000000(?![\s\S]))[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![\s\S])")
PROFILE_CACHE_BYTES = {
    "cache-8gib": str(8 << 30),
    "cache-12gib": str(12 << 30),
    "cache-16gib": str(16 << 30),
}
PROMPT_TOKENS = {
    "native-512": 512,
    "native-2048": 2048,
    "native-8192": 8192,
    "native-28672": 28672,
}
PROFILE_PROMPT_ORDER = {
    "cache-8gib": ("native-512", "native-2048", "native-28672", "native-8192"),
    "cache-12gib": ("native-2048", "native-8192", "native-512", "native-28672"),
    "cache-16gib": ("native-8192", "native-28672", "native-2048", "native-512"),
}

# The wire limits include the terminating LF for each record and every byte
# accepted by a stream.  They are intentionally small enough to bound parser
# memory before JSON decoding begins.
MAX_RECORD_BYTES = 1_048_576
MAX_STREAM_BYTES = 12_582_912
MAX_RECORD_COUNT = 12
MAX_JSON_NESTING = 64
JSON_READ_CHUNK_BYTES = 64 * 1024
EXPECTED_BINDINGS = (
    "manifest_sha256",
    "sequence_sha256",
    "profile_id",
    "prompt_order_index",
    "prompt_id",
    "input_sha256",
)
LIFECYCLE_EVENTS = ("request_accepted", "first_token", "request_complete")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _check_json_nesting(raw: bytes) -> None:
    """Reject deeply nested JSON before ``json.loads`` recurses."""

    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # double quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):  # object/array open
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise ValueError(
                    f"JSON nesting exceeds {MAX_JSON_NESTING} levels"
                )
        elif byte in (0x7D, 0x5D):  # object/array close
            depth -= 1
            if depth < 0:
                raise ValueError("invalid JSON nesting")


def loads_strict(raw: bytes) -> Any:
    if not isinstance(raw, bytes):
        raise TypeError("strict JSON input must be bytes")
    _check_json_nesting(raw)
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
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
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite number")
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
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"{path}: items must be unique")
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
        "manifest_sha256", "sequence_sha256", "input_sha256",
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


def _require_equal(record: dict[str, Any], key: str, expected: Any, source: str) -> None:
    if record.get(key) != expected:
        raise ValueError(f"$.{key}: does not match {source}")


def _validate_flattened_bindings(record: dict[str, Any]) -> None:
    runtime = record["runtime"]
    allocations = runtime["allocations"]
    categories = allocations["categories"]
    reports = allocations["reports"]
    qualification = allocations["qualification_total"]

    _require_equal(record, "instance_id", runtime["instance_id"], "runtime.instance_id")
    _require_equal(record, "snapshot_seq", runtime["snapshot_seq"], "runtime.snapshot_seq")
    _require_equal(
        record, "configured_prefill_rows", allocations["configured_prefill_rows"],
        "runtime.allocations.configured_prefill_rows",
    )
    _require_equal(
        record, "allocated_prefill_rows", allocations["allocated_prefill_rows"],
        "runtime.allocations.allocated_prefill_rows",
    )
    _require_equal(
        record, "expert_cache_limit_bytes",
        runtime["limits"]["expert_cache_limit_bytes"],
        "runtime.limits.expert_cache_limit_bytes",
    )
    _require_equal(
        record, "kv_allocated_bytes", categories["kv_state"]["current_bytes"],
        "runtime.allocations.categories.kv_state.current_bytes",
    )
    expert = categories["expert_cache_payload"]
    for key, source_key in (
        ("expert_cache_current_bytes", "current_bytes"),
        ("expert_cache_peak_bytes", "peak_bytes"),
    ):
        _require_equal(record, key, expert[source_key], f"runtime.allocations.categories.expert_cache_payload.{source_key}")
    for key, source_key in (
        ("qualification_total_current_bytes", "current_bytes"),
        ("qualification_total_bound_bytes", "bound_bytes"),
        ("qualification_total_peak_bytes", "peak_bytes"),
    ):
        _require_equal(record, key, qualification[source_key], f"runtime.allocations.qualification_total.{source_key}")
    _require_equal(
        record, "model_inode_resident_bytes",
        reports["model_source_resident"]["current_bytes"],
        "runtime.allocations.reports.model_source_resident.current_bytes",
    )
    for key in (
        "model_source_resident", "host_library_unattributed",
        "cuda_library_unattributed",
    ):
        _require_equal(
            record["external_attribution"], key,
            reports[key]["current_bytes"],
            f"runtime.allocations.reports.{key}.current_bytes",
        )

    expected_cache = PROFILE_CACHE_BYTES.get(record["profile_id"])
    expected_order = PROFILE_PROMPT_ORDER.get(record["profile_id"])
    if expected_cache is None or expected_order is None:
        raise ValueError("$.profile_id: unknown cache profile")
    if record["prompt_id"] != expected_order[record["prompt_order_index"]]:
        raise ValueError("$.prompt_order_index: does not match profile prompt order")
    for source, label in (
        (record["expert_cache_limit_bytes"], "record expert cache limit"),
        (runtime["limits"]["expert_cache_limit_bytes"], "runtime limits expert cache limit"),
        (runtime["config"]["ssd_streaming_cache_bytes"], "runtime config cache bytes"),
        (categories["expert_cache_payload"]["bound_bytes"], "expert cache bound"),
    ):
        if source != expected_cache:
            raise ValueError(f"$.profile_id: {label} is not the pinned profile ceiling")

    config = runtime["config"]
    limits = runtime["limits"]
    expected_config = {
        "context_tokens": 32768,
        "prefill_chunk_tokens": 4096,
        "session_slots": 1,
        "ssd_streaming": True,
        "ssd_streaming_cache_bytes": expected_cache,
    }
    expected_limits = {
        "effective_context_tokens": 32768,
        "effective_prefill_chunk_tokens": 4096,
        "effective_session_slots": 1,
        "expert_cache_limit_bytes": expected_cache,
    }
    if config != expected_config:
        raise ValueError("$.runtime.config: qualification settings are not pinned")
    if limits != expected_limits:
        raise ValueError("$.runtime.limits: qualification settings are not pinned")

    if record["event"] == "request_complete":
        request = record["request_metrics"]
        _require_equal(record, "request_id", request["request_id"], "request_metrics.request_id")
        _require_equal(record, "instance_id", request["instance_id"], "request_metrics.instance_id")
        if int(request["snapshot_seq"]) + 1 != int(record["snapshot_seq"]):
            raise ValueError("$.request_metrics.snapshot_seq: must immediately precede runtime snapshot_seq")
        _require_equal(record, "terminal_status", request["terminal_status"], "request_metrics.terminal_status")
        expected_tokens = PROMPT_TOKENS.get(record["prompt_id"])
        if expected_tokens is None or request["prompt_tokens"] != expected_tokens:
            raise ValueError("$.prompt_id: does not match request_metrics.prompt_tokens")


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
    if record["external_attribution"]["unrelated_process_inventory_stable"] is not True:
        raise ValueError("$.external_attribution.unrelated_process_inventory_stable must be true")
    _validate_flattened_bindings(record)
    if complete is True and record["event"] != "request_complete":
        raise ValueError("canned fixture must be a request_complete record")


def _validate_lifecycle(records: list[Any]) -> None:
    expected_events = ("request_accepted", "first_token", "request_complete")
    if len(records) != 12:
        raise ValueError("lifecycle JSONL must contain exactly 12 records")
    previous_monotonic = -1
    previous_snapshot = -1
    request_ids: set[str] = set()
    for index, record in enumerate(records):
        expected_repetition = index // len(expected_events)
        expected_event = expected_events[index % len(expected_events)]
        if record.get("repetition_index") != expected_repetition:
            raise ValueError(f"record {index + 1}: repetition_index is not {expected_repetition}")
        if record.get("event") != expected_event:
            raise ValueError(f"record {index + 1}: event is not {expected_event}")
        monotonic = int(record["monotonic_ns"])
        snapshot = int(record["runtime"]["snapshot_seq"])
        if monotonic <= previous_monotonic:
            raise ValueError(f"record {index + 1}: monotonic_ns is not strictly increasing")
        if snapshot <= previous_snapshot:
            raise ValueError(f"record {index + 1}: runtime snapshot_seq is not strictly increasing")
        previous_monotonic = monotonic
        previous_snapshot = snapshot
        repetition_start = expected_repetition * len(expected_events)
        if record["request_id"] != records[repetition_start]["request_id"]:
            raise ValueError(f"record {index + 1}: request_id changed within repetition")
        if index % len(expected_events) == 0:
            request_ids.add(record["request_id"])
        if index % len(expected_events) == 2:
            accepted_seq = int(records[repetition_start]["runtime"]["snapshot_seq"])
            first_seq = int(records[repetition_start + 1]["runtime"]["snapshot_seq"])
            complete_seq = snapshot
            metrics_seq = int(record["request_metrics"]["snapshot_seq"])
            if first_seq != accepted_seq + 1:
                raise ValueError(f"record {index + 1}: accepted/first runtime snapshots are not consecutive")
            if metrics_seq + 1 != complete_seq or metrics_seq != first_seq + 1:
                raise ValueError(f"record {index + 1}: request-finish sequence gap is invalid")
    if len(request_ids) != 4:
        raise ValueError("request_id must be distinct across repetitions")


def _copy_expected_bindings(expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a mapping")
    try:
        copied = dict(expected)
    except (TypeError, ValueError) as exc:
        raise TypeError("expected must be a mapping") from exc
    if set(copied) != set(EXPECTED_BINDINGS):
        raise ValueError("expected must contain exactly the six qualification bindings")
    for key in EXPECTED_BINDINGS:
        value = copied[key]
        if key == "prompt_order_index":
            if type(value) is not int:
                raise ValueError("expected.prompt_order_index must be a JSON integer")
        elif type(value) is not str:
            raise ValueError(f"expected.{key} must be a string")
    return copied


class QualificationRecordStream:
    """Incrementally validate one twelve-record qualification JSONL stream.

    This parser validates wire/schema/lifecycle structure only.  It does not
    launch a process, measure deadlines, compare numerical output, or declare
    qualification status.
    """

    def __init__(self, expected: Mapping[str, Any]) -> None:
        self._expected = _copy_expected_bindings(expected)
        self._line = bytearray()
        self._total_bytes = 0
        self._records: list[dict[str, Any]] = []
        self._drain_index = 0
        self._request_ids: set[str] = set()
        self._instance_id: str | None = None
        self._request_id: str | None = None
        self._accepted_snapshot: int | None = None
        self._first_snapshot: int | None = None
        self._previous_monotonic = -1
        self._previous_snapshot = -1
        self._failed = False
        self._finished = False

    def _fail(self, message: str) -> None:
        self._failed = True
        raise ValueError(message)

    def _ensure_open(self) -> None:
        if self._failed:
            raise ValueError("qualification record stream is failed")
        if self._finished:
            raise ValueError("qualification record stream is finished")

    def feed(self, chunk: bytes) -> None:
        self._ensure_open()
        if not isinstance(chunk, bytes):
            self._fail("qualification record chunks must be bytes")
        # Check the total before retaining any part of an unchecked chunk.
        if self._total_bytes + len(chunk) > MAX_STREAM_BYTES:
            self._fail("qualification JSONL stream exceeds byte limit")
        self._total_bytes += len(chunk)
        try:
            for byte in chunk:
                if len(self._line) + 1 > MAX_RECORD_BYTES:
                    self._fail("qualification JSONL record exceeds byte limit")
                self._line.append(byte)
                if byte == 0x0A:  # LF
                    line = bytes(self._line)
                    self._line.clear()
                    self._consume_line(line)
        except RecursionError as exc:
            self._fail(f"qualification record is too deeply nested: {exc}")

    def _consume_line(self, line: bytes) -> None:
        if len(self._records) >= MAX_RECORD_COUNT:
            self._fail("qualification JSONL stream contains too many records")
        if len(line) > MAX_RECORD_BYTES:
            self._fail("qualification JSONL record exceeds byte limit")
        if len(line) >= 2 and line[-2] == 0x0D:
            self._fail("qualification JSONL uses CRLF")
        try:
            record = loads_strict(line[:-1])
            validate_record(record)
        except (TypeError, ValueError, RecursionError) as exc:
            self._fail(str(exc))
        if type(record) is not dict:
            self._fail("qualification record root is not an object")
        self._accept_record(record)

    def _accept_record(self, record: dict[str, Any]) -> None:
        index = len(self._records)
        repetition = index // len(LIFECYCLE_EVENTS)
        event_index = index % len(LIFECYCLE_EVENTS)
        expected_event = LIFECYCLE_EVENTS[event_index]
        if record.get("repetition_index") != repetition:
            self._fail(
                f"record {index + 1}: repetition_index is not {repetition}"
            )
        if record.get("event") != expected_event:
            self._fail(f"record {index + 1}: event is not {expected_event}")
        for key in EXPECTED_BINDINGS:
            if record.get(key) != self._expected[key]:
                self._fail(f"$.{key}: does not match expected binding")

        monotonic = int(record["monotonic_ns"])
        snapshot = int(record["snapshot_seq"])
        if monotonic <= self._previous_monotonic:
            self._fail(f"record {index + 1}: monotonic_ns is not strictly increasing")
        if snapshot <= self._previous_snapshot:
            self._fail(f"record {index + 1}: snapshot_seq is not strictly increasing")
        self._previous_monotonic = monotonic
        self._previous_snapshot = snapshot

        instance_id = record["instance_id"]
        if self._instance_id is None:
            self._instance_id = instance_id
        elif instance_id != self._instance_id:
            self._fail(f"record {index + 1}: instance_id changed")

        request_id = record["request_id"]
        if event_index == 0:
            if request_id in self._request_ids:
                self._fail(f"record {index + 1}: request_id was reused")
            self._request_ids.add(request_id)
            self._request_id = request_id
            self._accepted_snapshot = snapshot
            self._first_snapshot = None
        elif request_id != self._request_id:
            self._fail(f"record {index + 1}: request_id changed within repetition")

        if event_index == 1:
            if self._accepted_snapshot is None or snapshot != self._accepted_snapshot + 1:
                self._fail(
                    f"record {index + 1}: accepted/first runtime snapshots are not consecutive"
                )
            self._first_snapshot = snapshot
        elif event_index == 2:
            metrics_snapshot = int(record["request_metrics"]["snapshot_seq"])
            if self._first_snapshot is None or metrics_snapshot != self._first_snapshot + 1:
                self._fail(
                    f"record {index + 1}: request-finish sequence gap is invalid"
                )
            if metrics_snapshot + 1 != snapshot:
                self._fail(
                    f"record {index + 1}: request-finish sequence gap is invalid"
                )
        self._records.append(record)

    def drain_records(self) -> tuple[dict[str, Any], ...]:
        """Copy new validated records, even after failure or finish.

        A returned prefix is evidence, not proof of a complete valid stream.
        Draining never changes the sticky feed/finish lifecycle.
        """
        end = len(self._records)
        drained = tuple(
            copy.deepcopy(record) for record in self._records[self._drain_index:end]
        )
        self._drain_index = end
        return drained

    def finish(self) -> tuple[dict[str, Any], ...]:
        if self._finished:
            raise ValueError("qualification record stream is finished")
        self._finished = True
        if self._failed:
            raise ValueError("qualification record stream is failed")
        if self._line:
            self._fail("qualification JSONL stream is not LF terminated")
        if not self._records:
            self._fail("qualification JSONL stream has no records")
        if len(self._records) != MAX_RECORD_COUNT:
            self._fail("qualification JSONL lifecycle is incomplete")
        return tuple(copy.deepcopy(record) for record in self._records)


def _iter_bounded_json_lines(handle: Any) -> Iterator[bytes]:
    """Yield LF-terminated record bodies from a bounded binary stream."""

    total = 0
    line = bytearray()
    while True:
        chunk = handle.read(JSON_READ_CHUNK_BYTES)
        if not chunk:
            break
        if not isinstance(chunk, bytes) or len(chunk) > JSON_READ_CHUNK_BYTES:
            raise ValueError("qualification input reader returned an invalid chunk")
        if total + len(chunk) > MAX_STREAM_BYTES:
            raise ValueError("qualification JSONL stream exceeds byte limit")
        total += len(chunk)
        for byte in chunk:
            if len(line) + 1 > MAX_RECORD_BYTES:
                raise ValueError("qualification JSONL record exceeds byte limit")
            line.append(byte)
            if byte == 0x0A:
                if len(line) >= 2 and line[-2] == 0x0D:
                    raise ValueError("qualification JSONL uses CRLF")
                yield bytes(line[:-1])
                line.clear()
    if line:
        raise ValueError("qualification JSONL input is not LF terminated")


def _read_bounded_records(handle: Any) -> list[Any]:
    records: list[Any] = []
    for line_number, body in enumerate(_iter_bounded_json_lines(handle), 1):
        if len(records) >= MAX_RECORD_COUNT:
            raise ValueError("qualification JSONL stream contains too many records")
        try:
            records.append(loads_strict(body))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    return records

def main() -> int:
    records: list[Any] = []
    try:
        records = _read_bounded_records(sys.stdin.buffer)
        if not records:
            raise ValueError("no JSONL records")
        for record in records:
            validate_record(record)
        if "--fixture" not in sys.argv:
            _validate_lifecycle(records)
    except (TypeError, ValueError, RecursionError) as exc:
        print(f"qualification-json: {exc}", file=sys.stderr)
        return 1
    print(f"qualification-json: {len(records)} strict record(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
