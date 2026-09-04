#!/usr/bin/env python3
"""Host-only tests for the incremental qualification-record parser.

The records come from the checked-in C-emitter fixture and are independently
assembled into the four-repetition lifecycle.  This file only exercises the
offline parser; it never launches DS4 or contacts a model, GPU, network,
or service.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/ds4-bench-qualification-v1.jsonl"
VALIDATOR_PATH = ROOT / "tests/validate_bench_qualification_json.py"

from qualification_records import (
    MAX_RECORD_BYTES,
    MAX_STREAM_BYTES,
    QualificationRecordStream,
)


_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "task20_existing_qualification_validator", VALIDATOR_PATH
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:
    raise RuntimeError(f"cannot load existing validator: {VALIDATOR_PATH}")
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)

EXPECTED_KEYS = (
    "manifest_sha256",
    "sequence_sha256",
    "profile_id",
    "prompt_order_index",
    "prompt_id",
    "input_sha256",
)
EVENTS = ("request_accepted", "first_token", "request_complete")


def _fixture_record() -> dict[str, Any]:
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise AssertionError("qualification fixture must remain one canned record")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise AssertionError("qualification fixture root must be an object")
    return record


def _lifecycle_records() -> list[dict[str, Any]]:
    """Build an independently valid four-request lifecycle from the fixture."""

    base = _fixture_record()
    records: list[dict[str, Any]] = []
    for repetition in range(4):
        request_id = f"123e4567-e89b-12d3-a456-4266141740{repetition + 1:02d}"
        for event_index, event in enumerate(EVENTS):
            record = copy.deepcopy(base)
            record["event"] = event
            record["repetition_index"] = repetition
            record["request_id"] = request_id
            record["monotonic_ns"] = str(1_000_000 + len(records))
            runtime_seq = 100 + repetition * 4 + (
                3 if event == "request_complete" else event_index
            )
            record["snapshot_seq"] = str(runtime_seq)
            record["runtime"]["snapshot_seq"] = str(runtime_seq)
            # Keep one real non-ASCII scalar in the valid fixture-derived
            # stream so tests feed a split UTF-8 code point, not only ASCII.
            record["runtime"]["model"]["id"] = "laguna-s-2.1-界"
            if event == "request_complete":
                request_metrics = record["request_metrics"]
                request_metrics["request_id"] = request_id
                request_metrics["instance_id"] = record["instance_id"]
                request_metrics["snapshot_seq"] = str(runtime_seq - 1)
            else:
                record.pop("request_metrics")
                record.pop("terminal_status")
            _VALIDATOR.validate_record(record)
            records.append(record)
    return records


def _expected(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: records[0][key] for key in EXPECTED_KEYS}


def _json_line(record: Any, *, ensure_ascii: bool = False) -> bytes:
    return (json.dumps(record, separators=(",", ":"), ensure_ascii=ensure_ascii) + "\n").encode(
        "utf-8"
    )


def _payload(records: list[dict[str, Any]], *, ensure_ascii: bool = False) -> bytes:
    return b"".join(_json_line(record, ensure_ascii=ensure_ascii) for record in records)


def _feed(stream: Any, payload: bytes) -> None:
    stream.feed(payload)


def _assert_rejected(test: unittest.TestCase, expected: dict[str, Any], payload: bytes) -> None:
    """Accept rejection either while feeding or at finish, but never recovery."""

    try:
        stream = QualificationRecordStream(expected)
    except (TypeError, ValueError):
        return
    try:
        _feed(stream, payload)
    except (TypeError, ValueError):
        return
    with test.assertRaises((TypeError, ValueError)):
        stream.finish()


class QualificationRecordStreamHostTest(unittest.TestCase):
    def test_valid_fixture_lifecycle_round_trips_across_split_utf8_chunks(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        payload = _payload(records)
        self.assertIn("界".encode("utf-8"), payload)

        stream = QualificationRecordStream(expected)
        # One-byte chunks deliberately split every multibyte UTF-8 sequence.
        for byte in payload:
            stream.feed(bytes((byte,)))

        self.assertEqual(stream.finish(), tuple(records))

    def test_finish_requires_data_complete_lf_and_twelve_record_lifecycle(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        cases = (
            b"",
            _payload(records)[:-1],
            _payload(records[:11]),
        )
        for candidate in cases:
            with self.subTest(payload_length=len(candidate)):
                stream = QualificationRecordStream(expected)
                try:
                    stream.feed(candidate)
                except (TypeError, ValueError):
                    continue
                with self.assertRaises(ValueError):
                    stream.finish()

    def test_expected_mapping_binds_all_six_values(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        changed: dict[str, Any] = {
            "manifest_sha256": "d" * 64,
            "sequence_sha256": "e" * 64,
            "profile_id": "cache-12gib",
            "prompt_order_index": 1,
            "prompt_id": "native-2048",
            "input_sha256": "f" * 64,
        }
        for key, value in changed.items():
            candidate = dict(expected)
            candidate[key] = value
            with self.subTest(binding=key):
                _assert_rejected(self, candidate, _payload(records))

        for key in EXPECTED_KEYS:
            candidate = dict(expected)
            del candidate[key]
            with self.subTest(missing_binding=key):
                with self.assertRaises((TypeError, ValueError)):
                    QualificationRecordStream(candidate)

    def test_bindings_cannot_change_after_the_first_repetition(self) -> None:
        baseline = _lifecycle_records()
        expected = _expected(baseline)
        changed: dict[str, Any] = {
            "manifest_sha256": "d" * 64,
            "sequence_sha256": "e" * 64,
            "profile_id": "cache-12gib",
            "prompt_order_index": 1,
            "prompt_id": "native-2048",
            "input_sha256": "f" * 64,
        }
        for key, value in changed.items():
            records = copy.deepcopy(baseline)
            records[3][key] = value
            with self.subTest(binding=key):
                _assert_rejected(self, expected, _payload(records))

    def test_instance_id_must_remain_stable_across_repetitions(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        changed_instance = "123e4567-e89b-12d3-a456-426614174099"
        records[3]["instance_id"] = changed_instance
        records[3]["runtime"]["instance_id"] = changed_instance
        _VALIDATOR.validate_record(records[3])
        _assert_rejected(self, expected, _payload(records))

    def test_duplicate_out_of_order_events_and_request_reuse_are_rejected(self) -> None:
        baseline = _lifecycle_records()
        expected = _expected(baseline)

        duplicate_event = copy.deepcopy(baseline)
        duplicate_event[1]["event"] = "request_accepted"

        out_of_order = copy.deepcopy(baseline)
        out_of_order[1], out_of_order[2] = out_of_order[2], out_of_order[1]

        reused_request = copy.deepcopy(baseline)
        reused_id = reused_request[0]["request_id"]
        for index in range(3, 6):
            reused_request[index]["request_id"] = reused_id
        reused_request[5]["request_metrics"]["request_id"] = reused_id
        for name, records in (
            ("duplicate_event", duplicate_event),
            ("out_of_order", out_of_order),
            ("reused_request", reused_request),
        ):
            with self.subTest(case=name):
                _assert_rejected(self, expected, _payload(records))

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_wrong_record_types(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        valid = _json_line(records[0])
        duplicate_key = valid.replace(
            b'"schema":"ds4.bench.qualification/v1",',
            b'"schema":"ds4.bench.qualification/v1","schema":"ds4.bench.qualification/v1",',
            1,
        )
        nonfinite = valid.replace(b'"monotonic_ns":"1000000"', b'"monotonic_ns":NaN', 1)
        infinity = valid.replace(b'"monotonic_ns":"1000000"', b'"monotonic_ns":Infinity', 1)
        candidates = (
            ("duplicate_key", duplicate_key),
            ("nan", nonfinite),
            ("infinity", infinity),
            ("array", b"[]\n"),
            ("string", b'"record"\n'),
            ("null", b"null\n"),
            ("number", b"1\n"),
        )
        for name, candidate in candidates:
            with self.subTest(case=name):
                _assert_rejected(self, expected, candidate)

    def test_crlf_and_invalid_utf8_are_rejected(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        _assert_rejected(self, expected, _payload(records).replace(b"\n", b"\r\n"))
        _assert_rejected(self, expected, b"\xff\n")

    def test_record_and_stream_byte_caps_are_hard_and_exported(self) -> None:
        self.assertEqual(MAX_RECORD_BYTES, 1_048_576)
        self.assertEqual(MAX_STREAM_BYTES, 12_582_912)
        records = _lifecycle_records()
        expected = _expected(records)

        line = _json_line(records[0])
        oversized_line = line[:-1] + b" " * (MAX_RECORD_BYTES - len(line) + 2) + b"\n"
        _assert_rejected(self, expected, oversized_line)

        # Keep each complete line below its cap, but take twelve of them to
        # exactly approach the total cap before one unfinished byte crosses it.
        padded_lines: list[bytes] = []
        for record in records:
            base_line = _json_line(record)
            padded_lines.append(
                base_line[:-1]
                + b" " * (MAX_RECORD_BYTES - len(base_line) - 1)
                + b"\n"
            )
        near_limit = b"".join(padded_lines)
        self.assertEqual(len(near_limit), MAX_STREAM_BYTES - 12)
        stream = QualificationRecordStream(expected)
        try:
            for padded_line in padded_lines:
                stream.feed(padded_line)
        except (TypeError, ValueError):
            self.fail("twelve valid padded records must fit below the stream cap")
        with self.assertRaises(ValueError):
            stream.feed(b" " * 13)

    def test_thirteenth_record_is_rejected(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        _assert_rejected(self, expected, _payload(records + [copy.deepcopy(records[0])]))

    def test_failed_stream_cannot_recover_with_valid_data(self) -> None:
        records = _lifecycle_records()
        expected = _expected(records)
        stream = QualificationRecordStream(expected)
        with self.assertRaises(ValueError):
            stream.feed(b"[]\n")
        with self.assertRaises(ValueError):
            stream.feed(_payload(records))
        with self.assertRaises(ValueError):
            stream.finish()

    def test_constructor_and_feed_reject_wrong_input_types(self) -> None:
        for expected in (None, [], "expected"):
            with self.subTest(expected_type=type(expected).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    QualificationRecordStream(expected)

        records = _lifecycle_records()
        stream = QualificationRecordStream(_expected(records))
        for chunk in (None, "{}\n", 7):
            with self.subTest(chunk_type=type(chunk).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    stream.feed(chunk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
