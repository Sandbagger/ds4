#!/usr/bin/env python3
"""Host-only RED tests for the resident qualification-record consumer.

Positive bytes come from the checked-in native resident emitter through a tiny
model-free helper.  No model, GPU, service, network, or production process is
started by these tests.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
EMITTER = (ROOT / "tests/test_bench_resident_qualification_emitter.c").resolve()
# Keep RED at the import boundary: a missing consumer fails before compilation.
import qualification_resident_records as resident_module
from qualification_resident_records import (
    EXPECTED_BINDINGS, MAX_JSON_NESTING, MAX_RECORD_BYTES, MAX_RECORD_COUNT,
    MAX_STREAM_BYTES, QualificationResidentRecordStream, validate_resident_record,
)
import qualification_records as qualification_records_module
from qualification_records import QualificationRecordStream, validate_record
from test_qualification_records import _expected as streamed_expected
from test_qualification_records import _lifecycle_records as streamed_lifecycle

EXPECTED_KEYS = ("manifest_sha256", "sequence_sha256", "profile_id",
                 "prompt_order_index", "prompt_id", "input_sha256")
PROMPTS = ("native-512", "native-2048", "native-8192", "native-28672")
EVENTS = ("request_accepted", "first_token", "request_complete")
_HELPER = f"""#define main old_fixture_main
#include \"{EMITTER}\"
#undef main
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
int main(void) {{
    alarm(15);
    ds4_runtime_wire_snapshot runtime;
    fill_runtime(&runtime);
    for (uint32_t order = 0u; order < 4u; order++) {{
        ds4_bench_resident_sequence resident;
        fill_sequence(&resident, order);
        for (uint32_t repetition = 0u; repetition < 4u; repetition++) {{
            const uint64_t base = 100u + 4u * repetition;
            const uint64_t accepted_time = 1000000u + 1000u * repetition;
            ds4_runtime_request_metrics metrics;
            fill_metrics(&metrics, request_ids[repetition], base + 3u,
                         prompt_tokens[order]);
            for (uint32_t event = 0u; event < 3u; event++) {{
                runtime.snapshot_seq = base +
                    (event == 0u ? 1u : event == 1u ? 2u : 4u);
                metrics.snapshot_seq = base + 3u;
                ds4_bench_resident_qualification_record record = {{
                    .sequence = &resident,
                    .event = (ds4_bench_qualification_event)event,
                    .request_id = request_ids[repetition],
                    .repetition_index = repetition,
                    .monotonic_ns = accepted_time +
                        (event == 0u ? 0u : event == 1u ? 100u : 200u),
                    .session_payload_bytes = 9000u,
                    .runtime_snapshot = &runtime,
                    .request_metrics = event == 2u ? &metrics : NULL,
                }};
                char error[256] = {{0}};
                if (!ds4_bench_resident_qualification_emit_record(
                        stdout, &record, error, sizeof(error))) {{
                    fprintf(stderr, \"resident emitter failed: %s\\n\", error);
                    return 2;
                }}
            }}
        }}
    }}
    alarm(0);
    return 0;
}}
"""

@lru_cache(maxsize=1)
def _native_payload() -> bytes:
    with tempfile.TemporaryDirectory(prefix="resident-records-") as directory:
        directory_path = Path(directory)
        source, executable = directory_path / "helper.c", directory_path / "helper"
        source.write_text(_HELPER, encoding="utf-8")
        result = subprocess.run([
            os.environ.get("CC", "cc"), "-std=c99", "-I", str(ROOT), str(source),
            str(ROOT / "ds4_bench_qualification.c"), str(ROOT / "ds4_runtime.c"),
            "-lm", "-pthread", "-o", str(executable)], capture_output=True,
            text=True, timeout=30, check=False)
        if result.returncode:
            raise AssertionError(f"resident fixture compile failed:\n{result.stderr}")
        result = subprocess.run([str(executable)], capture_output=True,
                                timeout=15, check=False)
        if result.returncode:
            raise AssertionError(f"resident fixture failed ({result.returncode}): "
                                 f"{result.stderr.decode(errors='replace')}")
        return result.stdout

@lru_cache(maxsize=1)
def _native_records() -> tuple[dict[str, Any], ...]:
    lines = _native_payload().splitlines(keepends=True)
    if len(lines) != 48 or any(not line.endswith(b"\n") for line in lines):
        raise AssertionError("resident helper must emit 48 LF-terminated records")
    return tuple(json.loads(line[:-1]) for line in lines)

def _fresh_slice(order: int = 0) -> list[dict[str, Any]]:
    records = copy.deepcopy(_native_records()[order * 12:(order + 1) * 12])
    for record in records:
        validate_resident_record(record)
    return records

def _expected(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: records[0][key] for key in EXPECTED_KEYS}

def _json_line(record: Any) -> bytes:
    return (json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n").encode()

def _payload(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_line(record) for record in records)

def _set(record: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        record = record[part]
    record[parts[-1]] = value

def _stream_reject(test: unittest.TestCase, expected: dict[str, Any], data: bytes) -> None:
    stream = QualificationResidentRecordStream(expected)
    try:
        stream.feed(data)
    except (TypeError, ValueError):
        return
    with test.assertRaises((TypeError, ValueError)):
        stream.finish()

def _feed_reject(test: unittest.TestCase, expected: dict[str, Any], data: bytes) -> None:
    stream = QualificationResidentRecordStream(expected)
    with test.assertRaises((TypeError, ValueError)):
        stream.feed(data)

def _reject_field(test: unittest.TestCase, path: str, value: Any,
                  *, index: int = 0, order: int = 0) -> None:
    records = _fresh_slice(order)
    expected = _expected(records)
    _set(records[index], path, value)
    with test.subTest(field=path, index=index):
        with test.assertRaises((TypeError, ValueError)):
            validate_resident_record(records[index])
        _stream_reject(test, expected, _payload(records))

def _assert_flattenings(test: unittest.TestCase, record: dict[str, Any]) -> None:
    runtime, external = record["runtime"], record["external_attribution"]
    allocations, reports = runtime["allocations"], runtime["allocations"]["reports"]
    for field, nested in (
        ("kv_allocated_bytes", allocations["categories"]["kv_state"]["current_bytes"]),
        ("expert_cache_current_bytes", allocations["categories"]["expert_cache_payload"]["current_bytes"]),
        ("expert_cache_peak_bytes", allocations["categories"]["expert_cache_payload"]["peak_bytes"]),
        ("qualification_total_current_bytes", allocations["qualification_total"]["current_bytes"]),
        ("qualification_total_bound_bytes", allocations["qualification_total"]["bound_bytes"]),
        ("qualification_total_peak_bytes", allocations["qualification_total"]["peak_bytes"]),
        ("model_inode_resident_bytes", reports["model_source_resident"]["current_bytes"]),
    ):
        test.assertEqual(record[field], nested)
    for name in ("model_source_resident", "host_library_unattributed", "cuda_library_unattributed"):
        test.assertEqual(external[name], reports[name]["current_bytes"])

class ResidentQualificationRecordTest(unittest.TestCase):
    def test_native_emitter_covers_four_valid_slices_and_preserves_footprints(self) -> None:
        self.assertEqual(EXPECTED_BINDINGS, EXPECTED_KEYS)
        self.assertEqual(_native_payload().count(b"\n"), 48)
        for order, prompt_id in enumerate(PROMPTS):
            records = _fresh_slice(order)
            self.assertEqual(len(records), 12)
            self.assertEqual({r["instance_id"] for r in records}, {records[0]["instance_id"]})
            self.assertEqual({r["prompt_order_index"] for r in records}, {order})
            self.assertEqual({r["prompt_id"] for r in records}, {prompt_id})
            for repetition in range(4):
                base = 100 + 4 * repetition
                group = records[repetition * 3:repetition * 3 + 3]
                self.assertEqual([r["event"] for r in group], list(EVENTS))
                self.assertEqual([int(r["snapshot_seq"]) for r in group], [base + 1, base + 2, base + 4])
                self.assertEqual([int(r["monotonic_ns"]) for r in group], [1000000 + 1000 * repetition, 1000100 + 1000 * repetition, 1000200 + 1000 * repetition])
                self.assertEqual(group[2]["request_metrics"]["snapshot_seq"], str(base + 3))
            record, runtime = records[0], records[0]["runtime"]
            allocations, reports = runtime["allocations"], runtime["allocations"]["reports"]
            self.assertFalse(runtime["config"]["ssd_streaming"])
            self.assertEqual(runtime["config"]["ssd_streaming_cache_bytes"], "0")
            self.assertEqual(runtime["limits"]["expert_cache_limit_bytes"], "0")
            self.assertGreater(int(allocations["categories"]["static_weights"]["current_bytes"]), 0)
            self.assertGreater(int(reports["model_mapping_registered"]["current_bytes"]), 0)
            self.assertGreater(int(reports["model_source_resident"]["current_bytes"]), 0)
            for digest in ("manifest_sha256", "sequence_sha256", "input_sha256"):
                self.assertEqual(len(record[digest]), 64)
                self.assertNotEqual(set(record[digest]), {"0"})
            _assert_flattenings(self, record)

    def test_stream_accepts_fragmented_and_coalesced_slices_and_returns_copies(self) -> None:
        native_lines = _native_payload().splitlines(keepends=True)
        for order in range(4):
            records, expected = _fresh_slice(order), None
            expected = _expected(records)
            lines = native_lines[order * 12:(order + 1) * 12]
            data = b"".join(lines)
            stream = QualificationResidentRecordStream(expected)
            if order == 0:
                stream.feed(lines[0][:-1]); self.assertEqual(stream.drain_records(), ())
                stream.feed(lines[0][-1:]); prefix = stream.drain_records()
                self.assertEqual(prefix, (records[0],)); prefix[0]["runtime"]["model"]["id"] = "drain-mutation"
                stream.feed(b"".join(lines[1:])); self.assertEqual(stream.drain_records(), tuple(records[1:]))
            elif order == 1:
                for chunk in (data[:1], data[1:2048], data[2048:]): stream.feed(chunk)
                self.assertEqual(stream.drain_records(), tuple(records))
            else:
                stream.feed(data); self.assertEqual(stream.drain_records(), tuple(records))
            self.assertEqual(stream.finish(), tuple(records))
            copied = QualificationResidentRecordStream(expected); copied.feed(data)
            finished = copied.finish(); finished[0]["runtime"]["model"]["id"] = "finish-mutation"
            self.assertEqual(copied.drain_records(), tuple(records))

    def test_failed_stream_is_sticky_but_releases_valid_drained_prefix(self) -> None:
        records, expected = _fresh_slice(), None
        expected = _expected(records)
        stream = QualificationResidentRecordStream(expected)
        with self.assertRaises((TypeError, ValueError)): stream.feed(_json_line(records[0]) + b"[]\n")
        self.assertEqual(stream.drain_records(), (records[0],)); self.assertEqual(stream.drain_records(), ())
        with self.assertRaises((TypeError, ValueError)): stream.feed(_json_line(records[1]))
        with self.assertRaises((TypeError, ValueError)): stream.finish()

    def test_finish_requires_twelve_lf_terminated_records(self) -> None:
        records, expected, data = _fresh_slice(), None, None
        expected, data = _expected(records), _payload(records)
        for candidate in (b"", data[:-1], _payload(records[:11])):
            with self.subTest(length=len(candidate)):
                stream = QualificationResidentRecordStream(expected)
                try: stream.feed(candidate)
                except (TypeError, ValueError): continue
                with self.assertRaises((TypeError, ValueError)): stream.finish()
        self.assertEqual(MAX_RECORD_COUNT, 12)

    def test_expected_mapping_is_exact_and_binds_all_six_values(self) -> None:
        records, expected = _fresh_slice(), None
        expected = _expected(records)
        for key, value in {"manifest_sha256": "d" * 64, "sequence_sha256": "e" * 64,
                            "profile_id": "resident-other", "prompt_order_index": 1,
                            "prompt_id": "native-2048", "input_sha256": "f" * 64}.items():
            candidate = dict(expected); candidate[key] = value
            _stream_reject(self, candidate, _payload(records))
        for key in EXPECTED_KEYS:
            candidate = {name: value for name, value in expected.items() if name != key}
            with self.assertRaises((TypeError, ValueError)): QualificationResidentRecordStream(candidate)
        with self.assertRaises((TypeError, ValueError)):
            QualificationResidentRecordStream({**expected, "extra": "nope"})

    def test_isolated_schema_profile_mode_cache_rows_and_flattening_rejections(self) -> None:
        cases = (("schema", "ds4.bench.qualification/v1"), ("mode", "streamed"),
                 ("profile_id", "cache-8gib"), ("expert_cache_limit_bytes", "1"),
                 ("expert_cache_current_bytes", "1"), ("expert_cache_peak_bytes", "1"),
                 ("runtime.config.ssd_streaming", True), ("runtime.config.ssd_streaming_cache_bytes", "1"),
                 ("runtime.limits.expert_cache_limit_bytes", "1"),
                 ("runtime.allocations.categories.expert_cache_payload.bound_bytes", "1"),
                 ("runtime.allocations.categories.expert_cache_payload.current_bytes", "1"),
                 ("runtime.allocations.categories.expert_cache_payload.peak_bytes", "1"),
                 ("configured_prefill_rows", 2048), ("allocated_prefill_rows", 2048),
                 ("runtime.allocations.configured_prefill_rows", 2048),
                 ("runtime.allocations.allocated_prefill_rows", 2048),
                 ("kv_allocated_bytes", "1"), ("runtime.allocations.categories.kv_state.current_bytes", "1"),
                 ("qualification_total_current_bytes", "1"), ("qualification_total_bound_bytes", "1"),
                 ("qualification_total_peak_bytes", "1"), ("model_inode_resident_bytes", "1"),
                 ("external_attribution.model_source_resident", "1"),
                 ("external_attribution.host_library_unattributed", "1"),
                 ("external_attribution.cuda_library_unattributed", "1"))
        for path, value in cases: _reject_field(self, path, value)

    def test_isolated_request_metrics_are_bound_to_nested_runtime_and_record(self) -> None:
        other = "123e4567-e89b-12d3-a456-426614174099"
        for path, value in (("request_metrics.request_id", other), ("request_metrics.instance_id", other),
                            ("request_metrics.snapshot_seq", "102"), ("request_metrics.prompt_tokens", 513),
                            ("request_metrics.terminal_status", "cancelled"), ("terminal_status", "cancelled")):
            _reject_field(self, path, value, index=2)
        records = _fresh_slice(); records[2].pop("request_metrics")
        with self.assertRaises((TypeError, ValueError)): validate_resident_record(records[2])
        _stream_reject(self, _expected(records), _payload(records))

    def test_lifecycle_order_instance_reuse_and_snapshot_gaps_are_stream_rejections(self) -> None:
        baseline, expected = _fresh_slice(), None
        expected = _expected(baseline)
        duplicate_event = copy.deepcopy(baseline); duplicate_event[1]["event"] = "request_accepted"
        wrong_repetition = copy.deepcopy(baseline); wrong_repetition[3]["repetition_index"] = 0
        instance_drift = copy.deepcopy(baseline); new_instance = "123e4567-e89b-12d3-a456-426614174099"
        instance_drift[3]["instance_id"] = new_instance; instance_drift[3]["runtime"]["instance_id"] = new_instance
        reused_request = copy.deepcopy(baseline); reused_id = reused_request[0]["request_id"]
        for index in range(3, 6): reused_request[index]["request_id"] = reused_id
        reused_request[5]["request_metrics"]["request_id"] = reused_id
        snapshot_gap = copy.deepcopy(baseline); snapshot_gap[0]["snapshot_seq"] = "102"; snapshot_gap[0]["runtime"]["snapshot_seq"] = "102"
        for name, candidate in (("duplicate_event", duplicate_event), ("wrong_repetition", wrong_repetition),
                                ("instance_drift", instance_drift), ("reused_request", reused_request),
                                ("snapshot_gap", snapshot_gap)):
            with self.subTest(case=name):
                for record in candidate: validate_resident_record(record)
                _stream_reject(self, expected, _payload(candidate))
        _reject_field(self, "prompt_order_index", 1)

    def test_old_streamed_consumer_rejects_resident_and_resident_rejects_streamed(self) -> None:
        resident = _fresh_slice()
        with self.assertRaises((TypeError, ValueError)): validate_record(resident[0])
        old_stream = QualificationRecordStream(_expected(resident))
        with self.assertRaises((TypeError, ValueError)): old_stream.feed(_payload(resident))
        streamed = streamed_lifecycle(); streamed_record = copy.deepcopy(streamed[0]); validate_record(streamed_record)
        with self.assertRaises((TypeError, ValueError)): validate_resident_record(streamed_record)
        _feed_reject(self, streamed_expected(streamed), _json_line(streamed_record))

    def test_strict_json_crlf_depth_and_wire_caps_are_bounded(self) -> None:
        records = _fresh_slice()
        expected, data = _expected(records), _payload(records)
        line = _native_payload().splitlines(keepends=True)[0]
        # Establish that the unmodified native line and complete normalized
        # stream reach feed successfully before each rejection case.
        QualificationResidentRecordStream(expected).feed(line)
        QualificationResidentRecordStream(expected).feed(data)
        duplicate = line.replace(b'"schema":"ds4.bench.resident-qualification/v1",', b'"schema":"ds4.bench.resident-qualification/v1","schema":"ds4.bench.resident-qualification/v1",', 1)
        nonfinite = line.replace(b'"monotonic_ns":"1000000"', b'"monotonic_ns":NaN', 1)
        for candidate in (duplicate, nonfinite, b"\xff\n"):
            _feed_reject(self, expected, candidate)
        _feed_reject(self, expected, data.replace(b"\n", b"\r\n"))
        nested = b"[" * (MAX_JSON_NESTING + 1) + b"0" + b"]" * (MAX_JSON_NESTING + 1) + b"\n"
        deep_stream = QualificationResidentRecordStream(expected)
        with mock.patch.object(qualification_records_module.json, "loads", wraps=qualification_records_module.json.loads) as loads:
            with self.assertRaises((TypeError, ValueError)): deep_stream.feed(nested)
            loads.assert_not_called()
        with mock.patch.object(resident_module, "MAX_RECORD_BYTES", len(line) - 1):
            _feed_reject(self, expected, line)
        with mock.patch.object(resident_module, "MAX_STREAM_BYTES", len(data) - 1):
            _feed_reject(self, expected, data)
        _feed_reject(self, expected, data + line)
        for bad_expected in (None, [], "expected"):
            with self.assertRaises((TypeError, ValueError)): QualificationResidentRecordStream(bad_expected)
        for chunk in (None, "{}\n", 7):
            stream = QualificationResidentRecordStream(expected)
            with self.assertRaises((TypeError, ValueError)): stream.feed(chunk)
        self.assertEqual(MAX_RECORD_BYTES, resident_module.MAX_RECORD_BYTES)
        self.assertEqual(MAX_STREAM_BYTES, resident_module.MAX_STREAM_BYTES)

if __name__ == "__main__":
    unittest.main(verbosity=2)
