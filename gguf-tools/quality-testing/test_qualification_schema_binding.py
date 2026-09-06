#!/usr/bin/env python3
"""RED tests for admission-bound qualification-record schema snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import qualification_admission as ADMISSION
import qualification_records as STREAMED
import qualification_resident_records as RESIDENT
import test_qualification_admission as ADMISSION_TEST
import test_qualification_records as STREAMED_TEST
import test_qualification_resident_records as RESIDENT_TEST


STREAMED_SCHEMA = "ds4-bench-qualification-v1.schema.json"
RESIDENT_SCHEMA = "ds4-bench-resident-qualification-v1.schema.json"
RUNTIME_SCHEMA = "ds4-runtime-v1.schema.json"
REQUEST_SCHEMA = "ds4-runtime-request-v1.schema.json"
SCHEMA_DOCUMENTS = (STREAMED_SCHEMA, RESIDENT_SCHEMA, RUNTIME_SCHEMA, REQUEST_SCHEMA)


@contextmanager
def _admitted(inputs: ADMISSION_TEST.Inputs):
    """Admit the real tiny fixture with ordinary host-only version claims."""

    def host_only_probe(role: str, artifact: Any) -> bytes:
        # This is deliberately not a native/process qualification claim.
        return inputs.version_bytes[role]

    with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
        with ADMISSION.admit_qualification_inputs(
            inputs.manifest_path,
            model_path=inputs.model,
            binaries=dict(inputs.binaries),
            version_probe=host_only_probe,
        ) as admission:
            yield admission


@contextmanager
def _poison_schema_paths():
    """Make every post-admission schema-path fallback fail loudly."""

    poison_root = Path("/task20-schema-binding-poison")
    selectors = (
        (STREAMED, "SCHEMA_PATH"),
        (STREAMED, "RUNTIME_SCHEMA_PATH"),
        (STREAMED, "REQUEST_SCHEMA_PATH"),
        (RESIDENT, "SCHEMA_PATH"),
    )
    with ExitStack() as stack:
        for module, name in selectors:
            stack.enter_context(
                mock.patch.object(module, name, poison_root / f"{module.__name__}-{name}")
            )
        stack.enter_context(
            mock.patch.object(STREAMED, "ROOT", poison_root)
        )
        stack.enter_context(
            mock.patch.object(
                Path, "read_bytes", side_effect=AssertionError("schema path read")
            )
        )
        stack.enter_context(
            mock.patch.object(
                Path, "read_text", side_effect=AssertionError("schema path read")
            )
        )
        stack.enter_context(
            mock.patch.object(
                Path, "open", side_effect=AssertionError("schema path open")
            )
        )
        stack.enter_context(
            mock.patch.object(
                Path, "is_file", side_effect=AssertionError("schema path selector")
            )
        )
        yield


def _streamed_case(
    manifest_sha256: str | None = None,
    sequence_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    records = STREAMED_TEST._lifecycle_records()
    for record in records:
        if manifest_sha256 is not None:
            record["manifest_sha256"] = manifest_sha256
        if sequence_sha256 is not None:
            record["sequence_sha256"] = sequence_sha256
    return records, STREAMED_TEST._expected(records), STREAMED_TEST._payload(records)


def _resident_case(
    manifest_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    records = RESIDENT_TEST._fresh_slice()
    if manifest_sha256 is not None:
        for record in records:
            record["manifest_sha256"] = manifest_sha256
    return records, RESIDENT_TEST._expected(records), RESIDENT_TEST._payload(records)


def _line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")


def _mutated_schema_bytes(path: Path) -> bytes:
    return path.read_bytes() + b"\n"


def _append_sequence_constraint(path: Path, digest: str) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    all_of = document.setdefault("allOf", [])
    if not isinstance(all_of, list):
        raise AssertionError("record schema allOf must remain a list")
    all_of.append({"properties": {"sequence_sha256": {"const": digest}}})
    path.write_text(
        json.dumps(document, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _assert_feed_rejected(
    test: unittest.TestCase,
    constructor: Callable[..., Any],
    expected: dict[str, Any],
    payload: bytes,
    admission: Any,
) -> None:
    stream = constructor(expected, input_admission=admission)
    with test.assertRaises((TypeError, ValueError, OSError)):
        stream.feed(payload)


def _overflow_number_line(record: dict[str, Any]) -> bytes:
    sentinel = "__task20_prefill_tokens_overflow_sentinel__"
    candidate = copy.deepcopy(record)
    candidate["request_metrics"]["prefill_tokens_per_second"] = sentinel
    line = _line(candidate)
    token = f'"prefill_tokens_per_second":"{sentinel}"'.encode("utf-8")
    if line.count(token) != 1:
        raise AssertionError("overflow sentinel must identify one complete field")
    overflow = line.replace(token, b'"prefill_tokens_per_second":1e999', 1)
    decoded = json.loads(overflow)
    value = decoded["request_metrics"]["prefill_tokens_per_second"]
    if not math.isinf(value) or value <= 0:
        raise AssertionError("1e999 must parse as positive infinity before validation")
    return overflow


class QualificationSchemaBindingTest(unittest.TestCase):
    def test_admission_exposes_detached_four_document_snapshots(self) -> None:
        with ADMISSION_TEST._tiny_inputs() as inputs:
            with _admitted(inputs) as admission:
                documents = admission.record_schema_documents
                self.assertIsInstance(documents, Mapping)
                self.assertEqual(set(documents), set(SCHEMA_DOCUMENTS))
                owners = {
                    owner.path.name: owner
                    for owner in admission._owners
                    if owner.path.parent == inputs.schema_root
                }
                for name in SCHEMA_DOCUMENTS:
                    payload = inputs.schema_bytes[name]
                    self.assertEqual(
                        documents[name], json.loads(payload.decode("utf-8"))
                    )
                    self.assertEqual(
                        owners[name].sha256, hashlib.sha256(payload).hexdigest()
                    )

                snapshot = copy.deepcopy(documents)
                documents[STREAMED_SCHEMA]["properties"]["schema"]["const"] = "caller-mutated"
                documents[RUNTIME_SCHEMA]["properties"]["schema"]["const"] = "caller-mutated"
                self.assertEqual(admission.record_schema_documents, snapshot)

    def test_fixed_streamed_and_resident_parsers_consume_only_admitted_snapshots(self) -> None:
        with ADMISSION_TEST._tiny_inputs() as inputs:
            with _admitted(inputs) as admission:
                streamed_records, streamed_expected, streamed_payload = _streamed_case(
                    admission.manifest_sha256
                )
                resident_records, resident_expected, resident_payload = _resident_case(
                    admission.manifest_sha256
                )
                with _poison_schema_paths():
                    streamed = STREAMED.QualificationRecordStream(
                        streamed_expected, input_admission=admission
                    )
                    streamed.feed(streamed_payload)
                    self.assertEqual(streamed.finish(), tuple(streamed_records))

                    resident = RESIDENT.QualificationResidentRecordStream(
                        resident_expected, input_admission=admission
                    )
                    resident.feed(resident_payload)
                    self.assertEqual(resident.finish(), tuple(resident_records))

    def test_snapshot_copy_mutation_cannot_weaken_current_or_new_parser(self) -> None:
        with ADMISSION_TEST._tiny_inputs() as inputs:
            with _admitted(inputs) as admission:
                records, expected, payload = _streamed_case(
                    admission.manifest_sha256
                )
                baseline = copy.deepcopy(admission.record_schema_documents)
                with _poison_schema_paths():
                    current = STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )
                    current.feed(_line(records[0]))
                    caller_copy = admission.record_schema_documents
                    for name in SCHEMA_DOCUMENTS:
                        caller_copy[name]["properties"]["schema"]["const"] = "caller-mutated"
                    self.assertEqual(admission.record_schema_documents, baseline)
                    current.feed(b"".join(_line(record) for record in records[1:]))
                    self.assertEqual(current.finish(), tuple(records))

                    fresh = STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )
                    fresh.feed(payload)
                    self.assertEqual(fresh.finish(), tuple(records))

    def test_independent_admissions_interleave_without_global_schema_state(self) -> None:
        left_digest = "a" * 64
        right_digest = "b" * 64
        with ADMISSION_TEST._tiny_inputs() as left_inputs:
            _append_sequence_constraint(
                left_inputs.schema_root / STREAMED_SCHEMA, left_digest
            )
            with _admitted(left_inputs) as left_admission:
                left_records, left_expected, _ = _streamed_case(
                    left_admission.manifest_sha256, left_digest
                )
                with ADMISSION_TEST._tiny_inputs() as right_inputs:
                    _append_sequence_constraint(
                        right_inputs.schema_root / STREAMED_SCHEMA, right_digest
                    )
                    with _admitted(right_inputs) as right_admission:
                        right_records, right_expected, _ = _streamed_case(
                            right_admission.manifest_sha256, right_digest
                        )
                        # Build cross-payload fixtures before trapping all
                        # schema-path and fixture-loader reads.
                        _, left_admission_expected, left_admission_payload = _streamed_case(
                            left_admission.manifest_sha256, right_digest
                        )
                        _, right_admission_expected, right_admission_payload = _streamed_case(
                            right_admission.manifest_sha256, left_digest
                        )
                        with _poison_schema_paths():
                            left = STREAMED.QualificationRecordStream(
                                left_expected, input_admission=left_admission
                            )
                            right = STREAMED.QualificationRecordStream(
                                right_expected, input_admission=right_admission
                            )
                            for left_record, right_record in zip(
                                left_records, right_records, strict=True
                            ):
                                left.feed(_line(left_record))
                                right.feed(_line(right_record))
                            self.assertEqual(left.finish(), tuple(left_records))
                            self.assertEqual(right.finish(), tuple(right_records))

                            # Keep each cross-payload's expected bindings aligned
                            # with its admission; only the admitted schema const
                            # differs, so this must reject the opposite snapshot.
                            self.assertEqual(
                                left_admission_expected["manifest_sha256"],
                                left_admission.manifest_sha256,
                            )
                            self.assertEqual(
                                right_admission_expected["manifest_sha256"],
                                right_admission.manifest_sha256,
                            )
                            _assert_feed_rejected(
                                self,
                                STREAMED.QualificationRecordStream,
                                left_admission_expected,
                                left_admission_payload,
                                left_admission,
                            )
                            _assert_feed_rejected(
                                self,
                                STREAMED.QualificationRecordStream,
                                right_admission_expected,
                                right_admission_payload,
                                right_admission,
                            )

    def test_explicit_none_preserves_legacy_schema_path_defaults(self) -> None:
        streamed_records, streamed_expected, streamed_payload = _streamed_case()
        resident_records, resident_expected, resident_payload = _resident_case()
        streamed = STREAMED.QualificationRecordStream(
            streamed_expected, input_admission=None
        )
        streamed.feed(streamed_payload)
        self.assertEqual(streamed.finish(), tuple(streamed_records))
        resident = RESIDENT.QualificationResidentRecordStream(
            resident_expected, input_admission=None
        )
        resident.feed(resident_payload)
        self.assertEqual(resident.finish(), tuple(resident_records))

    def test_nested_integer_and_finite_number_rejections_remain_strict(self) -> None:
        cases = (
            ("streamed", STREAMED.QualificationRecordStream, _streamed_case),
            ("resident", RESIDENT.QualificationResidentRecordStream, _resident_case),
        )
        for kind, constructor, fixture in cases:
            with self.subTest(kind=kind):
                with ADMISSION_TEST._tiny_inputs() as inputs:
                    with _admitted(inputs) as admission:
                        records, expected, payload = fixture(admission.manifest_sha256)
                        with _poison_schema_paths():
                            control = constructor(
                                expected, input_admission=admission
                            )
                            control.feed(payload)
                            self.assertEqual(control.finish(), tuple(records))

                            original_integer = records[2]["runtime"]["config"][
                                "context_tokens"
                            ]
                            self.assertIs(type(original_integer), int)
                            integral_float = copy.deepcopy(records[2])
                            integral_float["runtime"]["config"][
                                "context_tokens"
                            ] = float(original_integer)
                            boolean_integer = copy.deepcopy(records[2])
                            boolean_integer["runtime"]["config"][
                                "context_tokens"
                            ] = True
                            nan_number = copy.deepcopy(records[2])
                            nan_number["request_metrics"][
                                "prefill_tokens_per_second"
                            ] = float("nan")
                            prefix = b"".join(_line(record) for record in records[:2])
                            for label, bad_tail in (
                                ("integral-float", _line(integral_float)),
                                ("boolean-integer", _line(boolean_integer)),
                                ("nan", _line(nan_number)),
                                ("overflow", _overflow_number_line(records[2])),
                            ):
                                with self.subTest(field=label):
                                    _assert_feed_rejected(
                                        self,
                                        constructor,
                                        expected,
                                        prefix + bad_tail,
                                        admission,
                                    )

    def test_fixed_kind_rejects_the_other_record_document(self) -> None:
        with ADMISSION_TEST._tiny_inputs() as inputs:
            with _admitted(inputs) as admission:
                _, streamed_expected, streamed_payload = _streamed_case(
                    admission.manifest_sha256
                )
                _, resident_expected, resident_payload = _resident_case(
                    admission.manifest_sha256
                )
                with _poison_schema_paths():
                    # Each expected mapping matches the payload being fed.  The
                    # fixed class, not a binding mismatch, must reject the kind.
                    _assert_feed_rejected(
                        self,
                        STREAMED.QualificationRecordStream,
                        resident_expected,
                        resident_payload,
                        admission,
                    )
                    _assert_feed_rejected(
                        self,
                        RESIDENT.QualificationResidentRecordStream,
                        streamed_expected,
                        streamed_payload,
                        admission,
                    )

    def test_manifest_mismatch_and_wrong_or_closed_admission_reject(self) -> None:
        with ADMISSION_TEST._tiny_inputs() as inputs:
            with _admitted(inputs) as admission:
                _, expected, _ = _streamed_case(admission.manifest_sha256)
                mismatched = dict(expected)
                mismatched["manifest_sha256"] = "0" * 64
                with _poison_schema_paths():
                    with self.assertRaises((TypeError, ValueError, OSError)):
                        STREAMED.QualificationRecordStream(
                            mismatched, input_admission=admission
                        )
                    for wrong_admission in (object(), lambda: admission):
                        with self.subTest(wrong_admission=type(wrong_admission).__name__):
                            with self.assertRaises((TypeError, ValueError)):
                                STREAMED.QualificationRecordStream(
                                    expected, input_admission=wrong_admission
                                )

            with _poison_schema_paths():
                with self.assertRaises((TypeError, ValueError, OSError)):
                    STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )

    def test_schema_drift_before_between_records_and_before_finish_preserves_prefix(self) -> None:
        schema_name = STREAMED_SCHEMA

        constructed = mutated = feed_rejected = False
        try:
            with ADMISSION_TEST._tiny_inputs() as inputs:
                with _admitted(inputs) as admission:
                    records, expected, _ = _streamed_case(
                        admission.manifest_sha256
                    )
                    stream = STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )
                    constructed = True
                    (inputs.schema_root / schema_name).write_bytes(
                        _mutated_schema_bytes(inputs.schema_root / schema_name)
                    )
                    mutated = True
                    with _poison_schema_paths():
                        try:
                            stream.feed(_line(records[0]))
                        except (ValueError, OSError):
                            feed_rejected = True
                        else:
                            self.fail("schema drift before first record was accepted")
        except (ValueError, OSError):
            pass
        self.assertTrue(constructed, "drift setup did not reach parser construction")
        self.assertTrue(mutated, "drift setup did not reach mutation")
        self.assertTrue(feed_rejected, "drift before first record was not rejected")

        first_record_ok = mutated = feed_rejected = False
        prefix: tuple[dict[str, Any], ...] = ()
        try:
            with ADMISSION_TEST._tiny_inputs() as inputs:
                with _admitted(inputs) as admission:
                    records, expected, _ = _streamed_case(
                        admission.manifest_sha256
                    )
                    stream = STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )
                    with _poison_schema_paths():
                        stream.feed(_line(records[0]))
                    first_record_ok = True
                    (inputs.schema_root / schema_name).write_bytes(
                        _mutated_schema_bytes(inputs.schema_root / schema_name)
                    )
                    mutated = True
                    with _poison_schema_paths():
                        try:
                            stream.feed(_line(records[1]))
                        except (ValueError, OSError):
                            feed_rejected = True
                        else:
                            self.fail("schema drift between records was accepted")
                    prefix = stream.drain_records()
        except (ValueError, OSError):
            pass
        self.assertTrue(first_record_ok, "between-record setup lost its valid prefix")
        self.assertTrue(mutated, "between-record setup did not reach mutation")
        self.assertTrue(feed_rejected, "schema drift between records was not rejected")
        self.assertEqual(prefix, (records[0],))

        full_stream_ok = mutated = finish_rejected = False
        prefix = ()
        try:
            with ADMISSION_TEST._tiny_inputs() as inputs:
                with _admitted(inputs) as admission:
                    records, expected, _ = _streamed_case(
                        admission.manifest_sha256
                    )
                    stream = STREAMED.QualificationRecordStream(
                        expected, input_admission=admission
                    )
                    with _poison_schema_paths():
                        stream.feed(b"".join(_line(record) for record in records))
                    full_stream_ok = True
                    (inputs.schema_root / schema_name).write_bytes(
                        _mutated_schema_bytes(inputs.schema_root / schema_name)
                    )
                    mutated = True
                    with _poison_schema_paths():
                        try:
                            stream.finish()
                        except (ValueError, OSError):
                            finish_rejected = True
                        else:
                            self.fail("schema drift before finish was accepted")
                    prefix = stream.drain_records()
        except (ValueError, OSError):
            pass
        self.assertTrue(full_stream_ok, "finish setup did not reach a complete stream")
        self.assertTrue(mutated, "finish setup did not reach mutation")
        self.assertTrue(finish_rejected, "schema drift before finish was not rejected")
        self.assertEqual(prefix, tuple(records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
