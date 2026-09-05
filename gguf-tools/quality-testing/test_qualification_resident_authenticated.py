#!/usr/bin/env python3
"""Host-only RED tests for authenticated resident qualification records.

The resident bytes come from the checked-in native emitter helper.  Child
execution reuses the existing generated control fake, authentication probe,
and bounded Darwin descriptor adapter.  The tests never start DS4, a model,
GPU, network, or service.  A private simulated proc tree is test data, not
Linux-native executable-origin evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

# Keep the missing keyword at the call boundary.  The first explicit resident
# call is intentionally RED until the parent-owned record_kind seam exists.
from qualification_authenticated import run_authenticated_qualification_child
from qualification_controlled import run_qualification_controlled_child
from qualification_process import _qualification_child_transport, run_qualification_child
from qualification_supervisor import QualificationSliceMonitor

import compact_runtime_qualify as COMPACT
import qualification_process as PROCESS
import test_qualification_authenticated as AUTH_TEST
import test_qualification_resident_records as RESIDENT_TEST
from qualification_artifacts import open_qualification_artifact
from qualification_records import QualificationRecordStream, validate_record
from qualification_resident_records import (
    QualificationResidentRecordStream,
    validate_resident_record,
)
from test_qualification_records import (
    _expected as streamed_expected,
    _lifecycle_records as streamed_lifecycle,
)


POSIX_PROCESS_GROUPS = AUTH_TEST.POSIX_PROCESS_GROUPS
EXPECTED_KEYS = RESIDENT_TEST.EXPECTED_KEYS
MODEL_FD = COMPACT._QUALIFICATION_CONTROL_MODEL_FD
READY = COMPACT._QUALIFICATION_CONTROL_SAMPLE_READY
READY_ACK = COMPACT._QUALIFICATION_CONTROL_SAMPLE_READY_ACK
RESULT = COMPACT._QUALIFICATION_CONTROL_SAMPLE_RESULT
RESULT_ACK = COMPACT._QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK
MODEL_ACK = COMPACT._QUALIFICATION_CONTROL_MODEL_FD_ACK
WIRE = COMPACT._QUALIFICATION_CONTROL_MESSAGE
FD_ROOT = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


def _invoke_authenticated(
    records: list[dict[str, Any]],
    expected: dict[str, Any],
    record_kind: str | None,
    *,
    executable: Any,
    model_artifact: Any,
    probe: Any,
    metadata_path: Path,
    token: str,
    model: Path,
    fake: Path,
    payload: bytes,
    arguments: list[str],
    after_hook: Callable[[int, Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Invoke one child while borrowing caller-owned artifacts."""
    received_fds: list[int] = []
    preparation_refs: list[dict[str, Any]] = []
    before_refs: list[dict[str, Any]] = []
    after_refs: list[dict[str, Any]] = []

    def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
        fact = probe.callback("prepare", pid)
        received_fds.append(model_fd)
        fact.update(
            {
                "fd_inode": os.fstat(model_fd).st_ino,
                "sha256": evidence.sha256,
                "identity": evidence.identity.as_decimal_mapping(),
            }
        )
        preparation_refs.append(fact)
        return fact

    def before(pid: int, sequence: int) -> dict[str, Any]:
        fact = probe.callback("before", pid)
        fact["sequence"] = sequence
        before_refs.append(fact)
        return fact

    def after(pid: int, sequence: int) -> dict[str, Any]:
        fact = probe.callback("after", pid)
        fact["sequence"] = sequence
        if after_hook is not None:
            after_hook(sequence, model, fake)
        after_refs.append(fact)
        return fact

    call_kwargs: dict[str, Any] = {
        "executable_artifact": executable,
        "model_artifact": model_artifact,
        "probe": probe,
        "prepare_descriptor": prepare,
        "capture_before": before,
        "capture_after": after,
    }
    if record_kind is not None:
        call_kwargs["record_kind"] = record_kind
    result, requests = AUTH_TEST._call_bounded(
        arguments, expected, **call_kwargs
    )
    return {
        "result": result,
        "requests": requests,
        "probe": probe,
        "metadata_path": metadata_path,
        "token": token,
        "model": model,
        "payload": payload,
        "fake": fake,
        "arguments": arguments,
        "executable": executable,
        "model_artifact": model_artifact,
        "received_fds": received_fds,
        "preparation_refs": preparation_refs,
        "before_refs": before_refs,
        "after_refs": after_refs,
    }


@contextlib.contextmanager
def _authenticated_run(
    records: list[dict[str, Any]],
    expected: dict[str, Any],
    record_kind: str | None,
    *,
    mode: str = "complete",
    fake_hook: Callable[[Path], None] | None = None,
    after_hook: Callable[[int, Path, Path], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one owned fake while all retained owners remain in scope.

    ``record_kind=None`` deliberately omits the keyword.  That path is the
    diagnostic proving that the current streamed default rejects valid
    resident bytes for a contract reason rather than because the fake is bad.
    """
    payload = (
        f"authenticated {record_kind or 'default'} model bytes\0".encode("utf-8")
        * 29
    )
    with tempfile.TemporaryDirectory(prefix="qualification-resident-auth-") as tmp:
        model = Path(tmp) / "admitted-model.gguf"
        model.write_bytes(payload)
        with AUTH_TEST.CONTROLLED_TEST._controlled_fake_case(
            mode, records, model
        ) as (command, metadata_path, token):
            fake = Path(command[1])
            AUTH_TEST._adapt_controlled_fake(fake)
            if fake_hook is not None:
                fake_hook(fake)
            with (
                open_qualification_artifact(fake, executable=True) as executable,
                open_qualification_artifact(model) as model_artifact,
                tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-resident-auth-") as proc,
            ):
                probe = AUTH_TEST._AuthenticationProbe(
                    executable,
                    model_artifact,
                    Path(proc),
                    running_target=fake,
                )
                arguments = [
                    "--profile",
                    f"selected-{record_kind or 'default'}",
                    "--model",
                    "ignored.gguf",
                ]
                yield _invoke_authenticated(
                    records,
                    expected,
                    record_kind,
                    executable=executable,
                    model_artifact=model_artifact,
                    probe=probe,
                    metadata_path=metadata_path,
                    token=token,
                    model=model,
                    fake=fake,
                    payload=payload,
                    arguments=arguments,
                    after_hook=after_hook,
                )


def _enable_parent_payload(fake: Path) -> None:
    """Make one generated fake select a parent-specified JSON payload path."""
    source = fake.read_text(encoding="utf-8")
    needle = 'RECORDS = json.loads(json.dumps(CONFIG["records"]))\n'
    if source.count(needle) != 1:
        raise AssertionError("controlled fake record source seam changed")
    source = source.replace(
        needle,
        needle
        + "payload_paths = [value for index, value in enumerate(sys.argv)\n"
        + "                 if index and sys.argv[index - 1] == '--record-payload']\n"
        + "if len(payload_paths) != 1:\n"
        + "    raise SystemExit(43)\n"
        + "with open(payload_paths[0], 'r', encoding='utf-8') as handle:\n"
        + "    RECORDS = json.load(handle)\n",
        1,
    )
    fake.write_text(source, encoding="utf-8")
    fake.chmod(0o700)


@contextlib.contextmanager
def _shared_authenticated_session(
    cases: tuple[tuple[str, list[dict[str, Any]]], ...],
) -> Iterator[Callable[[str, list[dict[str, Any]], dict[str, Any]], dict[str, Any]]]:
    """Yield a runner that reuses one fake/model owner pair for two payloads."""
    if not cases:
        raise AssertionError("shared authenticated session needs payload cases")
    payload = b"one retained model owner for two payloads\0" * 29
    baseline_records = cases[0][1]
    with tempfile.TemporaryDirectory(prefix="qualification-resident-shared-") as tmp:
        directory = Path(tmp)
        model = directory / "admitted-model.gguf"
        model.write_bytes(payload)
        with AUTH_TEST.CONTROLLED_TEST._controlled_fake_case(
            "complete", baseline_records, model
        ) as (command, metadata_path, token):
            fake = Path(command[1])
            AUTH_TEST._adapt_controlled_fake(fake)
            payload_paths: dict[str, Path] = {}
            for record_kind, records in cases:
                path = directory / f"{record_kind}-payload.json"
                path.write_text(
                    json.dumps(records, separators=(",", ":"), ensure_ascii=False),
                    encoding="utf-8",
                )
                payload_paths[record_kind] = path
            _enable_parent_payload(fake)
            with (
                open_qualification_artifact(fake, executable=True) as executable,
                open_qualification_artifact(model) as model_artifact,
                tempfile.TemporaryDirectory(prefix="SIMULATED-PROC-resident-shared-") as proc,
            ):
                def run_once(
                    record_kind: str,
                    records: list[dict[str, Any]],
                    expected: dict[str, Any],
                ) -> dict[str, Any]:
                    try:
                        payload_path = payload_paths[record_kind]
                    except KeyError as exc:
                        raise AssertionError(f"unknown shared payload {record_kind!r}") from exc
                    probe = AUTH_TEST._AuthenticationProbe(
                        executable,
                        model_artifact,
                        Path(proc),
                        running_target=fake,
                    )
                    arguments = [
                        "--profile",
                        f"selected-{record_kind}",
                        "--model",
                        "ignored.gguf",
                        "--record-payload",
                        str(payload_path),
                    ]
                    return _invoke_authenticated(
                        records,
                        expected,
                        record_kind,
                        executable=executable,
                        model_artifact=model_artifact,
                        probe=probe,
                        metadata_path=metadata_path,
                        token=token,
                        model=model,
                        fake=fake,
                        payload=payload,
                        arguments=arguments,
                    )

                yield run_once

def _selected_stream(
    record_kind: str, expected: dict[str, Any]
) -> QualificationRecordStream | QualificationResidentRecordStream:
    if record_kind == "resident":
        return QualificationResidentRecordStream(expected)
    if record_kind == "streamed":
        return QualificationRecordStream(expected)
    raise AssertionError(f"test selected unknown record kind {record_kind!r}")


def _assert_full_ack_sequence(test: unittest.TestCase, run: dict[str, Any]) -> None:
    result = run["result"]
    expected_wire = [("receive", MODEL_FD, 0), ("send", MODEL_ACK, 0)]
    for sequence in range(1, 13):
        expected_wire.extend(
            (
                ("receive", READY, sequence),
                ("send", READY_ACK, sequence),
                ("receive", RESULT, sequence),
                ("send", RESULT_ACK, sequence),
            )
        )
    test.assertEqual(len(result.wire_records), len(expected_wire))
    model_identity = tuple(run["model_artifact"].identity)
    for index, (wire, expected) in enumerate(zip(result.wire_records, expected_wire)):
        AUTH_TEST.CONTROLLED_TEST._assert_wire_shape(test, wire)
        test.assertTrue(wire["complete"], index)
        values = WIRE.unpack(wire["payload"])
        test.assertEqual((wire["direction"], values[1], values[4]), expected)
        test.assertEqual(values[0], 1)
        test.assertEqual(values[2], WIRE.size)
        test.assertEqual(values[3], 0)
        if expected[1] == MODEL_FD:
            test.assertEqual(wire["descriptor_count"], 1)
            test.assertEqual(tuple(values[5:]), model_identity)
        else:
            test.assertEqual(wire["descriptor_count"], 0)
            if expected[1] in (RESULT, MODEL_ACK):
                test.assertEqual(tuple(values[5:]), model_identity)
            else:
                test.assertEqual(tuple(values[5:]), (0, 0, 0, 0))


def _assert_success(
    test: unittest.TestCase,
    run: dict[str, Any],
    records: list[dict[str, Any]],
    expected: dict[str, Any],
    record_kind: str,
) -> None:
    result = run["result"]
    AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(test, result)
    test.assertEqual(result.transport.reason, "complete")
    test.assertEqual(result.transport.returncode, 0)
    test.assertTrue(result.transport.cleanup_complete)
    test.assertIsNone(result.control_error)
    test.assertEqual(len(result.transport.records), 12)
    test.assertEqual(result.transport.stdout.count(b"\n"), 12)

    # Parse the retained raw bytes with the explicitly selected fixed parser.
    selected = _selected_stream(record_kind, expected)
    selected.feed(result.transport.stdout)
    parsed = selected.finish()
    test.assertIs(type(parsed), tuple)
    test.assertIs(type(result.transport.records), tuple)
    test.assertEqual(parsed, result.transport.records)
    test.assertEqual(len(parsed), len(records))
    for record in parsed:
        if record_kind == "resident":
            validate_resident_record(record)
        else:
            validate_record(record)

    probe = run["probe"]
    callback_names = [name for name, _pid in probe.callback_calls]
    test.assertEqual(
        callback_names,
        ["prepare"]
        + [name for _sequence in range(1, 13) for name in ("before", "after")],
    )
    callback_pids = {pid for _name, pid in probe.callback_calls}
    test.assertEqual(len(callback_pids), 1)
    self_pid = next(iter(callback_pids))
    test.assertTrue(
        all(owner is run["executable"] for _pid, owner in probe.auth_calls)
    )
    test.assertTrue(all(pid == self_pid for pid, _owner in probe.auth_calls))
    test.assertTrue(all(event[1] in ("executable", "model")
                        for event in probe.events
                        if event[0] == "verify"))
    AUTH_TEST._assert_authentication_boundaries(test, probe)

    test.assertEqual(result.preparation, run["preparation_refs"][0])
    for sequence, checkpoint in enumerate(result.checkpoints, 1):
        test.assertTrue(checkpoint["complete"])
        test.assertEqual(checkpoint["sequence"], sequence)
        test.assertEqual(checkpoint["before"], run["before_refs"][sequence - 1])
        test.assertEqual(checkpoint["after"], run["after_refs"][sequence - 1])
    test.assertEqual(len(result.checkpoints), 12)
    for fact in [
        run["preparation_refs"][0],
        *run["before_refs"],
        *run["after_refs"],
    ]:
        fact["caller_mutated"] = True
    test.assertNotIn("caller_mutated", result.preparation)
    test.assertNotIn("caller_mutated", result.checkpoints[0]["before"])
    test.assertNotIn("caller_mutated", result.checkpoints[0]["after"])

    evidence = result.model_evidence
    test.assertIsNotNone(evidence)
    test.assertEqual(evidence.sha256, hashlib.sha256(run["payload"]).hexdigest())
    test.assertEqual(evidence.identity, run["model_artifact"].identity)
    test.assertEqual(
        tuple(json.loads(line) for line in result.transport.stdout.splitlines()),
        result.transport.records,
    )
    for fd in run["received_fds"]:
        with test.assertRaises(OSError):
            os.fstat(fd)

    control_fd = AUTH_TEST._assert_spawn(
        test, run["requests"], run["probe"].executable_fd, run["arguments"]
    )
    metadata = AUTH_TEST._assert_cleanup(
        test, result, run["metadata_path"], run["token"]
    )
    test.assertEqual(
        metadata["argv"], run["arguments"] + ["--qualification-control-fd", str(control_fd)]
    )
    inherited = metadata["inherited_fd_inventory"]
    test.assertEqual(
        {entry["fd"] for entry in inherited},
        {run["executable"].fd, control_fd},
    )
    test.assertTrue(metadata["model_ack"])
    test.assertEqual(metadata["milestones"], list(range(1, 13)))
    _assert_full_ack_sequence(test, run)

    # The runner borrows both retained owners.  Cleanup must not invalidate
    # either owner, and no result object may grow a verdict/retry/publication.
    test.assertEqual(os.pread(run["executable"].fd, 2, 0), b"#!")
    test.assertEqual(os.pread(run["model_artifact"].fd, 16, 0), run["payload"][:16])
    run["executable"].verify()
    run["model_artifact"].verify()


def _assert_raw_first_record_matches(
    test: unittest.TestCase,
    run: dict[str, Any],
    expected: dict[str, Any],
    emitted_kind: str,
) -> None:
    raw_lines = run["result"].transport.stdout.splitlines()
    test.assertTrue(raw_lines, "the child did not emit its real first record")
    first = json.loads(raw_lines[0])
    for key in EXPECTED_KEYS:
        test.assertEqual(first[key], expected[key], key)
    if emitted_kind == "resident":
        validate_resident_record(first)
        test.assertEqual(first["schema"], "ds4.bench.resident-qualification/v1")
    else:
        validate_record(first)
        test.assertEqual(first["schema"], "ds4.bench.qualification/v1")


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "authenticated resident qualification needs POSIX process groups",
)
class QualificationResidentAuthenticatedHostTest(unittest.TestCase):
    def test_resident_then_streamed_successes_share_retained_owners(self) -> None:
        cases = (
            ("resident", RESIDENT_TEST._fresh_slice()),
            ("streamed", streamed_lifecycle()),
        )
        expected = {
            kind: (RESIDENT_TEST._expected(records) if kind == "resident"
                   else streamed_expected(records))
            for kind, records in cases
        }
        with _shared_authenticated_session(cases) as run_once:
            resident_kind, resident_records = cases[0]
            resident = run_once(resident_kind, resident_records, expected[resident_kind])
            _assert_success(
                self, resident, resident_records, expected[resident_kind], resident_kind
            )

            streamed_kind, streamed_records = cases[1]
            streamed = run_once(streamed_kind, streamed_records, expected[streamed_kind])
            self.assertIs(resident["fake"], streamed["fake"])
            self.assertIs(resident["executable"], streamed["executable"])
            self.assertIs(resident["model_artifact"], streamed["model_artifact"])
            self.assertIs(resident["model"], streamed["model"])
            self.assertEqual(resident["executable"].fd, streamed["executable"].fd)
            self.assertEqual(resident["model_artifact"].fd, streamed["model_artifact"].fd)
            self.assertNotEqual(resident["arguments"][-1], streamed["arguments"][-1])
            _assert_success(
                self, streamed, streamed_records, expected[streamed_kind], streamed_kind
            )

    def test_current_streamed_default_rejects_valid_resident_bytes_diagnostic(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)
        # No record_kind keyword here: this is the deliberate baseline proof.
        with _authenticated_run(records, expected, None) as run:
            result = run["result"]
            AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(self, result)
            self.assertEqual(result.transport.reason, "protocol_error")
            self.assertTrue(result.control_error)
            self.assertEqual(result.transport.records, ())
            _assert_raw_first_record_matches(self, run, expected, "resident")
            self.assertEqual(result.model_evidence.identity, run["model_artifact"].identity)
            AUTH_TEST._assert_cleanup(
                self, result, run["metadata_path"], run["token"]
            )
            self.assertEqual(result.wire_records[0]["direction"], "receive")
            self.assertEqual(result.wire_records[0]["descriptor_count"], 1)
            self.assertEqual(len(result.wire_records[0]["payload"]), WIRE.size)

    def test_selected_kind_rejects_the_other_real_record_contract_both_directions(self) -> None:
        cases = (
            ("resident-as-streamed", "streamed", RESIDENT_TEST._fresh_slice(), RESIDENT_TEST._expected, "resident"),
            ("streamed-as-resident", "resident", streamed_lifecycle(), streamed_expected, "streamed"),
        )
        for name, selected_kind, records, expected_factory, emitted_kind in cases:
            with self.subTest(case=name):
                expected = expected_factory(records)
                with _authenticated_run(records, expected, selected_kind) as run:
                    result = run["result"]
                    AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(self, result)
                    self.assertEqual(result.transport.reason, "protocol_error")
                    self.assertTrue(result.control_error)
                    _assert_raw_first_record_matches(self, run, expected, emitted_kind)
                    AUTH_TEST._assert_cleanup(
                        self, result, run["metadata_path"], run["token"]
                    )
                    self.assertIsNotNone(result.model_evidence)
                    self.assertEqual(
                        result.model_evidence.identity, run["model_artifact"].identity
                    )
                    self.assertEqual(
                        result.wire_records[0]["descriptor_count"], 1
                    )
                    self.assertFalse(
                        any(
                            wire["direction"] == "send"
                            and WIRE.unpack(wire["payload"])[1] == RESULT_ACK
                            and WIRE.unpack(wire["payload"])[4] == 12
                            for wire in result.wire_records
                        )
                    )

    def test_invalid_record_kind_never_reaches_popen_on_all_owner_layers(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)
        with tempfile.TemporaryDirectory(prefix="qualification-resident-preflight-") as tmp:
            directory = Path(tmp)
            executable_path = directory / "runner"
            executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable_path.chmod(0o700)
            model_path = directory / "model.gguf"
            model_path.write_bytes(b"preflight resident model")
            with (
                open_qualification_artifact(executable_path, executable=True) as executable,
                open_qualification_artifact(model_path) as model_artifact,
                mock.patch.object(PROCESS.subprocess, "Popen") as popen,
            ):
                callbacks = {
                    "prepare_descriptor": lambda *args: None,
                    "capture_before": lambda *args: None,
                    "capture_after": lambda *args: None,
                }
                command = ["/bin/false"]

                class KindAlias(str):
                    pass

                bad_kinds: tuple[Any, ...] = (
                    "not-a-kind",
                    KindAlias("resident"),
                    None,
                    b"resident",
                    "STREAMED",
                    7,
                    [],
                )
                for bad_kind in bad_kinds:
                    with self.subTest(record_kind=bad_kind):
                        with self.assertRaises((TypeError, ValueError)):
                            QualificationSliceMonitor(
                                expected, start_ns=0, record_kind=bad_kind
                            )
                        with self.assertRaises((TypeError, ValueError)):
                            with _qualification_child_transport(
                                command, expected, record_kind=bad_kind
                            ):
                                self.fail("invalid kind opened a child transport")
                        with self.assertRaises((TypeError, ValueError)):
                            run_qualification_child(
                                command, expected, record_kind=bad_kind
                            )
                        with self.assertRaises((TypeError, ValueError)):
                            run_qualification_controlled_child(
                                command,
                                expected,
                                **callbacks,
                                record_kind=bad_kind,
                            )
                        with self.assertRaises((TypeError, ValueError)):
                            run_authenticated_qualification_child(
                                ["--profile", "invalid"],
                                expected,
                                executable_artifact=executable,
                                model_artifact=model_artifact,
                                **callbacks,
                                record_kind=bad_kind,
                            )
                popen.assert_not_called()
                self.assertEqual(os.pread(executable.fd, 2, 0), b"#!")
                self.assertEqual(os.pread(model_artifact.fd, 8, 0), b"prefligh")
                executable.verify()
                model_artifact.verify()

    def test_resident_acceptance_deadline_keeps_one_valid_partial_record(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)

        def pause_after_acceptance(fake: Path) -> None:
            source = fake.read_text(encoding="utf-8")
            needle = (
                "        _metadata(\n"
                "            phase=\"milestone\",\n"
                "            milestone=sequence,\n"
                "            milestones=list(STATE[\"milestones\"]),\n"
                "        )\n"
            )
            self.assertEqual(source.count(needle), 1)
            source = source.replace(
                needle,
                needle
                + "        if MODE == 'stall' and sequence == 1:\n"
                + "            time.sleep(5.0)\n",
                1,
            )
            fake.write_text(source, encoding="utf-8")
            fake.chmod(0o700)

        with _authenticated_run(
            records,
            expected,
            "resident",
            mode="stall",
            fake_hook=pause_after_acceptance,
        ) as run:
            result = run["result"]
            AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(self, result)
            self.assertEqual(result.transport.reason, "timeout")
            self.assertEqual(result.transport.timeout_phase, "first_token")
            self.assertTrue(result.control_error)
            self.assertEqual(len(result.transport.records), 1)
            validate_resident_record(result.transport.records[0])
            self.assertEqual(result.transport.records[0]["event"], "request_accepted")
            self.assertEqual(len(result.checkpoints), 1)
            self.assertTrue(result.checkpoints[0]["complete"])
            self.assertEqual(len(result.wire_records), 7)
            for index, wire in enumerate(result.wire_records):
                AUTH_TEST.CONTROLLED_TEST._assert_wire_shape(self, wire)
                if index < 6:
                    self.assertTrue(wire["complete"], index)
                    self.assertEqual(len(wire["payload"]), WIRE.size, index)
            self.assertEqual(
                result.wire_records[6],
                {
                    "direction": "receive",
                    "payload": b"",
                    "complete": False,
                    "descriptor_count": 0,
                },
            )
            self.assertEqual(
                [
                    (
                        index,
                        WIRE.unpack(wire["payload"])[1],
                        WIRE.unpack(wire["payload"])[4],
                    )
                    for index, wire in enumerate(result.wire_records)
                    if wire["direction"] == "send"
                ],
                [
                    (1, MODEL_ACK, 0),
                    (3, READY_ACK, 1),
                    (5, RESULT_ACK, 1),
                ],
            )
            self.assertEqual(run["probe"].callback_calls[0][0], "prepare")
            self.assertEqual(
                [name for name, _pid in run["probe"].callback_calls[1:]],
                ["before", "after"],
            )
            metadata = AUTH_TEST._assert_cleanup(
                self, result, run["metadata_path"], run["token"]
            )
            self.assertEqual(metadata["milestones"], [1])
            self.assertEqual(result.model_evidence.identity, run["model_artifact"].identity)

    def test_resident_final_result_ack_is_refused_after_artifact_drift(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)

        def drift_after_final_capture(sequence: int, model: Path, _fake: Path) -> None:
            if sequence == 12:
                mutated = (b"resident model drift after capture\0" * 29)[: model.stat().st_size]
                with model.open("r+b") as handle:
                    handle.seek(0)
                    handle.write(mutated)
                    handle.flush()
                    os.fsync(handle.fileno())

        with _authenticated_run(
            records,
            expected,
            "resident",
            after_hook=drift_after_final_capture,
        ) as run:
            result = run["result"]
            AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(self, result)
            self.assertEqual(result.transport.reason, "protocol_error")
            self.assertTrue(result.control_error)
            self.assertEqual(len(result.transport.records), 11)
            for record in result.transport.records:
                validate_resident_record(record)
            self.assertEqual(len(result.checkpoints), 12)
            self.assertTrue(all(checkpoint["complete"] for checkpoint in result.checkpoints[:11]))
            self.assertFalse(result.checkpoints[-1]["complete"])
            self.assertEqual(len(result.wire_records), 49)
            last = result.wire_records[-1]
            self.assertEqual(last["direction"], "receive")
            values = WIRE.unpack(last["payload"])
            self.assertEqual(values[1], RESULT)
            self.assertEqual(values[4], 12)
            self.assertFalse(
                any(
                    wire["direction"] == "send"
                    and WIRE.unpack(wire["payload"])[1] == RESULT_ACK
                    and WIRE.unpack(wire["payload"])[4] == 12
                    for wire in result.wire_records
                )
            )
            self.assertIsNotNone(result.model_evidence)
            AUTH_TEST._assert_cleanup(
                self, result, run["metadata_path"], run["token"]
            )
            with self.assertRaises(ValueError):
                _ = run["model_artifact"].fd
            run["executable"].verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
