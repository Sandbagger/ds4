#!/usr/bin/env python3
"""Host-only tests for the Task-20 control-FD qualification runner."""

from __future__ import annotations

import array
import contextlib
import copy
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

# Keep this import before importing any existing fake/process helper.  The
# initial RED must fail here while qualification_controlled.py is absent,
# before a generated fake can start.
from qualification_controlled import run_qualification_controlled_child

import compact_runtime_qualify as COMPACT
import qualification_process as PROCESS
import test_compact_runtime_qualify as COMPACT_TEST
import test_qualification_process as PROCESS_TEST
from test_qualification_records import _expected, _lifecycle_records


POSIX_PROCESS_GROUPS = PROCESS_TEST.POSIX_PROCESS_GROUPS
EVENTS = PROCESS_TEST.EVENTS
WIRE = COMPACT._QUALIFICATION_CONTROL_MESSAGE
READY = COMPACT._QUALIFICATION_CONTROL_SAMPLE_READY
RESULT = COMPACT._QUALIFICATION_CONTROL_SAMPLE_RESULT
MODEL_ACK = COMPACT._QUALIFICATION_CONTROL_MODEL_FD_ACK


class _WatchdogExpired(BaseException):
    """Raised by the main-thread watchdog if a control wait wedges."""


def _call_bounded(
    command: list[str] | tuple[str, ...],
    expected: dict[str, Any],
    *,
    watchdog_seconds: float = 8.0,
    **kwargs: Any,
) -> Any:
    """Run one native child call behind a main-thread POSIX SIGALRM."""

    def _watchdog(_signum: int, _frame: Any) -> None:
        raise _WatchdogExpired(
            "run_qualification_controlled_child exceeded the test watchdog"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.setitimer(signal.ITIMER_REAL, watchdog_seconds)
    try:
        return run_qualification_controlled_child(command, expected, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _call_bounded_with_close_spy(
    close_descriptors: Any,
    command: list[str] | tuple[str, ...],
    expected: dict[str, Any],
    **kwargs: Any,
) -> Any:
    # Keep the instrumentation scoped to the native call.  Cleanup and all
    # fallback/diagnostic assertions run only after this mock is removed.
    with mock.patch.object(
        COMPACT.QualificationControl,
        "_close_descriptors",
        side_effect=close_descriptors,
    ):
        return _call_bounded(command, expected, **kwargs)


def _wire_message(
    message_type: int,
    sequence: int,
    identity: os.stat_result | tuple[int, int, int, int] | None = None,
) -> bytes:
    """Use the compact-runtime tests' canonical native wire encoding."""

    return COMPACT_TEST._qualification_control_wire_message(
        message_type, sequence, identity
    )


def _controlled_fake_source(
    mode: str,
    records: list[dict[str, Any]],
    metadata_path: Path,
    model_path: Path,
    token: str,
    *,
    stderr_burst: int = 0,
) -> str:
    """Generate a child implementing only the private control protocol."""

    config = repr(
        {
            "mode": mode,
            "records": records,
            "metadata": str(metadata_path),
            "model": str(model_path),
            "token": token,
            "stderr_burst": stderr_burst,
        }
    )
    return (
        "#!/usr/bin/env python3\n"
        "import array\n"
        "import copy\n"
        "import json\n"
        "import os\n"
        "import socket\n"
        "import struct\n"
        "import sys\n"
        "import time\n"
        f"CONFIG = {config}\n"
        r"""
WIRE = struct.Struct("@IIIIQQQQQ")
MODE = CONFIG["mode"]
RECORDS = json.loads(json.dumps(CONFIG["records"]))
METADATA = CONFIG["metadata"]
MODEL = CONFIG["model"]
TOKEN = CONFIG["token"]
STARTED_MONOTONIC_NS = time.monotonic_ns()
STATE = {"milestones": []}
PREVIOUS_MONOTONIC_NS = 0


def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise RuntimeError("short fake stream write")
        view = view[written:]


def _metadata(**extra):
    STATE.update(extra)
    data = {
        "schema": "ds4.qualification-process.fake/v1",
        "token": TOKEN,
        "parent_pid": os.getppid(),
        "leader_pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        # Metadata refreshes must never change this launch stamp.
        "started_monotonic_ns": STARTED_MONOTONIC_NS,
    }
    data.update(STATE)
    temporary = METADATA + ".next"
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(json.dumps(data, separators=(",", ":")) + "\n")
    os.replace(temporary, METADATA)


def _recv_exact(endpoint, size):
    payload = bytearray()
    while len(payload) < size:
        try:
            part = endpoint.recv(size - len(payload))
        except InterruptedError:
            continue
        if not part:
            raise EOFError("qualification control peer disconnected")
        payload.extend(part)
    return bytes(payload)


def _recv_message(endpoint):
    values = WIRE.unpack(_recv_exact(endpoint, WIRE.size))
    if values[0] != 1 or values[2] != WIRE.size or values[3] != 0:
        raise RuntimeError("invalid parent control message")
    return values


def _send_message(endpoint, message_type, sequence, identity=(0, 0, 0, 0)):
    payload = WIRE.pack(1, message_type, WIRE.size, 0, sequence, *identity)
    sent = endpoint.send(payload)
    if sent != len(payload):
        raise RuntimeError("short control send")


def _send_model(endpoint, descriptor, identity):
    payload = WIRE.pack(1, 1, WIRE.size, 0, 0, *identity)
    rights = array.array("i", [descriptor])
    sent = endpoint.sendmsg(
        [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
    )
    if sent != len(payload):
        raise RuntimeError("short model descriptor send")


def _send_extra_right(endpoint, descriptor):
    rights = array.array("i", [descriptor])
    sent = endpoint.sendmsg(
        [b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
    )
    if sent != 1:
        raise RuntimeError("short extra control send")


def _expect_ack(endpoint, message_type, sequence):
    values = _recv_message(endpoint)
    if values[1] != message_type or values[4] != sequence:
        raise RuntimeError("unexpected parent control acknowledgement")
    return values


def _emit_record(index):
    global PREVIOUS_MONOTONIC_NS
    value = time.monotonic_ns()
    if value <= PREVIOUS_MONOTONIC_NS:
        value = PREVIOUS_MONOTONIC_NS + 1
    PREVIOUS_MONOTONIC_NS = value
    record = copy.deepcopy(RECORDS[index])
    record["monotonic_ns"] = str(value)
    _write_all(
        1,
        json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        + b"\n",
    )


_metadata(phase="started", milestones=[])
control_fd_values = [
    int(value)
    for index, value in enumerate(sys.argv)
    if index and sys.argv[index - 1] == "--qualification-control-fd"
]
if len(control_fd_values) != 1:
    _metadata(phase="invalid_control_args", control_arg_count=len(control_fd_values))
    raise SystemExit(41)
control_fd = control_fd_values[0]
_metadata(phase="control_ready", control_fd=control_fd)
control = socket.socket(fileno=control_fd)
control.setblocking(True)
control.settimeout(5.0)
model_fd = -1
try:
    burst = int(CONFIG["stderr_burst"])
    if burst:
        _write_all(2, b"E" * burst)
    _metadata(phase="stderr_burst_written", stderr_burst_written=bool(burst))

    model_fd = os.open(MODEL, os.O_RDONLY)
    model_stat = os.fstat(model_fd)
    model_identity = (
        model_stat.st_dev,
        model_stat.st_ino,
        model_stat.st_size,
        model_stat.st_mtime_ns,
    )
    _send_model(control, model_fd, model_identity)
    _metadata(phase="model_offer_sent", model_offer_sent=True)
    try:
        ack = _expect_ack(control, 6, 0)
    except BaseException:
        _metadata(phase="model_ack_absent", model_ack=False, peer_closed=True)
        raise SystemExit(42)
    _metadata(phase="model_ack", model_ack=True, model_ack_values=list(ack))

    def checkpoint(sequence, index):
        _metadata(phase="ready_sent", ready_sent=sequence)
        _send_message(control, 2, sequence)
        _expect_ack(control, 3, sequence)
        _metadata(phase="ready_ack", ready_ack=sequence)
        _metadata(phase="result_sent", result_sent=sequence)
        _send_message(control, 4, sequence, model_identity)
        _expect_ack(control, 5, sequence)
        _emit_record(index)
        _metadata(phase="result_ack", result_ack=sequence)
        STATE["milestones"].append(sequence)
        _metadata(
            phase="milestone",
            milestone=sequence,
            milestones=list(STATE["milestones"]),
        )
        _write_all(2, ("milestone-%d\n" % sequence).encode("ascii"))

    if MODE == "stall":
        checkpoint(1, 0)
        _metadata(phase="stalled_before_ready2", stalled_before=2)
        time.sleep(5.0)
    else:
        for sequence in range(1, 13):
            checkpoint(sequence, sequence - 1)
        if MODE == "extra":
            _metadata(phase="extra_control_pending", extra_control_pending=True)
            _send_extra_right(control, model_fd)
            _metadata(phase="extra_control_sent", extra_control_sent=True)
            time.sleep(5.0)
finally:
    if model_fd >= 0:
        try:
            os.close(model_fd)
        except OSError:
            pass
"""
    )


@contextlib.contextmanager
def _controlled_fake_case(
    mode: str,
    records: list[dict[str, Any]],
    model_path: Path,
    *,
    stderr_burst: int = 0,
) -> Iterator[tuple[list[str], Path, str]]:
    """Yield a fake and clean only its authenticated process group."""

    with tempfile.TemporaryDirectory(prefix="qualification-controlled-") as name:
        directory = Path(name)
        metadata = directory / "owned-process.json"
        fake = directory / "qualification_controlled_fake.py"
        token = uuid.uuid4().hex
        fake.write_text(
            _controlled_fake_source(
                mode,
                records,
                metadata,
                model_path,
                token,
                stderr_burst=stderr_burst,
            ),
            encoding="utf-8",
        )
        fake.chmod(0o700)
        try:
            yield [sys.executable, str(fake)], metadata, token
        finally:
            PROCESS_TEST._cleanup_owned_group(metadata, token)


def _assert_controlled_shape(test: unittest.TestCase, result: Any) -> None:
    for field in (
        "transport", "model_evidence", "preparation", "checkpoints",
        "control_error", "wire_records",
    ):
        test.assertTrue(hasattr(result, field), f"missing controlled field {field}")
    for field in (
        "gate", "gates", "passed", "failed", "status", "authentication",
        "auth", "retry", "retries", "publication",
    ):
        test.assertFalse(hasattr(result, field), f"leaked verdict field {field}")
    test.assertIs(type(result.checkpoints), tuple)
    test.assertIs(type(result.wire_records), tuple)
    test.assertTrue(result.control_error is None or isinstance(result.control_error, str))


def _assert_wire_shape(test: unittest.TestCase, entry: Any) -> None:
    test.assertIs(type(entry), dict)
    test.assertEqual(set(entry), {"direction", "payload", "complete", "descriptor_count"})
    test.assertIn(entry["direction"], ("receive", "send"))
    test.assertIs(type(entry["payload"]), bytes)
    test.assertIs(type(entry["complete"]), bool)
    test.assertIs(type(entry["descriptor_count"]), int)
    test.assertGreaterEqual(entry["descriptor_count"], 0)


@unittest.skipUnless(POSIX_PROCESS_GROUPS, "controlled transport needs POSIX process groups")
class QualificationControlledHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _lifecycle_records()
        cls.expected = _expected(cls.records)

    def test_complete_handshake_and_twelve_brackets_drain_and_freeze(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-controlled-model-") as tmp:
            model = Path(tmp) / "tiny-model.gguf"
            payload = b"tiny regular model descriptor\0" * 257
            model.write_bytes(payload)
            with _controlled_fake_case(
                "complete", self.records, model, stderr_burst=128 << 10
            ) as (command, metadata_path, token):
                callback_log: list[tuple[Any, ...]] = []
                started_values: list[int] = []
                received_fds: list[int] = []
                preparation_refs: list[dict[str, Any]] = []
                before_refs: list[dict[str, Any]] = []
                after_refs: list[dict[str, Any]] = []
                before_snapshots: list[dict[str, Any]] = []
                after_snapshots: list[dict[str, Any]] = []

                def pid_fact(pid: int, sequence: int, phase: str) -> dict[str, Any]:
                    metadata = PROCESS_TEST._read_metadata(metadata_path)
                    started_values.append(metadata["started_monotonic_ns"])
                    return {
                        "pid": pid,
                        "sequence": sequence,
                        "phase": phase,
                        "sid": os.getsid(pid),
                        "pgid": os.getpgid(pid),
                    }

                def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                    callback_log.append(("prepare", pid))
                    received_fds.append(model_fd)
                    self.assertEqual(os.fstat(model_fd).st_ino, evidence.identity.inode)
                    metadata = PROCESS_TEST._read_metadata(metadata_path)
                    self.assertTrue(metadata.get("stderr_burst_written"))
                    self.assertFalse(metadata.get("model_ack", False))
                    fact = pid_fact(pid, 0, "prepare")
                    fact["sha256"] = evidence.sha256
                    fact["identity"] = evidence.identity.as_decimal_mapping()
                    preparation_refs.append(fact)
                    return fact

                def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                    callback_log.append(("before", pid, sequence))
                    metadata = PROCESS_TEST._read_metadata(metadata_path)
                    self.assertEqual(metadata.get("ready_sent"), sequence)
                    self.assertNotEqual(metadata.get("ready_ack"), sequence)
                    fact = pid_fact(pid, sequence, "before")
                    before_refs.append(fact)
                    before_snapshots.append(copy.deepcopy(fact))
                    return fact

                def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                    callback_log.append(("after", pid, sequence))
                    metadata = PROCESS_TEST._read_metadata(metadata_path)
                    self.assertEqual(metadata.get("result_sent"), sequence)
                    self.assertNotEqual(metadata.get("result_ack"), sequence)
                    fact = pid_fact(pid, sequence, "after")
                    after_refs.append(fact)
                    after_snapshots.append(copy.deepcopy(fact))
                    return fact

                with mock.patch.object(PROCESS, "MAX_STDERR_BYTES", 1 << 20):
                    result = _call_bounded(
                        command,
                        self.expected,
                        prepare_descriptor=prepare,
                        capture_before=capture_before,
                        capture_after=capture_after,
                        first_token_timeout_ns=2_000_000_000,
                        whole_request_timeout_ns=4_000_000_000,
                        idle_timeout_ns=2_000_000_000,
                        termination_grace_ns=100_000_000,
                        control_timeout_seconds=2.0,
                    )

                _assert_controlled_shape(self, result)
                metadata = PROCESS_TEST._read_metadata(metadata_path)
                PROCESS_TEST._assert_metadata(self, metadata, token)
                PROCESS_TEST._assert_valid_prefix(
                    self, result.transport, EVENTS, self.expected, metadata
                )
                self.assertEqual(result.transport.reason, "complete")
                self.assertEqual(result.transport.returncode, 0)
                self.assertTrue(result.transport.cleanup_complete)
                self.assertIsNone(result.transport.timeout_phase)
                self.assertFalse(result.transport.stderr_truncated)
                self.assertEqual(len(result.transport.records), 12)
                self.assertEqual(
                    tuple(json.loads(line) for line in result.transport.stdout.splitlines()),
                    result.transport.records,
                )
                burst = 128 << 10
                self.assertEqual(
                    result.transport.stderr,
                    b"E" * burst
                    + b"".join(
                        (f"milestone-{sequence}\n").encode("ascii")
                        for sequence in range(1, 13)
                    ),
                )
                pid = callback_log[0][1]
                self.assertEqual(
                    tuple(callback_log),
                    tuple(
                        [("prepare", pid)]
                        + [
                            item
                            for sequence in range(1, 13)
                            for item in (("before", pid, sequence), ("after", pid, sequence))
                        ]
                    ),
                )
                self.assertEqual(len(set(started_values)), 1)
                self.assertEqual(metadata["started_monotonic_ns"], started_values[0])
                self.assertEqual(metadata["milestones"], list(range(1, 13)))
                self.assertTrue(metadata.get("model_ack"))
                self.assertEqual(metadata["leader_pid"], pid)
                self.assertTrue(all(f["pid"] == pid for f in before_refs + after_refs))
                self.assertTrue(all(f["sid"] == f["pgid"] == pid for f in before_refs + after_refs))
                self.assertEqual(metadata["sid"], metadata["pgid"])
                self.assertEqual(metadata["pgid"], metadata["leader_pid"])

                evidence = result.model_evidence
                self.assertIsInstance(evidence, COMPACT.QualificationModelEvidence)
                self.assertEqual(evidence.sha256, hashlib.sha256(payload).hexdigest())
                stat_result = model.stat()
                self.assertEqual(evidence.identity.inode, stat_result.st_ino)
                self.assertEqual(evidence.identity.device, stat_result.st_dev)
                self.assertEqual(evidence.identity.size_bytes, stat_result.st_size)
                self.assertEqual(evidence.identity.mtime_ns, stat_result.st_mtime_ns)
                self.assertEqual(result.preparation, preparation_refs[0])
                for sequence, checkpoint in enumerate(result.checkpoints, 1):
                    self.assertEqual(
                        checkpoint,
                        {
                            "sequence": sequence,
                            "before": before_snapshots[sequence - 1],
                            "after": after_snapshots[sequence - 1],
                            "complete": True,
                        },
                    )
                self.assertEqual(len(result.checkpoints), 12)
                for fact in [preparation_refs[0], *before_refs, *after_refs]:
                    fact["mutated_after_return"] = True
                self.assertNotIn("mutated_after_return", result.preparation)
                self.assertNotIn("mutated_after_return", result.checkpoints[0]["before"])
                self.assertIsNone(result.control_error)

                self.assertEqual(len(result.wire_records), 50)
                for wire_record in result.wire_records:
                    _assert_wire_shape(self, wire_record)
                    self.assertTrue(wire_record["complete"])
                    self.assertEqual(len(wire_record["payload"]), WIRE.size)
                self.assertEqual(
                    sum(r["direction"] == "receive" for r in result.wire_records), 25
                )
                self.assertEqual(
                    sum(r["direction"] == "send" for r in result.wire_records), 25
                )
                self.assertEqual(result.wire_records[0]["descriptor_count"], 1)
                self.assertTrue(all(r["descriptor_count"] == 0 for r in result.wire_records[1:]))
                for fd in received_fds:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                PROCESS_TEST._assert_pid_gone(self, metadata["leader_pid"])
                PROCESS_TEST._assert_group_gone(self, metadata["pgid"])

    def test_prepare_failure_retains_model_evidence_and_closes_received_fd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-controlled-model-") as tmp:
            model = Path(tmp) / "tiny-model.gguf"
            payload = b"prepare failure model\0" * 37
            model.write_bytes(payload)
            with _controlled_fake_case("prepare_failure", self.records, model) as (
                command, metadata_path, token
            ):
                received_fds: list[int] = []
                callback_calls: list[str] = []

                def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                    del pid, evidence
                    callback_calls.append("prepare")
                    received_fds.append(model_fd)
                    self.assertEqual(os.fstat(model_fd).st_size, len(payload))
                    raise RuntimeError("prepare descriptor exploded")

                def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                    del pid, sequence
                    callback_calls.append("before")
                    return {"unexpected": "before"}

                def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                    del pid, sequence
                    callback_calls.append("after")
                    return {"unexpected": "after"}

                result = _call_bounded(
                    command,
                    self.expected,
                    prepare_descriptor=prepare,
                    capture_before=capture_before,
                    capture_after=capture_after,
                    watchdog_seconds=5.0,
                    first_token_timeout_ns=1_000_000_000,
                    whole_request_timeout_ns=2_000_000_000,
                    idle_timeout_ns=1_000_000_000,
                    termination_grace_ns=100_000_000,
                    control_timeout_seconds=1.0,
                )

                _assert_controlled_shape(self, result)
                metadata = PROCESS_TEST._read_metadata(metadata_path)
                PROCESS_TEST._assert_metadata(self, metadata, token)
                self.assertEqual(callback_calls, ["prepare"])
                self.assertEqual(result.transport.reason, "protocol_error")
                self.assertTrue(result.transport.cleanup_complete)
                self.assertIsNone(result.transport.timeout_phase)
                self.assertIsNone(result.preparation)
                self.assertEqual(result.checkpoints, ())
                self.assertIsInstance(result.control_error, str)
                self.assertLessEqual(len(result.control_error), 4096)
                self.assertIn("prepare", result.control_error.lower())
                self.assertIsInstance(result.model_evidence, COMPACT.QualificationModelEvidence)
                self.assertEqual(result.model_evidence.sha256, hashlib.sha256(payload).hexdigest())
                self.assertFalse(metadata.get("model_ack", False))
                self.assertIsNone(metadata.get("ready_sent"))
                self.assertEqual(metadata.get("milestones"), [])
                self.assertEqual(len(result.wire_records), 1)
                _assert_wire_shape(self, result.wire_records[0])
                self.assertEqual(result.wire_records[0]["direction"], "receive")
                self.assertEqual(result.wire_records[0]["descriptor_count"], 1)
                for fd in received_fds:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                PROCESS_TEST._assert_pid_gone(self, metadata["leader_pid"])
                PROCESS_TEST._assert_group_gone(self, metadata["pgid"])

    def test_control_wait_timeout_preserves_partial_transport_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-controlled-model-") as tmp:
            model = Path(tmp) / "tiny-model.gguf"
            model.write_bytes(b"stall after first checkpoint\0" * 31)
            with _controlled_fake_case("stall", self.records, model) as (
                command, metadata_path, token
            ):
                callback_log: list[tuple[str, int]] = []

                def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                    self.assertEqual(os.fstat(model_fd).st_ino, evidence.identity.inode)
                    callback_log.append(("prepare", pid))
                    return {"pid": pid, "phase": "prepared"}

                def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                    callback_log.append(("before", sequence))
                    return {"pid": pid, "sequence": sequence, "phase": "before"}

                def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                    callback_log.append(("after", sequence))
                    return {"pid": pid, "sequence": sequence, "phase": "after"}

                result = _call_bounded(
                    command,
                    self.expected,
                    prepare_descriptor=prepare,
                    capture_before=capture_before,
                    capture_after=capture_after,
                    watchdog_seconds=5.0,
                    first_token_timeout_ns=1_000_000_000,
                    whole_request_timeout_ns=3_000_000_000,
                    idle_timeout_ns=3_000_000_000,
                    termination_grace_ns=100_000_000,
                    control_timeout_seconds=3.0,
                )

                _assert_controlled_shape(self, result)
                metadata = PROCESS_TEST._read_metadata(metadata_path)
                PROCESS_TEST._assert_metadata(self, metadata, token)
                PROCESS_TEST._assert_valid_prefix(
                    self, result.transport, EVENTS[:1], self.expected, metadata
                )
                self.assertEqual(result.transport.reason, "timeout")
                self.assertEqual(result.transport.timeout_phase, "first_token")
                self.assertTrue(result.transport.cleanup_complete)
                self.assertEqual(len(result.transport.records), 1)
                self.assertEqual(
                    tuple(json.loads(line) for line in result.transport.stdout.splitlines()),
                    result.transport.records,
                )
                self.assertIn(b"milestone-1\n", result.transport.stderr)
                self.assertEqual(callback_log[1:], [("before", 1), ("after", 1)])
                self.assertEqual(len(result.checkpoints), 1)
                self.assertEqual(result.checkpoints[0]["sequence"], 1)
                self.assertTrue(result.checkpoints[0]["complete"])
                self.assertEqual(metadata.get("stalled_before"), 2)
                self.assertIsInstance(result.model_evidence, COMPACT.QualificationModelEvidence)
                self.assertLessEqual(len(result.wire_records), 50)
                for record in result.wire_records:
                    _assert_wire_shape(self, record)
                self.assertNotEqual(result.transport.reason, "protocol_error")
                PROCESS_TEST._assert_pid_gone(self, metadata["leader_pid"])
                PROCESS_TEST._assert_group_gone(self, metadata["pgid"])

    def test_extra_control_right_is_protocol_error_and_never_complete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qualification-controlled-model-") as tmp:
            model = Path(tmp) / "tiny-model.gguf"
            model.write_bytes(b"extra control descriptor model\0" * 29)
            with _controlled_fake_case("extra", self.records, model) as (
                command, metadata_path, token
            ):
                received_fds: list[int] = []
                closed_fds: list[int] = []
                original_close_descriptors = COMPACT.QualificationControl._close_descriptors

                def close_descriptors(descriptors: Any) -> None:
                    observed = list(descriptors)
                    closed_fds.extend(observed)
                    original_close_descriptors(observed)

                def prepare(pid: int, model_fd: int, evidence: Any) -> dict[str, Any]:
                    del evidence
                    received_fds.append(model_fd)
                    return {"pid": pid, "phase": "prepared"}

                def capture_before(pid: int, sequence: int) -> dict[str, Any]:
                    return {"pid": pid, "sequence": sequence, "phase": "before"}

                def capture_after(pid: int, sequence: int) -> dict[str, Any]:
                    return {"pid": pid, "sequence": sequence, "phase": "after"}

                result = _call_bounded_with_close_spy(
                    close_descriptors,
                    command,
                    self.expected,
                    prepare_descriptor=prepare,
                    capture_before=capture_before,
                    capture_after=capture_after,
                    watchdog_seconds=6.0,
                    first_token_timeout_ns=2_000_000_000,
                    whole_request_timeout_ns=4_000_000_000,
                    idle_timeout_ns=2_000_000_000,
                    termination_grace_ns=100_000_000,
                    control_timeout_seconds=1.0,
                )

                _assert_controlled_shape(self, result)
                metadata = PROCESS_TEST._read_metadata(metadata_path)
                PROCESS_TEST._assert_metadata(self, metadata, token)
                self.assertEqual(result.transport.reason, "protocol_error")
                self.assertTrue(result.transport.cleanup_complete)
                self.assertIsNone(result.transport.timeout_phase)
                self.assertEqual(len(result.transport.records), 12)
                self.assertEqual(len(result.checkpoints), 12)
                self.assertTrue(all(checkpoint["complete"] for checkpoint in result.checkpoints))
                self.assertIsInstance(result.control_error, str)
                self.assertLessEqual(len(result.control_error), 4096)
                self.assertIn("control", result.control_error.lower())
                self.assertNotEqual(result.transport.reason, "complete")
                # The captured wire byte proves the send.  TERM may precede
                # the fake's later metadata refresh after sendmsg returns.
                self.assertTrue(metadata.get("extra_control_pending"))
                self.assertEqual(metadata.get("milestones"), list(range(1, 13)))
                for record in result.wire_records:
                    _assert_wire_shape(self, record)
                self.assertGreaterEqual(len(result.wire_records), 50)
                extra = result.wire_records[50:]
                self.assertTrue(extra)
                self.assertEqual(extra[-1]["direction"], "receive")
                self.assertEqual(extra[-1]["payload"], b"x")
                self.assertEqual(extra[-1]["descriptor_count"], 1)
                model_fd_set = set(received_fds)
                unexpected_closed = [fd for fd in set(closed_fds) if fd not in model_fd_set]
                self.assertTrue(
                    unexpected_closed,
                    "the unexpected SCM_RIGHTS descriptor was not closed",
                )
                for fd in unexpected_closed:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                for fd in received_fds:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                PROCESS_TEST._assert_pid_gone(self, metadata["leader_pid"])
                PROCESS_TEST._assert_group_gone(self, metadata["pgid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
