#!/usr/bin/env python3
"""Host-only RED contract tests for the Task-20 child transport.

The child programs in this module are generated in a private temporary
folder.  They only emit qualification JSONL and metadata; they never launch
DS4, contact a model, use a GPU, network, or service.
"""

from __future__ import annotations

import contextlib
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

# Keep this import before creating any fake process.  The first RED run must
# fail here while qualification_process.py is not yet present.
import qualification_process as PROCESS
from qualification_process import run_qualification_child
from qualification_records import MAX_STREAM_BYTES
from test_qualification_records import _expected, _lifecycle_records


POSIX_PROCESS_GROUPS = (
    os.name == "posix" and hasattr(os, "fork") and hasattr(os, "killpg")
)
EVENTS = (
    "request_accepted",
    "first_token",
    "request_complete",
) * 4
RESULT_FIELDS = (
    "reason",
    "stdout",
    "stderr",
    "records",
    "returncode",
    "cleanup_complete",
    "timeout_phase",
    "stdout_truncated",
    "stderr_truncated",
)
FORBIDDEN_RESULT_FIELDS = (
    "passed",
    "failed",
    "qualification_status",
    "gates",
    "auth",
    "fd_control",
    "retries",
    "publication",
)


# The source is deliberately a generated Python script, rather than a shell
# fragment.  Every invocation writes its own exact PID/PGID metadata so the
# test's finally block can clean only the group it owns.
def _fake_source(
    mode: str,
    records: list[dict[str, Any]],
    metadata_path: Path,
    token: str,
) -> str:
    record_text = json.dumps(records, separators=(",", ":"), ensure_ascii=False)
    return f"""#!/usr/bin/env python3
import copy
import json
import os
import signal
import sys
import time

MODE = {mode!r}
TOKEN = {token!r}
METADATA = {str(metadata_path)!r}
RECORDS = json.loads({record_text!r})
PREVIOUS_MONOTONIC = 0


def _write_fd(fd, payload):
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except BrokenPipeError:
            return
        if written <= 0:
            return
        view = view[written:]


def _metadata(**extra):
    data = {{
        "schema": "ds4.qualification-process.fake/v1",
        "token": TOKEN,
        "parent_pid": os.getppid(),
        "leader_pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "started_monotonic_ns": time.monotonic_ns(),
    }}
    data.update(extra)
    with open(METADATA, "w", encoding="ascii") as handle:
        handle.write(json.dumps(data, separators=(",", ":")) + "\\n")


def _stamp(record):
    global PREVIOUS_MONOTONIC
    value = time.monotonic_ns()
    if value <= PREVIOUS_MONOTONIC:
        value = PREVIOUS_MONOTONIC + 1
    PREVIOUS_MONOTONIC = value
    record["monotonic_ns"] = str(value)


def _emit(index, *, newline=True):
    record = copy.deepcopy(RECORDS[index])
    # Refresh the observation clock immediately before each JSONL write.  The
    # resulting stamps are after process launch, unlike canned parser data.
    _stamp(record)
    encoded = json.dumps(
        record, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    _write_fd(1, encoded + (b"\\n" if newline else b""))


def _marker(value):
    _write_fd(2, value)


_metadata()
if MODE == "complete":
    # DEVNULL is part of the transport contract.  Reading one byte must see
    # EOF and must not wait for an inherited terminal or pipe.
    stdin_probe = sys.stdin.buffer.read(1)
    _metadata(stdin_eof=(stdin_probe == b""))
    for index in range(12):
        _emit(index)
    _marker(b"known-stderr-complete\\n")
    raise SystemExit(0)

if MODE == "nonzero":
    for index in range(3):
        _emit(index)
    _marker(b"known-stderr-partial\\n")
    raise SystemExit(7)

if MODE == "malformed":
    _emit(0)
    _write_fd(1, b"not-json\\n")
    raise SystemExit(0)

if MODE == "truncated":
    _emit(0)
    _emit(1, newline=False)
    raise SystemExit(0)

if MODE == "extra":
    for index in range(12):
        _emit(index)
    # A thirteenth line must be rejected as extra before it is fed as a
    # parser observation.
    _write_fd(1, json.dumps(RECORDS[0], separators=(",", ":")).encode("utf-8") + b"\\n")
    raise SystemExit(0)

if MODE == "stdout_limit":
    _write_fd(1, b"not-json\\n" + (b"S" * (1 << 20)))
    raise SystemExit(0)

if MODE == "stderr_limit":
    _write_fd(2, b"E" * (1 << 20))
    raise SystemExit(0)

if MODE == "startup_timeout":
    time.sleep(3)
    raise SystemExit(0)

if MODE == "first_token_timeout":
    _emit(0)
    time.sleep(3)
    raise SystemExit(0)

if MODE == "whole_request_timeout":
    _emit(0)
    _emit(1)
    time.sleep(3)
    raise SystemExit(0)

if MODE == "descendant":
    if not hasattr(os, "fork"):
        raise SystemExit(23)
    child_pid = os.fork()
    if child_pid == 0:
        # TERM is intentionally ignored.  The transport must escalate this
        # exact process group to KILL after the direct leader exits.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    _metadata(descendant_pid=child_pid)
    for index in range(12):
        _emit(index)
    _marker(b"known-stderr-descendant\\n")
    os._exit(0)

raise SystemExit(24)
"""


def _write_fake(
    directory: Path,
    mode: str,
    records: list[dict[str, Any]],
    metadata_path: Path,
    token: str,
) -> Path:
    fake = directory / "qualification_child_fake.py"
    fake.write_text(_fake_source(mode, records, metadata_path, token), encoding="utf-8")
    fake.chmod(0o700)
    return fake


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_METADATA_OVERFLOW = object()


def _load_bounded_metadata(path: Path) -> dict[str, Any] | object | None:
    """Load at most 4096 bytes, reading one extra byte only to detect overflow."""

    try:
        with path.open("rb") as handle:
            raw = handle.read(4_097)
    except (FileNotFoundError, OSError):
        return None
    if len(raw) > 4_096:
        return _METADATA_OVERFLOW
    try:
        value = json.loads(raw.decode("ascii"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None
    return value if isinstance(value, dict) else None


def _cleanup_owned_group(metadata_path: Path | None, token: str | None) -> None:
    """Best-effort fallback for exactly one metadata-authenticated process group."""

    if not POSIX_PROCESS_GROUPS or metadata_path is None or token is None:
        return
    data = _load_bounded_metadata(metadata_path)
    if data is _METADATA_OVERFLOW or not isinstance(data, dict):
        return
    if data.get("schema") != "ds4.qualification-process.fake/v1":
        return
    if data.get("token") != token:
        return
    parent_pid = data.get("parent_pid")
    leader_pid = data.get("leader_pid")
    pgid = data.get("pgid")
    sid = data.get("sid")
    if type(parent_pid) is not int or parent_pid <= 0 or parent_pid != os.getpid():
        return
    if any(type(value) is not int or value <= 1 for value in (leader_pid, pgid, sid)):
        return
    if not (sid == pgid == leader_pid):
        return
    if pgid == os.getpgrp():
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 0.25
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.005)
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # The leader is our exact known direct child.  Reap it if a broken
    # implementation returned before wait(), without probing unrelated PIDs.
    deadline = time.monotonic() + 0.5
    while _pid_exists(leader_pid) and time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(leader_pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            break
        if waited == leader_pid:
            break
        time.sleep(0.005)
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.005)


@contextlib.contextmanager
def _fake_case(
    mode: str,
    records: list[dict[str, Any]],
) -> Iterator[tuple[list[str], Path, str]]:
    """Yield one fake command and always clean its exact owned group."""

    with tempfile.TemporaryDirectory(prefix="qualification-process-") as name:
        directory = Path(name)
        metadata = directory / "owned-process.json"
        token = uuid.uuid4().hex
        fake = _write_fake(directory, mode, records, metadata, token)
        command = [sys.executable, str(fake)]
        try:
            yield command, metadata, token
        finally:
            _cleanup_owned_group(metadata, token)


@contextlib.contextmanager
def _no_child_cleanup_scope() -> Iterator[None]:
    """Keep invalid-argument tests explicit about their no-child cleanup path."""

    try:
        yield
    finally:
        # No metadata means there is no signalable child or process group.
        _cleanup_owned_group(None, None)


def _read_metadata(path: Path, timeout: float = 1.0) -> dict[str, Any]:
    """Read only the owned metadata prefix, with one byte for overflow detection."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _load_bounded_metadata(path)
        if value is _METADATA_OVERFLOW:
            raise AssertionError("owned fake metadata exceeds 4096 bytes")
        if isinstance(value, dict):
            return value
        time.sleep(0.005)
    raise AssertionError(f"fake metadata was not written: {path}")


class _WatchdogExpired(BaseException):
    """Raised by the main-thread SIGALRM watchdog around one child call."""


def _call_bounded(
    command: list[str] | tuple[str, ...],
    expected: dict[str, Any],
    **kwargs: int,
) -> Any:
    """Run the API behind a main-thread POSIX watchdog."""

    def _watchdog(_signum: int, _frame: Any) -> None:
        raise _WatchdogExpired("run_qualification_child exceeded the 2-second watchdog")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        return run_qualification_child(command, expected, **kwargs)
    finally:
        # Restore both pieces of process-global signal state even when the
        # child transport or this watchdog raises.
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _assert_group_gone(test: unittest.TestCase, pgid: int) -> None:
    deadline = time.monotonic() + 1.0
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    test.assertFalse(_group_exists(pgid), f"owned process group {pgid} survived")


def _assert_pid_gone(test: unittest.TestCase, pid: int) -> None:
    deadline = time.monotonic() + 1.0
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    test.assertFalse(_pid_exists(pid), f"owned direct child {pid} survived or was not reaped")


def _assert_result_shape(test: unittest.TestCase, result: Any) -> None:
    for field in RESULT_FIELDS:
        test.assertTrue(hasattr(result, field), f"result missing {field}")
    for field in FORBIDDEN_RESULT_FIELDS:
        test.assertFalse(hasattr(result, field), f"result leaked forbidden field {field}")
    test.assertIsInstance(result.stdout, bytes)
    test.assertIsInstance(result.stderr, bytes)
    test.assertIs(type(result.records), tuple)
    test.assertIs(type(result.cleanup_complete), bool)
    test.assertIs(type(result.stdout_truncated), bool)
    test.assertIs(type(result.stderr_truncated), bool)


def _assert_metadata(test: unittest.TestCase, metadata: dict[str, Any], token: str) -> None:
    test.assertEqual(metadata.get("schema"), "ds4.qualification-process.fake/v1")
    test.assertEqual(metadata.get("token"), token)
    test.assertIs(type(metadata.get("parent_pid")), int)
    test.assertEqual(metadata["parent_pid"], os.getpid())
    test.assertIs(type(metadata.get("leader_pid")), int)
    test.assertIs(type(metadata.get("pgid")), int)
    test.assertIs(type(metadata.get("sid")), int)
    test.assertGreater(metadata["parent_pid"], 0)
    test.assertGreater(metadata["leader_pid"], 0)
    test.assertGreater(metadata["pgid"], 0)
    test.assertGreater(metadata["sid"], 0)
    test.assertEqual(metadata["sid"], metadata["pgid"])
    test.assertEqual(metadata["pgid"], metadata["leader_pid"])
    # This proves start_new_session=True without ever touching the parent
    # process group in cleanup.
    test.assertNotEqual(metadata["pgid"], os.getpgrp())
    test.assertIs(type(metadata.get("started_monotonic_ns")), int)


def _assert_valid_prefix(
    test: unittest.TestCase,
    result: Any,
    expected_events: tuple[str, ...],
    expected_bindings: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    _assert_result_shape(test, result)
    test.assertEqual(
        tuple(record.get("event") for record in result.records), expected_events
    )
    previous = -1
    for record in result.records:
        test.assertIs(type(record), dict)
        for key, value in expected_bindings.items():
            test.assertEqual(record.get(key), value)
        timestamp = int(record["monotonic_ns"])
        test.assertGreater(timestamp, previous)
        test.assertGreaterEqual(timestamp, metadata["started_monotonic_ns"])
        previous = timestamp


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "Task-20 transport tests require POSIX process groups",
)
class QualificationProcessHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _lifecycle_records()
        cls.expected = _expected(cls.records)

    def test_complete_twelve_records_success_and_raw_output(self) -> None:
        with _fake_case("complete", self.records) as (command, metadata_path, token):
            result = _call_bounded(
                command,
                self.expected,
                first_token_timeout_ns=500_000_000,
                whole_request_timeout_ns=1_000_000_000,
                idle_timeout_ns=500_000_000,
                termination_grace_ns=30_000_000,
            )
            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_valid_prefix(self, result, EVENTS, self.expected, metadata)
            self.assertEqual(result.reason, "complete")
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.cleanup_complete)
            self.assertIsNone(result.timeout_phase)
            self.assertFalse(result.stdout_truncated)
            self.assertFalse(result.stderr_truncated)
            self.assertEqual(len(result.records), 12)
            decoded_stdout = tuple(
                json.loads(line) for line in result.stdout.splitlines()
            )
            self.assertEqual(decoded_stdout, result.records)
            self.assertTrue(result.stdout.endswith(b"\n"))
            self.assertEqual(result.stderr, b"known-stderr-complete\n")
            self.assertTrue(metadata.get("stdin_eof"))
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_nonzero_exit_keeps_partial_evidence_and_raw_streams(self) -> None:
        with _fake_case("nonzero", self.records) as (command, metadata_path, token):
            result = _call_bounded(
                command,
                self.expected,
                first_token_timeout_ns=500_000_000,
                whole_request_timeout_ns=1_000_000_000,
                idle_timeout_ns=500_000_000,
                termination_grace_ns=30_000_000,
            )
            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_valid_prefix(self, result, EVENTS[:3], self.expected, metadata)
            self.assertEqual(result.reason, "child_exit")
            self.assertEqual(result.returncode, 7)
            self.assertTrue(result.cleanup_complete)
            self.assertIsNone(result.timeout_phase)
            self.assertEqual(result.stderr, b"known-stderr-partial\n")
            self.assertFalse(result.stdout_truncated)
            self.assertFalse(result.stderr_truncated)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_malformed_truncated_and_extra_records_are_protocol_errors(self) -> None:
        cases = (
            ("malformed", EVENTS[:1]),
            ("truncated", EVENTS[:1]),
            ("extra", EVENTS),
        )
        for mode, prefix in cases:
            with self.subTest(mode=mode), _fake_case(mode, self.records) as (
                command,
                metadata_path,
                token,
            ):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=500_000_000,
                    whole_request_timeout_ns=1_000_000_000,
                    idle_timeout_ns=500_000_000,
                    termination_grace_ns=30_000_000,
                )
                metadata = _read_metadata(metadata_path)
                _assert_metadata(self, metadata, token)
                _assert_valid_prefix(self, result, prefix, self.expected, metadata)
                self.assertEqual(result.reason, "protocol_error")
                if result.cleanup_complete:
                    self.assertIs(type(result.returncode), int)
                self.assertTrue(result.cleanup_complete)
                self.assertIsNone(result.timeout_phase)
                self.assertFalse(result.stdout_truncated)
                self.assertFalse(result.stderr_truncated)
                if mode == "malformed":
                    self.assertIn(b"not-json\n", result.stdout)
                elif mode == "truncated":
                    self.assertFalse(result.stdout.endswith(b"\n"))
                else:
                    self.assertEqual(len(result.records), 12)
                _assert_pid_gone(self, metadata["leader_pid"])
                _assert_group_gone(self, metadata["pgid"])

    def test_stdout_and_stderr_ceilings_precede_parser_feed(self) -> None:
        self.assertEqual(PROCESS.MAX_STDOUT_BYTES, MAX_STREAM_BYTES)
        self.assertEqual(PROCESS.MAX_STDERR_BYTES, 65_536)
        for mode, stream_name in (("stdout_limit", "stdout"), ("stderr_limit", "stderr")):
            with self.subTest(stream=stream_name), mock.patch.object(
                PROCESS, "MAX_STDOUT_BYTES", 64
            ), mock.patch.object(PROCESS, "MAX_STDERR_BYTES", 48), _fake_case(
                mode, self.records
            ) as (command, metadata_path, token):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=500_000_000,
                    whole_request_timeout_ns=1_000_000_000,
                    idle_timeout_ns=500_000_000,
                    termination_grace_ns=30_000_000,
                )
                metadata = _read_metadata(metadata_path)
                _assert_metadata(self, metadata, token)
                _assert_result_shape(self, result)
                self.assertEqual(result.reason, "output_limit")
                self.assertEqual(result.records, ())
                self.assertTrue(result.cleanup_complete)
                self.assertIsNone(result.timeout_phase)
                self.assertLessEqual(len(result.stdout), 64)
                self.assertLessEqual(len(result.stderr), 48)
                if stream_name == "stdout":
                    self.assertTrue(result.stdout_truncated)
                    self.assertFalse(result.stderr_truncated)
                else:
                    self.assertFalse(result.stdout_truncated)
                    self.assertTrue(result.stderr_truncated)
                _assert_pid_gone(self, metadata["leader_pid"])
                _assert_group_gone(self, metadata["pgid"])

    def test_startup_timeout_is_bounded_and_reports_phase(self) -> None:
        with _fake_case("startup_timeout", self.records) as (command, metadata_path, token):
            result = _call_bounded(
                command,
                self.expected,
                first_token_timeout_ns=500_000_000,
                whole_request_timeout_ns=1_000_000_000,
                idle_timeout_ns=250_000_000,
                termination_grace_ns=30_000_000,
            )
            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_result_shape(self, result)
            self.assertEqual(result.reason, "timeout")
            self.assertEqual(result.timeout_phase, "startup")
            self.assertEqual(result.records, ())
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"")
            self.assertTrue(result.cleanup_complete)
            self.assertFalse(result.stdout_truncated)
            self.assertFalse(result.stderr_truncated)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_first_token_and_whole_request_timeouts_retain_records(self) -> None:
        cases = (
            (
                "first_token_timeout",
                "first_token",
                EVENTS[:1],
                {
                    "first_token_timeout_ns": 180_000_000,
                    "whole_request_timeout_ns": 900_000_000,
                    "idle_timeout_ns": 900_000_000,
                },
            ),
            (
                "whole_request_timeout",
                "request_complete",
                EVENTS[:2],
                {
                    "first_token_timeout_ns": 180_000_000,
                    "whole_request_timeout_ns": 350_000_000,
                    "idle_timeout_ns": 900_000_000,
                },
            ),
        )
        for mode, phase, prefix, timeouts in cases:
            with self.subTest(mode=mode), _fake_case(mode, self.records) as (
                command,
                metadata_path,
                token,
            ):
                result = _call_bounded(
                    command,
                    self.expected,
                    **timeouts,
                    termination_grace_ns=30_000_000,
                )
                metadata = _read_metadata(metadata_path)
                _assert_metadata(self, metadata, token)
                _assert_valid_prefix(self, result, prefix, self.expected, metadata)
                self.assertEqual(result.reason, "timeout")
                self.assertEqual(result.timeout_phase, phase)
                self.assertTrue(result.cleanup_complete)
                self.assertFalse(result.stdout_truncated)
                self.assertFalse(result.stderr_truncated)
                self.assertGreater(len(result.stdout), 0)
                _assert_pid_gone(self, metadata["leader_pid"])
                _assert_group_gone(self, metadata["pgid"])

    def test_exit_leader_with_descendant_held_pipes_reaps_child_and_group(self) -> None:
        with _fake_case("descendant", self.records) as (command, metadata_path, token):
            result = _call_bounded(
                command,
                self.expected,
                first_token_timeout_ns=900_000_000,
                whole_request_timeout_ns=1_800_000_000,
                idle_timeout_ns=900_000_000,
                termination_grace_ns=40_000_000,
            )
            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_valid_prefix(self, result, EVENTS, self.expected, metadata)
            self.assertEqual(result.reason, "complete")
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.cleanup_complete)
            self.assertEqual(result.stderr, b"known-stderr-descendant\n")
            descendant_pid = metadata.get("descendant_pid")
            self.assertIs(type(descendant_pid), int)
            self.assertGreater(descendant_pid, 1)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_pid_gone(self, descendant_pid)
            _assert_group_gone(self, metadata["pgid"])

    def test_bad_arguments_never_launch_and_exec_oserror_is_launch_error(self) -> None:
        valid = self.expected
        bad_calls: list[tuple[Any, Any, dict[str, Any], type[BaseException]]] = [
            (None, valid, {}, TypeError),
            ([], valid, {}, ValueError),
            ((), valid, {}, ValueError),
            ([sys.executable, 1], valid, {}, TypeError),
            ([""], valid, {}, ValueError),
            ([sys.executable, "bad\x00arg"], valid, {}, ValueError),
            ([sys.executable], None, {}, TypeError),
            ([sys.executable], {}, {}, ValueError),
            (
                [sys.executable],
                valid,
                {"first_token_timeout_ns": 0},
                ValueError,
            ),
            (
                [sys.executable],
                valid,
                {"whole_request_timeout_ns": -1},
                ValueError,
            ),
            (
                [sys.executable],
                valid,
                {"idle_timeout_ns": True},
                TypeError,
            ),
            (
                [sys.executable],
                valid,
                {"termination_grace_ns": 0},
                ValueError,
            ),
            (
                [sys.executable],
                valid,
                {
                    "first_token_timeout_ns": 5,
                    "whole_request_timeout_ns": 4,
                },
                ValueError,
            ),
        ]
        with _no_child_cleanup_scope(), mock.patch.object(PROCESS.subprocess, "Popen") as popen:
            for command, expected, kwargs, error_type in bad_calls:
                with self.subTest(command=repr(command), kwargs=kwargs):
                    with self.assertRaises(error_type):
                        _call_bounded(command, expected, **kwargs)
            popen.assert_not_called()

        with _no_child_cleanup_scope():
            result = _call_bounded(
                ["/definitely/not/a/qualification-child-executable"],
                valid,
                first_token_timeout_ns=100_000_000,
                whole_request_timeout_ns=200_000_000,
                idle_timeout_ns=100_000_000,
                termination_grace_ns=20_000_000,
            )
        _assert_result_shape(self, result)
        self.assertEqual(result.reason, "launch_error")
        self.assertIsNone(result.returncode)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.records, ())
        self.assertTrue(result.cleanup_complete)
        self.assertIsNone(result.timeout_phase)
        self.assertFalse(result.stdout_truncated)
        self.assertFalse(result.stderr_truncated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
