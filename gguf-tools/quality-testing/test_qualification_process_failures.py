#!/usr/bin/env python3
"""Host-only fault and interrupt tests for the Task-20 child transport.

The generated child fakes emit only qualification JSONL and metadata.  These
checks exercise process ownership, bounded cleanup, and stream error paths;
they never launch DS4, contact a model, use a GPU, network, or service.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import unittest
from typing import Any
from unittest import mock

import qualification_process as PROCESS
from test_qualification_process import (
    EVENTS,
    POSIX_PROCESS_GROUPS,
    _assert_group_gone,
    _assert_metadata,
    _assert_pid_gone,
    _assert_valid_prefix,
    _call_bounded,
    _fake_case,
    _WatchdogExpired,
)


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "Task-20 transport tests require POSIX process groups",
)
class QualificationProcessFailureHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Reuse the canonical lifecycle fixture and binding metadata from the
        # ordinary process-transport contract tests.
        from test_qualification_process import _expected, _lifecycle_records

        cls.records = _lifecycle_records()
        cls.expected = _expected(cls.records)

    def test_keyboard_interrupt_from_feed_rethrows_after_owned_cleanup(self) -> None:
        """A monitor BaseException must survive cleanup of exactly one group."""

        with _fake_case(
            "first_token_timeout", self.records
        ) as (command, metadata_path, token):
            injected = KeyboardInterrupt("one injected monitor interrupt")
            original_feed = PROCESS.QualificationSliceMonitor.feed
            feed_calls = 0
            validated: list[dict[str, Any]] = []

            def interrupting_feed(
                monitor: PROCESS.QualificationSliceMonitor,
                chunk: bytes,
                *,
                now_ns: int,
            ) -> None:
                nonlocal feed_calls
                feed_calls += 1
                original_feed(monitor, chunk, now_ns=now_ns)
                validated.extend(monitor.records)
                if feed_calls == 1:
                    raise injected
                raise AssertionError("monitor.feed was called after the injected interrupt")

            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_timer = signal.getitimer(signal.ITIMER_REAL)
            with mock.patch.object(
                PROCESS.QualificationSliceMonitor,
                "feed",
                new=interrupting_feed,
            ):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    _call_bounded(
                        command,
                        self.expected,
                        first_token_timeout_ns=500_000_000,
                        whole_request_timeout_ns=1_000_000_000,
                        idle_timeout_ns=500_000_000,
                        termination_grace_ns=30_000_000,
                    )

            metadata = _read_owned_metadata(metadata_path, token, self)
            self.assertIs(raised.exception, injected)
            self.assertEqual(feed_calls, 1)
            self.assertEqual(
                tuple(record.get("event") for record in validated), EVENTS[:1]
            )
            # _call_bounded is the main-thread watchdog.  An injected
            # BaseException must not leave its process-global signal state
            # armed or replaced.
            self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), previous_timer)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_watchdog_interrupts_term_grace_then_owned_group_is_gone(self) -> None:
        """SIGALRM during a long TERM grace still reaches KILL/reap cleanup."""

        with _fake_case(
            "descendant", self.records
        ) as (command, metadata_path, token):
            with self.assertRaises(_WatchdogExpired):
                _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=900_000_000,
                    whole_request_timeout_ns=1_800_000_000,
                    idle_timeout_ns=900_000_000,
                    # Deliberately exceeds the two-second native watchdog.
                    termination_grace_ns=2_500_000_000,
                )

            metadata = _read_owned_metadata(metadata_path, token, self)
            descendant_pid = metadata.get("descendant_pid")
            self.assertIs(type(descendant_pid), int)
            self.assertGreater(descendant_pid, 1)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_pid_gone(self, descendant_pid)
            _assert_group_gone(self, metadata["pgid"])
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_permission_denied_term_and_kill_do_not_block_complete_release(self) -> None:
        """EPERM for TERM/KILL is harmless once the complete child is reaped."""

        with _fake_case("complete", self.records) as (
            command,
            metadata_path,
            token,
        ):
            real_killpg = PROCESS.os.killpg
            calls: list[tuple[int, int]] = []

            def deny_positive_signals(pgid: int, signum: int) -> None:
                calls.append((pgid, signum))
                if signum in (signal.SIGTERM, signal.SIGKILL):
                    raise PermissionError(errno.EPERM, "denied test signal")
                # Signal 0 remains the real release/group-existence probe.
                real_killpg(pgid, signum)

            with mock.patch.object(
                PROCESS.os, "killpg", side_effect=deny_positive_signals
            ):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=500_000_000,
                    whole_request_timeout_ns=1_000_000_000,
                    idle_timeout_ns=500_000_000,
                    termination_grace_ns=30_000_000,
                )
                metadata = _read_owned_metadata(metadata_path, token, self)
                _assert_valid_prefix(self, result, EVENTS, self.expected, metadata)
                self.assertEqual(result.reason, "complete")
                self.assertEqual(result.returncode, 0)
                self.assertTrue(result.cleanup_complete)
                self.assertTrue(any(signum == signal.SIGTERM for _, signum in calls))
                self.assertTrue(any(signum == signal.SIGKILL for _, signum in calls))
                self.assertTrue(any(signum == 0 for _, signum in calls))
                _assert_pid_gone(self, metadata["leader_pid"])
                _assert_group_gone(self, metadata["pgid"])

    def test_select_error_after_validated_prefix_keeps_evidence_and_closes_streams(self) -> None:
        """A select fault keeps evidence and reports released process resources."""

        with _fake_case(
            "first_token_timeout", self.records
        ) as (command, metadata_path, token):
            real_select = PROCESS.select.select
            real_feed = PROCESS.QualificationSliceMonitor.feed
            real_popen = PROCESS.subprocess.Popen
            validated: list[dict[str, Any]] = []
            select_errors: list[BaseException] = []
            captured_processes: list[Any] = []
            injected = OSError(errno.EIO, "one injected select failure")

            def track_feed(
                monitor: PROCESS.QualificationSliceMonitor,
                chunk: bytes,
                *,
                now_ns: int,
            ) -> None:
                real_feed(monitor, chunk, now_ns=now_ns)
                validated.extend(monitor.records)

            def fail_once_after_prefix(
                read_fds: list[int],
                write_fds: list[int],
                error_fds: list[int],
                timeout: float | None = None,
            ) -> Any:
                if validated and not select_errors:
                    select_errors.append(injected)
                    raise injected
                return real_select(read_fds, write_fds, error_fds, timeout)

            def capture_popen(*args: Any, **kwargs: Any) -> Any:
                process = real_popen(*args, **kwargs)
                captured_processes.append(process)
                return process

            with (
                mock.patch.object(
                    PROCESS.QualificationSliceMonitor,
                    "feed",
                    new=track_feed,
                ),
                mock.patch.object(
                    PROCESS.select,
                    "select",
                    new=fail_once_after_prefix,
                ),
                mock.patch.object(
                    PROCESS.subprocess,
                    "Popen",
                    new=capture_popen,
                ),
            ):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=500_000_000,
                    whole_request_timeout_ns=1_000_000_000,
                    idle_timeout_ns=500_000_000,
                    termination_grace_ns=30_000_000,
                )

            metadata = _read_owned_metadata(metadata_path, token, self)
            _assert_valid_prefix(self, result, EVENTS[:1], self.expected, metadata)
            self.assertEqual(result.reason, "io_error")
            self.assertEqual(
                tuple(json.loads(line) for line in result.stdout.splitlines()),
                result.records,
            )
            self.assertTrue(result.stdout.endswith(b"\n"))
            self.assertEqual(len(select_errors), 1)
            self.assertIs(select_errors[0], injected)
            self.assertEqual(len(captured_processes), 1)
            self.assertTrue(captured_processes[0].stdout.closed)
            self.assertTrue(captured_processes[0].stderr.closed)
            # EOF is required for a complete protocol result, but an I/O
            # failure can still release every owned process resource.
            self.assertTrue(result.cleanup_complete)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_wait_ownership_loss_never_signals_and_metadata_fallback_cleans(self) -> None:
        """Lost wait ownership forbids signals; the fixture fallback owns cleanup."""

        with _fake_case("descendant", self.records) as (
            command,
            metadata_path,
            token,
        ):
            real_killpg = PROCESS.os.killpg
            calls: list[tuple[int, int]] = []

            def record_group_probe(pgid: int, signum: int) -> None:
                calls.append((pgid, signum))
                # Keep signal 0 real so the test observes the transport's
                # release probe without granting it any positive signal.
                if signum == 0:
                    real_killpg(pgid, signum)

            def lost_wait_probe() -> Any:
                def exited(_pid: int) -> bool:
                    raise ChildProcessError(errno.ECHILD, "wait ownership lost")

                return exited

            with (
                mock.patch.object(PROCESS, "_exit_probe", new=lost_wait_probe),
                mock.patch.object(
                    PROCESS.os, "killpg", side_effect=record_group_probe
                ),
            ):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=500_000_000,
                    whole_request_timeout_ns=1_000_000_000,
                    idle_timeout_ns=500_000_000,
                    termination_grace_ns=30_000_000,
                )
                metadata = _read_owned_metadata(metadata_path, token, self)
                _assert_metadata(self, metadata, token)
                descendant_pid = metadata.get("descendant_pid")
                self.assertIs(type(descendant_pid), int)
                self.assertGreater(descendant_pid, 1)
                self.assertEqual(result.reason, "io_error")
                self.assertIsNone(result.returncode)
                self.assertFalse(result.cleanup_complete)
                self.assertFalse(
                    any(signum in (signal.SIGTERM, signal.SIGKILL) for _, signum in calls)
                )
                leader_pid = metadata["leader_pid"]
                pgid = metadata["pgid"]

            # Both mocks are disabled before _fake_case's metadata-authenticated
            # exact-group fallback runs on context exit.  No broad PID scan is
            # needed or permitted here.

        _assert_pid_gone(self, leader_pid)
        _assert_pid_gone(self, descendant_pid)
        _assert_group_gone(self, pgid)


def _read_owned_metadata(
    path: Any, token: str, test: unittest.TestCase
) -> dict[str, Any]:
    """Reuse the existing bounded metadata reader without importing more helpers."""

    # The fixture's reader is intentionally private, but importing it here
    # keeps metadata reads bounded and authenticated exactly as in the parent
    # process tests.
    from test_qualification_process import _assert_metadata, _read_metadata

    metadata = _read_metadata(path)
    _assert_metadata(test, metadata, token)
    return metadata


if __name__ == "__main__":
    unittest.main(verbosity=2)
