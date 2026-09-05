#!/usr/bin/env python3
"""Host-only deadline regressions for the Task-20 child transport."""

from __future__ import annotations

import select
import unittest
from pathlib import Path
from unittest import mock

from test_qualification_process import (
    EVENTS,
    POSIX_PROCESS_GROUPS,
    _assert_group_gone,
    _assert_metadata,
    _assert_pid_gone,
    _assert_result_shape,
    _call_bounded,
    _fake_case,
    _read_metadata,
)
from test_qualification_records import _expected, _lifecycle_records


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "Task-20 transport deadline tests require POSIX process groups",
)
class QualificationProcessDeadlineHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _lifecycle_records()
        cls.expected = _expected(cls.records)

    def test_late_nonzero_exit_cannot_mask_startup_timeout(self) -> None:
        """A delayed parent observation must retain the monitor's startup timeout."""

        # This uses the owned startup fake but changes only its branch: it
        # writes no stdout/stderr, waits 20ms, and exits with status seven.
        with _fake_case("startup_timeout", self.records) as (
            command,
            metadata_path,
            token,
        ):
            fake = Path(command[1])
            source = fake.read_text(encoding="utf-8")
            old_branch = (
                'if MODE == "startup_timeout":\n'
                "    time.sleep(3)\n"
                "    raise SystemExit(0)\n"
            )
            new_branch = (
                'if MODE == "startup_timeout":\n'
                "    time.sleep(0.02)\n"
                "    raise SystemExit(7)\n"
            )
            self.assertEqual(source.count(old_branch), 1)
            fake.write_text(source.replace(old_branch, new_branch), encoding="utf-8")

            # Keep the mock's lifetime inside _fake_case.  Its finally-based
            # exact-PGID fallback therefore runs only after this scope closes.
            real_select = select.select
            select_calls = 0

            def delayed_first_select(readable, writable, exceptional, timeout):
                nonlocal select_calls
                select_calls += 1
                if select_calls == 1:
                    # Delay only the first parent observation.  Use the real
                    # selector on empty lists, so no descriptor becomes ready.
                    return real_select([], [], [], 0.25)
                return real_select(readable, writable, exceptional, timeout)

            import qualification_process as PROCESS

            with mock.patch.object(PROCESS.select, "select", delayed_first_select):
                result = _call_bounded(
                    command,
                    self.expected,
                    first_token_timeout_ns=100_000_000,
                    whole_request_timeout_ns=400_000_000,
                    idle_timeout_ns=100_000_000,
                    termination_grace_ns=10_000_000,
                )
            self.assertGreaterEqual(select_calls, 1)

            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_result_shape(self, result)
            self.assertEqual(result.reason, "timeout")
            self.assertEqual(result.timeout_phase, "startup")
            self.assertIs(type(result.returncode), int)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(result.records, ())
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"")
            self.assertTrue(result.cleanup_complete)
            self.assertFalse(result.stdout_truncated)
            self.assertFalse(result.stderr_truncated)
            # These assertions run before _fake_case's finally fallback.
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])

    def test_cleanup_grace_does_not_become_producer_exit_timeout(self) -> None:
        """A successful producer remains complete even when grace exceeds idle."""

        with _fake_case("complete", self.records) as (
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
                termination_grace_ns=650_000_000,
            )
            metadata = _read_metadata(metadata_path)
            _assert_metadata(self, metadata, token)
            _assert_result_shape(self, result)
            self.assertEqual(result.reason, "complete")
            self.assertIs(type(result.returncode), int)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(result.records), 12)
            self.assertEqual(
                tuple(record.get("event") for record in result.records), EVENTS
            )
            self.assertIsNone(result.timeout_phase)
            self.assertTrue(result.cleanup_complete)
            self.assertFalse(result.stdout_truncated)
            self.assertFalse(result.stderr_truncated)
            _assert_pid_gone(self, metadata["leader_pid"])
            _assert_group_gone(self, metadata["pgid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
