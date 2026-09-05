#!/usr/bin/env python3
"""Host-only tests for the qualification slice lifecycle supervisor.

These tests use the checked-in parser fixture and explicit nanosecond values.
They do not launch a process, contact a model, use a GPU or service, or read
outside this test module's repository.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any

from qualification_records import QualificationRecordStream
from qualification_supervisor import QualificationSliceMonitor, QualificationTimeout
from test_qualification_records import (
    _expected,
    _json_line,
    _lifecycle_records,
    _payload,
)


EVENTS_PER_REQUEST = 3


def _timed_records(times: list[int]) -> list[dict[str, Any]]:
    """Return fixture-derived records with explicit monotonic timestamps."""

    records = copy.deepcopy(_lifecycle_records())
    if len(times) != len(records):
        raise AssertionError("the lifecycle test needs twelve timestamps")
    for record, timestamp in zip(records, times):
        record["monotonic_ns"] = str(timestamp)
    return records


def _monitor(
    records: list[dict[str, Any]],
    *,
    start_ns: int = 0,
    first_token_timeout_ns: int = 5,
    whole_request_timeout_ns: int = 20,
    idle_timeout_ns: int = 7,
) -> QualificationSliceMonitor:
    return QualificationSliceMonitor(
        _expected(records),
        start_ns=start_ns,
        first_token_timeout_ns=first_token_timeout_ns,
        whole_request_timeout_ns=whole_request_timeout_ns,
        idle_timeout_ns=idle_timeout_ns,
    )


def _feed_one(
    monitor: QualificationSliceMonitor,
    record: dict[str, Any],
    *,
    now_ns: int | None = None,
) -> None:
    timestamp = int(record["monotonic_ns"])
    monitor.feed(
        _json_line(record),
        now_ns=timestamp if now_ns is None else now_ns,
    )


def _feed_all(monitor: QualificationSliceMonitor, records: list[dict[str, Any]]) -> None:
    for record in records:
        _feed_one(monitor, record)


def _successful_monitor() -> tuple[QualificationSliceMonitor, list[dict[str, Any]]]:
    records = _timed_records(
        [10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27]
    )
    monitor = _monitor(
        records,
        first_token_timeout_ns=5,
        whole_request_timeout_ns=10,
        idle_timeout_ns=10,
    )
    _feed_all(monitor, records)
    return monitor, records


def _assert_timeout(
    test: unittest.TestCase,
    monitor: QualificationSliceMonitor,
    *,
    now_ns: int,
    phase: str,
    deadline_ns: int,
) -> QualificationTimeout:
    with test.assertRaises(QualificationTimeout) as raised:
        monitor.check_deadline(now_ns)
    error = raised.exception
    test.assertIsInstance(error, TimeoutError)
    test.assertEqual(error.phase, phase)
    test.assertEqual(error.deadline_ns, deadline_ns)
    test.assertEqual(error.now_ns, now_ns)
    return error


class QualificationSliceMonitorHostTest(unittest.TestCase):
    def test_defaultconstants(self) -> None:
        records = _timed_records([10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27])
        monitor = QualificationSliceMonitor(_expected(records), start_ns=0)

        self.assertEqual(monitor.deadline_ns, 30_000_000_000)
        _feed_one(monitor, records[0])
        self.assertEqual(monitor.deadline_ns, 10 + 900_000_000_000)
        _feed_one(monitor, records[1])
        self.assertEqual(monitor.deadline_ns, 10 + 2_700_000_000_000)

    def test_allphases_including_startup_interrequest_and_exit(self) -> None:
        cases = (
            ("startup", "startup", 108, 107),
            ("first_token", "first_token", 107, 106),
            ("request_complete", "request_complete", 122, 121),
            ("between_requests", "between_requests", 113, 112),
            ("exit", "exit", 132, 131),
        )
        for name, phase, now_ns, deadline_ns in cases:
            with self.subTest(phase=name):
                records = _timed_records(
                    [
                        101,
                        103,
                        105,
                        108,
                        110,
                        112,
                        115,
                        117,
                        119,
                        122,
                        123,
                        124,
                    ]
                )
                monitor = _monitor(records, start_ns=100)
                if phase == "first_token":
                    _feed_one(monitor, records[0])
                elif phase == "request_complete":
                    _feed_one(monitor, records[0])
                    _feed_one(monitor, records[1])
                elif phase == "between_requests":
                    _feed_all(monitor, records[:EVENTS_PER_REQUEST])
                elif phase == "exit":
                    _feed_all(monitor, records)
                _assert_timeout(
                    self,
                    monitor,
                    now_ns=now_ns,
                    phase=phase,
                    deadline_ns=deadline_ns,
                )
                with self.assertRaises(ValueError):
                    monitor.check_deadline(now_ns + 1)

    def test_wholedeadline_is_from_acceptance_and_not_reset_at_first_token(self) -> None:
        records = _timed_records([100, 105, 110, 115, 116, 117, 120, 121, 122, 125, 126, 127])
        monitor = _monitor(
            records,
            first_token_timeout_ns=10,
            whole_request_timeout_ns=20,
            idle_timeout_ns=100,
        )

        _feed_one(monitor, records[0])
        _feed_one(monitor, records[1])
        self.assertEqual(monitor.deadline_ns, 120)
        monitor.check_deadline(120)
        _assert_timeout(
            self,
            monitor,
            now_ns=121,
            phase="request_complete",
            deadline_ns=120,
        )

    def test_delayed_acceptance_cannot_extend_first_token_deadline(self) -> None:
        records = _timed_records([10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27])
        monitor = _monitor(
            records,
            first_token_timeout_ns=10,
            whole_request_timeout_ns=30,
            idle_timeout_ns=100,
        )

        _feed_one(monitor, records[0], now_ns=15)
        self.assertEqual(monitor.deadline_ns, 20)
        monitor.check_deadline(20)
        _assert_timeout(
            self,
            monitor,
            now_ns=21,
            phase="first_token",
            deadline_ns=20,
        )

    def test_coalesced_late_first_token_is_not_hidden_by_complete_record(self) -> None:
        records = _timed_records([10, 25, 26, 30, 31, 32, 35, 36, 37, 40, 41, 42])
        monitor = _monitor(
            records,
            first_token_timeout_ns=10,
            whole_request_timeout_ns=30,
            idle_timeout_ns=100,
        )

        with self.assertRaises(QualificationTimeout) as raised:
            monitor.feed(_payload(records[:3]), now_ns=26)
        error = raised.exception
        self.assertEqual(error.phase, "first_token")
        self.assertEqual(error.deadline_ns, 20)
        self.assertEqual(error.now_ns, 25)
        self.assertEqual(
            [record["event"] for record in monitor.records],
            ["request_accepted", "first_token", "request_complete"],
        )
        with self.assertRaises(ValueError):
            monitor.feed(_json_line(records[2]), now_ns=26)

    def test_late_parent_receipt_cannot_bypass_an_existing_phase_deadline(self) -> None:
        records = _timed_records([1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18])
        for count, now_ns, phase, deadline in (
            (0, 6, "startup", 5),
            (3, 9, "between_requests", 8),
        ):
            with self.subTest(phase=phase):
                monitor = _monitor(
                    records, first_token_timeout_ns=10,
                    whole_request_timeout_ns=20, idle_timeout_ns=5,
                )
                _feed_all(monitor, records[:count])
                with self.assertRaises(QualificationTimeout) as raised:
                    monitor.feed(_json_line(records[count]), now_ns=now_ns)
                self.assertEqual(raised.exception.phase, phase)
                self.assertEqual(raised.exception.deadline_ns, deadline)
                self.assertEqual(raised.exception.now_ns, now_ns)
                self.assertEqual(monitor.records, tuple(records[:count + 1]))
                with self.assertRaises(ValueError):
                    monitor.finish(0, now_ns=now_ns)

    def test_parent_receipt_is_checked_for_each_new_coalesced_phase(self) -> None:
        records = _timed_records([1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18])
        monitor = _monitor(
            records, first_token_timeout_ns=1,
            whole_request_timeout_ns=20, idle_timeout_ns=100,
        )
        with self.assertRaises(QualificationTimeout) as raised:
            monitor.feed(_payload(records[:3]), now_ns=3)
        self.assertEqual(raised.exception.phase, "first_token")
        self.assertEqual(raised.exception.deadline_ns, 2)
        self.assertEqual(raised.exception.now_ns, 3)
        self.assertEqual(monitor.records, tuple(records[:3]))
        with self.assertRaises(ValueError):
            monitor.finish(0, now_ns=3)

    def test_inclusivebounds_allow_equal_clock_and_event_deadlines(self) -> None:
        cases = ("startup", "first_token", "request_complete", "between_requests", "exit")
        for phase in cases:
            with self.subTest(phase=phase):
                if phase == "startup":
                    records = _timed_records([107, 108, 109, 112, 113, 114, 117, 118, 119, 122, 123, 124])
                    monitor = _monitor(records, start_ns=100)
                    _feed_one(monitor, records[0])
                    self.assertEqual(monitor.records[0]["monotonic_ns"], "107")
                elif phase == "first_token":
                    records = _timed_records([101, 106, 110, 113, 114, 115, 118, 119, 120, 123, 124, 125])
                    monitor = _monitor(records, start_ns=100)
                    _feed_one(monitor, records[0])
                    _feed_one(monitor, records[1])
                elif phase == "request_complete":
                    records = _timed_records([101, 102, 121, 124, 125, 126, 129, 130, 131, 134, 135, 136])
                    monitor = _monitor(records, start_ns=100)
                    _feed_one(monitor, records[0])
                    _feed_one(monitor, records[1])
                    _feed_one(monitor, records[2])
                elif phase == "between_requests":
                    records = _timed_records([101, 102, 103, 110, 111, 112, 115, 116, 117, 120, 121, 122])
                    monitor = _monitor(records, start_ns=100)
                    _feed_all(monitor, records[:4])
                else:
                    records = _timed_records([101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118])
                    monitor = _monitor(records, start_ns=100)
                    _feed_all(monitor, records)
                    monitor.check_deadline(125)
                self.assertGreaterEqual(
                    monitor.deadline_ns,
                    int(monitor.records[-1]["monotonic_ns"]),
                )

    def test_exact_four_repetition_finish_requires_twelve_records_and_exit_zero(self) -> None:
        monitor, records = _successful_monitor()

        result = monitor.finish(0, now_ns=27)
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 12)
        self.assertEqual(result, tuple(records))
        self.assertEqual(monitor.records, tuple(records))
        self.assertIsNone(monitor.deadline_ns)

    def test_invalid_clock_limits_future_and_prestart_timestamps(self) -> None:
        valid_records = _timed_records([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
        valid_expected = _expected(valid_records)
        invalid_starts = (True, -1, 0.0, "0", None)
        for value in invalid_starts:
            with self.subTest(argument="start_ns", value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    QualificationSliceMonitor(valid_expected, start_ns=value)

        for name, value in (
            ("first_token_timeout_ns", 0),
            ("first_token_timeout_ns", -1),
            ("first_token_timeout_ns", True),
            ("first_token_timeout_ns", 1.0),
            ("first_token_timeout_ns", "1"),
            ("whole_request_timeout_ns", 0),
            ("whole_request_timeout_ns", -1),
            ("whole_request_timeout_ns", True),
            ("whole_request_timeout_ns", 1.0),
            ("whole_request_timeout_ns", "1"),
            ("idle_timeout_ns", 0),
            ("idle_timeout_ns", -1),
            ("idle_timeout_ns", True),
            ("idle_timeout_ns", 1.0),
            ("idle_timeout_ns", "1"),
        ):
            with self.subTest(argument=name, value=repr(value)):
                kwargs = {
                    "start_ns": 0,
                    "first_token_timeout_ns": 5,
                    "whole_request_timeout_ns": 20,
                    "idle_timeout_ns": 7,
                }
                kwargs[name] = value
                with self.assertRaises((TypeError, ValueError)):
                    QualificationSliceMonitor(valid_expected, **kwargs)
        with self.assertRaises((TypeError, ValueError)):
            QualificationSliceMonitor(
                valid_expected,
                start_ns=0,
                first_token_timeout_ns=21,
                whole_request_timeout_ns=20,
                idle_timeout_ns=7,
            )

        for operation in ("feed", "check_deadline", "finish"):
            for value in (True, -1, 0.0, "0", None):
                with self.subTest(operation=operation, now_ns=repr(value)):
                    monitor = _monitor(valid_records)
                    with self.assertRaises((TypeError, ValueError)):
                        if operation == "feed":
                            monitor.feed(_json_line(valid_records[0]), now_ns=value)
                        elif operation == "check_deadline":
                            monitor.check_deadline(value)
                        else:
                            monitor.finish(0, now_ns=value)
                    with self.assertRaises((TypeError, ValueError)):
                        monitor.check_deadline(0)

        future = _timed_records([11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25])
        monitor = _monitor(future)
        with self.assertRaises(ValueError):
            monitor.feed(_json_line(future[0]), now_ns=10)
        with self.assertRaises(ValueError):
            monitor.feed(_json_line(future[0]), now_ns=11)

        prestart = _timed_records([9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23])
        monitor = _monitor(prestart, start_ns=10)
        with self.assertRaises(ValueError):
            monitor.feed(_json_line(prestart[0]), now_ns=10)
        with self.assertRaises(ValueError):
            monitor.check_deadline(10)

    def test_parent_clock_cannot_precede_start(self) -> None:
        records = _timed_records([101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118])
        for operation in ("feed", "check_deadline"):
            with self.subTest(operation=operation):
                monitor = _monitor(records, start_ns=100)
                with self.assertRaises(ValueError):
                    if operation == "feed":
                        monitor.feed(b"", now_ns=99)
                    else:
                        monitor.check_deadline(99)
                with self.assertRaises(ValueError):
                    monitor.check_deadline(100)

    def test_parent_clock_cannot_regress_after_a_valid_observation(self) -> None:
        records = _timed_records([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
        for operation in ("feed", "check_deadline", "finish"):
            with self.subTest(operation=operation):
                monitor = _monitor(records, idle_timeout_ns=100)
                _feed_one(monitor, records[0], now_ns=3)
                with self.assertRaises(ValueError):
                    if operation == "feed":
                        monitor.feed(_json_line(records[1]), now_ns=2)
                    elif operation == "check_deadline":
                        monitor.check_deadline(2)
                    else:
                        monitor.finish(0, now_ns=2)
                with self.assertRaises(ValueError):
                    monitor.check_deadline(3)

    def test_malformed_truncated_nonzero_and_closed_streams_are_sticky(self) -> None:
        records = _timed_records([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])

        with self.subTest(case="malformed"):
            monitor = _monitor(records, idle_timeout_ns=100)
            _feed_one(monitor, records[0])
            with self.assertRaises(ValueError):
                monitor.feed(b"[]\n", now_ns=1)
            self.assertEqual(len(monitor.records), 1)
            with self.assertRaises(ValueError):
                monitor.feed(_json_line(records[1]), now_ns=2)
            with self.assertRaises(ValueError):
                monitor.finish(0, now_ns=2)

        with self.subTest(case="truncated"):
            monitor = _monitor(records, idle_timeout_ns=100)
            line = _json_line(records[0])
            monitor.feed(line[:-1], now_ns=1)
            self.assertEqual(monitor.records, ())
            with self.assertRaises(ValueError):
                monitor.finish(0, now_ns=1)
            with self.assertRaises(ValueError):
                monitor.feed(line, now_ns=1)

        for count in (1, 3, 11):
            with self.subTest(case="incomplete_lifecycle", count=count):
                monitor = _monitor(records, idle_timeout_ns=100)
                _feed_all(monitor, records[:count])
                with self.assertRaises(ValueError):
                    monitor.finish(0, now_ns=int(records[count - 1]["monotonic_ns"]))
                with self.assertRaises(ValueError):
                    monitor.feed(b"", now_ns=int(records[count - 1]["monotonic_ns"]))

        for exit_code in (1, -1, True, False, 1.0, "0"):
            with self.subTest(case="exit_code", value=repr(exit_code)):
                monitor, _ = _successful_monitor()
                with self.assertRaises(ValueError):
                    monitor.finish(exit_code, now_ns=27)
                with self.assertRaises(ValueError):
                    monitor.finish(0, now_ns=27)
                with self.assertRaises(ValueError):
                    monitor.feed(b"", now_ns=27)

        with self.subTest(case="closed"):
            monitor, _ = _successful_monitor()
            monitor.finish(0, now_ns=27)
            self.assertIsNone(monitor.deadline_ns)
            with self.assertRaises(ValueError):
                monitor.feed(b"", now_ns=27)
            with self.assertRaises(ValueError):
                monitor.finish(0, now_ns=27)

    def test_partial_prefix_waits_for_lf_and_records_are_mutation_isolated(self) -> None:
        records = _timed_records([10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27])
        monitor = _monitor(records, idle_timeout_ns=100)
        line = _json_line(records[0])
        split = len(line) // 2
        monitor.feed(line[:split], now_ns=10)
        self.assertEqual(monitor.records, ())
        self.assertEqual(monitor.deadline_ns, 100)
        monitor.feed(line[split:], now_ns=10)
        self.assertEqual(len(monitor.records), 1)
        self.assertEqual(monitor.deadline_ns, 15)

        successful, expected_records = _successful_monitor()
        observed = successful.records
        self.assertIsNot(observed, successful.records)
        self.assertIsNot(observed[0], successful.records[0])
        observed[0]["runtime"]["model"]["id"] = "changed"
        observed[1]["runtime"]["config"]["context_tokens"] = 1
        self.assertEqual(successful.records, tuple(expected_records))
        self.assertEqual(successful.records[0]["runtime"]["model"]["id"], "laguna-s-2.1-界")
        self.assertEqual(successful.records[1]["runtime"]["config"]["context_tokens"], 32768)


if __name__ == "__main__":
    unittest.main(verbosity=2)
