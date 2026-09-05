#!/usr/bin/env python3
"""Pure lifecycle and deadline supervision for qualification records.

Parent and record timestamps must share one host monotonic-clock epoch, as
Linux CLOCK_MONOTONIC producers and time.monotonic_ns() do.  The idle guard
bounds startup, between-request, and exit waits; it is not a performance gate.
This module launches no processes and authenticates no files or live identities.
Even a complete observed lifecycle is not a qualification verdict.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from qualification_records import QualificationRecordStream


_DEFAULT_FIRST_TOKEN_TIMEOUT_NS = 900_000_000_000
_DEFAULT_WHOLE_REQUEST_TIMEOUT_NS = 2_700_000_000_000
_DEFAULT_IDLE_TIMEOUT_NS = 30_000_000_000


class QualificationTimeout(TimeoutError):
    """A qualification lifecycle phase exceeded its inclusive deadline."""

    def __init__(self, phase: str, deadline_ns: int, now_ns: int) -> None:
        self.phase = phase
        self.deadline_ns = deadline_ns
        self.now_ns = now_ns
        super().__init__(
            f"qualification {phase} deadline exceeded: "
            f"now_ns={now_ns} deadline_ns={deadline_ns}"
        )


class QualificationSliceMonitor:
    """Validate and supervise one four-request qualification lifecycle.

    The record stream owns wire, schema, binding, and lifecycle validation.
    This class only drains its validated observations and applies lifecycle
    deadlines to their monotonic timestamps and to the parent clock.
    """

    def __init__(
        self,
        expected: Mapping[str, Any],
        *,
        start_ns: int,
        first_token_timeout_ns: int = _DEFAULT_FIRST_TOKEN_TIMEOUT_NS,
        whole_request_timeout_ns: int = _DEFAULT_WHOLE_REQUEST_TIMEOUT_NS,
        idle_timeout_ns: int = _DEFAULT_IDLE_TIMEOUT_NS,
    ) -> None:
        self._require_nonnegative_int("start_ns", start_ns)
        self._require_positive_int(
            "first_token_timeout_ns", first_token_timeout_ns
        )
        self._require_positive_int(
            "whole_request_timeout_ns", whole_request_timeout_ns
        )
        self._require_positive_int("idle_timeout_ns", idle_timeout_ns)
        if first_token_timeout_ns > whole_request_timeout_ns:
            raise ValueError(
                "first_token_timeout_ns must not exceed "
                "whole_request_timeout_ns"
            )

        # QualificationRecordStream is the single owner of record validation.
        self._stream = QualificationRecordStream(expected)
        self._start_ns = start_ns
        self._first_token_timeout_ns = first_token_timeout_ns
        self._whole_request_timeout_ns = whole_request_timeout_ns
        self._idle_timeout_ns = idle_timeout_ns

        self._records: list[dict[str, Any]] = []
        self._phase = "startup"
        self._deadline_ns: int | None = start_ns + idle_timeout_ns
        self._acceptance_ns: int | None = None
        self._completed_requests = 0
        self._last_parent_now_ns: int = start_ns
        self._failed = False
        self._finished = False

    @staticmethod
    def _require_builtin_int(name: str, value: Any) -> None:
        if type(value) is not int:
            raise TypeError(f"{name} must be a built-in integer")

    @classmethod
    def _require_nonnegative_int(cls, name: str, value: Any) -> None:
        cls._require_builtin_int(name, value)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def _require_positive_int(cls, name: str, value: Any) -> None:
        cls._require_builtin_int(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _ensure_open(self) -> None:
        if self._failed:
            raise ValueError("qualification slice monitor is failed")
        if self._finished:
            raise ValueError("qualification slice monitor is finished")

    def _fail(self, exc: Exception) -> None:
        self._failed = True
        raise exc

    def _accept_parent_now(self, now_ns: Any) -> int:
        try:
            self._require_nonnegative_int("now_ns", now_ns)
        except (TypeError, ValueError) as exc:
            self._failed = True
            raise
        if now_ns < self._last_parent_now_ns:
            self._failed = True
            raise ValueError("parent now_ns regressed")
        self._last_parent_now_ns = now_ns
        return now_ns

    def _timeout(self, *, phase: str, deadline_ns: int, now_ns: int) -> None:
        self._failed = True
        raise QualificationTimeout(phase, deadline_ns, now_ns)

    def _check_deadline_validated(self, now_ns: int) -> None:
        deadline_ns = self._deadline_ns
        if deadline_ns is not None and now_ns > deadline_ns:
            self._timeout(
                phase=self._phase,
                deadline_ns=deadline_ns,
                now_ns=now_ns,
            )

    def _check_event_timestamp(self, timestamp_ns: int, parent_now_ns: int) -> None:
        if timestamp_ns < self._start_ns or timestamp_ns > parent_now_ns:
            self._fail(
                ValueError(
                    "record monotonic_ns must lie within "
                    "[start_ns, now_ns]"
                )
            )
        deadline_ns = self._deadline_ns
        if deadline_ns is not None and timestamp_ns > deadline_ns:
            self._timeout(
                phase=self._phase,
                deadline_ns=deadline_ns,
                now_ns=timestamp_ns,
            )

    def _advance(self, record: dict[str, Any], timestamp_ns: int) -> None:
        event = record["event"]
        if event == "request_accepted":
            self._acceptance_ns = timestamp_ns
            self._phase = "first_token"
            self._deadline_ns = timestamp_ns + self._first_token_timeout_ns
            return
        if event == "first_token":
            # Whole-request time is measured from acceptance, not first token.
            if self._acceptance_ns is None:
                self._fail(ValueError("first_token has no request acceptance"))
            self._phase = "request_complete"
            self._deadline_ns = (
                self._acceptance_ns + self._whole_request_timeout_ns
            )
            return
        if event == "request_complete":
            self._completed_requests += 1
            self._acceptance_ns = None
            self._phase = (
                "exit" if self._completed_requests == 4 else "between_requests"
            )
            self._deadline_ns = timestamp_ns + self._idle_timeout_ns
            return
        self._fail(ValueError(f"unknown qualification event {event!r}"))

    def _evaluate_records(
        self,
        records: tuple[dict[str, Any], ...],
        *,
        parent_now_ns: int,
    ) -> None:
        # Records are appended by the caller before this loop.  If a later
        # event fails timing, the entire parser-validated drain remains
        # available as diagnostic evidence.
        for record in records:
            timestamp_ns = int(record["monotonic_ns"])
            self._check_event_timestamp(timestamp_ns, parent_now_ns)
            self._advance(record, timestamp_ns)

    def _drain_and_evaluate(
        self,
        *,
        parent_now_ns: int,
        parser_error: Exception | None,
    ) -> None:
        drained = self._stream.drain_records()
        if drained:
            self._records.extend(drained)
        try:
            self._evaluate_records(drained, parent_now_ns=parent_now_ns)
        except (QualificationTimeout, TypeError, ValueError):
            self._failed = True
            raise
        if parser_error is not None:
            self._failed = True
            raise parser_error

    def feed(self, chunk: bytes, *, now_ns: int) -> None:
        """Feed bytes received at parent-clock time ``now_ns``."""

        self._ensure_open()
        parent_now_ns = self._accept_parent_now(now_ns)
        parser_error: Exception | None = None
        try:
            self._stream.feed(chunk)
        except (TypeError, ValueError, RecursionError) as exc:
            parser_error = exc
        self._drain_and_evaluate(
            parent_now_ns=parent_now_ns,
            parser_error=parser_error,
        )
        self._check_deadline_validated(parent_now_ns)

    def check_deadline(self, now_ns: int) -> None:
        """Check the active phase deadline against the parent clock."""

        self._ensure_open()
        parent_now_ns = self._accept_parent_now(now_ns)
        self._check_deadline_validated(parent_now_ns)

    def finish(self, exit_code: int, *, now_ns: int) -> tuple[dict[str, Any], ...]:
        """Close the stream only after four valid repetitions and exit zero."""

        self._ensure_open()
        parent_now_ns = self._accept_parent_now(now_ns)
        if type(exit_code) is not int or exit_code != 0:
            self._fail(ValueError("qualification exit_code must be built-in integer 0"))

        parser_error: Exception | None = None
        try:
            self._stream.finish()
        except (TypeError, ValueError, RecursionError) as exc:
            parser_error = exc
        self._drain_and_evaluate(
            parent_now_ns=parent_now_ns,
            parser_error=parser_error,
        )
        self._check_deadline_validated(parent_now_ns)
        self._finished = True
        self._deadline_ns = None
        return self.records

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return deep-copied parser-validated observations."""

        return tuple(copy.deepcopy(record) for record in self._records)

    @property
    def deadline_ns(self) -> int | None:
        """Return the active or diagnostic deadline."""

        return self._deadline_ns
