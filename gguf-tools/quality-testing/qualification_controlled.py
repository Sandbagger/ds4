#!/usr/bin/env python3
"""Drive one benchmark control channel while supervising its live output.

This fixed twelve-checkpoint session returns observations, not qualification
verdicts.  Manifest admission, executable/runtime authentication, gate policy,
retries and publication remain responsibilities of the enclosing runner.
Callbacks are synchronous; descriptor preparation runs before MODEL_FD_ACK.
"""

from __future__ import annotations

import copy
import math
import select
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from compact_runtime_qualify import QualificationControl, QualificationModelEvidence
from qualification_process import (
    QualificationChildResult,
    _qualification_child_transport,
    _validated_command,
)


@dataclass(frozen=True)
class QualificationControlledResult:
    transport: QualificationChildResult
    model_evidence: QualificationModelEvidence | None
    preparation: Any
    checkpoints: tuple[dict[str, Any], ...]
    control_error: str | None
    wire_records: tuple[dict[str, Any], ...]


class _TransportStopped(BaseException):
    """Leave a control wait without replacing a prior transport failure."""


def run_qualification_controlled_child(
    command: list[str] | tuple[str, ...], expected: Mapping[str, Any], *,
    prepare_descriptor: Callable[[int, int, QualificationModelEvidence], Any],
    capture_before: Callable[[int, int], Any],
    capture_after: Callable[[int, int], Any],
    first_token_timeout_ns: int = 900_000_000_000,
    whole_request_timeout_ns: int = 2_700_000_000_000,
    idle_timeout_ns: int = 900_000_000_000,
    termination_grace_ns: int = 250_000_000,
    control_timeout_seconds: float = 30.0,
) -> QualificationControlledResult:
    """Prepare the offered model, bracket twelve samples, and require EOF.

    The setup idle default matches the child's 15-minute model-ACK budget.
    First-token and whole-request limits still originate at acceptance records.
    Parent snapshots are copied at capture, before a caller can reuse them.
    """
    argv = _validated_command(command)
    if any(arg == "--qualification-control-fd" or
           arg.startswith("--qualification-control-fd=") for arg in argv):
        raise ValueError("qualification control descriptor is owned by the runner")
    for name, callback in (("prepare_descriptor", prepare_descriptor),
                           ("capture_before", capture_before),
                           ("capture_after", capture_after)):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")

    transport = None
    model_observation: QualificationModelEvidence | None = None
    preparation = None
    checkpoints: list[dict[str, Any]] = []
    control_error: str | None = None

    def wait_ready(endpoint: Any, write: bool, timeout_seconds: float) -> bool:
        # QualificationControl owns protocol parsing and its own ACK budget.
        # This waiter adds live pipe drainage and the producer's milestones.
        assert transport is not None
        until = time.monotonic_ns() + math.ceil(timeout_seconds * 1_000_000_000)
        while True:
            transport.read_ready(0)
            if transport.reason is not None:
                raise _TransportStopped(transport.reason)
            transport.observe_exit()
            transport.check_clock()
            if transport.reason is not None:
                raise _TransportStopped(transport.reason)
            now = time.monotonic_ns()
            remaining = until - now
            if remaining <= 0:
                return False
            milestone = transport.monitor.deadline_ns
            wait_ns = min(10_000_000, remaining)
            if transport.terminal_ns is None and milestone is not None:
                wait_ns = max(0, min(wait_ns, milestone - now))
            readers = list(transport.pipes)
            if not write:
                readers.append(endpoint)
            try:
                readable, writable, _ = select.select(
                    readers, [endpoint] if write else [], [],
                    wait_ns / 1_000_000_000,
                )
            except (OSError, ValueError):
                transport.fail("io_error")
                raise _TransportStopped("control selector failed") from None
            transport.read_ready(0)
            transport.observe_exit()
            transport.check_clock()
            if transport.reason is not None:
                raise _TransportStopped(transport.reason)
            if endpoint in (writable if write else readable):
                return True
            if transport.exit_observed:
                # Exit may have raced the select snapshot.  Permit queued
                # control bytes/EOF, but never wait on an exited leader.
                readable, writable, _ = select.select(
                    [] if write else [endpoint], [endpoint] if write else [], [], 0
                )
                if endpoint in (writable if write else readable):
                    return True
                raise ValueError("qualification child exited before control readiness")

    with QualificationControl.create(
        timeout_seconds=control_timeout_seconds, wait_ready=wait_ready,
    ) as control:
        inherited = control.child_fd
        launched_argv = argv + ("--qualification-control-fd", str(inherited))
        with _qualification_child_transport(
            launched_argv, expected,
            first_token_timeout_ns=first_token_timeout_ns,
            whole_request_timeout_ns=whole_request_timeout_ns,
            idle_timeout_ns=idle_timeout_ns,
            termination_grace_ns=termination_grace_ns,
            pass_fds=(inherited,),
        ) as transport:
            control.close_child_endpoint()
            if transport.process is not None and transport.reason is None:
                pid = transport.process.pid

                def prepare(fd: int, evidence: QualificationModelEvidence) -> None:
                    nonlocal model_observation, preparation
                    model_observation = evidence
                    preparation = copy.deepcopy(prepare_descriptor(pid, fd, evidence))

                try:
                    control.receive_model(prepare_descriptor=prepare)
                    # ds4-bench brackets acceptance/first-token/completion for
                    # each of its four consecutive cold/warm repetitions.
                    for sequence in range(1, 13):
                        sample = {"sequence": sequence, "before": None,
                                  "after": None, "complete": False}

                        def before() -> Any:
                            checkpoints.append(sample)
                            sample["before"] = copy.deepcopy(capture_before(pid, sequence))
                            return sample["before"]

                        def after() -> Any:
                            sample["after"] = copy.deepcopy(capture_after(pid, sequence))
                            return sample["after"]

                        control.bracket_sample(
                            sequence, capture_before=before, capture_after=after,
                            sample_timeout_seconds=whole_request_timeout_ns / 1_000_000_000,
                        )
                        sample["complete"] = True
                    control.verify_model_unchanged()
                    control.finish()
                    transport.run()
                except _TransportStopped as exc:
                    control_error = f"qualification control stopped: {exc}"[:4096]
                    control.close()
                except Exception as exc:
                    transport.fail("protocol_error")
                    control_error = f"qualification control {type(exc).__name__}: {exc}"[:4096]
                    control.close()
        child_result = transport.result()
        wire_records = control.wire_records
    return QualificationControlledResult(
        child_result, model_observation, preparation,
        copy.deepcopy(tuple(checkpoints)), control_error, wire_records,
    )
