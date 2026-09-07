#!/usr/bin/env python3
"""Bind one controlled slice to retained model and native executable identities.

The existing owner reserves the PID; live authentication runs while the child
is waiting for an ACK.  Optional input admission binds record validation to
retained schema snapshots.  This composition returns observations, not native
qualification, a retry decision or publication.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from compact_runtime_qualify import QualificationModelEvidence, _qualification_file_identity
from qualification_artifacts import QualificationArtifact, authenticate_running_executable
from qualification_controlled import (
    QualificationControlledResult,
    run_qualification_controlled_child,
)
from qualification_process import _validated_command
from qualification_records import _validate_input_admission
from qualification_supervisor import _validate_record_kind
from qualification_version_probe import _descriptor_exec_path


def run_authenticated_qualification_child(
    arguments: list[str] | tuple[str, ...], expected: Mapping[str, Any], *,
    record_kind: str = "streamed",
    input_admission: Any = None,
    executable_artifact: QualificationArtifact,
    model_artifact: QualificationArtifact,
    prepare_descriptor: Callable[[int, int, QualificationModelEvidence], Any],
    capture_before: Callable[[int, int], Any],
    capture_after: Callable[[int, int], Any],
    first_token_timeout_ns: int = 900_000_000_000,
    whole_request_timeout_ns: int = 2_700_000_000_000,
    idle_timeout_ns: int = 900_000_000_000,
    termination_grace_ns: int = 250_000_000,
    control_timeout_seconds: float = 30.0,
) -> QualificationControlledResult:
    """Execute the pinned descriptor and check identities around every callback.

    ``arguments`` is nonempty and excludes argv[0].  The child inherits exactly
    the executable FD and its private control socket.  The received model FD is
    compared to the already admitted model, never to an independently selected
    pathname.  Hashing happens on initial receipt; subsequent owner change-time
    and identity checks avoid rereading the model at every checkpoint.
    """
    _validate_record_kind(record_kind)
    input_admission = _validate_input_admission(input_admission, expected)
    if input_admission is not None:
        admitted = input_admission.artifacts
        if (executable_artifact is not admitted["bench"]
                or model_artifact is not admitted["model"]):
            raise ValueError("authenticated slice requires the exact retained bench/model owners")
    if type(arguments) not in (list, tuple):
        raise TypeError("authenticated child arguments must be a built-in list or tuple")
    if not arguments:
        raise ValueError("authenticated child arguments must not be empty")
    # Reuse the command boundary without treating a user argument as argv[0].
    checked_arguments = _validated_command(("retained-executable", *arguments))[1:]
    if any(arg == "--qualification-control-fd" or
           arg.startswith("--qualification-control-fd=") for arg in checked_arguments):
        raise ValueError("qualification control descriptor is owned by the runner")
    for name, callback in (("prepare_descriptor", prepare_descriptor),
                           ("capture_before", capture_before),
                           ("capture_after", capture_after)):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    for name, artifact in (("executable", executable_artifact), ("model", model_artifact)):
        if not isinstance(artifact, QualificationArtifact):
            raise TypeError(f"authenticated {name} requires a live artifact owner")
        artifact.verify()
    executable_fd = executable_artifact.fd
    if not os.fstat(executable_fd).st_mode & 0o111:
        raise ValueError("authenticated executable must have execute permission")
    command = (_descriptor_exec_path(executable_fd), *checked_arguments)
    received_fd: int | None = None
    received_evidence: QualificationModelEvidence | None = None

    def verify_live(pid: int) -> None:
        # The controller supplies its exclusively owned, WNOWAIT-reserved PID.
        # Its current executable must still be bound before any ACK can let
        # allocation or the next sample continue.  Auth verifies its owner too.
        authenticate_running_executable(pid, executable_artifact)
        model_artifact.verify()
        if input_admission is not None:
            input_admission.verify()
        if received_fd is None or received_evidence is None:
            raise ValueError("authenticated model descriptor was not received")
        if (received_evidence.identity != model_artifact.identity
                or received_evidence.sha256 != model_artifact.sha256
                or _qualification_file_identity(received_fd) != model_artifact.identity):
            raise ValueError("opened model descriptor does not match the admitted model")

    def prepare(pid: int, descriptor: int, evidence: QualificationModelEvidence) -> Any:
        nonlocal received_fd, received_evidence
        received_fd = descriptor
        received_evidence = evidence
        verify_live(pid)
        value = prepare_descriptor(pid, descriptor, evidence)
        verify_live(pid)
        return value

    def before(pid: int, sequence: int) -> Any:
        verify_live(pid)
        value = capture_before(pid, sequence)
        verify_live(pid)
        return value

    def after(pid: int, sequence: int) -> Any:
        verify_live(pid)
        value = capture_after(pid, sequence)
        verify_live(pid)
        return value

    result = run_qualification_controlled_child(
        command, expected, record_kind=record_kind, input_admission=input_admission,
        prepare_descriptor=prepare,
        capture_before=before, capture_after=after, _executable_fd=executable_fd,
        first_token_timeout_ns=first_token_timeout_ns,
        whole_request_timeout_ns=whole_request_timeout_ns,
        idle_timeout_ns=idle_timeout_ns, termination_grace_ns=termination_grace_ns,
        control_timeout_seconds=control_timeout_seconds,
    )
    # The live proc check belongs BEFORE the final ACK, not after reaping.
    # Also reject file drift during the child's exit tail without discarding
    # the successfully collected control/output prefix or fabricating a PID.
    if result.transport.reason == "complete":
        try:
            executable_artifact.verify()
            model_artifact.verify()
            if input_admission is not None:
                input_admission.verify()
        except (OSError, ValueError) as exc:
            result = replace(
                result,
                transport=replace(result.transport, reason="protocol_error"),
                control_error=f"qualification final artifact {type(exc).__name__}: {exc}"[:4096],
            )
    return result
