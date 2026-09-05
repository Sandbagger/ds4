#!/usr/bin/env python3
"""Collect bounded version bytes by executing a retained Linux artifact FD.

This is an admission callback, not a version/schema or qualification verdict.
The caller retains the artifact owner.  A failed artifact verification closes
that owner; the probe itself never closes or transfers the borrowed FD.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from qualification_artifacts import QualificationArtifact
from qualification_process import _Transport, _owned_child_transport
from qualification_supervisor import QualificationTimeout

MAX_VERSION_STDOUT_BYTES = 65_536
MAX_VERSION_STDERR_BYTES = 65_536


def _positive_ns(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _descriptor_exec_path(descriptor: int) -> str:
    # /dev/fd is not an executable-file projection on the tested Darwin host.
    # Never replace this with a lookup of the mutable admitted pathname.
    if not sys.platform.startswith("linux"):
        raise ValueError("native descriptor execution requires Linux")
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("executable descriptor must be outside standard streams")
    return f"/proc/self/fd/{descriptor}"


class _VersionMonitor:
    """One inclusive deadline from launch admission through exit and pipe EOF.

    Progress bytes do not extend it.  JSON parsing belongs to input admission,
    not to the child transport or this raw-output clock.
    """

    def __init__(self, *, start_ns: int, timeout_ns: int) -> None:
        if type(start_ns) is not int or start_ns < 0:
            raise ValueError("version start_ns must be a nonnegative built-in integer")
        self.deadline_ns: int | None = start_ns + _positive_ns("timeout_ns", timeout_ns)
        self._last_now_ns = start_ns

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return ()

    def check_deadline(self, now_ns: int) -> None:
        if type(now_ns) is not int or now_ns < self._last_now_ns:
            raise ValueError("version parent clock is invalid or regressed")
        if self.deadline_ns is None:
            raise ValueError("version monitor is finished")
        self._last_now_ns = now_ns
        if now_ns > self.deadline_ns:
            raise QualificationTimeout("version", self.deadline_ns, now_ns)

    def feed(self, chunk: bytes, *, now_ns: int) -> None:
        self.check_deadline(now_ns)

    def finish(self, exit_code: int, *, now_ns: int) -> tuple[dict[str, Any], ...]:
        self.check_deadline(now_ns)
        if type(exit_code) is not int or exit_code != 0:
            raise ValueError("version child did not exit zero")
        self.deadline_ns = None
        return ()


class QualificationVersionProbe:
    """Callable ``probe(role, artifact) -> bytes`` with detached observations.

    Only server, bench and eval roles are accepted.  Failed started calls keep
    capped raw bytes, PID/exit observations, and actual release proof.  Native
    descriptor execution requires Linux; there is no pathname fallback.
    """

    def __init__(self, *, timeout_ns: int = 10_000_000_000,
                 termination_grace_ns: int = 250_000_000) -> None:
        self._timeout_ns = _positive_ns("timeout_ns", timeout_ns)
        self._grace_ns = _positive_ns("termination_grace_ns", termination_grace_ns)
        self._observations: list[dict[str, Any]] = []

    @property
    def observations(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(observation) for observation in self._observations)

    def _record(self, role: str, transport: _Transport, reason: str) -> None:
        # Snapshot directly: finalizing a monitor again could hide an interrupt
        # or replace the earlier transport error.  All fields are immutable.
        process = transport.process
        self._observations.append({
            "role": role,
            "pid": process.pid if process is not None else None,
            "reason": reason,
            "stdout": bytes(transport.buffers["stdout"]),
            "stderr": bytes(transport.buffers["stderr"]),
            "returncode": (process.returncode if process is not None
                           and transport.wait_owned else None),
            "cleanup_complete": transport.cleanup_complete,
            "stdout_truncated": transport.truncated["stdout"],
            "stderr_truncated": transport.truncated["stderr"],
        })

    def __call__(self, role: str, artifact: QualificationArtifact) -> bytes:
        if type(role) is not str or role not in ("server", "bench", "eval"):
            raise ValueError("version role must be server, bench or eval")
        if not isinstance(artifact, QualificationArtifact):
            raise TypeError("version probe needs a live qualification artifact owner")
        artifact.verify()
        descriptor = artifact.fd
        if not os.fstat(descriptor).st_mode & 0o111:
            raise ValueError("version artifact must be executable")
        command = (_descriptor_exec_path(descriptor), "--version-json")
        monitor = _VersionMonitor(start_ns=time.monotonic_ns(), timeout_ns=self._timeout_ns)
        # Retain the owner before __enter__: admission/attach can itself be
        # interrupted after a real child exists but before the context yields.
        transport = _Transport(
            monitor, self._grace_ns, stdout_limit=MAX_VERSION_STDOUT_BYTES,
            stderr_limit=MAX_VERSION_STDERR_BYTES,
        )
        error_reason = "io_error"
        try:
            with _owned_child_transport(command, transport, pass_fds=(descriptor,)):
                transport.run()
            result = transport.result()
            try:
                artifact.verify()
            except (OSError, ValueError) as exc:
                error_reason = "artifact_error"
                raise ValueError(f"qualification {role} version probe artifact_error") from exc
        except BaseException as exc:
            if transport.process is not None:
                if not isinstance(exc, Exception):
                    error_reason = "interrupted"
                self._record(role, transport, error_reason)
            raise
        # No record grammar is used here.  An internal raw-clock failure is an
        # I/O/supervision error, not evidence that the returned JSON was invalid.
        reason = "io_error" if result.reason == "protocol_error" else result.reason
        self._record(role, transport, reason)
        if reason != "complete":
            raise ValueError(f"qualification {role} version probe {reason}")
        return result.stdout
