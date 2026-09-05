#!/usr/bin/env python3
"""Bounded POSIX transport for one qualification lifecycle, not a verdict.

The caller owns this direct child exclusively (including reaping).  Keep its
PID reserved with WNOWAIT until group signaling has ended.  Cleanup covers the
created process group, not descendants that escape into another session.
No model, executable, descriptor, or evidence-file authentication happens here.
"""

from __future__ import annotations

import errno
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from qualification_records import MAX_STREAM_BYTES
from qualification_supervisor import (
    QualificationSliceMonitor,
    QualificationTimeout,
    _DEFAULT_FIRST_TOKEN_TIMEOUT_NS,
    _DEFAULT_IDLE_TIMEOUT_NS,
    _DEFAULT_WHOLE_REQUEST_TIMEOUT_NS,
)

MAX_STDOUT_BYTES = MAX_STREAM_BYTES
MAX_STDERR_BYTES = 65_536
_READ_BYTES = 65_536
_POLL_NS = 10_000_000
_RELEASE_TIMEOUT_NS = 1_000_000_000


@dataclass(frozen=True)
class QualificationChildResult:
    reason: str
    stdout: bytes
    stderr: bytes
    records: tuple[dict[str, Any], ...]
    returncode: int | None
    cleanup_complete: bool
    timeout_phase: str | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def _exit_probe() -> Callable[[int], bool]:
    """Observe exit without consuming the wait status or releasing the PID."""
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    if hasattr(os, "waitid"):
        def exited(pid: int) -> bool:
            return os.waitid(os.P_PID, pid, flags) is not None
        return exited
    if sys.platform != "darwin":
        raise OSError(errno.ENOSYS, "qualification transport requires waitid/WNOWAIT")

    # Darwin has libc waitid but some Python builds do not expose os.waitid.
    # Darwin siginfo_t starts with an int si_signo.  No other ABI fields are
    # decoded.  The aligned opaque storage exceeds Darwin's 104-byte struct.
    import ctypes

    class Siginfo(ctypes.Structure):
        _fields_ = [("signo", ctypes.c_int), ("opaque", ctypes.c_long * 128)]

    waitid = ctypes.CDLL(None, use_errno=True).waitid
    waitid.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Siginfo), ctypes.c_int]
    waitid.restype = ctypes.c_int

    def exited(pid: int) -> bool:
        info = Siginfo()
        if waitid(os.P_PID, pid, ctypes.byref(info), flags) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if info.signo not in (0, signal.SIGCHLD):
            raise OSError(errno.EPROTO, "unexpected waitid signal")
        return info.signo != 0

    return exited


class _Transport:
    def __init__(self, monitor: QualificationSliceMonitor, grace_ns: int,
                 exited: Callable[[int], bool]) -> None:
        self.monitor = monitor
        self.grace_ns = grace_ns
        self.exited = exited
        self.process: subprocess.Popen[bytes] | None = None
        self.owned_group = False
        self.pid_reserved = False
        self.wait_owned = True
        self.reason: str | None = None
        self.timeout_phase: str | None = None
        self.monitor_open = True
        self.io_open = True
        self.close_ok = True
        self.buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self.caps = {"stdout": MAX_STDOUT_BYTES, "stderr": MAX_STDERR_BYTES}
        self.truncated = {"stdout": False, "stderr": False}
        self.pipes: dict[int, tuple[str, Any]] = {}
        self.eof = {"stdout": False, "stderr": False}

    def fail(self, reason: str, phase: str | None = None) -> None:
        if self.reason is None:
            self.reason = reason
            self.timeout_phase = phase

    def check_clock(self) -> None:
        if self.monitor_open:
            try:
                self.monitor.check_deadline(time.monotonic_ns())
            except QualificationTimeout as exc:
                self.monitor_open = False
                self.fail("timeout", exc.phase)
            except (TypeError, ValueError):
                self.monitor_open = False
                self.fail("protocol_error")

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.pid_reserved = True
        pid = process.pid
        if pid <= 1 or pid in (os.getpid(), os.getpgrp(), os.getppid()):
            raise OSError(errno.EPERM, "unsafe qualification child process group")
        # Popen successfully completed setsid before exec.  The unreaped direct
        # child reserves this identity; no poll()/wait() occurs before KILL.
        self.owned_group = True
        for name in ("stdout", "stderr"):
            pipe = getattr(process, name)
            fd = pipe.fileno()
            self.pipes[fd] = (name, pipe)
        for fd in self.pipes:
            os.set_blocking(fd, False)

    def close_pipe(self, fd: int) -> None:
        _, pipe = self.pipes.pop(fd)
        try:
            pipe.close()
        except OSError:
            self.close_ok = False
            self.fail("io_error")

    def read_ready(self, timeout_ns: int) -> None:
        if not self.io_open:
            select.select([], [], [], timeout_ns / 1_000_000_000)
            return
        try:
            readable, _, _ = select.select(
                list(self.pipes), [], [], timeout_ns / 1_000_000_000
            )
        except (OSError, ValueError):
            self.io_open = False
            self.fail("io_error")
            return
        for fd in readable:
            name, _ = self.pipes[fd]
            room = self.caps[name] - len(self.buffers[name])
            # Once overflow is known, bounded reads only discard the excess
            # during cleanup.  Before then, read one extra byte to detect it.
            size = _READ_BYTES if self.truncated[name] else min(_READ_BYTES, room + 1)
            try:
                chunk = os.read(fd, size)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                self.io_open = False
                self.fail("io_error")
                return
            now_ns = time.monotonic_ns()  # receipt time, never sampled before read
            if not chunk:
                self.eof[name] = True
                self.close_pipe(fd)
                continue
            self.buffers[name].extend(chunk[:room])
            if self.truncated[name] or len(chunk) > room:
                self.truncated[name] = True
                self.fail("output_limit")
                continue  # an oversized chunk must never reach the parser
            if name == "stdout" and self.monitor_open:
                try:
                    self.monitor.feed(chunk, now_ns=now_ns)
                except QualificationTimeout as exc:
                    self.monitor_open = False
                    self.fail("timeout", exc.phase)
                except (TypeError, ValueError):
                    self.monitor_open = False
                    self.fail("protocol_error")
                except BaseException:
                    self.monitor_open = False
                    raise

    def observe_exit(self) -> bool:
        assert self.process is not None
        try:
            return self.exited(self.process.pid)
        except OSError:
            # Another reaper or a broken wait primitive removes our ownership
            # proof.  Never send a group signal after this failure.
            self.pid_reserved = False
            self.wait_owned = False
            self.fail("io_error")
            return True

    def run(self) -> None:
        while self.reason is None:
            if self.observe_exit():
                return  # proactively clean even if descendants hold the pipes
            deadline = self.monitor.deadline_ns
            assert deadline is not None
            wait_ns = max(0, min(_POLL_NS, deadline - time.monotonic_ns()))
            self.read_ready(wait_ns)
            if self.reason is None and not self.observe_exit():
                self.check_clock()

    def signal_group(self, signum: int) -> bool:
        assert self.process is not None
        if not self.owned_group or not self.pid_reserved:
            return False
        self.observe_exit()  # WNOWAIT verifies direct-child ownership again
        if not self.pid_reserved:
            return False
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        return True

    def group_gone(self) -> bool:
        assert self.process is not None
        if not self.owned_group:
            return False
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    def cleanup(self) -> bool:
        assert self.process is not None
        try:
            self.signal_group(signal.SIGTERM)
            until = time.monotonic_ns() + self.grace_ns
            while time.monotonic_ns() < until:
                self.read_ready(min(_POLL_NS, max(0, until - time.monotonic_ns())))
        except BaseException:
            self.monitor_open = False
            raise
        finally:
            # Even an interrupt in the TERM/drain phase must kill and reap.
            self.signal_group(signal.SIGKILL)
            self.pid_reserved = False  # no more signaling after this point
            until = time.monotonic_ns() + _RELEASE_TIMEOUT_NS
            try:
                while True:
                    self.process.poll()  # reaping is safe only after last signal
                    self.read_ready(0)
                    if self.process.returncode is not None and self.group_gone() and not self.pipes:
                        break
                    remaining = until - time.monotonic_ns()
                    if remaining <= 0:
                        break
                    self.read_ready(min(_POLL_NS, remaining))
            finally:
                for fd in list(self.pipes):
                    self.close_pipe(fd)
                # Covers an interrupt raised while draining after KILL, too.
                if self.process.returncode is None:
                    try:
                        self.process.wait(timeout=max(0, until - time.monotonic_ns()) / 1_000_000_000)
                    except subprocess.TimeoutExpired:
                        pass
        # Darwin can report EPERM when a group contains only the unreaped
        # zombie.  Signal acknowledgement is not release proof: the reaped
        # direct child, absent group, and both EOFs are the final authority.
        return (self.wait_owned and self.close_ok and self.process.returncode is not None
                and self.group_gone() and all(self.eof.values()))


def run_qualification_child(
    command: list[str] | tuple[str, ...], expected: Mapping[str, Any], *,
    first_token_timeout_ns: int = _DEFAULT_FIRST_TOKEN_TIMEOUT_NS,
    whole_request_timeout_ns: int = _DEFAULT_WHOLE_REQUEST_TIMEOUT_NS,
    idle_timeout_ns: int = _DEFAULT_IDLE_TIMEOUT_NS,
    termination_grace_ns: int = 250_000_000,
) -> QualificationChildResult:
    """Run one foreground producer, retain bounded observations, and clean up.

    Invalid arguments fail before launch.  The structured result describes
    transport only.  Interrupts propagate after bounded owned-group cleanup.
    """
    if type(command) not in (list, tuple):
        raise TypeError("command must be a built-in list or tuple")
    if not command:
        raise ValueError("command must not be empty")
    argv = tuple(command)
    if any(type(arg) is not str for arg in argv):
        raise TypeError("command arguments must be strings")
    if not argv[0] or any("\0" in arg for arg in argv):
        raise ValueError("command needs a program and NUL-free arguments")
    if type(termination_grace_ns) is not int:
        raise TypeError("termination_grace_ns must be a built-in integer")
    if termination_grace_ns <= 0:
        raise ValueError("termination_grace_ns must be positive")
    monitor = QualificationSliceMonitor(
        expected, start_ns=time.monotonic_ns(),
        first_token_timeout_ns=first_token_timeout_ns,
        whole_request_timeout_ns=whole_request_timeout_ns,
        idle_timeout_ns=idle_timeout_ns,
    )
    transport = _Transport(monitor, termination_grace_ns, _exit_probe())
    try:
        process = subprocess.Popen(
            argv, shell=False, start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
    except OSError:
        return QualificationChildResult("launch_error", b"", b"", (), None, True)
    try:
        try:
            transport.attach(process)
            transport.run()
        except OSError:
            transport.fail("io_error")
    finally:
        if sys.exc_info()[0] is not None:
            transport.monitor_open = False
        cleanup_complete = transport.cleanup()
    if transport.reason is None:
        if process.returncode != 0:
            transport.fail("child_exit")
        elif all(transport.eof.values()):
            try:
                monitor.finish(process.returncode, now_ns=time.monotonic_ns())
            except QualificationTimeout as exc:
                transport.fail("timeout", exc.phase)
            except (TypeError, ValueError):
                transport.fail("protocol_error")
    if transport.reason is None and not cleanup_complete:
        transport.fail("cleanup_error")
    return QualificationChildResult(
        reason=transport.reason or "complete",
        stdout=bytes(transport.buffers["stdout"]),
        stderr=bytes(transport.buffers["stderr"]), records=monitor.records,
        returncode=process.returncode if transport.wait_owned else None,
        cleanup_complete=cleanup_complete,
        timeout_phase=transport.timeout_phase,
        stdout_truncated=transport.truncated["stdout"],
        stderr_truncated=transport.truncated["stderr"],
    )
