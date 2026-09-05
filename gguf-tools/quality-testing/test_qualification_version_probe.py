#!/usr/bin/env python3
'''Host-only contract tests for the descriptor-bound version probe.

The generated fakes are tiny Python programs.  They emit only raw version
bytes and bounded ownership metadata; they never launch DS4, use a GPU, call a
service, or access a network.
'''

from __future__ import annotations

# This import must stay before every fixture or child-process definition.  The
# first RED run is required to fail here while the seam is still absent.
import qualification_version_probe as PROBE
from qualification_version_probe import QualificationVersionProbe

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

from qualification_artifacts import open_qualification_artifact
from test_qualification_process import (
    POSIX_PROCESS_GROUPS,
    _assert_group_gone,
    _assert_metadata,
    _assert_pid_gone,
    _cleanup_owned_group,
    _read_metadata,
    _WatchdogExpired,
)


OBSERVATION_KEYS = frozenset(
    {
        "role",
        "pid",
        "reason",
        "stdout",
        "stderr",
        "returncode",
        "cleanup_complete",
        "stdout_truncated",
        "stderr_truncated",
    }
)
FD_ROOT = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"


def _fake_source(
    mode: str,
    metadata_path: Path,
    token: str,
    stdout_payload: bytes,
    stderr_payload: bytes,
    exit_code: int = 0,
    marker_delay: float = 0.0,
) -> str:
    '''Build one bounded fake with atomic authenticated metadata.'''

    return f'''#!{str(Path(sys.executable).resolve())}
import json
import os
import signal
import sys
import time

MODE = {mode!r}
METADATA = {str(metadata_path)!r}
TOKEN = {token!r}
FD_ROOT = {FD_ROOT!r}
STDOUT = {stdout_payload!r}
STDERR = {stderr_payload!r}
EXIT_CODE = {exit_code!r}
MARKER_DELAY = {marker_delay!r}
STARTED_MONOTONIC_NS = time.monotonic_ns()


def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        try:
            count = os.write(fd, view)
        except BrokenPipeError:
            return
        if count <= 0:
            return
        view = view[count:]


def _fd_inventory():
    descriptors = []
    try:
        with os.scandir(FD_ROOT) as iterator:
            for index, entry in enumerate(iterator):
                if index >= 128:
                    raise RuntimeError("inherited fd inventory exceeded 128 entries")
                try:
                    fd = int(entry.name)
                except (TypeError, ValueError):
                    continue
                if fd > 2:
                    descriptors.append(fd)
    except OSError:
        return []

    result = []
    for fd in descriptors:
        try:
            status = os.fstat(fd)
        except OSError:
            continue
        result.append({{"fd": fd, "device": status.st_dev, "inode": status.st_ino}})
    return sorted(result, key=lambda item: item["fd"])



def _metadata(**extra):
    data = {{
        "schema": "ds4.qualification-process.fake/v1",
        "token": TOKEN,
        "parent_pid": os.getppid(),
        "leader_pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "started_monotonic_ns": STARTED_MONOTONIC_NS,
        "argv": sys.argv[1:],
        "fd_inventory": _fd_inventory(),
    }}
    data.update(extra)
    temporary = METADATA + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, separators=(",", ":")) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, METADATA)


_metadata()
stdin_probe = sys.stdin.buffer.read(1)
_metadata(stdin_eof=(stdin_probe == b""))
if MODE == "timeout":
    if not hasattr(os, "fork"):
        raise SystemExit(23)
    descendant_pid = os.fork()
    if descendant_pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for _ in range(5):
            time.sleep(1)
        os._exit(0)
    _metadata(descendant_pid=descendant_pid)
    _write_all(1, STDOUT)
    _write_all(2, STDERR)
    time.sleep(5)
    raise SystemExit(EXIT_CODE)

_write_all(1, STDOUT)
_write_all(2, STDERR)
if MARKER_DELAY:
    time.sleep(MARKER_DELAY)
raise SystemExit(EXIT_CODE)
'''


def _write_fake(
    directory: Path,
    filename: str,
    mode: str,
    metadata_path: Path,
    token: str,
    stdout_payload: bytes,
    stderr_payload: bytes,
    *,
    exit_code: int = 0,
    marker_delay: float = 0.0,
) -> Path:
    path = directory / filename
    path.write_text(
        _fake_source(
            mode,
            metadata_path,
            token,
            stdout_payload,
            stderr_payload,
            exit_code,
            marker_delay,
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


@contextlib.contextmanager
def _artifact_case(
    mode: str,
    stdout_payload: bytes,
    stderr_payload: bytes,
    *,
    exit_code: int = 0,
    marker_delay: float = 0.0,
) -> Iterator[tuple[Path, Path, str, Any]]:
    '''Yield one live artifact owner and clean only its authenticated group.'''

    with tempfile.TemporaryDirectory(prefix="qualification-version-") as name:
        directory = Path(name)
        metadata_path = directory / "owned-process.json"
        token = uuid.uuid4().hex
        fake = _write_fake(
            directory,
            "qualification_version_fake.py",
            mode,
            metadata_path,
            token,
            stdout_payload,
            stderr_payload,
            exit_code=exit_code,
            marker_delay=marker_delay,
        )
        try:
            with open_qualification_artifact(fake, executable=True) as artifact:
                yield fake, metadata_path, token, artifact
        finally:
            # This is a bounded, metadata-authenticated fallback for a broken
            # implementation.  It never scans or signals unrelated PIDs.
            _cleanup_owned_group(metadata_path, token)


def _call_probe_bounded(
    probe: QualificationVersionProbe,
    role: str,
    artifact: Any,
    *,
    watchdog_seconds: float = 2.0,
) -> bytes:
    '''Invoke one native probe behind the main-thread BaseException watchdog.

    Darwin cannot exec a regular Python or Mach-O file through /dev/fd on this
    host.  Only for this test helper, the production descriptor-path helper is
    patched and Popen is adapted to an absolute Python interpreter plus a
    bootstrap that reads the retained descriptor with os.pread.  The adapter
    checks the requested path and pass_fds, does not test Darwin kernel-native
    executable origin, and never reads the mutable artifact pathname.
    '''

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise unittest.SkipTest("version probe watchdog requires POSIX SIGALRM")

    def _watchdog(_signum: int, _frame: Any) -> None:
        raise _WatchdogExpired("QualificationVersionProbe exceeded its watchdog")

    def _invoke() -> bytes:
        if sys.platform != "darwin":
            return probe(role, artifact)

        descriptor = artifact.fd
        descriptor_path = f"{FD_ROOT}/{descriptor}"
        requested_descriptors: list[int] = []
        popen_requests: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
        # This source is deliberately tiny and only reads the inherited fd.
        bootstrap = (
            "import os,sys;"
            "fd=int(sys.argv[1]);"
            "fd_path=sys.argv[2];"
            "source=os.pread(fd, 1 << 20, 0);"
            "sys.argv=[fd_path, '--version-json'];"
            "namespace={'__name__':'__main__','__file__':fd_path};"
            "exec(compile(source.decode('utf-8'), fd_path, 'exec'), namespace, namespace)"
        )

        def _descriptor_exec_path(candidate: int) -> str:
            if candidate != descriptor:
                raise AssertionError("production requested a non-owner descriptor")
            requested_descriptors.append(candidate)
            return descriptor_path

        # Capture the currently installed Popen.  In the pathname-race test
        # this is the outer replacement wrapper, so the race remains exercised.
        current_popen = subprocess.Popen

        def _adapt_popen(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("args")
            if not isinstance(command, (list, tuple)) or not command:
                raise AssertionError("production Popen command was not a sequence")
            if tuple(command) != (descriptor_path, "--version-json"):
                raise AssertionError("production did not request the exact descriptor path and flag")
            pass_fds = kwargs.get("pass_fds", ())
            observed_pass_fds = tuple(pass_fds) if pass_fds is not None else ()
            if observed_pass_fds != (descriptor,):
                raise AssertionError("production inherited more or fewer than the artifact fd")
            popen_requests.append((tuple(command), observed_pass_fds))
            adapted_command = [
                sys.executable,
                "-c",
                bootstrap,
                str(descriptor),
                descriptor_path,
            ]
            if args:
                return current_popen(adapted_command, *args[1:], **kwargs)
            adapted_kwargs = dict(kwargs)
            adapted_kwargs["args"] = adapted_command
            return current_popen(**adapted_kwargs)

        with (
            mock.patch.object(
                PROBE,
                "_descriptor_exec_path",
                side_effect=_descriptor_exec_path,
            ),
            mock.patch.object(subprocess, "Popen", side_effect=_adapt_popen),
        ):
            result = probe(role, artifact)
        if requested_descriptors != [descriptor] or len(popen_requests) != 1:
            raise AssertionError("production did not make exactly one checked descriptor launch")
        return result

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.setitimer(signal.ITIMER_REAL, watchdog_seconds)
    try:
        return _invoke()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)



def _assert_observation(
    test: unittest.TestCase,
    observation: dict[str, Any],
    *,
    role: str,
    reason: str,
) -> None:
    test.assertIs(type(observation), dict)
    test.assertEqual(set(observation), OBSERVATION_KEYS)
    test.assertEqual(observation["role"], role)
    test.assertEqual(observation["reason"], reason)
    test.assertIsInstance(observation["stdout"], bytes)
    test.assertIsInstance(observation["stderr"], bytes)
    test.assertIs(type(observation["cleanup_complete"]), bool)
    test.assertIs(type(observation["stdout_truncated"]), bool)
    test.assertIs(type(observation["stderr_truncated"]), bool)


def _assert_owned_cleanup(
    test: unittest.TestCase,
    metadata_path: Path,
    token: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    metadata = _read_metadata(metadata_path)
    _assert_metadata(test, metadata, token)
    test.assertIs(type(observation["pid"]), int)
    test.assertEqual(observation["pid"], metadata["leader_pid"])
    test.assertTrue(observation["cleanup_complete"])
    _assert_pid_gone(test, metadata["leader_pid"])
    _assert_group_gone(test, metadata["pgid"])
    return metadata


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "version probe tests require POSIX process groups and fork",
)
class QualificationVersionProbeHostTest(unittest.TestCase):
    def test_success_raw_streams_flag_fd_only_inheritance_and_borrowed_owner(self) -> None:
        stdout = b"not-json\x00raw-version\xff\n"
        stderr = b"diagnostic\x00stderr\n"
        probe = QualificationVersionProbe(
            timeout_ns=2_000_000_000,
            termination_grace_ns=30_000_000,
        )
        with _artifact_case("success", stdout, stderr) as (
            fake,
            metadata_path,
            token,
            artifact,
        ):
            # A deliberately inheritable unrelated descriptor proves that the
            # child receives exactly the retained artifact fd.
            sentinel_path = fake.parent / "unrelated-sentinel"
            sentinel_path.write_bytes(b"must-not-be-inherited")
            sentinel_fd = os.open(sentinel_path, os.O_RDONLY)
            os.set_inheritable(sentinel_fd, True)
            try:
                descriptor = artifact.fd
                source_bytes = fake.read_bytes()
                source_digest = hashlib.sha256(source_bytes).digest()
                roles = ("server", "bench", "eval")
                for role in roles:
                    result = _call_probe_bounded(probe, role, artifact)
                    self.assertEqual(result, stdout)
                    metadata = _read_metadata(metadata_path)
                    _assert_metadata(self, metadata, token)
                    self.assertEqual(metadata["argv"], ["--version-json"])
                    self.assertTrue(metadata["stdin_eof"])
                    targets = metadata["fd_inventory"]
                    self.assertEqual(len(targets), 1)
                    artifact_stat = os.fstat(descriptor)
                    matching = [
                        item for item in targets
                        if item["device"] == artifact_stat.st_dev
                        and item["inode"] == artifact_stat.st_ino
                    ]
                    self.assertEqual(len(matching), 1)
                    self.assertEqual(matching[0]["fd"], descriptor)
                    sentinel_stat = os.fstat(sentinel_fd)
                    self.assertFalse(
                        any(
                            item["device"] == sentinel_stat.st_dev
                            and item["inode"] == sentinel_stat.st_ino
                            for item in targets
                        )
                    )
                    observation = probe.observations[-1]
                    _assert_observation(self, observation, role=role, reason="complete")
                    self.assertEqual(observation["stdout"], stdout)
                    self.assertEqual(observation["stderr"], stderr)
                    self.assertEqual(observation["returncode"], 0)
                    self.assertFalse(observation["stdout_truncated"])
                    self.assertFalse(observation["stderr_truncated"])
                    self.assertTrue(observation["cleanup_complete"])
                    self.assertEqual(observation["pid"], metadata["leader_pid"])
                    _assert_pid_gone(self, metadata["leader_pid"])
                    _assert_group_gone(self, metadata["pgid"])

                # The owner remains usable because the probe only borrowed it.
                self.assertEqual(os.pread(descriptor, len(source_bytes), 0), source_bytes)
                self.assertEqual(hashlib.sha256(os.pread(descriptor, 1 << 20, 0)).digest(), source_digest)
                first = probe.observations
                self.assertIs(type(first), tuple)
                self.assertEqual(len(first), 3)
                first[0]["stdout"] = b"tampered"
                first[0]["reason"] = "tampered"
                second = probe.observations
                self.assertIs(type(second), tuple)
                self.assertIsNot(first, second)
                self.assertEqual(second[0]["stdout"], stdout)
                self.assertEqual(second[0]["reason"], "complete")
                self.assertEqual(set(second[0]), OBSERVATION_KEYS)
            finally:
                os.close(sentinel_fd)

    def test_timeout_nonzero_and_output_limits_raise_with_prefixes_and_cleanup(self) -> None:
        cases = (
            ("timeout", b"timeout-prefix\n", b"timeout-stderr\n", 0, "timeout"),
            ("child-exit", b"child-prefix\n", b"child-stderr\n", 7, "child_exit"),
        )
        for mode, stdout, stderr, exit_code, reason in cases:
            with self.subTest(mode=mode), _artifact_case(
                mode,
                stdout,
                stderr,
                exit_code=exit_code,
            ) as (_, metadata_path, token, artifact):
                probe = QualificationVersionProbe(
                    timeout_ns=100_000_000,
                    termination_grace_ns=30_000_000,
                )
                with self.assertRaises(ValueError):
                    _call_probe_bounded(probe, "server", artifact)
                observations = probe.observations
                self.assertEqual(len(observations), 1)
                observation = observations[0]
                _assert_observation(self, observation, role="server", reason=reason)
                self.assertTrue(observation["cleanup_complete"])
                self.assertTrue(observation["stdout"].startswith(stdout))
                self.assertTrue(observation["stderr"].startswith(stderr))
                self.assertFalse(observation["stdout_truncated"])
                self.assertFalse(observation["stderr_truncated"])
                metadata = _assert_owned_cleanup(self, metadata_path, token, observation)
                if reason == "child_exit":
                    self.assertEqual(observation["returncode"], exit_code)
                else:
                    self.assertIsInstance(observation["returncode"], int)
                    descendant_pid = metadata.get("descendant_pid")
                    self.assertIs(type(descendant_pid), int)
                    _assert_pid_gone(self, descendant_pid)

        limit_cases = (
            ("stdout-limit", b"stdout-prefix-", b"", "stdout"),
            ("stderr-limit", b"", b"stderr-prefix-", "stderr"),
        )
        for mode, stdout_prefix, stderr_prefix, stream in limit_cases:
            with self.subTest(mode=mode), _artifact_case(
                "success",
                stdout_prefix + (b"S" * 96 if stream == "stdout" else b""),
                stderr_prefix + (b"E" * 96 if stream == "stderr" else b""),
            ) as (_, metadata_path, token, artifact):
                probe = QualificationVersionProbe(
                    timeout_ns=1_000_000_000,
                    termination_grace_ns=30_000_000,
                )
                with (
                    mock.patch.object(PROBE, "MAX_VERSION_STDOUT_BYTES", 24),
                    mock.patch.object(PROBE, "MAX_VERSION_STDERR_BYTES", 24),
                    self.assertRaises(ValueError),
                ):
                    _call_probe_bounded(probe, "bench", artifact)
                observation = probe.observations[0]
                _assert_observation(self, observation, role="bench", reason="output_limit")
                self.assertTrue(observation["cleanup_complete"])
                self.assertLessEqual(len(observation["stdout"]), 24)
                self.assertLessEqual(len(observation["stderr"]), 24)
                self.assertTrue(observation[stream].startswith(
                    stdout_prefix if stream == "stdout" else stderr_prefix
                ))
                self.assertTrue(observation[f"{stream}_truncated"])
                other = "stderr" if stream == "stdout" else "stdout"
                self.assertFalse(observation[f"{other}_truncated"])
                _assert_owned_cleanup(self, metadata_path, token, observation)

    def test_interrupt_propagates_after_owned_cleanup(self) -> None:
        with _artifact_case(
            "timeout",
            b"interrupt-prefix\n",
            b"interrupt-stderr\n",
        ) as (_, metadata_path, token, artifact):
            probe = QualificationVersionProbe(
                timeout_ns=100_000_000,
                # Force the external main-thread watchdog into cleanup.  The
                # interrupted call must still kill/reap this exact group.
                termination_grace_ns=3_000_000_000,
            )
            with self.assertRaises(_WatchdogExpired):
                _call_probe_bounded(probe, "eval", artifact)
            observations = probe.observations
            self.assertEqual(len(observations), 1)
            observation = observations[0]
            _assert_observation(self, observation, role="eval", reason="interrupted")
            self.assertTrue(observation["cleanup_complete"])
            self.assertTrue(observation["stdout"].startswith(b"interrupt-prefix\n"))
            self.assertTrue(observation["stderr"].startswith(b"interrupt-stderr\n"))
            _assert_owned_cleanup(self, metadata_path, token, observation)

    def test_interrupt_before_context_yield_keeps_started_observations(self) -> None:
        import qualification_process as PROCESS

        interrupted = KeyboardInterrupt("version admission interrupt")
        original_attach = PROCESS._Transport.attach
        admitted: list[int] = []

        def interrupt_after_attach(transport: Any, process: Any) -> None:
            original_attach(transport, process)
            # Establish actual child bytes/ownership before the injected signal,
            # but still interrupt __enter__, before the caller receives a yield.
            transport.run()
            admitted.append(process.pid)
            raise interrupted

        with _artifact_case("success", b"admitted-version", b"admitted-stderr") as (
            _fake, metadata_path, token, artifact,
        ):
            probe = QualificationVersionProbe(
                timeout_ns=1_000_000_000, termination_grace_ns=30_000_000,
            )
            with mock.patch.object(PROCESS._Transport, "attach", interrupt_after_attach):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    _call_probe_bounded(probe, "bench", artifact)
            self.assertIs(caught.exception, interrupted)
            self.assertEqual(len(admitted), 1)
            self.assertEqual(len(probe.observations), 1)
            observation = probe.observations[0]
            _assert_observation(self, observation, role="bench", reason="interrupted")
            self.assertEqual(observation["pid"], admitted[0])
            self.assertEqual(observation["stdout"], b"admitted-version")
            self.assertEqual(observation["stderr"], b"admitted-stderr")
            _assert_owned_cleanup(self, metadata_path, token, observation)

    def test_path_replacement_around_popen_executes_pinned_bytes_then_artifact_error(self) -> None:
        original = b"ORIGINAL-PINNED-VERSION\n"
        replacement = b"REPLACEMENT-MUST-NOT-RUN\n"
        with _artifact_case(
            "success",
            original,
            b"original-diagnostic\n",
            marker_delay=0.2,
        ) as (fake, metadata_path, token, artifact):
            replacement_path = fake.parent / "replacement.py"
            _write_fake(
                fake.parent,
                replacement_path.name,
                "success",
                fake.parent / "replacement-metadata.json",
                "replacement-token",
                replacement,
                b"replacement-diagnostic\n",
            )
            real_popen = subprocess.Popen
            popen_calls: list[tuple[Any, ...]] = []

            def replace_before_native_spawn(*args: Any, **kwargs: Any) -> Any:
                popen_calls.append(tuple(args))
                os.replace(replacement_path, fake)
                return real_popen(*args, **kwargs)

            probe = QualificationVersionProbe(
                timeout_ns=1_000_000_000,
                termination_grace_ns=30_000_000,
            )
            with mock.patch.object(
                subprocess, "Popen", side_effect=replace_before_native_spawn
            ):
                with self.assertRaises(ValueError):
                    _call_probe_bounded(probe, "eval", artifact)

            self.assertEqual(len(popen_calls), 1)
            self.assertTrue(fake.exists())
            self.assertFalse(replacement_path.exists())
            self.assertEqual(len(probe.observations), 1)
            observation = probe.observations[0]
            _assert_observation(self, observation, role="eval", reason="artifact_error")
            self.assertEqual(observation["stdout"], original)
            self.assertNotIn(replacement, observation["stdout"])
            self.assertEqual(observation["stderr"], b"original-diagnostic\n")
            _assert_owned_cleanup(self, metadata_path, token, observation)
            self.assertEqual(observation["returncode"], 0)
            with self.assertRaises(ValueError):
                _ = artifact.fd

    def test_invalid_role_closed_owner_and_bad_timeout_do_not_launch_or_record(self) -> None:
        with self.assertRaises(ValueError):
            QualificationVersionProbe(timeout_ns=0)
        with self.assertRaises(ValueError):
            QualificationVersionProbe(termination_grace_ns=0)

        with _artifact_case("success", b"must-not-run", b"") as (
            fake,
            metadata_path,
            _token,
            artifact,
        ):
            probe = QualificationVersionProbe(
                timeout_ns=500_000_000,
                termination_grace_ns=30_000_000,
            )
            with self.assertRaises(ValueError):
                probe("invalid", artifact)
            self.assertEqual(probe.observations, ())
            self.assertFalse(metadata_path.exists())

            artifact._close()
            with self.assertRaises(ValueError):
                probe("server", artifact)
            self.assertEqual(probe.observations, ())
            self.assertFalse(metadata_path.exists())
            self.assertFalse(fake.with_name("owned-process.json").exists())

    @unittest.skipUnless(
        sys.platform == "darwin",
        "Darwin-only direct descriptor execution fail-closed check",
    )
    def test_darwin_native_descriptor_exec_fails_closed_before_popen(self) -> None:
        with _artifact_case("success", b"must-not-run", b"") as (
            _fake,
            metadata_path,
            _token,
            artifact,
        ):
            probe = QualificationVersionProbe(
                timeout_ns=500_000_000,
                termination_grace_ns=30_000_000,
            )
            with mock.patch.object(subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    probe("server", artifact)
                popen.assert_not_called()
            self.assertEqual(probe.observations, ())
            self.assertFalse(metadata_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
