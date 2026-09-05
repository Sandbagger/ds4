#!/usr/bin/env python3
"""Contract tests for Task-20's descriptor-owned qualification artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Keep this import before creating any fixture.  The first RED run must fail
# here while the artifact-owner seam is not yet present.
import qualification_artifacts as ARTIFACTS
from qualification_artifacts import (
    QualificationFileIdentity,
    authenticate_running_executable,
    open_qualification_artifact,
)


class _WatchdogExpired(BaseException):
    """Raised if a FIFO or other accidental blocking operation survives."""


def _call_with_alarm(function):
    """Call one operation behind a main-thread POSIX SIGALRM watchdog."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise unittest.SkipTest("FIFO watchdog requires POSIX SIGALRM")

    def _watchdog(_signum, _frame):
        raise _WatchdogExpired("artifact operation blocked behind the watchdog")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _identity_mapping(path: Path, digest: str) -> dict[str, str]:
    observed = path.stat(follow_symlinks=False)
    return {
        "device": str(observed.st_dev),
        "inode": str(observed.st_ino),
        "size_bytes": str(observed.st_size),
        "mtime_ns": str(observed.st_mtime_ns),
        "sha256": digest,
    }


def _assert_closed(test: unittest.TestCase, descriptor: int) -> None:
    try:
        os.fstat(descriptor)
    except OSError as exc:
        test.assertEqual(exc.errno, errno.EBADF)
    else:
        test.fail(f"descriptor {descriptor} remained open")


class QualificationArtifactContractTest(unittest.TestCase):
    def test_restored_mtime_does_not_hide_changed_content(self):
        with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
            path = Path(name) / "artifact.bin"
            path.write_bytes(b"original bytes")
            with open_qualification_artifact(path) as artifact:
                descriptor = artifact.fd
                before = os.fstat(descriptor)
                path.write_bytes(b"modified bytes")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                after = os.fstat(descriptor)
                self.assertEqual(after.st_ino, before.st_ino)
                self.assertEqual(after.st_size, before.st_size)
                self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
                with self.assertRaises(ValueError):
                    artifact.verify()
                _assert_closed(self, descriptor)

    def test_size_cap_rejects_before_any_hash_read(self):
        with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
            path = Path(name) / "bounded.json"
            path.write_bytes(b"large input for a tiny cap")
            with mock.patch.object(ARTIFACTS.os, "pread") as read:
                with self.assertRaises(ValueError):
                    with open_qualification_artifact(path, max_bytes=4):
                        self.fail("oversized artifact was admitted")
                read.assert_not_called()
            for invalid in (True, 0, -1, 1.0, "4"):
                with self.subTest(limit=invalid):
                    with self.assertRaises(ValueError):
                        with open_qualification_artifact(path, max_bytes=invalid):
                            self.fail("invalid bound was admitted")
            with open_qualification_artifact(path, max_bytes=path.stat().st_size) as artifact:
                self.assertEqual(artifact.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_success_pins_bytes_identity_offset_and_closure(self):
        payload = b"task20 artifact bytes\x00" * 3
        with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
            directory = Path(name)
            path = directory / "artifact.bin"
            path.write_bytes(payload)
            path.chmod(0o700)
            digest = hashlib.sha256(payload).hexdigest()
            expected = _identity_mapping(path, digest)

            with open_qualification_artifact(path, expected=expected, executable=True) as artifact:
                descriptor = artifact.fd
                self.assertIsInstance(artifact.path, Path)
                self.assertEqual(artifact.path, path.absolute())
                self.assertIsInstance(artifact.identity, QualificationFileIdentity)
                self.assertEqual(artifact.identity.device, int(expected["device"]))
                self.assertEqual(artifact.identity.inode, int(expected["inode"]))
                self.assertEqual(artifact.identity.size_bytes, len(payload))
                self.assertEqual(artifact.sha256, digest)
                self.assertFalse(os.get_inheritable(descriptor))
                with self.assertRaises(OSError) as write_error:
                    os.write(descriptor, b"x")
                self.assertEqual(write_error.exception.errno, errno.EBADF)

                os.lseek(descriptor, 7, os.SEEK_SET)
                artifact.verify()
                self.assertEqual(os.lseek(descriptor, 0, os.SEEK_CUR), 7)
                self.assertEqual(os.pread(descriptor, len(payload), 0), payload)

            _assert_closed(self, descriptor)
            with self.assertRaises(ValueError):
                _ = artifact.fd
            with self.assertRaises(ValueError):
                artifact.verify()

    def test_rejects_symlinks_fifo_invalid_expected_and_path_inputs(self):
        with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
            directory = Path(name)
            real_directory = directory / "real"
            real_directory.mkdir()
            target = real_directory / "target.bin"
            target.write_bytes(b"small target")

            leaf = directory / "leaf.bin"
            leaf.symlink_to(target)
            with self.assertRaises(ValueError):
                with open_qualification_artifact(leaf):
                    pass

            ancestor = directory / "alias"
            ancestor.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaises(ValueError):
                with open_qualification_artifact(ancestor / target.name):
                    pass

            if hasattr(os, "mkfifo"):
                fifo = directory / "artifact.fifo"
                os.mkfifo(fifo)

                def open_fifo():
                    with open_qualification_artifact(fifo):
                        pass

                with self.assertRaises(ValueError):
                    _call_with_alarm(open_fifo)

            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            expected = _identity_mapping(target, digest)
            invalid_expected = [
                ("missing field", {key: value for key, value in expected.items() if key != "sha256"}),
                ("extra field", {**expected, "extra": "nope"}),
                ("integer field", {**expected, "device": 1}),
                ("boolean field", {**expected, "inode": True}),
                ("leading zero", {**expected, "size_bytes": "01"}),
                ("signed decimal", {**expected, "mtime_ns": "+1"}),
                ("negative decimal", {**expected, "device": "-1"}),
                ("uint64 overflow", {**expected, "device": str(1 << 64)}),
                ("wrong well-formed hash", {**expected, "sha256": "b" * 64}),
                ("noncanonical hash", {**expected, "sha256": "A" * 64}),
                ("short hash", {**expected, "sha256": "a" * 63}),
                ("identity mismatch", {**expected, "inode": str(int(expected["inode"]) + 1)}),
            ]
            for label, candidate in invalid_expected:
                with self.subTest(expected=label):
                    with self.assertRaises(ValueError):
                        with open_qualification_artifact(target, expected=candidate):
                            pass

            invalid_paths = (
                "relative-artifact.bin",
                str(directory / ".." / "target.bin"),
                str(directory / "has\ncontrol"),
                str(directory / "has\x00nul"),
            )
            for value in invalid_paths:
                with self.subTest(path=value):
                    with self.assertRaises((ValueError, TypeError)):
                        with open_qualification_artifact(value):
                            pass

            non_executable = directory / "not-executable.bin"
            non_executable.write_bytes(b"no execute bit")
            non_executable.chmod(0o600)
            with self.assertRaises(ValueError):
                with open_qualification_artifact(non_executable, executable=True):
                    pass

    def test_rejects_hash_races_and_retained_path_drift_closing_descriptors(self):
        payload = b"race payload" * 8
        for mode in ("replacement", "growth"):
            with self.subTest(race=mode), tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
                directory = Path(name)
                path = directory / "artifact.bin"
                path.write_bytes(payload)
                replacement = directory / "replacement.bin"
                replacement.write_bytes(b"replacement bytes")
                opened: list[int] = []
                pread_requests: list[tuple[int, int]] = []
                original_open = ARTIFACTS.os.open
                original_pread = ARTIFACTS.os.pread
                fired = False
                grown_size = None
                # The retained artifact descriptor is read-only.  A separate
                # writable handle makes the growth race a real path mutation.
                writer = os.open(path, os.O_WRONLY)

                def recording_open(*args, **kwargs):
                    descriptor = original_open(*args, **kwargs)
                    opened.append(descriptor)
                    return descriptor

                def racing_pread(descriptor, size, offset):
                    nonlocal fired, grown_size
                    pread_requests.append((size, offset))
                    if not fired:
                        fired = True
                        if mode == "replacement":
                            os.replace(replacement, path)
                        else:
                            os.ftruncate(writer, len(payload) + 1)
                            os.pwrite(writer, b"!", len(payload))
                            grown_size = os.fstat(writer).st_size
                    return original_pread(descriptor, size, offset)

                try:
                    with mock.patch.object(ARTIFACTS.os, "open", side_effect=recording_open):
                        with mock.patch.object(ARTIFACTS.os, "pread", side_effect=racing_pread):
                            with self.assertRaises(ValueError):
                                with open_qualification_artifact(path):
                                    pass
                finally:
                    os.close(writer)
                self.assertTrue(fired)
                self.assertTrue(pread_requests)
                for request_size, offset in pread_requests:
                    self.assertGreater(request_size, 0)
                    self.assertGreaterEqual(offset, 0)
                    self.assertLessEqual(offset + request_size, len(payload))
                if mode == "growth":
                    self.assertEqual(grown_size, len(payload) + 1)
                    self.assertEqual(path.stat().st_size, len(payload) + 1)
                for descriptor in opened:
                    _assert_closed(self, descriptor)

            with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
                directory = Path(name)
                path = directory / "artifact.bin"
                path.write_bytes(payload)
                replacement = directory / "replacement.bin"
                replacement.write_bytes(b"different path bytes")
                with open_qualification_artifact(path) as artifact:
                    descriptor = artifact.fd
                    os.replace(replacement, path)
                    with self.assertRaises(ValueError):
                        artifact.verify()
                    with self.assertRaises(ValueError):
                        _ = artifact.fd
                    _assert_closed(self, descriptor)

    def test_authenticates_terminal_proc_exe_and_rejects_invalid_pid(self):
        payload = b"executable artifact"
        with tempfile.TemporaryDirectory(prefix="task20-artifact-") as name:
            directory = Path(name)
            artifact_path = directory / "runner"
            artifact_path.write_bytes(payload)
            artifact_path.chmod(0o755)
            other_path = directory / "other-runner"
            other_path.write_bytes(b"not the runner")
            other_path.chmod(0o755)
            proc_root = directory / "proc"
            proc_root.mkdir()
            pid = 4242
            pid_directory = proc_root / str(pid)
            pid_directory.mkdir()
            (pid_directory / "exe").symlink_to(artifact_path)

            with mock.patch.object(ARTIFACTS, "_PROC_ROOT", proc_root):
                with open_qualification_artifact(artifact_path, executable=True) as artifact:
                    self.assertIsNone(authenticate_running_executable(pid, artifact))
                    self.assertEqual(os.fstat(artifact.fd).st_size, len(payload))

                    (pid_directory / "exe").unlink()
                    (pid_directory / "exe").symlink_to(other_path)
                    with self.assertRaises(ValueError):
                        authenticate_running_executable(pid, artifact)

                    (pid_directory / "exe").unlink()
                    (pid_directory / "exe").symlink_to(artifact_path)
                    os.replace(other_path, artifact_path)
                    with self.assertRaises(ValueError):
                        authenticate_running_executable(pid, artifact)

                real_pid_directory = directory / "real-pid"
                real_pid_directory.mkdir()
                (real_pid_directory / "exe").symlink_to(artifact_path)
                symlink_pid_directory = proc_root / "4343"
                symlink_pid_directory.symlink_to(real_pid_directory, target_is_directory=True)
                with open_qualification_artifact(artifact_path) as artifact:
                    self.assertEqual(os.fstat(artifact.fd).st_size, len(b"not the runner"))
                    with self.assertRaises(ValueError):
                        authenticate_running_executable(4343, artifact)

                with open_qualification_artifact(artifact_path) as artifact:
                    self.assertEqual(os.fstat(artifact.fd).st_size, len(b"not the runner"))
                    for invalid_pid in (0, -1, True, 1.0, "4242", "1/../4242"):
                        with self.subTest(pid=invalid_pid):
                            with self.assertRaises(ValueError):
                                authenticate_running_executable(invalid_pid, artifact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
