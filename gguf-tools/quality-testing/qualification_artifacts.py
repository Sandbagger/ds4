#!/usr/bin/env python3
"""Own immutable qualification file observations through retained descriptors.

A successful proc-exe check observes one reserved PID at one instant.  The
process transport, not this module, establishes and retains PID ownership.
No child, model, CUDA operation, output artifact or qualification verdict is
created here.  Callers may borrow ``fd`` but must not close it behind its owner.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compact_runtime_qualify import (
    QualificationFileIdentity,
    UINT64_MAX,
    _open_regular_nofollow,
    _qualification_file_identity,
    _sha256_open_descriptor,
)

_PROC_ROOT = Path("/proc")
_IDENTITY_KEYS = frozenset(("device", "inode", "size_bytes", "mtime_ns"))
_EXPECTED_KEYS = _IDENTITY_KEYS | {"sha256"}


def _normal_absolute_path(value: Path | str) -> Path:
    if not isinstance(value, (Path, str)):
        raise ValueError("qualification artifact path must be a Path or string")
    raw = str(value)
    raw.encode("utf-8", errors="strict")
    path = Path(raw)
    if (not raw or not path.is_absolute() or path.anchor != "/" or
            raw != str(path) or ".." in path.parts or
            any(unicodedata.category(char) == "Cc" for char in raw)):
        raise ValueError("qualification artifact requires a normal absolute UTF-8 path")
    return path


def _expected_identity(expected: Mapping[str, Any] | None) -> dict[str, str] | None:
    if expected is None:
        return None
    if not isinstance(expected, Mapping):
        raise ValueError("expected artifact identity must be a mapping")
    value = dict(expected)
    if set(value) != _EXPECTED_KEYS or any(type(key) is not str for key in value):
        raise ValueError("expected artifact identity has missing or extra fields")
    for key in _IDENTITY_KEYS:
        raw = value[key]
        if (type(raw) is not str or len(raw) > 20 or
                re.fullmatch(r"(?:0|[1-9][0-9]*)", raw) is None or
                int(raw) > UINT64_MAX):
            raise ValueError(f"expected artifact {key} must be a canonical uint64 string")
    if (type(value["sha256"]) is not str or
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("expected artifact SHA-256 must be lowercase hexadecimal")
    return value


@dataclass(frozen=True)
class QualificationArtifact:
    path: Path
    identity: QualificationFileIdentity
    sha256: str
    _fd: int | None = field(repr=False)
    _executable: bool = field(repr=False)
    _ctime_ns: int = field(repr=False)

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise ValueError("qualification artifact is closed")
        return self._fd

    def _close(self) -> None:
        descriptor = self._fd
        object.__setattr__(self, "_fd", None)
        if descriptor is not None:
            os.close(descriptor)

    def verify(self) -> None:
        """Check retained identity and the current nofollow path binding.

        Change-time also catches same-size writes with a restored mtime.
        This is a stat/path check, not a second full hash.  A failed check
        closes this owner and cannot be recovered by later path restoration.
        """
        try:
            if _qualification_file_identity(self.fd) != self.identity:
                raise ValueError("retained qualification artifact identity changed")
            if os.fstat(self.fd).st_ctime_ns != self._ctime_ns:
                raise ValueError("qualification artifact change-time changed")
            descriptor, observed = _open_regular_nofollow(self.path, "qualification artifact")
            try:
                if _qualification_file_identity(descriptor) != self.identity:
                    raise ValueError("qualification artifact path binding changed")
                if observed.st_ctime_ns != self._ctime_ns:
                    raise ValueError("qualification artifact path change-time changed")
                if self._executable and not observed.st_mode & 0o111:
                    raise ValueError("qualification artifact is no longer executable")
            finally:
                os.close(descriptor)
        except BaseException:
            self._close()
            raise


@contextmanager
def open_qualification_artifact(
    path: Path | str, *, expected: Mapping[str, Any] | None = None,
    executable: bool = False, max_bytes: int | None = None,
) -> Iterator[QualificationArtifact]:
    """Pin a regular file, authenticate its bytes, and close its owned fd."""
    source = _normal_absolute_path(path)
    expected_value = _expected_identity(expected)
    if type(executable) is not bool:
        raise ValueError("executable must be a built-in boolean")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("artifact byte cap must be a built-in positive integer")
    descriptor, observed = _open_regular_nofollow(source, "qualification artifact")
    artifact = None
    try:
        if max_bytes is not None and observed.st_size > max_bytes:
            raise ValueError("qualification artifact exceeds its byte cap")
        os.set_inheritable(descriptor, False)
        identity = _qualification_file_identity(descriptor)
        if executable and not observed.st_mode & 0o111:
            raise ValueError("qualification artifact has no execute bit")
        if expected_value is not None and any(
            str(getattr(identity, key)) != expected_value[key] for key in _IDENTITY_KEYS
        ):
            raise ValueError("qualification artifact identity does not match expected")
        digest = _sha256_open_descriptor(descriptor)
        if expected_value is not None and digest != expected_value["sha256"]:
            raise ValueError("qualification artifact SHA-256 does not match expected")
        artifact = QualificationArtifact(
            source, identity, digest, descriptor, executable, observed.st_ctime_ns,
        )
        artifact.verify()
        yield artifact
    finally:
        if artifact is None:
            os.close(descriptor)
        else:
            artifact._close()


def _open_proc_executable(pid: int) -> int:
    """Walk only real directories; permit the one kernel-owned exe link."""
    root = _normal_absolute_path(_PROC_ROOT)
    # Same fixed Darwin alias as the shared regular-file opener, for private
    # host fixtures.  Production qualification uses the fixed Linux /proc.
    if sys.platform == "darwin" and root.parts[:2] == ("/", "var"):
        if os.readlink("/var") == "private/var":
            root = Path("/private/var").joinpath(*root.parts[2:])
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("proc identity requires nofollow directory opens")
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | cloexec
    file_flags = os.O_RDONLY | cloexec | getattr(os, "O_NONBLOCK", 0)
    directory = -1
    descriptor = -1
    try:
        directory = os.open("/", directory_flags)
        for component in (*root.parts[1:], str(pid)):
            previous = directory
            directory = os.open(component, directory_flags, dir_fd=previous)
            os.close(previous)
        descriptor = os.open("exe", file_flags, dir_fd=directory)
        os.set_inheritable(descriptor, False)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        if directory >= 0:
            os.close(directory)


def authenticate_running_executable(pid: int, artifact: QualificationArtifact) -> None:
    """Match /proc/<owned-pid>/exe to the still-bound, already-hashed file.

    Invoke this while the transport reserves the exact PID and before an ACK
    can release the child.  This function performs no process-group signaling.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError("qualification PID must be a built-in positive integer")
    if not isinstance(artifact, QualificationArtifact):
        raise ValueError("qualification executable requires an artifact owner")
    try:
        artifact.verify()
        descriptor = _open_proc_executable(pid)
        try:
            if _qualification_file_identity(descriptor) != artifact.identity:
                raise ValueError("running executable does not match pinned artifact")
            artifact.verify()
        finally:
            os.close(descriptor)
    except BaseException as exc:
        artifact._close()
        if isinstance(exc, OSError):
            raise ValueError(f"cannot authenticate running executable: {exc}") from exc
        raise
