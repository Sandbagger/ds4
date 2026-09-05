"""Read-only, descriptor-relative authentication of a declared evidence index.

This verifies supplied index bytes against stored files, not gate provenance or
native qualification. The caller must derive references from all required gates
and independently authenticate bundle/index inputs. Reserved regular outputs are
excluded, not authenticated. Publication must reverify its own final inputs.

The root path selects a caller-owned POSIX directory. Its final component and
all paths beneath it must be non-symlinks. No snapshot guarantee is made against
writers after return; verification detects identity/namespace changes during
its bounded reads and final metadata pass.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from compact_runtime_schema import loads_strict
from qualification_evidence import (
    _validate_path,
    build_evidence_index,
    evidence_root_sha256,
)


MAX_INDEX_BYTES = 1_048_576
MAX_EVIDENCE_FILES = 4096
MAX_FILE_BYTES = 64 * 1_048_576
MAX_TOTAL_BYTES = 256 * 1_048_576
MAX_DIRECTORY_ENTRIES = 8192
MAX_PATH_DEPTH = 32
_READ_BYTES = 1_048_576


def _identity(value: os.stat_result) -> tuple[int, ...]:
    # ctime detects equal-length writes even if a writer restores mtime.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _same(before: os.stat_result, after: os.stat_result, path: str) -> None:
    if _identity(before) != _identity(after):
        raise ValueError(f"evidence identity changed: {path}")


def _hash_file(parent_fd: int, name: str, path: str, before: os.stat_result,
               expected: dict[str, str]) -> None:
    size = int(expected["size_bytes"])
    if before.st_size != size:
        raise ValueError(f"evidence size mismatch: {path}")
    # Metadata rejects existing special files. NONBLOCK also prevents a FIFO
    # swapped in between stat and open from blocking before the fstat check.
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                 dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"evidence is not a regular file: {path}")
        _same(before, opened, path)
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = os.read(fd, min(_READ_BYTES, remaining))
            if not chunk:
                raise ValueError(f"evidence shortened during read: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(f"evidence grew during read: {path}")
        _same(before, os.fstat(fd), path)
        _same(before, os.stat(name, dir_fd=parent_fd, follow_symlinks=False), path)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"evidence SHA-256 mismatch: {path}")
    finally:
        os.close(fd)


def _scan(root_fd: int, expected: dict[str, dict[str, str]],
          directories: set[str], reserved: set[str], *,
          authenticate: bool) -> dict[str, tuple[int, ...]]:
    snapshots: dict[str, tuple[int, ...]] = {}
    found: set[str] = set()
    count = 0

    def walk(fd: int, prefix: str) -> None:
        nonlocal count
        before = os.fstat(fd)
        snapshots[prefix] = _identity(before)
        # Limit while consuming the iterator, before materializing or sorting.
        names: list[str] = []
        with os.scandir(fd) as entries:
            for entry in entries:
                count += 1
                if count > MAX_DIRECTORY_ENTRIES:
                    raise ValueError("evidence directory entry limit exceeded")
                path = f"{prefix}/{entry.name}" if prefix else entry.name
                _validate_path(path, field="evidence path")
                if len(path.split("/")) > MAX_PATH_DEPTH:
                    raise ValueError("evidence path depth limit exceeded")
                names.append(entry.name)
        for name in sorted(names, key=lambda item: item.encode("utf-8")):
            path = f"{prefix}/{name}" if prefix else name
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                if path not in directories or path in reserved:
                    raise ValueError(f"unexpected evidence directory: {path}")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY |
                                os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                try:
                    _same(metadata, os.fstat(child), path)
                    walk(child, path)
                    _same(metadata, os.stat(name, dir_fd=fd,
                                            follow_symlinks=False), path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                if path not in expected and path not in reserved:
                    raise ValueError(f"extra evidence file: {path}")
                snapshots[path] = _identity(metadata)
                if path in expected:
                    found.add(path)
                    if authenticate:
                        _hash_file(fd, name, path, metadata, expected[path])
            else:
                raise ValueError(f"symlink or special evidence file: {path}")
        _same(before, os.fstat(fd), prefix or ".")

    walk(root_fd, "")
    if found != set(expected):
        raise ValueError("evidence directory is missing indexed files")
    return snapshots


def verify_evidence_files(
    root: Path | str,
    references: Any,
    index_bytes: bytes,
    *,
    bundle_path: str,
) -> str:
    """Verify the exact declared file set and return its canonical index hash.

    Reads only bounded regular files via no-follow descriptors. The index and
    reference arguments are unchanged. This function never writes or publishes.
    """
    if not isinstance(root, (str, Path)):
        raise TypeError("evidence root must be a Path or string")
    # Lexically strip trailing separators/dot components: POSIX otherwise
    # follows a final symlink in "link/" or "link/." despite O_NOFOLLOW.
    root = Path(root)
    if type(index_bytes) is not bytes:
        raise TypeError("evidence index must be bytes")
    if not index_bytes or len(index_bytes) > MAX_INDEX_BYTES:
        raise ValueError("evidence index byte limit exceeded or empty index")
    if type(references) not in (list, tuple):
        raise TypeError("evidence references must be a list or tuple")
    # Repeated references across gates are legitimate, but parsing stays bounded.
    if len(references) > MAX_DIRECTORY_ENTRIES:
        raise ValueError("evidence reference count limit exceeded")
    try:
        observations = loads_strict(index_bytes.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid evidence index JSON") from exc
    if type(observations) is not list or len(observations) > MAX_EVIDENCE_FILES:
        raise ValueError("evidence index file count limit exceeded or wrong shape")
    canonical = build_evidence_index(references, observations, bundle_path=bundle_path)
    if index_bytes != canonical:
        raise ValueError("evidence index bytes are not canonical")
    expected = {entry["path"]: entry for entry in observations}
    reserved = {bundle_path, f"{bundle_path}.sha256", "evidence-index.json"}
    total = 0
    for entry in observations:
        size = int(entry["size_bytes"])
        if size > MAX_FILE_BYTES:
            raise ValueError("evidence file byte limit exceeded")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("evidence total byte limit exceeded")
    directories: set[str] = set()
    for path in expected.keys() | reserved:
        parts = path.split("/")
        if len(parts) > MAX_PATH_DEPTH:
            raise ValueError("evidence path depth limit exceeded")
        directories.update("/".join(parts[:i]) for i in range(1, len(parts)))
    if directories & (expected.keys() | reserved):
        raise ValueError("evidence file and directory paths conflict")
    # O_NOFOLLOW on the final root component preserves the named-root contract.
    # Ancestor path selection is the caller's responsibility, not evidence.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        _same(before, os.stat(root, follow_symlinks=False), "root")
        first = _scan(fd, expected, directories, reserved, authenticate=True)
        final = _scan(fd, expected, directories, reserved, authenticate=False)
        if first != final:
            raise ValueError("evidence tree changed during verification")
        _same(before, os.fstat(fd), "root")
        _same(before, os.stat(root, follow_symlinks=False), "root")
    finally:
        os.close(fd)
    return evidence_root_sha256(index_bytes)


__all__ = ["verify_evidence_files"]
