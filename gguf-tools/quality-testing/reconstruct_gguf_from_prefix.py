#!/usr/bin/env python3
"""Reconstruct a GGUF from a corrected prefix and a verified old tensor tail.

The operation is deliberately byte-preserving: this tool never serializes
GGUF metadata.  It parses both layouts with ``compare_gguf_artifacts``, copies
the corrected artifact through its aligned tensor-data offset, then appends the
old artifact's complete tensor-data suffix.  A private temporary file is linked
to the requested output name only after an independent exact size/SHA-256 pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

from compare_gguf_artifacts import (
    DEFAULT_CHUNK_BYTES,
    MAX_CHUNK_BYTES,
    ArtifactInfo,
    ComparisonError,
    FileIdentity,
    parse_gguf_layout,
)


SCHEMA = "ds4.gguf-prefix-reconstruction/v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ReconstructionError(ValueError):
    """The requested reconstruction cannot be proven safe and exact."""


def _file_identity(stream: BinaryIO) -> FileIdentity:
    status = os.fstat(stream.fileno())
    return FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
    )


@contextmanager
def _open_input(path: Path, label: str) -> Iterator[BinaryIO]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    def opener(filename: str, flags: int) -> int:
        return os.open(filename, flags | nofollow | getattr(os, "O_CLOEXEC", 0))

    try:
        stream = open(path, "rb", buffering=0, opener=opener)
    except OSError as exc:
        raise ReconstructionError(f"cannot open {label} {path}: {exc}") from exc
    try:
        yield stream
    finally:
        stream.close()


def _sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ReconstructionError(f"{label} must be exactly 64 hexadecimal digits")
    return normalized


def _positive_size(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReconstructionError(f"{label} must be a positive integer")
    return value


def _parse_layout(
    stream: BinaryIO,
    path: Path,
    label: str,
    logical_size: int | None = None,
) -> ArtifactInfo:
    try:
        return parse_gguf_layout(stream, path, logical_size)
    except ComparisonError as exc:
        raise ReconstructionError(f"invalid {label}: {exc}") from exc


def _tensor_layout(artifact: ArtifactInfo) -> tuple[object, ...]:
    return (
        artifact.version,
        artifact.alignment,
        tuple(
            (
                tensor.name,
                tensor.dimensions,
                tensor.ggml_type,
                tensor.relative_offset,
                tensor.nbytes,
            )
            for tensor in artifact.tensors
        ),
    )


def _ranges_equal(
    left: BinaryIO,
    left_offset: int,
    right: BinaryIO,
    right_offset: int,
    size: int,
    chunk_bytes: int,
) -> bool:
    left.seek(left_offset)
    right.seek(right_offset)
    remaining = size
    while remaining:
        amount = min(remaining, chunk_bytes)
        left_payload = left.read(amount)
        right_payload = right.read(amount)
        if len(left_payload) != amount or len(right_payload) != amount:
            raise ReconstructionError("short read while comparing tensor directories")
        if left_payload != right_payload:
            return False
        remaining -= amount
    return True


def _validate_layouts(
    old_stream: BinaryIO,
    old: ArtifactInfo,
    prefix_stream: BinaryIO,
    new: ArtifactInfo,
    new_size: int,
    chunk_bytes: int,
) -> int:
    if _tensor_layout(old) != _tensor_layout(new):
        raise ReconstructionError("old and corrected tensor layouts differ")

    old_directory_bytes = old.tensor_directory_end - old.metadata_end
    new_directory_bytes = new.tensor_directory_end - new.metadata_end
    if old_directory_bytes != new_directory_bytes or not _ranges_equal(
        old_stream,
        old.metadata_end,
        prefix_stream,
        new.metadata_end,
        old_directory_bytes,
        chunk_bytes,
    ):
        raise ReconstructionError("old and corrected tensor directories differ")

    old_tail_bytes = old.identity.size - old.tensor_data_offset
    new_tail_bytes = new_size - new.tensor_data_offset
    if old_tail_bytes != new_tail_bytes:
        raise ReconstructionError(
            "tensor-data suffix size equation failed: "
            f"old={old_tail_bytes}, corrected={new_tail_bytes}"
        )
    return old_tail_bytes


def _hash_range(
    stream: BinaryIO,
    offset: int,
    size: int,
    digest: object,
    chunk_bytes: int,
    label: str,
) -> None:
    stream.seek(offset)
    remaining = size
    while remaining:
        payload = stream.read(min(remaining, chunk_bytes))
        if not payload:
            raise ReconstructionError(f"short read while hashing {label}")
        digest.update(payload)
        remaining -= len(payload)


def _copy_range(
    source: BinaryIO,
    offset: int,
    size: int,
    destination: BinaryIO,
    digests: tuple[object, ...],
    chunk_bytes: int,
    label: str,
) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        payload = source.read(min(remaining, chunk_bytes))
        if not payload:
            raise ReconstructionError(f"short read while copying {label}")
        for digest in digests:
            digest.update(payload)
        view = memoryview(payload)
        while view:
            written = destination.write(view)
            if written is None or written <= 0:
                raise ReconstructionError(f"short write while copying {label}")
            view = view[written:]
        remaining -= len(payload)


def _require_unchanged(
    stream: BinaryIO, expected: FileIdentity, label: str
) -> None:
    try:
        observed = _file_identity(stream)
    except OSError as exc:
        raise ReconstructionError(f"cannot restat {label}: {exc}") from exc
    if observed != expected:
        raise ReconstructionError(f"{label} changed during reconstruction")


def _verify_file(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    chunk_bytes: int,
) -> None:
    with _open_input(path, "temporary output") as stream:
        before = _file_identity(stream)
        if not stat.S_ISREG(before.mode):
            raise ReconstructionError("temporary output is not a regular file")
        if before.size != expected_size:
            raise ReconstructionError(
                f"temporary output size mismatch: expected {expected_size}, "
                f"observed {before.size}"
            )
        digest = hashlib.sha256()
        _hash_range(stream, 0, expected_size, digest, chunk_bytes, "temporary output")
        if digest.hexdigest() != expected_sha256:
            raise ReconstructionError(
                "temporary output SHA-256 mismatch: "
                f"expected {expected_sha256}, observed {digest.hexdigest()}"
            )
        _require_unchanged(stream, before, "temporary output")


def _ensure_output_available(output: Path, expected_size: int) -> None:
    if os.path.lexists(output):
        raise ReconstructionError(f"output already exists; refusing to replace it: {output}")
    try:
        parent_status = output.parent.stat()
    except OSError as exc:
        raise ReconstructionError(f"cannot inspect output directory: {exc}") from exc
    if not stat.S_ISDIR(parent_status.st_mode):
        raise ReconstructionError(f"output parent is not a directory: {output.parent}")
    try:
        filesystem = os.statvfs(output.parent)
    except OSError as exc:
        raise ReconstructionError(f"cannot inspect output free space: {exc}") from exc
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < expected_size:
        raise ReconstructionError(
            f"insufficient output space: need {expected_size} bytes, "
            f"only {available} bytes are available"
        )


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _same_inode(path: Path, expected: os.stat_result) -> bool:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    return (
        observed.st_dev == expected.st_dev
        and observed.st_ino == expected.st_ino
    )


def _publish_no_clobber(temporary_path: Path, output_path: Path) -> None:
    try:
        temporary_identity = temporary_path.lstat()
    except OSError as exc:
        raise ReconstructionError(f"cannot inspect temporary output: {exc}") from exc
    linked = False
    try:
        try:
            os.link(temporary_path, output_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ReconstructionError(
                f"output appeared during reconstruction; refusing to replace it: "
                f"{output_path}"
            ) from exc
        linked = True
        _sync_directory(output_path.parent)
        temporary_path.unlink()
        _sync_directory(output_path.parent)
    except (OSError, ReconstructionError) as exc:
        rollback_error: OSError | None = None
        if linked and _same_inode(output_path, temporary_identity):
            try:
                output_path.unlink()
                try:
                    _sync_directory(output_path.parent)
                except OSError:
                    pass
            except OSError as cleanup_exc:
                rollback_error = cleanup_exc
        if rollback_error is not None:
            raise ReconstructionError(
                "publication failed and the verified output could not be rolled back: "
                f"{rollback_error}"
            ) from exc
        if isinstance(exc, ReconstructionError):
            raise
        raise ReconstructionError(f"cannot publish reconstructed artifact: {exc}") from exc


def reconstruct_artifact(
    old_path: Path,
    prefix_path: Path,
    output_path: Path,
    *,
    old_size: int,
    old_sha256: str,
    new_size: int,
    new_sha256: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, object]:
    old_size = _positive_size(old_size, "old size")
    new_size = _positive_size(new_size, "new size")
    old_sha256 = _sha256(old_sha256, "old SHA-256")
    new_sha256 = _sha256(new_sha256, "new SHA-256")
    if type(chunk_bytes) is not int or chunk_bytes <= 0 or chunk_bytes > MAX_CHUNK_BYTES:
        raise ReconstructionError(
            f"chunk size must be between 1 and {MAX_CHUNK_BYTES} bytes"
        )

    old_path = Path(old_path)
    prefix_path = Path(prefix_path)
    output_path = Path(output_path)
    _ensure_output_available(output_path, new_size)

    temporary_path: Path | None = None
    with _open_input(old_path, "old artifact") as old_stream, _open_input(
        prefix_path, "corrected prefix"
    ) as prefix_stream:
        old = _parse_layout(old_stream, old_path, "old artifact")
        if old.identity.size != old_size:
            raise ReconstructionError(
                f"old artifact size mismatch: expected {old_size}, "
                f"observed {old.identity.size}"
            )
        new = _parse_layout(
            prefix_stream,
            prefix_path,
            "corrected prefix",
            logical_size=new_size,
        )
        if (
            old.identity.device == new.identity.device
            and old.identity.inode == new.identity.inode
        ):
            raise ReconstructionError("old artifact and corrected prefix are the same file")
        tail_bytes = _validate_layouts(
            old_stream,
            old,
            prefix_stream,
            new,
            new_size,
            chunk_bytes,
        )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w+b", buffering=0) as destination:
                old_digest = hashlib.sha256()
                output_digest = hashlib.sha256()
                _hash_range(
                    old_stream,
                    0,
                    old.tensor_data_offset,
                    old_digest,
                    chunk_bytes,
                    "old artifact prefix",
                )
                _copy_range(
                    prefix_stream,
                    0,
                    new.tensor_data_offset,
                    destination,
                    (output_digest,),
                    chunk_bytes,
                    "corrected prefix",
                )
                _copy_range(
                    old_stream,
                    old.tensor_data_offset,
                    tail_bytes,
                    destination,
                    (old_digest, output_digest),
                    chunk_bytes,
                    "old tensor-data suffix",
                )
                observed_old_sha256 = old_digest.hexdigest()
                if observed_old_sha256 != old_sha256:
                    raise ReconstructionError(
                        "old artifact SHA-256 mismatch: "
                        f"expected {old_sha256}, observed {observed_old_sha256}"
                    )
                observed_new_sha256 = output_digest.hexdigest()
                if observed_new_sha256 != new_sha256:
                    raise ReconstructionError(
                        "reconstructed SHA-256 mismatch: "
                        f"expected {new_sha256}, observed {observed_new_sha256}"
                    )
                if destination.tell() != new_size:
                    raise ReconstructionError(
                        f"reconstructed size mismatch: expected {new_size}, "
                        f"observed {destination.tell()}"
                    )
                _require_unchanged(old_stream, old.identity, "old artifact")
                _require_unchanged(prefix_stream, new.identity, "corrected prefix")
                os.fsync(destination.fileno())

            _verify_file(temporary_path, new_size, new_sha256, chunk_bytes)
            _require_unchanged(old_stream, old.identity, "old artifact")
            _require_unchanged(prefix_stream, new.identity, "corrected prefix")

            _publish_no_clobber(temporary_path, output_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    return {
        "schema": SCHEMA,
        "old": {
            "path": str(old_path),
            "size_bytes": str(old_size),
            "sha256": old_sha256,
            "tensor_data_offset": str(old.tensor_data_offset),
        },
        "new_prefix": {
            "path": str(prefix_path),
            "size_bytes": str(new.identity.size),
            "tensor_data_offset": str(new.tensor_data_offset),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": str(new_size),
            "sha256": new_sha256,
        },
        "copied": {
            "prefix_bytes": str(new.tensor_data_offset),
            "tensor_data_bytes": str(tail_bytes),
            "offset_delta": str(new.tensor_data_offset - old.tensor_data_offset),
        },
    }


def _positive_decimal(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a decimal integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive decimal integer")
    return parsed


def _chunk_bytes(value: str) -> int:
    parsed = _positive_decimal(value)
    if parsed > MAX_CHUNK_BYTES:
        raise argparse.ArgumentTypeError(
            f"must not exceed {MAX_CHUNK_BYTES} bytes"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-size", required=True, type=_positive_decimal)
    parser.add_argument("--old-sha256", required=True)
    parser.add_argument("--new-size", required=True, type=_positive_decimal)
    parser.add_argument("--new-sha256", required=True)
    parser.add_argument(
        "--chunk-bytes",
        type=_chunk_bytes,
        default=DEFAULT_CHUNK_BYTES,
        help=f"bounded streaming buffer size (default: {DEFAULT_CHUNK_BYTES})",
    )
    parser.add_argument("old", type=Path, help="complete older GGUF artifact")
    parser.add_argument(
        "corrected_prefix",
        type=Path,
        help="corrected GGUF bytes through at least tensor_data_offset",
    )
    parser.add_argument("output", type=Path, help="new no-clobber output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = reconstruct_artifact(
            args.old,
            args.corrected_prefix,
            args.output,
            old_size=args.old_size,
            old_sha256=args.old_sha256,
            new_size=args.new_size,
            new_sha256=args.new_sha256,
            chunk_bytes=args.chunk_bytes,
        )
    except ReconstructionError as exc:
        print(f"GGUF reconstruction: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
