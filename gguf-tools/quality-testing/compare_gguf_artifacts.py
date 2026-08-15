#!/usr/bin/env python3
"""Compare GGUF layout and tensor payload identity without loading the model.

The comparison is deliberately metadata-aware but weight-focused: metadata
values may differ while a successful exit still requires the complete tensor
directory and every named tensor payload to match.  File and tensor SHA-256
digests are calculated in one bounded-memory pass per artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Sequence


SCHEMA = "ds4.gguf-artifact-comparison/v1"
GGUF_HEADER_BYTES = 24
GGUF_DEFAULT_ALIGNMENT = 32
MAX_NAME_BYTES = 1 << 20
MAX_ALIGNMENT = 1 << 20
MAX_DIMS = 4
DEFAULT_CHUNK_BYTES = 8 << 20
MAX_CHUNK_BYTES = 64 << 20

GGUF_VALUE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}
GGUF_SCALAR_BYTES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


@dataclass(frozen=True)
class TensorTrait:
    name: str
    block_elements: int
    block_bytes: int


# GGML type IDs and serialized block sizes used by DS4.  This includes every
# conventional type through BF16 plus the newer DS4 writer formats.  Laguna
# Q4_K_M specifically exercises F32/F16, Q4_K, Q5_K/Q6_K, and BF16-compatible
# layout handling; keeping the surrounding DS4 set here makes unknown formats
# fail closed instead of being guessed from neighboring tensor offsets.
TENSOR_TRAITS = {
    0: TensorTrait("f32", 1, 4),
    1: TensorTrait("f16", 1, 2),
    2: TensorTrait("q4_0", 32, 18),
    3: TensorTrait("q4_1", 32, 20),
    6: TensorTrait("q5_0", 32, 22),
    7: TensorTrait("q5_1", 32, 24),
    8: TensorTrait("q8_0", 32, 34),
    9: TensorTrait("q8_1", 32, 36),
    10: TensorTrait("q2_k", 256, 84),
    11: TensorTrait("q3_k", 256, 110),
    12: TensorTrait("q4_k", 256, 144),
    13: TensorTrait("q5_k", 256, 176),
    14: TensorTrait("q6_k", 256, 210),
    15: TensorTrait("q8_k", 256, 292),
    16: TensorTrait("iq2_xxs", 256, 66),
    17: TensorTrait("iq2_xs", 256, 74),
    18: TensorTrait("iq3_xxs", 256, 98),
    19: TensorTrait("iq1_s", 256, 50),
    20: TensorTrait("iq4_nl", 32, 18),
    21: TensorTrait("iq3_s", 256, 110),
    22: TensorTrait("iq2_s", 256, 82),
    23: TensorTrait("iq4_xs", 256, 136),
    24: TensorTrait("i8", 1, 1),
    25: TensorTrait("i16", 1, 2),
    26: TensorTrait("i32", 1, 4),
    27: TensorTrait("i64", 1, 8),
    28: TensorTrait("f64", 1, 8),
    29: TensorTrait("iq1_m", 256, 56),
    30: TensorTrait("bf16", 1, 2),
    34: TensorTrait("tq1_0", 256, 54),
    35: TensorTrait("tq2_0", 256, 66),
    39: TensorTrait("mxfp4", 32, 17),
    40: TensorTrait("nvfp4", 64, 36),
    41: TensorTrait("q1_0", 128, 18),
}


class ComparisonError(ValueError):
    """A malformed or unsupported artifact cannot be compared safely."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class MetadataEntry:
    key: str
    value_type: int
    value_type_name: str
    payload_offset: int
    payload_bytes: int
    sha256: str = ""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    type_name: str
    relative_offset: int
    absolute_offset: int
    nbytes: int
    sha256: str = ""


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    identity: FileIdentity
    version: int
    alignment: int
    metadata_end: int
    tensor_directory_end: int
    tensor_data_offset: int
    metadata: tuple[MetadataEntry, ...]
    tensors: tuple[TensorInfo, ...]
    sha256: str = ""
    metadata_sha256: str = ""
    tensor_directory_sha256: str = ""


def _file_identity(stream: BinaryIO) -> FileIdentity:
    status = os.fstat(stream.fileno())
    return FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
    )


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    if size < 0:
        raise ComparisonError(f"{label} has a negative byte count")
    payload = stream.read(size)
    if len(payload) != size:
        raise ComparisonError(f"short read while parsing {label}")
    return payload


def _read_u32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<I", _read_exact(stream, 4, label))[0]


def _read_u64(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8, label))[0]


def _ensure_available(
    stream: BinaryIO, file_size: int, size: int, label: str
) -> None:
    position = stream.tell()
    if size < 0 or position < 0 or position > file_size or size > file_size - position:
        raise ComparisonError(f"{label} extends beyond the GGUF file")


def _skip_exact(stream: BinaryIO, file_size: int, size: int, label: str) -> None:
    _ensure_available(stream, file_size, size, label)
    stream.seek(size, os.SEEK_CUR)


def _read_name(stream: BinaryIO, file_size: int, label: str) -> str:
    length = _read_u64(stream, f"{label} length")
    if length > MAX_NAME_BYTES:
        raise ComparisonError(f"{label} exceeds {MAX_NAME_BYTES} bytes")
    _ensure_available(stream, file_size, length, label)
    try:
        return _read_exact(stream, length, label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComparisonError(f"{label} is not valid UTF-8") from exc


def _skip_string(stream: BinaryIO, file_size: int, label: str) -> None:
    length = _read_u64(stream, f"{label} length")
    _skip_exact(stream, file_size, length, label)


def _skip_value(
    stream: BinaryIO,
    file_size: int,
    value_type: int,
    label: str,
    depth: int = 0,
) -> None:
    if depth > 8:
        raise ComparisonError(f"{label} exceeds the metadata nesting limit")
    scalar_bytes = GGUF_SCALAR_BYTES.get(value_type)
    if scalar_bytes is not None:
        _skip_exact(stream, file_size, scalar_bytes, label)
        return
    if value_type == 8:
        _skip_string(stream, file_size, label)
        return
    if value_type != 9:
        raise ComparisonError(f"{label} has unsupported metadata type {value_type}")

    item_type = _read_u32(stream, f"{label} array type")
    count = _read_u64(stream, f"{label} array count")
    if item_type not in GGUF_VALUE_NAMES:
        raise ComparisonError(
            f"{label} has unsupported array metadata type {item_type}"
        )
    item_bytes = GGUF_SCALAR_BYTES.get(item_type)
    if item_bytes is not None:
        total = count * item_bytes
        _skip_exact(stream, file_size, total, label)
        return
    if count > file_size // 8 + 1:
        raise ComparisonError(f"{label} has an impossible array count")
    for index in range(count):
        _skip_value(
            stream,
            file_size,
            item_type,
            f"{label}[{index}]",
            depth + 1,
        )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tensor_nbytes(dimensions: tuple[int, ...], ggml_type: int, name: str) -> tuple[int, str]:
    trait = TENSOR_TRAITS.get(ggml_type)
    if trait is None:
        raise ComparisonError(f"tensor {name!r} has unsupported GGML type {ggml_type}")
    elements = 1
    for dimension in dimensions:
        if dimension == 0:
            raise ComparisonError(f"tensor {name!r} has a zero dimension")
        elements *= dimension
    if elements % trait.block_elements != 0:
        raise ComparisonError(
            f"tensor {name!r} element count {elements} is not divisible by "
            f"{trait.name} block size {trait.block_elements}"
        )
    return elements // trait.block_elements * trait.block_bytes, trait.name


def _parse_gguf(
    stream: BinaryIO, path: Path, identity: FileIdentity
) -> ArtifactInfo:
    if not stat.S_ISREG(identity.mode):
        raise ComparisonError(f"artifact is not a regular file: {path}")
    if identity.size < GGUF_HEADER_BYTES:
        raise ComparisonError(f"artifact is too small to be GGUF: {path}")

    stream.seek(0)
    if _read_exact(stream, 4, "GGUF magic") != b"GGUF":
        raise ComparisonError(f"artifact does not have GGUF magic: {path}")
    version = _read_u32(stream, "GGUF version")
    if version not in (2, 3):
        raise ComparisonError(f"unsupported GGUF version {version}: {path}")
    tensor_count = _read_u64(stream, "GGUF tensor count")
    metadata_count = _read_u64(stream, "GGUF metadata count")
    if tensor_count > identity.size // 24:
        raise ComparisonError("GGUF tensor count cannot fit in the artifact")
    if metadata_count > identity.size // 16:
        raise ComparisonError("GGUF metadata count cannot fit in the artifact")

    alignment = GGUF_DEFAULT_ALIGNMENT
    metadata: list[MetadataEntry] = []
    metadata_keys: set[str] = set()
    for index in range(metadata_count):
        key = _read_name(stream, identity.size, f"metadata key {index}")
        if key in metadata_keys:
            raise ComparisonError(f"duplicate GGUF metadata key {key!r}")
        metadata_keys.add(key)
        value_type = _read_u32(stream, f"metadata {key!r} type")
        value_type_name = GGUF_VALUE_NAMES.get(value_type)
        if value_type_name is None:
            raise ComparisonError(
                f"metadata {key!r} has unsupported type {value_type}"
            )
        payload_offset = stream.tell()
        if key == "general.alignment":
            if value_type != 4:
                raise ComparisonError("general.alignment is not a uint32")
            alignment = _read_u32(stream, "general.alignment")
        else:
            _skip_value(stream, identity.size, value_type, f"metadata {key!r}")
        payload_end = stream.tell()
        metadata.append(
            MetadataEntry(
                key=key,
                value_type=value_type,
                value_type_name=value_type_name,
                payload_offset=payload_offset,
                payload_bytes=payload_end - payload_offset,
            )
        )

    if (
        alignment == 0
        or alignment > MAX_ALIGNMENT
        or alignment & (alignment - 1) != 0
    ):
        raise ComparisonError(f"invalid GGUF alignment {alignment}")
    metadata_end = stream.tell()

    raw_tensors: list[
        tuple[str, tuple[int, ...], int, int, int, str]
    ] = []
    tensor_names: set[str] = set()
    for index in range(tensor_count):
        name = _read_name(stream, identity.size, f"tensor name {index}")
        if name in tensor_names:
            raise ComparisonError(f"duplicate GGUF tensor name {name!r}")
        tensor_names.add(name)
        dimension_count = _read_u32(stream, f"tensor {name!r} dimension count")
        if dimension_count == 0 or dimension_count > MAX_DIMS:
            raise ComparisonError(
                f"tensor {name!r} has unsupported dimension count {dimension_count}"
            )
        dimensions = tuple(
            _read_u64(stream, f"tensor {name!r} dimension {dimension}")
            for dimension in range(dimension_count)
        )
        ggml_type = _read_u32(stream, f"tensor {name!r} type")
        relative_offset = _read_u64(stream, f"tensor {name!r} offset")
        if relative_offset % alignment != 0:
            raise ComparisonError(
                f"tensor {name!r} relative offset is not {alignment}-byte aligned"
            )
        nbytes, type_name = _tensor_nbytes(dimensions, ggml_type, name)
        raw_tensors.append(
            (name, dimensions, ggml_type, relative_offset, nbytes, type_name)
        )

    tensor_directory_end = stream.tell()
    tensor_data_offset = _align(tensor_directory_end, alignment)
    if tensor_data_offset > identity.size:
        raise ComparisonError("GGUF tensor data begins beyond the artifact")

    tensors: list[TensorInfo] = []
    for name, dimensions, ggml_type, relative_offset, nbytes, type_name in raw_tensors:
        absolute_offset = tensor_data_offset + relative_offset
        if absolute_offset > identity.size or nbytes > identity.size - absolute_offset:
            raise ComparisonError(f"tensor {name!r} extends beyond the GGUF file")
        tensors.append(
            TensorInfo(
                name=name,
                dimensions=dimensions,
                ggml_type=ggml_type,
                type_name=type_name,
                relative_offset=relative_offset,
                absolute_offset=absolute_offset,
                nbytes=nbytes,
            )
        )

    ordered = sorted(tensors, key=lambda tensor: (tensor.absolute_offset, tensor.name))
    previous_end = tensor_data_offset
    for tensor in ordered:
        if tensor.absolute_offset < previous_end:
            raise ComparisonError(f"tensor {tensor.name!r} overlaps another tensor")
        previous_end = tensor.absolute_offset + tensor.nbytes

    return ArtifactInfo(
        path=path,
        identity=identity,
        version=version,
        alignment=alignment,
        metadata_end=metadata_end,
        tensor_directory_end=tensor_directory_end,
        tensor_data_offset=tensor_data_offset,
        metadata=tuple(metadata),
        tensors=tuple(tensors),
    )


def _hash_range(
    stream: BinaryIO, offset: int, size: int, chunk_bytes: int, label: str
) -> str:
    stream.seek(offset)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        payload = stream.read(min(chunk_bytes, remaining))
        if not payload:
            raise ComparisonError(f"short read while hashing {label}")
        digest.update(payload)
        remaining -= len(payload)
    return digest.hexdigest()


def _hash_artifact(
    stream: BinaryIO, artifact: ArtifactInfo, chunk_bytes: int
) -> ArtifactInfo:
    ordered = sorted(
        artifact.tensors,
        key=lambda tensor: (tensor.absolute_offset, tensor.name),
    )
    tensor_hashes = {tensor.name: hashlib.sha256() for tensor in ordered}
    tensor_hashed_bytes = {tensor.name: 0 for tensor in ordered}
    file_digest = hashlib.sha256()

    stream.seek(0)
    position = 0
    tensor_index = 0
    while position < artifact.identity.size:
        payload = stream.read(min(chunk_bytes, artifact.identity.size - position))
        if not payload:
            raise ComparisonError(f"short read while hashing {artifact.path}")
        file_digest.update(payload)
        chunk_end = position + len(payload)

        while (
            tensor_index < len(ordered)
            and ordered[tensor_index].absolute_offset
            + ordered[tensor_index].nbytes
            <= position
        ):
            tensor_index += 1
        scan = tensor_index
        while scan < len(ordered) and ordered[scan].absolute_offset < chunk_end:
            tensor = ordered[scan]
            tensor_end = tensor.absolute_offset + tensor.nbytes
            overlap_start = max(position, tensor.absolute_offset)
            overlap_end = min(chunk_end, tensor_end)
            if overlap_start < overlap_end:
                tensor_hashes[tensor.name].update(
                    payload[overlap_start - position : overlap_end - position]
                )
                tensor_hashed_bytes[tensor.name] += overlap_end - overlap_start
            if tensor_end <= chunk_end:
                scan += 1
                tensor_index = scan
            else:
                break
        position = chunk_end

    hashed_tensors: list[TensorInfo] = []
    for tensor in artifact.tensors:
        if tensor_hashed_bytes[tensor.name] != tensor.nbytes:
            raise ComparisonError(f"did not hash the complete tensor {tensor.name!r}")
        hashed_tensors.append(
            replace(tensor, sha256=tensor_hashes[tensor.name].hexdigest())
        )

    hashed_metadata = tuple(
        replace(
            entry,
            sha256=_hash_range(
                stream,
                entry.payload_offset,
                entry.payload_bytes,
                chunk_bytes,
                f"metadata {entry.key!r}",
            ),
        )
        for entry in artifact.metadata
    )
    metadata_sha256 = _hash_range(
        stream,
        GGUF_HEADER_BYTES,
        artifact.metadata_end - GGUF_HEADER_BYTES,
        chunk_bytes,
        "GGUF metadata",
    )
    tensor_directory_sha256 = _hash_range(
        stream,
        artifact.metadata_end,
        artifact.tensor_directory_end - artifact.metadata_end,
        chunk_bytes,
        "GGUF tensor directory",
    )
    if _file_identity(stream) != artifact.identity:
        raise ComparisonError(f"artifact changed while it was hashed: {artifact.path}")
    return replace(
        artifact,
        metadata=hashed_metadata,
        tensors=tuple(hashed_tensors),
        sha256=file_digest.hexdigest(),
        metadata_sha256=metadata_sha256,
        tensor_directory_sha256=tensor_directory_sha256,
    )


def _load_artifact(path: Path, chunk_bytes: int) -> ArtifactInfo:
    try:
        with path.open("rb") as stream:
            identity = _file_identity(stream)
            artifact = _parse_gguf(stream, path, identity)
            return _hash_artifact(stream, artifact, chunk_bytes)
    except OSError as exc:
        raise ComparisonError(f"cannot read artifact {path}: {exc}") from exc


def _artifact_record(artifact: ArtifactInfo) -> dict[str, object]:
    return {
        "path": str(artifact.path),
        "size_bytes": str(artifact.identity.size),
        "sha256": artifact.sha256,
        "gguf_version": artifact.version,
        "alignment": str(artifact.alignment),
        "metadata_count": len(artifact.metadata),
        "tensor_count": len(artifact.tensors),
        "metadata_end": str(artifact.metadata_end),
        "metadata_sha256": artifact.metadata_sha256,
        "tensor_directory_end": str(artifact.tensor_directory_end),
        "tensor_directory_sha256": artifact.tensor_directory_sha256,
        "tensor_data_offset": str(artifact.tensor_data_offset),
    }


def _metadata_side(entry: MetadataEntry | None) -> dict[str, object] | None:
    if entry is None:
        return None
    return {
        "value_type": entry.value_type,
        "value_type_name": entry.value_type_name,
        "payload_offset": str(entry.payload_offset),
        "payload_bytes": str(entry.payload_bytes),
        "sha256": entry.sha256,
    }


def _tensor_side(tensor: TensorInfo | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "dimensions": [str(dimension) for dimension in tensor.dimensions],
        "ggml_type": tensor.ggml_type,
        "type_name": tensor.type_name,
        "relative_offset": str(tensor.relative_offset),
        "absolute_offset": str(tensor.absolute_offset),
        "nbytes": str(tensor.nbytes),
        "sha256": tensor.sha256,
    }


def _ordered_union(base_names: Sequence[str], candidate_names: Sequence[str]) -> list[str]:
    ordered = list(base_names)
    seen = set(ordered)
    ordered.extend(sorted(name for name in candidate_names if name not in seen))
    return ordered


def compare_artifacts(
    base_path: Path, candidate_path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> dict[str, object]:
    if chunk_bytes <= 0 or chunk_bytes > MAX_CHUNK_BYTES:
        raise ComparisonError(
            f"chunk size must be between 1 and {MAX_CHUNK_BYTES} bytes"
        )
    base = _load_artifact(base_path, chunk_bytes)
    candidate = _load_artifact(candidate_path, chunk_bytes)

    base_metadata = {entry.key: entry for entry in base.metadata}
    candidate_metadata = {entry.key: entry for entry in candidate.metadata}
    metadata_names = _ordered_union(
        [entry.key for entry in base.metadata],
        [entry.key for entry in candidate.metadata],
    )
    metadata_records: list[dict[str, object]] = []
    metadata_differences: list[str] = []
    for name in metadata_names:
        base_entry = base_metadata.get(name)
        candidate_entry = candidate_metadata.get(name)
        equal = (
            base_entry is not None
            and candidate_entry is not None
            and base_entry.value_type == candidate_entry.value_type
            and base_entry.payload_bytes == candidate_entry.payload_bytes
            and base_entry.sha256 == candidate_entry.sha256
        )
        if not equal:
            metadata_differences.append(name)
        metadata_records.append(
            {
                "key": name,
                "base": _metadata_side(base_entry),
                "candidate": _metadata_side(candidate_entry),
                "equal": equal,
            }
        )

    base_tensors = {tensor.name: tensor for tensor in base.tensors}
    candidate_tensors = {tensor.name: tensor for tensor in candidate.tensors}
    tensor_names = _ordered_union(
        [tensor.name for tensor in base.tensors],
        [tensor.name for tensor in candidate.tensors],
    )
    tensor_records: list[dict[str, object]] = []
    layout_differences: list[str] = []
    payload_differences: list[str] = []
    for name in tensor_names:
        base_tensor = base_tensors.get(name)
        candidate_tensor = candidate_tensors.get(name)
        layout_equal = (
            base_tensor is not None
            and candidate_tensor is not None
            and base_tensor.dimensions == candidate_tensor.dimensions
            and base_tensor.ggml_type == candidate_tensor.ggml_type
            and base_tensor.relative_offset == candidate_tensor.relative_offset
            and base_tensor.nbytes == candidate_tensor.nbytes
        )
        payload_equal = (
            base_tensor is not None
            and candidate_tensor is not None
            and base_tensor.nbytes == candidate_tensor.nbytes
            and base_tensor.sha256 == candidate_tensor.sha256
        )
        if not layout_equal:
            layout_differences.append(name)
        if not payload_equal:
            payload_differences.append(name)
        tensor_records.append(
            {
                "name": name,
                "base": _tensor_side(base_tensor),
                "candidate": _tensor_side(candidate_tensor),
                "layout_equal": layout_equal,
                "payload_equal": payload_equal,
            }
        )

    tensor_layout_equal = not layout_differences
    tensor_payloads_equal = not payload_differences
    return {
        "schema": SCHEMA,
        "base": _artifact_record(base),
        "candidate": _artifact_record(candidate),
        "metadata_equal": not metadata_differences,
        "tensor_layout_equal": tensor_layout_equal,
        "tensor_payloads_equal": tensor_payloads_equal,
        "compatible": tensor_layout_equal and tensor_payloads_equal,
        "metadata_differences": metadata_differences,
        "layout_differences": layout_differences,
        "payload_differences": payload_differences,
        "metadata": metadata_records,
        "tensors": tensor_records,
    }


def _positive_chunk_bytes(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk size must be a decimal integer") from exc
    if parsed <= 0 or parsed > MAX_CHUNK_BYTES:
        raise argparse.ArgumentTypeError(
            f"chunk size must be between 1 and {MAX_CHUNK_BYTES} bytes"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-bytes",
        type=_positive_chunk_bytes,
        default=DEFAULT_CHUNK_BYTES,
        help=f"bounded streaming buffer size (default: {DEFAULT_CHUNK_BYTES})",
    )
    parser.add_argument("base", type=Path, help="older/reference GGUF artifact")
    parser.add_argument("candidate", type=Path, help="new/candidate GGUF artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_artifacts(args.base, args.candidate, args.chunk_bytes)
    except ComparisonError as exc:
        print(f"gguf artifact comparison: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
