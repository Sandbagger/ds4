#!/usr/bin/env python3
"""Build and verify the immutable Laguna compact-runtime benchmark manifest.

This task intentionally stops at manifest construction.  It neither launches a
qualification server nor publishes a result bundle.
"""

from __future__ import annotations

import argparse
import array
import ast
import base64
import ctypes
import errno
import hashlib
import json
import math
import mmap
import os
import platform
import re
import select
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence


ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "tests/test-vectors/laguna-resident/benchmark-32768.txt"
GENERATOR_PATH = ROOT / "tests/test-vectors/laguna-resident/generate_benchmark_prompt.py"
ORACLE_MANIFEST_PATH = ROOT / "tests/test-vectors/laguna-resident/manifest.json"
SCHEMA_PATH = ROOT / "schemas/compact-runtime-benchmark-v1.schema.json"

SCHEMA_ID = "ds4.compact-runtime-benchmark/v1"
MODEL_REPOSITORY = "poolside/Laguna-S-2.1-GGUF"
MODEL_REVISION = "706fa69799926b6afde1af9e24ca2a4923f110a1"
MODEL_FILENAME = "laguna-s-2.1-Q4_K_M.gguf"
MODEL_SIZE = 68_248_759_648
MODEL_SHA256 = "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
SEED_SIZE = 303_104
SEED_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"
GENERATOR_SHA256 = "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"
ORACLE_TOKENIZER_REVISION = "15c9b92502fed6bc26842e98d11a6347caadb08e"
LAGUNA_VOCAB_SIZE = 100_352

LAGUNA_TEMPLATE_REVISION = "poolside-laguna-s-2.1-native-nothink-v1"
LAGUNA_TEMPLATE_PREFIX = b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
LAGUNA_TEMPLATE_SUFFIX = b"</user>\n<assistant></think>"
LAGUNA_TEMPLATE_SHA256 = hashlib.sha256(
    LAGUNA_TEMPLATE_PREFIX + b"\0" + LAGUNA_TEMPLATE_SUFFIX
).hexdigest()

PROMPT_TARGETS = (512, 2048, 8192, 28672)
QUALIFICATION_SEQUENCE_SCHEMA = "ds4.qualification-sequence/v1"
QUALIFICATION_SEQUENCE_LINE_COUNT = 24
QUALIFICATION_SEQUENCE_MAX_BYTES = 16 << 20
PROFILE_SPECS = (
    ("cache-8gib", 8 << 30, (512, 2048, 28672, 8192)),
    ("cache-12gib", 12 << 30, (2048, 8192, 512, 28672)),
    ("cache-16gib", 16 << 30, (8192, 28672, 2048, 512)),
)
EVAL_CASE_IDS = (
    "recNu3MXkvWUzHZr9",
    "001b51d76b4d422988f2c11f104a2c6c",
    "aime2025-01",
    "compsec-076",
)
TOKEN_DUMP_ARGV = (
    "--dump-tokens",
    "--raw-prompt",
    "-m",
    "{model}",
    "--prompt-file",
    "{prompt}",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
PLACEHOLDER_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:fixme|todo|tbd|unknown|placeholder|changeme|example|n/?a|none|null)(?:$|[^a-z0-9])",
    re.I,
)
UINT64_MAX = (1 << 64) - 1
IJSON_SAFE_INTEGER_MAX = (1 << 53) - 1
QUALIFICATION_PLAN_MAX_BYTES = 256 << 20
UNAVOIDABLE_RESIDENCY_LIMIT_BYTES = 2 << 30
NVML_COMPUTE_API = "nvmlDeviceGetComputeRunningProcesses_v2"
NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VERSION_BUFFER_SIZE = 128
NVML_DEVICE_UUID_BUFFER_SIZE = 96
NVML_PROCESS_COUNT_LIMIT = 1 << 20
# Linux UAPI include/uapi/linux/mman.h.  Python's mmap module does not expose
# MADV_PAGEOUT on every supported interpreter, so keep the qualification-only
# retry explicit and Linux-gated.
LINUX_MADV_PAGEOUT = 21
WARM_STABILITY_REPETITIONS = 3
WARM_OWNED_CATEGORY_DRIFT_LIMIT_BYTES = 64 << 20
RUNTIME_OWNED_CATEGORY_NAMES = (
    "static_weights",
    "expert_cache_payload",
    "cache_metadata_address_tables",
    "kv_state",
    "graph_scratch",
    "pinned_staging",
    "other_host",
    "other_cuda",
)

TokenCounter = Callable[[bytes], int]


_QUALIFICATION_CONTROL_PROTOCOL_VERSION = 1
_QUALIFICATION_CONTROL_MODEL_FD = 1
_QUALIFICATION_CONTROL_SAMPLE_READY = 2
_QUALIFICATION_CONTROL_SAMPLE_READY_ACK = 3
_QUALIFICATION_CONTROL_SAMPLE_RESULT = 4
_QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK = 5
_QUALIFICATION_CONTROL_MODEL_FD_ACK = 6
_QUALIFICATION_CONTROL_MESSAGE = struct.Struct("@IIIIQQQQQ")
_QUALIFICATION_CONTROL_RIGHTS_CAPACITY = 8


class QualificationFileIdentity(NamedTuple):
    """Path-independent identity for one retained regular-file descriptor."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int

    def as_decimal_mapping(self) -> dict[str, str]:
        return {
            "device": str(self.device),
            "inode": str(self.inode),
            "size_bytes": str(self.size_bytes),
            "mtime_ns": str(self.mtime_ns),
        }


class QualificationModelEvidence(NamedTuple):
    """Digest and identity obtained only through the opened model descriptor."""

    identity: QualificationFileIdentity
    sha256: str


def _qualification_file_identity(
    descriptor: int,
) -> QualificationFileIdentity:
    try:
        observed = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"cannot stat qualification model descriptor: {exc}") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("qualification model descriptor is not a regular file")
    values = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )
    if any(type(value) is not int or value < 0 or value > UINT64_MAX for value in values):
        raise ValueError("qualification model descriptor identity is out of range")
    return QualificationFileIdentity(*values)


def _sha256_open_descriptor(descriptor: int) -> str:
    """Hash an opened descriptor without changing its shared file offset."""
    digest = hashlib.sha256()
    offset = 0
    while True:
        try:
            chunk = os.pread(descriptor, 8 << 20, offset)
        except InterruptedError:
            continue
        except OSError as exc:
            raise ValueError(
                f"cannot hash qualification model descriptor: {exc}"
            ) from exc
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _qualification_wait_ready(
    endpoint: socket.socket,
    write: bool,
    timeout: float,
) -> bool:
    readers = [] if write else [endpoint]
    writers = [endpoint] if write else []
    ready_readers, ready_writers, _ = select.select(
        readers, writers, [], timeout
    )
    return bool(ready_writers if write else ready_readers)


class QualificationControl:
    """Parent-owned half of DS4's private same-host qualification channel.

    The child endpoint is deliberately exposed only as an fd for a future
    launcher to place in ``pass_fds`` and name with
    ``--qualification-control-fd``.  The parent retains the descriptor sent
    through SCM_RIGHTS, but callers receive only its identity and digest.
    """

    def __init__(
        self,
        parent: socket.socket,
        child: socket.socket,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        wait_ready: Callable[[socket.socket, bool, float], bool],
    ) -> None:
        self._parent: socket.socket | None = parent
        self._child: socket.socket | None = child
        self._timeout_seconds = timeout_seconds
        self._monotonic = monotonic
        self._wait_ready = wait_ready
        self._model_fd = -1
        self._model_evidence: QualificationModelEvidence | None = None
        self._last_checkpoint_sequence = 0
        self._unsafe = False

    @classmethod
    def create(
        cls,
        *,
        timeout_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        wait_ready: Callable[[socket.socket, bool, float], bool] = (
            _qualification_wait_ready
        ),
    ) -> "QualificationControl":
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("qualification control timeout must be finite and positive")
        if not callable(monotonic) or not callable(wait_ready):
            raise ValueError("qualification control clock and waiter must be callable")
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            parent.set_inheritable(False)
            child.set_inheritable(False)
            parent.setblocking(False)
            return cls(
                parent,
                child,
                timeout_seconds=float(timeout_seconds),
                monotonic=monotonic,
                wait_ready=wait_ready,
            )
        except BaseException:
            parent.close()
            child.close()
            raise

    @property
    def parent_fd(self) -> int:
        if self._parent is None:
            return -1
        return self._parent.fileno()

    @property
    def child_fd(self) -> int:
        if self._child is None:
            raise ValueError("qualification control child endpoint is closed")
        return self._child.fileno()

    def close_child_endpoint(self) -> None:
        if self._child is not None:
            self._child.close()
            self._child = None

    def _close_parent_endpoint(self) -> None:
        if self._parent is not None:
            self._parent.close()
            self._parent = None

    def _close_model_descriptor(self) -> None:
        descriptor = self._model_fd
        self._model_fd = -1
        self._model_evidence = None
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _mark_unsafe(self) -> None:
        self._unsafe = True
        self._close_model_descriptor()
        self._close_parent_endpoint()
        self.close_child_endpoint()

    def _fail(self, message: str, error_type: type[Exception] = ValueError) -> None:
        self._mark_unsafe()
        raise error_type(message)

    def _require_usable(self) -> socket.socket:
        if self._unsafe:
            raise ValueError("qualification control is already unsafe")
        if self._parent is None:
            raise ValueError("qualification control is closed")
        return self._parent

    def _deadline(self, timeout_seconds: float | None = None) -> float:
        duration = (
            self._timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            self._fail(
                "qualification control deadline duration must be finite and positive"
            )
        try:
            now = float(self._monotonic())
        except BaseException as exc:
            self._fail(f"cannot read qualification monotonic clock: {exc}")
        if not math.isfinite(now):
            self._fail("qualification monotonic clock is not finite")
        deadline = now + float(duration)
        if not math.isfinite(deadline):
            self._fail("qualification control deadline is not finite")
        return deadline

    def _wait(self, *, write: bool, deadline: float, operation: str) -> None:
        endpoint = self._require_usable()
        while True:
            try:
                now = float(self._monotonic())
            except BaseException as exc:
                self._fail(f"cannot read qualification monotonic clock: {exc}")
            if not math.isfinite(now) or now >= deadline:
                self._fail(
                    f"qualification control timed out while waiting to {operation}",
                    TimeoutError,
                )
            remaining = deadline - now
            try:
                ready = self._wait_ready(endpoint, write, remaining)
            except InterruptedError:
                continue
            except Exception as exc:
                self._fail(
                    f"qualification control failed while waiting to {operation}: {exc}"
                )
            if ready:
                return
            self._fail(
                f"qualification control timed out while waiting to {operation}",
                TimeoutError,
            )

    @staticmethod
    def _close_descriptors(descriptors: Sequence[int]) -> None:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _receive_message(
        self,
        *,
        expected_type: int,
        expected_sequence: int,
        expect_model_fd: bool,
        deadline: float,
    ) -> tuple[QualificationFileIdentity, list[int]]:
        endpoint = self._require_usable()
        payload = bytearray()
        descriptors: list[int] = []
        integer_size = array.array("i").itemsize
        ancillary_size = socket.CMSG_SPACE(
            integer_size * _QUALIFICATION_CONTROL_RIGHTS_CAPACITY
        )
        receive_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
        try:
            while len(payload) < _QUALIFICATION_CONTROL_MESSAGE.size:
                self._wait(
                    write=False,
                    deadline=deadline,
                    operation="receive protocol message",
                )
                try:
                    part, ancillary, flags, _ = endpoint.recvmsg(
                        _QUALIFICATION_CONTROL_MESSAGE.size - len(payload),
                        ancillary_size,
                        receive_flags,
                    )
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError as exc:
                    self._fail(
                        f"qualification control failed to receive protocol message: {exc}"
                    )
                if not part:
                    self._fail(
                        "qualification control peer disconnected while receiving protocol message"
                    )
                ancillary_error = ""
                for level, kind, data in ancillary:
                    if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                        ancillary_error = (
                            "qualification control message carried unexpected ancillary data"
                        )
                        continue
                    if not data:
                        ancillary_error = (
                            "qualification control SCM_RIGHTS data is malformed"
                        )
                        continue
                    aligned_size = len(data) - (len(data) % integer_size)
                    rights = array.array("i")
                    if aligned_size:
                        rights.frombytes(data[:aligned_size])
                    for descriptor in rights:
                        try:
                            os.set_inheritable(descriptor, False)
                        except OSError as exc:
                            ancillary_error = (
                                "qualification control could not make a received "
                                f"descriptor close-on-exec: {exc}"
                            )
                        descriptors.append(descriptor)
                    if aligned_size != len(data):
                        ancillary_error = (
                            "qualification control SCM_RIGHTS data is malformed"
                        )
                if flags & (
                    getattr(socket, "MSG_CTRUNC", 0)
                    | getattr(socket, "MSG_TRUNC", 0)
                ):
                    self._fail(
                        "qualification control message or ancillary data was truncated"
                    )
                if ancillary_error:
                    self._fail(ancillary_error)
                payload.extend(part)

            values = _QUALIFICATION_CONTROL_MESSAGE.unpack(payload)
            protocol, message_type, size, reserved, sequence = values[:5]
            if (
                protocol != _QUALIFICATION_CONTROL_PROTOCOL_VERSION
                or message_type != expected_type
                or size != _QUALIFICATION_CONTROL_MESSAGE.size
                or reserved != 0
                or sequence != expected_sequence
            ):
                self._fail(
                    "qualification control received a mismatched protocol type, size, or sequence"
                )
            if expect_model_fd:
                if len(descriptors) != 1:
                    self._fail(
                        "qualification control model message requires exactly one SCM_RIGHTS descriptor"
                    )
            elif descriptors:
                self._fail(
                    "qualification control barrier message carried forbidden SCM_RIGHTS descriptors"
                )
            identity = QualificationFileIdentity(*values[5:])
            return identity, descriptors
        except BaseException:
            self._close_descriptors(descriptors)
            raise

    def _send_message(
        self,
        *,
        message_type: int,
        sequence: int,
        deadline: float,
        identity: QualificationFileIdentity | None = None,
    ) -> None:
        endpoint = self._require_usable()
        identity_fields = (0, 0, 0, 0) if identity is None else tuple(identity)
        payload = _QUALIFICATION_CONTROL_MESSAGE.pack(
            _QUALIFICATION_CONTROL_PROTOCOL_VERSION,
            message_type,
            _QUALIFICATION_CONTROL_MESSAGE.size,
            0,
            sequence,
            *identity_fields,
        )
        offset = 0
        flags = getattr(socket, "MSG_NOSIGNAL", 0)
        while offset != len(payload):
            self._wait(
                write=True,
                deadline=deadline,
                operation="send acknowledgement",
            )
            try:
                written = endpoint.send(payload[offset:], flags)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                self._fail(
                    f"qualification control failed to send acknowledgement: {exc}"
                )
            if written <= 0:
                self._fail(
                    "qualification control peer disconnected while sending acknowledgement"
                )
            offset += written

    def receive_model(
        self,
        *,
        prepare_descriptor: (
            Callable[[int, QualificationModelEvidence], Any] | None
        ) = None,
    ) -> QualificationModelEvidence:
        """Verify the opened model and acknowledge it before child allocation.

        ``prepare_descriptor`` is the qualification runner's narrow seam for
        exact-fd cold preparation.  It runs after the descriptor hash and
        identity checks but before MODEL_FD_ACK releases the child.
        """
        self._require_usable()
        if self._model_evidence is not None:
            self._fail("qualification model descriptor was already received")
        if prepare_descriptor is not None and not callable(prepare_descriptor):
            raise ValueError("qualification model descriptor preparer must be callable")
        wire_identity, descriptors = self._receive_message(
            expected_type=_QUALIFICATION_CONTROL_MODEL_FD,
            expected_sequence=0,
            expect_model_fd=True,
            deadline=self._deadline(),
        )
        descriptor = descriptors.pop()
        try:
            os.set_inheritable(descriptor, False)
            before = _qualification_file_identity(descriptor)
            if wire_identity != before:
                self._fail(
                    "qualification model wire identity does not match the opened descriptor"
                )
            digest = _sha256_open_descriptor(descriptor)
            after = _qualification_file_identity(descriptor)
            if after != before:
                self._fail(
                    "qualification model descriptor identity changed while hashing"
                )
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._mark_unsafe()
            raise
        self._model_fd = descriptor
        self._model_evidence = QualificationModelEvidence(before, digest)
        try:
            if prepare_descriptor is not None:
                prepare_descriptor(descriptor, self._model_evidence)
            self.verify_model_unchanged()
            self._send_message(
                message_type=_QUALIFICATION_CONTROL_MODEL_FD_ACK,
                sequence=0,
                deadline=self._deadline(),
                identity=before,
            )
        except BaseException:
            self._mark_unsafe()
            raise
        return self._model_evidence

    def verify_model_unchanged(self) -> QualificationFileIdentity:
        """Re-stat the retained received descriptor without exposing its fd."""
        self._require_usable()
        if self._model_evidence is None or self._model_fd < 0:
            raise ValueError("qualification model descriptor has not been received")
        try:
            observed = _qualification_file_identity(self._model_fd)
        except BaseException:
            self._mark_unsafe()
            raise
        if observed != self._model_evidence.identity:
            self._fail("qualification model descriptor identity changed after hashing")
        return observed

    def bracket_sample(
        self,
        checkpoint_sequence: int,
        *,
        capture_before: Callable[[], Any],
        capture_after: Callable[[], Any],
        sample_timeout_seconds: float | None = None,
    ) -> tuple[Any, Any]:
        """Hold the child at READY and RESULT while the parent inventories it."""
        self._require_usable()
        if self._model_evidence is None:
            raise ValueError("qualification model descriptor has not been received")
        if (
            isinstance(checkpoint_sequence, bool)
            or type(checkpoint_sequence) is not int
            or checkpoint_sequence <= self._last_checkpoint_sequence
            or checkpoint_sequence > UINT64_MAX
        ):
            self._fail(
                "qualification checkpoint sequence must be strictly increasing"
            )
        if not callable(capture_before) or not callable(capture_after):
            raise ValueError("qualification checkpoint inventories must be callable")
        if sample_timeout_seconds is not None and (
            isinstance(sample_timeout_seconds, bool)
            or not isinstance(sample_timeout_seconds, (int, float))
            or not math.isfinite(float(sample_timeout_seconds))
            or sample_timeout_seconds <= 0
        ):
            raise ValueError(
                "qualification sample timeout must be finite and positive"
            )

        ready_arrival_deadline = self._deadline(sample_timeout_seconds)
        ready_identity, _ = self._receive_message(
            expected_type=_QUALIFICATION_CONTROL_SAMPLE_READY,
            expected_sequence=checkpoint_sequence,
            expect_model_fd=False,
            deadline=ready_arrival_deadline,
        )
        if ready_identity != QualificationFileIdentity(0, 0, 0, 0):
            self._fail("qualification READY message carried a model identity")
        ready_ack_deadline = self._deadline()
        self.verify_model_unchanged()
        try:
            before = capture_before()
        except BaseException:
            self._mark_unsafe()
            raise
        self._send_message(
            message_type=_QUALIFICATION_CONTROL_SAMPLE_READY_ACK,
            sequence=checkpoint_sequence,
            deadline=ready_ack_deadline,
        )

        result_deadline = self._deadline(sample_timeout_seconds)
        result_identity, _ = self._receive_message(
            expected_type=_QUALIFICATION_CONTROL_SAMPLE_RESULT,
            expected_sequence=checkpoint_sequence,
            expect_model_fd=False,
            deadline=result_deadline,
        )
        if result_identity != self._model_evidence.identity:
            self._fail(
                "qualification RESULT model identity does not match the opened descriptor"
            )
        result_ack_deadline = self._deadline()
        try:
            after = capture_after()
            self.verify_model_unchanged()
        except BaseException:
            self._mark_unsafe()
            raise
        self._send_message(
            message_type=_QUALIFICATION_CONTROL_SAMPLE_RESULT_ACK,
            sequence=checkpoint_sequence,
            deadline=result_ack_deadline,
        )
        self._last_checkpoint_sequence = checkpoint_sequence
        return before, after

    def close(self) -> None:
        self._close_model_descriptor()
        self._close_parent_endpoint()
        self.close_child_endpoint()

    def __enter__(self) -> "QualificationControl":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()


class _NvmlProcessInfoV2(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value: {value}")
    return parsed


def loads_strict(payload: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
            parse_float=_strict_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def load_manifest(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read manifest {source}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"manifest is not UTF-8: {exc}") from exc
    value = loads_strict(text)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    validate_manifest(value)
    return value


def _canonical_string(value: str) -> str:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("unsupported lone surrogate outside the RFC 8785 domain")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("unsupported non-finite number outside the RFC 8785 domain")
    if math.copysign(1.0, value) < 0 and value == 0.0:
        return "0"
    supported = ((0.0, "0"), (1.0, "1"), (0.05, "0.05"))
    for expected, rendered in supported:
        if value == expected:
            return rendered
    raise ValueError(f"unsupported float outside the manifest RFC 8785 domain: {value!r}")


def _canonical_text(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return _canonical_string(value)
    if type(value) is int:
        if abs(value) > IJSON_SAFE_INTEGER_MAX:
            raise ValueError("unsupported integer outside the RFC 8785 I-JSON domain")
        return str(value)
    if type(value) is float:
        return _canonical_float(value)
    if type(value) is list:
        return "[" + ",".join(_canonical_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("RFC 8785 object keys must be strings")
        for key in value:
            _canonical_string(key)
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return "{" + ",".join(
            _canonical_string(key) + ":" + _canonical_text(value[key]) for key in keys
        ) + "}"
    raise ValueError(f"unsupported value outside the manifest RFC 8785 domain: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonicalize the manifest's explicitly supported RFC 8785 domain."""
    return _canonical_text(value).encode("utf-8")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def _mapping(
    value: Any,
    label: str,
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    unknown = sorted(actual - required)
    if unknown:
        raise ValueError(f"{label}: unknown key {unknown[0]!r}")
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{label}: missing key {missing[0]!r}")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: Any, label: str, *, allow_template_marker: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    if "\x00" in value or any(ord(char) < 0x20 for char in value):
        raise ValueError(f"{label} contains a control character")
    if not allow_template_marker and (
        PLACEHOLDER_RE.search(value) or re.search(r"<[^>]+>", value)
    ):
        raise ValueError(f"{label} contains a placeholder")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must use the JSON float number kind")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _finite(value: Any, label: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> str:
    text = _string(value, label)
    if not DECIMAL_RE.fullmatch(text) or int(text) > UINT64_MAX:
        raise ValueError(f"{label} must be a canonical uint64 decimal string")
    if positive and text == "0":
        raise ValueError(f"{label} must be positive")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    if text == "0" * 64:
        raise ValueError(f"{label} must not use an all-zero sentinel")
    return text


def _revision(value: Any, label: str) -> str:
    text = _string(value, label)
    if not REVISION_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase 40-hex revision")
    if text == "0" * 40:
        raise ValueError(f"{label} must not use an all-zero sentinel")
    return text


def validate_qualification_version(value: Any) -> dict[str, Any]:
    """Admit one exact clean CUDA build identity for qualification.

    This is deliberately a pure boundary: the future runner owns process I/O,
    parses stdout with ``loads_strict``, then passes only the resulting value
    here before it creates any qualification evidence or CUDA child.
    """
    required = {"schema", "revision", "dirty", "backend", "features"}
    version = _mapping(value, "qualification version", required)
    if version["schema"] != "ds4.version/v1":
        raise ValueError("qualification version schema is not ds4.version/v1")
    revision = _revision(
        version["revision"], "qualification version revision"
    )
    if type(version["dirty"]) is not bool:
        raise ValueError("qualification version dirty must be a boolean")
    if version["dirty"]:
        raise ValueError("qualification requires a clean build; dirty=true")
    if type(version["backend"]) is not str or version["backend"] != "cuda":
        raise ValueError("qualification version backend must be cuda")

    raw_features = _list(
        version["features"], "qualification version features"
    )
    features: list[str] = []
    for index, feature in enumerate(raw_features):
        if (
            type(feature) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]*", feature) is None
        ):
            raise ValueError(
                f"qualification version feature {index} is invalid"
            )
        features.append(feature)
    if features != sorted(features):
        raise ValueError("qualification version features must be sorted")
    if len(features) != len(set(features)):
        raise ValueError("qualification version features must be unique")
    missing = sorted({"laguna", "ssd_streaming"} - set(features))
    if missing:
        raise ValueError(
            "qualification version features must include " + ", ".join(missing)
        )
    return {
        "schema": "ds4.version/v1",
        "revision": revision,
        "dirty": False,
        "backend": "cuda",
        "features": list(features),
    }


def _base64(value: Any, label: str) -> bytes:
    text = _string(value, label)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be canonical base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != text:
        raise ValueError(f"{label} must be nonempty canonical base64")
    return decoded


def _validate_model(value: Any) -> None:
    required = {
        "path", "repository", "revision", "filename", "size_bytes",
        "sha256", "device", "inode", "mtime_ns",
    }
    model = _mapping(value, "model", required)
    path_text = _string(model["path"], "model.path")
    if "//" in path_text:
        raise ValueError("model.path must not contain repeated slashes")
    if not Path(path_text).is_absolute():
        raise ValueError("model.path must be absolute")
    if path_text.rsplit("/", 1)[-1] != MODEL_FILENAME:
        raise ValueError("model.path does not name the pinned artifact")
    expected = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
        "size_bytes": str(MODEL_SIZE),
        "sha256": MODEL_SHA256,
    }
    for key, wanted in expected.items():
        if model[key] != wanted:
            raise ValueError(f"model.{key} does not identify the pinned artifact")
    _revision(model["revision"], "model.revision")
    _sha256(model["sha256"], "model.sha256")
    for key in ("size_bytes", "device", "inode", "mtime_ns"):
        _decimal(model[key], f"model.{key}", positive=True)


def _validate_runtime(value: Any) -> None:
    required = {
        "source_revision", "oracle_tokenizer_revision", "executable_path",
        "executable_sha256", "device", "inode", "size_bytes", "mtime_ns",
        "token_dump_argv",
    }
    runtime = _mapping(value, "prompt_source.tokenizer_runtime", required)
    _revision(runtime["source_revision"], "tokenizer_runtime.source_revision")
    oracle_revision = _revision(
        runtime["oracle_tokenizer_revision"],
        "tokenizer_runtime.oracle_tokenizer_revision",
    )
    if oracle_revision != ORACLE_TOKENIZER_REVISION:
        raise ValueError("tokenizer runtime is not bound to the promoted oracle revision")
    executable = Path(_string(runtime["executable_path"], "tokenizer_runtime.executable_path"))
    if not executable.is_absolute():
        raise ValueError("tokenizer_runtime.executable_path must be absolute")
    _sha256(runtime["executable_sha256"], "tokenizer_runtime.executable_sha256")
    for key in ("device", "inode", "size_bytes", "mtime_ns"):
        _decimal(runtime[key], f"tokenizer_runtime.{key}", positive=True)
    argv = _list(runtime["token_dump_argv"], "tokenizer_runtime.token_dump_argv")
    if argv != list(TOKEN_DUMP_ARGV):
        raise ValueError("tokenizer_runtime.token_dump_argv is not the pinned raw-token invocation")
    for index, item in enumerate(argv):
        _string(item, f"tokenizer_runtime.token_dump_argv[{index}]", allow_template_marker=True)


def _validate_prompt_source(value: Any) -> None:
    required = {
        "seed_size_bytes", "seed_sha256", "generator_sha256",
        "template_revision", "template_prefix_base64", "template_suffix_base64",
        "template_sha256", "tokenizer_runtime",
    }
    source = _mapping(value, "prompt_source", required)
    if _decimal(source["seed_size_bytes"], "prompt_source.seed_size_bytes", positive=True) != str(SEED_SIZE):
        raise ValueError("prompt_source seed size is not pinned")
    if _sha256(source["seed_sha256"], "prompt_source.seed_sha256") != SEED_SHA256:
        raise ValueError("prompt_source seed digest is not pinned")
    if _sha256(source["generator_sha256"], "prompt_source.generator_sha256") != GENERATOR_SHA256:
        raise ValueError("prompt_source generator digest is not pinned")
    if source["template_revision"] != LAGUNA_TEMPLATE_REVISION:
        raise ValueError("prompt_source template revision is not pinned")
    prefix = _base64(source["template_prefix_base64"], "prompt_source.template_prefix_base64")
    suffix = _base64(source["template_suffix_base64"], "prompt_source.template_suffix_base64")
    if prefix != LAGUNA_TEMPLATE_PREFIX or suffix != LAGUNA_TEMPLATE_SUFFIX:
        raise ValueError("prompt_source template bytes are not the official Laguna form")
    if _sha256(source["template_sha256"], "prompt_source.template_sha256") != LAGUNA_TEMPLATE_SHA256:
        raise ValueError("prompt_source template digest is not pinned")
    _validate_runtime(source["tokenizer_runtime"])


def _validate_host(value: Any) -> None:
    required = {
        "hostname", "architecture", "kernel_release", "kernel_version",
        "cuda_driver_version", "cuda_runtime_version", "gpu_uuid",
        "filesystem", "nvme", "io",
    }
    host = _mapping(value, "host", required)
    for key in (
        "hostname", "architecture", "kernel_release", "kernel_version",
        "cuda_driver_version", "cuda_runtime_version",
    ):
        _string(host[key], f"host.{key}")
    gpu_uuid = _string(host["gpu_uuid"], "host.gpu_uuid")
    if not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("host.gpu_uuid is invalid")
    if gpu_uuid == "GPU-00000000-0000-0000-0000-000000000000":
        raise ValueError("host.gpu_uuid must not use an all-zero sentinel")

    filesystem = _mapping(
        host["filesystem"],
        "host.filesystem",
        {"mount_point", "type", "source", "device", "options"},
    )
    for key in filesystem:
        _string(filesystem[key], f"host.filesystem.{key}")
    if not re.fullmatch(r"[0-9]+:[0-9]+", filesystem["device"]):
        raise ValueError("host.filesystem.device must be major:minor")

    nvme = _mapping(
        host["nvme"],
        "host.nvme",
        {"device", "model", "serial", "firmware_revision"},
    )
    for key in nvme:
        _string(nvme[key], f"host.nvme.{key}")

    io_mode = _mapping(
        host["io"],
        "host.io",
        {"direct_io", "cold_preparation_advice", "runtime_disposal_advice"},
    )
    if type(io_mode["direct_io"]) is not bool:
        raise ValueError("host.io.direct_io must be boolean")
    if io_mode != {
        "direct_io": False,
        "cold_preparation_advice":
            "madvise_random+posix_fadvise_dontneed+madvise_dontneed+linux_madv_pageout_residual",
        "runtime_disposal_advice": "madvise_dontneed",
    }:
        raise ValueError("host.io does not match the reference advice mode")


def _validate_prompts(value: Any) -> None:
    prompts = _list(value, "prompts")
    if len(prompts) != len(PROMPT_TARGETS):
        raise ValueError("prompts must contain exactly four entries")
    seed = SEED_PATH.read_bytes()
    if len(seed) != SEED_SIZE or _sha256_bytes(seed) != SEED_SHA256:
        raise ValueError("checked-in benchmark seed does not match its pinned identity")
    required = {
        "id", "token_count", "payload_prefix_bytes", "rendered_size_bytes",
        "rendered_base64", "sha256",
    }
    for index, (item, target) in enumerate(zip(prompts, PROMPT_TARGETS, strict=True)):
        prompt = _mapping(item, f"prompts[{index}]", required)
        if prompt["id"] != f"native-{target}":
            raise ValueError("prompts are reordered or have an invalid id")
        if _integer(prompt["token_count"], f"prompts[{index}].token_count") != target:
            raise ValueError("prompts are reordered or have an invalid token count")
        prefix_bytes = int(_decimal(
            prompt["payload_prefix_bytes"],
            f"prompts[{index}].payload_prefix_bytes",
        ))
        rendered_size = int(_decimal(
            prompt["rendered_size_bytes"],
            f"prompts[{index}].rendered_size_bytes",
            positive=True,
        ))
        rendered = _base64(prompt["rendered_base64"], f"prompts[{index}].rendered_base64")
        if len(rendered) != rendered_size:
            raise ValueError(f"prompts[{index}] rendered size mismatch")
        if not rendered.startswith(LAGUNA_TEMPLATE_PREFIX) or not rendered.endswith(LAGUNA_TEMPLATE_SUFFIX):
            raise ValueError(f"prompts[{index}] is not an official native-template prompt")
        payload = rendered[len(LAGUNA_TEMPLATE_PREFIX) : -len(LAGUNA_TEMPLATE_SUFFIX)]
        if len(payload) != prefix_bytes or payload != seed[:prefix_bytes]:
            raise ValueError(f"prompts[{index}] is not an exact immutable seed prefix")
        digest = _sha256(prompt["sha256"], f"prompts[{index}].sha256")
        if digest != _sha256_bytes(rendered):
            raise ValueError(f"prompts[{index}] rendered-byte hash mismatch")


def _validate_sampling(value: Any) -> None:
    required = {
        "max_generated_tokens", "temperature", "top_k", "top_p", "min_p",
        "seed", "stop_sequences", "stop_token_policy",
    }
    sampling = _mapping(value, "sampling", required)
    expected = {
        "max_generated_tokens": 512,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "min_p": 0.05,
        "seed": 1,
        "stop_sequences": [],
        "stop_token_policy": "model-native",
    }
    for key in ("max_generated_tokens", "temperature", "top_k", "top_p", "seed"):
        _integer(sampling[key], f"sampling.{key}")
    for key in ("min_p",):
        _float(sampling[key], f"sampling.{key}")
    if dict(sampling) != expected:
        raise ValueError("sampling configuration is not the immutable reference configuration")


def _validate_execution(value: Any) -> None:
    required = {
        "qualification_cold_preparations", "fresh_process_runs",
        "same_process_warm_repetitions", "whole_request_timeout_seconds",
        "first_token_timeout_seconds", "warm_statistic", "scope",
    }
    execution = _mapping(value, "execution", required)
    expected = {
        "qualification_cold_preparations": 1,
        "fresh_process_runs": 1,
        "same_process_warm_repetitions": 3,
        "whole_request_timeout_seconds": 2700,
        "first_token_timeout_seconds": 900,
        "warm_statistic": "median-of-exactly-three",
        "scope": "each-profile-prompt-pair",
    }
    for key in (
        "qualification_cold_preparations", "fresh_process_runs",
        "same_process_warm_repetitions", "whole_request_timeout_seconds",
        "first_token_timeout_seconds",
    ):
        _integer(execution[key], f"execution.{key}")
    if dict(execution) != expected:
        raise ValueError("execution configuration is not the immutable reference protocol")


def _validate_profiles(value: Any) -> None:
    profiles = _list(value, "profiles")
    if len(profiles) != len(PROFILE_SPECS):
        raise ValueError("profiles must contain the fixed 8/12/16-GiB sweep")
    required = {"profile_id", "cache_bytes", "prompt_order"}
    for index, (item, expected) in enumerate(zip(profiles, PROFILE_SPECS, strict=True)):
        profile = _mapping(item, f"profiles[{index}]", required)
        profile_id, cache_bytes, prompt_order = expected
        if profile["profile_id"] != profile_id:
            raise ValueError("profile order is not the fixed 8/12/16-GiB order")
        if _decimal(profile["cache_bytes"], f"profiles[{index}].cache_bytes", positive=True) != str(cache_bytes):
            raise ValueError("cache profile order or ceiling is invalid")
        order = _list(profile["prompt_order"], f"profiles[{index}].prompt_order")
        if order != list(prompt_order):
            raise ValueError("profile prompt order is not the fixed counterbalanced order")
        if any(_integer(token, "profile prompt token count") not in PROMPT_TARGETS for token in order):
            raise ValueError("profile prompt order contains an unknown token count")


def validate_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "schema", "model", "host", "prompt_source", "prompts", "sampling",
        "execution", "qualification_preflight", "profiles", "eval_case_ids",
    }
    manifest = _mapping(value, "manifest", required)
    if manifest["schema"] != SCHEMA_ID:
        raise ValueError("manifest schema is not ds4.compact-runtime-benchmark/v1")
    _validate_model(manifest["model"])
    _validate_host(manifest["host"])
    _validate_prompt_source(manifest["prompt_source"])
    _validate_prompts(manifest["prompts"])
    _validate_sampling(manifest["sampling"])
    _validate_execution(manifest["execution"])
    _validate_qualification_preflight(
        manifest["qualification_preflight"],
        model_identity=manifest["model"],
        host_identity=manifest["host"],
        runtime_identity=manifest["prompt_source"]["tokenizer_runtime"],
    )
    _validate_profiles(manifest["profiles"])
    eval_ids = _list(manifest["eval_case_ids"], "eval_case_ids")
    if eval_ids != list(EVAL_CASE_IDS):
        raise ValueError("eval_case_ids must contain the four pinned cases in order")


def manifest_sha256(value: Mapping[str, Any]) -> str:
    validate_manifest(value)
    return _sha256_bytes(_canonical_bytes(value))


def build_qualification_sequence(
    manifest: Mapping[str, Any],
    profile_id: str,
    prompt_id: str,
) -> bytes:
    """Build one strict, deterministic qualification sequence.

    This is intentionally an immutable bridge for the benchmark frontend, not
    a general JSON or text parser.  Validation is repeated at this boundary so
    callers cannot substitute profile, prompt, sampling, or prompt-byte data.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("qualification sequence manifest must be an object")
    validate_manifest(manifest)
    if type(profile_id) is not str or not profile_id:
        raise ValueError("qualification sequence profile_id must be a nonempty string")
    if type(prompt_id) is not str or not prompt_id:
        raise ValueError("qualification sequence prompt_id must be a nonempty string")

    selected_profile: tuple[str, int, tuple[int, ...]] | None = None
    for candidate in PROFILE_SPECS:
        if candidate[0] == profile_id:
            selected_profile = candidate
            break
    if selected_profile is None:
        raise ValueError(f"unknown qualification sequence profile_id: {profile_id}")
    _, cache_bytes, prompt_order = selected_profile
    try:
        prompt_order_index = prompt_order.index(
            int(prompt_id.removeprefix("native-"))
            if prompt_id.startswith("native-")
            else -1
        )
    except ValueError:
        prompt_order_index = -1
    expected_prompt_id = ""
    if prompt_order_index >= 0:
        expected_prompt_id = f"native-{prompt_order[prompt_order_index]}"
    if prompt_id != expected_prompt_id:
        raise ValueError(
            "qualification sequence prompt_id does not match profile prompt order"
        )

    prompt = next(
        (item for item in manifest["prompts"] if item["id"] == prompt_id),
        None,
    )
    if prompt is None:
        raise ValueError(f"qualification sequence prompt is absent: {prompt_id}")
    rendered = _base64(prompt["rendered_base64"], f"prompts[{prompt_id}].rendered_base64")
    recorded_size = int(
        _decimal(prompt["rendered_size_bytes"], f"prompts[{prompt_id}].rendered_size_bytes", positive=True)
    )
    if len(rendered) != recorded_size:
        raise ValueError("qualification sequence rendered prompt size mismatch")
    if len(rendered) > QUALIFICATION_SEQUENCE_MAX_BYTES:
        raise ValueError("qualification sequence input exceeds the 16 MiB bound")
    recorded_digest = _sha256(prompt["sha256"], f"prompts[{prompt_id}].sha256")
    if _sha256_bytes(rendered) != recorded_digest:
        raise ValueError("qualification sequence rendered prompt SHA-256 mismatch")

    manifest_digest = manifest_sha256(manifest)
    lines = [
        f"schema={QUALIFICATION_SEQUENCE_SCHEMA}",
        f"manifest_sha256={manifest_digest}",
        f"profile_id={profile_id}",
        f"cache_bytes={cache_bytes}",
        f"prompt_order_index={prompt_order_index}",
        f"prompt_id={prompt_id}",
        f"prompt_tokens={prompt_order[prompt_order_index]}",
        "mode=streamed",
        f"input_size_bytes={len(rendered)}",
        f"input_sha256={_sha256_bytes(rendered)}",
        f"input_base64={base64.b64encode(rendered).decode('ascii')}",
        "max_generated_tokens=512",
        "temperature=0",
        "top_k=0",
        "top_p=1",
        "min_p=0.05",
        "seed=1",
        "stop_sequences_count=0",
        "stop_token_policy=model-native",
        "repetition_count=4",
        "repetition=0:cold",
        "repetition=1:warm-1",
        "repetition=2:warm-2",
        "repetition=3:warm-3",
    ]
    if len(lines) != QUALIFICATION_SEQUENCE_LINE_COUNT:
        raise AssertionError("qualification sequence line contract drifted")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_qualification_sequence_atomic(
    path: Path | str,
    payload: bytes,
) -> None:
    """Durably create one sequence without replacing a prior output."""
    target = Path(path)
    if not isinstance(payload, bytes):
        raise ValueError("qualification sequence payload must be bytes")
    if not payload or len(payload) > QUALIFICATION_SEQUENCE_MAX_BYTES:
        raise ValueError("qualification sequence payload is outside the 16 MiB bound")
    if not target.parent.is_dir():
        raise ValueError(
            f"qualification sequence output directory does not exist: {target.parent}"
        )
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ValueError(
                f"immutable qualification sequence output already exists: {target}"
            ) from exc
        temporary.unlink()
        temporary = None
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise ValueError(
            f"cannot atomically write qualification sequence {target}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _render_seed_prefix(seed: bytes, prefix_bytes: int) -> bytes:
    return LAGUNA_TEMPLATE_PREFIX + seed[:prefix_bytes] + LAGUNA_TEMPLATE_SUFFIX


def select_rendered_prompt(seed: bytes, target: int, token_counter: TokenCounter) -> tuple[bytes, int]:
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target token count must be positive")
    if not isinstance(seed, bytes) or not seed:
        raise ValueError("benchmark seed must be nonempty bytes")

    cache: dict[int, int] = {}

    def count(prefix_bytes: int) -> int:
        if prefix_bytes not in cache:
            observed = token_counter(_render_seed_prefix(seed, prefix_bytes))
            if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
                raise ValueError("token counter returned an invalid count")
            cache[prefix_bytes] = observed
        return cache[prefix_bytes]

    low_count = count(0)
    high_count = count(len(seed))
    if target < low_count or target > high_count:
        raise ValueError(
            f"cannot construct exactly {target} native-template tokens from the immutable seed "
            f"(range {low_count}..{high_count})"
        )

    low, high = 0, len(seed)
    insertion = 0
    while low <= high:
        middle = low + (high - low) // 2
        observed = count(middle)
        if observed == target:
            return _render_seed_prefix(seed, middle), middle
        if observed < target:
            low = middle + 1
        else:
            high = middle - 1
        insertion = low

    # BPE token counts can move at a suffix boundary.  Search a deterministic
    # neighbourhood around the monotonic insertion point, but never alter or
    # pad the seed.  Small injected fixtures are searched exhaustively.
    radius = len(seed) if len(seed) <= 8192 else 4096
    start = max(0, insertion - radius)
    stop = min(len(seed), insertion + radius)
    candidates = sorted(range(start, stop + 1), key=lambda item: (abs(item - insertion), item))
    for prefix_bytes in candidates:
        if count(prefix_bytes) == target:
            return _render_seed_prefix(seed, prefix_bytes), prefix_bytes
    before = count(max(0, insertion - 1))
    after = count(min(len(seed), insertion))
    raise ValueError(
        f"immutable seed has no verified prefix of exactly {target} native-template tokens "
        f"near byte {insertion} (adjacent counts {before}, {after})"
    )


def _run_git(arguments: Sequence[str], label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"cannot run git for {label}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"{label} failed: {detail}")
    return completed.stdout.strip()


def _stat_identity(path: Path) -> tuple[os.stat_result, dict[str, str]]:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"bound file is not regular: {path}")
    return info, {
        "device": str(info.st_dev),
        "inode": str(info.st_ino),
        "size_bytes": str(info.st_size),
        "mtime_ns": str(info.st_mtime_ns),
    }


def _same_stat(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
    )


def _uint64_int(value: Any, label: str, *, positive: bool = False) -> int:
    return int(_decimal(value, label, positive=positive))


def _open_regular_nofollow(path: Path | str, label: str) -> tuple[int, os.stat_result]:
    source = Path(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ValueError(f"{label} cannot be opened without following symlinks")
    if not source.is_absolute() or ".." in source.parts:
        raise ValueError(f"{label} requires an absolute path without traversal")

    # Qualification runs on Linux, where each component is opened relative to
    # its already-verified parent descriptor.  macOS test hosts conventionally
    # spell /private/var as /var; normalize only that fixed system alias before
    # applying the same component-wise walk.
    if sys.platform == "darwin" and source.parts[:2] == ("/", "var"):
        try:
            var_target = os.readlink("/var")
        except OSError:
            var_target = ""
        if var_target == "private/var":
            source = Path("/private/var").joinpath(*source.parts[2:])

    cloexec = getattr(os, "O_CLOEXEC", 0)
    parent_flags = os.O_RDONLY | directory_flag | nofollow | cloexec
    file_flags = os.O_RDONLY | nofollow | cloexec
    parts = source.parts[1:]
    if not parts:
        raise ValueError(f"{label} must name a regular file")

    parent_descriptor = -1
    descriptor = -1
    try:
        parent_descriptor = os.open("/", os.O_RDONLY | directory_flag | cloexec)
        for component in parts[:-1]:
            previous_descriptor = parent_descriptor
            parent_descriptor = os.open(
                component,
                parent_flags,
                dir_fd=previous_descriptor,
            )
            os.close(previous_descriptor)
        descriptor = os.open(parts[-1], file_flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError(
            f"{label} cannot be opened without symlink traversal: {exc}"
        ) from exc
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError(f"{label} must be an opened regular file")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_nofollow(path: Path | str, label: str) -> bytes:
    descriptor, before = _open_regular_nofollow(path, label)
    try:
        if before.st_size <= 0 or before.st_size > QUALIFICATION_PLAN_MAX_BYTES:
            raise ValueError(f"{label} size is outside the qualification bound")
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            try:
                chunk = os.pread(
                    descriptor,
                    min(8 << 20, before.st_size - offset),
                    offset,
                )
            except OSError as exc:
                raise ValueError(f"cannot read opened {label}: {exc}") from exc
            if not chunk:
                raise ValueError(f"opened {label} ended before its recorded size")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if not _same_stat(before, after):
            raise ValueError(f"{label} identity changed while reading")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _qualification_model(value: Any) -> Mapping[str, Any]:
    required = {
        "device",
        "filename",
        "inode",
        "mtime_ns",
        "repository",
        "revision",
        "sha256",
        "size_bytes",
    }
    model = _mapping(value, "qualification plan model", required)
    if model["filename"] != MODEL_FILENAME:
        raise ValueError("qualification plan model filename is not Laguna")
    if model["repository"] != MODEL_REPOSITORY:
        raise ValueError("qualification plan model repository is not pinned")
    if model["revision"] != MODEL_REVISION:
        raise ValueError("qualification plan model revision is not pinned")
    _sha256(model["sha256"], "qualification plan model SHA-256")
    for key in ("device", "inode", "mtime_ns", "size_bytes"):
        _decimal(
            model[key],
            f"qualification plan model {key}",
            positive=True,
        )
    return model


def _ledger_safe_page_ranges(
    ledger: Any,
    *,
    page_size: int,
    model_size: int,
) -> list[tuple[int, int]]:
    if not isinstance(ledger, Mapping):
        raise ValueError("qualification plan ledger must be an object")
    if "file_size" not in ledger or "tensor_ranges" not in ledger:
        raise ValueError("qualification plan ledger lacks file size or tensor ranges")
    ledger_size_text = _decimal(
        ledger["file_size"], "qualification plan ledger file size", positive=True
    )
    if int(ledger_size_text) != model_size:
        raise ValueError("qualification plan ledger and model sizes do not match")
    tensors = _list(ledger["tensor_ranges"], "qualification plan tensor ranges")
    if not tensors:
        raise ValueError("qualification plan tensor range ledger is empty")

    source_ranges: list[tuple[int, int]] = []
    safe_ranges: list[tuple[int, int]] = []
    for index, raw in enumerate(tensors):
        if not isinstance(raw, Mapping):
            raise ValueError(f"qualification tensor range {index} must be an object")
        missing = {"class", "source_offset", "source_bytes"} - set(raw)
        if missing:
            raise ValueError(
                f"qualification tensor range {index} lacks {sorted(missing)[0]}"
            )
        if raw["class"] not in {"STATIC", "ROUTED_EXPERT"}:
            raise ValueError(f"qualification tensor range {index} has an unknown class")
        offset = _uint64_int(
            raw["source_offset"], f"qualification tensor range {index} offset"
        )
        length = _uint64_int(
            raw["source_bytes"],
            f"qualification tensor range {index} bytes",
            positive=True,
        )
        end = offset + length
        if end > UINT64_MAX or end > model_size:
            raise ValueError(f"qualification tensor range {index} exceeds the model")
        source_ranges.append((offset, end))
        safe_start = ((offset + page_size - 1) // page_size) * page_size
        safe_end = (end // page_size) * page_size
        if safe_end > safe_start:
            safe_ranges.append((safe_start, safe_end))

    source_ranges.sort()
    for index in range(1, len(source_ranges)):
        if source_ranges[index][0] < source_ranges[index - 1][1]:
            raise ValueError("qualification tensor ranges overlap")

    safe_ranges.sort()
    union: list[tuple[int, int]] = []
    for start, end in safe_ranges:
        if not union or start > union[-1][1]:
            union.append((start, end))
        elif end > union[-1][1]:
            union[-1] = (union[-1][0], end)
    return union


def _validated_advice_ranges(
    ranges: Any,
    *,
    page_size: int,
    model_size: int,
) -> tuple[list[tuple[int, int]], int]:
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page size must be a positive integer")
    if page_size & (page_size - 1):
        raise ValueError("page size must be a power of two")
    if type(model_size) is not int or model_size <= 0 or model_size > UINT64_MAX:
        raise ValueError("model size must be a positive uint64")
    records = _list(ranges, "safe page ranges")
    normalized: list[tuple[int, int]] = []
    total = 0
    previous_end = -1
    for index, value in enumerate(records):
        record = _mapping(value, f"safe page range {index}", {"bytes", "offset"})
        offset = _uint64_int(record["offset"], f"safe page range {index} offset")
        length = _uint64_int(
            record["bytes"], f"safe page range {index} bytes", positive=True
        )
        end = offset + length
        if offset % page_size or length % page_size:
            raise ValueError(f"safe page range {index} is not page aligned")
        if end > UINT64_MAX or end > model_size:
            raise ValueError(f"safe page range {index} exceeds the model")
        if offset <= previous_end:
            raise ValueError("safe page ranges are not a normalized union")
        total += length
        if total > UINT64_MAX:
            raise ValueError("safe page range coverage overflows uint64")
        normalized.append((offset, length))
        previous_end = end
    return normalized, total


def _validate_qualification_plan(
    value: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], int, int, int]:
    required = {
        "allocation",
        "ledger",
        "ledger_sha256",
        "model",
        "page_cache",
        "schema",
    }
    plan = _mapping(value, "qualification plan", required)
    if plan["schema"] != "ds4.laguna.qualification-plan/v1":
        raise ValueError("qualification plan schema is not supported")
    if not isinstance(plan["allocation"], Mapping):
        raise ValueError("qualification plan allocation must be an object")

    ledger_digest = _sha256(
        plan["ledger_sha256"], "qualification plan ledger SHA-256"
    )
    observed_ledger_digest = _sha256_bytes(canonical_json_bytes(plan["ledger"]))
    if observed_ledger_digest != ledger_digest:
        raise ValueError("qualification plan ledger digest mismatch")

    model = _qualification_model(plan["model"])
    model_size = _uint64_int(
        model["size_bytes"], "qualification plan model size", positive=True
    )
    page_cache = _mapping(
        plan["page_cache"],
        "qualification plan page cache",
        {
            "eligible_unique_bytes",
            "mapped_page_bytes",
            "page_size",
            "ranges",
            "unavoidable_bytes",
        },
    )
    page_size = _uint64_int(
        page_cache["page_size"], "qualification plan page size", positive=True
    )
    if page_size & (page_size - 1):
        raise ValueError("qualification plan page size must be a power of two")
    mapped_page_bytes = _uint64_int(
        page_cache["mapped_page_bytes"],
        "qualification plan mapped page bytes",
        positive=True,
    )
    expected_mapped = ((model_size + page_size - 1) // page_size) * page_size
    if expected_mapped > UINT64_MAX or mapped_page_bytes != expected_mapped:
        raise ValueError("qualification plan mapped page coverage is inconsistent")

    normalized, covered_bytes = _validated_advice_ranges(
        page_cache["ranges"], page_size=page_size, model_size=model_size
    )
    eligible_bytes = _uint64_int(
        page_cache["eligible_unique_bytes"],
        "qualification plan eligible bytes",
    )
    if eligible_bytes != covered_bytes:
        raise ValueError("qualification plan eligible coverage does not match its ranges")
    if eligible_bytes > mapped_page_bytes:
        raise ValueError("qualification plan eligible coverage exceeds the mapping")
    unavoidable_bytes = mapped_page_bytes - eligible_bytes
    recorded_unavoidable = _uint64_int(
        page_cache["unavoidable_bytes"],
        "qualification plan unavoidable bytes",
    )
    if recorded_unavoidable != unavoidable_bytes:
        raise ValueError("qualification plan unavoidable residency is inconsistent")
    if unavoidable_bytes > UNAVOIDABLE_RESIDENCY_LIMIT_BYTES:
        raise ValueError("unavoidable residency exceeds the 2 GiB limit")

    expected_union = _ledger_safe_page_ranges(
        plan["ledger"], page_size=page_size, model_size=model_size
    )
    observed_union = [(offset, offset + length) for offset, length in normalized]
    if observed_union != expected_union:
        raise ValueError("page cache does not match the ledger safe tensor union")
    ranges = _list(page_cache["ranges"], "qualification plan safe page ranges")
    return model, ranges, page_size, model_size, unavoidable_bytes


def _posix_fadvise_dontneed(descriptor: int, offset: int, length: int) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise OSError(errno.ENOSYS, "posix_fadvise DONTNEED is unavailable")
    os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)


def advise_safe_page_ranges(
    descriptor: int,
    ranges: Any,
    *,
    page_size: int,
    model_size: int,
    advise: Callable[[int, int, int], None] = _posix_fadvise_dontneed,
) -> dict[str, Any]:
    """Advise every validated safe range and retain exact failure accounting."""
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("cold-preparation descriptor is invalid")
    normalized, eligible_bytes = _validated_advice_ranges(
        ranges, page_size=page_size, model_size=model_size
    )
    report: dict[str, Any] = {
        "eligible_calls": len(normalized),
        "eligible_bytes": str(eligible_bytes),
        "attempted_calls": 0,
        "attempted_bytes": "0",
        "successful_calls": 0,
        "successful_bytes": "0",
        "failed_calls": 0,
        "failed_bytes": "0",
        "errno_buckets": {},
    }
    attempted_bytes = 0
    successful_bytes = 0
    failed_bytes = 0
    for offset, length in normalized:
        report["attempted_calls"] += 1
        attempted_bytes += length
        try:
            advise(descriptor, offset, length)
        except OSError as exc:
            report["failed_calls"] += 1
            failed_bytes += length
            bucket = errno.errorcode.get(exc.errno, "UNKNOWN")
            buckets = report["errno_buckets"]
            buckets[bucket] = buckets.get(bucket, 0) + 1
        else:
            report["successful_calls"] += 1
            successful_bytes += length
    report["attempted_bytes"] = str(attempted_bytes)
    report["successful_bytes"] = str(successful_bytes)
    report["failed_bytes"] = str(failed_bytes)
    return report


def _dispose_resident_eligible_pages(
    descriptor: int,
    file_size: int,
    page_size: int,
    resident_bits: bytes,
    eligible_ranges: Sequence[tuple[int, int]],
    *,
    advise: Callable[[int, int, int], None] = _posix_fadvise_dontneed,
    madvise_advice: int | None = None,
) -> dict[str, Any]:
    """Dispose only sampled-hot eligible runs through the retained mapping."""
    def counters() -> dict[str, Any]:
        return {
            "attempted_calls": 0,
            "attempted_bytes": "0",
            "successful_calls": 0,
            "successful_bytes": "0",
            "failed_calls": 0,
            "failed_bytes": "0",
            "errno_buckets": {},
        }

    report: dict[str, Any] = {
        "random_access_madvise": counters(),
        "mapping_touch_pages": 0,
        "mapping_touch_bytes": "0",
        "madvise": counters(),
        "fadvise": counters(),
    }
    if not hasattr(mmap, "MADV_RANDOM") or not hasattr(mmap, "MADV_DONTNEED"):
        raise OSError(
            errno.ENOSYS,
            "mmap MADV_RANDOM/DONTNEED is unavailable",
        )
    if madvise_advice is None:
        madvise_advice = mmap.MADV_DONTNEED
    if type(madvise_advice) is not int or madvise_advice < 0:
        raise ValueError("residual disposal mmap advice is invalid")
    if len(resident_bits) != (file_size + page_size - 1) // page_size:
        raise ValueError("residual disposal residency vector is invalid")
    mapping = mmap.mmap(
        descriptor,
        file_size,
        flags=mmap.MAP_PRIVATE,
        prot=mmap.PROT_READ,
    )
    try:
        random_access = report["random_access_madvise"]
        random_access["attempted_calls"] = 1
        random_access["attempted_bytes"] = str(file_size)
        try:
            mapping.madvise(mmap.MADV_RANDOM, 0, file_size)
        except OSError as exc:
            random_access["failed_calls"] = 1
            random_access["failed_bytes"] = str(file_size)
            bucket = errno.errorcode.get(exc.errno, "UNKNOWN")
            random_access["errno_buckets"][bucket] = 1
        else:
            random_access["successful_calls"] = 1
            random_access["successful_bytes"] = str(file_size)
        for offset, length in eligible_ranges:
            first_page = offset // page_size
            final_page = (offset + length) // page_size
            page = first_page
            while page < final_page:
                if resident_bits[page] == 0:
                    page += 1
                    continue
                run_first = page
                while page < final_page and resident_bits[page] != 0:
                    _ = mapping[page * page_size]
                    report["mapping_touch_pages"] += 1
                    page += 1
                run_bytes = (page - run_first) * page_size
                for label, operation in (
                    (
                        "madvise",
                        lambda: mapping.madvise(
                            madvise_advice,
                            run_first * page_size,
                            run_bytes,
                        ),
                    ),
                    (
                        "fadvise",
                        lambda: advise(
                            descriptor,
                            run_first * page_size,
                            run_bytes,
                        ),
                    ),
                ):
                    specific = report[label]
                    specific["attempted_calls"] += 1
                    specific["attempted_bytes"] = str(
                        int(specific["attempted_bytes"]) + run_bytes
                    )
                    try:
                        operation()
                    except OSError as exc:
                        specific["failed_calls"] += 1
                        specific["failed_bytes"] = str(
                            int(specific["failed_bytes"]) + run_bytes
                        )
                        bucket = errno.errorcode.get(exc.errno, "UNKNOWN")
                        buckets = specific["errno_buckets"]
                        buckets[bucket] = buckets.get(bucket, 0) + 1
                    else:
                        specific["successful_calls"] += 1
                        specific["successful_bytes"] = str(
                            int(specific["successful_bytes"]) + run_bytes
                        )
    finally:
        mapping.close()
    report["mapping_touch_bytes"] = str(
        report["mapping_touch_pages"] * page_size
    )
    return report


def sample_exact_inode_residency(
    descriptor: int,
    file_size: int,
    page_size: int,
) -> bytes:
    """Return one location-preserving residency bit per mapped inode page."""
    if page_size != mmap.PAGESIZE:
        raise ValueError("qualification plan page size differs from the host page size")
    if file_size <= 0:
        raise ValueError("cannot sample an empty model inode")
    page_count = (file_size + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * page_count)()
    mapping = mmap.mmap(
        descriptor,
        file_size,
        flags=mmap.MAP_PRIVATE,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    anchor = None
    try:
        anchor = ctypes.c_char.from_buffer(mapping)
        libc = ctypes.CDLL(None, use_errno=True)
        mincore = libc.mincore
        mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        mincore.restype = ctypes.c_int
        ctypes.set_errno(0)
        if mincore(ctypes.addressof(anchor), file_size, vector) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        return bytes(value & 1 for value in vector)
    finally:
        del anchor
        mapping.close()


def _model_identity_matches(
    recorded: Mapping[str, Any],
    observed: os.stat_result,
) -> None:
    for key, actual in (
        ("device", observed.st_dev),
        ("inode", observed.st_ino),
        ("size_bytes", observed.st_size),
        ("mtime_ns", observed.st_mtime_ns),
    ):
        if recorded[key] != str(actual):
            raise ValueError(f"model identity mismatch: {key}")


def _load_cold_preparation_plan(
    plan_path: Path | str,
    expected_plan_sha256: str,
) -> tuple[
    Mapping[str, Any], Mapping[str, Any], list[Any], int, int, int, str
]:
    expected_digest = _sha256(
        expected_plan_sha256, "expected qualification plan SHA-256"
    )
    plan_bytes = _read_regular_nofollow(plan_path, "qualification plan")
    observed_digest = _sha256_bytes(plan_bytes)
    if observed_digest != expected_digest:
        raise ValueError("qualification plan digest mismatch")
    try:
        plan_text = plan_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("qualification plan is not UTF-8") from exc
    plan = loads_strict(plan_text)
    if not isinstance(plan, Mapping):
        raise ValueError("qualification plan root must be an object")
    if canonical_json_bytes(plan) != plan_bytes:
        raise ValueError("qualification plan bytes are not canonical")
    model, ranges, page_size, model_size, unavoidable_bytes = (
        _validate_qualification_plan(plan)
    )
    if page_size != mmap.PAGESIZE:
        raise ValueError("qualification plan page size differs from the host page size")
    return (
        plan, model, ranges, page_size, model_size, unavoidable_bytes,
        observed_digest,
    )


def _cold_prepare_open_descriptor(
    descriptor: int,
    loaded_plan: tuple[
        Mapping[str, Any], Mapping[str, Any], list[Any], int, int, int, str
    ],
    *,
    advise: Callable[[int, int, int], None],
    sample_residency: Callable[[int, int, int], int | bytes],
) -> dict[str, Any]:
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("cold-preparation descriptor is invalid")
    plan, model, ranges, page_size, model_size, unavoidable_bytes, \
        observed_digest = loaded_plan
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("cold-preparation descriptor is not open") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("cold-preparation descriptor is not a regular file")
    original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)

    try:
        _model_identity_matches(model, before)
        report = advise_safe_page_ranges(
            descriptor,
            ranges,
            page_size=page_size,
            model_size=model_size,
            advise=advise,
        )
        after_advice = os.fstat(descriptor)
        if not _same_stat(before, after_advice):
            raise ValueError("model identity changed during cold preparation")
        if report["failed_calls"] != 0:
            raise ValueError(
                "cold-preparation advice failed; evidence is invalid: "
                f"{report['errno_buckets']}"
            )
        page_count = (model_size + page_size - 1) // page_size
        normalized_ranges, _ = _validated_advice_ranges(
            ranges, page_size=page_size, model_size=model_size
        )

        def exact_residency_bits() -> bytes:
            sample = sample_residency(descriptor, model_size, page_size)
            if type(sample) is int:
                # Legacy synthetic fixtures may state that every page is cold.
                # Any nonzero scalar is location-ambiguous and cannot qualify.
                if sample != 0:
                    raise ValueError(
                        "nonzero exact-inode residency requires location-aware page bits"
                    )
                return bytes(page_count)
            if isinstance(sample, (bytes, bytearray, memoryview)):
                bits = bytes(sample)
                if len(bits) != page_count or any(bit not in (0, 1) for bit in bits):
                    raise ValueError("exact-inode residency page bits are invalid")
                return bits
            raise ValueError("exact-inode residency sample is invalid")

        def eligible_resident_pages(bits: bytes) -> int:
            view = memoryview(bits)
            return sum(
                sum(view[offset // page_size:(offset + length) // page_size])
                for offset, length in normalized_ranges
            )

        resident_bits = exact_residency_bits()
        after_initial_sample = os.fstat(descriptor)
        if not _same_stat(before, after_initial_sample):
            raise ValueError("model identity changed during residency measurement")
        initial_resident_eligible_pages = eligible_resident_pages(resident_bits)
        empty_advice_report = {
            "attempted_calls": 0,
            "attempted_bytes": "0",
            "successful_calls": 0,
            "successful_bytes": "0",
            "failed_calls": 0,
            "failed_bytes": "0",
            "errno_buckets": {},
        }
        residual_report: dict[str, Any] = {
            "initial_resident_eligible_pages": initial_resident_eligible_pages,
            "pageout_retry_pages": 0,
            "residency_samples": 1,
            "mapping_touch_pages": 0,
            "mapping_touch_bytes": "0",
            "random_access_madvise": {
                **empty_advice_report, "errno_buckets": {}
            },
            "madvise": {**empty_advice_report, "errno_buckets": {}},
            "fadvise": {**empty_advice_report, "errno_buckets": {}},
        }

        def merge_disposal_report(source: Mapping[str, Any]) -> None:
            for key in ("mapping_touch_pages",):
                total = int(residual_report[key]) + int(source[key])
                if total > UINT64_MAX:
                    raise ValueError("residual disposal accounting overflow")
                residual_report[key] = total
            total_touch_bytes = (
                int(residual_report["mapping_touch_bytes"])
                + int(source["mapping_touch_bytes"])
            )
            if total_touch_bytes > UINT64_MAX:
                raise ValueError("residual disposal accounting overflow")
            residual_report["mapping_touch_bytes"] = str(total_touch_bytes)
            for label in ("random_access_madvise", "madvise", "fadvise"):
                target = residual_report[label]
                observed = source[label]
                for key in (
                    "attempted_calls", "successful_calls", "failed_calls"
                ):
                    total = int(target[key]) + int(observed[key])
                    if total > UINT64_MAX:
                        raise ValueError("residual disposal accounting overflow")
                    target[key] = total
                for key in (
                    "attempted_bytes", "successful_bytes", "failed_bytes"
                ):
                    total = int(target[key]) + int(observed[key])
                    if total > UINT64_MAX:
                        raise ValueError("residual disposal accounting overflow")
                    target[key] = str(total)
                for bucket, count in observed["errno_buckets"].items():
                    target["errno_buckets"][bucket] = (
                        target["errno_buckets"].get(bucket, 0) + count
                    )

        if initial_resident_eligible_pages != 0:
            try:
                residual = _dispose_resident_eligible_pages(
                    descriptor,
                    model_size,
                    page_size,
                    resident_bits,
                    normalized_ranges,
                    advise=advise,
                )
            except OSError as exc:
                raise ValueError(
                    "an eligible page remains resident after cold preparation"
                ) from exc
            merge_disposal_report(residual)
            residual_report["residency_samples"] = 2
            if (residual_report["random_access_madvise"]["failed_calls"] != 0 or
                    residual_report["madvise"]["failed_calls"] != 0 or
                    residual_report["fadvise"]["failed_calls"] != 0):
                raise ValueError(
                    "residual eligible-page disposal failed; evidence is invalid"
                )
            resident_bits = exact_residency_bits()
            after_dontneed_sample = os.fstat(descriptor)
            if not _same_stat(before, after_dontneed_sample):
                raise ValueError("model identity changed during residency measurement")
            pageout_retry_pages = eligible_resident_pages(resident_bits)
            if pageout_retry_pages != 0:
                if sys.platform != "linux":
                    raise ValueError(
                        "an eligible page remains resident; "
                        "Linux MADV_PAGEOUT disposal is unavailable"
                    )
                try:
                    pageout = _dispose_resident_eligible_pages(
                        descriptor,
                        model_size,
                        page_size,
                        resident_bits,
                        normalized_ranges,
                        advise=advise,
                        madvise_advice=LINUX_MADV_PAGEOUT,
                    )
                except OSError as exc:
                    raise ValueError(
                        "an eligible page remains resident after cold preparation"
                    ) from exc
                merge_disposal_report(pageout)
                residual_report["pageout_retry_pages"] = pageout_retry_pages
                residual_report["residency_samples"] = 3
                if (residual_report["random_access_madvise"]["failed_calls"] != 0 or
                        residual_report["madvise"]["failed_calls"] != 0 or
                        residual_report["fadvise"]["failed_calls"] != 0):
                    raise ValueError(
                        "residual eligible-page disposal failed; evidence is invalid"
                    )
                resident_bits = exact_residency_bits()
        if eligible_resident_pages(resident_bits) != 0:
            raise ValueError("an eligible page remains resident after cold preparation")
        resident_bytes = sum(resident_bits) * page_size
        after_sample = os.fstat(descriptor)
        if not _same_stat(before, after_sample):
            raise ValueError("model identity changed during residency measurement")
        _model_identity_matches(model, after_sample)
        result = {
            "plan_sha256": observed_digest,
            "ledger_sha256": plan["ledger_sha256"],
            "model_identity": dict(model),
            "page_size": str(page_size),
            "eligible_ranges": [
                {"offset": str(offset), "bytes": str(length)}
                for offset, length in normalized_ranges
            ],
            **report,
            "residual_disposal": residual_report,
            "resident_bytes_after": str(resident_bytes),
            "unavoidable_bytes": str(unavoidable_bytes),
        }
        return result
    finally:
        os.lseek(descriptor, original_offset, os.SEEK_SET)


def cold_prepare_descriptor_from_plan(
    descriptor: int,
    plan_path: Path | str,
    expected_plan_sha256: str,
    *,
    advise: Callable[[int, int, int], None] = _posix_fadvise_dontneed,
    sample_residency: Callable[[int, int, int], int | bytes] = (
        sample_exact_inode_residency
    ),
) -> dict[str, Any]:
    """Cold-prepare a caller-owned exact model descriptor without closing it."""
    loaded_plan = _load_cold_preparation_plan(
        plan_path, expected_plan_sha256
    )
    return _cold_prepare_open_descriptor(
        descriptor,
        loaded_plan,
        advise=advise,
        sample_residency=sample_residency,
    )


def cold_prepare_from_plan(
    model_path: Path | str,
    plan_path: Path | str,
    expected_plan_sha256: str,
    *,
    advise: Callable[[int, int, int], None] = _posix_fadvise_dontneed,
    sample_residency: Callable[[int, int, int], int | bytes] = (
        sample_exact_inode_residency
    ),
) -> dict[str, Any]:
    """Open and cold-prepare only the plan's descriptor-bound safe union."""
    loaded_plan = _load_cold_preparation_plan(
        plan_path, expected_plan_sha256
    )
    model = loaded_plan[1]
    descriptor, _ = _open_regular_nofollow(model_path, "model")
    try:
        if Path(model_path).name != model["filename"]:
            raise ValueError("opened model filename does not match the plan")
        return _cold_prepare_open_descriptor(
            descriptor,
            loaded_plan,
            advise=advise,
            sample_residency=sample_residency,
        )
    finally:
        os.close(descriptor)


def _nvml_inventory_by_pid(
    value: Any,
    label: str,
    *,
    gpu_uuid: str,
) -> dict[int, int]:
    inventory = _mapping(value, label, {"api", "gpu_uuid", "processes"})
    if inventory["api"] != NVML_COMPUTE_API:
        raise ValueError(f"{label} NVML API/version does not match {NVML_COMPUTE_API}")
    if inventory["gpu_uuid"] != gpu_uuid:
        raise ValueError(f"{label} GPU UUID does not match the qualification device")
    processes = _list(inventory["processes"], f"{label} NVML processes")
    by_pid: dict[int, int] = {}
    for index, raw in enumerate(processes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} NVML process {index} must be an object")
        if "used_gpu_memory_bytes" not in raw:
            raise ValueError(f"{label} NVML process {index} usage is missing")
        process = _mapping(
            raw,
            f"{label} NVML process {index}",
            {"pid", "used_gpu_memory_bytes"},
        )
        pid = _integer(process["pid"], f"{label} NVML process {index} PID")
        if pid <= 0 or pid > 0xFFFFFFFF:
            raise ValueError(f"{label} NVML process {index} PID is invalid")
        try:
            usage = _uint64_int(
                process["used_gpu_memory_bytes"],
                f"{label} NVML process {index} usage",
            )
        except ValueError as exc:
            raise ValueError(
                f"{label} NVML process {index} usage representation is invalid"
            ) from exc
        if usage == UINT64_MAX:
            raise ValueError(f"{label} NVML process {index} usage is unknown")
        if pid in by_pid:
            raise ValueError(f"{label} contains duplicate NVML process PID {pid}")
        by_pid[pid] = usage
    return by_pid


def validate_nvml_checkpoint(
    frozen_inventory: Any,
    before_inventory: Any,
    after_inventory: Any,
    *,
    ds4_pid: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    """Validate one process-scoped NVML v2 checkpoint without baseline math."""
    if type(ds4_pid) is not int or ds4_pid <= 0 or ds4_pid > 0xFFFFFFFF:
        raise ValueError("DS4 PID is invalid")
    if not isinstance(gpu_uuid, str) or not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("qualification GPU UUID is invalid")
    frozen = _nvml_inventory_by_pid(
        frozen_inventory, "frozen pre-child inventory", gpu_uuid=gpu_uuid
    )
    before = _nvml_inventory_by_pid(
        before_inventory, "checkpoint-before inventory", gpu_uuid=gpu_uuid
    )
    after = _nvml_inventory_by_pid(
        after_inventory, "checkpoint-after inventory", gpu_uuid=gpu_uuid
    )
    if ds4_pid in frozen:
        raise ValueError("DS4 PID already exists in the frozen pre-child inventory")
    if ds4_pid not in before or ds4_pid not in after:
        raise ValueError("DS4 NVML process usage is missing")
    before_other = {pid: usage for pid, usage in before.items() if pid != ds4_pid}
    after_other = {pid: usage for pid, usage in after.items() if pid != ds4_pid}
    if before_other != frozen or after_other != frozen:
        raise ValueError("unrelated NVML process inventory changed")
    if before[ds4_pid] != after[ds4_pid]:
        raise ValueError("DS4 NVML process usage changed during the checkpoint")
    return {
        "api": NVML_COMPUTE_API,
        "gpu_uuid": gpu_uuid,
        "ds4_pid": ds4_pid,
        "ds4_process_bytes": str(after[ds4_pid]),
    }


def _warm_stability_sample(value: Any, label: str) -> tuple[dict[str, int], int, int, int, int]:
    required = {
        "owned_category_current_bytes",
        "cache_acquire_hits_before",
        "cache_acquire_hits_after",
        "model_file_read_bytes_before",
        "model_file_read_bytes_after",
    }
    sample = _mapping(value, label, required)
    raw_categories = _mapping(
        sample["owned_category_current_bytes"],
        f"{label} owned categories",
        set(RUNTIME_OWNED_CATEGORY_NAMES),
    )
    categories = {
        name: _uint64_int(raw_categories[name], f"{label} {name}")
        for name in RUNTIME_OWNED_CATEGORY_NAMES
    }
    hit_before = _uint64_int(
        sample["cache_acquire_hits_before"], f"{label} hits before"
    )
    hit_after = _uint64_int(
        sample["cache_acquire_hits_after"], f"{label} hits after"
    )
    read_before = _uint64_int(
        sample["model_file_read_bytes_before"], f"{label} reads before"
    )
    read_after = _uint64_int(
        sample["model_file_read_bytes_after"], f"{label} reads after"
    )
    if hit_after < hit_before or read_after < read_before:
        raise ValueError(f"{label} cumulative cache counters regressed")
    return categories, hit_before, hit_after, read_before, read_after


def validate_warm_stability_samples(
    cold_sample: Any,
    warm_samples: Any,
) -> None:
    """Validate the bounded, same-process warm-repetition evidence."""
    warm = _list(warm_samples, "warm stability samples")
    if len(warm) != WARM_STABILITY_REPETITIONS:
        raise ValueError("warm stability requires exactly three repetitions")
    _, cold_hit_before, cold_hit_after, cold_read_before, cold_read_after = \
        _warm_stability_sample(cold_sample, "cold sample")
    cold_read_delta = cold_read_after - cold_read_before
    if cold_read_delta == 0:
        raise ValueError("cold sample must read routed model-file bytes")
    previous_hits = cold_hit_after
    previous_reads = cold_read_after
    parsed: list[tuple[dict[str, int], int, int, int, int]] = []
    for index, raw in enumerate(warm):
        current = _warm_stability_sample(raw, f"warm sample {index}")
        categories, hit_before, hit_after, read_before, read_after = current
        if hit_before != previous_hits or read_before != previous_reads:
            raise ValueError("warm cumulative counters are not contiguous")
        if hit_after == hit_before:
            raise ValueError("every warm repetition must increase cache hits")
        if read_after - read_before > cold_read_delta:
            raise ValueError("a warm repetition read more routed bytes than cold")
        previous_hits = hit_after
        previous_reads = read_after
        parsed.append(current)

    baseline = parsed[0][0]
    for name in RUNTIME_OWNED_CATEGORY_NAMES:
        values = [sample[0][name] for sample in parsed]
        if any(
            abs(value - baseline[name]) > WARM_OWNED_CATEGORY_DRIFT_LIMIT_BYTES
            for value in values
        ):
            raise ValueError(f"warm owned category {name} exceeded its drift limit")
        if any(values[index] > values[index - 1] for index in range(1, len(values))) \
                and all(values[index] >= values[index - 1] for index in range(1, len(values))):
            raise ValueError(f"warm owned category {name} grew monotonically")


def _validate_cold_preparation(value: Any) -> Mapping[str, Any]:
    required = {
        "plan_sha256",
        "ledger_sha256",
        "model_identity",
        "page_size",
        "eligible_ranges",
        "eligible_calls",
        "eligible_bytes",
        "attempted_calls",
        "attempted_bytes",
        "successful_calls",
        "successful_bytes",
        "failed_calls",
        "failed_bytes",
        "errno_buckets",
        "residual_disposal",
        "resident_bytes_after",
        "unavoidable_bytes",
    }
    cold = _mapping(value, "qualification preflight cold preparation", required)
    _sha256(cold["plan_sha256"], "cold-preparation plan SHA-256")
    _sha256(cold["ledger_sha256"], "cold-preparation ledger SHA-256")
    model = _qualification_model(cold["model_identity"])
    model_size = _uint64_int(
        model["size_bytes"], "cold-preparation model size", positive=True
    )
    page_size = _uint64_int(
        cold["page_size"], "cold-preparation page size", positive=True
    )
    if page_size & (page_size - 1):
        raise ValueError("cold-preparation page size must be a power of two")
    normalized, eligible_bytes = _validated_advice_ranges(
        cold["eligible_ranges"], page_size=page_size, model_size=model_size
    )
    expected_ranges = [
        {"offset": str(offset), "bytes": str(length)}
        for offset, length in normalized
    ]
    if cold["eligible_ranges"] != expected_ranges:
        raise ValueError("cold-preparation eligible ranges are not canonical")

    counts = {
        key: _integer(cold[key], f"cold-preparation {key}")
        for key in (
            "eligible_calls",
            "attempted_calls",
            "successful_calls",
            "failed_calls",
        )
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError("cold-preparation call counts must be nonnegative")
    byte_counts = {
        key: _uint64_int(cold[key], f"cold-preparation {key}")
        for key in (
            "eligible_bytes",
            "attempted_bytes",
            "successful_bytes",
            "failed_bytes",
        )
    }
    if counts["eligible_calls"] != len(normalized):
        raise ValueError("cold-preparation eligible call count is inconsistent")
    if byte_counts["eligible_bytes"] != eligible_bytes:
        raise ValueError("cold-preparation eligible byte count is inconsistent")
    if (
        counts["attempted_calls"] != counts["eligible_calls"]
        or byte_counts["attempted_bytes"] != eligible_bytes
        or counts["successful_calls"] + counts["failed_calls"]
        != counts["attempted_calls"]
        or byte_counts["successful_bytes"] + byte_counts["failed_bytes"]
        != byte_counts["attempted_bytes"]
    ):
        raise ValueError("cold-preparation attempt accounting is inconsistent")
    if counts["failed_calls"] != 0 or byte_counts["failed_bytes"] != 0:
        raise ValueError("cold-preparation evidence contains an advice failure")
    if (
        counts["successful_calls"] != counts["eligible_calls"]
        or byte_counts["successful_bytes"] != eligible_bytes
    ):
        raise ValueError("cold-preparation coverage is incomplete")
    buckets = _mapping(cold["errno_buckets"], "cold-preparation errno buckets", set())
    if buckets:
        raise ValueError("successful cold-preparation evidence has errno buckets")

    residual = _mapping(
        cold["residual_disposal"],
        "cold-preparation residual disposal",
        {
            "initial_resident_eligible_pages",
            "pageout_retry_pages",
            "residency_samples",
            "mapping_touch_pages",
            "mapping_touch_bytes",
            "random_access_madvise",
            "madvise",
            "fadvise",
        },
    )
    initial_pages = _integer(
        residual["initial_resident_eligible_pages"],
        "cold-preparation initial resident eligible pages",
    )
    pageout_retry_pages = _integer(
        residual["pageout_retry_pages"],
        "cold-preparation PAGEOUT retry pages",
    )
    residency_samples = _integer(
        residual["residency_samples"],
        "cold-preparation residency sample count",
    )
    touch_pages = _integer(
        residual["mapping_touch_pages"],
        "cold-preparation mapping-touch pages",
    )
    touch_bytes = _uint64_int(
        residual["mapping_touch_bytes"],
        "cold-preparation mapping-touch bytes",
    )
    if (initial_pages < 0 or pageout_retry_pages < 0 or
            residency_samples not in (1, 2, 3) or touch_pages < 0):
        raise ValueError("cold-preparation residual disposal counts are invalid")
    if (initial_pages > eligible_bytes // page_size or
            pageout_retry_pages > initial_pages):
        raise ValueError("cold-preparation residual disposal exceeds eligible pages")

    def validate_residual_advice(label: str) -> tuple[int, int]:
        advice = _mapping(
            residual[label],
            f"cold-preparation residual {label}",
            {
                "attempted_calls", "attempted_bytes", "successful_calls",
                "successful_bytes", "failed_calls", "failed_bytes",
                "errno_buckets",
            },
        )
        advice_counts = {
            key: _integer(advice[key], f"cold-preparation residual {label} {key}")
            for key in ("attempted_calls", "successful_calls", "failed_calls")
        }
        advice_bytes = {
            key: _uint64_int(
                advice[key], f"cold-preparation residual {label} {key}"
            )
            for key in ("attempted_bytes", "successful_bytes", "failed_bytes")
        }
        if any(count < 0 for count in advice_counts.values()) or (
            advice_counts["successful_calls"] + advice_counts["failed_calls"]
            != advice_counts["attempted_calls"]
        ) or (
            advice_bytes["successful_bytes"] + advice_bytes["failed_bytes"]
            != advice_bytes["attempted_bytes"]
        ):
            raise ValueError(
                f"cold-preparation residual {label} accounting is inconsistent"
            )
        buckets = _mapping(
            advice["errno_buckets"],
            f"cold-preparation residual {label} errno buckets",
            set(),
        )
        if advice_counts["failed_calls"] != 0 or advice_bytes["failed_bytes"] != 0:
            raise ValueError(f"cold-preparation residual {label} failed")
        if buckets:
            raise ValueError(
                f"successful cold-preparation residual {label} has errno buckets"
            )
        return advice_counts["attempted_calls"], advice_bytes["attempted_bytes"]

    random_calls, random_bytes = validate_residual_advice(
        "random_access_madvise"
    )
    madvise_calls, madvise_bytes = validate_residual_advice("madvise")
    fadvise_calls, fadvise_bytes = validate_residual_advice("fadvise")
    if initial_pages == 0:
        if (pageout_retry_pages != 0 or residency_samples != 1 or
                touch_pages != 0 or touch_bytes != 0 or
                random_calls != 0 or random_bytes != 0 or
                madvise_calls != 0 or madvise_bytes != 0 or
                fadvise_calls != 0 or fadvise_bytes != 0):
            raise ValueError(
                "cold-preparation unnecessary residual disposal was attempted"
            )
    elif (
        residency_samples != (3 if pageout_retry_pages != 0 else 2)
        or touch_pages != initial_pages + pageout_retry_pages
        or touch_bytes != touch_pages * page_size
        or random_calls != 1 + int(pageout_retry_pages != 0)
        or model_size > UINT64_MAX // random_calls
        or random_bytes != model_size * random_calls
        or madvise_calls < 1 + int(pageout_retry_pages != 0)
        or madvise_calls > touch_pages
        or fadvise_calls != madvise_calls
        or madvise_bytes != touch_bytes
        or fadvise_bytes != touch_bytes
    ):
        raise ValueError("cold-preparation residual disposal is incomplete")

    mapped_bytes = ((model_size + page_size - 1) // page_size) * page_size
    unavoidable_bytes = _uint64_int(
        cold["unavoidable_bytes"], "cold-preparation unavoidable bytes"
    )
    if unavoidable_bytes != mapped_bytes - eligible_bytes:
        raise ValueError("cold-preparation unavoidable byte count is inconsistent")
    if unavoidable_bytes > UNAVOIDABLE_RESIDENCY_LIMIT_BYTES:
        raise ValueError("cold-preparation unavoidable residency exceeds the 2 GiB limit")
    resident_bytes = _uint64_int(
        cold["resident_bytes_after"], "cold-preparation resident bytes"
    )
    if resident_bytes > unavoidable_bytes:
        raise ValueError("cold-preparation resident bytes exceed unavoidable coverage")
    return cold


def _canonical_nvml_inventory(
    value: Any,
    label: str,
    *,
    gpu_uuid: str,
) -> dict[str, Any]:
    by_pid = _nvml_inventory_by_pid(value, label, gpu_uuid=gpu_uuid)
    return {
        "api": NVML_COMPUTE_API,
        "gpu_uuid": gpu_uuid,
        "processes": [
            {"pid": pid, "used_gpu_memory_bytes": str(by_pid[pid])}
            for pid in sorted(by_pid)
        ],
    }


def collect_nvml_pre_child(gpu_uuid: str) -> dict[str, Any]:
    """Collect one process inventory through NVML without touching CUDA."""
    if not isinstance(gpu_uuid, str) or not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("pre-child NVML GPU UUID is invalid")
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        raise ValueError(f"cannot load libnvidia-ml.so.1: {exc}") from exc
    try:
        init = nvml.nvmlInit_v2
        shutdown = nvml.nvmlShutdown
        get_version = nvml.nvmlSystemGetNVMLVersion
        get_device = nvml.nvmlDeviceGetHandleByUUID
        get_device_uuid = nvml.nvmlDeviceGetUUID
        get_processes = nvml.nvmlDeviceGetComputeRunningProcesses_v2
    except AttributeError as exc:
        raise ValueError(f"NVML lacks the required v2 process API: {exc}") from exc

    init.argtypes = []
    init.restype = ctypes.c_int
    shutdown.argtypes = []
    shutdown.restype = ctypes.c_int
    get_version.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    get_version.restype = ctypes.c_int
    get_device.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    get_device.restype = ctypes.c_int
    get_device_uuid.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    get_device_uuid.restype = ctypes.c_int
    get_processes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_NvmlProcessInfoV2),
    ]
    get_processes.restype = ctypes.c_int

    initialized = False
    primary_error: BaseException | None = None
    try:
        rc = init()
        if rc != NVML_SUCCESS:
            raise ValueError(f"nvmlInit_v2 failed with NVML status {rc}")
        initialized = True

        version_buffer = ctypes.create_string_buffer(NVML_VERSION_BUFFER_SIZE)
        rc = get_version(version_buffer, ctypes.c_uint(len(version_buffer)))
        if rc != NVML_SUCCESS:
            raise ValueError(
                f"nvmlSystemGetNVMLVersion failed with NVML status {rc}"
            )
        try:
            library_version = version_buffer.value.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("NVML library version is not ASCII") from exc
        _string(library_version, "NVML library version")

        device = ctypes.c_void_p()
        rc = get_device(gpu_uuid.encode("ascii"), ctypes.byref(device))
        if rc != NVML_SUCCESS or not device.value:
            raise ValueError(
                f"nvmlDeviceGetHandleByUUID failed with NVML status {rc}"
            )
        uuid_buffer = ctypes.create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
        rc = get_device_uuid(device, uuid_buffer, ctypes.c_uint(len(uuid_buffer)))
        if rc != NVML_SUCCESS:
            raise ValueError(f"nvmlDeviceGetUUID failed with NVML status {rc}")
        try:
            observed_uuid = uuid_buffer.value.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("NVML device UUID is not ASCII") from exc
        if observed_uuid != gpu_uuid:
            raise ValueError("NVML device UUID does not match the requested GPU")

        count = ctypes.c_uint(0)
        rc = get_processes(device, ctypes.byref(count), None)
        if rc == NVML_SUCCESS and count.value == 0:
            raw_inventory = {
                "api": NVML_COMPUTE_API,
                "gpu_uuid": gpu_uuid,
                "processes": [],
            }
        else:
            if rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
                raise ValueError(
                    f"{NVML_COMPUTE_API} count query failed with NVML status {rc}"
                )
            records: list[dict[str, Any]] | None = None
            for _attempt in range(4):
                capacity = count.value
                if capacity == 0 or capacity > NVML_PROCESS_COUNT_LIMIT:
                    raise ValueError("NVML process count is outside the qualification bound")
                array = (_NvmlProcessInfoV2 * capacity)()
                returned = ctypes.c_uint(capacity)
                rc = get_processes(device, ctypes.byref(returned), array)
                if rc == NVML_ERROR_INSUFFICIENT_SIZE:
                    count = returned
                    continue
                if rc != NVML_SUCCESS:
                    raise ValueError(
                        f"{NVML_COMPUTE_API} failed with NVML status {rc}"
                    )
                if returned.value > capacity:
                    raise ValueError("NVML returned more process records than allocated")
                records = [
                    {
                        "pid": int(array[index].pid),
                        "used_gpu_memory_bytes": str(
                            int(array[index].usedGpuMemory)
                        ),
                    }
                    for index in range(returned.value)
                ]
                break
            if records is None:
                raise ValueError("NVML process inventory did not stabilize")
            raw_inventory = {
                "api": NVML_COMPUTE_API,
                "gpu_uuid": gpu_uuid,
                "processes": records,
            }

        inventory = _canonical_nvml_inventory(
            raw_inventory, "pre-child NVML inventory", gpu_uuid=gpu_uuid
        )
        return {
            "library_version": library_version,
            "inventory": inventory,
        }
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if initialized:
            shutdown_rc = shutdown()
            if shutdown_rc != NVML_SUCCESS and primary_error is None:
                raise ValueError(
                    f"nvmlShutdown failed with NVML status {shutdown_rc}"
                )


def _qualification_binding_payload(
    cold_preparation: Mapping[str, Any],
    nvml_pre_child: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cold_preparation": dict(cold_preparation),
        "nvml_pre_child": dict(nvml_pre_child),
        "runtime": {
            "source_revision": runtime_identity["source_revision"],
            "executable_sha256": runtime_identity["executable_sha256"],
        },
    }


def freeze_qualification_preflight(
    cold_preparation: Mapping[str, Any],
    *,
    gpu_uuid: str,
    runtime_identity: Mapping[str, Any],
    captured_at_unix_ns: str,
    nvml_query: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze already-collected cold evidence and one no-CUDA NVML query."""
    if not isinstance(gpu_uuid, str) or not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("qualification preflight GPU UUID is invalid")
    frozen_runtime = loads_strict(
        canonical_json_bytes(runtime_identity).decode("utf-8")
    )
    _validate_runtime(frozen_runtime)
    _decimal(
        captured_at_unix_ns,
        "qualification preflight capture timestamp",
        positive=True,
    )
    frozen_cold = loads_strict(
        canonical_json_bytes(cold_preparation).decode("utf-8")
    )
    cold = _validate_cold_preparation(frozen_cold)
    if not callable(nvml_query):
        raise ValueError("qualification preflight NVML query is not callable")
    query = _mapping(
        nvml_query(gpu_uuid),
        "qualification preflight NVML query result",
        {"library_version", "inventory"},
    )
    library_version = _string(
        query["library_version"], "qualification preflight NVML library version"
    )
    inventory = _canonical_nvml_inventory(
        query["inventory"], "qualification preflight NVML inventory", gpu_uuid=gpu_uuid
    )
    nvml_pre_child = {
        "capture_phase": "pre-child-before-cuda-initialization",
        "captured_at_unix_ns": captured_at_unix_ns,
        "library_version": library_version,
        "inventory": inventory,
        "inventory_sha256": _sha256_bytes(canonical_json_bytes(inventory)),
    }
    result = {
        "cold_preparation": cold,
        "cold_preparation_sha256": _sha256_bytes(
            canonical_json_bytes(cold)
        ),
        "nvml_pre_child": nvml_pre_child,
        "binding_sha256": _sha256_bytes(
            canonical_json_bytes(
                _qualification_binding_payload(
                    cold, nvml_pre_child, frozen_runtime
                )
            )
        ),
    }
    return result


def _validate_qualification_preflight(
    value: Any,
    *,
    model_identity: Mapping[str, Any],
    host_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> None:
    preflight = _mapping(
        value,
        "qualification_preflight",
        {
            "cold_preparation",
            "cold_preparation_sha256",
            "nvml_pre_child",
            "binding_sha256",
        },
    )
    cold = _validate_cold_preparation(preflight["cold_preparation"])
    cold_digest = _sha256(
        preflight["cold_preparation_sha256"],
        "qualification preflight cold-preparation SHA-256",
    )
    if cold_digest != _sha256_bytes(canonical_json_bytes(cold)):
        raise ValueError("qualification preflight cold-preparation digest mismatch")
    expected_model = {
        key: model_identity[key]
        for key in (
            "device",
            "filename",
            "inode",
            "mtime_ns",
            "repository",
            "revision",
            "sha256",
            "size_bytes",
        )
    }
    _require_identity_match(cold["model_identity"], expected_model, "cold model")

    nvml = _mapping(
        preflight["nvml_pre_child"],
        "qualification preflight NVML baseline",
        {
            "capture_phase",
            "captured_at_unix_ns",
            "library_version",
            "inventory",
            "inventory_sha256",
        },
    )
    if nvml["capture_phase"] != "pre-child-before-cuda-initialization":
        raise ValueError("qualification preflight NVML capture phase is invalid")
    _decimal(
        nvml["captured_at_unix_ns"],
        "qualification preflight NVML capture timestamp",
        positive=True,
    )
    _string(
        nvml["library_version"], "qualification preflight NVML library version"
    )
    gpu_uuid = host_identity["gpu_uuid"]
    canonical_inventory = _canonical_nvml_inventory(
        nvml["inventory"],
        "qualification preflight NVML inventory",
        gpu_uuid=gpu_uuid,
    )
    if nvml["inventory"] != canonical_inventory:
        raise ValueError("qualification preflight NVML inventory is not PID-sorted")
    inventory_digest = _sha256(
        nvml["inventory_sha256"], "qualification preflight NVML inventory SHA-256"
    )
    if inventory_digest != _sha256_bytes(canonical_json_bytes(canonical_inventory)):
        raise ValueError("qualification preflight NVML inventory digest mismatch")

    binding_digest = _sha256(
        preflight["binding_sha256"], "qualification preflight binding SHA-256"
    )
    expected_binding = _sha256_bytes(
        canonical_json_bytes(
            _qualification_binding_payload(cold, nvml, runtime_identity)
        )
    )
    if binding_digest != expected_binding:
        raise ValueError("qualification preflight binding digest mismatch")


def _oracle_tokenizer_revision() -> str:
    oracle = load_json_file_strict(ORACLE_MANIFEST_PATH)
    try:
        revision = oracle["provenance"]["tokenizer_runtime_commit"]
    except (KeyError, TypeError) as exc:
        raise ValueError("resident oracle lacks tokenizer runtime provenance") from exc
    return _revision(revision, "resident oracle tokenizer runtime commit")


def load_json_file_strict(path: Path) -> Any:
    try:
        return loads_strict(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc


def bind_runtime_identity(executable: Path | None = None) -> dict[str, Any]:
    executable = executable or ROOT / "ds4"
    if executable.is_symlink():
        raise ValueError(f"revision-bound tokenizer executable must not be a symlink: {executable}")
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"revision-bound repo ./ds4 binary is absent: {exc}") from exc
    if resolved != (ROOT / "ds4").resolve(strict=False):
        raise ValueError("tokenizer executable must be the revision-bound repo ./ds4 binary")
    before, identity = _stat_identity(resolved)
    if not os.access(resolved, os.X_OK):
        raise ValueError("revision-bound repo ./ds4 binary is not executable")

    revision = _run_git(["rev-parse", "HEAD"], "DS4 revision lookup")
    _revision(revision, "DS4 source revision")
    if _run_git(["status", "--short", "--untracked-files=no"], "DS4 cleanliness check"):
        raise ValueError("DS4 tracked worktree is dirty; tokenizer source identity is not immutable")
    oracle_revision = _oracle_tokenizer_revision()
    _run_git(
        ["merge-base", "--is-ancestor", oracle_revision, revision],
        "tokenizer provenance ancestry check",
    )
    digest = _sha256_file(resolved)
    after, _ = _stat_identity(resolved)
    if not _same_stat(before, after):
        raise ValueError("revision-bound repo ./ds4 binary changed while hashing")
    return {
        "source_revision": revision,
        "oracle_tokenizer_revision": oracle_revision,
        "executable_path": str(resolved),
        "executable_sha256": digest,
        **identity,
        "token_dump_argv": list(TOKEN_DUMP_ARGV),
    }


def _assert_runtime_unchanged(runtime: Mapping[str, Any]) -> None:
    path = Path(runtime["executable_path"])
    _, identity = _stat_identity(path)
    for key in ("device", "inode", "size_bytes", "mtime_ns"):
        if identity[key] != runtime[key]:
            raise ValueError("tokenizer executable identity changed during manifest construction")
    if _sha256_file(path) != runtime["executable_sha256"]:
        raise ValueError("tokenizer executable digest changed during manifest construction")
    if _run_git(["rev-parse", "HEAD"], "DS4 revision recheck") != runtime["source_revision"]:
        raise ValueError("DS4 source revision changed during manifest construction")
    if _run_git(["status", "--short", "--untracked-files=no"], "DS4 cleanliness recheck"):
        raise ValueError("DS4 tracked worktree became dirty during manifest construction")


class RawDs4TokenCounter:
    def __init__(self, model: Path, runtime: Mapping[str, Any]) -> None:
        self.model = model
        self.runtime = runtime

    def __call__(self, rendered: bytes) -> int:
        parent = self.model.parent if os.access(self.model.parent, os.W_OK) else Path(tempfile.gettempdir())
        fd, name = tempfile.mkstemp(prefix=".laguna-manifest-prompt-", dir=parent)
        prompt_path = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            command = [
                self.runtime["executable_path"],
                "--dump-tokens",
                "--raw-prompt",
                "-m",
                str(self.model),
                "--prompt-file",
                str(prompt_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                )
            except OSError as exc:
                raise ValueError(f"DS4 tokenizer invocation failed: {exc}") from exc
            if completed.returncode != 0:
                detail = (
                    completed.stderr.decode("utf-8", errors="replace").strip()
                    or completed.stdout.decode("utf-8", errors="replace").strip()
                )
                raise ValueError(f"DS4 tokenizer invocation failed: {detail}")
            raw_first_line = completed.stdout.split(b"\n", 1)[0]
            try:
                first_line = raw_first_line.decode("ascii")
                tokens = ast.literal_eval(first_line)
            except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
                raise ValueError(f"DS4 token dump is invalid: {exc}") from exc
            if not isinstance(tokens, list) or any(
                type(token) is not int or token < 0 or token >= LAGUNA_VOCAB_SIZE
                for token in tokens
            ):
                raise ValueError("DS4 token dump first line is not an integer token list")
            return len(tokens)
        finally:
            try:
                prompt_path.unlink()
            except FileNotFoundError:
                pass


def bind_model_identity(model_path: Path) -> dict[str, str]:
    try:
        resolved = model_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve pinned model: {exc}") from exc
    if resolved.name != MODEL_FILENAME:
        raise ValueError(f"model filename must be {MODEL_FILENAME}")
    try:
        descriptor = os.open(resolved, os.O_RDONLY)
    except OSError as exc:
        raise ValueError(f"cannot open pinned model: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != MODEL_SIZE:
            raise ValueError("model does not have the pinned regular-file size")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if not _same_stat(before, after):
            raise ValueError("model inode changed while hashing")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError("model SHA-256 does not match the pinned Laguna artifact")
    return {
        "path": str(resolved),
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "filename": MODEL_FILENAME,
        "size_bytes": str(before.st_size),
        "sha256": digest.hexdigest(),
        "device": str(before.st_dev),
        "inode": str(before.st_ino),
        "mtime_ns": str(before.st_mtime_ns),
    }


def _unescape_mount(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _filesystem_identity(path: Path) -> dict[str, str]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read Linux mount identity: {exc}") from exc
    resolved = str(path.resolve())
    candidates: list[tuple[int, list[str], list[str]]] = []
    for line in lines:
        fields = line.split()
        if "-" not in fields or len(fields) < 10:
            continue
        separator = fields.index("-")
        mount_point = _unescape_mount(fields[4])
        if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
            candidates.append((len(mount_point), fields, fields[separator + 1 :]))
    if not candidates:
        raise ValueError("model path is not covered by /proc/self/mountinfo")
    _, fields, tail = max(candidates, key=lambda item: item[0])
    if len(tail) < 3:
        raise ValueError("model mount identity is incomplete")
    return {
        "mount_point": _unescape_mount(fields[4]),
        "type": tail[0],
        "source": _unescape_mount(tail[1]),
        "device": fields[2],
        "options": fields[5],
    }


def _read_nonempty(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="strict").strip()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not value:
        raise ValueError(f"{label} is empty")
    return value


def _nvme_identity(filesystem: Mapping[str, str]) -> dict[str, str]:
    source = Path(filesystem["source"])
    block = source.name
    sys_block = Path("/sys/class/block") / block
    try:
        resolved = sys_block.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"filesystem source is not a directly attributable block device: {exc}") from exc
    current = resolved
    while current.name and not re.fullmatch(r"nvme[0-9]+n[0-9]+", current.name):
        if current.parent == current:
            break
        current = current.parent
    if not re.fullmatch(r"nvme[0-9]+n[0-9]+", current.name):
        raise ValueError("filesystem source cannot be bound to one NVMe namespace")
    device = current / "device"
    return {
        "device": current.name,
        "model": _read_nonempty(device / "model", "NVMe model"),
        "serial": _read_nonempty(device / "serial", "NVMe serial"),
        "firmware_revision": _read_nonempty(device / "firmware_rev", "NVMe firmware"),
    }


def _cuda_driver_version() -> str:
    text = _read_nonempty(Path("/proc/driver/nvidia/version"), "CUDA driver version")
    match = re.search(
        r"Kernel Module(?:\s+for\s+\S+)?\s+([0-9][0-9.]*)",
        text,
    )
    if not match:
        raise ValueError("CUDA driver version is not parseable")
    return match.group(1)


def _cuda_runtime_version() -> str:
    version_json = Path("/usr/local/cuda/version.json")
    if version_json.is_file():
        value = load_json_file_strict(version_json)
        try:
            version = value["cuda"]["version"]
        except (KeyError, TypeError) as exc:
            raise ValueError("CUDA runtime version.json lacks cuda.version") from exc
        return _string(version, "CUDA runtime version")
    text = _read_nonempty(Path("/usr/local/cuda/version.txt"), "CUDA runtime version")
    match = re.search(r"([0-9]+(?:\.[0-9]+)+)", text)
    if not match:
        raise ValueError("CUDA runtime version is not parseable")
    return match.group(1)


def _single_gpu_uuid() -> str:
    paths = sorted(Path("/proc/driver/nvidia/gpus").glob("*/information"))
    uuids: list[str] = []
    for path in paths:
        text = _read_nonempty(path, f"GPU identity {path}")
        match = re.search(r"^GPU UUID:\s*(\S+)\s*$", text, re.M)
        if match:
            uuids.append(match.group(1))
    if len(uuids) != 1 or not GPU_UUID_RE.fullmatch(uuids[0]):
        raise ValueError("qualification host must expose exactly one attributable GPU UUID")
    return uuids[0]


def collect_host_identity(model_path: Path) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise ValueError("compact qualification manifest host identity requires Linux")
    uname = platform.uname()
    filesystem = _filesystem_identity(model_path)
    return {
        "hostname": _string(uname.node, "host name"),
        "architecture": _string(uname.machine, "host architecture"),
        "kernel_release": _string(uname.release, "kernel release"),
        "kernel_version": _string(uname.version, "kernel version"),
        "cuda_driver_version": _cuda_driver_version(),
        "cuda_runtime_version": _cuda_runtime_version(),
        "gpu_uuid": _single_gpu_uuid(),
        "filesystem": filesystem,
        "nvme": _nvme_identity(filesystem),
        "io": {
            "direct_io": False,
                "cold_preparation_advice":
                "madvise_random+posix_fadvise_dontneed+madvise_dontneed+linux_madv_pageout_residual",
            "runtime_disposal_advice": "madvise_dontneed",
        },
    }


def _validate_fixed_sources(seed: bytes) -> None:
    if len(seed) != SEED_SIZE or _sha256_bytes(seed) != SEED_SHA256:
        raise ValueError("benchmark seed bytes do not match the pinned generator output")
    if _sha256_file(GENERATOR_PATH) != GENERATOR_SHA256:
        raise ValueError("benchmark prompt generator does not match pinned provenance")


def _validate_prepared_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "schema", "model", "host", "prompt_source", "prompts", "sampling",
        "execution", "profiles", "eval_case_ids",
    }
    prepared = _mapping(value, "prepared manifest", required)
    if prepared["schema"] != SCHEMA_ID:
        raise ValueError("prepared manifest schema is not supported")
    _validate_model(prepared["model"])
    _validate_host(prepared["host"])
    _validate_prompt_source(prepared["prompt_source"])
    _validate_prompts(prepared["prompts"])
    _validate_sampling(prepared["sampling"])
    _validate_execution(prepared["execution"])
    _validate_profiles(prepared["profiles"])
    eval_ids = _list(prepared["eval_case_ids"], "prepared eval_case_ids")
    if eval_ids != list(EVAL_CASE_IDS):
        raise ValueError("prepared eval_case_ids are not pinned")


def prepare_manifest(
    model_path: Path | str,
    *,
    token_counter: TokenCounter | None = None,
    host_identity: Mapping[str, Any] | None = None,
    model_identity: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    seed_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Complete every model read and tokenizer subprocess before cold preparation."""
    model_path = Path(model_path)
    seed = seed_bytes if seed_bytes is not None else SEED_PATH.read_bytes()
    _validate_fixed_sources(seed)

    injected_model = model_identity is not None
    injected_runtime = runtime_identity is not None
    model = dict(model_identity) if model_identity is not None else bind_model_identity(model_path)
    runtime = dict(runtime_identity) if runtime_identity is not None else bind_runtime_identity()
    host = dict(host_identity) if host_identity is not None else collect_host_identity(Path(model["path"]))
    counter = token_counter or RawDs4TokenCounter(Path(model["path"]), runtime)

    prompts = []
    for target in PROMPT_TARGETS:
        rendered, prefix_bytes = select_rendered_prompt(seed, target, counter)
        if counter(rendered) != target:
            raise ValueError(f"final prompt verification failed for {target} native-template tokens")
        prompts.append({
            "id": f"native-{target}",
            "token_count": target,
            "payload_prefix_bytes": str(prefix_bytes),
            "rendered_size_bytes": str(len(rendered)),
            "rendered_base64": base64.b64encode(rendered).decode("ascii"),
            "sha256": _sha256_bytes(rendered),
        })

    if not injected_runtime:
        _assert_runtime_unchanged(runtime)
    if not injected_model:
        current = Path(model["path"]).stat()
        if (
            str(current.st_dev) != model["device"]
            or str(current.st_ino) != model["inode"]
            or str(current.st_size) != model["size_bytes"]
            or str(current.st_mtime_ns) != model["mtime_ns"]
        ):
            raise ValueError("pinned model identity changed during manifest construction")

    prepared: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "model": model,
        "host": host,
        "prompt_source": {
            "seed_size_bytes": str(len(seed)),
            "seed_sha256": _sha256_bytes(seed),
            "generator_sha256": _sha256_file(GENERATOR_PATH),
            "template_revision": LAGUNA_TEMPLATE_REVISION,
            "template_prefix_base64": base64.b64encode(LAGUNA_TEMPLATE_PREFIX).decode("ascii"),
            "template_suffix_base64": base64.b64encode(LAGUNA_TEMPLATE_SUFFIX).decode("ascii"),
            "template_sha256": LAGUNA_TEMPLATE_SHA256,
            "tokenizer_runtime": runtime,
        },
        "prompts": prompts,
        "sampling": {
            "max_generated_tokens": 512,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "min_p": 0.05,
            "seed": 1,
            "stop_sequences": [],
            "stop_token_policy": "model-native",
        },
        "execution": {
            "qualification_cold_preparations": 1,
            "fresh_process_runs": 1,
            "same_process_warm_repetitions": 3,
            "whole_request_timeout_seconds": 2700,
            "first_token_timeout_seconds": 900,
            "warm_statistic": "median-of-exactly-three",
            "scope": "each-profile-prompt-pair",
        },
        "profiles": [
            {
                "profile_id": profile_id,
                "cache_bytes": str(cache_bytes),
                "prompt_order": list(prompt_order),
            }
            for profile_id, cache_bytes, prompt_order in PROFILE_SPECS
        ],
        "eval_case_ids": list(EVAL_CASE_IDS),
    }
    _validate_prepared_manifest(prepared)
    return prepared


def finalize_manifest(
    prepared_manifest: Mapping[str, Any],
    qualification_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Purely bind frozen preflight evidence into already-prepared static bytes."""
    prepared = loads_strict(
        canonical_json_bytes(prepared_manifest).decode("utf-8")
    )
    _validate_prepared_manifest(prepared)
    frozen_preflight = loads_strict(
        canonical_json_bytes(qualification_preflight).decode("utf-8")
    )
    _validate_qualification_preflight(
        frozen_preflight,
        model_identity=prepared["model"],
        host_identity=prepared["host"],
        runtime_identity=prepared["prompt_source"]["tokenizer_runtime"],
    )
    manifest = dict(prepared)
    manifest["qualification_preflight"] = frozen_preflight
    validate_manifest(manifest)
    return manifest


def build_manifest(
    model_path: Path | str,
    *,
    token_counter: TokenCounter | None = None,
    host_identity: Mapping[str, Any] | None = None,
    model_identity: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    seed_bytes: bytes | None = None,
    qualification_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience API for callers that already hold trusted preflight evidence."""
    if qualification_preflight is None:
        raise ValueError(
            "qualification preflight evidence is required; capture it before manifest build"
        )
    prepared = prepare_manifest(
        model_path,
        token_counter=token_counter,
        host_identity=host_identity,
        model_identity=model_identity,
        runtime_identity=runtime_identity,
        seed_bytes=seed_bytes,
    )
    return finalize_manifest(prepared, qualification_preflight)


def _require_identity_match(
    recorded: Mapping[str, Any],
    observed: Mapping[str, Any],
    label: str,
) -> None:
    if dict(recorded) != dict(observed):
        differing = sorted(
            key for key in set(recorded) | set(observed)
            if recorded.get(key) != observed.get(key)
        )
        detail = differing[0] if differing else "identity"
        raise ValueError(f"recorded {label} binding does not match {detail}")


def verify_manifest_bindings(
    value: Mapping[str, Any],
    *,
    model_identity: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    token_counter: TokenCounter | None = None,
) -> None:
    validate_manifest(value)
    recorded_model = value["model"]
    recorded_runtime = value["prompt_source"]["tokenizer_runtime"]
    observed_model = (
        dict(model_identity)
        if model_identity is not None
        else bind_model_identity(Path(recorded_model["path"]))
    )
    observed_runtime = (
        dict(runtime_identity)
        if runtime_identity is not None
        else bind_runtime_identity()
    )
    _require_identity_match(recorded_model, observed_model, "model")
    _require_identity_match(recorded_runtime, observed_runtime, "tokenizer runtime")

    counter = token_counter or RawDs4TokenCounter(
        Path(observed_model["path"]), observed_runtime
    )
    for prompt in value["prompts"]:
        rendered = base64.b64decode(prompt["rendered_base64"], validate=True)
        observed = counter(rendered)
        recorded = prompt["token_count"]
        if observed != recorded:
            raise ValueError(
                f"{prompt['id']} token count mismatch: recorded {recorded}, observed {observed}"
            )
    if runtime_identity is None:
        _assert_runtime_unchanged(observed_runtime)
    if model_identity is None:
        current = Path(observed_model["path"]).stat()
        for key, actual in (
            ("device", current.st_dev),
            ("inode", current.st_ino),
            ("size_bytes", current.st_size),
            ("mtime_ns", current.st_mtime_ns),
        ):
            if str(actual) != observed_model[key]:
                raise ValueError("model identity changed during manifest verification")


def write_manifest_atomic(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    validate_manifest(value)
    if not target.parent.is_dir():
        raise ValueError(f"manifest output directory does not exist: {target.parent}")
    payload = _canonical_bytes(value) + b"\n"
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ValueError(f"immutable manifest output already exists: {target}") from exc
        temporary.unlink()
        temporary = None
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise ValueError(f"cannot atomically write manifest {target}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="build or verify an immutable manifest")
    actions = manifest.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="build the immutable reference manifest")
    build.add_argument("--model", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--qualification-plan", required=True, type=Path)
    build.add_argument(
        "--trusted-qualification-plan-sha256",
        required=True,
        help=(
            "plan digest received over the controlling harness channel from "
            "the plan-producing DS4 invocation"
        ),
    )
    verify = actions.add_parser("verify", help="verify an immutable reference manifest")
    verify.add_argument("--manifest", required=True, type=Path)

    sequence = commands.add_parser(
        "sequence", help="build one immutable qualification input sequence"
    )
    sequence_actions = sequence.add_subparsers(dest="action", required=True)
    sequence_build = sequence_actions.add_parser(
        "build", help="build one qualification sequence from a manifest"
    )
    sequence_build.add_argument("--manifest", required=True, type=Path)
    sequence_build.add_argument("--profile-id", required=True)
    sequence_build.add_argument("--prompt-id", required=True)
    sequence_build.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "sequence":
            value = load_manifest(args.manifest)
            payload = build_qualification_sequence(
                value, args.profile_id, args.prompt_id
            )
            write_qualification_sequence_atomic(args.output, payload)
            print(
                f"manifest_sha256={manifest_sha256(value)} "
                f"profile_id={args.profile_id} prompt_id={args.prompt_id} "
                f"output={args.output}"
            )
            return 0
        if args.action == "build":
            prepared = prepare_manifest(args.model)
            cold_preparation = cold_prepare_from_plan(
                args.model,
                args.qualification_plan,
                args.trusted_qualification_plan_sha256,
            )
            runtime_identity = prepared["prompt_source"]["tokenizer_runtime"]
            host_identity = prepared["host"]
            nvml_capture = collect_nvml_pre_child(host_identity["gpu_uuid"])
            captured_at_unix_ns = str(time.time_ns())
            preflight = freeze_qualification_preflight(
                cold_preparation,
                gpu_uuid=host_identity["gpu_uuid"],
                runtime_identity=runtime_identity,
                captured_at_unix_ns=captured_at_unix_ns,
                nvml_query=lambda gpu_uuid: nvml_capture,
            )
            value = finalize_manifest(prepared, preflight)
            write_manifest_atomic(args.output, value)
            print(f"manifest_sha256={manifest_sha256(value)} output={args.output}")
            return 0
        value = load_manifest(args.manifest)
        verify_manifest_bindings(value)
        print(
            f"manifest_sha256={manifest_sha256(value)} "
            f"prompts={len(value['prompts'])} profiles={len(value['profiles'])}"
        )
        return 0
    except ValueError as exc:
        print(f"compact-runtime-qualify: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
