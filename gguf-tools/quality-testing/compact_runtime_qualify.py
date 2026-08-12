#!/usr/bin/env python3
"""Build and verify the immutable Laguna compact-runtime benchmark manifest.

This task intentionally stops at manifest construction.  It neither launches a
qualification server nor publishes a result bundle.
"""

from __future__ import annotations

import argparse
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
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


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

TokenCounter = Callable[[bytes], int]


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
        "cold_preparation_advice": "posix_fadvise_dontneed",
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
        "execution", "profiles", "eval_case_ids",
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
    _validate_profiles(manifest["profiles"])
    eval_ids = _list(manifest["eval_case_ids"], "eval_case_ids")
    if eval_ids != list(EVAL_CASE_IDS):
        raise ValueError("eval_case_ids must contain the four pinned cases in order")


def manifest_sha256(value: Mapping[str, Any]) -> str:
    validate_manifest(value)
    return _sha256_bytes(_canonical_bytes(value))


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
    if nofollow is None:
        raise ValueError(f"{label} cannot be opened without following symlinks")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(
            f"{label} cannot be opened without following symlinks: {exc}"
        ) from exc
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


def sample_exact_inode_residency(
    descriptor: int,
    file_size: int,
    page_size: int,
) -> int:
    """Count resident pages for one opened inode through mincore(2)."""
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
        return sum(1 for value in vector if value & 1) * page_size
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


def cold_prepare_from_plan(
    model_path: Path | str,
    plan_path: Path | str,
    expected_plan_sha256: str,
    *,
    advise: Callable[[int, int, int], None] = _posix_fadvise_dontneed,
    sample_residency: Callable[[int, int, int], int] = sample_exact_inode_residency,
) -> dict[str, Any]:
    """Cold-prepare only the plan's descriptor-bound, safe full-page union."""
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

    descriptor, before = _open_regular_nofollow(model_path, "model")
    try:
        if Path(model_path).name != model["filename"]:
            raise ValueError("opened model filename does not match the plan")
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
        resident_bytes = sample_residency(descriptor, model_size, page_size)
        if (
            type(resident_bytes) is not int
            or resident_bytes < 0
            or resident_bytes > ((model_size + page_size - 1) // page_size) * page_size
            or resident_bytes % page_size != 0
        ):
            raise ValueError("exact-inode residency sample is invalid")
        after_sample = os.fstat(descriptor)
        if not _same_stat(before, after_sample):
            raise ValueError("model identity changed during residency measurement")
        _model_identity_matches(model, after_sample)
        result = {
            "plan_sha256": observed_digest,
            "ledger_sha256": plan["ledger_sha256"],
            "model_identity": dict(model),
            "page_size": str(page_size),
            **report,
            "resident_bytes_after": str(resident_bytes),
            "unavoidable_bytes": str(unavoidable_bytes),
        }
        if report["failed_calls"] != 0:
            raise ValueError(
                "cold-preparation advice failed; evidence is invalid: "
                f"{report['errno_buckets']}"
            )
        return result
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
        usage = process["used_gpu_memory_bytes"]
        if type(usage) is not int or usage < 0:
            raise ValueError(f"{label} NVML process {index} usage is missing")
        if usage == UINT64_MAX:
            raise ValueError(f"{label} NVML process {index} usage is unknown")
        if usage > UINT64_MAX:
            raise ValueError(f"{label} NVML process {index} usage is invalid")
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
            "cold_preparation_advice": "posix_fadvise_dontneed",
            "runtime_disposal_advice": "madvise_dontneed",
        },
    }


def _validate_fixed_sources(seed: bytes) -> None:
    if len(seed) != SEED_SIZE or _sha256_bytes(seed) != SEED_SHA256:
        raise ValueError("benchmark seed bytes do not match the pinned generator output")
    if _sha256_file(GENERATOR_PATH) != GENERATOR_SHA256:
        raise ValueError("benchmark prompt generator does not match pinned provenance")


def build_manifest(
    model_path: Path | str,
    *,
    token_counter: TokenCounter | None = None,
    host_identity: Mapping[str, Any] | None = None,
    model_identity: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    seed_bytes: bytes | None = None,
) -> dict[str, Any]:
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

    manifest: dict[str, Any] = {
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
    validate_manifest(manifest)
    return manifest


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
    verify = actions.add_parser("verify", help="verify an immutable reference manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "build":
            value = build_manifest(args.model)
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
