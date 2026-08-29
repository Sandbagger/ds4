#!/usr/bin/env python3
"""Promote and verify the pinned Laguna Poolside oracle fixture."""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


VOCAB_SIZE = 100352
VECTOR_BYTES = VOCAB_SIZE * 4
CONTINUATION_TOKENS = 8
CONTINUATION_BYTES = CONTINUATION_TOKENS * 4
MAX_CONTRACT_FILE_BYTES = 16 * 1024 * 1024

CONTRACT_COMMIT = "a250e43722945e293f6044bc7254c4806d5a7912"
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
CAPTURE_MANIFEST_SHA256 = (
    "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
)
GENERATOR_SHA256 = "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"
BENCHMARK_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"

CAPTURE_MODEL = {
    "repository": "poolside/Laguna-S-2.1-GGUF",
    "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
    "file": "laguna-s-2.1-Q4_K_M.gguf",
    "size": 68248759648,
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
}
PROMOTED_MODEL = {
    "repository": "poolside/Laguna-S-2.1-GGUF",
    "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
    "filename": "laguna-s-2.1-Q4_K_M.gguf",
    "size": 68248759648,
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
}

CASE_SPECS = (
    ("short", "laguna-ds4", 22, 1024),
    ("swa-513", "raw", 513, 1024),
    ("yarn-8193", "raw", 8193, 8202),
    ("deep-32768", "raw", 32768, 32768),
)

FIXED_INPUT_FILES = {
    "cases.json",
    "short.txt",
    "generate_benchmark_prompt.py",
    "benchmark-32768.txt",
}
MATERIALIZED_PROMPT_FILES = {
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
}
PROMOTED_VECTOR_FILES = {
    f"{case_id}.llama.f32" for case_id, _, _, _ in CASE_SPECS
}
PROMOTED_OUTPUT_FILES = (
    MATERIALIZED_PROMPT_FILES
    | PROMOTED_VECTOR_FILES
    | {"yarn-8193.continuation.i32", "manifest.json"}
)
PROMOTED_FILES = FIXED_INPUT_FILES | PROMOTED_OUTPUT_FILES

CAPTURE_ARTIFACT_FILES = {
    "short.prompt",
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
    "short.tokens.i32",
    "swa-513.tokens.i32",
    "yarn-8193.tokens.i32",
    "deep-32768.tokens.i32",
    "short.logits.f32",
    "swa-513.logits.f32",
    "yarn-8193.logits.f32",
    "deep-32768.logits.f32",
    "yarn-8193.continuation.i32",
    "yarn-8193.step-00.logits.f32",
    "yarn-8193.step-01.logits.f32",
    "yarn-8193.step-02.logits.f32",
    "yarn-8193.step-03.logits.f32",
    "yarn-8193.step-04.logits.f32",
    "yarn-8193.step-05.logits.f32",
    "yarn-8193.step-06.logits.f32",
    "yarn-8193.step-07.logits.f32",
}

CUDA_LIMITS = {
    "centered_rms": 0.04,
    "centered_max_abs": 0.20,
    "top20_overlap": 18,
    "argmax_equal": True,
    "continuation_equal": True,
}

SHORT_RENDERED_PREFIX = b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
SHORT_RENDERED_SUFFIX = b"</user>\n<assistant></think>"


class ContractError(RuntimeError):
    """A fixture, capture, or provenance record violates the contract."""


def fail(message: str) -> None:
    raise ContractError(message)


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=duplicate_rejecting_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: top-level JSON value must be an object")
    return value


def _read_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # Opening an attacker-swapped FIFO/device must not block before fstat can
    # reject its non-regular mode.  O_NONBLOCK is harmless for regular files.
    flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _directory_open_flags() -> int:
    # Keep directory opens portable: O_NONBLOCK is for file reads only and is
    # not needed (or consistently accepted) with O_DIRECTORY.
    flags = _read_open_flags() & ~getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    return flags


def read_bytes_with_stat(
    path: Path, dir_fd: int | None = None
) -> tuple[bytes, os.stat_result]:
    """Read one regular file through one descriptor and return its inode."""
    fd: int | None = None
    target: str | Path = path if dir_fd is None else path.name
    try:
        try:
            if dir_fd is None:
                fd = os.open(target, _read_open_flags())
            else:
                fd = os.open(target, _read_open_flags(), dir_fd=dir_fd)
        except OSError as exc:
            fail(f"cannot open regular file {path}: {exc}")
        try:
            metadata = os.fstat(fd)
        except OSError as exc:
            fail(f"cannot stat regular file {path}: {exc}")
        if (
            not isinstance(metadata.st_size, int)
            or metadata.st_size < 0
            or metadata.st_size > MAX_CONTRACT_FILE_BYTES
        ):
            fail(
                f"regular file size out of bounds for {path}: "
                f"{metadata.st_size!r} bytes (limit {MAX_CONTRACT_FILE_BYTES})"
            )
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"missing regular file: {path}")
        try:
            payload = os.read(fd, metadata.st_size)
            after = os.fstat(fd)
        except (OSError, OverflowError) as exc:
            fail(f"cannot read {path}: {exc}")
        if len(payload) != metadata.st_size:
            fail(
                f"short read for {path}: expected {metadata.st_size} bytes, "
                f"got {len(payload)}"
            )
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
        ):
            fail(f"regular file changed while reading: {path}")
        return payload, metadata
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def read_bytes(path: Path) -> bytes:
    payload, _ = read_bytes_with_stat(path)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return load_json_bytes(read_bytes(path), path)


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    return value


def require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        fail(f"{label} keys mismatch: missing={missing} extra={extra}")


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be a boolean")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        fail(f"{label} must be a string")
    return value


def require_hex(value: Any, length: int, label: str) -> str:
    text = require_string(value, label)
    if len(text) != length or re.fullmatch(r"[0-9a-f]+", text) is None:
        fail(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def require_commit(value: Any, label: str) -> str:
    return require_hex(value, 40, label)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_safe_name(value: Any, expected: str, label: str) -> str:
    name = require_string(value, label)
    if name != expected or Path(name).name != name:
        fail(f"{label} must be {expected!r}")
    return name


def encode_i32(values: list[int]) -> bytes:
    encoded = array.array("i", values)
    if encoded.itemsize != 4:
        fail("runtime int32 representation is not four bytes")
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def decode_i32_payload(
    payload: bytes, count: int, label: str, expected_sha: str | None = None
) -> tuple[bytes, list[int]]:
    expected_bytes = count * 4
    if len(payload) != expected_bytes:
        fail(f"{label}: expected {expected_bytes} bytes, got {len(payload)}")
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        fail(f"{label}: SHA-256 mismatch")
    values = array.array("i")
    if values.itemsize != 4:
        fail("runtime int32 representation is not four bytes")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if any(token < 0 or token >= VOCAB_SIZE for token in result):
        fail(f"{label}: token ID outside vocabulary")
    return payload, result


def decode_f32_payload(
    payload: bytes, label: str, expected_sha: str | None = None
) -> tuple[bytes, list[float]]:
    if len(payload) != VECTOR_BYTES:
        fail(f"{label}: expected {VECTOR_BYTES} bytes, got {len(payload)}")
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        fail(f"{label}: SHA-256 mismatch")
    values = array.array("f")
    if values.itemsize != 4:
        fail("runtime float32 representation is not four bytes")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if len(result) != VOCAB_SIZE:
        fail(f"{label}: expected {VOCAB_SIZE} logits")
    if not all(math.isfinite(item) for item in result):
        fail(f"{label}: non-finite logit")
    return payload, result


def validate_token_list(value: Any, expected_count: int, label: str) -> list[int]:
    items = require_list(value, label)
    if len(items) != expected_count:
        fail(f"{label}: expected {expected_count} IDs, got {len(items)}")
    result = [require_int(item, f"{label}[{index}]") for index, item in enumerate(items)]
    if any(token < 0 or token >= VOCAB_SIZE for token in result):
        fail(f"{label}: token ID outside vocabulary")
    return result


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def actual_files(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        fail(f"fixture directory does not exist: {root}")
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        fail(f"cannot list {root}: {exc}")
    files: set[str] = set()
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            fail(f"{root}: fixture must contain regular files only ({entry.name})")
        files.add(entry.name)
    return files


def require_model(value: Any, expected: dict[str, Any], label: str) -> dict[str, Any]:
    model = require_object(value, label)
    require_exact_keys(model, set(expected), label)
    if model != expected:
        fail(f"{label}: pinned model identity mismatch")
    return model


def validate_fixture_inputs(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"promotion destination must be a populated fixture directory: {root}")

    generator_payload = read_bytes(root / "generate_benchmark_prompt.py")
    if sha256_bytes(generator_payload) != GENERATOR_SHA256:
        fail("prompt generator SHA-256 mismatch against fixed input")
    benchmark_payload = read_bytes(root / "benchmark-32768.txt")
    if sha256_bytes(benchmark_payload) != BENCHMARK_SHA256:
        fail("benchmark seed SHA-256 mismatch against fixed input")
    short_payload = read_bytes(root / "short.txt")
    if short_payload != b"Explain why a ring buffer wraps, in two sentences.\n":
        fail("short.txt does not match the fixed contract")

    cases_path = root / "cases.json"
    cases_payload = read_bytes(cases_path)
    cases = load_json_bytes(cases_payload, cases_path)
    require_exact_keys(
        cases,
        {"schema", "vocab_size", "continuation_case", "continuation_tokens", "cases"},
        "cases.json",
    )
    if require_string(cases["schema"], "cases.json.schema") != "laguna-resident-oracle-v1":
        fail("cases.json schema mismatch")
    if require_int(cases["vocab_size"], "cases.json.vocab_size") != VOCAB_SIZE:
        fail("cases.json vocabulary mismatch")
    if require_string(cases["continuation_case"], "cases.json.continuation_case") != "yarn-8193":
        fail("cases.json continuation case mismatch")
    if (
        require_int(cases["continuation_tokens"], "cases.json.continuation_tokens")
        != CONTINUATION_TOKENS
    ):
        fail("cases.json continuation length mismatch")
    entries = require_list(cases["cases"], "cases.json.cases")
    if len(entries) != len(CASE_SPECS):
        fail("cases.json must contain exactly four cases")
    for index, (case_id, render, frontier, context) in enumerate(CASE_SPECS):
        entry = require_object(entries[index], f"cases.json.cases[{index}]")
        expected_keys = {"id", "render", "prompt", "ctx"}
        if case_id != "short":
            expected_keys.add("frontier")
        require_exact_keys(entry, expected_keys, f"cases.json.cases[{index}]")
        if require_string(entry["id"], f"cases.json.cases[{index}].id") != case_id:
            fail(f"cases.json case {index} identity mismatch")
        if require_string(entry["render"], f"cases.json.cases[{index}].render") != render:
            fail(f"cases.json case {index} render mismatch")
        expected_prompt = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        require_safe_name(entry["prompt"], expected_prompt, f"cases.json.{case_id}.prompt")
        if require_int(entry["ctx"], f"cases.json.{case_id}.ctx") != context:
            fail(f"cases.json {case_id} context mismatch")
        if case_id != "short" and require_int(
            entry["frontier"], f"cases.json.{case_id}.frontier"
        ) != frontier:
            fail(f"cases.json {case_id} frontier mismatch")

    return {
        "payloads": {
            "generate_benchmark_prompt.py": generator_payload,
            "benchmark-32768.txt": benchmark_payload,
            "short.txt": short_payload,
            "cases.json": cases_payload,
        },
        "generator": generator_payload,
        "benchmark": benchmark_payload,
        "short": short_payload,
        "cases": cases,
    }


def validate_cuda_thresholds(value: Any, label: str) -> dict[str, Any]:
    thresholds = require_object(value, label)
    require_exact_keys(thresholds, set(CUDA_LIMITS), label)
    rms = require_number(thresholds["centered_rms"], f"{label}.centered_rms")
    maximum = require_number(thresholds["centered_max_abs"], f"{label}.centered_max_abs")
    overlap = require_int(thresholds["top20_overlap"], f"{label}.top20_overlap")
    argmax_equal = require_bool(thresholds["argmax_equal"], f"{label}.argmax_equal")
    continuation_equal = require_bool(
        thresholds["continuation_equal"], f"{label}.continuation_equal"
    )
    actual = {
        "centered_rms": rms,
        "centered_max_abs": maximum,
        "top20_overlap": overlap,
        "argmax_equal": argmax_equal,
        "continuation_equal": continuation_equal,
    }
    if actual != CUDA_LIMITS:
        fail(f"{label}: thresholds differ from the fixed CUDA admission contract")
    return actual


def validate_capture(root: Path, expected_oracle: str = "llama") -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"capture directory does not exist: {root}")

    capture_path = root / "capture.json"
    capture_payload = read_bytes(capture_path)
    actual_capture_sha = sha256_bytes(capture_payload)
    if actual_capture_sha != CAPTURE_MANIFEST_SHA256:
        fail("Poolside capture.json trust-anchor SHA-256 mismatch")
    capture = load_json_bytes(capture_payload, capture_path)

    if expected_oracle != "llama":
        fail(f"unsupported capture oracle: {expected_oracle}")
    require_exact_keys(
        capture,
        {
            "schema",
            "oracle",
            "runtime_commit",
            "vocab_size",
            "seed_token_count",
            "model",
            "cases",
            "continuation",
            "files",
        },
        "Poolside capture",
    )
    if require_string(capture["schema"], "Poolside capture.schema") != "laguna-resident-capture-v1":
        fail("Poolside capture schema mismatch")
    if require_string(capture["oracle"], "Poolside capture.oracle") != "llama":
        fail("Poolside capture oracle identity mismatch")
    runtime_commit = require_commit(capture["runtime_commit"], "Poolside capture.runtime_commit")
    if runtime_commit != LLAMA_COMMIT:
        fail("Poolside capture runtime commit does not match the pinned revision")
    if require_int(capture["vocab_size"], "Poolside capture.vocab_size") != VOCAB_SIZE:
        fail("Poolside capture vocabulary mismatch")
    seed_token_count = require_int(
        capture["seed_token_count"], "Poolside capture.seed_token_count"
    )
    if seed_token_count != 61440:
        fail("Poolside capture seed token count must be exactly 61440")
    require_model(capture["model"], CAPTURE_MODEL, "Poolside capture.model")

    files = require_object(capture["files"], "Poolside capture.files")
    require_exact_keys(files, CAPTURE_ARTIFACT_FILES, "Poolside capture.files")
    found_files = actual_files(root)
    expected_files = CAPTURE_ARTIFACT_FILES | {"capture.json"}
    if found_files != expected_files:
        fail(
            "Poolside capture file allowlist mismatch: "
            f"missing={sorted(expected_files - found_files)} "
            f"extra={sorted(found_files - expected_files)}"
        )

    # Capture artifact payloads are read exactly once after the trust anchor has
    # been checked.  All decoding and promotion below uses these byte strings.
    artifact_payloads: dict[str, bytes] = {}
    for name in sorted(CAPTURE_ARTIFACT_FILES):
        require_safe_name(name, name, f"Poolside capture.files[{name}]")
        expected_sha = require_hex(files[name], 64, f"Poolside capture.files[{name}]")
        payload = read_bytes(root / name)
        if sha256_bytes(payload) != expected_sha:
            fail(f"Poolside capture: SHA-256 mismatch for {name}")
        artifact_payloads[name] = payload

    entries = require_list(capture["cases"], "Poolside capture.cases")
    if len(entries) != len(CASE_SPECS):
        fail("Poolside capture must contain exactly four cases")
    decoded_cases: list[dict[str, Any]] = []
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        case = require_object(entries[index], f"Poolside capture.cases[{index}]")
        require_exact_keys(
            case,
            {
                "id",
                "render",
                "prompt",
                "frontier",
                "context",
                "prompt_file",
                "tokens_file",
                "logits_file",
                "token_count",
                "argmax",
            },
            f"Poolside capture.cases[{index}]",
        )
        if require_string(case["id"], f"Poolside capture.{case_id}.id") != case_id:
            fail(f"Poolside capture case {index} identity mismatch")
        if require_string(case["render"], f"Poolside capture.{case_id}.render") != render:
            fail(f"Poolside capture case {index} render mismatch")
        expected_prompt_ref = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        require_safe_name(case["prompt"], expected_prompt_ref, f"Poolside capture.{case_id}.prompt")
        frontier = require_int(case["frontier"], f"Poolside capture.{case_id}.frontier")
        if frontier <= 0 or (fixed_frontier != frontier):
            fail(f"Poolside capture.{case_id}: invalid frontier")
        if require_int(case["context"], f"Poolside capture.{case_id}.context") != context:
            fail(f"Poolside capture.{case_id}: context mismatch")
        if require_int(case["token_count"], f"Poolside capture.{case_id}.token_count") != frontier:
            fail(f"Poolside capture.{case_id}: token count/frontier mismatch")
        prompt_name = require_safe_name(
            case["prompt_file"], f"{case_id}.prompt", f"Poolside capture.{case_id}.prompt_file"
        )
        tokens_name = require_safe_name(
            case["tokens_file"],
            f"{case_id}.tokens.i32",
            f"Poolside capture.{case_id}.tokens_file",
        )
        logits_name = require_safe_name(
            case["logits_file"],
            f"{case_id}.logits.f32",
            f"Poolside capture.{case_id}.logits_file",
        )
        prompt = artifact_payloads[prompt_name]
        if not prompt:
            fail(f"Poolside capture.{case_id}: empty prompt")
        try:
            prompt.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            fail(f"Poolside capture.{case_id}: prompt is not valid UTF-8")
        _, tokens = decode_i32_payload(
            artifact_payloads[tokens_name], frontier, f"Poolside capture.{case_id}.tokens"
        )
        logits_payload, logits = decode_f32_payload(
            artifact_payloads[logits_name], f"Poolside capture.{case_id}.logits"
        )
        recorded_argmax = require_int(case["argmax"], f"Poolside capture.{case_id}.argmax")
        if recorded_argmax < 0 or recorded_argmax >= VOCAB_SIZE:
            fail(f"Poolside capture.{case_id}: argmax outside vocabulary")
        if recorded_argmax != argmax(logits):
            fail(f"Poolside capture.{case_id}: recorded argmax mismatch")
        decoded_cases.append(
            {
                "id": case_id,
                "render": render,
                "frontier": frontier,
                "context": context,
                "prompt": prompt,
                "tokens": tokens,
                "logits": logits,
                "logits_payload": logits_payload,
                "argmax": recorded_argmax,
                "prompt_name": prompt_name,
            }
        )

    continuation = require_object(capture["continuation"], "Poolside capture.continuation")
    require_exact_keys(
        continuation,
        {"case", "tokens_file", "logits_files", "argmax"},
        "Poolside capture.continuation",
    )
    if require_string(continuation["case"], "Poolside capture.continuation.case") != "yarn-8193":
        fail("Poolside capture continuation case mismatch")
    continuation_name = require_safe_name(
        continuation["tokens_file"],
        "yarn-8193.continuation.i32",
        "Poolside capture.continuation.tokens_file",
    )
    token_payload, continuation_ids = decode_i32_payload(
        artifact_payloads[continuation_name],
        CONTINUATION_TOKENS,
        "Poolside capture.continuation",
    )
    recorded_ids = validate_token_list(
        continuation["argmax"], CONTINUATION_TOKENS, "Poolside capture.continuation.argmax"
    )
    if continuation_ids != recorded_ids:
        fail("Poolside capture continuation binary/manifest mismatch")
    logits_names = require_list(
        continuation["logits_files"], "Poolside capture.continuation.logits_files"
    )
    if len(logits_names) != CONTINUATION_TOKENS:
        fail("Poolside capture continuation must contain eight logit rows")
    for step, value in enumerate(logits_names):
        name = require_safe_name(
            value,
            f"yarn-8193.step-{step:02d}.logits.f32",
            f"Poolside capture.continuation.logits_files[{step}]",
        )
        _, logits = decode_f32_payload(
            artifact_payloads[name], f"Poolside capture continuation step {step}"
        )
        if argmax(logits) != recorded_ids[step]:
            fail(f"Poolside capture continuation argmax mismatch at step {step}")
    if recorded_ids[0] != decoded_cases[2]["argmax"]:
        fail("Poolside capture YaRN frontier argmax differs from continuation step zero")

    return {
        "capture_sha256": actual_capture_sha,
        "runtime_commit": runtime_commit,
        "model": dict(CAPTURE_MODEL),
        "cases": decoded_cases,
        "continuation_ids": recorded_ids,
        "continuation_payload": token_payload,
        "seed_token_count": seed_token_count,
        "payloads": artifact_payloads,
    }


def parse_dump_tokens(stdout: str, label: str) -> list[int]:
    lines = stdout.splitlines()
    first_line = lines[0].strip() if lines else ""
    try:
        value = ast.literal_eval(first_line)
    except (SyntaxError, ValueError) as exc:
        fail(f"{label}: cannot parse ds4 --dump-tokens output: {exc}")
    if not isinstance(value, list):
        fail(f"{label}: ds4 --dump-tokens did not return a list")
    result = [require_int(item, f"{label}.ds4_tokens[{index}]") for index, item in enumerate(value)]
    if any(token < 0 or token >= VOCAB_SIZE for token in result):
        fail(f"{label}: ds4 token outside vocabulary")
    return result


def dump_ds4_tokens(ds4: Path, prompt_path: Path, label: str) -> list[int]:
    model_path = os.environ.get("LAGUNA_MODEL", PROMOTED_MODEL["filename"])
    command = [
        str(ds4),
        "--dump-tokens",
        "--raw-prompt",
        "-m",
        model_path,
        "--prompt-file",
        str(prompt_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        fail(f"{label}: ds4 --dump-tokens timed out: {exc}")
    except (OSError, UnicodeError) as exc:
        fail(f"{label}: cannot execute ds4: {exc}")
    if completed.returncode != 0:
        fail(f"{label}: ds4 --dump-tokens failed: {completed.stderr.strip()}")
    return parse_dump_tokens(completed.stdout, label)


def git_output(location: Path, args: list[str], label: str) -> str:
    command = ["git", "-C", str(location), *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        fail(f"{label}: git command timed out: {exc}")
    except (OSError, UnicodeError) as exc:
        fail(f"{label}: git command failed: {exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        fail(f"{label}: git command failed: {detail}")
    return completed.stdout


def discover_tokenizer_provenance(ds4: Path) -> dict[str, Any]:
    try:
        executable = ds4.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        fail(f"cannot resolve tokenizer executable: {exc}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        fail(f"tokenizer executable is not executable: {ds4}")

    top_text = git_output(executable.parent, ["rev-parse", "--show-toplevel"], "tokenizer repository")
    top_name = top_text.strip()
    if not top_name:
        fail("tokenizer repository top-level is empty")
    try:
        repository = Path(top_name).resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve tokenizer repository top-level: {exc}")
    if not repository.is_dir():
        fail("tokenizer repository top-level is not a directory")

    head_text = git_output(repository, ["rev-parse", "HEAD"], "tokenizer HEAD")
    head = require_commit(head_text.strip(), "tokenizer runtime HEAD")

    status_text = git_output(
        repository,
        ["status", "--short", "--untracked-files=no"],
        "tokenizer tracked status",
    )
    if status_text.strip():
        fail("tokenizer repository has dirty tracked files")

    contract_commit = require_commit(CONTRACT_COMMIT, "fixed contract commit")
    git_output(
        repository,
        ["cat-file", "-e", f"{contract_commit}^{{commit}}"],
        "fixed contract commit resolution",
    )
    git_output(
        repository,
        ["merge-base", "--is-ancestor", contract_commit, head],
        "fixed contract ancestor proof",
    )
    return {
        "repository": repository,
        "head": head,
        "contract_commit": contract_commit,
    }


def rendered_short_prompt(short_payload: bytes) -> bytes:
    return SHORT_RENDERED_PREFIX + short_payload + SHORT_RENDERED_SUFFIX


def validate_capture_determinism(capture: dict[str, Any], inputs: dict[str, Any]) -> None:
    cases = capture["cases"]
    if cases[0]["prompt"] != rendered_short_prompt(inputs["short"]):
        fail("short: captured rendered prompt differs from deterministic rendering")
    benchmark = inputs["benchmark"]
    deep_tokens = cases[3]["tokens"]
    for index, (case_id, _, frontier, _) in enumerate(CASE_SPECS[1:], start=1):
        prompt = cases[index]["prompt"]
        if len(prompt) > len(benchmark):
            fail(f"{case_id}: captured prompt is longer than benchmark seed")
        if prompt != benchmark[: len(prompt)]:
            fail(f"{case_id}: captured prompt is not an exact benchmark prefix")
        if cases[index]["tokens"] != deep_tokens[:frontier]:
            fail(f"{case_id}: captured token IDs are not the benchmark frontier prefix")


def validate_destination_for_promotion(
    destination: Path,
    expected_prompts: dict[str, bytes],
) -> tuple[dict[str, Any], dict[str, tuple[int, int]], int, tuple[int, int]]:
    found = actual_files(destination)
    partial_outputs = found & (PROMOTED_VECTOR_FILES | {"yarn-8193.continuation.i32"})
    if "manifest.json" not in found and partial_outputs:
        fail(
            "stale partial publication; existing output paths require explicit cleanup: "
            + ", ".join(sorted(partial_outputs))
        )
    if "manifest.json" in found:
        fail("promotion output already exists: manifest.json (no-clobber)")

    allowed = FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
    missing = FIXED_INPUT_FILES - found
    extra = found - allowed
    if missing or extra:
        fail(
            "pre-promotion fixture set mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    inputs = validate_fixture_inputs(destination)
    existing_prompts: dict[str, tuple[int, int]] = {}
    for name, expected in sorted(expected_prompts.items()):
        if name not in found:
            continue
        actual, metadata = read_bytes_with_stat(destination / name)
        if actual != expected:
            fail(f"existing prompt bytes mismatch for {name}")
        existing_prompts[name] = (metadata.st_dev, metadata.st_ino)
    destination_fd, destination_identity = open_owned_directory(
        destination, "promotion destination"
    )
    return inputs, existing_prompts, destination_fd, destination_identity


def open_owned_directory(path: Path, label: str) -> tuple[int, tuple[int, int]]:
    fd: int | None = None
    try:
        try:
            fd = os.open(path, _directory_open_flags())
        except OSError as exc:
            fail(f"{label} cannot be opened as a directory: {exc}")
        try:
            metadata = os.fstat(fd)
        except OSError as exc:
            fail(f"{label} cannot be stat'ed: {exc}")
        if not stat.S_ISDIR(metadata.st_mode):
            fail(f"{label} is not a directory")
        return fd, (metadata.st_dev, metadata.st_ino)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def assert_owned_directory(
    path: Path,
    fd: int,
    identity: tuple[int, int],
    label: str,
) -> None:
    try:
        descriptor_stat = os.fstat(fd)
    except OSError as exc:
        fail(f"{label} ownership was lost: {exc}")
    if (
        not stat.S_ISDIR(descriptor_stat.st_mode)
        or descriptor_stat.st_dev != identity[0]
        or descriptor_stat.st_ino != identity[1]
    ):
        fail(f"{label} ownership was lost")
    try:
        path_stat = path.lstat()
    except OSError as exc:
        fail(f"{label} path was removed or replaced: {exc}")
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != identity[0]
        or path_stat.st_ino != identity[1]
    ):
        fail(f"{label} path was removed or replaced")


def revalidate_existing_prompts(
    destination: Path,
    destination_fd: int,
    destination_identity: tuple[int, int],
    expected_prompts: dict[str, bytes],
    existing_prompts: dict[str, tuple[int, int]],
) -> None:
    assert_owned_directory(
        destination,
        destination_fd,
        destination_identity,
        "promotion destination",
    )
    for name, expected in sorted(expected_prompts.items()):
        if name not in existing_prompts:
            continue
        actual, metadata = read_bytes_with_stat(destination / name, destination_fd)
        identity = (metadata.st_dev, metadata.st_ino)
        if identity != existing_prompts[name]:
            fail(f"existing prompt inode changed for {name}")
        if actual != expected:
            fail(f"existing prompt bytes changed for {name}")


def revalidate_fixed_inputs(
    destination: Path,
    destination_fd: int,
    destination_identity: tuple[int, int],
    final_inputs: dict[str, Any],
) -> None:
    assert_owned_directory(
        destination,
        destination_fd,
        destination_identity,
        "promotion destination",
    )
    payloads = final_inputs["payloads"]
    for name in sorted(FIXED_INPUT_FILES):
        try:
            actual, _ = read_bytes_with_stat(destination / name, destination_fd)
        except ContractError:
            raise
        if actual != payloads[name]:
            fail(f"destination fixed input changed before manifest publication: {name}")


def stage_entry_identity(stage_fd: int, name: str) -> tuple[int, int]:
    fd: int | None = None
    try:
        try:
            fd = os.open(name, _read_open_flags(), dir_fd=stage_fd)
        except OSError as exc:
            fail(f"staged output cannot be opened: {name}: {exc}")
        try:
            metadata = os.fstat(fd)
        except OSError as exc:
            fail(f"staged output cannot be stat'ed: {name}: {exc}")
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"staged output is not a regular file: {name}")
        return metadata.st_dev, metadata.st_ino
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _record_stage_entry(
    stage_fd: int | None,
    name: str,
    owned_entries: dict[str, tuple[int, int]] | None,
) -> None:
    if stage_fd is None or owned_entries is None:
        return
    try:
        owned_entries[name] = stage_entry_identity(stage_fd, name)
    except ContractError:
        # Preserve the original write failure.  Cleanup will leave an entry
        # whose ownership could not be proved rather than deleting it blindly.
        pass


def write_stage_file(
    stage: Path,
    name: str,
    payload: bytes,
    stage_fd: int | None = None,
    owned_entries: dict[str, tuple[int, int]] | None = None,
) -> None:
    if stage_fd is None:
        try:
            (stage / name).write_bytes(payload)
        except OSError as exc:
            fail(f"cannot stage {name}: {exc}")
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        try:
            fd = os.open(name, flags, 0o600, dir_fd=stage_fd)
        except OSError as exc:
            fail(f"cannot stage {name}: {exc}")
        view = memoryview(payload)
        offset = 0
        try:
            while offset < len(view):
                written = os.write(fd, view[offset:])
                if written <= 0:
                    fail(f"cannot stage {name}: short write")
                offset += written
        except OSError as exc:
            _record_stage_entry(stage_fd, name, owned_entries)
            fail(f"cannot stage {name}: {exc}")
        except BaseException:
            _record_stage_entry(stage_fd, name, owned_entries)
            raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    _record_stage_entry(stage_fd, name, owned_entries)


def unlink_owned_links(
    published: list[tuple[str, int, int]], destination_fd: int | None, destination: Path
) -> None:
    for name, device, inode in reversed(published):
        if destination_fd is not None:
            try:
                current = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
            except OSError:
                continue
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_dev != device
                or current.st_ino != inode
            ):
                continue
            try:
                os.unlink(name, dir_fd=destination_fd)
            except OSError:
                pass
            continue
        target = destination / name
        try:
            current = target.lstat()
        except OSError:
            continue
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != device
            or current.st_ino != inode
        ):
            continue
        try:
            target.unlink()
        except OSError:
            pass


def cleanup_stage(
    stage: Path | None,
    stage_fd: int | None,
    stage_identity: tuple[int, int] | None,
    owned_entries: dict[str, tuple[int, int]],
) -> None:
    if stage is None:
        return
    if stage_fd is not None and stage_identity is not None:
        try:
            descriptor_stat = os.fstat(stage_fd)
        except OSError:
            descriptor_stat = None
        if descriptor_stat is not None and (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and descriptor_stat.st_dev == stage_identity[0]
            and descriptor_stat.st_ino == stage_identity[1]
        ):
            for name, (device, inode) in owned_entries.items():
                try:
                    current = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
                except OSError:
                    continue
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_dev != device
                    or current.st_ino != inode
                ):
                    continue
                try:
                    os.unlink(name, dir_fd=stage_fd)
                except OSError:
                    pass
    try:
        if stage_fd is not None:
            os.close(stage_fd)
    except OSError:
        pass

    try:
        current = stage.lstat()
    except OSError:
        return
    if (
        stage_identity is None
        or not stat.S_ISDIR(current.st_mode)
        or current.st_dev != stage_identity[0]
        or current.st_ino != stage_identity[1]
    ):
        return
    try:
        os.rmdir(stage)
    except OSError:
        pass


def remove_owned_directory(
    path: Path, fd: int | None, identity: tuple[int, int] | None
) -> None:
    if identity is None:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        return
    owned_descriptor = False
    if fd is not None:
        try:
            descriptor_stat = os.fstat(fd)
        except OSError:
            descriptor_stat = None
        owned_descriptor = descriptor_stat is not None and (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and descriptor_stat.st_dev == identity[0]
            and descriptor_stat.st_ino == identity[1]
        )
    try:
        current = path.lstat()
    except OSError:
        current = None
    if owned_descriptor and current is not None and (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == identity[0]
        and current.st_ino == identity[1]
    ):
        try:
            os.rmdir(path)
        except OSError:
            pass
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def build_promoted_manifest(
    capture: dict[str, Any],
    ds4_tokens: dict[str, list[int]],
    ds4_seed_token_count: int,
    tokenizer_runtime_commit: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest_cases: list[dict[str, Any]] = []
    vector_payloads: dict[str, bytes] = {}
    for case_id, render, frontier, context in CASE_SPECS:
        case = next(item for item in capture["cases"] if item["id"] == case_id)
        actual_tokens = ds4_tokens[case_id]
        if len(actual_tokens) != frontier or actual_tokens != case["tokens"]:
            fail(f"{case_id}: DS4 and Poolside token arrays differ")
        name = f"{case_id}.llama.f32"
        payload = case["logits_payload"]
        vector_payloads[name] = payload
        manifest_cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt_hex": case["prompt"].hex(),
                "prompt_sha256": sha256_bytes(case["prompt"]),
                "poolside_tokens": list(case["tokens"]),
                "ds4_tokens": list(actual_tokens),
                "frontier": frontier,
                "context": context,
                "vector": {
                    "file": name,
                    "sha256": sha256_bytes(payload),
                    "argmax": case["argmax"],
                },
            }
        )

    continuation_ids = capture["continuation_ids"]
    continuation_payload = capture["continuation_payload"]
    continuation_name = "yarn-8193.continuation.i32"
    payloads = dict(vector_payloads)
    payloads[continuation_name] = continuation_payload
    manifest = {
        "schema": "laguna-resident-promoted-v2",
        "oracle_policy": "single-poolside-v1",
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": CONTINUATION_TOKENS,
        "model": dict(PROMOTED_MODEL),
        "oracle": {
            "name": "poolside-llama",
            "runtime_commit": LLAMA_COMMIT,
            "capture_manifest_sha256": capture["capture_sha256"],
        },
        "provenance": {
            "contract_commit": CONTRACT_COMMIT,
            "tokenizer_runtime_commit": tokenizer_runtime_commit,
            "generator_sha256": GENERATOR_SHA256,
            "benchmark_sha256": BENCHMARK_SHA256,
            "poolside_seed_token_count": capture["seed_token_count"],
            "ds4_seed_token_count": ds4_seed_token_count,
        },
        "thresholds": {"cuda_admission": dict(CUDA_LIMITS)},
        "cases": manifest_cases,
        "continuation": {
            "case": "yarn-8193",
            "file": continuation_name,
            "sha256": sha256_bytes(continuation_payload),
            "argmax": list(continuation_ids),
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    payloads["manifest.json"] = manifest_payload
    return manifest, payloads


def verify_promoted(
    root: Path,
    contract_commit: str,
    tokenizer_runtime_commit: str,
    llama_commit: str,
    capture_manifest_sha256: str,
    gguf_size: int,
    gguf_sha256: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        fail(f"promoted fixture directory does not exist: {root}")
    found_files = actual_files(root)
    if found_files != PROMOTED_FILES:
        fail(
            f"promoted file set mismatch: missing={sorted(PROMOTED_FILES - found_files)} "
            f"extra={sorted(found_files - PROMOTED_FILES)}"
        )

    supplied_contract = require_commit(contract_commit, "supplied contract commit")
    supplied_tokenizer = require_commit(
        tokenizer_runtime_commit, "supplied tokenizer runtime commit"
    )
    supplied_llama = require_commit(llama_commit, "supplied llama commit")
    supplied_capture = require_hex(
        capture_manifest_sha256, 64, "supplied capture manifest SHA-256"
    )
    supplied_size = require_int(gguf_size, "supplied GGUF size")
    supplied_hash = require_hex(gguf_sha256, 64, "supplied GGUF SHA-256")
    if supplied_contract != CONTRACT_COMMIT:
        fail("supplied contract commit does not match the fixed contract pin")
    if supplied_llama != LLAMA_COMMIT:
        fail("supplied llama commit does not match the pinned runtime")
    if supplied_size != PROMOTED_MODEL["size"] or supplied_hash != PROMOTED_MODEL["sha256"]:
        fail("supplied GGUF identity does not match the pinned model")

    inputs = validate_fixture_inputs(root)
    manifest = load_json(root / "manifest.json")
    require_exact_keys(
        manifest,
        {
            "schema",
            "oracle_policy",
            "vocab_size",
            "continuation_case",
            "continuation_tokens",
            "model",
            "oracle",
            "provenance",
            "thresholds",
            "cases",
            "continuation",
        },
        "manifest",
    )
    if require_string(manifest["schema"], "manifest.schema") != "laguna-resident-promoted-v2":
        fail("unknown promoted manifest schema")
    if require_string(manifest["oracle_policy"], "manifest.oracle_policy") != "single-poolside-v1":
        fail("manifest oracle policy mismatch")
    if require_int(manifest["vocab_size"], "manifest.vocab_size") != VOCAB_SIZE:
        fail("manifest vocabulary size mismatch")
    if (
        require_string(manifest["continuation_case"], "manifest.continuation_case")
        != "yarn-8193"
    ):
        fail("manifest continuation case mismatch")
    if (
        require_int(manifest["continuation_tokens"], "manifest.continuation_tokens")
        != CONTINUATION_TOKENS
    ):
        fail("manifest continuation length mismatch")
    require_model(manifest["model"], PROMOTED_MODEL, "manifest.model")

    oracle = require_object(manifest["oracle"], "manifest.oracle")
    require_exact_keys(oracle, {"name", "runtime_commit", "capture_manifest_sha256"}, "manifest.oracle")
    if require_string(oracle["name"], "manifest.oracle.name") != "poolside-llama":
        fail("manifest oracle name mismatch")
    if require_commit(oracle["runtime_commit"], "manifest.oracle.runtime_commit") != supplied_llama:
        fail("manifest oracle runtime commit mismatch")
    if require_hex(oracle["capture_manifest_sha256"], 64, "manifest.oracle.capture_manifest_sha256") != supplied_capture:
        fail("manifest oracle capture digest mismatch")

    provenance = require_object(manifest["provenance"], "manifest.provenance")
    require_exact_keys(
        provenance,
        {
            "contract_commit",
            "tokenizer_runtime_commit",
            "generator_sha256",
            "benchmark_sha256",
            "poolside_seed_token_count",
            "ds4_seed_token_count",
        },
        "manifest.provenance",
    )
    if require_commit(provenance["contract_commit"], "manifest.provenance.contract_commit") != supplied_contract:
        fail("manifest contract provenance mismatch")
    if require_commit(
        provenance["tokenizer_runtime_commit"], "manifest.provenance.tokenizer_runtime_commit"
    ) != supplied_tokenizer:
        fail("manifest tokenizer runtime provenance mismatch")
    if require_hex(provenance["generator_sha256"], 64, "manifest.provenance.generator_sha256") != GENERATOR_SHA256:
        fail("manifest generator SHA-256 mismatch")
    if require_hex(provenance["benchmark_sha256"], 64, "manifest.provenance.benchmark_sha256") != BENCHMARK_SHA256:
        fail("manifest benchmark SHA-256 mismatch")
    if require_int(
        provenance["poolside_seed_token_count"], "manifest.provenance.poolside_seed_token_count"
    ) != 61440:
        fail("manifest Poolside seed token count mismatch")
    ds4_seed_count = require_int(
        provenance["ds4_seed_token_count"], "manifest.provenance.ds4_seed_token_count"
    )
    if ds4_seed_count < 32768:
        fail("manifest DS4 seed token count is below 32768")

    thresholds = require_object(manifest["thresholds"], "manifest.thresholds")
    require_exact_keys(thresholds, {"cuda_admission"}, "manifest.thresholds")
    validate_cuda_thresholds(thresholds["cuda_admission"], "manifest.thresholds.cuda_admission")

    entries = require_list(manifest["cases"], "manifest.cases")
    if len(entries) != len(CASE_SPECS):
        fail("manifest must contain exactly four cases")
    case_argmaxes: dict[str, int] = {}
    deep_tokens: list[int] | None = None
    for index, (case_id, render, frontier, context) in enumerate(CASE_SPECS):
        case = require_object(entries[index], f"manifest.cases[{index}]")
        require_exact_keys(
            case,
            {
                "id",
                "render",
                "prompt_hex",
                "prompt_sha256",
                "poolside_tokens",
                "ds4_tokens",
                "frontier",
                "context",
                "vector",
            },
            f"manifest.cases[{index}]",
        )
        if require_string(case["id"], f"manifest.cases[{index}].id") != case_id:
            fail(f"manifest case {index} identity mismatch")
        if require_string(case["render"], f"manifest.{case_id}.render") != render:
            fail(f"manifest {case_id} render mismatch")
        if require_int(case["frontier"], f"manifest.{case_id}.frontier") != frontier:
            fail(f"{case_id}: frontier mismatch")
        if require_int(case["context"], f"manifest.{case_id}.context") != context:
            fail(f"{case_id}: context mismatch")

        prompt_hex = require_string(case["prompt_hex"], f"manifest.{case_id}.prompt_hex")
        try:
            prompt = bytes.fromhex(prompt_hex)
        except ValueError:
            fail(f"{case_id}: prompt_hex is invalid")
        if prompt.hex() != prompt_hex or not prompt:
            fail(f"{case_id}: prompt_hex is noncanonical or empty")
        try:
            prompt.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            fail(f"{case_id}: rendered prompt is not valid UTF-8")
        if require_hex(case["prompt_sha256"], 64, f"manifest.{case_id}.prompt_sha256") != sha256_bytes(prompt):
            fail(f"{case_id}: prompt SHA-256 mismatch")
        if case_id == "short":
            expected_prompt = rendered_short_prompt(inputs["short"])
        else:
            prompt_name = require_safe_name(
                f"{case_id}.prompt", f"{case_id}.prompt", f"manifest.{case_id}.prompt_file"
            )
            expected_prompt = read_bytes(root / prompt_name)
            if len(expected_prompt) > len(inputs["benchmark"]):
                fail(f"{case_id}: prompt is longer than benchmark seed")
            if expected_prompt != inputs["benchmark"][: len(expected_prompt)]:
                fail(f"{case_id}: prompt is not an exact benchmark prefix")
        if prompt != expected_prompt:
            fail(f"{case_id}: manifest prompt bytes differ from deterministic input")

        poolside_tokens = validate_token_list(
            case["poolside_tokens"], frontier, f"manifest.{case_id}.poolside_tokens"
        )
        ds4_tokens = validate_token_list(
            case["ds4_tokens"], frontier, f"manifest.{case_id}.ds4_tokens"
        )
        if poolside_tokens != ds4_tokens:
            fail(f"{case_id}: Poolside and DS4 token arrays differ")
        if case_id != "short":
            if deep_tokens is None:
                deep_tokens = poolside_tokens if case_id == "deep-32768" else None
        if case_id == "deep-32768":
            deep_tokens = poolside_tokens
        elif case_id != "short" and deep_tokens is not None:
            if poolside_tokens != deep_tokens[:frontier]:
                fail(f"{case_id}: token IDs are not the benchmark frontier prefix")

        vector = require_object(case["vector"], f"manifest.{case_id}.vector")
        require_exact_keys(vector, {"file", "sha256", "argmax"}, f"manifest.{case_id}.vector")
        vector_name = require_safe_name(
            vector["file"], f"{case_id}.llama.f32", f"manifest.{case_id}.vector.file"
        )
        vector_sha = require_hex(vector["sha256"], 64, f"manifest.{case_id}.vector.sha256")
        vector_payload, vector_values = decode_f32_payload(
            read_bytes(root / vector_name), f"manifest.{case_id}.vector", vector_sha
        )
        del vector_payload
        recorded_argmax = require_int(vector["argmax"], f"manifest.{case_id}.vector.argmax")
        if recorded_argmax < 0 or recorded_argmax >= VOCAB_SIZE:
            fail(f"{case_id}: vector argmax outside vocabulary")
        if recorded_argmax != argmax(vector_values):
            fail(f"{case_id}: recorded vector argmax mismatch")
        case_argmaxes[case_id] = recorded_argmax

    # The fixed case order places the deep row last.  Check all shorter rows
    # against it after decoding so the comparison is independent of list order.
    if deep_tokens is None:
        fail("manifest deep benchmark case is missing")
    for index, (case_id, _, frontier, _) in enumerate(CASE_SPECS[1:], start=1):
        row = validate_token_list(
            entries[index]["poolside_tokens"], frontier, f"manifest.{case_id}.poolside_tokens"
        )
        if row != deep_tokens[:frontier]:
            fail(f"{case_id}: token IDs are not the benchmark frontier prefix")

    continuation = require_object(manifest["continuation"], "manifest.continuation")
    require_exact_keys(
        continuation,
        {"case", "file", "sha256", "argmax"},
        "manifest.continuation",
    )
    if require_string(continuation["case"], "manifest.continuation.case") != "yarn-8193":
        fail("manifest continuation case mismatch")
    continuation_name = require_safe_name(
        continuation["file"],
        "yarn-8193.continuation.i32",
        "manifest.continuation.file",
    )
    continuation_sha = require_hex(continuation["sha256"], 64, "manifest.continuation.sha256")
    _, continuation_ids = decode_i32_payload(
        read_bytes(root / continuation_name),
        CONTINUATION_TOKENS,
        "manifest.continuation",
        continuation_sha,
    )
    recorded_ids = validate_token_list(
        continuation["argmax"], CONTINUATION_TOKENS, "manifest.continuation.argmax"
    )
    if continuation_ids != recorded_ids:
        fail("manifest continuation binary and argmax IDs differ")
    if recorded_ids[0] != case_argmaxes["yarn-8193"]:
        fail("manifest YaRN frontier argmax differs from continuation step zero")


def promote(ds4: Path, llama_root: Path, destination: Path) -> None:
    lock_path = destination.with_name(f".{destination.name}.lock")
    # Pin the caller-requested destination before any path-based preflight.  The
    # held descriptor lets the later under-lock validation reject a retargeted
    # path even when the replacement contains an otherwise valid fixture.
    requested_destination_fd, requested_destination_identity = open_owned_directory(
        destination, "requested destination"
    )
    try:
        capture = validate_capture(llama_root, "llama")
        inputs = validate_fixture_inputs(destination)
        validate_capture_determinism(capture, inputs)

        expected_prompts = {
            case["prompt_name"]: case["prompt"]
            for case in capture["cases"]
            if case["id"] != "short"
        }

        provenance = discover_tokenizer_provenance(ds4)
        tokenizer_runtime_commit = provenance["head"]

        ds4_tokens: dict[str, list[int]] = {}
        with tempfile.TemporaryDirectory(prefix="laguna-tokenize-") as tokenize_name:
            tokenize_root = Path(tokenize_name)
            for case in capture["cases"]:
                write_stage_file(tokenize_root, case["prompt_name"], case["prompt"])
            benchmark_name = "benchmark-32768.txt"
            write_stage_file(tokenize_root, benchmark_name, inputs["benchmark"])

            for case in capture["cases"]:
                case_id = case["id"]
                actual = dump_ds4_tokens(
                    ds4,
                    tokenize_root / case["prompt_name"],
                    case_id,
                )
                if actual != case["tokens"]:
                    fail(f"{case_id}: DS4 and Poolside token arrays differ")
                ds4_tokens[case_id] = actual
            benchmark_tokens = dump_ds4_tokens(
                ds4,
                tokenize_root / benchmark_name,
                benchmark_name,
            )
            if len(benchmark_tokens) < 32768:
                fail("DS4 benchmark seed has fewer than 32768 tokens")
            if benchmark_tokens[:32768] != capture["cases"][3]["tokens"]:
                fail("DS4 and Poolside benchmark first 32768 token IDs differ")

        _, payloads = build_promoted_manifest(
            capture,
            ds4_tokens,
            len(benchmark_tokens),
            tokenizer_runtime_commit,
        )
        manifest_payload = payloads["manifest.json"]
    except BaseException:
        try:
            os.close(requested_destination_fd)
        except OSError:
            pass
        raise

    lock_fd: int | None = None
    lock_identity: tuple[int, int] | None = None
    destination_fd: int | None = None
    destination_identity: tuple[int, int] | None = None
    stage: Path | None = None
    stage_fd: int | None = None
    stage_identity: tuple[int, int] | None = None
    owned_stage_entries: dict[str, tuple[int, int]] = {}
    published: list[tuple[str, int, int]] = []
    try:
        try:
            os.mkdir(lock_path)
        except FileExistsError:
            fail(f"promotion lock is already held: {lock_path}")
        except OSError as exc:
            fail(f"cannot acquire promotion lock {lock_path}: {exc}")

        # Capture the lock inode immediately, then retain a directory fd so a
        # replacement at the sibling path can never be mistaken for our lock.
        try:
            lock_path_stat = lock_path.lstat()
        except OSError as exc:
            fail(f"cannot inspect promotion lock {lock_path}: {exc}")
        if not stat.S_ISDIR(lock_path_stat.st_mode):
            fail(f"promotion lock is not a directory: {lock_path}")
        lock_identity = (lock_path_stat.st_dev, lock_path_stat.st_ino)
        lock_fd, opened_lock_identity = open_owned_directory(
            lock_path, "promotion lock"
        )
        if opened_lock_identity != lock_identity:
            fail("promotion lock was replaced while opening")

        (
            final_inputs,
            existing_prompts,
            destination_fd,
            destination_identity,
        ) = validate_destination_for_promotion(destination, expected_prompts)
        # The final destination fd was opened under our lock.  It must still
        # identify the exact directory requested before any staging begins.
        assert_owned_directory(
            destination,
            requested_destination_fd,
            requested_destination_identity,
            "requested destination",
        )
        if destination_identity != requested_destination_identity:
            fail("promotion destination differs from requested destination")
        for name in FIXED_INPUT_FILES:
            if final_inputs["payloads"][name] != inputs["payloads"][name]:
                fail(f"destination fixed input changed during promotion: {name}")

        # The validator captured the destination inode under the lock.  A
        # replacement made immediately after it returns is therefore rejected
        # instead of receiving any publication links.
        assert_owned_directory(
            destination,
            destination_fd,
            destination_identity,
            "promotion destination",
        )

        try:
            stage = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.tmp-",
                    dir=str(destination.parent),
                )
            )
        except OSError as exc:
            fail(f"cannot create sibling promotion staging directory: {exc}")

        # Record the stage path inode immediately after mkdtemp, then compare
        # it with the held O_DIRECTORY fd before using either identity.
        try:
            stage_path_stat = stage.lstat()
        except OSError as exc:
            fail(f"cannot inspect promotion staging directory: {exc}")
        if not stat.S_ISDIR(stage_path_stat.st_mode):
            fail("promotion staging path is not a directory")
        stage_identity = (stage_path_stat.st_dev, stage_path_stat.st_ino)
        stage_fd, opened_stage_identity = open_owned_directory(
            stage, "promotion staging directory"
        )
        if opened_stage_identity != stage_identity:
            fail("promotion staging directory was replaced while opening")

        for name in sorted(FIXED_INPUT_FILES):
            write_stage_file(
                stage,
                name,
                final_inputs["payloads"][name],
                stage_fd,
                owned_stage_entries,
            )
        for name, prompt_payload in sorted(expected_prompts.items()):
            write_stage_file(
                stage,
                name,
                prompt_payload,
                stage_fd,
                owned_stage_entries,
            )
        for name in sorted(PROMOTED_VECTOR_FILES):
            write_stage_file(
                stage,
                name,
                payloads[name],
                stage_fd,
                owned_stage_entries,
            )
        write_stage_file(
            stage,
            "yarn-8193.continuation.i32",
            payloads["yarn-8193.continuation.i32"],
            stage_fd,
            owned_stage_entries,
        )
        write_stage_file(
            stage,
            "manifest.json",
            manifest_payload,
            stage_fd,
            owned_stage_entries,
        )

        assert_owned_directory(
            stage, stage_fd, stage_identity, "promotion stage directory"
        )
        verify_promoted(
            stage,
            CONTRACT_COMMIT,
            tokenizer_runtime_commit,
            LLAMA_COMMIT,
            capture["capture_sha256"],
            PROMOTED_MODEL["size"],
            PROMOTED_MODEL["sha256"],
        )
        assert_owned_directory(
            stage, stage_fd, stage_identity, "promotion stage directory"
        )

        final_provenance = discover_tokenizer_provenance(ds4)
        if final_provenance["head"] != tokenizer_runtime_commit:
            fail("tokenizer repository HEAD changed during promotion")

        assert_owned_directory(
            lock_path, lock_fd, lock_identity, "promotion lock"
        )
        assert_owned_directory(
            destination,
            destination_fd,
            destination_identity,
            "promotion destination",
        )
        revalidate_existing_prompts(
            destination,
            destination_fd,
            destination_identity,
            expected_prompts,
            existing_prompts,
        )
        assert_owned_directory(
            lock_path, lock_fd, lock_identity, "promotion lock"
        )

        link_names = [
            name for name in sorted(expected_prompts) if name not in existing_prompts
        ]
        link_names.extend(sorted(PROMOTED_VECTOR_FILES))
        link_names.extend(["yarn-8193.continuation.i32", "manifest.json"])
        for name in link_names:
            # Both sibling resources must still be the inodes captured by this
            # invocation immediately before every no-clobber link.
            assert_owned_directory(lock_path, lock_fd, lock_identity, "promotion lock")
            assert_owned_directory(
                destination,
                destination_fd,
                destination_identity,
                "promotion destination",
            )
            assert_owned_directory(
                stage, stage_fd, stage_identity, "promotion stage directory"
            )
            if name == "manifest.json":
                # Recheck every pre-existing prompt after all prior links and
                # immediately before exposing the manifest.
                revalidate_existing_prompts(
                    destination,
                    destination_fd,
                    destination_identity,
                    expected_prompts,
                    existing_prompts,
                )
                revalidate_fixed_inputs(
                    destination,
                    destination_fd,
                    destination_identity,
                    final_inputs,
                )
                assert_owned_directory(lock_path, lock_fd, lock_identity, "promotion lock")
                assert_owned_directory(
                    destination,
                    destination_fd,
                    destination_identity,
                    "promotion destination",
                )
                assert_owned_directory(
                    stage, stage_fd, stage_identity, "promotion stage directory"
                )
            expected_identity = owned_stage_entries.get(name)
            if expected_identity is None:
                fail(f"staged output ownership is missing: {name}")
            if stage_entry_identity(stage_fd, name) != expected_identity:
                fail(f"staged output inode changed before publication: {name}")
            try:
                os.link(
                    name,
                    name,
                    src_dir_fd=stage_fd,
                    dst_dir_fd=destination_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                # A wrapper or filesystem may report an error after creating
                # the link.  Record it only if the destination names our
                # staged inode; never claim an external inode as ours.
                try:
                    current = os.stat(
                        name, dir_fd=destination_fd, follow_symlinks=False
                    )
                except OSError:
                    current = None
                if current is not None and (
                    stat.S_ISREG(current.st_mode)
                    and current.st_dev == expected_identity[0]
                    and current.st_ino == expected_identity[1]
                ):
                    published.append((name, *expected_identity))
                fail(f"publication link failed for {name}: {exc}")
            try:
                destination_stat = os.stat(
                    name, dir_fd=destination_fd, follow_symlinks=False
                )
            except OSError as exc:
                fail(f"publication link cannot inspect destination {name}: {exc}")
            if (
                not stat.S_ISREG(destination_stat.st_mode)
                or destination_stat.st_dev != expected_identity[0]
                or destination_stat.st_ino != expected_identity[1]
            ):
                # If the source inode changed during os.link, the destination
                # link was still created by this invocation.  Record only that
                # actual destination inode so cleanup removes our link while
                # preserving the externally supplied source replacement.
                try:
                    source_after_identity = stage_entry_identity(stage_fd, name)
                except ContractError:
                    source_after_identity = expected_identity
                if source_after_identity != expected_identity and stat.S_ISREG(
                    destination_stat.st_mode
                ):
                    published.append(
                        (name, destination_stat.st_dev, destination_stat.st_ino)
                    )
                fail(f"publication link destination inode differs from staged inode for {name}")
            published.append((name, *expected_identity))
    except BaseException:
        unlink_owned_links(published, destination_fd, destination)
        raise
    finally:
        cleanup_stage(stage, stage_fd, stage_identity, owned_stage_entries)
        stage_fd = None
        remove_owned_directory(lock_path, lock_fd, lock_identity)
        lock_fd = None
        if destination_fd is not None:
            try:
                os.close(destination_fd)
            except OSError:
                pass
        try:
            os.close(requested_destination_fd)
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-promoted", type=Path)
    parser.add_argument("--contract-commit")
    parser.add_argument("--tokenizer-runtime-commit")
    parser.add_argument("--llama-commit")
    parser.add_argument("--capture-manifest-sha256")
    parser.add_argument("--gguf-size", type=int)
    parser.add_argument("--gguf-sha256")
    parser.add_argument("--ds4", type=Path)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--promote", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verify_mode = args.verify_promoted is not None
    verify_values = (
        args.contract_commit,
        args.tokenizer_runtime_commit,
        args.llama_commit,
        args.capture_manifest_sha256,
        args.gguf_size,
        args.gguf_sha256,
    )
    promote_values = (args.ds4, args.llama, args.promote)
    try:
        if verify_mode:
            if not all(value is not None for value in verify_values) or any(
                value is not None for value in promote_values
            ):
                fail("verification mode requires the seven v2 identity arguments only")
            verify_promoted(
                args.verify_promoted,
                args.contract_commit,
                args.tokenizer_runtime_commit,
                args.llama_commit,
                args.capture_manifest_sha256,
                args.gguf_size,
                args.gguf_sha256,
            )
            print(f"verified={args.verify_promoted} cases=4 vectors=4 oracle=poolside")
            return 0
        if not all(value is not None for value in promote_values) or any(
            value is not None for value in verify_values
        ):
            fail("promotion mode requires --ds4, --llama, and --promote only")
        promote(args.ds4, args.llama, args.promote)
        print(f"promoted={args.promote} cases=4 vectors=4 oracle=poolside")
        return 0
    except ContractError as exc:
        mode = "verification" if verify_mode else "promotion"
        print(f"{mode} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
