#!/usr/bin/env python3
"""Promote and verify the pinned single-Poolside Laguna oracle fixture."""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
import os
import re
import shutil
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
    "repository": CAPTURE_MODEL["repository"],
    "revision": CAPTURE_MODEL["revision"],
    "filename": CAPTURE_MODEL["file"],
    "size": CAPTURE_MODEL["size"],
    "sha256": CAPTURE_MODEL["sha256"],
}

CASE_SPECS = (
    ("short", "laguna-ds4", None, 1024),
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
PROMOTED_FILES = (
    FIXED_INPUT_FILES
    | MATERIALIZED_PROMPT_FILES
    | PROMOTED_VECTOR_FILES
    | {"yarn-8193.continuation.i32", "manifest.json"}
)
CAPTURE_ARTIFACT_FILES = (
    {
        artifact
        for case_id, _, _, _ in CASE_SPECS
        for artifact in (
            f"{case_id}.prompt",
            f"{case_id}.tokens.i32",
            f"{case_id}.logits.f32",
        )
    }
    | {f"yarn-8193.step-{step:02d}.logits.f32" for step in range(8)}
    | {"yarn-8193.continuation.i32"}
)
CUDA_LIMITS = {
    "centered_rms": 0.04,
    "centered_max_abs": 0.20,
    "top20_overlap": 18,
    "argmax_equal": True,
    "continuation_equal": True,
}
SHORT_BYTES = b"Explain why a ring buffer wraps, in two sentences.\n"
SHORT_PREFIX = b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
SHORT_SUFFIX = b"</user>\n<assistant></think>"


class ContractError(RuntimeError):
    """A fixture, capture, or source checkout violates the pinned contract."""


def fail(message: str) -> None:
    raise ContractError(message)


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    fail(f"non-finite JSON number: {value}")


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=duplicate_rejecting_object,
            parse_constant=reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label}: top-level JSON value must be an object")
    return value


def read_bytes(path: Path) -> bytes:
    path = Path(path)
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            fail(f"missing regular file: {path}")
        return path.read_bytes()
    except FileNotFoundError:
        fail(f"missing regular file: {path}")
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        fail(f"{label} must be a string")
    return value


def require_int(value: Any, label: str) -> int:
    if type(value) is not int:
        fail(f"{label} must be an integer")
    return value


def require_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        fail(f"{label} must be a finite floating-point number")
    return value


def require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} must be a boolean")
    return value


def require_hex(value: Any, length: int, label: str) -> str:
    text = require_string(value, label)
    if len(text) != length or re.fullmatch(r"[0-9a-f]+", text) is None:
        fail(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        fail(
            f"{label} keys mismatch: missing={sorted(keys - actual)} "
            f"extra={sorted(actual - keys)}"
        )


def require_safe_name(value: Any, expected: str, label: str) -> str:
    name = require_string(value, label)
    if name != expected or Path(name).name != name:
        fail(f"{label} must be {expected!r}")
    return name


def actual_files(root: Path) -> set[str]:
    root = Path(root)
    try:
        root_mode = root.lstat().st_mode
        if not stat.S_ISDIR(root_mode):
            fail(f"fixture directory does not exist: {root}")
        entries = list(root.iterdir())
    except FileNotFoundError:
        fail(f"fixture directory does not exist: {root}")
    except OSError as exc:
        fail(f"cannot list {root}: {exc}")
    files: set[str] = set()
    for entry in entries:
        try:
            mode = entry.lstat().st_mode
        except OSError as exc:
            fail(f"cannot inspect {entry}: {exc}")
        if not stat.S_ISREG(mode):
            fail(f"{root}: fixture must contain regular files only: {entry}")
        files.add(entry.name)
    return files


def decode_f32_payload(payload: bytes, label: str) -> list[float]:
    if len(payload) != VECTOR_BYTES:
        fail(f"{label}: expected {VECTOR_BYTES} bytes, got {len(payload)}")
    values = array.array("f")
    values.frombytes(payload)
    if values.itemsize != 4:
        fail("runtime float32 representation is not four bytes")
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if len(result) != VOCAB_SIZE:
        fail(f"{label}: expected {VOCAB_SIZE} logits")
    if not all(math.isfinite(value) for value in result):
        fail(f"{label}: non-finite logit")
    return result


def decode_i32_payload(payload: bytes, count: int, label: str) -> list[int]:
    expected_bytes = count * 4
    if len(payload) != expected_bytes:
        fail(f"{label}: expected {expected_bytes} bytes, got {len(payload)}")
    values = array.array("i")
    values.frombytes(payload)
    if values.itemsize != 4:
        fail("runtime int32 representation is not four bytes")
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if any(token < 0 or token >= VOCAB_SIZE for token in result):
        fail(f"{label}: token ID outside vocabulary")
    return result


def require_token_list(value: Any, count: int, label: str) -> list[int]:
    items = require_list(value, label)
    if len(items) != count:
        fail(f"{label}: expected {count} IDs, got {len(items)}")
    tokens = [require_int(item, f"{label}[{index}]") for index, item in enumerate(items)]
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        fail(f"{label}: token ID outside vocabulary")
    return tokens


def vector_argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def decode_canonical_hex(value: Any, label: str) -> bytes:
    text = require_string(value, label)
    try:
        payload = bytes.fromhex(text)
    except ValueError:
        fail(f"{label} is invalid")
    if not payload or payload.hex() != text:
        fail(f"{label} is noncanonical or empty")
    return payload


def require_utf8(payload: bytes, label: str) -> None:
    if not payload:
        fail(f"{label} must not be empty")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        fail(f"{label} is not valid UTF-8")


def validate_fixture_inputs(root: Path) -> dict[str, bytes]:
    payloads = {name: read_bytes(root / name) for name in sorted(FIXED_INPUT_FILES)}
    short = payloads["short.txt"]
    require_utf8(short, "short.txt")
    if short != SHORT_BYTES:
        fail("short.txt does not match the fixed contract")
    if sha256_bytes(payloads["generate_benchmark_prompt.py"]) != GENERATOR_SHA256:
        fail("prompt generator SHA-256 mismatch")
    benchmark = payloads["benchmark-32768.txt"]
    require_utf8(benchmark, "benchmark-32768.txt")
    if sha256_bytes(benchmark) != BENCHMARK_SHA256:
        fail("benchmark seed SHA-256 mismatch")

    cases = load_json_bytes(payloads["cases.json"], "cases.json")
    require_exact_keys(
        cases,
        {"schema", "vocab_size", "continuation_case", "continuation_tokens", "cases"},
        "cases.json",
    )
    if cases["schema"] != "laguna-resident-oracle-v1":
        fail("cases.json schema mismatch")
    if require_int(cases["vocab_size"], "cases.json.vocab_size") != VOCAB_SIZE:
        fail("cases.json vocabulary mismatch")
    if cases["continuation_case"] != "yarn-8193":
        fail("cases.json continuation case mismatch")
    if require_int(cases["continuation_tokens"], "cases.json.continuation_tokens") != 8:
        fail("cases.json continuation length mismatch")
    entries = require_list(cases["cases"], "cases.json.cases")
    if len(entries) != len(CASE_SPECS):
        fail("cases.json must contain exactly four cases")
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        entry = require_object(entries[index], f"cases.json.cases[{index}]")
        keys = {"id", "render", "prompt", "ctx"}
        if case_id != "short":
            keys.add("frontier")
        require_exact_keys(entry, keys, f"cases.json.cases[{index}]")
        expected_prompt = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        if (
            entry["id"] != case_id
            or entry["render"] != render
            or entry["prompt"] != expected_prompt
        ):
            fail(f"cases.json case {index} identity mismatch")
        if require_int(entry["ctx"], f"cases.json.{case_id}.ctx") != context:
            fail(f"cases.json {case_id} context mismatch")
        if case_id != "short" and require_int(
            entry["frontier"], f"cases.json.{case_id}.frontier"
        ) != fixed_frontier:
            fail(f"cases.json {case_id} frontier mismatch")
    return payloads


def validate_cuda_limits(value: Any, label: str) -> None:
    limits = require_object(value, label)
    require_exact_keys(limits, set(CUDA_LIMITS), label)
    if require_float(limits["centered_rms"], f"{label}.centered_rms") != 0.04:
        fail("CUDA centered RMS threshold differs from the fixed contract")
    if require_float(limits["centered_max_abs"], f"{label}.centered_max_abs") != 0.20:
        fail("CUDA centered maximum threshold differs from the fixed contract")
    if require_int(limits["top20_overlap"], f"{label}.top20_overlap") != 18:
        fail("CUDA top-20 threshold differs from the fixed contract")
    if not require_bool(limits["argmax_equal"], f"{label}.argmax_equal"):
        fail("CUDA argmax requirement must be true")
    if not require_bool(limits["continuation_equal"], f"{label}.continuation_equal"):
        fail("CUDA continuation requirement must be true")


def verify_promoted(
    root: Path,
    contract_commit: str,
    tokenizer_runtime_commit: str,
    llama_commit: str,
    capture_manifest_sha256: str,
    gguf_size: int,
    gguf_sha256: str,
) -> None:
    root = Path(root)
    found = actual_files(root)
    if found != PROMOTED_FILES:
        fail(
            f"promoted file set mismatch: missing={sorted(PROMOTED_FILES - found)} "
            f"extra={sorted(found - PROMOTED_FILES)}"
        )
    fixed = validate_fixture_inputs(root)
    manifest = load_json_bytes(read_bytes(root / "manifest.json"), "manifest.json")
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
    if manifest["schema"] != "laguna-resident-promoted-v2":
        fail("unknown promoted manifest schema")
    if manifest["oracle_policy"] != "single-poolside-v1":
        fail("unknown oracle policy")
    if require_int(manifest["vocab_size"], "manifest.vocab_size") != VOCAB_SIZE:
        fail("manifest vocabulary mismatch")
    if manifest["continuation_case"] != "yarn-8193":
        fail("manifest continuation case mismatch")
    if require_int(manifest["continuation_tokens"], "manifest.continuation_tokens") != 8:
        fail("manifest continuation length mismatch")

    model = require_object(manifest["model"], "manifest.model")
    require_exact_keys(model, set(PROMOTED_MODEL), "manifest.model")
    if model != PROMOTED_MODEL:
        fail("manifest model identity mismatch")
    if type(gguf_size) is not int or gguf_size != PROMOTED_MODEL["size"]:
        fail("supplied GGUF size does not match the pinned model")
    if require_hex(gguf_sha256, 64, "supplied GGUF SHA-256") != PROMOTED_MODEL["sha256"]:
        fail("supplied GGUF SHA-256 does not match the pinned model")

    supplied_contract = require_hex(contract_commit, 40, "supplied contract commit")
    supplied_tokenizer = require_hex(
        tokenizer_runtime_commit, 40, "supplied tokenizer runtime commit"
    )
    supplied_llama = require_hex(llama_commit, 40, "supplied Poolside runtime commit")
    supplied_capture = require_hex(
        capture_manifest_sha256, 64, "supplied capture manifest SHA-256"
    )
    if supplied_contract != CONTRACT_COMMIT:
        fail("supplied contract commit does not match the pinned contract")
    if supplied_llama != LLAMA_COMMIT:
        fail("supplied Poolside runtime commit does not match the pinned runtime")

    oracle = require_object(manifest["oracle"], "manifest.oracle")
    require_exact_keys(
        oracle, {"name", "runtime_commit", "capture_manifest_sha256"}, "manifest.oracle"
    )
    if oracle["name"] != "poolside-llama":
        fail("manifest oracle name mismatch")
    if require_hex(oracle["runtime_commit"], 40, "manifest.oracle.runtime_commit") != supplied_llama:
        fail("manifest Poolside runtime commit mismatch")
    if require_hex(
        oracle["capture_manifest_sha256"],
        64,
        "manifest.oracle.capture_manifest_sha256",
    ) != supplied_capture:
        fail("manifest capture trust anchor mismatch")

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
    if require_hex(provenance["contract_commit"], 40, "provenance.contract_commit") != supplied_contract:
        fail("manifest contract commit mismatch")
    if require_hex(
        provenance["tokenizer_runtime_commit"], 40, "provenance.tokenizer_runtime_commit"
    ) != supplied_tokenizer:
        fail("manifest tokenizer runtime commit mismatch")
    if require_hex(provenance["generator_sha256"], 64, "provenance.generator_sha256") != GENERATOR_SHA256:
        fail("manifest generator SHA-256 mismatch")
    if require_hex(provenance["benchmark_sha256"], 64, "provenance.benchmark_sha256") != BENCHMARK_SHA256:
        fail("manifest benchmark SHA-256 mismatch")
    if require_int(
        provenance["poolside_seed_token_count"], "provenance.poolside_seed_token_count"
    ) != 61440:
        fail("Poolside benchmark seed token count mismatch")
    if require_int(
        provenance["ds4_seed_token_count"], "provenance.ds4_seed_token_count"
    ) < 32768:
        fail("DS4 benchmark seed has fewer than 32768 tokens")

    thresholds = require_object(manifest["thresholds"], "manifest.thresholds")
    require_exact_keys(thresholds, {"cuda_admission"}, "manifest.thresholds")
    validate_cuda_limits(thresholds["cuda_admission"], "manifest.thresholds.cuda_admission")

    benchmark = fixed["benchmark-32768.txt"]
    manifest_cases = require_list(manifest["cases"], "manifest.cases")
    if len(manifest_cases) != len(CASE_SPECS):
        fail("manifest must contain exactly four cases")
    case_argmax: dict[str, int] = {}
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        case = require_object(manifest_cases[index], f"manifest.cases[{index}]")
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
        if case["id"] != case_id or case["render"] != render:
            fail(f"manifest case {index} identity mismatch")
        frontier = require_int(case["frontier"], f"{case_id}.frontier")
        if frontier <= 0 or (fixed_frontier is not None and frontier != fixed_frontier):
            fail(f"{case_id}: frontier mismatch")
        if require_int(case["context"], f"{case_id}.context") != context:
            fail(f"{case_id}: context mismatch")
        prompt = decode_canonical_hex(case["prompt_hex"], f"{case_id}.prompt_hex")
        if require_hex(case["prompt_sha256"], 64, f"{case_id}.prompt_sha256") != sha256_bytes(prompt):
            fail(f"{case_id}: prompt SHA-256 mismatch")
        require_utf8(prompt, f"{case_id} prompt")
        if case_id == "short":
            expected_prompt = SHORT_PREFIX + fixed["short.txt"] + SHORT_SUFFIX
        else:
            expected_prompt = read_bytes(root / f"{case_id}.prompt")
            require_utf8(expected_prompt, f"{case_id}.prompt")
            if len(expected_prompt) > len(benchmark) or benchmark[: len(expected_prompt)] != expected_prompt:
                fail(f"{case_id}: prompt is not the deterministic benchmark prefix")
        if prompt != expected_prompt:
            fail(f"{case_id}: manifest prompt bytes differ from the fixture input")
        poolside_tokens = require_token_list(
            case["poolside_tokens"], frontier, f"{case_id}.poolside_tokens"
        )
        ds4_tokens = require_token_list(case["ds4_tokens"], frontier, f"{case_id}.ds4_tokens")
        if poolside_tokens != ds4_tokens:
            fail(f"{case_id}: Poolside and DS4 token arrays differ")

        vector = require_object(case["vector"], f"{case_id}.vector")
        require_exact_keys(vector, {"file", "sha256", "argmax"}, f"{case_id}.vector")
        name = require_safe_name(
            vector["file"], f"{case_id}.llama.f32", f"{case_id}.vector.file"
        )
        payload = read_bytes(root / name)
        expected_sha = require_hex(vector["sha256"], 64, f"{case_id}.vector.sha256")
        if sha256_bytes(payload) != expected_sha:
            fail(f"{name}: SHA-256 mismatch")
        values = decode_f32_payload(payload, name)
        actual_argmax = vector_argmax(values)
        if require_int(vector["argmax"], f"{case_id}.vector.argmax") != actual_argmax:
            fail(f"{case_id}: vector argmax mismatch")
        case_argmax[case_id] = actual_argmax

    continuation = require_object(manifest["continuation"], "manifest.continuation")
    require_exact_keys(continuation, {"case", "file", "sha256", "argmax"}, "manifest.continuation")
    if continuation["case"] != "yarn-8193":
        fail("manifest continuation case mismatch")
    continuation_name = require_safe_name(
        continuation["file"], "yarn-8193.continuation.i32", "manifest.continuation.file"
    )
    continuation_payload = read_bytes(root / continuation_name)
    if require_hex(continuation["sha256"], 64, "manifest.continuation.sha256") != sha256_bytes(
        continuation_payload
    ):
        fail("continuation SHA-256 mismatch")
    continuation_ids = decode_i32_payload(
        continuation_payload, CONTINUATION_TOKENS, continuation_name
    )
    recorded_ids = require_token_list(
        continuation["argmax"], CONTINUATION_TOKENS, "manifest.continuation.argmax"
    )
    if continuation_ids != recorded_ids:
        fail("continuation payload and recorded IDs differ")
    if continuation_ids[0] != case_argmax["yarn-8193"]:
        fail("continuation step zero differs from the yarn terminal argmax")


def validate_capture(root: Path, expected_capture_sha256: str) -> dict[str, Any]:
    root = Path(root)
    capture_payload = read_bytes(root / "capture.json")
    if sha256_bytes(capture_payload) != expected_capture_sha256:
        fail("Poolside capture.json trust-anchor SHA-256 mismatch")
    capture = load_json_bytes(capture_payload, "capture.json")
    found = actual_files(root)
    expected_files = CAPTURE_ARTIFACT_FILES | {"capture.json"}
    if found != expected_files:
        fail(
            f"capture file set mismatch: missing={sorted(expected_files - found)} "
            f"extra={sorted(found - expected_files)}"
        )
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
        "capture",
    )
    if capture["schema"] != "laguna-resident-capture-v1":
        fail("capture schema mismatch")
    if capture["oracle"] != "llama":
        fail("capture oracle mismatch")
    if require_hex(capture["runtime_commit"], 40, "capture.runtime_commit") != LLAMA_COMMIT:
        fail("capture runtime commit mismatch")
    if require_int(capture["vocab_size"], "capture.vocab_size") != VOCAB_SIZE:
        fail("capture vocabulary mismatch")
    if require_int(capture["seed_token_count"], "capture.seed_token_count") != 61440:
        fail("capture seed token count mismatch")
    model = require_object(capture["model"], "capture.model")
    require_exact_keys(model, set(CAPTURE_MODEL), "capture.model")
    if model != CAPTURE_MODEL:
        fail("capture model identity mismatch")

    file_hashes = require_object(capture["files"], "capture.files")
    require_exact_keys(file_hashes, set(CAPTURE_ARTIFACT_FILES), "capture.files")
    retained: dict[str, bytes] = {}
    for name in sorted(CAPTURE_ARTIFACT_FILES):
        expected_sha = require_hex(file_hashes[name], 64, f"capture.files.{name}")
        payload = read_bytes(root / name)
        if sha256_bytes(payload) != expected_sha:
            fail(f"capture artifact SHA-256 mismatch: {name}")
        retained[name] = payload

    cases_value = require_list(capture["cases"], "capture.cases")
    if len(cases_value) != len(CASE_SPECS):
        fail("capture must contain exactly four cases")
    retained_cases: list[dict[str, Any]] = []
    primary_argmax: dict[str, int] = {}
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        case = require_object(cases_value[index], f"capture.cases[{index}]")
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
            f"capture.cases[{index}]",
        )
        expected_prompt_ref = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        if (
            case["id"] != case_id
            or case["render"] != render
            or case["prompt"] != expected_prompt_ref
        ):
            fail(f"capture case {index} identity mismatch")
        frontier = require_int(case["frontier"], f"capture.{case_id}.frontier")
        if frontier <= 0 or (fixed_frontier is not None and frontier != fixed_frontier):
            fail(f"capture {case_id} frontier mismatch")
        if require_int(case["context"], f"capture.{case_id}.context") != context:
            fail(f"capture {case_id} context mismatch")
        if require_int(case["token_count"], f"capture.{case_id}.token_count") != frontier:
            fail(f"capture {case_id} token count mismatch")
        prompt_name = require_safe_name(
            case["prompt_file"], f"{case_id}.prompt", f"capture.{case_id}.prompt_file"
        )
        tokens_name = require_safe_name(
            case["tokens_file"], f"{case_id}.tokens.i32", f"capture.{case_id}.tokens_file"
        )
        logits_name = require_safe_name(
            case["logits_file"], f"{case_id}.logits.f32", f"capture.{case_id}.logits_file"
        )
        prompt = retained[prompt_name]
        require_utf8(prompt, prompt_name)
        tokens = decode_i32_payload(retained[tokens_name], frontier, tokens_name)
        logits = decode_f32_payload(retained[logits_name], logits_name)
        actual_argmax = vector_argmax(logits)
        if require_int(case["argmax"], f"capture.{case_id}.argmax") != actual_argmax:
            fail(f"capture {case_id} argmax mismatch")
        primary_argmax[case_id] = actual_argmax
        retained_cases.append(
            {
                "id": case_id,
                "render": render,
                "frontier": frontier,
                "context": context,
                "prompt": prompt,
                "tokens": tokens,
                "logits": retained[logits_name],
                "argmax": actual_argmax,
            }
        )

    continuation = require_object(capture["continuation"], "capture.continuation")
    require_exact_keys(
        continuation, {"case", "tokens_file", "logits_files", "argmax"}, "capture.continuation"
    )
    if continuation["case"] != "yarn-8193":
        fail("capture continuation case mismatch")
    tokens_name = require_safe_name(
        continuation["tokens_file"],
        "yarn-8193.continuation.i32",
        "capture.continuation.tokens_file",
    )
    logits_files = require_list(continuation["logits_files"], "capture.continuation.logits_files")
    expected_step_files = [f"yarn-8193.step-{step:02d}.logits.f32" for step in range(8)]
    if logits_files != expected_step_files:
        fail("capture continuation logit file order mismatch")
    continuation_ids = decode_i32_payload(
        retained[tokens_name], CONTINUATION_TOKENS, tokens_name
    )
    recorded_ids = require_token_list(
        continuation["argmax"], CONTINUATION_TOKENS, "capture.continuation.argmax"
    )
    if continuation_ids != recorded_ids:
        fail("capture continuation payload and recorded IDs differ")
    for step, name in enumerate(expected_step_files):
        logits = decode_f32_payload(retained[name], name)
        if vector_argmax(logits) != continuation_ids[step]:
            fail(f"capture continuation step {step} argmax mismatch")
    if continuation_ids[0] != primary_argmax["yarn-8193"]:
        fail("capture continuation step zero differs from the yarn terminal argmax")
    return {
        "manifest": capture,
        "files": retained,
        "cases": retained_cases,
        "continuation": continuation_ids,
        "continuation_payload": retained[tokens_name],
    }


def validate_destination(
    destination: Path, capture_data: dict[str, Any], *, reject_outputs: bool
) -> dict[str, bytes]:
    found = actual_files(destination)
    unknown = found - PROMOTED_FILES
    if unknown:
        fail(f"destination has unknown files: {sorted(unknown)}")
    missing_fixed = FIXED_INPUT_FILES - found
    if missing_fixed:
        fail(f"destination is missing fixed inputs: {sorted(missing_fixed)}")
    if reject_outputs:
        if "manifest.json" in found:
            fail(f"destination already contains manifest: {destination / 'manifest.json'}")
        stale = sorted(
            destination / name
            for name in (PROMOTED_VECTOR_FILES | {"yarn-8193.continuation.i32"})
            if name in found
        )
        if stale:
            fail("stale partial promotion outputs: " + ", ".join(str(path) for path in stale))
    fixed = validate_fixture_inputs(destination)
    capture_files = capture_data["files"]
    for name in sorted(MATERIALIZED_PROMPT_FILES & found):
        payload = read_bytes(destination / name)
        if payload != capture_files[name]:
            fail(f"existing prompt differs from the capture: {destination / name}")
    return fixed


def validate_capture_relationships(
    capture_data: dict[str, Any], fixed: dict[str, bytes]
) -> None:
    cases = {case["id"]: case for case in capture_data["cases"]}
    expected_short = SHORT_PREFIX + fixed["short.txt"] + SHORT_SUFFIX
    if cases["short"]["prompt"] != expected_short:
        fail("capture short prompt differs from the fixed rendered prompt")
    benchmark = fixed["benchmark-32768.txt"]
    deep_tokens = cases["deep-32768"]["tokens"]
    for case_id in ("swa-513", "yarn-8193", "deep-32768"):
        case = cases[case_id]
        prompt = case["prompt"]
        require_utf8(prompt, f"capture {case_id} prompt")
        if len(prompt) > len(benchmark) or benchmark[: len(prompt)] != prompt:
            fail(f"capture {case_id} prompt is not the deterministic benchmark prefix")
        frontier = case["frontier"]
        if case["tokens"] != deep_tokens[:frontier]:
            fail(f"capture {case_id} tokens differ from the deep-token prefix")


def run_git(repo_or_path: Path, arguments: list[str], label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_or_path), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        fail(f"{label} failed: {exc}")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"{label} failed: {detail}")
    return completed.stdout.strip()


def discover_repository(ds4: Path) -> tuple[Path, Path, str]:
    try:
        executable = Path(ds4).resolve(strict=True)
        mode = executable.lstat().st_mode
    except (OSError, RuntimeError) as exc:
        fail(f"cannot resolve DS4 executable: {exc}")
    if not stat.S_ISREG(mode) or not os.access(executable, os.X_OK):
        fail(f"DS4 executable is not an executable regular file: {executable}")
    top = run_git(executable.parent, ["rev-parse", "--show-toplevel"], "DS4 repository discovery")
    repository = Path(top)
    head = run_git(repository, ["rev-parse", "HEAD"], "DS4 HEAD discovery")
    require_hex(head, 40, "DS4 HEAD")
    if run_git(
        repository,
        ["status", "--short", "--untracked-files=no"],
        "DS4 tracked-clean check",
    ):
        fail("DS4 repository has tracked changes")
    contract = require_hex(CONTRACT_COMMIT, 40, "pinned contract commit")
    run_git(
        repository,
        ["cat-file", "-e", f"{contract}^{{commit}}"],
        "contract commit resolution",
    )
    run_git(
        repository,
        ["merge-base", "--is-ancestor", contract, head],
        "contract ancestry check",
    )
    return executable, repository, head


def recheck_repository(repository: Path, expected_head: str) -> None:
    head = run_git(repository, ["rev-parse", "HEAD"], "DS4 HEAD recheck")
    require_hex(head, 40, "DS4 HEAD")
    if head != expected_head:
        fail("DS4 HEAD changed during promotion")
    if run_git(
        repository,
        ["status", "--short", "--untracked-files=no"],
        "DS4 tracked-clean recheck",
    ):
        fail("DS4 repository became dirty during promotion")


def parse_tokenizer_output(output: str, label: str) -> list[int]:
    lines = output.splitlines()
    machine_readable = lines[0].strip() if lines else ""
    try:
        value = ast.literal_eval(machine_readable)
    except (SyntaxError, ValueError) as exc:
        fail(f"{label}: invalid token dump: {exc}")
    items = require_list(value, label)
    tokens = [require_int(item, f"{label}[{index}]") for index, item in enumerate(items)]
    if not tokens:
        fail(f"{label}: token dump is empty")
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        fail(f"{label}: token ID outside vocabulary")
    return tokens


def tokenize_retained_prompts(
    executable: Path,
    destination: Path,
    capture_data: dict[str, Any],
    fixed: dict[str, bytes],
) -> tuple[dict[str, list[int]], int]:
    model = os.environ.get("LAGUNA_MODEL")
    if not model:
        fail("LAGUNA_MODEL must name the pinned GGUF for tokenizer validation")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tokenizer-", dir=destination.parent)
    )
    try:
        payloads = {
            f"{case['id']}.prompt": case["prompt"] for case in capture_data["cases"]
        }
        payloads["benchmark-32768.txt"] = fixed["benchmark-32768.txt"]
        observed: dict[str, list[int]] = {}
        for name in (
            "short.prompt",
            "swa-513.prompt",
            "yarn-8193.prompt",
            "deep-32768.prompt",
            "benchmark-32768.txt",
        ):
            path = temporary / name
            try:
                path.write_bytes(payloads[name])
                completed = subprocess.run(
                    [
                        str(executable),
                        "--dump-tokens",
                        "--raw-prompt",
                        "-m",
                        model,
                        "--prompt-file",
                        str(path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                fail(f"DS4 tokenizer invocation failed for {name}: {exc}")
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                fail(f"DS4 tokenizer invocation failed for {name}: {detail}")
            observed[name] = parse_tokenizer_output(completed.stdout, f"DS4 tokens for {name}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    for case in capture_data["cases"]:
        tokens = observed[f"{case['id']}.prompt"]
        if tokens != case["tokens"]:
            fail(f"DS4 and Poolside tokens differ for {case['id']}")
    benchmark_tokens = observed["benchmark-32768.txt"]
    if len(benchmark_tokens) < 32768:
        fail("DS4 benchmark token dump is shorter than 32768 tokens")
    deep_tokens = next(
        case["tokens"] for case in capture_data["cases"] if case["id"] == "deep-32768"
    )
    if benchmark_tokens[:32768] != deep_tokens:
        fail("DS4 benchmark tokens differ from the deep capture prefix")
    return observed, len(benchmark_tokens)


def build_manifest(
    capture_data: dict[str, Any],
    capture_sha256: str,
    tokenizer_runtime_commit: str,
    ds4_seed_token_count: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in capture_data["cases"]:
        case_id = case["id"]
        prompt = case["prompt"]
        vector_name = f"{case_id}.llama.f32"
        cases.append(
            {
                "id": case_id,
                "render": case["render"],
                "prompt_hex": prompt.hex(),
                "prompt_sha256": sha256_bytes(prompt),
                "poolside_tokens": list(case["tokens"]),
                "ds4_tokens": list(case["tokens"]),
                "frontier": case["frontier"],
                "context": case["context"],
                "vector": {
                    "file": vector_name,
                    "sha256": sha256_bytes(case["logits"]),
                    "argmax": case["argmax"],
                },
            }
        )
    continuation_payload = capture_data["continuation_payload"]
    return {
        "schema": "laguna-resident-promoted-v2",
        "oracle_policy": "single-poolside-v1",
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": CONTINUATION_TOKENS,
        "model": dict(PROMOTED_MODEL),
        "oracle": {
            "name": "poolside-llama",
            "runtime_commit": LLAMA_COMMIT,
            "capture_manifest_sha256": capture_sha256,
        },
        "provenance": {
            "contract_commit": CONTRACT_COMMIT,
            "tokenizer_runtime_commit": tokenizer_runtime_commit,
            "generator_sha256": GENERATOR_SHA256,
            "benchmark_sha256": BENCHMARK_SHA256,
            "poolside_seed_token_count": 61440,
            "ds4_seed_token_count": ds4_seed_token_count,
        },
        "thresholds": {"cuda_admission": dict(CUDA_LIMITS)},
        "cases": cases,
        "continuation": {
            "case": "yarn-8193",
            "file": "yarn-8193.continuation.i32",
            "sha256": sha256_bytes(continuation_payload),
            "argmax": list(capture_data["continuation"]),
        },
    }


def json_payload(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_stage(
    stage: Path,
    fixed: dict[str, bytes],
    capture_data: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    try:
        for name, payload in fixed.items():
            (stage / name).write_bytes(payload)
        for name in MATERIALIZED_PROMPT_FILES:
            (stage / name).write_bytes(capture_data["files"][name])
        for case in capture_data["cases"]:
            (stage / f"{case['id']}.llama.f32").write_bytes(case["logits"])
        (stage / "yarn-8193.continuation.i32").write_bytes(
            capture_data["continuation_payload"]
        )
        (stage / "manifest.json").write_bytes(json_payload(manifest))
    except OSError as exc:
        fail(f"cannot stage promoted fixture: {exc}")


def clean_published(created: list[tuple[Path, int, int]]) -> None:
    for target, device, inode in reversed(created):
        try:
            status = os.lstat(target)
            if stat.S_ISREG(status.st_mode) and status.st_dev == device and status.st_ino == inode:
                target.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def promote(
    ds4: Path,
    llama_root: Path,
    destination: Path,
    *,
    expected_capture_sha256: str = CAPTURE_MANIFEST_SHA256,
) -> None:
    destination = Path(destination)
    capture_data = validate_capture(Path(llama_root), expected_capture_sha256)
    fixed = validate_destination(destination, capture_data, reject_outputs=False)
    validate_capture_relationships(capture_data, fixed)
    executable, repository, head = discover_repository(Path(ds4))
    _, ds4_seed_token_count = tokenize_retained_prompts(
        executable, destination, capture_data, fixed
    )
    manifest = build_manifest(
        capture_data, expected_capture_sha256, head, ds4_seed_token_count
    )

    lock = destination.with_name(f".{destination.name}.lock")
    acquired = False
    stage: Path | None = None
    try:
        try:
            os.mkdir(lock)
            acquired = True
        except FileExistsError:
            fail(f"promotion lock is already held: {lock}")
        except OSError as exc:
            fail(f"cannot acquire promotion lock {lock}: {exc}")

        fixed = validate_destination(destination, capture_data, reject_outputs=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        write_stage(stage, fixed, capture_data, manifest)
        verify_promoted(
            stage,
            CONTRACT_COMMIT,
            head,
            LLAMA_COMMIT,
            expected_capture_sha256,
            int(PROMOTED_MODEL["size"]),
            str(PROMOTED_MODEL["sha256"]),
        )
        recheck_repository(repository, head)

        destination_files = actual_files(destination)
        publication_names = [
            *sorted(MATERIALIZED_PROMPT_FILES - destination_files),
            *sorted(PROMOTED_VECTOR_FILES),
            "yarn-8193.continuation.i32",
            "manifest.json",
        ]
        created: list[tuple[Path, int, int]] = []
        try:
            for name in publication_names:
                source = stage / name
                target = destination / name
                source_status = os.lstat(source)
                os.link(source, target)
                created.append((target, source_status.st_dev, source_status.st_ino))
        except Exception as exc:
            clean_published(created)
            if isinstance(exc, ContractError):
                raise
            fail(f"publication failed: {exc}")
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if acquired:
            try:
                os.rmdir(lock)
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
    promotion_values = (args.ds4, args.llama, args.promote)
    try:
        if verify_mode:
            if not all(value is not None for value in verify_values) or any(
                value is not None for value in promotion_values
            ):
                fail("verify mode requires all identity arguments and no promotion arguments")
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
        if not all(value is not None for value in promotion_values) or any(
            value is not None for value in verify_values
        ):
            fail("promotion mode requires --ds4, --llama, and --promote only")
        promote(
            args.ds4,
            args.llama,
            args.promote,
            expected_capture_sha256=CAPTURE_MANIFEST_SHA256,
        )
        print(f"promoted={args.promote} cases=4 vectors=4 oracle=poolside")
        return 0
    except ContractError as exc:
        mode = "verification" if verify_mode else "promotion"
        print(f"{mode} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
