#!/usr/bin/env python3
"""Promote and verify the pinned Laguna Metal/Poolside oracle fixtures."""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


VOCAB_SIZE = 100352
VECTOR_BYTES = VOCAB_SIZE * 4
CONTINUATION_TOKENS = 8
CONTINUATION_BYTES = CONTINUATION_TOKENS * 4
GENERATOR_SHA256 = "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"
BENCHMARK_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"

MODEL = {
    "repository": "poolside/Laguna-S-2.1-GGUF",
    "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
    "file": "laguna-s-2.1-Q4_K_M.gguf",
    "size": 68248759648,
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
}
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"

CASE_SPECS = (
    ("short", "laguna-ds4", None, 1024),
    ("swa-513", "raw", 513, 1024),
    ("yarn-8193", "raw", 8193, 8202),
    ("deep-32768", "raw", 32768, 32768),
)
ORACLES = ("metal", "llama")
FIXTURE_INPUT_FILES = {
    "cases.json",
    "short.txt",
    "generate_benchmark_prompt.py",
    "benchmark-32768.txt",
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
}
PROMOTION_LIMITS = {
    "centered_rms": 0.02,
    "centered_max_abs": 0.10,
    "top20_overlap": 18,
    "argmax_equal": True,
    "continuation_equal": True,
}
CUDA_LIMITS = {
    "centered_rms": 0.04,
    "centered_max_abs": 0.20,
    "top20_overlap": 18,
    "argmax_equal": True,
    "teacher_forced_ids_equal": True,
}


class ContractError(RuntimeError):
    """A fixture or capture violates the pinned contract."""


def fail(message: str) -> None:
    raise ContractError(message)


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=duplicate_rejecting_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: top-level JSON value must be an object")
    return value


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_f32(values: list[float]) -> bytes:
    encoded = array.array("f", values)
    if encoded.itemsize != 4:
        fail("runtime float32 representation is not four bytes")
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def encode_i32(values: list[int]) -> bytes:
    encoded = array.array("i", values)
    if encoded.itemsize != 4:
        fail("runtime int32 representation is not four bytes")
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def read_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            fail(f"missing regular file: {path}")
        return path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def require_safe_name(value: Any, expected: str, label: str) -> str:
    name = require_string(value, label)
    if name != expected or Path(name).name != name:
        fail(f"{label} must be {expected!r}")
    return name


def decode_f32(path: Path, expected_sha: str | None = None) -> tuple[bytes, list[float]]:
    payload = read_bytes(path)
    if len(payload) != VECTOR_BYTES:
        fail(f"{path.name}: expected {VECTOR_BYTES} bytes, got {len(payload)}")
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        fail(f"{path.name}: SHA-256 mismatch")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if len(result) != VOCAB_SIZE:
        fail(f"{path.name}: expected {VOCAB_SIZE} logits")
    if not all(math.isfinite(item) for item in result):
        fail(f"{path.name}: non-finite logit")
    return payload, result


def decode_i32(path: Path, count: int, expected_sha: str | None = None) -> tuple[bytes, list[int]]:
    payload = read_bytes(path)
    expected_bytes = count * 4
    if len(payload) != expected_bytes:
        fail(f"{path.name}: expected {expected_bytes} bytes, got {len(payload)}")
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        fail(f"{path.name}: SHA-256 mismatch")
    values = array.array("i")
    values.frombytes(payload)
    if values.itemsize != 4:
        fail("runtime int32 representation is not four bytes")
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    if any(token < 0 or token >= VOCAB_SIZE for token in result):
        fail(f"{path.name}: token ID outside vocabulary")
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


def top_indices(values: list[float], count: int) -> set[int]:
    ranked = heapq.nlargest(count, range(len(values)), key=lambda index: (values[index], -index))
    return set(ranked)


def compare_logits(left: list[float], right: list[float]) -> dict[str, Any]:
    differences = [a - b for a, b in zip(left, right, strict=True)]
    mean_difference = math.fsum(differences) / len(differences)
    centered = [value - mean_difference for value in differences]
    rms = math.sqrt(math.fsum(value * value for value in centered) / len(centered))
    maximum = max(abs(value) for value in centered)
    left_argmax = argmax(left)
    right_argmax = argmax(right)
    return {
        "centered_rms": rms,
        "centered_max_abs": maximum,
        "top20_overlap": len(top_indices(left, 20) & top_indices(right, 20)),
        "argmax_equal": left_argmax == right_argmax,
    }


def check_metric_limits(metrics: dict[str, Any], label: str) -> None:
    if require_number(metrics.get("centered_rms"), f"{label}.centered_rms") > 0.02:
        fail(f"{label}: centered RMS exceeds 0.02")
    if require_number(metrics.get("centered_max_abs"), f"{label}.centered_max_abs") > 0.10:
        fail(f"{label}: centered max absolute error exceeds 0.10")
    if require_int(metrics.get("top20_overlap"), f"{label}.top20_overlap") < 18:
        fail(f"{label}: top-20 overlap is below 18")
    if not require_bool(metrics.get("argmax_equal"), f"{label}.argmax_equal"):
        fail(f"{label}: oracle argmax differs")


def require_model(value: Any, label: str) -> dict[str, Any]:
    model = require_object(value, label)
    require_exact_keys(model, set(MODEL), label)
    if model != MODEL:
        fail(f"{label}: pinned model identity mismatch")
    return model


def require_commit(value: Any, label: str) -> str:
    return require_hex(value, 40, label)


def expected_promoted_files() -> set[str]:
    files = set(FIXTURE_INPUT_FILES) | {"manifest.json", "yarn-8193.continuation.i32"}
    for case_id, _, _, _ in CASE_SPECS:
        for oracle in ORACLES:
            files.add(f"{case_id}.{oracle}.f32")
    return files


def expected_promotion_outputs() -> set[str]:
    return expected_promoted_files() - FIXTURE_INPUT_FILES


def actual_files(root: Path) -> set[str]:
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        fail(f"cannot list {root}: {exc}")
    files: set[str] = set()
    for entry in entries:
        if entry.name == "__pycache__" and entry.is_dir() and not entry.is_symlink():
            continue
        if entry.is_symlink() or not entry.is_file():
            fail(f"{root}: fixture must contain regular files only")
        files.add(entry.name)
    return files


def validate_fixture_inputs(root: Path) -> None:
    short = read_bytes(root / "short.txt")
    if short != b"Explain why a ring buffer wraps, in two sentences.\n":
        fail("short.txt does not match the fixed contract")
    generator = read_bytes(root / "generate_benchmark_prompt.py")
    benchmark = read_bytes(root / "benchmark-32768.txt")
    if sha256_bytes(generator) != GENERATOR_SHA256:
        fail("prompt generator SHA-256 mismatch")
    if sha256_bytes(benchmark) != BENCHMARK_SHA256:
        fail("benchmark seed SHA-256 mismatch")

    cases = load_json(root / "cases.json")
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
    if require_int(cases["continuation_tokens"], "cases.json.continuation_tokens") != CONTINUATION_TOKENS:
        fail("cases.json continuation length mismatch")
    entries = require_list(cases["cases"], "cases.json.cases")
    if len(entries) != len(CASE_SPECS):
        fail("cases.json must contain exactly four cases")
    for index, (case_id, render, frontier, context) in enumerate(CASE_SPECS):
        entry = require_object(entries[index], f"cases.json.cases[{index}]")
        expected_keys = {"id", "render", "prompt", "ctx"}
        if frontier is not None:
            expected_keys.add("frontier")
        require_exact_keys(entry, expected_keys, f"cases.json.cases[{index}]")
        expected_prompt = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        if entry["id"] != case_id or entry["render"] != render or entry["prompt"] != expected_prompt:
            fail(f"cases.json case {index} identity mismatch")
        if require_int(entry["ctx"], f"cases.json.{case_id}.ctx") != context:
            fail(f"cases.json {case_id} context mismatch")
        if frontier is not None and require_int(
            entry["frontier"], f"cases.json.{case_id}.frontier"
        ) != frontier:
            fail(f"cases.json {case_id} frontier mismatch")


def verify_promoted(
    root: Path,
    dwarfstar_commit: str,
    llama_commit: str,
    gguf_size: int,
    gguf_sha256: str,
) -> None:
    if not root.is_dir():
        fail(f"promoted fixture directory does not exist: {root}")
    expected_files = expected_promoted_files()
    found_files = actual_files(root)
    if found_files != expected_files:
        fail(
            f"promoted file set mismatch: missing={sorted(expected_files - found_files)} "
            f"extra={sorted(found_files - expected_files)}"
        )
    validate_fixture_inputs(root)

    manifest = load_json(root / "manifest.json")
    require_exact_keys(
        manifest,
        {
            "schema",
            "vocab_size",
            "continuation_case",
            "continuation_tokens",
            "model",
            "runtimes",
            "dwarfstar_commit",
            "thresholds",
            "seed",
            "cases",
            "continuation",
        },
        "manifest",
    )
    if manifest["schema"] != "laguna-resident-promoted-v1":
        fail("unknown promoted manifest schema")
    if require_int(manifest["vocab_size"], "manifest.vocab_size") != VOCAB_SIZE:
        fail("manifest vocabulary size mismatch")
    if manifest["continuation_case"] != "yarn-8193":
        fail("manifest continuation case mismatch")
    if require_int(manifest["continuation_tokens"], "manifest.continuation_tokens") != CONTINUATION_TOKENS:
        fail("manifest continuation length mismatch")
    require_model(manifest["model"], "manifest.model")
    if gguf_size != MODEL["size"] or gguf_sha256 != MODEL["sha256"]:
        fail("supplied GGUF identity does not match the pinned model")

    dwarfstar_commit = require_commit(dwarfstar_commit, "supplied dwarfstar commit")
    llama_commit = require_commit(llama_commit, "supplied llama commit")
    if llama_commit != LLAMA_COMMIT:
        fail("supplied Poolside llama commit does not match the pinned runtime")
    runtimes = require_object(manifest["runtimes"], "manifest.runtimes")
    require_exact_keys(runtimes, {"metal_commit", "llama_commit"}, "manifest.runtimes")
    if require_commit(runtimes["metal_commit"], "manifest.runtimes.metal_commit") != dwarfstar_commit:
        fail("DwarfStar/Metal runtime commit mismatch")
    if require_commit(runtimes["llama_commit"], "manifest.runtimes.llama_commit") != llama_commit:
        fail("Poolside llama runtime commit mismatch")
    if require_commit(manifest["dwarfstar_commit"], "manifest.dwarfstar_commit") != dwarfstar_commit:
        fail("DwarfStar contract commit mismatch")

    thresholds = require_object(manifest["thresholds"], "manifest.thresholds")
    require_exact_keys(thresholds, {"promotion", "cuda_admission"}, "manifest.thresholds")
    if require_object(thresholds["promotion"], "manifest.thresholds.promotion") != PROMOTION_LIMITS:
        fail("promotion thresholds differ from the fixed contract")
    if require_object(thresholds["cuda_admission"], "manifest.thresholds.cuda_admission") != CUDA_LIMITS:
        fail("CUDA admission thresholds differ from the fixed contract")

    seed = require_object(manifest["seed"], "manifest.seed")
    require_exact_keys(
        seed,
        {
            "generator_sha256",
            "benchmark_sha256",
            "poolside_token_count",
            "dwarfstar_token_count",
        },
        "manifest.seed",
    )
    if require_hex(seed["generator_sha256"], 64, "manifest.seed.generator_sha256") != GENERATOR_SHA256:
        fail("manifest generator SHA-256 mismatch")
    if require_hex(seed["benchmark_sha256"], 64, "manifest.seed.benchmark_sha256") != BENCHMARK_SHA256:
        fail("manifest benchmark SHA-256 mismatch")
    if require_int(seed["poolside_token_count"], "manifest.seed.poolside_token_count") < 32768:
        fail("Poolside benchmark seed has fewer than 32768 tokens")
    if require_int(seed["dwarfstar_token_count"], "manifest.seed.dwarfstar_token_count") < 32768:
        fail("DwarfStar benchmark seed has fewer than 32768 tokens")

    cases = require_list(manifest["cases"], "manifest.cases")
    if len(cases) != len(CASE_SPECS):
        fail("manifest must contain exactly four cases")
    case_argmaxes: dict[str, dict[str, int]] = {}
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        case = require_object(cases[index], f"manifest.cases[{index}]")
        require_exact_keys(
            case,
            {
                "id",
                "render",
                "prompt_hex",
                "prompt_sha256",
                "metal_tokens",
                "llama_tokens",
                "frontier",
                "context",
                "oracles",
                "metrics",
            },
            f"manifest.cases[{index}]",
        )
        if case["id"] != case_id or case["render"] != render:
            fail(f"manifest case {index} identity/render mismatch")
        frontier = require_int(case["frontier"], f"{case_id}.frontier")
        if frontier <= 0 or (fixed_frontier is not None and frontier != fixed_frontier):
            fail(f"{case_id}: invalid frontier")
        if require_int(case["context"], f"{case_id}.context") != context:
            fail(f"{case_id}: context mismatch")
        prompt_hex = require_string(case["prompt_hex"], f"{case_id}.prompt_hex")
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
        if require_hex(case["prompt_sha256"], 64, f"{case_id}.prompt_sha256") != sha256_bytes(prompt):
            fail(f"{case_id}: prompt SHA-256 mismatch")
        if case_id == "short":
            expected_prompt = (
                b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
                + read_bytes(root / "short.txt")
                + b"</user>\n<assistant></think>"
            )
        else:
            expected_prompt = read_bytes(root / f"{case_id}.prompt")
        if prompt != expected_prompt:
            fail(f"{case_id}: manifest prompt bytes differ from checked-in input")
        metal_tokens = validate_token_list(case["metal_tokens"], frontier, f"{case_id}.metal_tokens")
        llama_tokens = validate_token_list(case["llama_tokens"], frontier, f"{case_id}.llama_tokens")
        if metal_tokens != llama_tokens:
            fail(f"{case_id}: Metal and Poolside token arrays differ")

        oracle_entries = require_object(case["oracles"], f"{case_id}.oracles")
        require_exact_keys(oracle_entries, set(ORACLES), f"{case_id}.oracles")
        vectors: dict[str, list[float]] = {}
        case_argmaxes[case_id] = {}
        for oracle in ORACLES:
            entry = require_object(oracle_entries[oracle], f"{case_id}.{oracle}")
            require_exact_keys(entry, {"file", "sha256", "argmax"}, f"{case_id}.{oracle}")
            name = require_safe_name(entry["file"], f"{case_id}.{oracle}.f32", f"{case_id}.{oracle}.file")
            expected_sha = require_hex(entry["sha256"], 64, f"{case_id}.{oracle}.sha256")
            _, vector = decode_f32(root / name, expected_sha)
            recorded_argmax = require_int(entry["argmax"], f"{case_id}.{oracle}.argmax")
            if recorded_argmax != argmax(vector):
                fail(f"{case_id}.{oracle}: recorded argmax mismatch")
            vectors[oracle] = vector
            case_argmaxes[case_id][oracle] = recorded_argmax

        recorded_metrics = require_object(case["metrics"], f"{case_id}.metrics")
        require_exact_keys(
            recorded_metrics,
            {"centered_rms", "centered_max_abs", "top20_overlap", "argmax_equal"},
            f"{case_id}.metrics",
        )
        computed_metrics = compare_logits(vectors["metal"], vectors["llama"])
        for name in ("centered_rms", "centered_max_abs"):
            recorded = require_number(recorded_metrics[name], f"{case_id}.metrics.{name}")
            if not math.isclose(recorded, float(computed_metrics[name]), rel_tol=1e-7, abs_tol=1e-9):
                fail(f"{case_id}: recorded {name} does not match vectors")
        for name in ("top20_overlap", "argmax_equal"):
            if recorded_metrics[name] != computed_metrics[name]:
                fail(f"{case_id}: recorded {name} does not match vectors")
        check_metric_limits(recorded_metrics, f"{case_id}.metrics")

    continuation = require_object(manifest["continuation"], "manifest.continuation")
    require_exact_keys(
        continuation,
        {"case", "file", "sha256", "metal_argmax", "llama_argmax"},
        "manifest.continuation",
    )
    if continuation["case"] != "yarn-8193":
        fail("continuation case mismatch")
    name = require_safe_name(
        continuation["file"],
        "yarn-8193.continuation.i32",
        "manifest.continuation.file",
    )
    expected_sha = require_hex(continuation["sha256"], 64, "manifest.continuation.sha256")
    _, continuation_ids = decode_i32(root / name, CONTINUATION_TOKENS, expected_sha)
    metal_ids = validate_token_list(
        continuation["metal_argmax"], CONTINUATION_TOKENS, "manifest.continuation.metal_argmax"
    )
    llama_ids = validate_token_list(
        continuation["llama_argmax"], CONTINUATION_TOKENS, "manifest.continuation.llama_argmax"
    )
    if metal_ids != llama_ids or metal_ids != continuation_ids:
        fail("Metal, Poolside, and promoted continuation IDs differ")
    for oracle, ids in (("metal", metal_ids), ("llama", llama_ids)):
        if case_argmaxes["yarn-8193"][oracle] != ids[0]:
            fail(f"{oracle}: YaRN frontier argmax differs from continuation step zero")


def validate_capture(root: Path, expected_oracle: str) -> dict[str, Any]:
    if not root.is_dir():
        fail(f"capture directory does not exist: {root}")
    capture = load_json(root / "capture.json")
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
            *( {"dwarfstar_commit"} if expected_oracle == "metal" else set() ),
        },
        f"{expected_oracle} capture",
    )
    if capture["schema"] != "laguna-resident-capture-v1":
        fail(f"{expected_oracle}: unknown capture schema")
    if capture["oracle"] != expected_oracle:
        fail(f"{expected_oracle}: capture oracle identity mismatch")
    runtime_commit = require_commit(capture["runtime_commit"], f"{expected_oracle}.runtime_commit")
    if expected_oracle == "llama" and runtime_commit != LLAMA_COMMIT:
        fail("Poolside capture runtime commit does not match the pinned revision")
    if expected_oracle == "metal":
        if require_commit(capture["dwarfstar_commit"], "metal.dwarfstar_commit") != runtime_commit:
            fail("Metal runtime and DwarfStar commits differ")
    if require_int(capture["vocab_size"], f"{expected_oracle}.vocab_size") != VOCAB_SIZE:
        fail(f"{expected_oracle}: vocabulary size mismatch")
    seed_token_count = require_int(
        capture["seed_token_count"], f"{expected_oracle}.seed_token_count"
    )
    if seed_token_count < 32768:
        fail(f"{expected_oracle}: benchmark seed has fewer than 32768 tokens")
    require_model(capture["model"], f"{expected_oracle}.model")

    files = require_object(capture["files"], f"{expected_oracle}.files")
    expected_names = set(files) | {"capture.json"}
    if actual_files(root) != expected_names:
        fail(f"{expected_oracle}: capture file set mismatch")
    for name, recorded_sha in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            fail(f"{expected_oracle}: unsafe capture filename")
        expected_sha = require_hex(recorded_sha, 64, f"{expected_oracle}.files[{name}]")
        if sha256_bytes(read_bytes(root / name)) != expected_sha:
            fail(f"{expected_oracle}: SHA-256 mismatch for {name}")

    cases = require_list(capture["cases"], f"{expected_oracle}.cases")
    if len(cases) != len(CASE_SPECS):
        fail(f"{expected_oracle}: expected four cases")
    decoded_cases = []
    for index, (case_id, render, fixed_frontier, context) in enumerate(CASE_SPECS):
        case = require_object(cases[index], f"{expected_oracle}.cases[{index}]")
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
            f"{expected_oracle}.cases[{index}]",
        )
        if case["id"] != case_id or case["render"] != render:
            fail(f"{expected_oracle}: case {index} identity/render mismatch")
        expected_prompt_ref = "short.txt" if case_id == "short" else f"{case_id}.prompt"
        if case["prompt"] != expected_prompt_ref:
            fail(f"{expected_oracle}.{case_id}: prompt reference mismatch")
        frontier = require_int(case["frontier"], f"{expected_oracle}.{case_id}.frontier")
        if frontier <= 0 or (fixed_frontier is not None and frontier != fixed_frontier):
            fail(f"{expected_oracle}.{case_id}: invalid frontier")
        if require_int(case["context"], f"{expected_oracle}.{case_id}.context") != context:
            fail(f"{expected_oracle}.{case_id}: context mismatch")
        if require_int(case["token_count"], f"{expected_oracle}.{case_id}.token_count") != frontier:
            fail(f"{expected_oracle}.{case_id}: token count/frontier mismatch")
        prompt_name = require_safe_name(
            case["prompt_file"], f"{case_id}.prompt", f"{expected_oracle}.{case_id}.prompt_file"
        )
        tokens_name = require_safe_name(
            case["tokens_file"],
            f"{case_id}.tokens.i32",
            f"{expected_oracle}.{case_id}.tokens_file",
        )
        logits_name = require_safe_name(
            case["logits_file"],
            f"{case_id}.logits.f32",
            f"{expected_oracle}.{case_id}.logits_file",
        )
        prompt = read_bytes(root / prompt_name)
        if not prompt:
            fail(f"{expected_oracle}.{case_id}: empty prompt")
        _, tokens = decode_i32(root / tokens_name, frontier)
        logits_payload, logits = decode_f32(root / logits_name)
        recorded_argmax = require_int(case["argmax"], f"{expected_oracle}.{case_id}.argmax")
        if recorded_argmax != argmax(logits):
            fail(f"{expected_oracle}.{case_id}: argmax mismatch")
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
                "prompt_path": root / prompt_name,
            }
        )

    continuation = require_object(capture["continuation"], f"{expected_oracle}.continuation")
    require_exact_keys(
        continuation,
        {"case", "tokens_file", "logits_files", "argmax"},
        f"{expected_oracle}.continuation",
    )
    if continuation["case"] != "yarn-8193":
        fail(f"{expected_oracle}: continuation case mismatch")
    token_name = require_safe_name(
        continuation["tokens_file"],
        "yarn-8193.continuation.i32",
        f"{expected_oracle}.continuation.tokens_file",
    )
    token_payload, token_ids = decode_i32(root / token_name, CONTINUATION_TOKENS)
    recorded_ids = validate_token_list(
        continuation["argmax"], CONTINUATION_TOKENS, f"{expected_oracle}.continuation.argmax"
    )
    if token_ids != recorded_ids:
        fail(f"{expected_oracle}: continuation binary/manifest mismatch")
    logits_names = require_list(continuation["logits_files"], f"{expected_oracle}.continuation.logits_files")
    if len(logits_names) != CONTINUATION_TOKENS:
        fail(f"{expected_oracle}: continuation must contain eight logit rows")
    for step, value in enumerate(logits_names):
        name = require_safe_name(
            value,
            f"yarn-8193.step-{step:02d}.logits.f32",
            f"{expected_oracle}.continuation.logits_files[{step}]",
        )
        _, logits = decode_f32(root / name)
        if argmax(logits) != recorded_ids[step]:
            fail(f"{expected_oracle}: teacher-forced argmax mismatch at step {step}")
    if recorded_ids[0] != decoded_cases[2]["argmax"]:
        fail(f"{expected_oracle}: YaRN frontier argmax differs from continuation step zero")

    return {
        "runtime_commit": runtime_commit,
        "model": capture["model"],
        "cases": decoded_cases,
        "continuation_ids": recorded_ids,
        "continuation_payload": token_payload,
        "seed_token_count": seed_token_count,
    }


def parse_dump_tokens(stdout: str, label: str) -> list[int]:
    first_line = stdout.splitlines()[0].strip() if stdout.splitlines() else ""
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
    model_path = os.environ.get("LAGUNA_MODEL", str(MODEL["file"]))
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
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        fail(f"{label}: cannot execute {ds4}: {exc}")
    if completed.returncode != 0:
        fail(f"{label}: ds4 --dump-tokens failed: {completed.stderr.strip()}")
    return parse_dump_tokens(completed.stdout, label)


def verify_ds4_tokens(ds4: Path, prompt_path: Path, expected: list[int], label: str) -> None:
    actual = dump_ds4_tokens(ds4, prompt_path, label)
    if actual != expected:
        fail(f"{label}: DwarfStar and Poolside token arrays differ")


def require_ds4_token(value: Any, label: str) -> int:
    token = require_object(value, label)
    require_exact_keys(token, {"id", "text", "bytes"}, label)
    token_id = require_int(token["id"], f"{label}.id")
    if token_id < 0 or token_id >= VOCAB_SIZE:
        fail(f"{label}.id outside vocabulary")
    require_string(token["text"], f"{label}.text")
    raw_bytes = require_list(token["bytes"], f"{label}.bytes")
    for index, item in enumerate(raw_bytes):
        byte = require_int(item, f"{label}.bytes[{index}]")
        if byte < 0 or byte > 255:
            fail(f"{label}.bytes[{index}] outside byte range")
    return token_id


def discover_dwarfstar_commit(ds4: Path) -> str:
    try:
        executable = ds4.resolve(strict=True)
        completed = subprocess.run(
            ["git", "-C", str(executable.parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        fail(f"cannot resolve DwarfStar executable: {exc}")
    if completed.returncode != 0:
        fail(f"cannot determine DwarfStar commit: {completed.stderr.strip()}")
    return require_commit(completed.stdout.strip(), "DwarfStar runtime commit")


def validate_native_metal_capture(
    root: Path, ds4: Path, llama: dict[str, Any]
) -> dict[str, Any]:
    if not root.is_dir():
        fail(f"Metal capture directory does not exist: {root}")
    expected_files = {
        *(f"{case_id}.logits.json" for case_id, _, _, _ in CASE_SPECS),
        "yarn-8193.continuation.json",
    }
    found_files = actual_files(root)
    if found_files != expected_files:
        fail(
            f"Metal capture file set mismatch: missing={sorted(expected_files - found_files)} "
            f"extra={sorted(found_files - expected_files)}"
        )

    model_env = os.environ.get("LAGUNA_MODEL")
    if not model_env:
        fail("native Metal promotion requires LAGUNA_MODEL")
    expected_model = Path(model_env).resolve()
    if expected_model.name != MODEL["file"]:
        fail("LAGUNA_MODEL filename does not match the pinned artifact")
    runtime_commit = discover_dwarfstar_commit(ds4)

    decoded_cases = []
    for (case_id, render, fixed_frontier, context), llama_case in zip(
        CASE_SPECS, llama["cases"], strict=True
    ):
        raw = load_json(root / f"{case_id}.logits.json")
        require_exact_keys(
            raw,
            {
                "source",
                "model",
                "backend",
                "quant_bits",
                "prompt_tokens",
                "ctx",
                "vocab",
                "argmax_token",
                "argmax_logit",
                "logits",
            },
            f"Metal {case_id} logits",
        )
        if raw["source"] != "ds4" or raw["backend"] != "metal":
            fail(f"Metal {case_id}: source/backend mismatch")
        recorded_model = Path(require_string(raw["model"], f"Metal {case_id}.model")).resolve()
        if recorded_model != expected_model:
            fail(f"Metal {case_id}: model path differs from LAGUNA_MODEL")
        if require_int(raw["quant_bits"], f"Metal {case_id}.quant_bits") <= 0:
            fail(f"Metal {case_id}: invalid routed quantization")
        frontier = len(llama_case["tokens"])
        if fixed_frontier is not None and frontier != fixed_frontier:
            fail(f"Metal {case_id}: Poolside frontier mismatch")
        if require_int(raw["prompt_tokens"], f"Metal {case_id}.prompt_tokens") != frontier:
            fail(f"Metal {case_id}: prompt token count mismatch")
        if require_int(raw["ctx"], f"Metal {case_id}.ctx") != context:
            fail(f"Metal {case_id}: context mismatch")
        if require_int(raw["vocab"], f"Metal {case_id}.vocab") != VOCAB_SIZE:
            fail(f"Metal {case_id}: vocabulary mismatch")
        raw_logits = require_list(raw["logits"], f"Metal {case_id}.logits")
        if len(raw_logits) != VOCAB_SIZE:
            fail(f"Metal {case_id}: expected {VOCAB_SIZE} logits")
        logits = [
            require_number(value, f"Metal {case_id}.logits[{index}]")
            for index, value in enumerate(raw_logits)
        ]
        logits_payload = encode_f32(logits)
        recorded_argmax = require_ds4_token(raw["argmax_token"], f"Metal {case_id}.argmax_token")
        if recorded_argmax != argmax(logits):
            fail(f"Metal {case_id}: recorded argmax mismatch")
        if not math.isclose(
            require_number(raw["argmax_logit"], f"Metal {case_id}.argmax_logit"),
            logits[recorded_argmax],
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            fail(f"Metal {case_id}: argmax logit mismatch")
        verify_ds4_tokens(ds4, llama_case["prompt_path"], llama_case["tokens"], case_id)
        decoded_cases.append(
            {
                "id": case_id,
                "render": render,
                "frontier": frontier,
                "context": context,
                "prompt": llama_case["prompt"],
                "tokens": llama_case["tokens"],
                "logits": logits,
                "logits_payload": logits_payload,
                "argmax": recorded_argmax,
                "prompt_path": llama_case["prompt_path"],
            }
        )

    continuation = load_json(root / "yarn-8193.continuation.json")
    require_exact_keys(
        continuation,
        {"source", "prompt_tokens", "ctx", "top_k", "steps"},
        "Metal continuation",
    )
    if continuation["source"] != "ds4":
        fail("Metal continuation source mismatch")
    if require_int(continuation["prompt_tokens"], "Metal continuation.prompt_tokens") != 8193:
        fail("Metal continuation prompt-token mismatch")
    if require_int(continuation["ctx"], "Metal continuation.ctx") != 8202:
        fail("Metal continuation context mismatch")
    if require_int(continuation["top_k"], "Metal continuation.top_k") != 20:
        fail("Metal continuation top-k mismatch")
    steps = require_list(continuation["steps"], "Metal continuation.steps")
    if len(steps) != CONTINUATION_TOKENS:
        fail("Metal continuation must contain eight steps")
    continuation_ids = []
    for index, value in enumerate(steps):
        step = require_object(value, f"Metal continuation.steps[{index}]")
        require_exact_keys(step, {"step", "selected", "top_logprobs"}, f"Metal continuation.steps[{index}]")
        if require_int(step["step"], f"Metal continuation.steps[{index}].step") != index:
            fail("Metal continuation step index mismatch")
        continuation_ids.append(
            require_ds4_token(step["selected"], f"Metal continuation.steps[{index}].selected")
        )
        require_list(step["top_logprobs"], f"Metal continuation.steps[{index}].top_logprobs")
    if continuation_ids[0] != decoded_cases[2]["argmax"]:
        fail("Metal YaRN frontier argmax differs from continuation step zero")

    return {
        "runtime_commit": runtime_commit,
        "model": MODEL,
        "cases": decoded_cases,
        "continuation_ids": continuation_ids,
        "continuation_payload": encode_i32(continuation_ids),
    }


def promote(ds4: Path, metal_root: Path, llama_root: Path, destination: Path) -> None:
    llama = validate_capture(llama_root, "llama")
    metal = (
        validate_capture(metal_root, "metal")
        if (metal_root / "capture.json").is_file()
        else validate_native_metal_capture(metal_root, ds4, llama)
    )
    if metal["model"] != llama["model"]:
        fail("Metal and Poolside model identities differ")
    if not destination.is_dir():
        fail(f"promotion destination must be the populated fixture directory: {destination}")
    found_inputs = actual_files(destination)
    if found_inputs != FIXTURE_INPUT_FILES:
        fail(
            f"pre-promotion fixture set mismatch: missing={sorted(FIXTURE_INPUT_FILES - found_inputs)} "
            f"extra={sorted(found_inputs - FIXTURE_INPUT_FILES)}"
        )
    validate_fixture_inputs(destination)
    for llama_case in llama["cases"]:
        case_id = llama_case["id"]
        if case_id != "short" and read_bytes(destination / f"{case_id}.prompt") != llama_case["prompt"]:
            fail(f"{case_id}: materialized fixture prompt differs from Poolside capture")

    dwarfstar_seed = dump_ds4_tokens(
        ds4, destination / "benchmark-32768.txt", "benchmark-32768"
    )
    if len(dwarfstar_seed) < 32768:
        fail("DwarfStar benchmark seed has fewer than 32768 tokens")
    if dwarfstar_seed[:32768] != llama["cases"][3]["tokens"]:
        fail("DwarfStar and Poolside benchmark first 32768 token IDs differ")

    manifest_cases = []
    vector_payloads: dict[str, bytes] = {}
    for spec, metal_case, llama_case in zip(CASE_SPECS, metal["cases"], llama["cases"], strict=True):
        case_id, render, _, context = spec
        if metal_case["id"] != case_id or llama_case["id"] != case_id:
            fail(f"{case_id}: capture ordering mismatch")
        if metal_case["render"] != render or llama_case["render"] != render:
            fail(f"{case_id}: capture render mismatch")
        if metal_case["context"] != context or llama_case["context"] != context:
            fail(f"{case_id}: capture context mismatch")
        if metal_case["frontier"] != llama_case["frontier"]:
            fail(f"{case_id}: capture frontiers differ")
        if metal_case["prompt"] != llama_case["prompt"]:
            fail(f"{case_id}: exact prompt bytes differ")
        if metal_case["tokens"] != llama_case["tokens"]:
            fail(f"{case_id}: Metal and Poolside token arrays differ")
        verify_ds4_tokens(ds4, metal_case["prompt_path"], llama_case["tokens"], case_id)

        metrics = compare_logits(metal_case["logits"], llama_case["logits"])
        check_metric_limits(metrics, f"{case_id}.metrics")
        oracles: dict[str, dict[str, Any]] = {}
        for oracle, case in (("metal", metal_case), ("llama", llama_case)):
            name = f"{case_id}.{oracle}.f32"
            payload = case["logits_payload"]
            vector_payloads[name] = payload
            oracles[oracle] = {
                "file": name,
                "sha256": sha256_bytes(payload),
                "argmax": case["argmax"],
            }
        prompt = metal_case["prompt"]
        manifest_cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt_hex": prompt.hex(),
                "prompt_sha256": sha256_bytes(prompt),
                "metal_tokens": metal_case["tokens"],
                "llama_tokens": llama_case["tokens"],
                "frontier": metal_case["frontier"],
                "context": context,
                "oracles": oracles,
                "metrics": metrics,
            }
        )

    metal_continuation = metal["continuation_ids"]
    llama_continuation = llama["continuation_ids"]
    if metal_continuation != llama_continuation:
        fail("Metal and Poolside teacher-forced continuations differ")
    continuation_payload = llama["continuation_payload"]
    if continuation_payload != metal["continuation_payload"]:
        fail("Metal and Poolside continuation binaries differ")
    if metal_continuation[0] != metal["cases"][2]["argmax"]:
        fail("Metal YaRN frontier argmax differs from continuation step zero")
    if llama_continuation[0] != llama["cases"][2]["argmax"]:
        fail("Poolside YaRN frontier argmax differs from continuation step zero")

    manifest = {
        "schema": "laguna-resident-promoted-v1",
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": CONTINUATION_TOKENS,
        "model": MODEL,
        "runtimes": {
            "metal_commit": metal["runtime_commit"],
            "llama_commit": llama["runtime_commit"],
        },
        "dwarfstar_commit": metal["runtime_commit"],
        "thresholds": {
            "promotion": PROMOTION_LIMITS,
            "cuda_admission": CUDA_LIMITS,
        },
        "seed": {
            "generator_sha256": GENERATOR_SHA256,
            "benchmark_sha256": BENCHMARK_SHA256,
            "poolside_token_count": llama["seed_token_count"],
            "dwarfstar_token_count": len(dwarfstar_seed),
        },
        "cases": manifest_cases,
        "continuation": {
            "case": "yarn-8193",
            "file": "yarn-8193.continuation.i32",
            "sha256": sha256_bytes(continuation_payload),
            "metal_argmax": metal_continuation,
            "llama_argmax": llama_continuation,
        },
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    published: list[Path] = []
    try:
        for name in FIXTURE_INPUT_FILES:
            shutil.copy2(destination / name, temporary / name)
        for name, payload in vector_payloads.items():
            (temporary / name).write_bytes(payload)
        (temporary / "yarn-8193.continuation.i32").write_bytes(continuation_payload)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_promoted(
            temporary,
            metal["runtime_commit"],
            llama["runtime_commit"],
            int(MODEL["size"]),
            str(MODEL["sha256"]),
        )
        output_names = expected_promotion_outputs() - {"manifest.json"}
        for name in sorted(output_names):
            target = destination / name
            os.replace(temporary / name, target)
            published.append(target)
        target = destination / "manifest.json"
        os.replace(temporary / "manifest.json", target)
        published.append(target)
    except Exception:
        for path in reversed(published):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-promoted", type=Path)
    parser.add_argument("--dwarfstar-commit")
    parser.add_argument("--llama-commit")
    parser.add_argument("--gguf-size", type=int)
    parser.add_argument("--gguf-sha256")
    parser.add_argument("--ds4", type=Path)
    parser.add_argument("--metal", type=Path)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--promote", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verify_mode = args.verify_promoted is not None
    verify_values = (args.dwarfstar_commit, args.llama_commit, args.gguf_size, args.gguf_sha256)
    promote_values = (args.ds4, args.metal, args.llama, args.promote)
    try:
        if verify_mode:
            if not all(value is not None for value in verify_values) or any(
                value is not None for value in promote_values
            ):
                fail("verify mode requires all identity arguments and no promotion arguments")
            verify_promoted(
                args.verify_promoted,
                args.dwarfstar_commit,
                args.llama_commit,
                args.gguf_size,
                args.gguf_sha256,
            )
            print(f"verified={args.verify_promoted} cases=4 vectors=8")
            return 0
        if not all(value is not None for value in promote_values) or any(
            value is not None for value in verify_values
        ):
            fail("promotion mode requires --ds4, --metal, --llama, and --promote only")
        promote(args.ds4, args.metal, args.llama, args.promote)
        print(f"promoted={args.promote} cases=4 vectors=8")
        return 0
    except ContractError as exc:
        mode = "verification" if verify_mode else "promotion"
        print(f"{mode} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
