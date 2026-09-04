#!/usr/bin/env python3
"""Compare canonical Poolside/DS4 Laguna layer captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Any


WIDTH = 3072
TOKENS = 22
LAYER_COUNT = 48
FLOAT_SIZE = 4
LAYER_VALUES = WIDTH * TOKENS
LAYER_BYTES = LAYER_VALUES * FLOAT_SIZE
LAYER0_WIDTHS = {
    "attn-norm": 3072,
    "q-proj": 6144,
    "k-proj": 1024,
    "v-proj": 1024,
    "gate-proj": 48,
    "q-rope": 6144,
    "k-rope": 1024,
    "attn-gated": 6144,
    "attn-o-proj": 3072,
    "ffn-inp": 3072,
    "ffn-norm": 3072,
    "ffn-out": 3072,
}
LAYER0_STAGES = tuple(LAYER0_WIDTHS)


class DiagnosticError(RuntimeError):
    pass


def read_canonical_f32(path: Path, expected_bytes: int | None) -> tuple[bytes, array]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DiagnosticError(f"{path}: {error.strerror or error}") from error

    if expected_bytes is not None and len(payload) != expected_bytes:
        raise DiagnosticError(
            f"{path.name}: expected {expected_bytes} bytes, got {len(payload)}"
        )
    if not payload or len(payload) % FLOAT_SIZE:
        raise DiagnosticError(
            f"{path.name}: expected a non-empty little-endian float32 file"
        )

    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return payload, values


def metrics(
    reference_path: Path,
    candidate_path: Path,
    expected_bytes: int | None,
    value_start: int = 0,
    value_count: int | None = None,
    *,
    row_width: int | None = None,
) -> dict[str, Any]:
    if row_width is not None and (type(row_width) is not int or row_width <= 0):
        raise DiagnosticError(
            f"invalid row_width {row_width!r}: expected None or a positive integer"
        )

    reference_bytes, reference = read_canonical_f32(reference_path, expected_bytes)
    candidate_bytes, candidate = read_canonical_f32(candidate_path, expected_bytes)
    if len(reference) != len(candidate):
        raise DiagnosticError(
            f"{candidate_path.name}: expected {len(reference) * FLOAT_SIZE} bytes "
            f"to match {reference_path.name}, got {len(candidate) * FLOAT_SIZE}"
        )
    value_end = len(reference) if value_count is None else value_start + value_count
    if value_start < 0 or value_end < value_start or value_end > len(reference):
        raise DiagnosticError(
            f"{reference_path.name}/{candidate_path.name}: invalid metric slice "
            f"{value_start}:{value_end} for {len(reference)} floats"
        )
    reference = reference[value_start:value_end]
    candidate = candidate[value_start:value_end]
    byte_start = value_start * FLOAT_SIZE
    byte_end = value_end * FLOAT_SIZE
    reference_bytes = reference_bytes[byte_start:byte_end]
    candidate_bytes = candidate_bytes[byte_start:byte_end]

    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    exact_hash = reference_sha256 == candidate_sha256

    # Validate selected values before the exact-byte fast path.  The raw bytes
    # are canonical little-endian F32, so bit identity is read directly rather
    # than reconstructed through host floats or a repack operation.
    first_mismatch: dict[str, Any] | None = None
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise DiagnosticError(
                f"{reference_path.name}/{candidate_path.name}: "
                f"non-finite value at float index {value_start + index}"
            )
        if first_mismatch is not None:
            continue
        byte_offset = index * FLOAT_SIZE
        reference_bits = int.from_bytes(
            reference_bytes[byte_offset : byte_offset + FLOAT_SIZE], "little"
        )
        candidate_bits = int.from_bytes(
            candidate_bytes[byte_offset : byte_offset + FLOAT_SIZE], "little"
        )
        if reference_bits != candidate_bits:
            flat_index = value_start + index
            first_mismatch = {
                "flat_index": flat_index,
                "token_index": (
                    flat_index // row_width if row_width is not None else None
                ),
                "element_index": (
                    flat_index % row_width if row_width is not None else None
                ),
                "reference_bits": f"0x{reference_bits:08x}",
                "candidate_bits": f"0x{candidate_bits:08x}",
                "reference_value": expected,
                "candidate_value": actual,
            }

    if reference_bytes == candidate_bytes:
        rms = 0.0
        relative_rms = 0.0
        max_abs = 0.0
        cosine = 1.0
    else:
        sum_delta_squared = 0.0
        sum_reference_squared = 0.0
        sum_candidate_squared = 0.0
        dot_product = 0.0
        max_abs = 0.0

        for expected, actual in zip(reference, candidate):
            delta = float(actual) - float(expected)
            absolute_delta = abs(delta)
            if absolute_delta > max_abs:
                max_abs = absolute_delta
            sum_delta_squared += delta * delta
            sum_reference_squared += float(expected) * float(expected)
            sum_candidate_squared += float(actual) * float(actual)
            dot_product += float(expected) * float(actual)

        count = len(reference)
        rms = math.sqrt(sum_delta_squared / count)
        reference_rms = math.sqrt(sum_reference_squared / count)
        relative_rms = rms / max(reference_rms, 1.0e-30)
        denominator = math.sqrt(sum_reference_squared * sum_candidate_squared)
        if denominator == 0.0:
            cosine = 1.0 if sum_reference_squared == sum_candidate_squared else 0.0
        else:
            cosine = dot_product / denominator

    return {
        "rms": rms,
        "relative_rms": relative_rms,
        "max_abs": max_abs,
        "cosine": cosine,
        "exact_hash": exact_hash,
        "reference_sha256": reference_sha256,
        "candidate_sha256": candidate_sha256,
        "first_mismatch": first_mismatch,
    }


def optional_pair(reference: Path, candidate: Path, name: str) -> tuple[Path, Path] | None:
    reference_path = reference / name
    candidate_path = candidate / name
    if reference_path.exists() != candidate_path.exists():
        missing = candidate_path if reference_path.exists() else reference_path
        raise DiagnosticError(f"missing paired diagnostic file: {missing}")
    if not reference_path.exists():
        return None
    return reference_path, candidate_path


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    if not reference.is_dir():
        raise DiagnosticError(f"reference directory does not exist: {reference}")
    if not candidate.is_dir():
        raise DiagnosticError(f"candidate directory does not exist: {candidate}")

    embedding_pair = optional_pair(reference, candidate, "embd.f32")
    embedding = None
    if embedding_pair is not None:
        embedding = metrics(*embedding_pair, LAYER_BYTES, row_width=WIDTH)
        embedding["last_token"] = metrics(
            *embedding_pair,
            LAYER_BYTES,
            value_start=(TOKENS - 1) * WIDTH,
            value_count=WIDTH,
            row_width=WIDTH,
        )

    layer0_pairs = {
        stage: optional_pair(
            reference, candidate, f"layer-00-{stage}.f32"
        )
        for stage in LAYER0_STAGES
    }
    if any(pair is not None for pair in layer0_pairs.values()) and not all(
        pair is not None for pair in layer0_pairs.values()
    ):
        missing = next(stage for stage, pair in layer0_pairs.items() if pair is None)
        raise DiagnosticError(
            f"incomplete layer-0 diagnostics: missing layer-00-{missing}.f32"
        )
    layer0_checkpoints: dict[str, dict[str, Any]] = {}
    for stage, pair in layer0_pairs.items():
        if pair is None:
            continue
        width = LAYER0_WIDTHS[stage]
        expected_bytes = width * TOKENS * FLOAT_SIZE
        result = metrics(*pair, expected_bytes, row_width=width)
        result["last_token"] = metrics(
            *pair,
            expected_bytes,
            value_start=(TOKENS - 1) * width,
            value_count=width,
            row_width=width,
        )
        result["width"] = width
        layer0_checkpoints[stage] = result

    layers: list[dict[str, Any]] = []
    for layer in range(LAYER_COUNT):
        name = f"layer-{layer:02d}.f32"
        result = metrics(
            reference / name,
            candidate / name,
            LAYER_BYTES,
            row_width=WIDTH,
        )
        result["last_token"] = metrics(
            reference / name,
            candidate / name,
            LAYER_BYTES,
            value_start=(TOKENS - 1) * WIDTH,
            value_count=WIDTH,
            row_width=WIDTH,
        )
        result["layer"] = layer
        layers.append(result)

    largest_increase = None
    previous_stage = "embd"
    previous_relative_rms = (
        embedding["last_token"]["relative_rms"] if embedding is not None else 0.0
    )
    for result in layers:
        current_relative_rms = result["last_token"]["relative_rms"]
        increase = current_relative_rms - previous_relative_rms
        increase_candidate = {
            "layer": result["layer"],
            "previous_stage": previous_stage,
            "increase": increase,
        }
        if largest_increase is None or increase > largest_increase["increase"]:
            largest_increase = increase_candidate
        previous_stage = f"l_out-{result['layer']}"
        previous_relative_rms = current_relative_rms

    logits_pair = optional_pair(reference, candidate, "logits.f32")
    logits = None
    if logits_pair is not None:
        logits = metrics(*logits_pair, None)

    first_divergence = None
    if embedding is not None and not embedding["exact_hash"]:
        first_divergence = {"stage": "embd", "layer": None}
    else:
        for stage, result in layer0_checkpoints.items():
            if not result["exact_hash"]:
                first_divergence = {"stage": stage, "layer": 0}
                break
        if first_divergence is None:
            for result in layers:
                if not result["exact_hash"]:
                    first_divergence = {"stage": "l_out", "layer": result["layer"]}
                    break
        if first_divergence is None and logits is not None and not logits["exact_hash"]:
            first_divergence = {"stage": "logits", "layer": None}

    return {
        "schema": "laguna-layer-comparison/v1",
        "shape": {"width": WIDTH, "tokens": TOKENS},
        "embedding": embedding,
        "layer0_checkpoints": layer0_checkpoints,
        "layers": layers,
        "logits": logits,
        "first_divergence": first_divergence,
        "largest_last_token_relative_rms_increase": largest_increase,
    }


def metric_rows(report: dict[str, Any]):
    if report["embedding"] is not None:
        yield "embd", report["embedding"]
    for stage, result in report["layer0_checkpoints"].items():
        yield f"l0-{stage}", result
    for result in report["layers"]:
        yield f"l_out-{result['layer']}", result
    if report["logits"] is not None:
        yield "logits", report["logits"]


def _first_mismatch_value(value: Any) -> str:
    return "null" if value is None else str(value)


def print_table(report: dict[str, Any]) -> None:
    print("stage\trms\trelative_rms\tmax_abs\tcosine\texact_hash")
    for stage, result in metric_rows(report):
        print(
            f"{stage}\t{result['rms']:.9g}\t{result['relative_rms']:.9g}\t"
            f"{result['max_abs']:.9g}\t{result['cosine']:.12g}\t"
            f"{str(result['exact_hash']).lower()}"
        )

    mismatch_rows = [
        (stage, result.get("first_mismatch"))
        for stage, result in metric_rows(report)
        if result.get("first_mismatch") is not None
    ]
    if not mismatch_rows:
        print("first_mismatch=none")
    else:
        for stage, mismatch in mismatch_rows:
            assert mismatch is not None
            print(
                f"first_mismatch={stage} "
                f"flat_index={mismatch['flat_index']} "
                f"token_index={_first_mismatch_value(mismatch['token_index'])} "
                f"element_index={_first_mismatch_value(mismatch['element_index'])} "
                f"reference_bits={mismatch['reference_bits']} "
                f"candidate_bits={mismatch['candidate_bits']} "
                f"reference_value={_first_mismatch_value(mismatch['reference_value'])} "
                f"candidate_value={_first_mismatch_value(mismatch['candidate_value'])}"
            )

    divergence = report["first_divergence"]
    if divergence is None:
        print("first_divergence=none")
    elif divergence["layer"] is None:
        print(f"first_divergence={divergence['stage']}")
    else:
        print(f"first_divergence={divergence['stage']}-{divergence['layer']}")
    increase = report["largest_last_token_relative_rms_increase"]
    if increase is not None:
        print(
            "largest_last_token_relative_rms_increase="
            f"layer-{increase['layer']} previous={increase['previous_stage']} "
            f"delta={increase['increase']:.9g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare(args.reference, args.candidate)
    except DiagnosticError as error:
        print(f"compare_laguna_layers: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
