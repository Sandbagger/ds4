#!/usr/bin/env python3
"""Compare direct Poolside and DS4 Laguna MoE captures in execution order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "laguna-moe-execution-comparison/v1"
RUN_SCHEMA = "laguna-token513-direct-capture-run/v1"
LAYER = 1
EXPERTS_USED = 10
EXPERTS_TOTAL = 256
EMBEDDING = 3072
EXPERT_MID = 1024
Q8_1_BYTES = EMBEDDING // 32 * 36
DOWN_Q8_1_BYTES = EXPERTS_USED * EXPERT_MID // 32 * 36
PINNED_MODEL = {
    "bytes": 68248759648,
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
}
PINNED_PREFIX = {
    "count": 512,
    "bytes": 2048,
    "sha256": "569aa6394783e0f17558db421ba26480d7a530d44dd2219bcc9aac2c09a3b559",
}
PINNED_DEVICE = {
    "name": "NVIDIA GB10",
    "compute_capability": "12.1",
    "driver": "580.126.09",
}
PINNED_POOLSIDE_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
PINNED_DS4_CAPTURE_COMMIT = "1d009d4f134af0f069730702a6247c077e18fdbd"


class ComparisonError(RuntimeError):
    """A capture or microscope fixture violated the comparison contract."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    filename: str | None
    dtype: str
    elements: int
    shape: tuple[int, ...]
    kind: str
    width: int


ROOT_STAGES = (
    StageSpec("ffn_norm", None, "f32-le", EMBEDDING, (EMBEDDING,), "row", EMBEDDING),
    StageSpec(
        "router_logits",
        "layer-01-router-logits.f32",
        "f32-le",
        EXPERTS_TOTAL,
        (EXPERTS_TOTAL,),
        "router_logits",
        EXPERTS_TOTAL,
    ),
    StageSpec(
        "selected",
        "layer-01-router-selected.i32",
        "i32-le",
        EXPERTS_USED,
        (EXPERTS_USED,),
        "selected",
        EXPERTS_USED,
    ),
    StageSpec(
        "router_weights",
        "layer-01-router-weights.f32",
        "f32-le",
        EXPERTS_USED,
        (1, EXPERTS_USED, 1),
        "slot",
        EXPERTS_USED,
    ),
)

ROUTED_STAGES = (
    StageSpec(
        "gate",
        "layer-01-ffn-moe-gate.f32",
        "f32-le",
        EXPERTS_USED * EXPERT_MID,
        (EXPERT_MID, EXPERTS_USED, 1),
        "routed",
        EXPERT_MID,
    ),
    StageSpec(
        "up",
        "layer-01-ffn-moe-up.f32",
        "f32-le",
        EXPERTS_USED * EXPERT_MID,
        (EXPERT_MID, EXPERTS_USED, 1),
        "routed",
        EXPERT_MID,
    ),
    StageSpec(
        "swiglu",
        "layer-01-ffn-moe-swiglu.f32",
        "f32-le",
        EXPERTS_USED * EXPERT_MID,
        (EXPERT_MID, EXPERTS_USED, 1),
        "routed",
        EXPERT_MID,
    ),
    StageSpec(
        "col_l2",
        "layer-01-ffn-moe-col-l2.f32",
        "f32-le",
        EXPERTS_USED,
        (1, EXPERTS_USED, 1),
        "routed",
        1,
    ),
    StageSpec(
        "down_input",
        "layer-01-ffn-moe-down-input.f32",
        "f32-le",
        EXPERTS_USED * EXPERT_MID,
        (EXPERT_MID, EXPERTS_USED, 1),
        "routed",
        EXPERT_MID,
    ),
    StageSpec(
        "down",
        "layer-01-ffn-moe-down.f32",
        "f32-le",
        EXPERTS_USED * EMBEDDING,
        (EMBEDDING, EXPERTS_USED, 1),
        "routed",
        EMBEDDING,
    ),
    StageSpec(
        "weighted",
        "layer-01-ffn-moe-weighted.f32",
        "f32-le",
        EXPERTS_USED * EMBEDDING,
        (EMBEDDING, EXPERTS_USED, 1),
        "routed",
        EMBEDDING,
    ),
)

FINAL_STAGES = (
    StageSpec(
        "routed_sum",
        "layer-01-ffn-moe-out.f32",
        "f32-le",
        EMBEDDING,
        (EMBEDDING, 1, 1),
        "row",
        EMBEDDING,
    ),
    StageSpec(
        "shared",
        "layer-01-ffn-shared-out.f32",
        "f32-le",
        EMBEDDING,
        (EMBEDDING, 1, 1),
        "row",
        EMBEDDING,
    ),
    StageSpec(
        "combined",
        "layer-01-ffn-out.f32",
        "f32-le",
        EMBEDDING,
        (EMBEDDING, 1, 1),
        "row",
        EMBEDDING,
    ),
)

ALL_STAGES = ROOT_STAGES + ROUTED_STAGES + FINAL_STAGES
STAGE_BY_NAME = {stage.name: stage for stage in ALL_STAGES}
SEMANTIC_ORDER = [
    "ffn_norm",
    "router_logits",
    "selected",
    "router_weights",
    "for each slot: gate, up, swiglu, col_l2, down_input, "
    "down_input_q8_1, down, weighted",
    "routed_sum",
    "shared",
    "combined",
]
MICROSCOPE_FILES = {
    "input.f32": EMBEDDING * 4,
    "input.q8_1": Q8_1_BYTES,
    "weight-row.q4k": 12 * 144,
    "poolside-output.f32": 4,
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_exact(path: Path, expected_bytes: int, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ComparisonError(f"cannot read {label} {path}: {error}") from error
    if len(payload) != expected_bytes:
        raise ComparisonError(
            f"{label} {path} is {len(payload)} bytes; expected {expected_bytes}"
        )
    return payload


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ComparisonError(f"{label} is not a directory: {path}")


def bits_at(payload: bytes, index: int) -> int:
    return int.from_bytes(payload[index * 4 : index * 4 + 4], "little")


def bits_text(bits: int) -> str:
    return f"0x{bits:08x}"


def i32_values(payload: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(payload) // 4}i", payload)


def f32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def f32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def replay_routed_sum(
    down_payload: bytes,
    weights_payload: bytes,
) -> tuple[bytes, bytes]:
    down = f32_values(down_payload)
    weights = f32_values(weights_payload)
    weighted = [0.0] * (EXPERTS_USED * EMBEDDING)
    routed_sum = [f32_round(0.0)] * EMBEDDING
    for slot in range(EXPERTS_USED):
        weight = weights[slot]
        base = slot * EMBEDDING
        for row in range(EMBEDDING):
            product = f32_round(down[base + row] * weight)
            weighted[base + row] = product
            routed_sum[row] = f32_round(routed_sum[row] + product)
    return (
        struct.pack(f"<{len(weighted)}f", *weighted),
        struct.pack(f"<{len(routed_sum)}f", *routed_sum),
    )


def replay_binding(
    runtime: str,
    replay_weighted: bytes,
    replay_sum: bytes,
    captured_weighted: bytes,
    captured_sum: bytes,
) -> dict[str, Any]:
    weighted_exact = replay_weighted == captured_weighted
    routed_sum_exact = replay_sum == captured_sum
    if not weighted_exact or not routed_sum_exact:
        failed = []
        if not weighted_exact:
            failed.append("weighted")
        if not routed_sum_exact:
            failed.append("routed_sum")
        raise ComparisonError(
            f"{runtime} F32 replay does not reproduce captured "
            + " and ".join(failed)
        )
    return {
        "weighted_exact": True,
        "weighted_elements": EXPERTS_USED * EMBEDDING,
        "weighted_sha256": sha256(replay_weighted),
        "routed_sum_exact": True,
        "routed_sum_elements": EMBEDDING,
        "routed_sum_sha256": sha256(replay_sum),
    }


def delta_metrics(candidate: bytes, reference: bytes) -> dict[str, Any]:
    candidate_values = f32_values(candidate)
    reference_values = f32_values(reference)
    deltas = [
        candidate_value - reference_value
        for candidate_value, reference_value in zip(
            candidate_values, reference_values, strict=True
        )
    ]
    unequal_rows = sum(
        candidate[index : index + 4] != reference[index : index + 4]
        for index in range(0, len(reference), 4)
    )
    squared_sum = sum(delta * delta for delta in deltas)
    return {
        "unequal_rows": unequal_rows,
        "rows": len(reference_values),
        "max_absolute_delta": max((abs(delta) for delta in deltas), default=0.0),
        "rms_delta": math.sqrt(squared_sum / len(deltas)),
        "l2_delta": math.sqrt(squared_sum),
    }


def parse_bits(value: Any, label: str) -> int:
    if not isinstance(value, str) or len(value) != 10 or not value.startswith("0x"):
        raise ComparisonError(f"{label} must be a 32-bit hexadecimal string")
    try:
        parsed = int(value, 16)
    except ValueError as error:
        raise ComparisonError(f"{label} must be hexadecimal") from error
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise ComparisonError(f"{label} is outside uint32 range")
    return parsed


def mismatch_record(
    stage: StageSpec,
    index: int,
    poolside: bytes,
    ds4: bytes,
    poolside_selected: tuple[int, ...],
    ds4_selected: tuple[int, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage.name,
        "poolside_bits": bits_text(bits_at(poolside, index)),
        "ds4_bits": bits_text(bits_at(ds4, index)),
    }
    if stage.kind == "routed":
        slot, row = divmod(index, stage.width)
        poolside_expert = poolside_selected[slot]
        ds4_expert = ds4_selected[slot]
        record.update(
            {
                "slot": slot,
                "expert": poolside_expert
                if poolside_expert == ds4_expert
                else None,
                "poolside_expert": poolside_expert,
                "ds4_expert": ds4_expert,
                "row": row,
            }
        )
    elif stage.kind == "slot":
        poolside_expert = poolside_selected[index]
        ds4_expert = ds4_selected[index]
        record.update(
            {
                "slot": index,
                "expert": poolside_expert
                if poolside_expert == ds4_expert
                else None,
                "poolside_expert": poolside_expert,
                "ds4_expert": ds4_expert,
            }
        )
    elif stage.kind == "selected":
        record.update(
            {
                "slot": index,
                "poolside_value": poolside_selected[index],
                "ds4_value": ds4_selected[index],
            }
        )
    elif stage.kind == "router_logits":
        record["expert_index"] = index
    else:
        record["row"] = index
    return record


def first_mismatch(
    stage: StageSpec,
    poolside: bytes,
    ds4: bytes,
    poolside_selected: tuple[int, ...],
    ds4_selected: tuple[int, ...],
    indices: Iterable[int] | None = None,
) -> dict[str, Any] | None:
    candidates = range(stage.elements) if indices is None else indices
    for index in candidates:
        offset = index * 4
        if poolside[offset : offset + 4] != ds4[offset : offset + 4]:
            return mismatch_record(
                stage,
                index,
                poolside,
                ds4,
                poolside_selected,
                ds4_selected,
            )
    return None


def compare_stage(
    stage: StageSpec,
    poolside: bytes,
    ds4: bytes,
    poolside_selected: tuple[int, ...],
    ds4_selected: tuple[int, ...],
    poolside_label: str,
    ds4_label: str,
) -> dict[str, Any]:
    equal_count = 0
    for index in range(stage.elements):
        offset = index * 4
        if poolside[offset : offset + 4] == ds4[offset : offset + 4]:
            equal_count += 1
    mismatch_count = stage.elements - equal_count
    return {
        "dtype": stage.dtype,
        "shape": list(stage.shape),
        "elements": stage.elements,
        "equal_count": equal_count,
        "mismatch_count": mismatch_count,
        "exact": mismatch_count == 0,
        "first_mismatch": first_mismatch(
            stage, poolside, ds4, poolside_selected, ds4_selected
        ),
        "poolside": {
            "artifact": poolside_label,
            "sha256": sha256(poolside),
        },
        "ds4": {"artifact": ds4_label, "sha256": sha256(ds4)},
    }


def load_microscope(
    fixture: Path,
) -> tuple[dict[str, Any], bytes, dict[str, bytes], dict[str, dict[str, Any]]]:
    require_directory(fixture, "microscope fixture")
    manifest_path = fixture / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read microscope manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ComparisonError("microscope manifest must be a JSON object")
    if manifest.get("schema") != "q4k-mmvq-microscope-fixture/v1":
        raise ComparisonError("unexpected microscope fixture schema")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict):
        raise ComparisonError("microscope manifest files must be an object")
    if not set(MICROSCOPE_FILES).issubset(declared_files):
        raise ComparisonError("microscope manifest is missing a required file")

    payloads: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name in sorted(declared_files):
        if Path(name).name != name:
            raise ComparisonError(f"microscope file name is not local: {name}")
        entry = declared_files[name]
        if not isinstance(entry, dict):
            raise ComparisonError(f"microscope file entry is not an object: {name}")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_sha256, str)
        ):
            raise ComparisonError(f"invalid microscope file contract: {name}")
        payload = read_exact(fixture / name, expected_bytes, "microscope file")
        actual_sha256 = sha256(payload)
        if actual_sha256 != expected_sha256:
            raise ComparisonError(f"microscope file hash mismatch: {name}")
        payloads[name] = payload
        bindings[name] = {
            "bytes": expected_bytes,
            "sha256": actual_sha256,
        }
    for name, expected_bytes in MICROSCOPE_FILES.items():
        if len(payloads[name]) != expected_bytes:
            raise ComparisonError(
                f"microscope {name} has {len(payloads[name])} bytes; "
                f"expected {expected_bytes}"
            )
    return manifest, manifest_bytes, payloads, bindings


def load_run_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read run manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ComparisonError("run manifest must be a JSON object")
    if manifest.get("schema") != RUN_SCHEMA:
        raise ComparisonError("unexpected run manifest schema")
    if manifest.get("token") != 513 or manifest.get("layer") != LAYER:
        raise ComparisonError("run manifest must identify token 513, layer 1")
    if manifest.get("resume_token") != 3612:
        raise ComparisonError("run manifest must identify resume token 3612")
    model = manifest.get("model")
    prefix = manifest.get("prefix")
    if not isinstance(model, dict) or not isinstance(prefix, dict):
        raise ComparisonError("run manifest model and prefix must be objects")
    for label, identity in (("model", model), ("prefix", prefix)):
        size = identity.get("bytes")
        digest = identity.get("sha256")
        if (
            not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ComparisonError(f"run manifest {label} identity is invalid")
    if model != PINNED_MODEL:
        raise ComparisonError("run manifest model is not the pinned model")
    if prefix != PINNED_PREFIX:
        raise ComparisonError("run manifest prefix is not the pinned prefix")
    if manifest.get("device") != PINNED_DEVICE:
        raise ComparisonError("run manifest device is not the pinned device")
    runtimes = manifest.get("runtimes")
    if not isinstance(runtimes, dict) or not all(
        isinstance(runtimes.get(runtime), dict)
        for runtime in ("poolside", "ds4")
    ):
        raise ComparisonError("run manifest must identify both runtimes")
    poolside = runtimes["poolside"]
    ds4 = runtimes["ds4"]
    if poolside.get("commit") != PINNED_POOLSIDE_COMMIT:
        raise ComparisonError("run manifest Poolside commit is not pinned")
    if ds4.get("capture_code_commit") != PINNED_DS4_CAPTURE_COMMIT:
        raise ComparisonError("run manifest DS4 capture commit is not pinned")
    for runtime, entry, fields in (
        (
            "Poolside",
            poolside,
            ("producer_source_sha256", "producer_binary_sha256"),
        ),
        (
            "DS4",
            ds4,
            ("capture_probe_binary_sha256",),
        ),
    ):
        for field in fields:
            value = entry.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ComparisonError(
                    f"run manifest {runtime} {field} is not a SHA-256"
                )
    controls = manifest.get("controls")
    if not isinstance(controls, dict):
        raise ComparisonError("run manifest controls must be an object")
    if controls.get("release_vs_hook_null") != "bit-exact":
        raise ComparisonError(
            "run manifest release versus hook-null control is not bit-exact"
        )
    if controls.get("hook_null_vs_hook_active") != "bit-exact":
        raise ComparisonError(
            "run manifest hook-null versus active control is not bit-exact"
        )
    if controls.get("logits_bytes") != 401408:
        raise ComparisonError("run manifest control logits byte count is invalid")
    for field in (
        "release_probe_binary_sha256",
        "hook_probe_binary_sha256",
        "logits_sha256",
    ):
        value = controls.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ComparisonError(
                f"run manifest control {field} is not a SHA-256"
            )
    return manifest, payload


def require_manifest_artifact(
    manifest: dict[str, Any],
    runtime: str,
    name: str,
    payload: bytes,
) -> None:
    captures = manifest.get("captures")
    capture = captures.get(runtime) if isinstance(captures, dict) else None
    artifacts = capture.get("artifacts") if isinstance(capture, dict) else None
    contract = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(contract, dict):
        raise ComparisonError(
            f"run manifest is missing {runtime} artifact contract: {name}"
        )
    if contract.get("bytes") != len(payload):
        raise ComparisonError(
            f"run manifest artifact byte count does not match {runtime} {name}"
        )
    if contract.get("sha256") != sha256(payload):
        raise ComparisonError(
            f"run manifest artifact hash does not match {runtime} {name}"
        )


def validate_origin(manifest: dict[str, Any]) -> tuple[dict[str, Any], StageSpec]:
    origin = manifest.get("origin")
    if not isinstance(origin, dict):
        raise ComparisonError("microscope origin must be an object")
    if origin.get("token") != 513 or origin.get("layer") != LAYER:
        raise ComparisonError("microscope origin must be token 513, layer 1")
    projection = origin.get("projection")
    stage = STAGE_BY_NAME.get(projection) if isinstance(projection, str) else None
    if stage is None or stage.name not in ("gate", "up"):
        raise ComparisonError(
            "microscope origin projection must be a routed gate/up Q4_K stage"
        )
    slot = origin.get("selected_slot")
    row = origin.get("row")
    expert = origin.get("expert")
    if not isinstance(slot, int) or slot < 0 or slot >= EXPERTS_USED:
        raise ComparisonError("microscope origin selected_slot is invalid")
    if not isinstance(row, int) or row < 0 or row >= stage.width:
        raise ComparisonError("microscope origin row is invalid")
    if not isinstance(expert, int) or expert < 0 or expert >= EXPERTS_TOTAL:
        raise ComparisonError("microscope origin expert is invalid")
    return origin, stage


def build_report(
    poolside_moe: Path,
    poolside_ffn_norm_path: Path,
    ds4_dir: Path,
    microscope_fixture: Path,
    run_manifest_path: Path,
) -> dict[str, Any]:
    require_directory(poolside_moe, "Poolside MoE capture")
    require_directory(ds4_dir, "DS4 capture")
    manifest, manifest_bytes, fixture_payloads, fixture_files = load_microscope(
        microscope_fixture
    )
    run_manifest, run_manifest_bytes = load_run_manifest(run_manifest_path)
    if manifest.get("model") != run_manifest["model"]:
        raise ComparisonError(
            "run manifest model does not match microscope model identity"
        )

    poolside_payloads: dict[str, bytes] = {}
    ds4_payloads: dict[str, bytes] = {}
    poolside_labels: dict[str, str] = {}
    ds4_labels: dict[str, str] = {}
    for stage in ALL_STAGES:
        expected_bytes = stage.elements * 4
        if stage.name == "ffn_norm":
            poolside_path = poolside_ffn_norm_path
            ds4_path = ds4_dir / "layer-01-ffn-norm.f32"
        else:
            assert stage.filename is not None
            poolside_path = poolside_moe / stage.filename
            ds4_path = ds4_dir / stage.filename
        poolside_payloads[stage.name] = read_exact(
            poolside_path, expected_bytes, f"Poolside {stage.name}"
        )
        ds4_payloads[stage.name] = read_exact(
            ds4_path, expected_bytes, f"DS4 {stage.name}"
        )
        poolside_labels[stage.name] = poolside_path.name
        ds4_labels[stage.name] = ds4_path.name

    ds4_q8 = read_exact(
        ds4_dir / "layer-01-ffn-moe-input.q8_1",
        Q8_1_BYTES,
        "DS4 routed input Q8_1",
    )
    ds4_down_q8 = read_exact(
        ds4_dir / "layer-01-ffn-moe-down-input.q8_1",
        DOWN_Q8_1_BYTES,
        "DS4 down input Q8_1",
    )
    fixture_ffn_norm = fixture_payloads["input.f32"]
    fixture_q8 = fixture_payloads["input.q8_1"]
    if poolside_payloads["ffn_norm"] != fixture_ffn_norm:
        raise ComparisonError(
            "Poolside ffn_norm does not match microscope input.f32"
        )
    if ds4_payloads["ffn_norm"] != fixture_ffn_norm:
        raise ComparisonError("DS4 ffn_norm does not match microscope input.f32")
    if ds4_q8 != fixture_q8:
        raise ComparisonError("DS4 Q8 input does not match microscope input.q8_1")

    poolside_selected = i32_values(poolside_payloads["selected"])
    ds4_selected = i32_values(ds4_payloads["selected"])
    for runtime, selected in (
        ("Poolside", poolside_selected),
        ("DS4", ds4_selected),
    ):
        if any(expert < 0 or expert >= EXPERTS_TOTAL for expert in selected):
            raise ComparisonError(f"{runtime} selected expert is outside 0..255")

    stages = {
        stage.name: compare_stage(
            stage,
            poolside_payloads[stage.name],
            ds4_payloads[stage.name],
            poolside_selected,
            ds4_selected,
            poolside_labels[stage.name],
            ds4_labels[stage.name],
        )
        for stage in ALL_STAGES
    }

    poolside_replay_weighted, poolside_replay_sum = replay_routed_sum(
        poolside_payloads["down"], poolside_payloads["router_weights"]
    )
    ds4_replay_weighted, ds4_replay_sum = replay_routed_sum(
        ds4_payloads["down"], ds4_payloads["router_weights"]
    )
    poolside_replay = replay_binding(
        "Poolside",
        poolside_replay_weighted,
        poolside_replay_sum,
        poolside_payloads["weighted"],
        poolside_payloads["routed_sum"],
    )
    ds4_replay = replay_binding(
        "DS4",
        ds4_replay_weighted,
        ds4_replay_sum,
        ds4_payloads["weighted"],
        ds4_payloads["routed_sum"],
    )
    _, routing_only_sum = replay_routed_sum(
        poolside_payloads["down"], ds4_payloads["router_weights"]
    )
    _, expert_only_sum = replay_routed_sum(
        ds4_payloads["down"], poolside_payloads["router_weights"]
    )
    counterfactual_decomposition = {
        "reference": "poolside_routed_sum",
        "arithmetic": "slot-order IEEE-754 binary32 multiply then add",
        "poolside_replay": poolside_replay,
        "ds4_replay": ds4_replay,
        "routing_only": delta_metrics(
            routing_only_sum, poolside_payloads["routed_sum"]
        ),
        "expert_only": delta_metrics(
            expert_only_sum, poolside_payloads["routed_sum"]
        ),
        "combined": delta_metrics(
            ds4_replay_sum, poolside_payloads["routed_sum"]
        ),
    }

    first_overall: dict[str, Any] | None = None
    for stage in ROOT_STAGES:
        first_overall = stages[stage.name]["first_mismatch"]
        if first_overall is not None:
            break

    first_routed: dict[str, Any] | None = None
    if first_overall is None:
        for slot in range(EXPERTS_USED):
            for stage in ROUTED_STAGES:
                start = slot * stage.width
                mismatch = first_mismatch(
                    stage,
                    poolside_payloads[stage.name],
                    ds4_payloads[stage.name],
                    poolside_selected,
                    ds4_selected,
                    range(start, start + stage.width),
                )
                if mismatch is not None:
                    first_overall = mismatch
                    first_routed = mismatch
                    break
            if first_overall is not None:
                break
    else:
        # Root divergence does not suppress the independent routed-expert
        # answer; downstream capture comparison can still be informative.
        pass

    if first_routed is None:
        for slot in range(EXPERTS_USED):
            for stage in ROUTED_STAGES:
                start = slot * stage.width
                mismatch = first_mismatch(
                    stage,
                    poolside_payloads[stage.name],
                    ds4_payloads[stage.name],
                    poolside_selected,
                    ds4_selected,
                    range(start, start + stage.width),
                )
                if mismatch is not None:
                    first_routed = mismatch
                    break
            if first_routed is not None:
                break

    if first_overall is None:
        for stage in FINAL_STAGES:
            first_overall = stages[stage.name]["first_mismatch"]
            if first_overall is not None:
                break

    origin, origin_stage = validate_origin(manifest)
    oracle = manifest.get("oracle")
    if not isinstance(oracle, dict):
        raise ComparisonError("microscope oracle must be an object")
    poolside_oracle_bits = parse_bits(
        oracle.get("poolside_float32_bits"),
        "microscope Poolside oracle bits",
    )
    ds4_oracle_bits = parse_bits(
        oracle.get("ds4_serial_float32_bits"),
        "microscope DS4 oracle bits",
    )
    fixture_output_bits = bits_at(fixture_payloads["poolside-output.f32"], 0)
    if fixture_output_bits != poolside_oracle_bits:
        raise ComparisonError(
            "microscope poolside-output.f32 does not match its oracle bits"
        )
    origin_slot = origin["selected_slot"]
    origin_index = origin_slot * origin_stage.width + origin["row"]
    direct_poolside_bits = bits_at(
        poolside_payloads[origin_stage.name], origin_index
    )
    direct_ds4_bits = bits_at(ds4_payloads[origin_stage.name], origin_index)
    if poolside_selected[origin_slot] != origin["expert"]:
        raise ComparisonError(
            "Poolside selected expert at microscope origin does not match manifest"
        )
    if ds4_selected[origin_slot] != origin["expert"]:
        raise ComparisonError(
            "DS4 selected expert at microscope origin does not match manifest"
        )
    if direct_poolside_bits != poolside_oracle_bits:
        raise ComparisonError(
            "Poolside direct output does not match microscope oracle"
        )
    if direct_poolside_bits != fixture_output_bits:
        raise ComparisonError(
            "Poolside direct output does not match microscope fixture output"
        )
    if direct_ds4_bits != ds4_oracle_bits:
        raise ComparisonError("DS4 direct output does not match microscope oracle")
    if first_routed is None:
        raise ComparisonError(
            "microscope origin exists but captures have no first routed mismatch"
        )
    expected_origin_coordinate = {
        "stage": origin_stage.name,
        "slot": origin_slot,
        "expert": origin["expert"],
        "row": origin["row"],
    }
    actual_origin_coordinate = {
        key: first_routed.get(key) for key in expected_origin_coordinate
    }
    if actual_origin_coordinate != expected_origin_coordinate:
        raise ComparisonError(
            "microscope origin does not match first routed mismatch: "
            f"expected {expected_origin_coordinate}, got {actual_origin_coordinate}"
        )

    for stage in ALL_STAGES:
        require_manifest_artifact(
            run_manifest,
            "poolside",
            stage.name,
            poolside_payloads[stage.name],
        )
        require_manifest_artifact(
            run_manifest,
            "ds4",
            stage.name,
            ds4_payloads[stage.name],
        )
    require_manifest_artifact(
        run_manifest, "ds4", "moe_input_q8_1", ds4_q8
    )
    require_manifest_artifact(
        run_manifest, "ds4", "down_input_q8_1", ds4_down_q8
    )
    declared_microscope_sha256 = run_manifest.get(
        "microscope_manifest_sha256"
    )
    if declared_microscope_sha256 != sha256(manifest_bytes):
        raise ComparisonError(
            "run manifest microscope hash does not match fixture manifest"
        )

    weight_mismatches = []
    for slot in range(EXPERTS_USED):
        if (
            poolside_payloads["router_weights"][slot * 4 : slot * 4 + 4]
            == ds4_payloads["router_weights"][slot * 4 : slot * 4 + 4]
        ):
            continue
        weight_mismatches.append(
            {
                "slot": slot,
                "expert": (
                    poolside_selected[slot]
                    if poolside_selected[slot] == ds4_selected[slot]
                    else None
                ),
                "poolside_bits": bits_text(
                    bits_at(poolside_payloads["router_weights"], slot)
                ),
                "ds4_bits": bits_text(
                    bits_at(ds4_payloads["router_weights"], slot)
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "semantic_order": SEMANTIC_ORDER,
        "shape": {
            "layer": LAYER,
            "experts_total": EXPERTS_TOTAL,
            "experts_used": EXPERTS_USED,
            "embedding": EMBEDDING,
            "expert_mid": EXPERT_MID,
        },
        "run_manifest": {
            "schema": run_manifest["schema"],
            "artifact": run_manifest_path.name,
            "sha256": sha256(run_manifest_bytes),
            "captures_bound": True,
            "token": run_manifest["token"],
            "layer": run_manifest["layer"],
            "model": run_manifest["model"],
            "prefix": run_manifest["prefix"],
            "resume_token": run_manifest["resume_token"],
            "device": run_manifest["device"],
            "runtimes": run_manifest["runtimes"],
            "controls": run_manifest.get("controls", {}),
        },
        "routing": {
            "selected_ids": {
                "poolside": list(poolside_selected),
                "ds4": list(ds4_selected),
                "exact": poolside_selected == ds4_selected,
            },
            "router_logits_mismatch_count": stages["router_logits"][
                "mismatch_count"
            ],
            "weight_mismatches": weight_mismatches,
        },
        "stages": stages,
        "unpaired_boundaries": {
            "down_input_q8_1": {
                "semantic_after": "down_input",
                "semantic_before": "down",
                "status": "unavailable_for_comparison",
                "poolside_observed": False,
                "poolside_reason": (
                    "Poolside graph callbacks expose the F32 down input; "
                    "Q8_1 quantization is internal to the CUDA Q4_K multiply"
                ),
                "ds4_observed": True,
                "ds4_artifact": "layer-01-ffn-moe-down-input.q8_1",
                "ds4_bytes": len(ds4_down_q8),
                "ds4_sha256": sha256(ds4_down_q8),
            }
        },
        "counterfactual_decomposition": counterfactual_decomposition,
        "first_overall_mismatch": first_overall,
        "first_routed_expert_mismatch": first_routed,
        "microscope": {
            "schema": manifest["schema"],
            "manifest_sha256": sha256(manifest_bytes),
            "origin": origin,
            "oracle": oracle,
            "files": fixture_files,
            "input_binding": {
                "ffn_norm_exact": True,
                "q8_1_exact": True,
                "poolside_q8_1_observed": False,
                "poolside_q8_1_binding": (
                    "inferred from byte-exact F32 input and the pinned "
                    "microscope quantization; not a Poolside runtime capture"
                ),
                "microscope_ffn_norm_sha256": sha256(fixture_ffn_norm),
                "poolside_ffn_norm_sha256": sha256(
                    poolside_payloads["ffn_norm"]
                ),
                "ds4_ffn_norm_sha256": sha256(ds4_payloads["ffn_norm"]),
                "microscope_q8_1_sha256": sha256(fixture_q8),
                "ds4_q8_1_sha256": sha256(ds4_q8),
            },
            "origin_binding": {
                "poolside_selected_expert": poolside_selected[origin_slot],
                "ds4_selected_expert": ds4_selected[origin_slot],
                "poolside_matches_manifest": True,
                "ds4_matches_manifest": True,
            },
            "direct_output_binding": {
                "poolside_bits": bits_text(direct_poolside_bits),
                "ds4_bits": bits_text(direct_ds4_bits),
                "poolside_matches_oracle": True,
                "ds4_matches_oracle": True,
                "poolside_matches_fixture_output": True,
            },
        },
    }


def mismatch_text(label: str, mismatch: dict[str, Any] | None) -> str:
    if mismatch is None:
        return f"{label}=none"
    fields = [f"{label} stage={mismatch['stage']}"]
    for key in ("slot", "expert", "row", "expert_index"):
        if key in mismatch:
            fields.append(f"{key}={mismatch[key]}")
    fields.append(f"poolside_bits={mismatch['poolside_bits']}")
    fields.append(f"ds4_bits={mismatch['ds4_bits']}")
    return " ".join(fields)


def print_human_report(report: dict[str, Any]) -> None:
    for stage in ALL_STAGES:
        result = report["stages"][stage.name]
        first = result["first_mismatch"]
        first_text = "none" if first is None else mismatch_text("", first).strip()
        print(
            f"stage={stage.name} equal={result['equal_count']}/{result['elements']} "
            f"mismatches={result['mismatch_count']} first={first_text}"
        )
    print(mismatch_text("first_overall_mismatch", report["first_overall_mismatch"]))
    print(
        mismatch_text(
            "first_routed_expert_mismatch",
            report["first_routed_expert_mismatch"],
        )
    )
    microscope = report["microscope"]
    origin = microscope["origin"]
    binding = microscope["direct_output_binding"]
    print(
        "microscope_origin "
        f"token={origin['token']} layer={origin['layer']} "
        f"projection={origin['projection']} slot={origin['selected_slot']} "
        f"expert={origin['expert']} row={origin['row']} "
        f"poolside_matches_oracle={str(binding['poolside_matches_oracle']).lower()} "
        f"ds4_matches_oracle={str(binding['ds4_matches_oracle']).lower()}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poolside-moe", required=True, type=Path)
    parser.add_argument("--poolside-ffn-norm", required=True, type=Path)
    parser.add_argument("--ds4", required=True, type=Path)
    parser.add_argument("--microscope-fixture", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    options = parse_args(argv)
    try:
        report = build_report(
            options.poolside_moe,
            options.poolside_ffn_norm,
            options.ds4,
            options.microscope_fixture,
            options.run_manifest,
        )
        payload = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        options.json_out.write_text(payload, encoding="utf-8")
    except (ComparisonError, OSError) as error:
        print(f"compare_laguna_moe_execution: {error}", file=sys.stderr)
        return 2
    print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
