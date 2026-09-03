#!/usr/bin/env python3
"""Contract tests for the immutable Laguna compact-runtime manifest."""

from __future__ import annotations

import base64
import array
import copy
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "gguf-tools/quality-testing/compact_runtime_qualify.py"
SCHEMA_PATH = ROOT / "schemas/compact-runtime-benchmark-v1.schema.json"

SPEC = importlib.util.spec_from_file_location("compact_runtime_qualify", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load manifest builder: {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

TEST_PAGE_SIZE = TOOL.mmap.PAGESIZE

PREFIX = b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
SUFFIX = b"</user>\n<assistant></think>"
TARGETS = [512, 2048, 8192, 28672]
SEED = (
    b"The ring buffer stores each item at a position modulo its fixed capacity.\n"
    * 4096
)

MODEL_IDENTITY = {
    "path": "/models/laguna-s-2.1-Q4_K_M.gguf",
    "repository": "poolside/Laguna-S-2.1-GGUF",
    "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
    "filename": "laguna-s-2.1-Q4_K_M.gguf",
    "size_bytes": "68248759648",
    "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
    "device": "259",
    "inode": "9001",
    "mtime_ns": "1785600000000000000",
}

RUNTIME_IDENTITY = {
    "source_revision": "a" * 40,
    "oracle_tokenizer_revision": "15c9b92502fed6bc26842e98d11a6347caadb08e",
    "executable_path": "/src/ds4/ds4",
    "executable_sha256": "b" * 64,
    "device": "16777234",
    "inode": "7002",
    "size_bytes": "1234567",
    "mtime_ns": "1785600100000000000",
    "token_dump_argv": [
        "--dump-tokens",
        "--raw-prompt",
        "-m",
        "{model}",
        "--prompt-file",
        "{prompt}",
    ],
}

HOST_IDENTITY = {
    "hostname": "dgx-spark",
    "architecture": "aarch64",
    "kernel_release": "6.14.0-1012-nvidia",
    "kernel_version": "#12-NVIDIA SMP PREEMPT_DYNAMIC",
    "cuda_driver_version": "580.65.06",
    "cuda_runtime_version": "13.0.0",
    "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
    "filesystem": {
        "mount_point": "/home",
        "type": "ext4",
        "source": "/dev/nvme0n1p2",
        "device": "259:2",
        "options": "rw,relatime",
    },
    "nvme": {
        "device": "nvme0n1",
        "model": "SAMSUNG MZVL21T0HCLR-00B00",
        "serial": "S6P2NL0T123456",
        "firmware_revision": "GXA7401Q",
    },
    "io": {
        "direct_io": False,
        "cold_preparation_advice":
            "madvise_random+posix_fadvise_dontneed+madvise_dontneed+linux_madv_pageout_residual",
        "runtime_disposal_advice": "madvise_dontneed",
    },
}

WARM_OWNED_CATEGORY_NAMES = (
    "static_weights",
    "expert_cache_payload",
    "cache_metadata_address_tables",
    "kv_state",
    "graph_scratch",
    "pinned_staging",
    "other_host",
    "other_cuda",
)


def _warm_stability_sample(
    categories: dict[str, int],
    *,
    hit_before: int,
    hit_after: int,
    read_before: int,
    read_after: int,
) -> dict[str, object]:
    return {
        "owned_category_current_bytes": {
            name: str(categories[name]) for name in WARM_OWNED_CATEGORY_NAMES
        },
        "cache_acquire_hits_before": str(hit_before),
        "cache_acquire_hits_after": str(hit_after),
        "model_file_read_bytes_before": str(read_before),
        "model_file_read_bytes_after": str(read_after),
    }


def _warm_stability_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    categories = {
        "static_weights": 4 << 30,
        "expert_cache_payload": 8 << 30,
        "cache_metadata_address_tables": 1 << 20,
        "kv_state": 0,
        "graph_scratch": 0,
        "pinned_staging": 24 << 20,
        "other_host": 2 << 20,
        "other_cuda": 0,
    }
    cold = _warm_stability_sample(
        categories,
        hit_before=100,
        hit_after=105,
        read_before=1_000,
        read_after=5_000,
    )
    second_categories = dict(categories)
    second_categories["other_host"] += 64 << 20
    warm = [
        _warm_stability_sample(
            categories,
            hit_before=105,
            hit_after=110,
            read_before=5_000,
            read_after=7_000,
        ),
        _warm_stability_sample(
            second_categories,
            hit_before=110,
            hit_after=114,
            read_before=7_000,
            read_after=7_500,
        ),
        _warm_stability_sample(
            categories,
            hit_before=114,
            hit_after=115,
            read_before=7_500,
            read_after=7_500,
        ),
    ]
    return cold, warm


def deterministic_token_count(rendered: bytes) -> int:
    if not rendered.startswith(PREFIX) or not rendered.endswith(SUFFIX):
        raise AssertionError("builder did not use the pinned Laguna wrapper")
    payload = rendered[len(PREFIX) : -len(SUFFIX)]
    if payload != SEED[: len(payload)]:
        raise AssertionError("builder did not select an immutable seed prefix")
    return len(payload) + 17


def build_fixture() -> dict:
    return TOOL.build_manifest(
        Path(MODEL_IDENTITY["path"]),
        token_counter=deterministic_token_count,
        host_identity=copy.deepcopy(HOST_IDENTITY),
        model_identity=copy.deepcopy(MODEL_IDENTITY),
        runtime_identity=copy.deepcopy(RUNTIME_IDENTITY),
        seed_bytes=SEED,
        qualification_preflight=_qualification_preflight_fixture(),
    )


Mutation = tuple[str, Callable[[dict], None]]
PathPart = str | int


def _node_at(manifest: dict, path: tuple[PathPart, ...]) -> dict:
    node = manifest
    for part in path:
        node = node[part]
    return node


def _set(path: tuple[PathPart, ...], key: str, value: object) -> Callable[[dict], None]:
    return lambda manifest: _node_at(manifest, path).__setitem__(key, value)


def _delete(path: tuple[PathPart, ...], key: str) -> Callable[[dict], None]:
    return lambda manifest: _node_at(manifest, path).__delitem__(key)


NVML_COMPUTE_API = "nvmlDeviceGetComputeRunningProcesses_v2"
NVML_LIBRARY_VERSION = "13.580.126.09"
PRE_CHILD_CAPTURED_AT_UNIX_NS = "1785600200000000000"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_cold_preparation_fixture(
    directory: Path,
    *,
    ranges: list[dict[str, str]] | None = None,
) -> tuple[Path, Path, dict, str]:
    """Write a sparse model and the smallest Task-13 qualification plan."""
    page_size = TEST_PAGE_SIZE
    model = directory / "laguna-s-2.1-Q4_K_M.gguf"
    with model.open("wb") as handle:
        handle.truncate(8 * page_size)
        handle.seek(page_size)
        handle.write(b"original descriptor bytes")
    identity = model.stat()
    ledger = {
        "file_size": str(identity.st_size),
        "tensor_ranges": [
            {
                "class": "STATIC",
                "source_bytes": str(2 * page_size),
                "source_offset": str(page_size),
            },
            {
                "class": "ROUTED_EXPERT",
                "source_bytes": str(page_size),
                "source_offset": str(4 * page_size),
            },
        ],
    }
    safe_ranges = ranges or [
        {"bytes": str(2 * page_size), "offset": str(page_size)},
        {"bytes": str(page_size), "offset": str(4 * page_size)},
    ]
    plan = {
        "allocation": {"profile_id": "cache-8gib"},
        "ledger": ledger,
        "ledger_sha256": hashlib.sha256(
            TOOL.canonical_json_bytes(ledger)
        ).hexdigest(),
        "model": {
            "device": str(identity.st_dev),
            "filename": model.name,
            "inode": str(identity.st_ino),
            "mtime_ns": str(identity.st_mtime_ns),
            "repository": "poolside/Laguna-S-2.1-GGUF",
            "revision": "706fa69799926b6afde1af9e24ca2a4923f110a1",
            "sha256": _file_sha256(model),
            "size_bytes": str(identity.st_size),
        },
        "page_cache": {
            "eligible_unique_bytes": str(3 * page_size),
            "mapped_page_bytes": str(identity.st_size),
            "page_size": str(page_size),
            "ranges": safe_ranges,
            "unavoidable_bytes": str(identity.st_size - 3 * page_size),
        },
        "schema": "ds4.laguna.qualification-plan/v1",
    }
    plan_path = directory / "qualification-plan.json"
    plan_bytes = TOOL.canonical_json_bytes(plan)
    plan_path.write_bytes(plan_bytes)
    return model, plan_path, plan, hashlib.sha256(plan_bytes).hexdigest()


def _nvml_inventory(
    processes: list[dict[str, object]],
    *,
    gpu_uuid: str = HOST_IDENTITY["gpu_uuid"],
    api: str = NVML_COMPUTE_API,
) -> dict[str, object]:
    return {"api": api, "gpu_uuid": gpu_uuid, "processes": processes}


def _qualification_cold_preparation(
    model_identity: dict[str, object] = MODEL_IDENTITY,
) -> dict[str, object]:
    page_size = 4096
    model_size = int(model_identity["size_bytes"])
    eligible_bytes = (model_size // page_size) * page_size
    if eligible_bytes == 0:
        raise AssertionError("qualification fixture model needs one complete page")
    return {
        "plan_sha256": "c" * 64,
        "ledger_sha256": "d" * 64,
        "model_identity": {
            key: value for key, value in model_identity.items() if key != "path"
        },
        "page_size": str(page_size),
        "eligible_ranges": [
            {"offset": "0", "bytes": str(eligible_bytes)},
        ],
        "eligible_calls": 1,
        "eligible_bytes": str(eligible_bytes),
        "attempted_calls": 1,
        "attempted_bytes": str(eligible_bytes),
        "successful_calls": 1,
        "successful_bytes": str(eligible_bytes),
        "failed_calls": 0,
        "failed_bytes": "0",
        "errno_buckets": {},
        "residual_disposal": {
            "initial_resident_eligible_pages": 0,
            "pageout_retry_pages": 0,
            "residency_samples": 1,
            "mapping_touch_pages": 0,
            "mapping_touch_bytes": "0",
            "random_access_madvise": {
                "attempted_calls": 0,
                "attempted_bytes": "0",
                "successful_calls": 0,
                "successful_bytes": "0",
                "failed_calls": 0,
                "failed_bytes": "0",
                "errno_buckets": {},
            },
            "madvise": {
                "attempted_calls": 0,
                "attempted_bytes": "0",
                "successful_calls": 0,
                "successful_bytes": "0",
                "failed_calls": 0,
                "failed_bytes": "0",
                "errno_buckets": {},
            },
            "fadvise": {
                "attempted_calls": 0,
                "attempted_bytes": "0",
                "successful_calls": 0,
                "successful_bytes": "0",
                "failed_calls": 0,
                "failed_bytes": "0",
                "errno_buckets": {},
            },
        },
        "resident_bytes_after": "0",
        "unavoidable_bytes": str(
            ((model_size + page_size - 1) // page_size) * page_size
            - eligible_bytes
        ),
    }


def _qualification_preflight_fixture(
    *,
    model_identity: dict[str, object] = MODEL_IDENTITY,
    runtime_identity: dict[str, object] = RUNTIME_IDENTITY,
    host_identity: dict[str, object] = HOST_IDENTITY,
) -> dict[str, object]:
    return TOOL.freeze_qualification_preflight(
        _qualification_cold_preparation(model_identity),
        gpu_uuid=str(host_identity["gpu_uuid"]),
        runtime_identity=copy.deepcopy(runtime_identity),
        captured_at_unix_ns=PRE_CHILD_CAPTURED_AT_UNIX_NS,
        nvml_query=lambda gpu_uuid: {
            "library_version": NVML_LIBRARY_VERSION,
            "inventory": _nvml_inventory([], gpu_uuid=gpu_uuid),
        },
    )


def _qualification_binding_payload(
    cold_preparation: dict[str, object],
    nvml_pre_child: dict[str, object],
) -> dict[str, object]:
    return {
        "cold_preparation": cold_preparation,
        "nvml_pre_child": nvml_pre_child,
        "runtime": {
            "source_revision": RUNTIME_IDENTITY["source_revision"],
            "executable_sha256": RUNTIME_IDENTITY["executable_sha256"],
        },
    }


def schema_expressible_mutations() -> list[Mutation]:
    mutations: list[Mutation] = [
        (
            "uint64 max plus one",
            _set(("prompts", 0), "payload_prefix_bytes", "18446744073709551616"),
        ),
        (
            "uint64 larger overflow",
            _set(("prompts", 0), "payload_prefix_bytes", "99999999999999999999"),
        ),
        (
            "positive uint64 max plus one",
            _set(("model",), "device", "18446744073709551616"),
        ),
        (
            "positive uint64 larger overflow",
            _set(
                ("prompt_source", "tokenizer_runtime"),
                "inode",
                "99999999999999999999",
            ),
        ),
        ("base64 invalid padding", _set(("prompts", 0), "rendered_base64", "Zg===")),
        ("base64 invalid alphabet", _set(("prompts", 0), "rendered_base64", "!!!!")),
        (
            "base64 noncanonical low bits",
            _set(("prompts", 0), "rendered_base64", "Zh=="),
        ),
        (
            "model path relative",
            _set(("model",), "path", "models/laguna-s-2.1-Q4_K_M.gguf"),
        ),
        (
            "model path repeated slash",
            _set(("model",), "path", "/models//laguna-s-2.1-Q4_K_M.gguf"),
        ),
        (
            "model path wrong basename",
            _set(("model",), "path", "/models/not-laguna.gguf"),
        ),
        (
            "model path placeholder segment",
            _set(("model",), "path", "/models/TODO/laguna-s-2.1-Q4_K_M.gguf"),
        ),
        (
            "model path angle placeholder segment",
            _set(
                ("model",),
                "path",
                "/models/<placeholder>/laguna-s-2.1-Q4_K_M.gguf",
            ),
        ),
        (
            "model path embedded generic marker",
            _set(("model",), "path", "/models/<x>/laguna-s-2.1-Q4_K_M.gguf"),
        ),
        (
            "model path tab segment",
            _set(("model",), "path", "/models/has\ttab/laguna-s-2.1-Q4_K_M.gguf"),
        ),
        (
            "model path trailing slash",
            _set(("model",), "path", "/models/laguna-s-2.1-Q4_K_M.gguf/"),
        ),
    ]

    positive_fields = [
        (("model",), "device"),
        (("model",), "inode"),
        (("model",), "mtime_ns"),
        (("prompt_source", "tokenizer_runtime"), "device"),
        (("prompt_source", "tokenizer_runtime"), "inode"),
        (("prompt_source", "tokenizer_runtime"), "size_bytes"),
        (("prompt_source", "tokenizer_runtime"), "mtime_ns"),
        *((('prompts', index), "rendered_size_bytes") for index in range(4)),
    ]
    mutations.extend(
        (
            f"positive uint64 zero at {'.'.join(map(str, path + (key,)))}",
            _set(path, key, "0"),
        )
        for path, key in positive_fields
    )

    required_fields = [
        ((), "schema"),
        (("model",), "path"),
        (("host",), "hostname"),
        (("host", "filesystem"), "mount_point"),
        (("host", "nvme"), "device"),
        (("host", "io"), "direct_io"),
        (("prompt_source",), "seed_size_bytes"),
        (("prompt_source", "tokenizer_runtime"), "source_revision"),
        (("prompts", 0), "id"),
        (("sampling",), "max_generated_tokens"),
        (("execution",), "qualification_cold_preparations"),
        ((), "qualification_preflight"),
        (("qualification_preflight",), "binding_sha256"),
        (("qualification_preflight", "cold_preparation"), "eligible_ranges"),
        (("qualification_preflight", "nvml_pre_child"), "library_version"),
        (("profiles", 0), "profile_id"),
    ]
    for path, key in required_fields:
        location = ".".join(map(str, path + (key,)))
        mutations.extend(
            [
                (f"unknown key beside {location}", _set(path, "extra", "forbidden")),
                (f"missing required {location}", _delete(path, key)),
                (f"null required {location}", _set(path, key, None)),
            ]
        )
    return mutations


def schema_validates_string(schema: dict, subschema: dict, value: str) -> bool:
    """Evaluate the Draft 2020-12 string constraints used by this schema."""
    if "$ref" in subschema:
        resolved: object = schema
        for part in subschema["$ref"].removeprefix("#/").split("/"):
            if not isinstance(resolved, dict):
                raise AssertionError(f"invalid schema reference: {subschema['$ref']}")
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        if not isinstance(resolved, dict):
            raise AssertionError(f"invalid schema reference: {subschema['$ref']}")
        return schema_validates_string(schema, resolved, value)

    if subschema.get("type") == "string" and not isinstance(value, str):
        return False
    if "const" in subschema and value != subschema["const"]:
        return False
    if "minLength" in subschema and len(value) < subschema["minLength"]:
        return False
    if "pattern" in subschema and re.search(subschema["pattern"], value) is None:
        return False
    if "anyOf" in subschema and not any(
        schema_validates_string(schema, alternative, value)
        for alternative in subschema["anyOf"]
    ):
        return False
    if "not" in subschema and schema_validates_string(schema, subschema["not"], value):
        return False
    return True


class CompactRuntimeManifestContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_fixture()
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.schema_validator = Draft202012Validator(cls.schema)

    def test_checked_schema_matches_strict_validator_mutation_corpus(self) -> None:
        self.assertTrue(self.schema_validator.is_valid(self.manifest))
        TOOL.validate_manifest(self.manifest)

        mismatches = []
        for label, mutate in schema_expressible_mutations():
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            schema_accepts = self.schema_validator.is_valid(changed)
            try:
                TOOL.validate_manifest(changed)
                custom_accepts = True
            except ValueError:
                custom_accepts = False
            if schema_accepts or custom_accepts or schema_accepts != custom_accepts:
                mismatches.append(
                    f"{label}: schema_accepts={schema_accepts} "
                    f"custom_accepts={custom_accepts}"
                )

        self.assertEqual(mismatches, [], "schema/custom mismatches:\n" + "\n".join(mismatches))

    def test_schema_exists_parses_and_closes_every_object(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "ds4.compact-runtime-benchmark/v1")

        def assert_closed(node: object, location: str) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False, location)
                for key, value in node.items():
                    assert_closed(value, f"{location}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    assert_closed(value, f"{location}[{index}]")

        assert_closed(schema, "schema")
        self.assertIn("?!0{64}", schema["$defs"]["sha256"]["pattern"])
        self.assertIn("?!0{40}", schema["$defs"]["revision"]["pattern"])

    def test_identity_schema_rejects_mixed_case_runtime_placeholders(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        def field(definition: str, name: str) -> dict:
            return schema["$defs"][definition]["properties"][name]

        cases = [
            (field("model", "path"), "/models/FixMe/model.gguf"),
            (field("filesystem", "mount_point"), "/mnt/ToDo"),
            (field("filesystem", "type"), "TbD"),
            (field("filesystem", "source"), "/dev/UnKnOwN"),
            (field("filesystem", "options"), "rw,PlaceHolder"),
            (field("nvme", "device"), "ChangeMe"),
            (field("nvme", "model"), "an ExAmPlE device"),
            (field("nvme", "serial"), "N/A"),
            (field("nvme", "firmware_revision"), "NoNe"),
            (field("host", "hostname"), "NuLl"),
        ]
        for field_schema, value in cases:
            with self.subTest(value=value):
                self.assertFalse(schema_validates_string(schema, field_schema, value))

    def test_identity_schema_accepts_real_values(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        def field(definition: str, name: str) -> dict:
            return schema["$defs"][definition]["properties"][name]

        cases = [
            (field("model", "path"), MODEL_IDENTITY["path"]),
            *(
                (field("filesystem", name), value)
                for name, value in HOST_IDENTITY["filesystem"].items()
                if name != "device"
            ),
            *(
                (field("nvme", name), value)
                for name, value in HOST_IDENTITY["nvme"].items()
            ),
            *(
                (field("host", name), HOST_IDENTITY[name])
                for name in (
                    "hostname",
                    "architecture",
                    "kernel_release",
                    "kernel_version",
                    "cuda_driver_version",
                    "cuda_runtime_version",
                )
            ),
            (field("tokenizer_runtime", "executable_path"), RUNTIME_IDENTITY["executable_path"]),
        ]
        for field_schema, value in cases:
            with self.subTest(value=value):
                self.assertTrue(schema_validates_string(schema, field_schema, value))

    def test_builds_exact_native_template_prompts_and_hashes(self) -> None:
        prompts = self.manifest["prompts"]
        self.assertEqual([item["token_count"] for item in prompts], TARGETS)
        for item, target in zip(prompts, TARGETS, strict=True):
            rendered = base64.b64decode(item["rendered_base64"], validate=True)
            self.assertTrue(rendered.startswith(PREFIX))
            self.assertTrue(rendered.endswith(SUFFIX))
            payload = rendered[len(PREFIX) : -len(SUFFIX)]
            self.assertEqual(payload, SEED[: target - 17])
            self.assertEqual(item["payload_prefix_bytes"], str(len(payload)))
            self.assertEqual(item["rendered_size_bytes"], str(len(rendered)))
            self.assertEqual(item["sha256"], hashlib.sha256(rendered).hexdigest())
            self.assertEqual(deterministic_token_count(rendered), target)

        source = self.manifest["prompt_source"]
        self.assertEqual(source["template_revision"], "poolside-laguna-s-2.1-native-nothink-v1")
        self.assertEqual(base64.b64decode(source["template_prefix_base64"]), PREFIX)
        self.assertEqual(base64.b64decode(source["template_suffix_base64"]), SUFFIX)
        self.assertEqual(source["seed_sha256"], hashlib.sha256(SEED).hexdigest())
        self.assertEqual(source["tokenizer_runtime"], RUNTIME_IDENTITY)

    def test_freezes_sampling_execution_profiles_and_eval_cases(self) -> None:
        self.assertEqual(
            self.manifest["sampling"],
            {
                "max_generated_tokens": 512,
                "temperature": 0,
                "top_k": 0,
                "top_p": 1,
                "min_p": 0.05,
                "seed": 1,
                "stop_sequences": [],
                "stop_token_policy": "model-native",
            },
        )
        self.assertEqual(
            self.manifest["execution"],
            {
                "qualification_cold_preparations": 1,
                "fresh_process_runs": 1,
                "same_process_warm_repetitions": 3,
                "whole_request_timeout_seconds": 2700,
                "first_token_timeout_seconds": 900,
                "warm_statistic": "median-of-exactly-three",
                "scope": "each-profile-prompt-pair",
            },
        )
        self.assertEqual(
            [profile["cache_bytes"] for profile in self.manifest["profiles"]],
            [str(8 << 30), str(12 << 30), str(16 << 30)],
        )
        self.assertEqual(
            [profile["prompt_order"] for profile in self.manifest["profiles"]],
            [
                [512, 2048, 28672, 8192],
                [2048, 8192, 512, 28672],
                [8192, 28672, 2048, 512],
            ],
        )
        self.assertEqual(
            self.manifest["eval_case_ids"],
            [
                "recNu3MXkvWUzHZr9",
                "001b51d76b4d422988f2c11f104a2c6c",
                "aime2025-01",
                "compsec-076",
            ],
        )

    def test_binds_complete_model_host_and_runtime_identity(self) -> None:
        self.assertEqual(self.manifest["model"], MODEL_IDENTITY)
        self.assertEqual(self.manifest["host"], HOST_IDENTITY)
        TOOL.validate_manifest(self.manifest)

    def test_rejects_placeholders_missing_bytes_reordering_and_stale_hashes(self) -> None:
        cases: list[tuple[str, callable]] = [
            ("placeholder", lambda d: d["host"].__setitem__("hostname", "TODO")),
            ("missing bytes", lambda d: d["prompts"][0].__setitem__("rendered_base64", "")),
            ("hash", lambda d: d["prompts"][0].__setitem__("sha256", "0" * 64)),
            ("prompt order", lambda d: d["prompts"].reverse()),
            ("profile order", lambda d: d["profiles"].reverse()),
            ("cache order", lambda d: d["profiles"][0].__setitem__("cache_bytes", str(12 << 30))),
            (
                "tokenizer revision",
                lambda d: d["prompt_source"]["tokenizer_runtime"].__setitem__(
                    "oracle_tokenizer_revision", "c" * 40
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                changed = copy.deepcopy(self.manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    TOOL.validate_manifest(changed)

    def test_rejects_result_fields_before_hashing(self) -> None:
        for key in ("result", "results", "status", "measurements"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed[key] = {}
                with self.assertRaisesRegex(ValueError, "result|unknown key"):
                    TOOL.manifest_sha256(changed)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            TOOL.loads_strict('{"schema":"x","schema":"y"}')
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    TOOL.loads_strict('{"value":' + value + "}")
        changed = copy.deepcopy(self.manifest)
        changed["sampling"]["min_p"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            TOOL.validate_manifest(changed)

    def test_numeric_fields_require_the_exact_json_number_kind(self) -> None:
        mutations = [
            lambda d: d["sampling"].__setitem__("max_generated_tokens", False),
            lambda d: d["sampling"].__setitem__("max_generated_tokens", 512.0),
            lambda d: d["sampling"].__setitem__("temperature", 0.0),
            lambda d: d["sampling"].__setitem__("top_k", False),
            lambda d: d["sampling"].__setitem__("top_k", 0.0),
            lambda d: d["sampling"].__setitem__("top_p", 1.0),
            lambda d: d["sampling"].__setitem__("seed", True),
            lambda d: d["sampling"].__setitem__("seed", 1.0),
            lambda d: d["execution"].__setitem__("fresh_process_runs", True),
            lambda d: d["execution"].__setitem__("same_process_warm_repetitions", 3.0),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "integer|float|number kind"):
                TOOL.validate_manifest(changed)

    def test_strict_parser_preserves_zero_number_spelling(self) -> None:
        canonical = json.dumps(self.manifest, separators=(",", ":"))
        float_spelling = canonical.replace('"temperature":0', '"temperature":0.0', 1)
        parsed_integer = TOOL.loads_strict(canonical)
        parsed_float = TOOL.loads_strict(float_spelling)
        self.assertIs(type(parsed_integer["sampling"]["temperature"]), int)
        self.assertIs(type(parsed_float["sampling"]["temperature"]), float)
        with self.assertRaisesRegex(ValueError, "integer|number kind"):
            TOOL.validate_manifest(parsed_float)

    def test_supported_domain_canonical_json_matches_rfc8785_boundaries(self) -> None:
        value = {"\U0001f600": 1, "\u20ac": 2, "a": "\b\n", "n": [0.0, 1.0, 0.05]}
        self.assertEqual(
            TOOL.canonical_json_bytes(value),
            b'{"a":"\\b\\n","n":[0,1,0.05],"\xe2\x82\xac":2,"\xf0\x9f\x98\x80":1}',
        )
        with self.assertRaisesRegex(ValueError, "unsupported.*RFC 8785"):
            TOOL.canonical_json_bytes({"outside_contract": 1e30})

    def test_manifest_hash_is_canonical_and_validated(self) -> None:
        digest = TOOL.manifest_sha256(self.manifest)
        reordered = dict(reversed(list(self.manifest.items())))
        self.assertEqual(TOOL.manifest_sha256(reordered), digest)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_atomic_writer_leaves_only_complete_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "compact-runtime-benchmark-v1.json"
            TOOL.write_manifest_atomic(output, self.manifest)
            loaded = TOOL.load_manifest(output)
            self.assertEqual(loaded, self.manifest)
            self.assertEqual(list(Path(tmp).iterdir()), [output])

    def test_atomic_writer_never_replaces_a_raced_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "compact-runtime-benchmark-v1.json"
            real_link = os.link

            def race(source: os.PathLike | str | bytes,
                     target: os.PathLike | str | bytes,
                     **kwargs: object) -> None:
                Path(target).write_bytes(b"raced-target\n")
                real_link(source, target, **kwargs)

            with mock.patch.object(TOOL.os, "link", side_effect=race):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    TOOL.write_manifest_atomic(output, self.manifest)
            self.assertEqual(output.read_bytes(), b"raced-target\n")
            self.assertEqual(list(Path(tmp).iterdir()), [output])

    @contextlib.contextmanager
    def fake_cli_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "source.txt").write_text("tokenizer source\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "source.txt"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(root), "-c", "user.name=DS4 Test",
                    "-c", "user.email=ds4@example.invalid", "commit", "-q", "-m", "source",
                ],
                check=True,
                capture_output=True,
            )
            oracle_revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            invocation_log = Path(tmp) / "tokenizer-invocations.jsonl"
            executable = root / "ds4"
            executable.write_text(
                """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args[:3] != ["--dump-tokens", "--raw-prompt", "-m"] or args[4] != "--prompt-file" or len(args) != 6:
    print("bad argv", file=sys.stderr)
    raise SystemExit(2)
rendered = pathlib.Path(args[5]).read_bytes()
prefix = b"\\xe3\\x80\\x88|EOS|\\xe3\\x80\\x89<user>"
suffix = b"</user>\\n<assistant></think>"
if not rendered.startswith(prefix) or not rendered.endswith(suffix):
    print("bad rendered prompt", file=sys.stderr)
    raise SystemExit(2)
payload = rendered[len(prefix):-len(suffix)]
count = len(payload) // 10 + 17
with open(os.environ["FAKE_DS4_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
print("[" + ",".join(["0"] * count) + "]")
""",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            oracle = root / "oracle.json"
            oracle.write_text(
                json.dumps({"provenance": {"tokenizer_runtime_commit": oracle_revision}}),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(root), "add", "ds4", "oracle.json"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(root), "-c", "user.name=DS4 Test",
                    "-c", "user.email=ds4@example.invalid", "commit", "-q", "-m", "runtime",
                ],
                check=True,
                capture_output=True,
            )

            model, plan, _, plan_sha256 = _write_cold_preparation_fixture(Path(tmp))
            model_bytes = model.read_bytes()
            output = Path(tmp) / "compact-runtime-benchmark-v1.json"
            events: list[object] = []
            guard = {"cold_complete": False, "enforce": True}
            real_cold_prepare = TOOL.cold_prepare_from_plan
            real_bind_model = TOOL.bind_model_identity
            real_token_count = TOOL.RawDs4TokenCounter.__call__

            def bind_model_for_cli(model_path: Path) -> dict[str, str]:
                if guard["cold_complete"] and guard["enforce"]:
                    raise AssertionError("model binding occurred after cold preparation")
                events.append("model-access")
                return real_bind_model(model_path)

            def count_tokens_for_cli(
                counter: object,
                rendered: bytes,
            ) -> int:
                if guard["cold_complete"] and guard["enforce"]:
                    raise AssertionError("tokenizer subprocess ran after cold preparation")
                events.append("tokenizer-subprocess")
                return real_token_count(counter, rendered)

            def cold_prepare_for_cli(
                model_path: Path,
                plan_path: Path,
                expected_sha256: str,
            ) -> dict[str, object]:
                result = real_cold_prepare(
                    model_path,
                    plan_path,
                    expected_sha256,
                    advise=lambda descriptor, offset, length: None,
                    sample_residency=lambda descriptor, file_size, page_size: bytes(
                        (file_size + page_size - 1) // page_size
                    ),
                )
                guard["cold_complete"] = True
                events.append(("cold", model_path, plan_path, expected_sha256))
                return result

            def collect_nvml_for_cli(gpu_uuid: str) -> dict[str, object]:
                self.assertTrue(
                    guard["cold_complete"],
                    "pre-child NVML capture must follow cold preparation",
                )
                events.append(("nvml", gpu_uuid))
                return {
                    "library_version": NVML_LIBRARY_VERSION,
                    "inventory": _nvml_inventory(
                        [
                            {
                                "pid": 9001,
                                "used_gpu_memory_bytes": str(1 << 30),
                            }
                        ],
                        gpu_uuid=gpu_uuid,
                    ),
                }

            def capture_time_ns() -> int:
                self.assertEqual(events[-1], ("nvml", HOST_IDENTITY["gpu_uuid"]))
                events.append("clock")
                return int(PRE_CHILD_CAPTURED_AT_UNIX_NS)

            patches = [
                mock.patch.object(TOOL, "ROOT", root),
                mock.patch.object(TOOL, "ORACLE_MANIFEST_PATH", oracle),
                mock.patch.object(TOOL, "ORACLE_TOKENIZER_REVISION", oracle_revision),
                mock.patch.object(TOOL, "MODEL_SIZE", len(model_bytes)),
                mock.patch.object(TOOL, "MODEL_SHA256", hashlib.sha256(model_bytes).hexdigest()),
                mock.patch.object(
                    TOOL, "collect_host_identity", return_value=copy.deepcopy(HOST_IDENTITY)
                ),
                mock.patch.object(
                    TOOL, "bind_model_identity", side_effect=bind_model_for_cli
                ),
                mock.patch.object(
                    TOOL.RawDs4TokenCounter,
                    "__call__",
                    side_effect=count_tokens_for_cli,
                    autospec=True,
                ),
                mock.patch.object(
                    TOOL, "cold_prepare_from_plan", side_effect=cold_prepare_for_cli
                ),
                mock.patch.object(
                    TOOL, "collect_nvml_pre_child", side_effect=collect_nvml_for_cli
                ),
                mock.patch.object(TOOL.time, "time_ns", side_effect=capture_time_ns),
                mock.patch.dict(os.environ, {"FAKE_DS4_LOG": str(invocation_log)}),
            ]
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                yield {
                    "root": root,
                    "model": model,
                    "plan": plan,
                    "plan_sha256": plan_sha256,
                    "output": output,
                    "log": invocation_log,
                    "model_bytes": model_bytes,
                    "events": events,
                    "guard": guard,
                }

    def fake_cli_build_argv(self, fixture: dict[str, object]) -> list[str]:
        return [
            "manifest", "build",
            "--model", str(fixture["model"]),
            "--output", str(fixture["output"]),
            "--qualification-plan", str(fixture["plan"]),
            "--trusted-qualification-plan-sha256", str(fixture["plan_sha256"]),
        ]

    def build_with_fake_cli(self, fixture: dict[str, object]) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                TOOL.main(self.fake_cli_build_argv(fixture)),
                0,
            )
        fixture["guard"]["enforce"] = False

    def test_cli_build_and_verify_bind_and_retokenize_real_artifacts(self) -> None:
        with self.fake_cli_environment() as fixture:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    TOOL.main(self.fake_cli_build_argv(fixture)),
                    0,
                )
                build_events = list(fixture["events"])
                fixture["guard"]["enforce"] = False
                self.assertEqual(
                    TOOL.main(["manifest", "verify", "--manifest", str(fixture["output"])]),
                    0,
                )
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("manifest_sha256=", stdout.getvalue())
            events = build_events
            cold_event = (
                "cold",
                fixture["model"],
                fixture["plan"],
                fixture["plan_sha256"],
            )
            cold_index = events.index(cold_event)
            self.assertIn("model-access", events[:cold_index])
            self.assertIn("tokenizer-subprocess", events[:cold_index])
            self.assertEqual(
                events[cold_index : cold_index + 3],
                [cold_event, ("nvml", HOST_IDENTITY["gpu_uuid"]), "clock"],
            )
            self.assertNotIn("model-access", events[cold_index + 1 :])
            self.assertNotIn("tokenizer-subprocess", events[cold_index + 1 :])
            manifest = TOOL.load_manifest(fixture["output"])
            self.assertEqual(
                manifest["qualification_preflight"]["nvml_pre_child"]
                ["captured_at_unix_ns"],
                PRE_CHILD_CAPTURED_AT_UNIX_NS,
            )
            invocations = [
                json.loads(line) for line in Path(fixture["log"]).read_text().splitlines()
            ]
            self.assertGreater(len(invocations), 8)
            for argv in invocations:
                self.assertEqual(argv[:3], ["--dump-tokens", "--raw-prompt", "-m"])
                self.assertEqual(argv[3], str(Path(fixture["model"]).resolve()))
                self.assertEqual(argv[4], "--prompt-file")

    def test_verify_rejects_a_512_token_prompt_relabelled_as_2048(self) -> None:
        with self.fake_cli_environment() as fixture:
            self.build_with_fake_cli(fixture)
            manifest = TOOL.load_manifest(fixture["output"])
            first = manifest["prompts"][0]
            second = manifest["prompts"][1]
            for key in (
                "payload_prefix_bytes", "rendered_size_bytes", "rendered_base64", "sha256"
            ):
                second[key] = first[key]
            tampered = Path(fixture["output"]).with_name("tampered.json")
            TOOL.write_manifest_atomic(tampered, manifest)
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    TOOL.main(["manifest", "verify", "--manifest", str(tampered)]),
                    1,
                )
            self.assertIn("recorded 2048", stderr.getvalue())
            self.assertIn("observed 512", stderr.getvalue())

    def test_verify_rejects_changed_model_binding(self) -> None:
        with self.fake_cli_environment() as fixture:
            self.build_with_fake_cli(fixture)
            model = Path(fixture["model"])
            original = fixture["model_bytes"]
            assert isinstance(original, bytes)
            model.write_bytes(b"x" + original[1:])
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    TOOL.main(["manifest", "verify", "--manifest", str(fixture["output"])]),
                    1,
                )
            self.assertIn("SHA-256", stderr.getvalue())

    def test_verify_rejects_changed_runtime_binding(self) -> None:
        with self.fake_cli_environment() as fixture:
            self.build_with_fake_cli(fixture)
            executable = Path(fixture["root"]) / "ds4"
            executable.write_text(
                executable.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    TOOL.main(["manifest", "verify", "--manifest", str(fixture["output"])]),
                    1,
                )
            self.assertIn("dirty", stderr.getvalue())

    def test_runtime_binding_requires_repo_ds4_and_counter_rejects_bad_dump(self) -> None:
        with self.fake_cli_environment() as fixture:
            other = Path(fixture["root"]) / "other"
            other.write_text("not ds4\n", encoding="utf-8")
            other.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "repo ./ds4"):
                TOOL.bind_runtime_identity(other)

            bad = Path(fixture["root"]) / "bad-tokenizer"
            bad.write_text("#!/bin/sh\nprintf 'not-a-list\\n'\n", encoding="utf-8")
            bad.chmod(0o755)
            runtime = copy.deepcopy(RUNTIME_IDENTITY)
            runtime["executable_path"] = str(bad)
            counter = TOOL.RawDs4TokenCounter(Path(fixture["model"]), runtime)
            with self.assertRaisesRegex(ValueError, "token dump is invalid"):
                counter(PREFIX + b"payload" + SUFFIX)

    def test_rejects_identity_placeholders_and_zero_sentinels(self) -> None:
        mutations = [
            lambda d: d["host"].__setitem__("cuda_driver_version", "FIXME"),
            lambda d: d["host"]["nvme"].__setitem__("serial", "example"),
            lambda d: d["host"].__setitem__(
                "gpu_uuid", "GPU-00000000-0000-0000-0000-000000000000"
            ),
            lambda d: d["prompt_source"]["tokenizer_runtime"].__setitem__(
                "source_revision", "0" * 40
            ),
            lambda d: d["prompt_source"]["tokenizer_runtime"].__setitem__(
                "executable_sha256", "0" * 64
            ),
            lambda d: d["prompt_source"]["tokenizer_runtime"].__setitem__(
                "executable_path", "<path>"
            ),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "placeholder|sentinel|zero|invalid"):
                TOOL.validate_manifest(changed)

    def test_exact_prefix_selection_fails_instead_of_padding(self) -> None:
        def impossible(rendered: bytes) -> int:
            self.assertTrue(rendered.startswith(PREFIX) and rendered.endswith(SUFFIX))
            payload = rendered[len(PREFIX) : -len(SUFFIX)]
            return len(payload) * 2

        with self.assertRaisesRegex(ValueError, "exactly 513 native-template tokens"):
            TOOL.select_rendered_prompt(SEED[:1024], 513, impossible)

    def test_parser_exposes_manifest_and_sequence_commands(self) -> None:
        build = TOOL.parse_args([
            "manifest", "build", "--model", "/m", "--output", "/o",
            "--qualification-plan", "/p",
            "--trusted-qualification-plan-sha256", "c" * 64,
        ])
        verify = TOOL.parse_args(["manifest", "verify", "--manifest", "/m"])
        self.assertEqual((build.command, build.action), ("manifest", "build"))
        self.assertEqual(build.qualification_plan, Path("/p"))
        self.assertEqual(build.trusted_qualification_plan_sha256, "c" * 64)
        self.assertEqual((verify.command, verify.action), ("manifest", "verify"))
        sequence = TOOL.parse_args([
            "sequence", "build", "--manifest", "/m", "--profile-id", "cache-8gib",
            "--prompt-id", "native-512", "--output", "/o",
        ])
        self.assertEqual((sequence.command, sequence.action), ("sequence", "build"))
        self.assertEqual(sequence.manifest, Path("/m"))
        self.assertEqual(sequence.profile_id, "cache-8gib")
        self.assertEqual(sequence.prompt_id, "native-512")
        self.assertEqual(sequence.output, Path("/o"))
        for missing in (
            "--qualification-plan",
            "--trusted-qualification-plan-sha256",
        ):
            argv = [
                "manifest", "build", "--model", "/m", "--output", "/o",
                "--qualification-plan", "/p",
                "--trusted-qualification-plan-sha256", "c" * 64,
            ]
            index = argv.index(missing)
            del argv[index : index + 2]
            with self.subTest(missing=missing), self.assertRaises(SystemExit):
                TOOL.parse_args(argv)
        with mock.patch.dict(
            os.environ,
            {
                "DS4_QUALIFICATION_PLAN": "/env/plan",
                "DS4_QUALIFICATION_PLAN_SHA256": "c" * 64,
            },
        ), self.assertRaises(SystemExit):
            TOOL.parse_args([
                "manifest", "build", "--model", "/m", "--output", "/o"
            ])
        with self.assertRaises(SystemExit):
            TOOL.parse_args([
                "manifest", "build", "--model", "/m", "--output", "/o",
                "--qualification-plan", "/p",
                "--qualification-plan-sha256", "c" * 64,
            ])
        with self.assertRaises(SystemExit):
            TOOL.parse_args(["run"])


    def test_build_qualification_sequence_is_byte_exact_for_all_profile_orders(self) -> None:
        original = copy.deepcopy(self.manifest)
        digest = TOOL.manifest_sha256(self.manifest)
        expected_profiles = (
            ("cache-8gib", 8589934592, (512, 2048, 28672, 8192)),
            ("cache-12gib", 12884901888, (2048, 8192, 512, 28672)),
            ("cache-16gib", 17179869184, (8192, 28672, 2048, 512)),
        )
        for profile_id, cache_bytes, order in expected_profiles:
            for order_index, token_count in enumerate(order):
                prompt_id = f"native-{token_count}"
                with self.subTest(profile_id=profile_id, prompt_id=prompt_id):
                    rendered = base64.b64decode(
                        next(
                            item for item in self.manifest["prompts"]
                            if item["id"] == prompt_id
                        )["rendered_base64"],
                        validate=True,
                    )
                    expected = "\n".join([
                        "schema=ds4.qualification-sequence/v1",
                        f"manifest_sha256={digest}",
                        f"profile_id={profile_id}",
                        f"cache_bytes={cache_bytes}",
                        f"prompt_order_index={order_index}",
                        f"prompt_id={prompt_id}",
                        f"prompt_tokens={token_count}",
                        "mode=streamed",
                        f"input_size_bytes={len(rendered)}",
                        f"input_sha256={hashlib.sha256(rendered).hexdigest()}",
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
                    ]).encode("ascii") + b"\n"
                    self.assertEqual(
                        TOOL.build_qualification_sequence(
                            self.manifest, profile_id, prompt_id
                        ),
                        expected,
                    )
        self.assertEqual(self.manifest, original)

    def test_build_qualification_sequence_validates_manifest_and_rejects_mutation(self) -> None:
        mutations: list[tuple[str, Callable[[dict], None]]] = [
            ("unknown top-level key", lambda value: value.__setitem__("extra", 1)),
            ("missing sampling", lambda value: value.__delitem__("sampling")),
            (
                "reordered profile",
                lambda value: value["profiles"].__setitem__(
                    0, {**value["profiles"][0], "prompt_order": [2048, 512, 28672, 8192]}
                ),
            ),
            (
                "stale prompt bytes",
                lambda value: value["prompts"][0].__setitem__(
                    "rendered_base64", base64.b64encode(b"changed").decode("ascii")
                ),
            ),
            (
                "wrong sampling",
                lambda value: value["sampling"].__setitem__("temperature", 1),
            ),
        ]
        for label, mutate in mutations:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises(ValueError):
                TOOL.build_qualification_sequence(changed, "cache-8gib", "native-512")
        for profile_id, prompt_id in (("cache-4gib", "native-512"), ("cache-8gib", "native-999")):
            with self.subTest(profile_id=profile_id, prompt_id=prompt_id), self.assertRaisesRegex(
                ValueError, "profile|prompt"
            ):
                TOOL.build_qualification_sequence(self.manifest, profile_id, prompt_id)

    def test_sequence_bounds_cover_worst_case_canonical_base64_without_giant_buffers(self) -> None:
        max_input = getattr(TOOL, "QUALIFICATION_SEQUENCE_MAX_INPUT_BYTES", None)
        max_serialized = getattr(
            TOOL, "QUALIFICATION_SEQUENCE_MAX_SERIALIZED_BYTES", None
        )
        self.assertIsNotNone(max_input)
        self.assertIsNotNone(max_serialized)
        if max_input is None or max_serialized is None:
            return

        sample = TOOL.build_qualification_sequence(
            self.manifest, "cache-8gib", "native-512"
        )
        self.assertEqual(sample.count(b"\n"), TOOL.QUALIFICATION_SEQUENCE_LINE_COUNT)
        base64_line = next(
            line for line in sample.splitlines(keepends=True)
            if line.startswith(b"input_base64=")
        )
        encoded_input = base64_line[len(b"input_base64=") : -1]
        fixed_line_overhead = len(sample) - len(encoded_input)
        worst_case_base64 = 4 * ((max_input + 2) // 3)
        self.assertLessEqual(
            worst_case_base64 + fixed_line_overhead,
            max_serialized,
            "24-line sequence overhead plus canonical base64 must fit",
        )
        self.assertGreater(max_serialized, max_input)

        header = (ROOT / "ds4_bench_sequence.h").read_text(encoding="utf-8")

        def header_bound(macro: str) -> int:
            match = re.search(
                rf"#define\s+{macro}\s+\((\d+)u\s*\*\s*1024u\s*\*\s*1024u\)",
                header,
            )
            self.assertIsNotNone(match, f"missing byte bound macro {macro}")
            assert match is not None
            return int(match.group(1)) << 20

        self.assertEqual(max_input, header_bound("DS4_BENCH_SEQUENCE_MAX_INPUT_BYTES"))
        self.assertEqual(max_serialized, header_bound("DS4_BENCH_SEQUENCE_MAX_FILE_BYTES"))

    def test_sequence_build_cli_is_atomic_no_replace_and_model_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest_path = directory / "manifest.json"
            output = directory / "sequence"
            TOOL.write_manifest_atomic(manifest_path, self.manifest)
            expected = TOOL.build_qualification_sequence(
                self.manifest, "cache-8gib", "native-512"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "sequence", "build",
                "--manifest", str(manifest_path),
                "--profile-id", "cache-8gib",
                "--prompt-id", "native-512",
                "--output", str(output),
            ]
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(TOOL.main(argv), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(output.read_bytes(), expected)
            self.assertIn(f"manifest_sha256={TOOL.manifest_sha256(self.manifest)}", stdout.getvalue())
            first = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr := io.StringIO()):
                self.assertEqual(TOOL.main(argv), 1)
            self.assertEqual(output.read_bytes(), first)
            self.assertRegex(stderr.getvalue(), "already exists|replace")
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["manifest.json", "sequence"])

    def test_sequence_atomic_writer_never_replaces_a_raced_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sequence"
            payload = b"candidate-sequence\n"
            real_link = os.link

            def race(source: os.PathLike | str | bytes,
                     target: os.PathLike | str | bytes,
                     **kwargs: object) -> None:
                Path(target).write_bytes(b"racing-sequence\n")
                real_link(source, target, **kwargs)

            with mock.patch.object(TOOL.os, "link", side_effect=race):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    TOOL.write_qualification_sequence_atomic(output, payload)
            self.assertEqual(output.read_bytes(), b"racing-sequence\n")
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()), ["sequence"])

class WarmStabilityContractTest(unittest.TestCase):
    def test_freezes_warm_stability_constants_and_accepts_boundary_evidence(self) -> None:
        self.assertEqual(TOOL.WARM_STABILITY_REPETITIONS, 3)
        self.assertEqual(
            TOOL.WARM_OWNED_CATEGORY_DRIFT_LIMIT_BYTES,
            64 << 20,
        )
        self.assertEqual(
            tuple(TOOL.RUNTIME_OWNED_CATEGORY_NAMES),
            WARM_OWNED_CATEGORY_NAMES,
        )
        cold, warm = _warm_stability_fixture()
        TOOL.validate_warm_stability_samples(cold, warm)

    def test_requires_exactly_three_same_process_warm_samples(self) -> None:
        cold, warm = _warm_stability_fixture()
        for samples in (warm[:2], warm + [copy.deepcopy(warm[-1])]):
            with self.subTest(count=len(samples)), self.assertRaises(ValueError):
                TOOL.validate_warm_stability_samples(cold, samples)

    def test_rejects_owned_category_drift_beyond_the_inclusive_limit(self) -> None:
        cold, warm = _warm_stability_fixture()
        changed = copy.deepcopy(warm)
        first = int(
            changed[0]["owned_category_current_bytes"]["other_host"]
        )
        changed[1]["owned_category_current_bytes"]["other_host"] = str(
            first + (64 << 20) + 1
        )
        with self.assertRaises(ValueError):
            TOOL.validate_warm_stability_samples(cold, changed)

    def test_rejects_any_nonconstant_monotonically_growing_category(self) -> None:
        cold, warm = _warm_stability_fixture()
        for values in ((0, 1, 2), (0, 0, 1)):
            with self.subTest(values=values):
                changed = copy.deepcopy(warm)
                for sample, value in zip(changed, values, strict=True):
                    sample["owned_category_current_bytes"]["other_cuda"] = str(
                        value
                    )
                with self.assertRaises(ValueError):
                    TOOL.validate_warm_stability_samples(cold, changed)

    def test_each_warm_sample_adds_hits_and_reads_no_more_than_cold(self) -> None:
        cold, warm = _warm_stability_fixture()
        zero_hits = copy.deepcopy(warm)
        zero_hits[1]["cache_acquire_hits_after"] = zero_hits[1][
            "cache_acquire_hits_before"
        ]
        excessive_reads = copy.deepcopy(warm)
        cold_read_delta = (
            int(cold["model_file_read_bytes_after"])
            - int(cold["model_file_read_bytes_before"])
        )
        excessive_reads[1]["model_file_read_bytes_after"] = str(
            int(excessive_reads[1]["model_file_read_bytes_before"])
            + cold_read_delta
            + 1
        )
        for label, changed in (
            ("zero hit delta", zero_hits),
            ("warm read delta", excessive_reads),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                TOOL.validate_warm_stability_samples(cold, changed)

    def test_requires_real_cold_io_and_contiguous_same_process_counters(self) -> None:
        cold, warm = _warm_stability_fixture()

        zero_cold_io = copy.deepcopy(cold)
        zero_cold_io["model_file_read_bytes_after"] = zero_cold_io[
            "model_file_read_bytes_before"
        ]
        zero_cold_warm = copy.deepcopy(warm)
        read_cursor = zero_cold_io["model_file_read_bytes_after"]
        for sample in zero_cold_warm:
            sample["model_file_read_bytes_before"] = read_cursor
            sample["model_file_read_bytes_after"] = read_cursor

        discontinuous = copy.deepcopy(warm)
        discontinuous[1]["cache_acquire_hits_before"] = "112"
        discontinuous[1]["cache_acquire_hits_after"] = "116"
        discontinuous[2]["cache_acquire_hits_before"] = "116"
        discontinuous[2]["cache_acquire_hits_after"] = "117"

        for label, changed_cold, changed_warm in (
            ("zero cold routed I/O", zero_cold_io, zero_cold_warm),
            ("counter discontinuity", cold, discontinuous),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                TOOL.validate_warm_stability_samples(
                    changed_cold,
                    changed_warm,
                )

    def test_rejects_counter_regression_and_uint64_overflow(self) -> None:
        cold, warm = _warm_stability_fixture()

        hit_regression = copy.deepcopy(warm)
        hit_regression[1]["cache_acquire_hits_after"] = str(
            int(hit_regression[1]["cache_acquire_hits_before"]) - 1
        )
        read_regression = copy.deepcopy(warm)
        read_regression[1]["model_file_read_bytes_after"] = str(
            int(read_regression[1]["model_file_read_bytes_before"]) - 1
        )
        cross_sample_regression = copy.deepcopy(warm)
        cross_sample_regression[0]["cache_acquire_hits_before"] = str(
            int(cold["cache_acquire_hits_after"]) - 1
        )
        counter_overflow = copy.deepcopy(warm)
        counter_overflow[1]["model_file_read_bytes_after"] = str(1 << 64)
        category_overflow = copy.deepcopy(warm)
        category_overflow[1]["owned_category_current_bytes"][
            "static_weights"
        ] = str(1 << 64)

        for label, changed in (
            ("hit regression", hit_regression),
            ("read regression", read_regression),
            ("cross-sample regression", cross_sample_regression),
            ("counter overflow", counter_overflow),
            ("category overflow", category_overflow),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                TOOL.validate_warm_stability_samples(cold, changed)

    def test_rejects_malformed_sample_and_category_counts(self) -> None:
        cold, warm = _warm_stability_fixture()
        missing_category = copy.deepcopy(warm)
        del missing_category[0]["owned_category_current_bytes"]["kv_state"]
        extra_category = copy.deepcopy(warm)
        extra_category[0]["owned_category_current_bytes"]["unknown"] = "0"
        missing_counter = copy.deepcopy(warm)
        del missing_counter[0]["model_file_read_bytes_after"]
        noncanonical_counter = copy.deepcopy(warm)
        noncanonical_counter[0]["cache_acquire_hits_after"] = "0110"
        numeric_category = copy.deepcopy(warm)
        numeric_category[0]["owned_category_current_bytes"]["kv_state"] = 0
        boolean_counter = copy.deepcopy(warm)
        boolean_counter[0]["cache_acquire_hits_after"] = True

        for label, changed in (
            ("missing category", missing_category),
            ("extra category", extra_category),
            ("missing counter", missing_counter),
            ("noncanonical counter", noncanonical_counter),
            ("numeric category", numeric_category),
            ("boolean counter", boolean_counter),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                TOOL.validate_warm_stability_samples(cold, changed)


class ColdPreparationContractTest(unittest.TestCase):
    def test_caller_owned_descriptor_survives_path_swap_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp).resolve()
            )
            descriptor = os.open(model, os.O_RDONLY)
            original = os.fstat(descriptor)
            os.lseek(descriptor, 7, os.SEEK_SET)
            moved = model.with_name("opened-original.gguf")
            model.rename(moved)
            model.write_bytes(b"replacement")
            try:
                seen: list[int] = []

                def advise(fd: int, offset: int, length: int) -> None:
                    seen.append(fd)

                result = TOOL.cold_prepare_descriptor_from_plan(
                    descriptor,
                    plan_path,
                    plan_sha256,
                    advise=advise,
                    sample_residency=lambda fd, size, page: bytes(
                        (size + page - 1) // page
                    ),
                )
                self.assertTrue(seen)
                self.assertEqual(set(seen), {descriptor})
                self.assertEqual(os.lseek(descriptor, 0, os.SEEK_CUR), 7)
                self.assertEqual(os.fstat(descriptor).st_ino, original.st_ino)
                self.assertEqual(
                    result["model_identity"]["inode"], str(original.st_ino)
                )

                def failing_advice(fd: int, offset: int, length: int) -> None:
                    os.lseek(fd, 3, os.SEEK_SET)
                    raise OSError(errno.EIO, "injected")

                with self.assertRaisesRegex(ValueError, "advice failed"):
                    TOOL.cold_prepare_descriptor_from_plan(
                        descriptor,
                        plan_path,
                        plan_sha256,
                        advise=failing_advice,
                        sample_residency=lambda fd, size, page: bytes(
                            (size + page - 1) // page
                        ),
                    )
                self.assertEqual(os.lseek(descriptor, 0, os.SEEK_CUR), 7)
                self.assertEqual(os.fstat(descriptor).st_ino, original.st_ino)
            finally:
                os.close(descriptor)

    def test_cold_preparation_is_descriptor_bound_and_reports_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model, plan_path, plan, plan_sha256 = _write_cold_preparation_fixture(
                directory
            )
            replacement = directory / "replacement.gguf"
            replacement.write_bytes(b"replacement pathname")
            advised: list[tuple[int, int, int]] = []
            sampled_descriptors: list[int] = []

            def advise(descriptor: int, offset: int, length: int) -> None:
                self.assertEqual(os.pread(descriptor, 8, TEST_PAGE_SIZE), b"original")
                advised.append((descriptor, offset, length))
                if len(advised) == 1:
                    os.replace(replacement, model)

            def sample_residency(
                descriptor: int, file_size: int, page_size: int
            ) -> int:
                sampled_descriptors.append(descriptor)
                self.assertEqual(
                    (file_size, page_size), (8 * TEST_PAGE_SIZE, TEST_PAGE_SIZE)
                )
                self.assertEqual(os.pread(descriptor, 8, TEST_PAGE_SIZE), b"original")
                return 0

            result = TOOL.cold_prepare_from_plan(
                model,
                plan_path,
                plan_sha256,
                advise=advise,
                sample_residency=sample_residency,
            )

            self.assertEqual(
                [(offset, length) for _, offset, length in advised],
                [
                    (TEST_PAGE_SIZE, 2 * TEST_PAGE_SIZE),
                    (4 * TEST_PAGE_SIZE, TEST_PAGE_SIZE),
                ],
            )
            self.assertEqual(len({descriptor for descriptor, _, _ in advised}), 1)
            self.assertEqual(sampled_descriptors, [advised[0][0]])
            self.assertEqual(result["plan_sha256"], plan_sha256)
            self.assertEqual(result["ledger_sha256"], plan["ledger_sha256"])
            self.assertEqual(result["model_identity"], plan["model"])
            self.assertEqual(result["page_size"], str(TEST_PAGE_SIZE))
            self.assertEqual(
                {
                    key: result[key]
                    for key in (
                        "eligible_calls",
                        "attempted_calls",
                        "successful_calls",
                        "failed_calls",
                    )
                },
                {
                    "eligible_calls": 2,
                    "attempted_calls": 2,
                    "successful_calls": 2,
                    "failed_calls": 0,
                },
            )
            self.assertEqual(
                {
                    key: result[key]
                    for key in (
                        "eligible_bytes",
                        "attempted_bytes",
                        "successful_bytes",
                        "failed_bytes",
                    )
                },
                {
                    "eligible_bytes": str(3 * TEST_PAGE_SIZE),
                    "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                    "successful_bytes": str(3 * TEST_PAGE_SIZE),
                    "failed_bytes": "0",
                },
            )
            self.assertEqual(result["errno_buckets"], {})
            self.assertEqual(result["resident_bytes_after"], "0")
            self.assertEqual(
                result["unavoidable_bytes"], str(5 * TEST_PAGE_SIZE)
            )

    def test_plan_and_ledger_digests_and_opened_identity_are_independent_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, plan, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp)
            )
            no_advice = lambda descriptor, offset, length: None
            no_residency = lambda descriptor, file_size, page_size: 0

            with self.subTest("outer plan digest"):
                with self.assertRaisesRegex(ValueError, "plan.*digest|SHA-256"):
                    TOOL.cold_prepare_from_plan(
                        model,
                        plan_path,
                        "f" * 64,
                        advise=no_advice,
                        sample_residency=no_residency,
                    )

            with self.subTest("embedded ledger digest"):
                changed = copy.deepcopy(plan)
                changed["ledger_sha256"] = "e" * 64
                changed_bytes = TOOL.canonical_json_bytes(changed)
                plan_path.write_bytes(changed_bytes)
                with self.assertRaisesRegex(ValueError, "ledger.*digest|ledger.*SHA-256"):
                    TOOL.cold_prepare_from_plan(
                        model,
                        plan_path,
                        hashlib.sha256(changed_bytes).hexdigest(),
                        advise=no_advice,
                        sample_residency=no_residency,
                    )

            with self.subTest("opened model identity"):
                changed = copy.deepcopy(plan)
                changed["model"]["inode"] = str(int(changed["model"]["inode"]) + 1)
                changed_bytes = TOOL.canonical_json_bytes(changed)
                plan_path.write_bytes(changed_bytes)
                with self.assertRaisesRegex(ValueError, "model.*identity|inode"):
                    TOOL.cold_prepare_from_plan(
                        model,
                        plan_path,
                        hashlib.sha256(changed_bytes).hexdigest(),
                        advise=no_advice,
                        sample_residency=no_residency,
                    )

            self.assertRegex(plan_sha256, r"^[0-9a-f]{64}$")

    def test_plan_and_model_paths_are_opened_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                directory
            )
            model_link = directory / "model-link.gguf"
            plan_link = directory / "plan-link.json"
            model_link.symlink_to(model)
            plan_link.symlink_to(plan_path)
            kwargs = {
                "advise": lambda descriptor, offset, length: None,
                "sample_residency": lambda descriptor, file_size, page_size: 0,
            }
            for label, model_arg, plan_arg in (
                ("model", model_link, plan_path),
                ("plan", model, plan_link),
            ):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "symlink|follow"):
                        TOOL.cold_prepare_from_plan(
                            model_arg, plan_arg, plan_sha256, **kwargs
                        )

    def test_parent_directory_symlinks_are_not_traversed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            real = directory / "real"
            real.mkdir()
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(real)
            via_symlink = directory / "via-symlink"
            via_symlink.symlink_to(real, target_is_directory=True)
            kwargs = {
                "advise": lambda descriptor, offset, length: None,
                "sample_residency": lambda descriptor, file_size, page_size: 0,
            }
            for label, model_arg, plan_arg in (
                (
                    "model parent",
                    via_symlink / model.name,
                    plan_path,
                ),
                (
                    "plan parent",
                    model,
                    via_symlink / plan_path.name,
                ),
            ):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        ValueError, "symlink|without following|traversal"
                    ):
                        TOOL.cold_prepare_from_plan(
                            model_arg, plan_arg, plan_sha256, **kwargs
                        )

    def test_host_page_size_mismatch_fails_before_any_advice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp)
            )
            advice_calls: list[tuple[int, int]] = []
            observed_error: ValueError | None = None
            with mock.patch.object(TOOL.mmap, "PAGESIZE", 2 * TEST_PAGE_SIZE):
                try:
                    TOOL.cold_prepare_from_plan(
                        model,
                        plan_path,
                        plan_sha256,
                        advise=lambda descriptor, offset, length: advice_calls.append(
                            (offset, length)
                        ),
                        sample_residency=lambda descriptor, file_size, page_size: 0,
                    )
                except ValueError as exc:
                    observed_error = exc

            self.assertEqual(
                advice_calls,
                [],
                "host page-size validation must precede the first cache advice",
            )
            self.assertIsNotNone(observed_error)
            self.assertRegex(str(observed_error), "host page|page size")

    def test_plan_ranges_must_be_a_complete_normalized_safe_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, plan, _ = _write_cold_preparation_fixture(Path(tmp))
            cases = {
                "missing coverage": lambda value: value["page_cache"].__setitem__(
                    "eligible_unique_bytes", str(4 * TEST_PAGE_SIZE)
                ),
                "duplicate coverage": lambda value: value["page_cache"]["ranges"].append(
                    copy.deepcopy(value["page_cache"]["ranges"][0])
                ),
                "unaligned range": lambda value: value["page_cache"]["ranges"][0].__setitem__(
                    "offset", str(TEST_PAGE_SIZE + 1)
                ),
            }
            for label, mutate in cases.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(plan)
                    mutate(changed)
                    raw = TOOL.canonical_json_bytes(changed)
                    plan_path.write_bytes(raw)
                    with self.assertRaisesRegex(
                        ValueError, "coverage|normalized|page|range|eligible"
                    ):
                        TOOL.cold_prepare_from_plan(
                            model,
                            plan_path,
                            hashlib.sha256(raw).hexdigest(),
                            advise=lambda descriptor, offset, length: None,
                            sample_residency=lambda descriptor, file_size, page_size: 0,
                        )

    def test_digest_consistent_page_cache_cannot_claim_pages_outside_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, plan, _ = _write_cold_preparation_fixture(Path(tmp))
            changed = copy.deepcopy(plan)
            changed["page_cache"]["ranges"][1] = {
                "bytes": str(TEST_PAGE_SIZE),
                "offset": str(6 * TEST_PAGE_SIZE),
            }
            raw = TOOL.canonical_json_bytes(changed)
            plan_path.write_bytes(raw)
            with self.assertRaisesRegex(
                ValueError, "page.*ledger|ledger.*page|safe.*tensor"
            ):
                TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    hashlib.sha256(raw).hexdigest(),
                    advise=lambda descriptor, offset, length: None,
                    sample_residency=lambda descriptor, file_size, page_size: 0,
                )

    def test_derived_unavoidable_residency_above_two_gib_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, plan, _ = _write_cold_preparation_fixture(Path(tmp))
            model_size = (2 << 30) + (3 * TEST_PAGE_SIZE)
            with model.open("r+b") as handle:
                handle.truncate(model_size)
            identity = model.stat()
            changed = copy.deepcopy(plan)
            changed["model"].update(
                {
                    "device": str(identity.st_dev),
                    "inode": str(identity.st_ino),
                    "mtime_ns": str(identity.st_mtime_ns),
                    "size_bytes": str(identity.st_size),
                }
            )
            changed["ledger"]["file_size"] = str(identity.st_size)
            changed["ledger_sha256"] = hashlib.sha256(
                TOOL.canonical_json_bytes(changed["ledger"])
            ).hexdigest()
            changed["page_cache"].update(
                {
                    "eligible_unique_bytes": str(TEST_PAGE_SIZE),
                    "mapped_page_bytes": str(identity.st_size),
                    "ranges": [
                        {
                            "bytes": str(TEST_PAGE_SIZE),
                            "offset": str(TEST_PAGE_SIZE),
                        }
                    ],
                    "unavoidable_bytes": str(identity.st_size - TEST_PAGE_SIZE),
                }
            )
            raw = TOOL.canonical_json_bytes(changed)
            plan_path.write_bytes(raw)
            with self.assertRaisesRegex(
                ValueError, "unavoidable.*(?:2 GiB|bound|limit)"
            ):
                TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    hashlib.sha256(raw).hexdigest(),
                    advise=lambda descriptor, offset, length: None,
                    sample_residency=lambda descriptor, file_size, page_size: 0,
                )

    def test_advice_failure_counts_every_range_and_errno_without_short_circuiting(self) -> None:
        ranges = [
            {"bytes": str(2 * TEST_PAGE_SIZE), "offset": str(TEST_PAGE_SIZE)},
            {"bytes": str(TEST_PAGE_SIZE), "offset": str(4 * TEST_PAGE_SIZE)},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            model, _, _, _ = _write_cold_preparation_fixture(Path(tmp))
            descriptor = os.open(model, os.O_RDONLY)
            calls: list[tuple[int, int]] = []

            def advise(_descriptor: int, offset: int, length: int) -> None:
                calls.append((offset, length))
                if offset == TEST_PAGE_SIZE:
                    raise OSError(errno.EIO, os.strerror(errno.EIO))

            try:
                report = TOOL.advise_safe_page_ranges(
                    descriptor,
                    ranges,
                    page_size=TEST_PAGE_SIZE,
                    model_size=8 * TEST_PAGE_SIZE,
                    advise=advise,
                )
            finally:
                os.close(descriptor)

            self.assertEqual(
                calls,
                [
                    (TEST_PAGE_SIZE, 2 * TEST_PAGE_SIZE),
                    (4 * TEST_PAGE_SIZE, TEST_PAGE_SIZE),
                ],
            )
            self.assertEqual(
                report,
                {
                    "eligible_calls": 2,
                    "eligible_bytes": str(3 * TEST_PAGE_SIZE),
                    "attempted_calls": 2,
                    "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                    "successful_calls": 1,
                    "successful_bytes": str(TEST_PAGE_SIZE),
                    "failed_calls": 1,
                    "failed_bytes": str(2 * TEST_PAGE_SIZE),
                    "errno_buckets": {"EIO": 1},
                },
            )

    def test_residual_disposal_touches_only_hot_eligible_contiguous_runs(self) -> None:
        page_size = TEST_PAGE_SIZE
        descriptor = 41
        resident = bytes((1, 1, 1, 1, 1, 0, 1, 0))
        eligible = (
            (page_size, 2 * page_size),
            (4 * page_size, 2 * page_size),
        )
        events: list[tuple[object, ...]] = []

        class FakeMapping:
            def __getitem__(self, offset: int) -> int:
                events.append(("touch", offset))
                return 0

            def madvise(self, advice: int, offset: int, length: int) -> None:
                events.append(("madvise", advice, offset, length))

            def close(self) -> None:
                events.append(("close",))

        def make_mapping(
            fd: int,
            length: int,
            *,
            flags: int,
            prot: int,
        ) -> FakeMapping:
            events.append(("mmap", fd, length, flags, prot))
            return FakeMapping()

        def advise(fd: int, offset: int, length: int) -> None:
            events.append(("fadvise", fd, offset, length))

        random_advice = 4141
        advice = 4242
        with mock.patch.object(
            TOOL.mmap, "MADV_DONTNEED", advice, create=True
        ), mock.patch.object(
            TOOL.mmap, "MADV_RANDOM", random_advice, create=True
        ), mock.patch.object(TOOL.mmap, "mmap", side_effect=make_mapping):
            report = TOOL._dispose_resident_eligible_pages(
                descriptor,
                8 * page_size,
                page_size,
                resident,
                eligible,
                advise=advise,
            )

        self.assertEqual(
            events,
            [
                (
                    "mmap",
                    descriptor,
                    8 * page_size,
                    TOOL.mmap.MAP_PRIVATE,
                    TOOL.mmap.PROT_READ,
                ),
                ("madvise", random_advice, 0, 8 * page_size),
                ("touch", page_size),
                ("touch", 2 * page_size),
                ("madvise", advice, page_size, 2 * page_size),
                ("fadvise", descriptor, page_size, 2 * page_size),
                ("touch", 4 * page_size),
                ("madvise", advice, 4 * page_size, page_size),
                ("fadvise", descriptor, 4 * page_size, page_size),
                ("close",),
            ],
        )
        self.assertEqual(
            report,
            {
                "random_access_madvise": {
                    "attempted_calls": 1,
                    "attempted_bytes": str(8 * page_size),
                    "successful_calls": 1,
                    "successful_bytes": str(8 * page_size),
                    "failed_calls": 0,
                    "failed_bytes": "0",
                    "errno_buckets": {},
                },
                "mapping_touch_pages": 3,
                "mapping_touch_bytes": str(3 * page_size),
                "madvise": {
                    "attempted_calls": 2,
                    "attempted_bytes": str(3 * page_size),
                    "successful_calls": 2,
                    "successful_bytes": str(3 * page_size),
                    "failed_calls": 0,
                    "failed_bytes": "0",
                    "errno_buckets": {},
                },
                "fadvise": {
                    "attempted_calls": 2,
                    "attempted_bytes": str(3 * page_size),
                    "successful_calls": 2,
                    "successful_bytes": str(3 * page_size),
                    "failed_calls": 0,
                    "failed_bytes": "0",
                    "errno_buckets": {},
                },
            },
        )

    def test_residual_disposal_attempts_and_accounts_both_calls_on_failures(self) -> None:
        page_size = TEST_PAGE_SIZE
        descriptor = 42
        resident = bytes((0, 1, 1, 0, 1, 0, 0, 0))
        eligible = (
            (page_size, 2 * page_size),
            (4 * page_size, page_size),
        )
        events: list[tuple[object, ...]] = []

        class FakeMapping:
            def __getitem__(self, offset: int) -> int:
                events.append(("touch", offset))
                return 0

            def madvise(self, advice: int, offset: int, length: int) -> None:
                events.append(("madvise", advice, offset, length))
                if advice == 4242 and offset == page_size:
                    raise OSError(errno.EIO, "injected madvise failure")

            def close(self) -> None:
                events.append(("close",))

        def advise(fd: int, offset: int, length: int) -> None:
            events.append(("fadvise", offset, length))
            if offset == 4 * page_size:
                raise OSError(errno.ENOSPC, "injected fadvise failure")

        with mock.patch.object(
            TOOL.mmap, "MADV_DONTNEED", 4242, create=True
        ), mock.patch.object(
            TOOL.mmap, "MADV_RANDOM", 4141, create=True
        ), mock.patch.object(TOOL.mmap, "mmap", return_value=FakeMapping()):
            report = TOOL._dispose_resident_eligible_pages(
                descriptor,
                8 * page_size,
                page_size,
                resident,
                eligible,
                advise=advise,
            )

        self.assertEqual(
            [event for event in events if event[0] in ("madvise", "fadvise")],
            [
                ("madvise", 4141, 0, 8 * page_size),
                ("madvise", 4242, page_size, 2 * page_size),
                ("fadvise", page_size, 2 * page_size),
                ("madvise", 4242, 4 * page_size, page_size),
                ("fadvise", 4 * page_size, page_size),
            ],
        )
        self.assertEqual(events[-1], ("close",))
        self.assertEqual(
            report["madvise"],
            {
                "attempted_calls": 2,
                "attempted_bytes": str(3 * page_size),
                "successful_calls": 1,
                "successful_bytes": str(page_size),
                "failed_calls": 1,
                "failed_bytes": str(2 * page_size),
                "errno_buckets": {"EIO": 1},
            },
        )
        self.assertEqual(
            report["fadvise"],
            {
                "attempted_calls": 2,
                "attempted_bytes": str(3 * page_size),
                "successful_calls": 1,
                "successful_bytes": str(2 * page_size),
                "failed_calls": 1,
                "failed_bytes": str(page_size),
                "errno_buckets": {"ENOSPC": 1},
            },
        )

    def test_residual_disposal_is_a_noop_when_no_eligible_page_is_hot(self) -> None:
        page_size = TEST_PAGE_SIZE
        events: list[tuple[object, ...]] = []

        class FakeMapping:
            def __getitem__(self, offset: int) -> int:
                events.append(("unexpected touch", offset))
                return 0

            def madvise(self, advice: int, offset: int, length: int) -> None:
                events.append(("unexpected madvise", offset, length))

            def close(self) -> None:
                events.append(("close",))

        def advise(fd: int, offset: int, length: int) -> None:
            events.append(("unexpected fadvise", offset, length))

        with mock.patch.object(
            TOOL.mmap, "MADV_DONTNEED", 4242, create=True
        ), mock.patch.object(
            TOOL.mmap, "MADV_RANDOM", 4141, create=True
        ), mock.patch.object(TOOL.mmap, "mmap", return_value=FakeMapping()):
            report = TOOL._dispose_resident_eligible_pages(
                43,
                8 * page_size,
                page_size,
                bytes((1, 0, 0, 1, 0, 0, 1, 0)),
                ((page_size, 2 * page_size), (4 * page_size, 2 * page_size)),
                advise=advise,
            )

        self.assertEqual(events, [("unexpected madvise", 0, 8 * page_size),
                                  ("close",)])
        self.assertEqual(report["mapping_touch_pages"], 0)
        self.assertEqual(report["mapping_touch_bytes"], "0")
        for label in ("madvise", "fadvise"):
            self.assertEqual(report[label]["attempted_calls"], 0)
            self.assertEqual(report[label]["attempted_bytes"], "0")

    def test_final_hot_resample_is_rejected_after_residual_disposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp).resolve()
            )
            descriptor = os.open(model, os.O_RDONLY)
            page_count = os.fstat(descriptor).st_size // TEST_PAGE_SIZE
            hot = bytearray(page_count)
            hot[1] = 1
            samples = iter((bytes(hot), bytes(hot), bytes(hot)))

            class FakeMapping:
                def __getitem__(self, offset: int) -> int:
                    return 0

                def madvise(self, advice: int, offset: int, length: int) -> None:
                    return None

                def close(self) -> None:
                    return None

            try:
                with mock.patch.object(
                    TOOL.mmap, "MADV_DONTNEED", 4242, create=True
                ), mock.patch.object(
                    TOOL.mmap, "MADV_RANDOM", 4141, create=True
                ), mock.patch.object(
                    TOOL.mmap, "mmap", return_value=FakeMapping()
                ), mock.patch.object(TOOL.sys, "platform", "linux"):
                    with self.assertRaisesRegex(
                        ValueError, "eligible.*resident|resident.*eligible"
                    ):
                        TOOL.cold_prepare_descriptor_from_plan(
                            descriptor,
                            plan_path,
                            plan_sha256,
                            advise=lambda fd, offset, length: None,
                            sample_residency=lambda fd, size, page: next(samples),
                        )
                with self.assertRaises(StopIteration):
                    next(samples)
                os.fstat(descriptor)
            finally:
                os.close(descriptor)

    def test_residual_disposal_uses_one_bounded_pageout_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp).resolve()
            )
            page_count = model.stat().st_size // TEST_PAGE_SIZE
            initial = bytearray(page_count)
            initial[1:4] = b"\1\1\1"
            after_dontneed = bytearray(page_count)
            after_dontneed[2] = 1
            samples = iter(
                (bytes(initial), bytes(after_dontneed), bytes(page_count))
            )
            advice_kinds: list[int] = []

            class FakeMapping:
                def __getitem__(self, offset: int) -> int:
                    return 0

                def madvise(self, advice: int, offset: int, length: int) -> None:
                    if advice != 4141:
                        advice_kinds.append(advice)

                def close(self) -> None:
                    return None

            with mock.patch.object(
                TOOL.mmap, "MADV_RANDOM", 4141, create=True
            ), mock.patch.object(
                TOOL.mmap, "MADV_DONTNEED", 4242, create=True
            ), mock.patch.object(
                TOOL.mmap, "mmap", return_value=FakeMapping()
            ), mock.patch.object(TOOL.sys, "platform", "linux"):
                result = TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    plan_sha256,
                    advise=lambda fd, offset, length: None,
                    sample_residency=lambda fd, size, page: next(samples),
                )

            self.assertEqual(advice_kinds, [4242, TOOL.LINUX_MADV_PAGEOUT])
            self.assertEqual(result["resident_bytes_after"], "0")
            self.assertEqual(
                result["residual_disposal"]["pageout_retry_pages"], 1
            )
            self.assertEqual(result["residual_disposal"]["residency_samples"], 3)
            self.assertEqual(result["residual_disposal"]["mapping_touch_pages"], 3)
            self.assertEqual(
                result["residual_disposal"]["random_access_madvise"],
                {
                    "attempted_calls": 2,
                    "attempted_bytes": str(2 * model.stat().st_size),
                    "successful_calls": 2,
                    "successful_bytes": str(2 * model.stat().st_size),
                    "failed_calls": 0,
                    "failed_bytes": "0",
                    "errno_buckets": {},
                },
            )

    def test_pageout_evidence_requires_two_disposal_runs(self) -> None:
        cold = _qualification_cold_preparation()
        page_size = int(cold["page_size"])
        residual = cold["residual_disposal"]
        residual.update(
            {
                "initial_resident_eligible_pages": 2,
                "pageout_retry_pages": 1,
                "residency_samples": 3,
                "mapping_touch_pages": 3,
                "mapping_touch_bytes": str(3 * page_size),
            }
        )
        model_size = int(cold["model_identity"]["size_bytes"])
        residual["random_access_madvise"].update(
            {
                "attempted_calls": 2,
                "attempted_bytes": str(2 * model_size),
                "successful_calls": 2,
                "successful_bytes": str(2 * model_size),
            }
        )
        for label in ("madvise", "fadvise"):
            residual[label].update(
                {
                    "attempted_calls": 1,
                    "attempted_bytes": str(3 * page_size),
                    "successful_calls": 1,
                    "successful_bytes": str(3 * page_size),
                }
            )
        with self.assertRaisesRegex(ValueError, "residual disposal"):
            TOOL._validate_cold_preparation(cold)

    def test_location_aware_residency_rejects_only_hot_eligible_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp)
            )
            page_size = TEST_PAGE_SIZE
            page_count = 8
            safe_page = bytearray(page_count)
            safe_page[1] = 1
            unavoidable_page = bytearray(page_count)
            unavoidable_page[0] = 1
            kwargs = {
                "advise": lambda descriptor, offset, length: None,
            }

            with self.assertRaisesRegex(
                ValueError, "eligible.*resident|resident.*eligible"
            ):
                TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    plan_sha256,
                    sample_residency=lambda descriptor, file_size, observed_page_size: bytes(
                        safe_page
                    ),
                    **kwargs,
                )

            result = TOOL.cold_prepare_from_plan(
                model,
                plan_path,
                plan_sha256,
                sample_residency=lambda descriptor, file_size, observed_page_size: bytes(
                    unavoidable_page
                ),
                **kwargs,
            )
            self.assertEqual(result["resident_bytes_after"], str(page_size))
            self.assertLessEqual(
                int(result["resident_bytes_after"]),
                int(result["unavoidable_bytes"]),
            )

    def test_active_eligible_pages_are_disposed_and_resampled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp)
            )
            hot = bytearray(8)
            hot[1] = 1
            hot[2] = 1
            hot[4] = 1
            samples = iter((bytes(hot), bytes(8)))
            disposal_calls: list[tuple[int, int, int, bytes, object]] = []

            def dispose(
                descriptor: int,
                file_size: int,
                page_size: int,
                resident_bits: bytes,
                eligible_ranges: object,
                **kwargs: object,
            ) -> dict[str, object]:
                disposal_calls.append(
                    (
                        descriptor,
                        file_size,
                        page_size,
                        resident_bits,
                        eligible_ranges,
                    )
                )
                return {
                    "random_access_madvise": {
                        "attempted_calls": 1,
                        "attempted_bytes": str(file_size),
                        "successful_calls": 1,
                        "successful_bytes": str(file_size),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                    "mapping_touch_pages": 3,
                    "mapping_touch_bytes": str(3 * TEST_PAGE_SIZE),
                    "madvise": {
                        "attempted_calls": 2,
                        "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                        "successful_calls": 2,
                        "successful_bytes": str(3 * TEST_PAGE_SIZE),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                    "fadvise": {
                        "attempted_calls": 2,
                        "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                        "successful_calls": 2,
                        "successful_bytes": str(3 * TEST_PAGE_SIZE),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                }

            with mock.patch.object(
                TOOL,
                "_dispose_resident_eligible_pages",
                side_effect=dispose,
                create=True,
            ):
                result = TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    plan_sha256,
                    advise=lambda descriptor, offset, length: None,
                    sample_residency=lambda descriptor, file_size, page_size: next(
                        samples
                    ),
                )

            self.assertEqual(len(disposal_calls), 1)
            self.assertEqual(disposal_calls[0][1:4], (
                8 * TEST_PAGE_SIZE,
                TEST_PAGE_SIZE,
                bytes(hot),
            ))
            self.assertEqual(result["resident_bytes_after"], "0")
            self.assertEqual(
                result["residual_disposal"],
                {
                    "initial_resident_eligible_pages": 3,
                    "pageout_retry_pages": 0,
                    "residency_samples": 2,
                    "mapping_touch_pages": 3,
                    "mapping_touch_bytes": str(3 * TEST_PAGE_SIZE),
                    "random_access_madvise": {
                        "attempted_calls": 1,
                        "attempted_bytes": str(8 * TEST_PAGE_SIZE),
                        "successful_calls": 1,
                        "successful_bytes": str(8 * TEST_PAGE_SIZE),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                    "madvise": {
                        "attempted_calls": 2,
                        "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                        "successful_calls": 2,
                        "successful_bytes": str(3 * TEST_PAGE_SIZE),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                    "fadvise": {
                        "attempted_calls": 2,
                        "attempted_bytes": str(3 * TEST_PAGE_SIZE),
                        "successful_calls": 2,
                        "successful_bytes": str(3 * TEST_PAGE_SIZE),
                        "failed_calls": 0,
                        "failed_bytes": "0",
                        "errno_buckets": {},
                    },
                },
            )

    def test_descriptor_identity_change_during_measurement_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model, plan_path, _, plan_sha256 = _write_cold_preparation_fixture(
                Path(tmp)
            )

            def mutate_open_inode(_descriptor: int, _offset: int, _length: int) -> None:
                with model.open("r+b") as writable:
                    writable.truncate(7 * TEST_PAGE_SIZE)

            with self.assertRaisesRegex(ValueError, "identity.*changed|size.*changed"):
                TOOL.cold_prepare_from_plan(
                    model,
                    plan_path,
                    plan_sha256,
                    advise=mutate_open_inode,
                    sample_residency=lambda descriptor, file_size, page_size: 0,
                )


class NvmlCheckpointContractTest(unittest.TestCase):
    DS4_PID = 4242
    OTHER_PID = 9001

    def test_stable_inventory_returns_only_the_ds4_process_usage(self) -> None:
        other = {"pid": self.OTHER_PID, "used_gpu_memory_bytes": str(1 << 30)}
        ds4 = {"pid": self.DS4_PID, "used_gpu_memory_bytes": str(6 << 30)}
        frozen = _nvml_inventory([other])
        before = _nvml_inventory([other, ds4])
        after = _nvml_inventory([copy.deepcopy(other), copy.deepcopy(ds4)])
        result = TOOL.validate_nvml_checkpoint(
            frozen,
            before,
            after,
            ds4_pid=self.DS4_PID,
            gpu_uuid=HOST_IDENTITY["gpu_uuid"],
        )
        self.assertEqual(result["api"], NVML_COMPUTE_API)
        self.assertEqual(result["gpu_uuid"], HOST_IDENTITY["gpu_uuid"])
        self.assertEqual(result["ds4_pid"], self.DS4_PID)
        self.assertEqual(result["ds4_process_bytes"], str(6 << 30))

    def test_unrelated_process_change_is_invalid_even_when_checkpoint_is_stable(self) -> None:
        ds4 = {"pid": self.DS4_PID, "used_gpu_memory_bytes": str(6 << 30)}
        cases = {
            "changed since frozen baseline": (
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str(1 << 30)}],
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str((1 << 30) + 4096)}],
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str((1 << 30) + 4096)}],
            ),
            "appeared": ([], [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": "1"}], [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": "1"}]),
            "disappeared": ([{"pid": self.OTHER_PID, "used_gpu_memory_bytes": "1"}], [], []),
            "changed inside checkpoint": (
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str(1 << 30)}],
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str(1 << 30)}],
                [{"pid": self.OTHER_PID, "used_gpu_memory_bytes": str((1 << 30) + 4096)}],
            ),
        }
        for label, (frozen_other, before_other, after_other) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, "unrelated.*process|inventory.*changed"
                ):
                    TOOL.validate_nvml_checkpoint(
                        _nvml_inventory(frozen_other),
                        _nvml_inventory(before_other + [copy.deepcopy(ds4)]),
                        _nvml_inventory(after_other + [copy.deepcopy(ds4)]),
                        ds4_pid=self.DS4_PID,
                        gpu_uuid=HOST_IDENTITY["gpu_uuid"],
                    )

    def test_missing_or_unknown_process_usage_is_never_coerced_to_zero(self) -> None:
        other = {"pid": self.OTHER_PID, "used_gpu_memory_bytes": str(1 << 30)}
        cases = {
            "missing process": [],
            "missing usage": [{"pid": self.DS4_PID}],
            "null usage": [{"pid": self.DS4_PID, "used_gpu_memory_bytes": None}],
            "NVML unknown sentinel": [
                {"pid": self.DS4_PID, "used_gpu_memory_bytes": str((1 << 64) - 1)}
            ],
        }
        for label, ds4_processes in cases.items():
            with self.subTest(label=label):
                checkpoint = _nvml_inventory([copy.deepcopy(other)] + ds4_processes)
                with self.assertRaisesRegex(ValueError, "missing|unknown|usage"):
                    TOOL.validate_nvml_checkpoint(
                        _nvml_inventory([copy.deepcopy(other)]),
                        checkpoint,
                        copy.deepcopy(checkpoint),
                        ds4_pid=self.DS4_PID,
                        gpu_uuid=HOST_IDENTITY["gpu_uuid"],
                    )

    def test_nvml_api_and_gpu_uuid_must_match_the_frozen_target(self) -> None:
        ds4 = {"pid": self.DS4_PID, "used_gpu_memory_bytes": str(6 << 30)}
        for label, changed in (
            ("API", _nvml_inventory([ds4], api="nvmlDeviceGetComputeRunningProcesses_v3")),
            (
                "UUID",
                _nvml_inventory(
                    [ds4],
                    gpu_uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
                ),
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "API|version|UUID|GPU"):
                    TOOL.validate_nvml_checkpoint(
                        _nvml_inventory([]),
                        changed,
                        copy.deepcopy(changed),
                        ds4_pid=self.DS4_PID,
                        gpu_uuid=HOST_IDENTITY["gpu_uuid"],
                    )

    def test_schema_defines_closed_cold_preparation_and_nvml_inventory_records(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cold = schema["$defs"]["cold_preparation"]
        self.assertIs(cold["additionalProperties"], False)
        self.assertEqual(
            set(cold["required"]),
            {
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
            },
        )
        inventory = schema["$defs"]["nvml_process_inventory"]
        self.assertIs(inventory["additionalProperties"], False)
        self.assertEqual(
            set(inventory["required"]), {"api", "gpu_uuid", "processes"}
        )
        self.assertEqual(inventory["properties"]["api"]["const"], NVML_COMPUTE_API)
        process = schema["$defs"]["nvml_process"]
        self.assertIs(process["additionalProperties"], False)
        self.assertEqual(
            set(process["required"]), {"pid", "used_gpu_memory_bytes"}
        )

    def test_schema_and_validator_share_one_nvml_byte_representation(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = Draft202012Validator(
            {
                "$schema": schema["$schema"],
                "$ref": "#/$defs/nvml_process",
                "$defs": schema["$defs"],
            }
        )
        cases = [
            {"pid": self.DS4_PID, "used_gpu_memory_bytes": "6442450944"},
            {"pid": self.DS4_PID, "used_gpu_memory_bytes": 6 << 30},
        ]
        shared_acceptance = 0
        for process in cases:
            schema_accepts = validator.is_valid(process)
            other_usage = "1073741824" if isinstance(
                process["used_gpu_memory_bytes"], str
            ) else 1 << 30
            other = {
                "pid": self.OTHER_PID,
                "used_gpu_memory_bytes": other_usage,
            }
            checkpoint = _nvml_inventory([other, process])
            try:
                TOOL.validate_nvml_checkpoint(
                    _nvml_inventory([other]),
                    checkpoint,
                    copy.deepcopy(checkpoint),
                    ds4_pid=self.DS4_PID,
                    gpu_uuid=HOST_IDENTITY["gpu_uuid"],
                )
            except ValueError:
                validator_accepts = False
            else:
                validator_accepts = True
            with self.subTest(value=process["used_gpu_memory_bytes"]):
                self.assertEqual(schema_accepts, validator_accepts)
            shared_acceptance += int(schema_accepts and validator_accepts)
        self.assertEqual(
            shared_acceptance,
            1,
            "exactly one canonical NVML byte representation must be accepted",
        )

    def test_nvml_unknown_usage_sentinel_is_schema_invalid(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = Draft202012Validator(
            {
                "$schema": schema["$schema"],
                "$ref": "#/$defs/nvml_process",
                "$defs": schema["$defs"],
            }
        )
        candidates = [
            {"pid": self.DS4_PID, "used_gpu_memory_bytes": str((1 << 64) - 1)},
            {"pid": self.DS4_PID, "used_gpu_memory_bytes": (1 << 64) - 1},
        ]
        for process in candidates:
            with self.subTest(value=process["used_gpu_memory_bytes"]):
                self.assertFalse(validator.is_valid(process))


class NvmlPreChildCollectorContractTest(unittest.TestCase):
    def test_collector_uses_only_nvml_v2_and_returns_every_process_sorted(self) -> None:
        calls: list[object] = []
        observed_device_uuid = [HOST_IDENTITY["gpu_uuid"]]

        class FakeFunction:
            def __init__(self, name: str, callback: Callable[..., int]) -> None:
                self.name = name
                self.callback = callback
                self.argtypes: object = None
                self.restype: object = None

            def __call__(self, *args: object) -> int:
                calls.append(self.name)
                return self.callback(*args)

        def success(*args: object) -> int:
            return 0

        def version(buffer: object, capacity: object) -> int:
            self.assertGreaterEqual(capacity.value, len(NVML_LIBRARY_VERSION) + 1)
            buffer.value = NVML_LIBRARY_VERSION.encode("ascii")
            return 0

        def device_by_uuid(uuid: object, output: object) -> int:
            self.assertEqual(uuid, HOST_IDENTITY["gpu_uuid"].encode("ascii"))
            output._obj.value = 0x1234
            return 0

        def device_uuid(device: object, buffer: object, capacity: object) -> int:
            self.assertEqual(device.value, 0x1234)
            encoded = observed_device_uuid[0].encode("ascii")
            self.assertGreaterEqual(capacity.value, len(encoded) + 1)
            buffer.value = encoded
            return 0

        def processes(device: object, count: object, records: object) -> int:
            self.assertEqual(device.value, 0x1234)
            if records is None:
                count._obj.value = 2
                return 7
            count._obj.value = 2
            records[0].pid = 9001
            records[0].usedGpuMemory = 1 << 30
            records[1].pid = 123
            records[1].usedGpuMemory = 256 << 20
            return 0

        class FakeNvml:
            nvmlInit_v2 = FakeFunction("nvmlInit_v2", success)
            nvmlShutdown = FakeFunction("nvmlShutdown", success)
            nvmlSystemGetNVMLVersion = FakeFunction(
                "nvmlSystemGetNVMLVersion", version
            )
            nvmlDeviceGetHandleByUUID = FakeFunction(
                "nvmlDeviceGetHandleByUUID", device_by_uuid
            )
            nvmlDeviceGetUUID = FakeFunction("nvmlDeviceGetUUID", device_uuid)
            nvmlDeviceGetComputeRunningProcesses_v2 = FakeFunction(
                "nvmlDeviceGetComputeRunningProcesses_v2", processes
            )

        loaded: list[str] = []

        def load_nvml(name: str, *, use_errno: bool = False) -> FakeNvml:
            loaded.append(name)
            self.assertFalse(use_errno)
            return FakeNvml()

        with mock.patch.object(TOOL.ctypes, "CDLL", side_effect=load_nvml):
            result = TOOL.collect_nvml_pre_child(HOST_IDENTITY["gpu_uuid"])

        self.assertEqual(loaded, ["libnvidia-ml.so.1"])
        self.assertEqual(
            result,
            {
                "library_version": NVML_LIBRARY_VERSION,
                "inventory": _nvml_inventory(
                    [
                        {
                            "pid": 123,
                            "used_gpu_memory_bytes": str(256 << 20),
                        },
                        {
                            "pid": 9001,
                            "used_gpu_memory_bytes": str(1 << 30),
                        },
                    ]
                ),
            },
        )
        self.assertEqual(calls[0], "nvmlInit_v2")
        self.assertEqual(calls[-1], "nvmlShutdown")
        self.assertEqual(calls.count("nvmlDeviceGetUUID"), 1)
        self.assertEqual(
            calls.count("nvmlDeviceGetComputeRunningProcesses_v2"), 2
        )

        observed_device_uuid[0] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
        with mock.patch.object(TOOL.ctypes, "CDLL", side_effect=load_nvml):
            with self.assertRaisesRegex(ValueError, "UUID|device"):
                TOOL.collect_nvml_pre_child(HOST_IDENTITY["gpu_uuid"])
        self.assertEqual(calls[-1], "nvmlShutdown")


class QualificationPreflightManifestContractTest(unittest.TestCase):
    def freeze_preflight(
        self,
        *,
        captured_at_unix_ns: str = PRE_CHILD_CAPTURED_AT_UNIX_NS,
    ) -> tuple[dict[str, object], list[str]]:
        query_calls: list[str] = []
        inventory = _nvml_inventory(
            [
                {"pid": 9001, "used_gpu_memory_bytes": str(1 << 30)},
                {"pid": 123, "used_gpu_memory_bytes": str(256 << 20)},
            ]
        )

        def query_nvml_without_cuda(gpu_uuid: str) -> dict[str, object]:
            query_calls.append(gpu_uuid)
            return {
                "library_version": NVML_LIBRARY_VERSION,
                "inventory": copy.deepcopy(inventory),
            }

        preflight = TOOL.freeze_qualification_preflight(
            _qualification_cold_preparation(),
            gpu_uuid=HOST_IDENTITY["gpu_uuid"],
            runtime_identity=copy.deepcopy(RUNTIME_IDENTITY),
            captured_at_unix_ns=captured_at_unix_ns,
            nvml_query=query_nvml_without_cuda,
        )
        return preflight, query_calls

    def build_manifest_with_preflight(
        self,
        preflight: dict[str, object],
    ) -> dict:
        return TOOL.build_manifest(
            Path(MODEL_IDENTITY["path"]),
            token_counter=deterministic_token_count,
            host_identity=copy.deepcopy(HOST_IDENTITY),
            model_identity=copy.deepcopy(MODEL_IDENTITY),
            runtime_identity=copy.deepcopy(RUNTIME_IDENTITY),
            seed_bytes=SEED,
            qualification_preflight=copy.deepcopy(preflight),
        )

    def test_builder_freezes_exact_cold_and_pre_child_nvml_evidence(self) -> None:
        with mock.patch.object(
            TOOL,
            "_cuda_runtime_version",
            side_effect=AssertionError("pre-child NVML capture must not touch CUDA"),
        ) as cuda_probe:
            preflight, query_calls = self.freeze_preflight()

        self.assertEqual(query_calls, [HOST_IDENTITY["gpu_uuid"]])
        cuda_probe.assert_not_called()
        cold = _qualification_cold_preparation()
        expected_inventory = _nvml_inventory(
            [
                {"pid": 123, "used_gpu_memory_bytes": str(256 << 20)},
                {"pid": 9001, "used_gpu_memory_bytes": str(1 << 30)},
            ]
        )
        nvml_pre_child = {
            "capture_phase": "pre-child-before-cuda-initialization",
            "captured_at_unix_ns": PRE_CHILD_CAPTURED_AT_UNIX_NS,
            "library_version": NVML_LIBRARY_VERSION,
            "inventory": expected_inventory,
            "inventory_sha256": hashlib.sha256(
                TOOL.canonical_json_bytes(expected_inventory)
            ).hexdigest(),
        }
        expected = {
            "cold_preparation": cold,
            "cold_preparation_sha256": hashlib.sha256(
                TOOL.canonical_json_bytes(cold)
            ).hexdigest(),
            "nvml_pre_child": nvml_pre_child,
            "binding_sha256": hashlib.sha256(
                TOOL.canonical_json_bytes(
                    _qualification_binding_payload(cold, nvml_pre_child)
                )
            ).hexdigest(),
        }
        self.assertEqual(preflight, expected)
        self.assertEqual(
            preflight["cold_preparation"]["eligible_ranges"],
            cold["eligible_ranges"],
        )
        self.assertEqual(
            preflight["cold_preparation"]["model_identity"],
            {key: value for key, value in MODEL_IDENTITY.items() if key != "path"},
        )

        manifest = self.build_manifest_with_preflight(preflight)
        self.assertEqual(manifest["qualification_preflight"], expected)
        TOOL.validate_manifest(manifest)
        TOOL.verify_manifest_bindings(
            manifest,
            model_identity=copy.deepcopy(MODEL_IDENTITY),
            runtime_identity=copy.deepcopy(RUNTIME_IDENTITY),
            token_counter=deterministic_token_count,
        )

    def test_stale_inner_digests_and_runtime_binding_reject_substitution(self) -> None:
        preflight, _ = self.freeze_preflight()
        manifest = self.build_manifest_with_preflight(preflight)
        mutations = {
            "eligible range": lambda value: value["qualification_preflight"]
            ["cold_preparation"]["eligible_ranges"][0].__setitem__("offset", "4096"),
            "descriptor identity": lambda value: value["qualification_preflight"]
            ["cold_preparation"]["model_identity"].__setitem__("inode", "9002"),
            "inventory": lambda value: value["qualification_preflight"]
            ["nvml_pre_child"]["inventory"]["processes"][0].__setitem__(
                "used_gpu_memory_bytes", str(257 << 20)
            ),
            "library version": lambda value: value["qualification_preflight"]
            ["nvml_pre_child"].__setitem__("library_version", "13.999.0"),
            "capture time": lambda value: value["qualification_preflight"]
            ["nvml_pre_child"].__setitem__(
                "captured_at_unix_ns", str(int(PRE_CHILD_CAPTURED_AT_UNIX_NS) + 1)
            ),
            "runtime build": lambda value: value["prompt_source"]
            ["tokenizer_runtime"].__setitem__("executable_sha256", "e" * 64),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(manifest)
                mutate(changed)
                with self.assertRaisesRegex(
                    ValueError, "digest|binding|identity|range|coverage"
                ):
                    TOOL.validate_manifest(changed)

        original_sha256 = TOOL.manifest_sha256(manifest)
        fresh_preflight, _ = self.freeze_preflight(
            captured_at_unix_ns=str(int(PRE_CHILD_CAPTURED_AT_UNIX_NS) + 1)
        )
        stale_substitution = copy.deepcopy(manifest)
        stale_substitution["qualification_preflight"]["nvml_pre_child"] = (
            copy.deepcopy(fresh_preflight["nvml_pre_child"])
        )
        with self.assertRaisesRegex(ValueError, "digest|binding"):
            TOOL.validate_manifest(stale_substitution)

        intentionally_rebuilt = copy.deepcopy(manifest)
        intentionally_rebuilt["qualification_preflight"] = fresh_preflight
        TOOL.validate_manifest(intentionally_rebuilt)
        self.assertNotEqual(
            TOOL.manifest_sha256(intentionally_rebuilt),
            original_sha256,
            "a fresh baseline must change the immutable whole-manifest identity",
        )

    def test_schema_requires_closed_top_level_qualification_preflight(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertIn("qualification_preflight", schema["required"])
        self.assertEqual(
            schema["properties"]["qualification_preflight"],
            {"$ref": "#/$defs/qualification_preflight"},
        )

        preflight = schema["$defs"]["qualification_preflight"]
        self.assertIs(preflight["additionalProperties"], False)
        self.assertEqual(
            set(preflight["required"]),
            {
                "cold_preparation",
                "cold_preparation_sha256",
                "nvml_pre_child",
                "binding_sha256",
            },
        )
        self.assertEqual(
            preflight["properties"]["cold_preparation"],
            {"$ref": "#/$defs/cold_preparation"},
        )
        self.assertEqual(
            preflight["properties"]["nvml_pre_child"],
            {"$ref": "#/$defs/nvml_pre_child"},
        )

        cold = schema["$defs"]["cold_preparation"]
        self.assertIn("eligible_ranges", cold["required"])
        self.assertEqual(
            cold["properties"]["eligible_ranges"]["items"],
            {"$ref": "#/$defs/qualification_page_range"},
        )
        nvml_pre_child = schema["$defs"]["nvml_pre_child"]
        self.assertIs(nvml_pre_child["additionalProperties"], False)
        self.assertEqual(
            set(nvml_pre_child["required"]),
            {
                "capture_phase",
                "captured_at_unix_ns",
                "library_version",
                "inventory",
                "inventory_sha256",
            },
        )
        self.assertEqual(
            nvml_pre_child["properties"]["capture_phase"]["const"],
            "pre-child-before-cuda-initialization",
        )
        self.assertEqual(
            nvml_pre_child["properties"]["inventory"],
            {"$ref": "#/$defs/nvml_process_inventory"},
        )
        self.assertEqual(
            nvml_pre_child["properties"]["library_version"],
            {"$ref": "#/$defs/identity"},
        )
        self.assertEqual(
            nvml_pre_child["properties"]["captured_at_unix_ns"],
            {"$ref": "#/$defs/positive_uint64"},
        )
        self.assertEqual(
            schema["$defs"]["nvml_process_inventory"]["properties"]["api"]["const"],
            NVML_COMPUTE_API,
        )


QUALIFICATION_CONTROL_MESSAGE = struct.Struct("@IIIIQQQQQ")


def _qualification_control_wire_message(
    message_type: int,
    sequence: int,
    identity: os.stat_result | tuple[int, int, int, int] | None = None,
) -> bytes:
    if identity is None:
        fields = (0, 0, 0, 0)
    elif isinstance(identity, os.stat_result):
        fields = (
            identity.st_dev,
            identity.st_ino,
            identity.st_size,
            identity.st_mtime_ns,
        )
    else:
        fields = identity
    return QUALIFICATION_CONTROL_MESSAGE.pack(
        1,
        message_type,
        QUALIFICATION_CONTROL_MESSAGE.size,
        0,
        sequence,
        *fields,
    )


def _recv_control_wire_message(endpoint: socket.socket) -> tuple[int, int, tuple[int, ...]]:
    payload = bytearray()
    while len(payload) != QUALIFICATION_CONTROL_MESSAGE.size:
        part = endpoint.recv(QUALIFICATION_CONTROL_MESSAGE.size - len(payload))
        if not part:
            raise AssertionError("qualification control disconnected")
        payload.extend(part)
    values = QUALIFICATION_CONTROL_MESSAGE.unpack(payload)
    if values[0] != 1 or values[2] != QUALIFICATION_CONTROL_MESSAGE.size:
        raise AssertionError("invalid qualification control message")
    return values[1], values[4], values[5:]


class QualificationVersionAdmissionContractTest(unittest.TestCase):
    def _valid_version(self) -> dict[str, object]:
        return {
            "schema": "ds4.version/v1",
            "revision": "1234567890abcdef1234567890abcdef12345678",
            "dirty": False,
            "backend": "cuda",
            "features": ["laguna", "ssd_streaming"],
        }

    def test_accepts_only_a_clean_cuda_build_with_required_features(self) -> None:
        version = self._valid_version()
        admitted = TOOL.validate_qualification_version(version)
        self.assertEqual(admitted, version)
        self.assertIsNot(admitted, version)
        self.assertIsNot(admitted["features"], version["features"])

    def test_rejects_nonclosed_or_mistyped_version_payloads(self) -> None:
        cases: dict[str, Callable[[dict[str, object]], None]] = {
            "missing": lambda value: value.pop("revision"),
            "extra": lambda value: value.__setitem__("path", "/tmp/ds4"),
            "schema": lambda value: value.__setitem__("schema", "ds4.version/v2"),
            "dirty number": lambda value: value.__setitem__("dirty", 0),
            "features tuple": lambda value: value.__setitem__(
                "features", ("laguna", "ssd_streaming")
            ),
            "feature number": lambda value: value.__setitem__(
                "features", ["laguna", 7, "ssd_streaming"]
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                changed = self._valid_version()
                mutate(changed)
                with self.assertRaisesRegex(
                    ValueError, "missing|unknown|schema|boolean|array|feature"
                ):
                    TOOL.validate_qualification_version(changed)

    def test_rejects_invalid_or_placeholder_revisions(self) -> None:
        for revision in (
            "0" * 40,
            "a" * 39,
            "A" * 40,
            "g" * 40,
            7,
        ):
            with self.subTest(revision=revision):
                changed = self._valid_version()
                changed["revision"] = revision
                with self.assertRaisesRegex(ValueError, "revision|40-hex|sentinel"):
                    TOOL.validate_qualification_version(changed)

    def test_rejects_dirty_or_non_cuda_builds(self) -> None:
        mutations: dict[str, Callable[[dict[str, object]], None]] = {
            "dirty": lambda value: value.__setitem__("dirty", True),
            "cpu": lambda value: value.__setitem__("backend", "cpu"),
            "metal": lambda value: value.__setitem__("backend", "metal"),
            "rocm": lambda value: value.__setitem__("backend", "rocm"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = self._valid_version()
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "clean|dirty|CUDA|cuda|backend"):
                    TOOL.validate_qualification_version(changed)

    def test_requires_sorted_unique_laguna_and_ssd_streaming_features(self) -> None:
        candidates: dict[str, object] = {
            "missing laguna": ["ssd_streaming"],
            "missing streaming": ["laguna"],
            "unsorted": ["ssd_streaming", "laguna"],
            "duplicate": ["laguna", "laguna", "ssd_streaming"],
            "empty": [],
            "invalid spelling": ["laguna", "ssd-streaming"],
        }
        for label, features in candidates.items():
            with self.subTest(label=label):
                changed = self._valid_version()
                changed["features"] = features
                with self.assertRaisesRegex(
                    ValueError, "feature|sorted|unique|laguna|ssd_streaming"
                ):
                    TOOL.validate_qualification_version(changed)


class QualificationControlParentContractTest(unittest.TestCase):
    def _channel_and_child(self, **kwargs: object) -> tuple[object, socket.socket]:
        control = TOOL.QualificationControl.create(**kwargs)
        self.assertFalse(os.get_inheritable(control.child_fd))
        child = socket.socket(fileno=os.dup(control.child_fd))
        child.settimeout(1.0)
        control.close_child_endpoint()
        return control, child

    def _send_model(
        self,
        child: socket.socket,
        descriptor: int,
        *,
        passed_descriptors: tuple[int, ...] | None = None,
        identity: os.stat_result | tuple[int, int, int, int] | None = None,
    ) -> os.stat_result:
        observed = os.fstat(descriptor)
        rights = (descriptor,) if passed_descriptors is None else passed_descriptors
        ancillary = []
        if rights:
            ancillary = [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    array.array("i", rights),
                )
            ]
        sent = child.sendmsg(
            [
                _qualification_control_wire_message(
                    1,
                    0,
                    observed if identity is None else identity,
                )
            ],
            ancillary,
        )
        self.assertEqual(sent, QUALIFICATION_CONTROL_MESSAGE.size)
        return observed

    def _assert_model_ack(
        self, child: socket.socket, expected: os.stat_result
    ) -> None:
        self.assertEqual(
            _recv_control_wire_message(child),
            (
                6,
                0,
                (
                    expected.st_dev,
                    expected.st_ino,
                    expected.st_size,
                    expected.st_mtime_ns,
                ),
            ),
        )

    def test_receives_hashes_and_retains_one_exact_opened_model_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            payload = b"opened descriptor identity\0" * 4096
            model.write_bytes(payload)
            descriptor = os.open(model, os.O_RDONLY)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            try:
                expected = self._send_model(child, descriptor)
                hash_descriptor = TOOL._sha256_open_descriptor

                def hash_cloexec(received_fd: int) -> str:
                    self.assertFalse(os.get_inheritable(received_fd))
                    return hash_descriptor(received_fd)

                with mock.patch.object(
                    TOOL,
                    "_sha256_open_descriptor",
                    side_effect=hash_cloexec,
                ):
                    evidence = control.receive_model()
                self._assert_model_ack(child, expected)
                self.assertEqual(
                    evidence.identity.inode,
                    expected.st_ino,
                    "child must remain blocked until the parent acknowledges the "
                    "verified model identity",
                )
                self.assertEqual(evidence.sha256, hashlib.sha256(payload).hexdigest())
                self.assertEqual(evidence.identity.device, expected.st_dev)
                self.assertEqual(evidence.identity.inode, expected.st_ino)
                self.assertEqual(evidence.identity.size_bytes, expected.st_size)
                self.assertEqual(evidence.identity.mtime_ns, expected.st_mtime_ns)
                self.assertEqual(control.verify_model_unchanged(), evidence.identity)
                with self.assertRaisesRegex(ValueError, "already.*received"):
                    control.receive_model()
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_model_ack_waits_for_descriptor_bound_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"cold preparation before allocation")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            order: list[str] = []
            try:
                expected = self._send_model(child, descriptor)

                def prepare(received_fd: int, evidence: object) -> None:
                    order.append("prepare")
                    self.assertEqual(os.fstat(received_fd).st_ino, expected.st_ino)
                    self.assertEqual(evidence.identity.inode, expected.st_ino)
                    child.settimeout(0.01)
                    with self.assertRaises(socket.timeout):
                        child.recv(1)
                    child.settimeout(1.0)

                evidence = control.receive_model(prepare_descriptor=prepare)
                order.append("returned")
                self._assert_model_ack(child, expected)
                self.assertEqual(order, ["prepare", "returned"])
                self.assertEqual(evidence.identity.inode, expected.st_ino)
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_model_receive_rejects_missing_multiple_and_mismatched_rights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            descriptor = os.open(model, os.O_RDONLY)
            cases = {
                "missing": (),
                "multiple": (descriptor, descriptor),
            }
            try:
                for label, rights in cases.items():
                    with self.subTest(label=label):
                        control, child = self._channel_and_child(timeout_seconds=1.0)
                        try:
                            self._send_model(
                                child,
                                descriptor,
                                passed_descriptors=rights,
                            )
                            with self.assertRaisesRegex(
                                ValueError, "exactly one.*descriptor|SCM_RIGHTS"
                            ):
                                control.receive_model()
                        finally:
                            control.close()
                            child.close()

                control, child = self._channel_and_child(timeout_seconds=1.0)
                try:
                    status = os.fstat(descriptor)
                    changed = (
                        status.st_dev,
                        status.st_ino,
                        status.st_size + 1,
                        status.st_mtime_ns,
                    )
                    self._send_model(child, descriptor, identity=changed)
                    with self.assertRaisesRegex(ValueError, "identity.*match"):
                        control.receive_model()
                finally:
                    control.close()
                    child.close()
            finally:
                os.close(descriptor)

    def test_model_hash_has_pre_and_post_fstat_identity_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"before")
            descriptor = os.open(model, os.O_RDWR)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            try:
                self._send_model(child, descriptor)

                def mutate_during_hash(received_fd: int) -> str:
                    digest = hashlib.sha256(os.pread(received_fd, 6, 0)).hexdigest()
                    os.ftruncate(received_fd, 7)
                    return digest

                with mock.patch.object(
                    TOOL,
                    "_sha256_open_descriptor",
                    side_effect=mutate_during_hash,
                ):
                    with self.assertRaisesRegex(ValueError, "changed.*hash"):
                        control.receive_model()
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_ready_and_result_acknowledgements_bracket_parent_inventories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"checkpoint model")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            order: list[str] = []
            child_failure: list[BaseException] = []
            try:
                identity = self._send_model(child, descriptor)
                evidence = control.receive_model()
                self._assert_model_ack(child, identity)

                def child_checkpoint() -> None:
                    try:
                        order.append("ready-sent")
                        child.sendall(_qualification_control_wire_message(2, 7))
                        self.assertEqual(
                            _recv_control_wire_message(child),
                            (3, 7, (0, 0, 0, 0)),
                        )
                        order.append("ready-ack")
                        order.append("result-sent")
                        child.sendall(
                            _qualification_control_wire_message(4, 7, identity)
                        )
                        self.assertEqual(
                            _recv_control_wire_message(child),
                            (5, 7, (0, 0, 0, 0)),
                        )
                        order.append("result-ack")
                    except BaseException as exc:
                        child_failure.append(exc)

                worker = threading.Thread(target=child_checkpoint)
                worker.start()
                before, after = control.bracket_sample(
                    7,
                    capture_before=lambda: order.append("before") or "before-evidence",
                    capture_after=lambda: order.append("after") or "after-evidence",
                )
                worker.join(1.0)
                self.assertFalse(worker.is_alive(), "child remained blocked at a barrier")
                if child_failure:
                    raise child_failure[0]
                self.assertEqual(before, "before-evidence")
                self.assertEqual(after, "after-evidence")
                self.assertEqual(
                    order,
                    [
                        "ready-sent",
                        "before",
                        "ready-ack",
                        "result-sent",
                        "after",
                        "result-ack",
                    ],
                )
                self.assertEqual(control.verify_model_unchanged(), evidence.identity)
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    control.bracket_sample(
                        7,
                        capture_before=lambda: None,
                        capture_after=lambda: None,
                    )
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_wrong_ready_sequence_and_result_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"checkpoint model")
            descriptor = os.open(model, os.O_RDONLY)
            try:
                control, child = self._channel_and_child(timeout_seconds=1.0)
                try:
                    self._send_model(child, descriptor)
                    control.receive_model()
                    child.sendall(_qualification_control_wire_message(2, 10))
                    with self.assertRaisesRegex(ValueError, "sequence"):
                        control.bracket_sample(
                            9,
                            capture_before=lambda: None,
                            capture_after=lambda: None,
                        )
                    with self.assertRaisesRegex(ValueError, "unsafe"):
                        control.bracket_sample(
                            11,
                            capture_before=lambda: None,
                            capture_after=lambda: None,
                        )
                finally:
                    control.close()
                    child.close()

                control, child = self._channel_and_child(timeout_seconds=1.0)
                try:
                    identity = self._send_model(child, descriptor)
                    control.receive_model()
                    self._assert_model_ack(child, identity)

                    def mismatched_result() -> None:
                        child.sendall(_qualification_control_wire_message(2, 1))
                        _recv_control_wire_message(child)
                        changed = (
                            identity.st_dev,
                            identity.st_ino,
                            identity.st_size + 1,
                            identity.st_mtime_ns,
                        )
                        child.sendall(_qualification_control_wire_message(4, 1, changed))

                    worker = threading.Thread(target=mismatched_result)
                    worker.start()
                    with self.assertRaisesRegex(ValueError, "identity"):
                        control.bracket_sample(
                            1,
                            capture_before=lambda: None,
                            capture_after=lambda: None,
                        )
                    worker.join(1.0)
                finally:
                    control.close()
                    child.close()
            finally:
                os.close(descriptor)

    def test_post_receive_protocol_failure_closes_the_retained_model_fd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"retained model")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            received_fds: list[int] = []
            hash_descriptor = TOOL._sha256_open_descriptor
            try:
                self._send_model(child, descriptor)

                def remember_received_fd(received_fd: int) -> str:
                    received_fds.append(received_fd)
                    return hash_descriptor(received_fd)

                with mock.patch.object(
                    TOOL,
                    "_sha256_open_descriptor",
                    side_effect=remember_received_fd,
                ):
                    control.receive_model()
                self.assertEqual(len(received_fds), 1)
                os.fstat(received_fds[0])

                child.sendall(_qualification_control_wire_message(2, 8))
                with self.assertRaisesRegex(ValueError, "sequence"):
                    control.bracket_sample(
                        7,
                        capture_before=lambda: None,
                        capture_after=lambda: None,
                    )
                with self.assertRaises(OSError) as raised:
                    os.fstat(received_fds[0])
                self.assertEqual(raised.exception.errno, errno.EBADF)
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_child_work_and_parent_acknowledgements_use_separate_deadlines(
        self,
    ) -> None:
        class FakeMonotonic:
            value = 100.0

            def __call__(self) -> float:
                return self.value

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"sampling deadline model")
            descriptor = os.open(model, os.O_RDONLY)
            control, child = self._channel_and_child(timeout_seconds=1.0)
            clock = FakeMonotonic()
            receives: list[tuple[int, float]] = []
            sends: list[tuple[int, float, float]] = []
            try:
                self._send_model(child, descriptor)
                evidence = control.receive_model()
                control._monotonic = clock

                def receive_message(
                    *,
                    expected_type: int,
                    expected_sequence: int,
                    expect_model_fd: bool,
                    deadline: float,
                ) -> tuple[object, list[int]]:
                    self.assertEqual(expected_sequence, 17)
                    self.assertFalse(expect_model_fd)
                    receives.append((expected_type, deadline))
                    if expected_type == 2:
                        clock.value = deadline - 0.05
                        return TOOL.QualificationFileIdentity(0, 0, 0, 0), []
                    self.assertEqual(expected_type, 4)
                    clock.value = deadline - 0.05
                    return evidence.identity, []

                def send_message(
                    *, message_type: int, sequence: int, deadline: float
                ) -> None:
                    self.assertEqual(sequence, 17)
                    sends.append((message_type, deadline, clock.value))
                    if message_type == 3:
                        clock.value += 0.1

                def capture_before() -> str:
                    clock.value += 0.1
                    return "before"

                def capture_after() -> str:
                    clock.value += 0.2
                    return "after"

                with (
                    mock.patch.object(
                        control, "_receive_message", side_effect=receive_message
                    ),
                    mock.patch.object(
                        control, "_send_message", side_effect=send_message
                    ),
                ):
                    observed = control.bracket_sample(
                        17,
                        capture_before=capture_before,
                        capture_after=capture_after,
                        sample_timeout_seconds=5.0,
                    )

                self.assertEqual(observed, ("before", "after"))
                self.assertEqual(receives[0], (2, 105.0))
                self.assertEqual(sends[0], (3, 105.95, 105.05))
                self.assertEqual(receives[1][0], 4)
                self.assertAlmostEqual(receives[1][1], 110.15)
                self.assertEqual(sends[1][0], 5)
                self.assertAlmostEqual(
                    sends[1][1],
                    111.1,
                    msg="RESULT_ACK needs a fresh child-compatible control deadline",
                )
                self.assertAlmostEqual(sends[1][2], 110.3)
            finally:
                control.close()
                child.close()
                os.close(descriptor)

    def test_malformed_or_truncated_rights_close_every_aligned_prefix_fd(self) -> None:
        class FakeEndpoint:
            def __init__(self, response: tuple[bytes, list[tuple[int, int, bytes]], int, object]):
                self.response = response
                self.closed = False

            def recvmsg(
                self, _size: int, _ancillary_size: int, _flags: int
            ) -> tuple[bytes, list[tuple[int, int, bytes]], int, object]:
                return self.response

            def close(self) -> None:
                self.closed = True

        wire = _qualification_control_wire_message(1, 0)
        cases = {
            "non-int-aligned": lambda descriptor: (
                struct.pack("@i", descriptor) + b"x",
                0,
                "malformed",
            ),
            "MSG_CTRUNC": lambda descriptor: (
                struct.pack("@i", descriptor),
                socket.MSG_CTRUNC,
                "truncated",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"ancillary model")
            for label, response in cases.items():
                with self.subTest(label=label):
                    received_fd = os.open(model, os.O_RDONLY)
                    ancillary, flags, error = response(received_fd)
                    parent = FakeEndpoint(
                        (
                            wire,
                            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancillary)],
                            flags,
                            None,
                        )
                    )
                    child = FakeEndpoint((b"", [], 0, None))
                    control = TOOL.QualificationControl(
                        parent,
                        child,
                        timeout_seconds=1.0,
                        monotonic=lambda: 1.0,
                        wait_ready=lambda _endpoint, _write, _timeout: True,
                    )
                    try:
                        with self.assertRaisesRegex(ValueError, error):
                            control.receive_model()
                        with self.assertRaises(OSError) as raised:
                            os.fstat(received_fd)
                        self.assertEqual(raised.exception.errno, errno.EBADF)
                    finally:
                        control.close()
                        try:
                            os.close(received_fd)
                        except OSError:
                            pass

    def test_fake_monotonic_deadline_bounds_a_missing_model_message(self) -> None:
        class FakeMonotonic:
            value = 100.0

            def __call__(self) -> float:
                return self.value

        clock = FakeMonotonic()
        waits: list[float] = []

        def wait_ready(_socket: socket.socket, _write: bool, timeout: float) -> bool:
            waits.append(timeout)
            clock.value += timeout
            return False

        control, child = self._channel_and_child(
            timeout_seconds=0.25,
            monotonic=clock,
            wait_ready=wait_ready,
        )
        try:
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                control.receive_model()
            self.assertEqual(waits, [0.25])
            with self.assertRaisesRegex(ValueError, "unsafe"):
                control.receive_model()
        finally:
            control.close()
            child.close()

        control, child = self._channel_and_child(timeout_seconds=1.0)
        child.close()
        try:
            with self.assertRaisesRegex(ValueError, "disconnected"):
                control.receive_model()
            with self.assertRaisesRegex(ValueError, "unsafe"):
                control.receive_model()
        finally:
            control.close()

    def test_close_owns_both_socket_ends_and_the_received_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            descriptor = os.open(model, os.O_RDONLY)
            control = TOOL.QualificationControl.create(timeout_seconds=1.0)
            parent_fd = control.parent_fd
            child_fd = control.child_fd
            child = socket.socket(fileno=os.dup(child_fd))
            try:
                self._send_model(child, descriptor)
                control.receive_model()
                control.close()
                for owned in (parent_fd, child_fd):
                    with self.assertRaises(OSError):
                        os.fstat(owned)
                control.close()
            finally:
                child.close()
                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
