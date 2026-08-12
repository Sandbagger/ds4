#!/usr/bin/env python3
"""Contract tests for the immutable Laguna compact-runtime manifest."""

from __future__ import annotations

import base64
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
import subprocess
import tempfile
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
        "cold_preparation_advice": "posix_fadvise_dontneed",
        "runtime_disposal_advice": "madvise_dontneed",
    },
}


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

            model = Path(tmp) / "laguna-s-2.1-Q4_K_M.gguf"
            model_bytes = b"fake pinned Laguna model\n"
            model.write_bytes(model_bytes)
            output = Path(tmp) / "compact-runtime-benchmark-v1.json"
            patches = [
                mock.patch.object(TOOL, "ROOT", root),
                mock.patch.object(TOOL, "ORACLE_MANIFEST_PATH", oracle),
                mock.patch.object(TOOL, "ORACLE_TOKENIZER_REVISION", oracle_revision),
                mock.patch.object(TOOL, "MODEL_SIZE", len(model_bytes)),
                mock.patch.object(TOOL, "MODEL_SHA256", hashlib.sha256(model_bytes).hexdigest()),
                mock.patch.object(
                    TOOL, "collect_host_identity", return_value=copy.deepcopy(HOST_IDENTITY)
                ),
                mock.patch.dict(os.environ, {"FAKE_DS4_LOG": str(invocation_log)}),
            ]
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                yield {
                    "root": root,
                    "model": model,
                    "output": output,
                    "log": invocation_log,
                    "model_bytes": model_bytes,
                }

    def build_with_fake_cli(self, fixture: dict[str, object]) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                TOOL.main([
                    "manifest", "build", "--model", str(fixture["model"]),
                    "--output", str(fixture["output"]),
                ]),
                0,
            )

    def test_cli_build_and_verify_bind_and_retokenize_real_artifacts(self) -> None:
        with self.fake_cli_environment() as fixture:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    TOOL.main([
                        "manifest", "build", "--model", str(fixture["model"]),
                        "--output", str(fixture["output"]),
                    ]),
                    0,
                )
                self.assertEqual(
                    TOOL.main(["manifest", "verify", "--manifest", str(fixture["output"])]),
                    0,
                )
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("manifest_sha256=", stdout.getvalue())
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

    def test_parser_exposes_only_the_two_manifest_commands(self) -> None:
        build = TOOL.parse_args(["manifest", "build", "--model", "/m", "--output", "/o"])
        verify = TOOL.parse_args(["manifest", "verify", "--manifest", "/m"])
        self.assertEqual((build.command, build.action), ("manifest", "build"))
        self.assertEqual((verify.command, verify.action), ("manifest", "verify"))
        with self.assertRaises(SystemExit):
            TOOL.parse_args(["run"])


class ColdPreparationContractTest(unittest.TestCase):
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
                "eligible_calls",
                "eligible_bytes",
                "attempted_calls",
                "attempted_bytes",
                "successful_calls",
                "successful_bytes",
                "failed_calls",
                "failed_bytes",
                "errno_buckets",
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


if __name__ == "__main__":
    unittest.main()
