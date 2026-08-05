#!/usr/bin/env python3
"""Contract tests for the immutable Laguna compact-runtime manifest."""

from __future__ import annotations

import base64
import copy
import contextlib
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
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "gguf-tools/quality-testing/compact_runtime_qualify.py"
SCHEMA_PATH = ROOT / "schemas/compact-runtime-benchmark-v1.schema.json"

SPEC = importlib.util.spec_from_file_location("compact_runtime_qualify", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load manifest builder: {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


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

    def test_rejects_unknown_keys_at_every_manifest_level(self) -> None:
        mutations = [
            ((), "extra"),
            (("model",), "extra"),
            (("host",), "extra"),
            (("host", "filesystem"), "extra"),
            (("host", "nvme"), "extra"),
            (("host", "io"), "extra"),
            (("prompt_source",), "extra"),
            (("prompt_source", "tokenizer_runtime"), "extra"),
            (("prompts", 0), "extra"),
            (("sampling",), "extra"),
            (("execution",), "extra"),
            (("profiles", 0), "extra"),
        ]
        for path, key in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.manifest)
                node = changed
                for part in path:
                    node = node[part]
                node[key] = "forbidden"
                with self.assertRaisesRegex(ValueError, "unknown key"):
                    TOOL.validate_manifest(changed)

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


if __name__ == "__main__":
    unittest.main()
