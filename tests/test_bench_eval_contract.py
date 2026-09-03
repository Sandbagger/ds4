#!/usr/bin/env python3
"""Host-only RED gates for Task 19 benchmark lifecycle integration.

The CLI cases build the host-only benchmark once, feed it an exact sequence
produced by the checked-in Python builder, and stop at a missing-model or
qualification-only compatibility boundary.  They never load a model, open a
prompt file, initialize a GPU, or open a network connection.

The JSONL schema and canned fixture are checked with the dependency-free
validator beside this file.  The model-free C emitter target intentionally
names the next production header/API; it is a separate RED boundary until that
module exists.  Source inspection is limited to the engine/session ownership
shape.  It is not used as evidence for emitted event ordering or field values.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_SOURCE_PATH = ROOT / "ds4_bench.c"
BENCH_SOURCE = BENCH_SOURCE_PATH.read_text(encoding="utf-8")
BUILDER = ROOT / "gguf-tools/quality-testing/compact_runtime_qualify.py"
SCHEMA_PATH = ROOT / "schemas/ds4-bench-qualification-v1.schema.json"
FIXTURE_PATH = ROOT / "tests/fixtures/ds4-bench-qualification-v1.jsonl"
VALIDATOR = ROOT / "tests/validate_bench_qualification_json.py"
QUALIFICATION_SEQUENCE_OPTION = "--qualification-sequence"
QUALIFICATION_MANIFEST_SHA256_OPTION = "--qualification-manifest-sha256"
QUALIFICATION_SEQUENCE_SHA256_OPTION = "--qualification-sequence-sha256"
MISSING_MODEL = "/definitely/missing/task19-model.gguf"
MISSING_PROMPT = "/definitely/missing/task19-prompt.txt"
NUL_RENDERED_SIZE = 59
NUL_RENDERED_SHA256 = "c6f7134b47f7fb0ed0693100ca855fec1ab9c5ba626007777f4c020c141d71d9"
NUL_RENDERED_BASE64 = "44CIfEVPU3zjgIk8dXNlcj5vZmZpY2lhbABwcm9tcHQ8L3VzZXI+Cjxhc3Npc3RhbnQ+PC90aGluaz4="
LIFECYCLE_EVENTS = ("request_accepted", "first_token", "request_complete")
EXTERNAL_ATTRIBUTION_FIELDS = (
    "model_source_resident",
    "host_library_unattributed",
    "cuda_library_unattributed",
    "unrelated_process_inventory_stable",
)
BASE_SCHEMA_FIELDS = (
    "schema",
    "manifest_sha256",
    "sequence_sha256",
    "profile_id",
    "prompt_order_index",
    "prompt_id",
    "input_sha256",
    "event",
    "request_id",
    "instance_id",
    "snapshot_seq",
    "repetition_index",
    "monotonic_ns",
    "mode",
    "session_payload_bytes",
    "kv_allocated_bytes",
    "configured_prefill_rows",
    "allocated_prefill_rows",
    "expert_cache_limit_bytes",
    "expert_cache_current_bytes",
    "expert_cache_peak_bytes",
    "qualification_total_current_bytes",
    "qualification_total_bound_bytes",
    "qualification_total_peak_bytes",
    "model_inode_resident_bytes",
    "external_attribution",
    "runtime",
)
UINT64_FIELDS = (
    "snapshot_seq",
    "monotonic_ns",
    "session_payload_bytes",
    "kv_allocated_bytes",
    "expert_cache_limit_bytes",
    "expert_cache_current_bytes",
    "expert_cache_peak_bytes",
    "qualification_total_current_bytes",
    "qualification_total_bound_bytes",
    "qualification_total_peak_bytes",
    "model_inode_resident_bytes",
)
INTEGER_FIELDS = (
    "prompt_order_index",
    "repetition_index",
    "configured_prefill_rows",
    "allocated_prefill_rows",
)
PROFILE_PROMPT_ORDER = {
    "cache-8gib": ("native-512", "native-2048", "native-28672", "native-8192"),
    "cache-12gib": ("native-2048", "native-8192", "native-512", "native-28672"),
    "cache-16gib": ("native-8192", "native-28672", "native-2048", "native-512"),
}
POST_PARSE_DIAGNOSTICS = (
    re.compile(r"cannot\s+(?:open|stat)\s+model", re.I),
    re.compile(r"failed\s+to\s+open\s+model", re.I),
    re.compile(r"qualification(?:[- ]only)?[^\n]*(?:requires|supports|incompatible|unsupported)", re.I),
    re.compile(r"(?:requires|unsupported|unavailable|incompatible)[^\n]*(?:cuda|streaming|qualification)", re.I),
    re.compile(r"(?:cuda|streaming)[^\n]*(?:requires|unsupported|unavailable|incompatible)", re.I),
)


def _matching_brace(source: str, brace: int) -> int:
    """Find a C brace pair while ignoring comments, strings, and chars."""

    if brace < 0 or brace >= len(source) or source[brace] != "{":
        raise AssertionError("brace scanner did not start at an opening brace")
    depth = 0
    index = brace
    state = "code"
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if byte == "/" and following == "/":
                state = "line-comment"
                index += 2
                continue
            if byte == "/" and following == "*":
                state = "block-comment"
                index += 2
                continue
            if byte == '"':
                state = "string"
                index += 1
                continue
            if byte == "'":
                state = "char"
                index += 1
                continue
            if byte == "{":
                depth += 1
            elif byte == "}":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
            continue
        if state == "line-comment":
            if byte in "\r\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if byte == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if byte == "\\":
            index += 2
        elif byte == '"' and state == "string":
            state = "code"
            index += 1
        elif byte == "'" and state == "char":
            state = "code"
            index += 1
        else:
            index += 1
    raise AssertionError("unterminated C brace body")


def _first_code_brace(source: str, start: int) -> int:
    """Return the first opening brace after start outside C lexical literals."""

    index = start
    state = "code"
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if byte == "/" and following == "/":
                state = "line-comment"
                index += 2
                continue
            if byte == "/" and following == "*":
                state = "block-comment"
                index += 2
                continue
            if byte == '"':
                state = "string"
                index += 1
                continue
            if byte == "'":
                state = "char"
                index += 1
                continue
            if byte == "{":
                return index
            index += 1
            continue
        if state == "line-comment":
            if byte in "\r\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if byte == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if byte == "\\":
            index += 2
        elif (state == "string" and byte == '"') or (state == "char" and byte == "'"):
            state = "code"
            index += 1
        else:
            index += 1
    raise AssertionError("no code brace after function signature")


def _function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing source function {signature}")
    brace = _first_code_brace(source, start + len(signature))
    end = _matching_brace(source, brace)
    return source[brace + 1 : end]


def _function_bodies(source: str) -> list[tuple[str, str]]:
    """Find simple function definitions and scan bodies lexically."""

    pattern = re.compile(
        r"(?ms)^(?:static\s+)?[A-Za-z_][^;{}]*?\b"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*?\)\s*\{"
    )
    functions: list[tuple[str, str]] = []
    for match in pattern.finditer(source):
        name = match.group("name")
        if name in {"if", "for", "while", "switch"}:
            continue
        brace = match.end() - 1
        end = _matching_brace(source, brace)
        functions.append((name, source[brace + 1 : end]))
    return functions


def _strip_c_comments(source: str) -> str:
    """Remove comments without interpreting comment markers in string literals."""

    output: list[str] = []
    index = 0
    state = "code"
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if byte == "/" and following == "/":
                output.extend((" ", " "))
                state = "line-comment"
                index += 2
            elif byte == "/" and following == "*":
                output.extend((" ", " "))
                state = "block-comment"
                index += 2
            elif byte == '"':
                output.append(byte)
                state = "string"
                index += 1
            elif byte == "'":
                output.append(byte)
                state = "char"
                index += 1
            else:
                output.append(byte)
                index += 1
            continue
        if state == "line-comment":
            if byte in "\r\n":
                output.append(byte)
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block-comment":
            if byte == "*" and following == "/":
                output.extend((" ", " "))
                state = "code"
                index += 2
            else:
                output.append("\n" if byte in "\r\n" else " ")
                index += 1
            continue
        output.append(byte)
        if byte == "\\" and index + 1 < len(source):
            output.append(source[index + 1])
            index += 2
        elif (state == "string" and byte == '"') or (state == "char" and byte == "'"):
            state = "code"
            index += 1
        else:
            index += 1
    return "".join(output)


def _c_code_view(source: str) -> str:
    return _strip_c_comments(source).replace(r'\"', '"')


def _loop_body_with_four_repetitions(source: str) -> tuple[str, str] | None:
    for match in re.finditer(r"\bfor\s*\((?P<header>[^{}]*)\)\s*\{", source, re.S):
        header = match.group("header")
        if not re.search(r"\brepetition_index\s*=\s*0\b", header):
            continue
        if not re.search(r"\brepetition_index\s*<\s*4\b", header):
            continue
        brace = match.end() - 1
        end = _matching_brace(source, brace)
        return header, source[brace + 1 : end]
    return None


def _run_validator(
    payload: bytes, args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        cwd=ROOT,
        input=payload,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _lifecycle_fixture() -> list[dict[str, Any]]:
    base = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    events = ("request_accepted", "first_token", "request_complete")
    records: list[dict[str, Any]] = []
    for repetition in range(4):
        request_id = f"123e4567-e89b-12d3-a456-42661417400{repetition + 1}"
        for event_index, event in enumerate(events):
            record = json.loads(json.dumps(base))
            record["event"] = event
            record["repetition_index"] = repetition
            record["request_id"] = request_id
            record["monotonic_ns"] = str(1000000 + len(records))
            runtime_seq = 100 + repetition * 4 + (3 if event == "request_complete" else event_index)
            snapshot_seq = str(runtime_seq)
            record["snapshot_seq"] = snapshot_seq
            record["runtime"]["snapshot_seq"] = snapshot_seq
            if event == "request_complete":
                request = record["request_metrics"]
                request["request_id"] = request_id
                request["instance_id"] = record["instance_id"]
                request["snapshot_seq"] = str(runtime_seq - 1)
            else:
                record.pop("request_metrics")
                record.pop("terminal_status")
            records.append(record)
    return records


def _sequence_digests(sequence: Path) -> tuple[str, str]:
    raw = sequence.read_bytes()
    manifest_digest = next(
        line.split(b"=", 1)[1].decode("ascii")
        for line in raw.splitlines()
        if line.startswith(b"manifest_sha256=")
    )
    return manifest_digest, hashlib.sha256(raw).hexdigest()


class BenchQualificationMakefileContractTest(unittest.TestCase):
    def test_emitter_target_runs_in_aggregate_make_test(self) -> None:
        lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        start = next(
            index for index, line in enumerate(lines) if line.startswith("test:")
        )
        dependencies = lines[start]
        while dependencies.rstrip().endswith("\\"):
            start += 1
            dependencies += " " + lines[start].strip()
        self.assertIn("test-bench-qualification-emitter", dependencies.split())

    def test_production_translation_unit_gate_is_direct_aggregate_and_cleaned(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        lines = makefile.splitlines()
        phony = next(line for line in lines if line.startswith(".PHONY:"))
        self.assertIn("test-bench-qualification-production-compile", phony.split())

        target = re.search(
            r"(?m)^test-bench-qualification-production-compile: (?P<deps>[^\n]+)\n"
            r"(?P<recipe>\t[^\n]+)$",
            makefile,
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.group("deps").split(), ["ds4_bench.c", "ds4_gpu.h"])
        recipe = target.group("recipe")
        for flag in (
            "$(CC) $(CFLAGS)",
            "-DDS4_NO_GPU",
            "-DDS4_BENCH_QUALIFICATION_TEST_BACKEND",
            "-Werror=implicit-function-declaration",
        ):
            self.assertIn(flag, recipe)
        self.assertRegex(
            recipe,
            r"-c\s+-o\s+tests/test_bench_qualification_production_compile\.o\s+ds4_bench\.c$",
        )
        self.assertNotIn("test_bench_qualification_lifecycle.c", recipe)

        start = next(index for index, line in enumerate(lines) if line.startswith("test:"))
        dependencies = lines[start]
        while dependencies.rstrip().endswith("\\"):
            start += 1
            dependencies += " " + lines[start].strip()
        self.assertIn("test-bench-qualification-production-compile", dependencies.split())

        clean = next(line for line in lines if line.startswith("\trm -f "))
        self.assertIn("tests/test_bench_qualification_production_compile.o", clean)

    def test_bench_directly_includes_gpu_api(self) -> None:
        self.assertRegex(BENCH_SOURCE, r'(?m)^#include "ds4_gpu\.h"$')

    def test_bench_object_rules_depend_on_gpu_api(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target_name in ("ds4_bench.o", "ds4_bench_cpu.o"):
            with self.subTest(target=target_name):
                rule = re.search(
                    rf"(?m)^{re.escape(target_name)}: (?P<deps>[^\n]+)$",
                    makefile,
                )
                self.assertIsNotNone(rule)
                assert rule is not None
                self.assertIn("ds4_gpu.h", rule.group("deps").split())

    def test_linux_identity_hashes_proc_exe_directly(self) -> None:
        identity = _function_body(
            BENCH_SOURCE, "static bool qualification_build_identity("
        )
        self.assertRegex(
            identity,
            r'(?ms)#if\s+defined\(__linux__\)\s*\n\s*'
            r'const int fd = open\("/proc/self/exe", flags\);',
        )
        self.assertNotRegex(
            BENCH_SOURCE, r'\breadlink\s*\(\s*"/proc/self/exe"'
        )


class CScannerContractTest(unittest.TestCase):
    def test_brace_scanner_ignores_braces_in_c_literals(self) -> None:
        source = (
            'static int example(void) {\n'
            '  puts("{ a string brace } and \\\" quote");\n'
            '  if (1) { puts("}"); }\n'
            '  return 0;\n'
            '}\n'
        )
        body = _function_body(source, "static int example(void)")
        self.assertIn('puts("{ a string brace }', body)
        self.assertIn('if (1) {', body)
        self.assertNotIn("unterminated", body)


class BenchQualificationSchemaContractTest(unittest.TestCase):
    def test_closed_schema_and_canonical_canned_jsonl_fixture(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], "ds4.bench.qualification/v1")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(tuple(schema["required"]), BASE_SCHEMA_FIELDS)
        self.assertEqual(
            set(schema["properties"]),
            set(BASE_SCHEMA_FIELDS) | {"request_metrics", "terminal_status"},
        )
        external = schema["$defs"]["external_attribution"]
        self.assertIs(external["additionalProperties"], False)
        self.assertEqual(set(external["required"]), set(EXTERNAL_ATTRIBUTION_FIELDS))
        self.assertEqual(
            set(external["properties"]), set(EXTERNAL_ATTRIBUTION_FIELDS)
        )
        self.assertEqual(
            schema["properties"]["runtime"]["$ref"],
            "ds4-runtime-v1.schema.json",
        )
        self.assertEqual(
            schema["properties"]["request_metrics"]["$ref"],
            "ds4-runtime-request-v1.schema.json",
        )
        self.assertEqual(
            schema["properties"]["sequence_sha256"]["$ref"],
            "#/$defs/sha256",
        )
        fixture_record = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertRegex(
            fixture_record["sequence_sha256"],
            r"^(?!0{64}$)[0-9a-f]{64}$",
        )
        conditional = json.dumps(schema["allOf"])
        for marker in ("request_complete", "terminal_status", "request_metrics"):
            self.assertIn(marker, conditional)

        fixture = FIXTURE_PATH.read_bytes()
        result = _run_validator(fixture, ("--fixture",))
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_schema_rejects_extra_fields_terminal_early_nonfinite_and_wrong_kinds(self) -> None:
        record = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        def rejected(mutator: Any) -> None:
            candidate = json.loads(json.dumps(record))
            mutator(candidate)
            result = _run_validator(
                (json.dumps(candidate, separators=(",", ":")) + "\n").encode(),
                ("--fixture",),
            )
            self.assertNotEqual(result.returncode, 0, result.stdout.decode())

        rejected(lambda value: value.__setitem__("unexpected", 1))
        rejected(lambda value: value["external_attribution"].__setitem__("extra", "0"))
        rejected(lambda value: value.__setitem__("terminal_status", None))
        rejected(lambda value: value.__setitem__("event", "first_token"))
        rejected(lambda value: value.__setitem__("monotonic_ns", "NaN"))
        rejected(lambda value: value.__setitem__("sequence_sha256", "A" * 64))
        rejected(lambda value: value.__setitem__("sequence_sha256", "0" * 64))
        rejected(lambda value: value.pop("sequence_sha256"))
        rejected(lambda value: value.__setitem__("snapshot_seq", 42))
        rejected(lambda value: value.__setitem__("repetition_index", "0"))

    def test_canonical_bindings_and_lifecycle_invariants_are_rejected_when_tampered(self) -> None:
        records = _lifecycle_fixture()
        payload = ("\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n").encode()
        result = _run_validator(payload)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

        def rejected(mutator: Any) -> None:
            candidate = json.loads(json.dumps(records))
            mutator(candidate)
            candidate_payload = ("\n".join(json.dumps(record, separators=(",", ":")) for record in candidate) + "\n").encode()
            result = _run_validator(candidate_payload)
            self.assertNotEqual(result.returncode, 0, result.stdout.decode())

        rejected(lambda value: value[0].__setitem__("instance_id", "123e4567-e89b-12d3-a456-426614174099"))
        rejected(lambda value: value[1].__setitem__("snapshot_seq", "999"))
        rejected(lambda value: value[2].__setitem__("kv_allocated_bytes", "1"))
        rejected(lambda value: value[2].__setitem__("expert_cache_current_bytes", "1"))
        rejected(lambda value: value[2].__setitem__("qualification_total_current_bytes", "1"))
        rejected(lambda value: value[2].__setitem__("model_inode_resident_bytes", "1"))
        rejected(lambda value: value[2]["external_attribution"].__setitem__("model_source_resident", "1"))
        rejected(lambda value: value[0].__setitem__("profile_id", "cache-12gib"))
        rejected(lambda value: value[0].__setitem__("prompt_id", "native-2048"))
        rejected(lambda value: value[0].__setitem__("prompt_order_index", 1))
        rejected(lambda value: value[2].__setitem__("prompt_id", "native-2048"))
        rejected(lambda value: value[2]["request_metrics"].__setitem__("request_id", "123e4567-e89b-12d3-a456-426614174099"))
        rejected(lambda value: value[2]["request_metrics"].__setitem__("instance_id", "123e4567-e89b-12d3-a456-426614174099"))
        rejected(lambda value: value[4].__setitem__("request_id", value[0]["request_id"]))
        rejected(lambda value: value[1].__setitem__("request_id", "123e4567-e89b-12d3-a456-426614174099"))
        rejected(lambda value: value[4].__setitem__("monotonic_ns", value[3]["monotonic_ns"]))
        rejected(lambda value: value[4]["runtime"].__setitem__("snapshot_seq", value[3]["runtime"]["snapshot_seq"]))
        rejected(lambda value: value[0]["runtime"]["build"]["features"].__setitem__(1, "laguna"))
        rejected(lambda value: value[2].__setitem__("terminal_status", "cancelled"))
        rejected(lambda value: value[0]["runtime"]["config"].__setitem__("context_tokens", 1))
        rejected(lambda value: value[0]["runtime"]["config"].__setitem__("ssd_streaming", False))
        rejected(lambda value: value[0]["runtime"]["limits"].__setitem__("effective_prefill_chunk_tokens", 1))
        rejected(lambda value: value[2]["request_metrics"].__setitem__("prefill_tokens_per_second", 1e999))
        raw_overflow = payload.replace(b'"prefill_tokens_per_second":512.0', b'"prefill_tokens_per_second":1e999')
        self.assertNotEqual(_run_validator(raw_overflow).returncode, 0)


class BenchQualificationPreflightCliTest(unittest.TestCase):
    """Exercise the real host-only CLI without accepting CPU qualification."""

    bench = ROOT / "ds4-bench"

    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            ["make", "cpu"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                "host-only ds4-bench build failed: "
                + (result.stderr or result.stdout)[-4000:]
            )
        if not cls.bench.is_file():
            raise AssertionError(f"host build did not produce {cls.bench}")

    def _generate_sequence(
        self,
        directory: Path,
        profile_id: str = "cache-8gib",
        prompt_id: str = "native-512",
    ) -> Path:
        """Use the production sequence builder with a tiny deterministic manifest.

        The checked-in production manifest is intentionally bound to a real
        model and host. This host-only gate supplies the builder's pure
        sequence function with the same pinned profile constants and compact
        official-template prompts, while bypassing only manifest identity
        validation. No profile order or sequence line is duplicated here.
        """
        spec = importlib.util.spec_from_file_location("task19_sequence_builder", BUILDER)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        prompts = []
        for target in builder.PROMPT_TARGETS:
            prompt_id_for_target = f"native-{target}"
            payload = b"task19 canonical prompt bytes " + str(target).encode("ascii")
            rendered = builder.LAGUNA_TEMPLATE_PREFIX + payload + builder.LAGUNA_TEMPLATE_SUFFIX
            prompts.append({
                "id": prompt_id_for_target,
                "token_count": target,
                "payload_prefix_bytes": str(len(payload)),
                "rendered_size_bytes": str(len(rendered)),
                "rendered_base64": base64.b64encode(rendered).decode("ascii"),
                "sha256": hashlib.sha256(rendered).hexdigest(),
            })
        manifest = {
            "schema": builder.SCHEMA_ID,
            "prompts": prompts,
            "profiles": [
                {
                    "profile_id": candidate_id,
                    "cache_bytes": str(cache_bytes),
                    "prompt_order": list(prompt_order),
                }
                for candidate_id, cache_bytes, prompt_order in builder.PROFILE_SPECS
            ],
        }
        original_validator = builder.validate_manifest
        builder.validate_manifest = lambda _value: None
        try:
            sequence_bytes = builder.build_qualification_sequence(
                manifest, profile_id, prompt_id
            )
        finally:
            builder.validate_manifest = original_validator
        output = directory / f"sequence-{profile_id}-{prompt_id}.txt"
        output.write_bytes(sequence_bytes)
        self.assertEqual(sequence_bytes.count(b"\n"), builder.QUALIFICATION_SEQUENCE_LINE_COUNT)
        self.assertTrue(sequence_bytes.startswith(b"schema=ds4.qualification-sequence/v1\n"))
        self.assertTrue(sequence_bytes.endswith(b"repetition=3:warm-3\n"))
        return output

    def _authenticated_sequence_args(self, sequence: Path) -> tuple[str, ...]:
        manifest_digest, sequence_digest = _sequence_digests(sequence)
        return (
            QUALIFICATION_SEQUENCE_OPTION,
            str(sequence),
            QUALIFICATION_MANIFEST_SHA256_OPTION,
            manifest_digest,
            QUALIFICATION_SEQUENCE_SHA256_OPTION,
            sequence_digest,
        )

    def _invoke(
        self,
        directory: Path,
        sequence_args: tuple[str, ...],
        extra: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["DS4_LOCK_FILE"] = str(directory / "task19-private.lock")
        return subprocess.run(
            [
                str(self.bench),
                *sequence_args,
                "--model",
                MISSING_MODEL,
                "--backend",
                "cuda",
                *extra,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def _assert_rejected_before_model_or_prompt_open(
        self,
        directory: Path,
        sequence_args: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        result = self._invoke(directory, sequence_args)
        stderr = result.stderr.lower()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("unknown option", stderr)
        self.assertRegex(stderr, r"qualification[- ]sequence")
        self.assertNotIn("cannot open model", stderr)
        self.assertNotIn("failed to open model", stderr)
        self.assertNotIn(MISSING_MODEL.lower(), stderr)
        self.assertNotIn(MISSING_PROMPT.lower(), stderr)
        return result

    def _assert_reached_post_parse_boundary(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
        stderr = result.stderr
        self.assertNotIn("unknown option", stderr.lower(), stderr)
        self.assertTrue(
            any(pattern.search(stderr) for pattern in POST_PARSE_DIAGNOSTICS),
            "valid sequence did not reach a distinct model/compatibility boundary:\n"
            + stderr,
        )

    def test_python_builder_all_profile_prompt_pairs_reach_post_parse_boundary(self) -> None:
        spec = importlib.util.spec_from_file_location("task19_sequence_pairs", BUILDER)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        pairs = [
            (profile_id, f"native-{target}")
            for profile_id, _cache_bytes, _prompt_order in builder.PROFILE_SPECS
            for target in builder.PROMPT_TARGETS
        ]
        self.assertEqual(len(pairs), 12)
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            for profile_id, prompt_id in pairs:
                with self.subTest(profile_id=profile_id, prompt_id=prompt_id):
                    sequence = self._generate_sequence(directory, profile_id, prompt_id)
                    result = self._invoke(
                        directory, self._authenticated_sequence_args(sequence)
                    )
                    # A host-only binary may reject the unavailable CUDA backend or
                    # the deliberately missing model. It must not accept CPU as
                    # qualification.
                    self.assertIn(result.returncode, (1, 2), result.stderr)
                    self._assert_reached_post_parse_boundary(result)

    def test_malformed_empty_and_nul_decoded_inputs_reject_before_model_open(self) -> None:
        cases = (("empty-file", b""), ("empty-object", b"{}\n"), ("malformed", b"{\n"))
        for label, payload in cases:
            with self.subTest(sequence=label), tempfile.TemporaryDirectory(
                prefix="task19-sequence-"
            ) as directory_name:
                directory = Path(directory_name)
                sequence = directory / f"{label}.txt"
                sequence.write_bytes(payload)
                self._assert_rejected_before_model_or_prompt_open(
                    directory,
                    (
                        QUALIFICATION_SEQUENCE_OPTION,
                        str(sequence),
                        QUALIFICATION_MANIFEST_SHA256_OPTION,
                        "a" * 64,
                        QUALIFICATION_SEQUENCE_SHA256_OPTION,
                        "b" * 64,
                    ),
                )

        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            sequence = self._generate_sequence(directory)
            lines = sequence.read_bytes().splitlines()
            canonical = next(line.split(b"=", 1)[1] for line in lines if line.startswith(b"input_base64="))
            canonical_rendered = base64.b64decode(canonical, validate=True)
            nul_payload = (
                b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
                + b"official\x00prompt"
                + b"</user>\n<assistant></think>"
            )
            self.assertEqual(canonical_rendered[: len(b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>")], b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>")
            self.assertEqual(canonical_rendered[-len(b"</user>\n<assistant></think>"):], b"</user>\n<assistant></think>")
            self.assertEqual(len(nul_payload), NUL_RENDERED_SIZE)
            self.assertEqual(hashlib.sha256(nul_payload).hexdigest(), NUL_RENDERED_SHA256)
            self.assertEqual(base64.b64encode(nul_payload).decode("ascii"), NUL_RENDERED_BASE64)
            self.assertIn(b"\x00", base64.b64decode(NUL_RENDERED_BASE64, validate=True))
            replacements = {
                b"input_size_bytes=": f"input_size_bytes={NUL_RENDERED_SIZE}".encode(),
                b"input_sha256=": b"input_sha256=" + NUL_RENDERED_SHA256.encode(),
                b"input_base64=": b"input_base64=" + NUL_RENDERED_BASE64.encode(),
            }
            mutated: list[bytes] = []
            for line in lines:
                replacement = next(
                    (value for key, value in replacements.items() if line.startswith(key)),
                    None,
                )
                mutated.append(replacement if replacement is not None else line)
            nul_sequence = directory / "decoded-nul.txt"
            nul_sequence.write_bytes(b"\n".join(mutated) + b"\n")
            self._assert_rejected_before_model_or_prompt_open(
                directory, self._authenticated_sequence_args(nul_sequence)
            )

    def test_authenticated_sequence_digest_options_are_valid_and_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            sequence = self._generate_sequence(directory)
            manifest_digest, sequence_digest = _sequence_digests(sequence)
            authenticated = (
                QUALIFICATION_SEQUENCE_OPTION,
                str(sequence),
                QUALIFICATION_MANIFEST_SHA256_OPTION,
                manifest_digest,
                QUALIFICATION_SEQUENCE_SHA256_OPTION,
                sequence_digest,
            )
            valid = self._invoke(directory, authenticated)
            self.assertIn(valid.returncode, (1, 2), valid.stderr)
            self._assert_reached_post_parse_boundary(valid)

            for omitted, args in (
                (
                    QUALIFICATION_MANIFEST_SHA256_OPTION,
                    (
                        QUALIFICATION_SEQUENCE_OPTION,
                        str(sequence),
                        QUALIFICATION_SEQUENCE_SHA256_OPTION,
                        sequence_digest,
                    ),
                ),
                (
                    QUALIFICATION_SEQUENCE_SHA256_OPTION,
                    (
                        QUALIFICATION_SEQUENCE_OPTION,
                        str(sequence),
                        QUALIFICATION_MANIFEST_SHA256_OPTION,
                        manifest_digest,
                    ),
                ),
            ):
                with self.subTest(case="missing", option=omitted):
                    result = self._invoke(directory, args)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    stderr = result.stderr.lower()
                    self.assertNotIn("unknown option", stderr)
                    self.assertRegex(stderr, r"required|missing|once|digest")
                    self.assertNotIn(MISSING_MODEL.lower(), stderr)
                    self.assertNotIn(MISSING_PROMPT.lower(), stderr)

            for option, value in (
                (QUALIFICATION_MANIFEST_SHA256_OPTION, manifest_digest),
                (QUALIFICATION_SEQUENCE_SHA256_OPTION, sequence_digest),
            ):
                with self.subTest(case="duplicate", option=option):
                    result = self._invoke(directory, authenticated + (option, value))
                    self.assertEqual(result.returncode, 2, result.stderr)
                    stderr = result.stderr.lower()
                    self.assertNotIn("unknown option", stderr)
                    self.assertRegex(stderr, r"duplicate|once|only")
                    self.assertNotIn(MISSING_MODEL.lower(), stderr)
                    self.assertNotIn(MISSING_PROMPT.lower(), stderr)

                with self.subTest(case="empty", option=option):
                    empty_args = list(authenticated)
                    empty_args[empty_args.index(option) + 1] = ""
                    result = self._invoke(directory, tuple(empty_args))
                    self.assertEqual(result.returncode, 2, result.stderr)
                    stderr = result.stderr.lower()
                    self.assertNotIn("unknown option", stderr)
                    self.assertRegex(stderr, r"empty|non-empty|digest")
                    self.assertNotIn(MISSING_MODEL.lower(), stderr)
                    self.assertNotIn(MISSING_PROMPT.lower(), stderr)

    def test_authenticated_digest_mismatch_rejects_before_model_prompt_or_engine_access(self) -> None:
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            sequence = self._generate_sequence(directory)
            manifest_digest, sequence_digest = _sequence_digests(sequence)
            wrong_manifest = "f" * 64 if manifest_digest != "f" * 64 else "e" * 64
            wrong_sequence = "f" * 64 if sequence_digest != "f" * 64 else "e" * 64
            cases = (
                (
                    "manifest",
                    wrong_manifest,
                    sequence_digest,
                ),
                (
                    "sequence",
                    manifest_digest,
                    wrong_sequence,
                ),
            )
            for label, expected_manifest, expected_sequence in cases:
                with self.subTest(digest=label):
                    result = self._invoke(
                        directory,
                        (
                            QUALIFICATION_SEQUENCE_OPTION,
                            str(sequence),
                            QUALIFICATION_MANIFEST_SHA256_OPTION,
                            expected_manifest,
                            QUALIFICATION_SEQUENCE_SHA256_OPTION,
                            expected_sequence,
                        ),
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    stderr = result.stderr.lower()
                    self.assertNotIn("unknown option", stderr)
                    self.assertRegex(stderr, r"manifest|sequence|digest|sha-256|mismatch")
                    self.assertNotIn("cannot open model", stderr)
                    self.assertNotIn("failed to open model", stderr)
                    self.assertNotIn(MISSING_MODEL.lower(), stderr)
                    self.assertNotIn(MISSING_PROMPT.lower(), stderr)

    def test_duplicate_and_empty_sequence_options_have_stable_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            sequence = self._generate_sequence(directory)
            authenticated = self._authenticated_sequence_args(sequence)
            duplicate = self._assert_rejected_before_model_or_prompt_open(
                directory,
                authenticated + (QUALIFICATION_SEQUENCE_OPTION, str(sequence)),
            )
            self.assertRegex(duplicate.stderr.lower(), r"duplicate|once|only")

            empty = self._assert_rejected_before_model_or_prompt_open(
                directory,
                (
                    QUALIFICATION_SEQUENCE_OPTION,
                    "",
                    QUALIFICATION_MANIFEST_SHA256_OPTION,
                    "a" * 64,
                    QUALIFICATION_SEQUENCE_SHA256_OPTION,
                    "b" * 64,
                ),
            )
            self.assertRegex(empty.stderr.lower(), r"empty|requires|path|invalid")

    def test_cpu_and_prompt_file_are_explicitly_incompatible_with_qualification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory_name:
            directory = Path(directory_name)
            sequence = self._generate_sequence(directory)
            for extra, pattern in (
                (("--cpu",), r"cpu|backend|qualification"),
                (("--prompt-file", MISSING_PROMPT), r"prompt|qualification|sequence"),
            ):
                with self.subTest(extra=extra):
                    result = self._invoke(
                        directory,
                        self._authenticated_sequence_args(sequence),
                        extra,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertNotIn("unknown option", result.stderr.lower())
                    self.assertRegex(result.stderr.lower(), pattern)
                    self.assertNotIn(MISSING_MODEL.lower(), result.stderr.lower())
                    self.assertNotIn(MISSING_PROMPT.lower(), result.stderr.lower())


class BenchQualificationAuthenticatedSequenceSourceContractTest(unittest.TestCase):
    def test_bench_parser_requires_one_nonempty_digest_value_for_each_option(self) -> None:
        parse_body = _function_body(
            BENCH_SOURCE,
            "static bench_config parse_options(int argc, char **argv)",
        )
        for option, field in (
            (
                QUALIFICATION_MANIFEST_SHA256_OPTION,
                "qualification_manifest_sha256",
            ),
            (
                QUALIFICATION_SEQUENCE_SHA256_OPTION,
                "qualification_sequence_sha256",
            ),
        ):
            with self.subTest(option=option):
                marker = f'!strcmp(arg, "{option}")'
                start = parse_body.find(marker)
                self.assertGreaterEqual(start, 0, f"missing parser branch for {option}")
                end = parse_body.find("} else if", start)
                self.assertGreater(end, start, f"unterminated parser branch for {option}")
                branch = parse_body[start:end]
                self.assertIn(f"if (c.{field}_set)", branch)
                self.assertIn("may only be specified once", branch)
                self.assertIn("non-empty", branch)
                self.assertRegex(branch, rf"c\.{field}\s*=")
                self.assertRegex(branch, rf"c\.{field}_set\s*=\s*true")

    def test_bench_wires_both_trusted_digest_fields_without_json_sequence_parsing(self) -> None:
        source = _c_code_view(BENCH_SOURCE)
        self.assertIn("ds4_bench_sequence_parse_file_trusted(", source)
        self.assertIn("qualification_manifest_sha256", source)
        self.assertIn("qualification_sequence_sha256", source)
        self.assertNotIn("json_parse", source)

    def test_trusted_parser_is_declared_by_public_sequence_header(self) -> None:
        header = (ROOT / "ds4_bench_sequence.h").read_text(encoding="utf-8")
        self.assertIn("sequence_sha256", header)
        self.assertIn("ds4_bench_sequence_parse_file_trusted(", header)


class BenchQualificationLifecycleSourceContractTest(unittest.TestCase):
    """Keep only the engine/session ownership guard in source inspection."""

    def test_one_engine_encloses_qualification_runner_with_fresh_sessions(self) -> None:
        functions = _function_bodies(BENCH_SOURCE)
        runners: list[tuple[str, str, str]] = []
        for name, body in functions:
            loop = _loop_body_with_four_repetitions(body)
            if loop is None:
                continue
            _header, loop_body = loop
            code = _c_code_view(loop_body)
            if "ds4_session_create(" in code and "ds4_session_free(" in code:
                runners.append((name, body, loop_body))
        self.assertEqual(
            len(runners),
            1,
            "qualification runner must create and free a fresh session in each 0..3 repetition",
        )
        runner_name, _runner_body, loop_body = runners[0]
        loop_code = _c_code_view(loop_body)
        self.assertRegex(loop_code, r"\brepetition_index\b")
        self.assertIn("ds4_session_create(", loop_code)
        self.assertIn("ds4_session_free(", loop_code)

        main = _c_code_view(_function_body(BENCH_SOURCE, "int main(int argc, char **argv)"))
        opens = [
            position
            for marker in (
                "ds4_engine_create_with_gpu_config(",
                "ds4_engine_open(",
            )
            for position in (main.find(marker),)
            if position >= 0
        ]
        self.assertTrue(opens, "benchmark has no engine-open boundary")
        engine_close = main.rfind("ds4_engine_close(")
        self.assertGreaterEqual(engine_close, 0, "benchmark has no engine-close boundary")
        runner_call = re.search(rf"\b{re.escape(runner_name)}\s*\(", main)
        self.assertIsNotNone(runner_call, "main does not call qualification runner")
        assert runner_call is not None
        self.assertLess(min(opens), runner_call.start())
        self.assertLess(runner_call.start(), engine_close)


if __name__ == "__main__":
    unittest.main(verbosity=2)
