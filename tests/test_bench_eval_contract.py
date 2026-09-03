#!/usr/bin/env python3
"""Host-only RED contract for Task 19 Step 1 benchmark evidence.

The current revision has no model-free lifecycle stdio seam.  The qualification
preflight cases therefore build and invoke the real host-only ``ds4-bench``
CLI, while the lifecycle checks stay bounded to a concrete emitter/loop source
seam that the CLI must call.  They do not run a model, allocate a GPU, open a
network socket, or inspect live output.  The later model-free executable hook
remains the authoritative green target; this source contract only makes the
missing public behavior fail loudly first.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_SOURCE = (ROOT / "ds4_bench.c").read_text(encoding="utf-8")
QUALIFICATION_SEQUENCE_OPTION = "--qualification-sequence"
LIFECYCLE_EVENTS = ("request_accepted", "first_token", "request_complete")
REQUIRED_SAMPLE_FIELDS = (
    "schema",
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
)
EXTERNAL_ATTRIBUTION_FIELDS = (
    "model_source_resident",
    "host_library_unattributed",
    "cuda_library_unattributed",
    "unrelated_process_inventory_stable",
)
# These values are uint64-like on the wire and must be JSON strings.  Prefill
# row counts and repetition_index are intentionally excluded: the existing
# ds4.runtime/v1 ABI defines those as JSON integers.
UINT64_SAMPLE_FIELDS = (
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
INTEGER_SAMPLE_FIELDS = (
    "configured_prefill_rows",
    "allocated_prefill_rows",
    "repetition_index",
)
MISSING_MODEL = "/definitely/missing/task19-model.gguf"
MISSING_PROMPT = "/definitely/missing/task19-prompt.txt"


def _function_body(source: str, signature: str) -> str:
    """Return one braced C function body without requiring a compiler."""

    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing source function {signature}")
    brace = source.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing function body for {signature}")
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"unterminated function {signature}")


def _function_bodies(source: str) -> list[tuple[str, str]]:
    """Return top-level-looking C function names and bodies.

    This deliberately only supports the small source seam used here.  It is
    not a C parser and is never used to infer behavior from unrelated text.
    """

    pattern = re.compile(
        r"(?ms)^(?:static\s+)?[A-Za-z_][^;{}]*?\b"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*?\)\s*\{"
    )
    functions: list[tuple[str, str]] = []
    keywords = {"if", "for", "while", "switch"}
    for match in pattern.finditer(source):
        name = match.group("name")
        if name in keywords:
            continue
        brace = match.end() - 1
        depth = 0
        for index in range(brace, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    functions.append((name, source[brace + 1 : index]))
                    break
        else:
            raise AssertionError(f"unterminated function {name}")
    return functions


def _c_code_view(source: str) -> str:
    """Return a bounded C view with comments removed and JSON unescaped."""

    without_comments = re.sub(
        r"(?s)/\*.*?\*/|//[^\r\n]*",
        "",
        source,
    )
    return without_comments.replace(r'\"', '"')


def _qualification_option_branch(parser: str) -> str:
    start = parser.find(QUALIFICATION_SEQUENCE_OPTION)
    if start < 0:
        return ""
    end = parser.find("} else if", start)
    if end < 0:
        end = parser.find("} else {", start)
    return parser[start:] if end < 0 else parser[start:end]


def _qualification_emitters() -> list[tuple[str, str]]:
    """Find concrete functions that own all lifecycle literals and flushing."""

    emitters = []
    for name, body in _function_bodies(BENCH_SOURCE):
        view = _c_code_view(body)
        if (
            all(f'"{event}"' in view for event in LIFECYCLE_EVENTS)
            and re.search(r"\b(?:fprintf|fputs|fwrite)\s*\(", body)
            and re.search(r"\bfflush\s*\(", body)
        ):
            emitters.append((name, body))
    return emitters


class BenchQualificationSequenceContractTest(unittest.TestCase):
    """RED source contract for the qualification-only sequence frontend."""

    def test_sequence_parser_precedes_engine_open_in_main(self) -> None:
        parser = _c_code_view(
            _function_body(BENCH_SOURCE, "static bench_config parse_options")
        )
        self.assertTrue(
            QUALIFICATION_SEQUENCE_OPTION in parser,
            "ds4-bench has no qualification-only --qualification-sequence FILE option",
        )
        branch = _qualification_option_branch(parser)
        self.assertTrue(branch, "qualification-sequence parser branch is missing")
        self.assertRegex(
            branch,
            r"(?:exit\s*\(\s*2\s*\)|return\s+2|invalid|reject)",
            "invalid qualification sequences must be rejected by the CLI",
        )

        main = _c_code_view(
            _function_body(BENCH_SOURCE, "int main(int argc, char **argv)")
        )
        opens = [
            position
            for marker in (
                "ds4_engine_create_with_gpu_config(",
                "ds4_engine_open(",
            )
            for position in [main.find(marker)]
            if position >= 0
        ]
        self.assertTrue(opens, "benchmark has no engine-open boundary")
        parse_position = main.find("parse_options(argc, argv)")
        self.assertGreaterEqual(parse_position, 0, "main does not parse benchmark options")
        self.assertLess(
            parse_position,
            min(opens),
            "qualification input must be validated before model allocation",
        )


class BenchQualificationPreflightCliTest(unittest.TestCase):
    """Exercise the real CLI's model-free invalid-sequence boundary."""

    bench = ROOT / "ds4-bench"

    @classmethod
    def setUpClass(cls) -> None:
        # ``cpu`` is the repository's normal host-only build target.  Build once
        # for this class, then every case reaches parse/preflight only.
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

    def _invoke(
        self,
        temporary: Path,
        sequence_argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["DS4_LOCK_FILE"] = str(temporary / "private-task19.lock")
        return subprocess.run(
            [
                str(self.bench),
                *sequence_argv,
                "--model",
                MISSING_MODEL,
                "--prompt-file",
                MISSING_PROMPT,
                "--cpu",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def _assert_sequence_rejected_before_model_or_prompt_open(
        self,
        temporary: Path,
        sequence_argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        result = self._invoke(temporary, sequence_argv)
        stderr = result.stderr.lower()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn(
            "unknown option",
            stderr,
            "the planned option must not take the current unknown-option path",
        )
        self.assertRegex(
            stderr,
            r"qualification[- ]sequence",
            "invalid sequence must produce a qualification-sequence diagnostic",
        )
        self.assertNotIn("cannot open model", stderr)
        self.assertNotIn("failed to open model", stderr)
        self.assertNotIn("failed to open", stderr)
        self.assertNotIn(MISSING_PROMPT.lower(), stderr)
        return result

    def test_malformed_and_empty_manifest_slice_reject_before_model_open(self) -> None:
        for label, payload in (("malformed", "{\n"), ("empty-object", "{}\n")):
            with self.subTest(sequence=label), tempfile.TemporaryDirectory(
                prefix="task19-sequence-"
            ) as directory:
                temporary = Path(directory)
                sequence = temporary / f"{label}.json"
                sequence.write_text(payload, encoding="utf-8")
                self._assert_sequence_rejected_before_model_or_prompt_open(
                    temporary,
                    (QUALIFICATION_SEQUENCE_OPTION, str(sequence)),
                )

    def test_duplicate_and_empty_sequence_options_reject_before_model_open(self) -> None:
        with tempfile.TemporaryDirectory(prefix="task19-sequence-") as directory:
            temporary = Path(directory)
            sequence = temporary / "duplicate.json"
            sequence.write_text("{}\n", encoding="utf-8")
            duplicate = self._assert_sequence_rejected_before_model_or_prompt_open(
                temporary,
                (
                    QUALIFICATION_SEQUENCE_OPTION,
                    str(sequence),
                    QUALIFICATION_SEQUENCE_OPTION,
                    str(sequence),
                ),
            )
            self.assertRegex(
                duplicate.stderr.lower(),
                r"(?:duplicate|once|only)",
                "duplicate qualification sequence options need a stable diagnostic",
            )

            empty = self._assert_sequence_rejected_before_model_or_prompt_open(
                temporary,
                (QUALIFICATION_SEQUENCE_OPTION, ""),
            )
            self.assertRegex(
                empty.stderr.lower(),
                r"(?:empty|requires|path|invalid)",
                "an empty qualification sequence path needs a stable diagnostic",
            )


class BenchQualificationEvidenceContractTest(unittest.TestCase):
    def test_every_qualification_sample_has_the_closed_runtime_field_set(self) -> None:
        view = _c_code_view(BENCH_SOURCE)
        missing = [
            field
            for field in REQUIRED_SAMPLE_FIELDS
            if f'"{field}"' not in view
        ]
        self.assertEqual(
            missing,
            [],
            "qualification samples are missing closed fields: " + ", ".join(missing),
        )
        missing_external = [
            field
            for field in EXTERNAL_ATTRIBUTION_FIELDS
            if f'"{field}"' not in view
        ]
        self.assertEqual(
            missing_external,
            [],
            "external_attribution is missing closed fields: "
            + ", ".join(missing_external),
        )

        # The legacy CSV label described a serialized snapshot payload as live
        # KV.  It must not remain in the machine-readable benchmark header.
        headers = re.findall(
            r'fprintf\s*\(\s*out\s*,\s*"([^"]*ctx_tokens[^"]*)"',
            BENCH_SOURCE,
        )
        self.assertTrue(headers, "benchmark has no machine-readable sample header")
        self.assertFalse(
            any("kvcache_bytes" in header for header in headers),
            "ambiguous kvcache_bytes remains in the machine-readable header",
        )

        # Runtime-owned accounting is read from the existing APIs, not inferred
        # by differencing unrelated snapshots.
        for marker in (
            "ds4_engine_runtime_snapshot(",
            "ds4_runtime_request_metrics_json(",
            "ds4_engine_laguna_external_checkpoint(",
            "ds4_session_payload_bytes(",
        ):
            self.assertIn(marker, BENCH_SOURCE, f"benchmark does not use {marker}")

        # Byte/time fields are canonical decimal JSON strings.  Row counts and
        # repetition_index intentionally use the integer ABI instead.
        for field in UINT64_SAMPLE_FIELDS:
            position = view.find(f'"{field}"')
            self.assertGreaterEqual(position, 0)
            self.assertRegex(
                view[position : position + 512],
                r"(?:PRIu64|%llu|json_u64)",
                f"{field} is not emitted through a uint64 JSON writer",
            )
        for field in INTEGER_SAMPLE_FIELDS:
            position = view.find(f'"{field}"')
            self.assertGreaterEqual(position, 0)
            self.assertRegex(
                view[position : position + 512],
                r"(?:PRIu32|%u|%d|json_u32|integer)",
                f"{field} is not emitted as an integer JSON value",
            )

    def test_lifecycle_emitter_and_loop_are_concrete_but_provisional(self) -> None:
        """Bind provisional source checks to live call seams, not dead text."""

        emitters = _qualification_emitters()
        self.assertEqual(
            len(emitters),
            1,
            "qualification lifecycle needs one concrete emitter with all events and fflush",
        )
        emitter_name, emitter_body = emitters[0]
        emitter_view = _c_code_view(emitter_body)
        for event in LIFECYCLE_EVENTS:
            self.assertIn(f'"{event}"', emitter_view)
        self.assertRegex(emitter_body, r"\bfflush\s*\(")
        last_event = max(
            emitter_view.find(f'"{event}"') for event in LIFECYCLE_EVENTS
        )
        first_flush = emitter_view.find("fflush")
        self.assertGreater(
            first_flush,
            last_event,
            "the concrete emitter must flush after constructing each record",
        )

        # terminal_status belongs only to request_complete.  This source check
        # is intentionally narrow; the future model-free hook must validate the
        # same rule against actual JSONL records.
        complete = emitter_view.find('"request_complete"')
        terminal = emitter_view.find('"terminal_status"')
        self.assertGreaterEqual(complete, 0)
        self.assertGreaterEqual(terminal, 0)
        self.assertGreater(
            terminal,
            complete,
            "terminal_status must be gated by request_complete",
        )
        for earlier in ("request_accepted", "first_token"):
            self.assertNotIn(
                '"terminal_status"',
                emitter_view[emitter_view.find(f'"{earlier}"') : complete],
                f"{earlier} must not claim terminal_status",
            )

        main = _function_body(BENCH_SOURCE, "int main(int argc, char **argv)")
        main_view = _c_code_view(main)
        emitter_call = re.search(
            rf"\b{re.escape(emitter_name)}\s*\(", main_view
        )
        self.assertIsNotNone(
            emitter_call,
            "main does not call the qualification lifecycle emitter",
        )

        loops = []
        for name, body in _function_bodies(BENCH_SOURCE):
            code = _c_code_view(body)
            if not re.search(
                r"for\s*\(\s*(?:int\s+)?repetition_index\s*=\s*0\s*;"
                r"[^;]*<\s*4\b",
                code,
            ):
                continue
            if re.search(rf"\b{re.escape(emitter_name)}\s*\(", code):
                loops.append((name, body))
        self.assertEqual(
            len(loops),
            1,
            "qualification loop must emit cold plus three warm records through the live emitter",
        )
        loop_name, loop_body = loops[0]
        self.assertRegex(
            _c_code_view(loop_body),
            r"repetition_index",
            "qualification loop does not bind repetition_index 0..3",
        )
        if loop_name != "main":
            loop_call = re.search(
                rf"\b{re.escape(loop_name)}\s*\(",
                main_view,
            )
            self.assertIsNotNone(
                loop_call,
                "main does not call the concrete qualification loop",
            )
            open_positions = [
                main_view.find(marker)
                for marker in (
                    "ds4_engine_create_with_gpu_config(",
                    "ds4_engine_open(",
                )
                if main_view.find(marker) >= 0
            ]
            engine_close = main_view.rfind("ds4_engine_close(")
            self.assertTrue(open_positions, "main has no engine-open boundary")
            self.assertGreaterEqual(engine_close, 0, "main has no engine-close boundary")
            assert loop_call is not None
            self.assertLess(min(open_positions), loop_call.start())
            self.assertLess(loop_call.start(), engine_close)


if __name__ == "__main__":
    unittest.main()
