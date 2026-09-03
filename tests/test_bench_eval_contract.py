#!/usr/bin/env python3
"""Host-only RED contract for Task 19 Step 1 benchmark evidence.

This revision has no model-free ``ds4-bench`` stdio/test seam.  The contract
therefore stays bounded to the public CLI/output source: it checks that the
qualification option is validated before engine creation and that the
qualification JSONL writer carries the required closed fields.  It does not
run a model, allocate a GPU, open a network socket, or inspect live output.
The later model-free executable hook remains the authoritative green target;
this source contract only makes the missing public behavior fail loudly first.
"""

from __future__ import annotations

import re
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
UINT64_SAMPLE_FIELDS = (
    "snapshot_seq",
    "monotonic_ns",
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
)


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


def _c_json_view(source: str) -> str:
    """Make escaped C JSON keys searchable as their wire spelling."""

    return source.replace(r'\"', '"')


def _qualification_option_branch(parser: str) -> str:
    start = parser.find(QUALIFICATION_SEQUENCE_OPTION)
    if start < 0:
        return ""
    end = parser.find("} else if", start)
    if end < 0:
        end = parser.find("} else {", start)
    return parser[start:] if end < 0 else parser[start:end]


def _engine_open_positions(source: str) -> list[int]:
    return [
        position
        for marker in (
            "ds4_engine_create_with_gpu_config(",
            "ds4_engine_open(",
        )
        for position in [source.find(marker)]
        if position >= 0
    ]


class BenchQualificationSequenceContractTest(unittest.TestCase):
    """RED source contract for the qualification-only sequence frontend."""

    def test_sequence_is_rejected_before_any_model_engine_allocation(self) -> None:
        parser = _function_body(BENCH_SOURCE, "static bench_config parse_options")
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

        main = _function_body(BENCH_SOURCE, "int main(int argc, char **argv)")
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
        self.assertLess(
            main.find("parse_options(argc, argv)"),
            min(opens),
            "qualification input must be validated before model allocation",
        )

        preallocation = BENCH_SOURCE[: min(position for position in _engine_open_positions(BENCH_SOURCE))]
        for marker in (
            "qualification_sequence",
            "manifest",
            "prompt_id",
            "prompt_tokens",
            "order",
            "input",
            "repetition_index",
        ):
            self.assertIn(
                marker,
                preallocation.lower(),
                f"qualification preflight does not validate sequence {marker}",
            )
        self.assertRegex(
            preallocation,
            r"(?:==|!=|<|>)\s*4\b|\b4\s*(?:repetitions|requests)",
            "qualification sequence must contain exactly four repetitions",
        )

    def test_qualification_sequence_keeps_one_engine_for_cold_plus_three_warm(self) -> None:
        view = _c_json_view(BENCH_SOURCE)
        self.assertTrue(
            "repetition_index" in view,
            "qualification lifecycle records do not bind a repetition index",
        )
        self.assertRegex(
            BENCH_SOURCE,
            r"(?:for|while)\s*\([^\n]*repetition|repetition[^\n]*4",
            "qualification sequence has no explicit four-repetition loop",
        )
        engine_open = min(_engine_open_positions(BENCH_SOURCE))
        engine_close = BENCH_SOURCE.rfind("ds4_engine_close(")
        repetition = BENCH_SOURCE.find("repetition_index")
        self.assertGreaterEqual(engine_close, 0, "benchmark has no engine close boundary")
        self.assertLess(
            engine_open,
            repetition,
            "the cold and warm sequence must run after one engine is opened",
        )
        self.assertLess(
            repetition,
            engine_close,
            "the engine must stay alive through all four repetitions",
        )

    def test_every_qualification_sample_has_unambiguous_runtime_evidence(self) -> None:
        view = _c_json_view(BENCH_SOURCE)
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
        # KV.  The machine-readable benchmark header must use the unambiguous
        # name instead.  A separate JSONL sequence writer may coexist with the
        # ordinary CSV path, so only reject the ambiguous legacy label here.
        headers = re.findall(
            r"fprintf\s*\(\s*out\s*,\s*\"([^\"]*ctx_tokens[^\"]*)\"",
            BENCH_SOURCE,
        )
        self.assertTrue(headers, "benchmark has no machine-readable sample header")
        self.assertFalse(
            any("kvcache_bytes" in header for header in headers),
            "ambiguous kvcache_bytes remains in the machine-readable header",
        )

        # These are the existing production accounting seams.  A benchmark
        # record must report the runtime tracker and request metrics, not infer
        # live allocation from unrelated before/after snapshots.
        for marker in (
            "ds4_engine_runtime_snapshot(",
            "ds4_runtime_request_metrics_json(",
            "ds4_engine_laguna_external_checkpoint(",
            "ds4_session_payload_bytes(",
        ):
            self.assertIn(marker, BENCH_SOURCE, f"benchmark does not use {marker}")

        # Runtime-owned uint64 measurements use the existing canonical decimal
        # JSON writers; they must not be emitted as floating-point numbers.
        for field in UINT64_SAMPLE_FIELDS:
            position = view.find(f'"{field}"')
            self.assertGreaterEqual(position, 0)
            self.assertRegex(
                view[position : position + 512],
                r"(?:PRIu64|%llu|json_u64)",
                f"{field} is not emitted through a uint64 JSON writer",
            )

    def test_lifecycle_jsonl_distinguishes_events_and_flushes_each_record(self) -> None:
        view = _c_json_view(BENCH_SOURCE)
        positions: list[int] = []
        for event in LIFECYCLE_EVENTS:
            marker = f'"{event}"'
            position = view.find(marker)
            self.assertGreaterEqual(
                position,
                0,
                f"qualification JSONL has no {event} lifecycle record",
            )
            positions.append(position)
            self.assertRegex(
                view[position : position + 4096],
                r"fflush\s*\(",
                f"{event} record is not flushed before the child can block",
            )
        self.assertEqual(
            positions,
            sorted(positions),
            "qualification lifecycle records are not ordered cold request, first token, complete",
        )
        self.assertIn(
            QUALIFICATION_SEQUENCE_OPTION,
            view,
            "lifecycle JSONL is not gated behind the qualification-only option",
        )

        terminal = view.find('"terminal_status"')
        self.assertGreaterEqual(
            terminal,
            0,
            "request_complete is missing its terminal_status field",
        )
        accepted, first, complete = positions
        self.assertLess(
            first,
            terminal,
            "terminal_status must belong to request_complete, not an earlier milestone",
        )
        self.assertGreaterEqual(
            terminal,
            complete,
            "terminal_status must be emitted only with request_complete",
        )
        self.assertNotIn(
            '"terminal_status"',
            view[accepted:first],
            "request_accepted must not claim terminal status",
        )
        self.assertNotIn(
            '"terminal_status"',
            view[first:complete],
            "first_token must not claim terminal status",
        )


if __name__ == "__main__":
    unittest.main()
