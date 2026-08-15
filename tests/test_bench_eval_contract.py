#!/usr/bin/env python3
"""Source contracts for qualification-safe benchmark and eval evidence."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = (ROOT / "ds4_bench.c").read_text(encoding="utf-8")
EVAL = (ROOT / "ds4_eval.c").read_text(encoding="utf-8")
QUALIFIER = (
    ROOT / "gguf-tools/quality-testing/compact_runtime_qualify.py"
).read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
VECTOR_README = (ROOT / "tests/test-vectors/README.md").read_text(encoding="utf-8")

CASE_IDS = (
    "recNu3MXkvWUzHZr9",
    "001b51d76b4d422988f2c11f104a2c6c",
    "aime2025-01",
    "compsec-076",
)


class BenchQualificationContractTests(unittest.TestCase):
    def test_qualification_sequence_is_explicit_and_manifest_bound(self) -> None:
        self.assertIn('"--qualification-sequence"', BENCH)
        self.assertIn("qualification_sequence_path", BENCH)
        self.assertIn("validate_qualification_sequence", BENCH)
        self.assertIn("repetitions != 4", BENCH)

    def test_qualification_records_have_ordered_lifecycle_milestones(self) -> None:
        for milestone in (
            "request_accepted",
            "first_token",
            "request_complete",
        ):
            self.assertIn(f'"{milestone}"', BENCH, milestone)
        self.assertIn("repetition_index", BENCH)
        self.assertIn("accepted_monotonic_ns", BENCH)
        self.assertIn("fflush", BENCH)

    def test_samples_distinguish_payload_from_live_allocations(self) -> None:
        self.assertNotIn("kvcache_bytes", BENCH)
        for field in (
            "session_payload_bytes",
            "kv_allocated_bytes",
            "configured_prefill_rows",
            "allocated_prefill_rows",
            "expert_cache_bound_bytes",
            "expert_cache_current_bytes",
            "expert_cache_peak_bytes",
            "qualification_total_current_bytes",
            "qualification_total_peak_bytes",
            "model_source_resident_bytes",
            "external_attribution",
            "runtime_snapshot",
            "request_metrics",
            "resident_mode",
        ):
            self.assertIn(field, BENCH, field)
        self.assertIn("ds4_engine_runtime_snapshot", BENCH)
        self.assertIn("ds4_runtime_request_metrics_json", BENCH)


class EvalQualificationContractTests(unittest.TestCase):
    def test_case_selection_uses_stable_ids_only(self) -> None:
        self.assertIn('"--case-id"', EVAL)
        self.assertIn("case_ids", EVAL)
        self.assertIn("duplicate --case-id", EVAL)
        self.assertIn("unknown --case-id", EVAL)
        for case_id in CASE_IDS:
            self.assertIn(case_id, EVAL, case_id)

    def test_machine_records_bind_result_and_runtime_identity(self) -> None:
        self.assertIn('"--result-jsonl"', EVAL)
        for field in (
            "case_id",
            "answer",
            "grade",
            "terminal_status",
            "request_metrics",
            "runtime_snapshot",
            "evidence_sha256",
        ):
            self.assertIn(field, EVAL, field)
        self.assertIn("ds4_engine_runtime_snapshot", EVAL)
        self.assertIn("ds4_runtime_request_metrics_json", EVAL)


class QualificationHarnessContractTests(unittest.TestCase):
    def test_smoke_eval_surface_is_present(self) -> None:
        self.assertRegex(
            QUALIFIER,
            r'add_parser\(\s*"smoke-eval"',
        )
        self.assertIn("run_smoke_eval", QUALIFIER)
        self.assertIn("CASE_IDS", QUALIFIER)
        self.assertIn("resident", QUALIFIER)
        self.assertIn("streamed", QUALIFIER)

    def test_task19_contract_is_in_focused_make_target(self) -> None:
        target = re.search(
            r"^test-laguna-bench-eval-contract:\n(?P<body>(?:\t.*\n)+)",
            MAKEFILE,
            re.MULTILINE,
        )
        self.assertIsNotNone(target)
        self.assertIn("tests/test_bench_eval_contract.py", target.group("body"))

    def test_reference_runbook_pins_curve_and_excludes_service_policy(self) -> None:
        combined = "\n".join((README, CONTRIBUTING, VECTOR_README))
        for value in (
            "32,768", "4,096", "one session slot", "8/12/16-GiB",
            "one cold plus exactly three", "45-minute", "15-minute",
            "recNu3MXkvWUzHZr9", "001b51d76b4d422988f2c11f104a2c6c",
            "aime2025-01", "compsec-076",
        ):
            self.assertIn(value, combined, value)
        for command in ("run", "verify", "publish", "verify-bundle"):
            self.assertIn(f"compact_runtime_qualify.py \\\n+  {command}", README)
        for excluded in (
            "drop_caches", "legacy whole-map", "deprecated expert-count",
            "daemonization", "port selection", "peer eviction", "co-residency",
        ):
            self.assertIn(excluded, combined, excluded)


if __name__ == "__main__":
    unittest.main()
