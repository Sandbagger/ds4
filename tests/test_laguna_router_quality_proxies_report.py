#!/usr/bin/env python3
"""Consistency contract for the Laguna router and quality-proxy verdict."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "tests/oracle-producers/laguna-c7/token513-router-quality-proxies.json"
)
REPORT = (
    ROOT
    / "docs/superpowers/plans/2026-08-11-laguna-router-quality-proxies.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LagunaRouterQualityProxiesReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.report = REPORT.read_text(encoding="utf-8")

    def test_router_cause_and_decision_are_structured(self) -> None:
        self.assertEqual(
            self.result["schema"], "laguna-router-quality-proxies/v1"
        )
        router = self.result["router_microscope"]
        self.assertEqual(router["ds4_bits"], "0xbea377b8")
        self.assertEqual(router["poolside_bits"], "0xbea377ba")
        self.assertEqual(router["direct_ds4_capture_bits"], router["ds4_bits"])
        self.assertEqual(
            router["direct_poolside_capture_bits"], router["poolside_bits"]
        )
        self.assertEqual(router["cause"], "f32_reduction_topology")
        self.assertEqual(router["causal_scope"], "captured_row_0_scalar")
        self.assertLess(router["ds4_abs_error_vs_fp64"], router["poolside_abs_error_vs_fp64"])
        self.assertFalse(router["production_change_recommended"])

    def test_quality_proxy_is_paired_and_does_not_overclaim(self) -> None:
        quality = self.result["quality_proxy"]
        self.assertEqual(quality["cases"], 100)
        self.assertEqual(quality["scored_tokens"], 2342)
        self.assertEqual(quality["unchanged_prefill_control_positions"], 100)
        self.assertEqual(quality["candidate_sensitive_scored_positions"], 2242)
        self.assertEqual(quality["serial"]["first_token_matches"], 90)
        self.assertEqual(quality["poolside_reduction"]["first_token_matches"], 90)
        self.assertFalse(quality["api_logprobs"]["available"])
        self.assertGreater(
            quality["poolside_reduction"]["average_nll"],
            quality["serial"]["average_nll"],
        )
        self.assertFalse(self.result["evaluation_scope"]["planner_eval_available"])
        self.assertFalse(self.result["evaluation_scope"]["planner_claim_supported"])
        self.assertFalse(
            self.result["evaluation_scope"]["laguna_vs_flash_claim_supported"]
        )
        self.assertFalse(
            self.result["evaluation_scope"]["coding_task_verifier_used"]
        )

    def test_capability_proxy_is_bound_to_the_pinned_sequence(self) -> None:
        capability = self.result["capability_proxy"]
        self.assertEqual(capability["case_sequence"], [1, 2, 3, 76])
        self.assertEqual(capability["requested_cases"], 4)
        self.assertEqual(capability["reported_suite_denominator"], 92)
        self.assertEqual(capability["not_run"], 88)
        self.assertEqual(capability["repetitions_per_mode"], 1)
        self.assertFalse(
            capability["comparison"]["within_mode_output_reproducibility_control"]
        )
        self.assertIn("serial", capability)
        self.assertIn("poolside_reduction", capability)

    def test_report_is_bound_and_keeps_the_default_unchanged(self) -> None:
        self.assertIn(sha256(RESULT), self.report)
        self.assertIn("model-quality proxy", self.report.lower())
        self.assertIn("capability proxy", self.report.lower())
        self.assertIn("not a planner/orchestrator evaluation", self.report.lower())
        self.assertEqual(self.result["decision"]["default"], "serial")
        self.assertEqual(self.result["decision"]["experimental"], "poolside")
        self.assertFalse(self.result["decision"]["promote_poolside_to_default"])


if __name__ == "__main__":
    unittest.main()
