#!/usr/bin/env python3
"""Consistency contract for the feature-gated Poolside MMVQ experiment."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "tests/oracle-producers/laguna-c7/token513-poolside-mmvq-experiment.json"
)
REPORT = (
    ROOT
    / "docs/superpowers/plans/2026-08-11-laguna-poolside-mmvq-experiment.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LagunaPoolsideMmvqExperimentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.report = REPORT.read_text(encoding="utf-8")

    def test_feature_gate_and_scope_are_explicit(self) -> None:
        self.assertEqual(
            self.result["schema"], "laguna-poolside-mmvq-experiment/v1"
        )
        feature = self.result["feature"]
        self.assertEqual(feature["environment"], "DS4_MM_VQ_REDUCTION")
        self.assertEqual(feature["experimental_value"], "poolside")
        self.assertEqual(feature["default"], "serial")
        self.assertEqual(feature["tokens"], 1)
        self.assertEqual(feature["weight_type"], "Q4_K")
        self.assertEqual(feature["activation_type"], "Q8_1")
        self.assertEqual(feature["laguna_shape"]["experts_total"], 256)
        self.assertEqual(feature["laguna_shape"]["experts_used"], 10)

    def test_expert_parity_and_downstream_limits_are_durable(self) -> None:
        numerical = self.result["numerical"]
        for stage in ("gate", "up", "swiglu", "col_l2", "down_input", "down"):
            with self.subTest(stage=stage):
                self.assertEqual(
                    numerical["poolside_reduction_vs_poolside"][stage][
                        "mismatch_count"
                    ],
                    0,
                )
        self.assertLess(
            numerical["poolside_reduction_vs_poolside"]["layer_1"]["l2_delta"],
            numerical["serial_vs_poolside"]["layer_1"]["l2_delta"],
        )
        self.assertGreater(
            numerical["poolside_reduction_vs_poolside"]["final_logits"]["l2_delta"],
            numerical["serial_vs_poolside"]["final_logits"]["l2_delta"],
        )
        self.assertEqual(
            numerical["unchanged_router"]["selected_id_mismatch_count"], 0
        )

    def test_behavior_and_performance_are_measured_not_inferred(self) -> None:
        behavior = self.result["behavior"]
        self.assertEqual(behavior["steps"], 32)
        self.assertIn("serial", behavior)
        self.assertIn("poolside_reduction", behavior)
        self.assertIn("reference", behavior)

        performance = self.result["performance"]
        self.assertEqual(performance["paired_runs"], 3)
        self.assertGreaterEqual(
            performance["steady_decode"]["candidate_to_serial_ratio"], 0.98
        )
        self.assertIn("nsys", performance)
        self.assertFalse(performance["hardware_counters"]["available"])

    def test_report_is_bound_and_keeps_router_and_gb10_caveats(self) -> None:
        self.assertIn(sha256(RESULT), self.report)
        self.assertIn("router", self.report.lower())
        self.assertIn("GB10", self.report)
        self.assertIn("experimental", self.report.lower())
        self.assertIn("four-warp", self.report.lower())


if __name__ == "__main__":
    unittest.main()
