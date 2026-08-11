#!/usr/bin/env python3
"""Consistency contract for the durable Laguna token-513 diagnosis."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "tests/oracle-producers/laguna-c7"
RUN_MANIFEST = CAPTURE_DIR / "token513-layer1-run.json"
COMPARISON = CAPTURE_DIR / "token513-layer1-comparison.json"
REPORT = (
    ROOT
    / "docs/superpowers/plans/2026-08-11-laguna-token513-direct-capture.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LagunaToken513ComparisonReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_manifest = json.loads(
            RUN_MANIFEST.read_text(encoding="utf-8")
        )
        cls.comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
        cls.report = REPORT.read_text(encoding="utf-8")

    def test_comparison_is_bound_to_complete_run_manifest(self) -> None:
        binding = self.comparison["run_manifest"]
        self.assertTrue(binding["captures_bound"])
        self.assertEqual(binding["sha256"], sha256(RUN_MANIFEST))
        self.assertEqual(binding["model"], self.run_manifest["model"])
        self.assertEqual(binding["prefix"], self.run_manifest["prefix"])
        self.assertEqual(binding["resume_token"], 3612)
        controls = binding["controls"]
        self.assertEqual(controls["release_vs_hook_null"], "bit-exact")
        self.assertEqual(controls["hook_null_vs_hook_active"], "bit-exact")
        self.assertEqual(
            controls["logits_sha256"],
            "e837bc5b6ad4dbfe74e17a1d4e7552eaae6756c46f30c4548c00d63c6e7d21c2",
        )

    def test_execution_order_and_first_mismatches_are_durable(self) -> None:
        stages = self.comparison["stages"]
        expected_counts = {
            "ffn_norm": (3072, 0),
            "router_logits": (92, 164),
            "selected": (10, 0),
            "router_weights": (8, 2),
            "gate": (1557, 8683),
            "up": (1465, 8775),
            "swiglu": (750, 9490),
            "col_l2": (7, 3),
            "down_input": (797, 9443),
            "down": (4384, 26336),
            "weighted": (4222, 26498),
            "routed_sum": (382, 2690),
            "shared": (3072, 0),
            "combined": (664, 2408),
        }
        for stage, (equal, unequal) in expected_counts.items():
            with self.subTest(stage=stage):
                self.assertEqual(stages[stage]["equal_count"], equal)
                self.assertEqual(stages[stage]["mismatch_count"], unequal)

        self.assertEqual(
            self.comparison["first_overall_mismatch"],
            {
                "stage": "router_logits",
                "expert_index": 0,
                "poolside_bits": "0xbea377ba",
                "ds4_bits": "0xbea377b8",
            },
        )
        self.assertEqual(
            self.comparison["first_routed_expert_mismatch"],
            {
                "stage": "gate",
                "slot": 0,
                "expert": 144,
                "poolside_expert": 144,
                "ds4_expert": 144,
                "row": 0,
                "poolside_bits": "0xbdcdf5ed",
                "ds4_bits": "0xbdcdf5ef",
            },
        )

    def test_router_and_counterfactual_details_are_not_prose_only(self) -> None:
        routing = self.comparison["routing"]
        self.assertEqual(
            routing["selected_ids"]["poolside"],
            [144, 15, 106, 165, 240, 108, 253, 102, 34, 226],
        )
        self.assertTrue(routing["selected_ids"]["exact"])
        self.assertEqual(
            [entry["slot"] for entry in routing["weight_mismatches"]],
            [0, 7],
        )

        decomposition = self.comparison["counterfactual_decomposition"]
        self.assertTrue(decomposition["poolside_replay"]["weighted_exact"])
        self.assertTrue(decomposition["poolside_replay"]["routed_sum_exact"])
        self.assertTrue(decomposition["ds4_replay"]["weighted_exact"])
        self.assertTrue(decomposition["ds4_replay"]["routed_sum_exact"])
        self.assertEqual(decomposition["routing_only"]["unequal_rows"], 1851)
        self.assertEqual(decomposition["expert_only"]["unequal_rows"], 2682)
        self.assertEqual(decomposition["combined"]["unequal_rows"], 2690)
        self.assertAlmostEqual(
            decomposition["routing_only"]["l2_delta"],
            4.508155154480043e-08,
        )
        self.assertAlmostEqual(
            decomposition["expert_only"]["l2_delta"],
            1.0126285517335878e-07,
        )

    def test_report_scopes_claims_to_observed_evidence(self) -> None:
        self.assertIn(
            "Status: routed-expert causal diagnosis complete; "
            "router attribution remains open.",
            self.report,
        )
        self.assertIn(sha256(RUN_MANIFEST), self.report)
        self.assertIn(sha256(COMPARISON), self.report)
        self.assertIn(
            "consistent with their differing F32 reduction topologies",
            self.report,
        )
        self.assertIn(
            "Poolside's runtime Q8_1 bytes were not directly observed",
            self.report,
        )
        self.assertNotIn(
            "Thus the router and expert differences are two concrete kernels",
            self.report,
        )


if __name__ == "__main__":
    unittest.main()
