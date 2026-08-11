#!/usr/bin/env python3
"""Contract tests for the Laguna token-513 Poolside capture recipe."""

from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "tests/oracle-producers/laguna-c7/"
    "poolside-token513-layer1-capture.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoolsideToken513CaptureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_identity_and_exact_invocation_are_pinned(self) -> None:
        manifest = self.manifest
        self.assertEqual(
            manifest["schema"],
            "laguna-poolside-token513-layer1-capture/v1",
        )
        self.assertEqual(
            manifest["poolside"]["commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(
            manifest["model"],
            {
                "bytes": 68248759648,
                "sha256": (
                    "e163b2c98908809a71245d6bb68b2226994d9969cb2a438e"
                    "ccb72196a1c4147a"
                ),
            },
        )
        self.assertEqual(manifest["capture"]["prefix_tokens"], 512)
        self.assertEqual(manifest["capture"]["resume_token"], 3612)
        self.assertEqual(manifest["capture"]["resume_position"], 512)
        self.assertEqual(manifest["capture"]["logical_token_number"], 513)
        self.assertEqual(manifest["capture"]["layer"], 1)

        run = manifest["run"]
        self.assertEqual(run["environment"]["LD_LIBRARY_PATH"], "{poolside_build}/bin")
        self.assertEqual(
            run["argv"],
            [
                "{probe_binary}",
                "--model",
                "{model}",
                "--tokens",
                "{prefix_i32}",
                "--out",
                "{empty_output_dir}",
                "--flash-attn",
                "auto",
                "--detail-layer",
                "1",
                "--token-count",
                "1",
            ],
        )

    def test_prefix_recipe_reconstructs_the_captured_binary(self) -> None:
        prefix = self.manifest["prefix"]
        ids = (
            prefix["pattern_ids"] * prefix["full_repetitions"]
            + prefix["tail_ids"]
        )
        self.assertEqual(len(ids), prefix["count"])
        payload = struct.pack(f"<{len(ids)}i", *ids)
        self.assertEqual(len(payload), prefix["bytes"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), prefix["sha256"])
        self.assertEqual(
            prefix["sha256"],
            "569aa6394783e0f17558db421ba26480d7a530d44dd2219bc"
            "c9aac2c09a3b559",
        )

    def test_tracked_capture_assets_are_content_addressed(self) -> None:
        for name, entry in self.manifest["producer"].items():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.stat().st_size, entry["bytes"], name)
            self.assertEqual(sha256(path), entry["sha256"], name)

    def test_execution_order_outputs_include_l2_and_down_input(self) -> None:
        outputs = self.manifest["outputs"]["layer_1_execution_order"]
        callbacks = [entry["callback"] for entry in outputs]
        self.assertEqual(
            callbacks,
            [
                "ffn_inp-1",
                "ffn_moe_logits-1",
                "ffn_moe_topk-1",
                "ffn_moe_weights_scaled-1",
                "ffn_moe_gate-1",
                "ffn_moe_up-1",
                "ffn_moe_swiglu-1",
                "ffn_moe_col_l2-1",
                "ffn_moe_down_input-1",
                "ffn_moe_down-1",
                "ffn_moe_weighted-1",
                "ffn_moe_out-1",
                "ffn_shexp-1",
                "ffn_out-1",
            ],
        )
        by_callback = {entry["callback"]: entry for entry in outputs}
        self.assertEqual(by_callback["ffn_moe_col_l2-1"]["bytes"], 40)
        self.assertEqual(by_callback["ffn_moe_down_input-1"]["bytes"], 40960)
        self.assertEqual(by_callback["ffn_moe_down_input-1"]["dtype"], "f32-le")
        self.assertFalse(self.manifest["capture"]["q8_1_down_input_exposed"])


if __name__ == "__main__":
    unittest.main()
