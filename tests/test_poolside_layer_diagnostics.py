#!/usr/bin/env python3

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "gguf-tools/quality-testing/probe_poolside_laguna_layers.cpp"
COMPARATOR = ROOT / "gguf-tools/quality-testing/compare_laguna_layers.py"
DS4_SOURCE = ROOT / "ds4.c"
WIDTH = 3072
TOKENS = 22
LAYERS = 48
VALUES_PER_LAYER = WIDTH * TOKENS
BYTES_PER_LAYER = VALUES_PER_LAYER * 4
LAYER0_TARGETS = (
    ("attn-norm", "attn_norm-0", 3072),
    ("q-proj", "Qcur-0", 6144),
    ("k-proj", "Kcur-0", 1024),
    ("v-proj", "Vcur-0", 1024),
    ("gate-proj", "attn_gate_proj-0", 48),
    ("q-rope", "Qcur_rope-0", 6144),
    ("k-rope", "Kcur_rope-0", 1024),
    ("attn-gated", "attn_gated-0", 6144),
    ("attn-o-proj", "attn_o_proj-0", 3072),
    ("ffn-inp", "ffn_inp-0", 3072),
    ("ffn-norm", "ffn_norm-0", 3072),
    ("ffn-out", "ffn_out-0", 3072),
)
LAYER0_STAGES = tuple(stage for stage, _, _ in LAYER0_TARGETS)


class PoolsideLayerDiagnosticsTest(unittest.TestCase):
    def test_probe_source_exposes_the_exact_short_prompt_callback_contract(self):
        self.assertTrue(PROBE.is_file(), f"missing diagnostic probe: {PROBE}")
        source = PROBE.read_text(encoding="utf-8")

        for required in (
            "--model",
            "--tokens",
            "--out",
            "--flash-attn",
            "--detail-layer",
            "cb_eval",
            "ggml_backend_tensor_get",
            "ggml_is_contiguous",
            "GGML_TYPE_F32",
            '"embd"',
            '"l_out-"',
            '"h_nextn"',
            '"result_output"',
            "layer-%02d.f32",
            "embd.f32",
            "logits.f32",
            "duplicate",
            "LLAMA_FLASH_ATTN_TYPE_AUTO",
            "LLAMA_FLASH_ATTN_TYPE_DISABLED",
            "context_params.flash_attn_type = options.flash_attn_type",
        ):
            self.assertIn(required, source)

        self.assertRegex(source, r"kWidth\s*=\s*3072")
        self.assertRegex(source, r"kTokens\s*=\s*22")
        self.assertRegex(source, r"kLayers\s*=\s*48")
        for stage, callback, _ in LAYER0_TARGETS:
            callback_base = callback.removesuffix("-0")
            self.assertIn(f'"{callback_base}"', source)
            self.assertIn(f'"{stage}"', source)

    def test_detail_layer_selector_defaults_to_zero_in_both_generators(self):
        probe_source = PROBE.read_text(encoding="utf-8")
        ds4_source = DS4_SOURCE.read_text(encoding="utf-8")

        self.assertRegex(probe_source, r"detail_layer\s*=\s*0")
        self.assertIn("--detail-layer must be 0 or 1", probe_source)
        self.assertIn("detail_head_count", probe_source)
        self.assertRegex(probe_source, r"detail_layer\s*==\s*0\s*\?\s*48\s*:\s*72")
        self.assertIn('"layer-%02d-%s.f32"', probe_source)
        self.assertIn("state.detail_layer", probe_source)

        self.assertIn('getenv("DS4_LAGUNA_DIAG_LAYER")', ds4_source)
        self.assertIn(
            "DS4_LAGUNA_DIAG_LAYER must be 0 or 1",
            ds4_source,
        )
        self.assertIn("laguna_graph_diag_detail_layer", ds4_source)
        self.assertIn("il == (uint32_t)detail_layer", ds4_source)

    def test_comparator_reports_every_layer_and_the_first_exact_divergence(self):
        self.assertTrue(COMPARATOR.is_file(), f"missing comparator: {COMPARATOR}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()

            one_layer = struct.pack("<f", 1.0) * VALUES_PER_LAYER
            changed_layer = bytearray(one_layer)
            last_token_offset = (TOKENS - 1) * WIDTH * 4
            struct.pack_into("<f", changed_layer, last_token_offset, 2.0)

            for layer in range(LAYERS):
                name = f"layer-{layer:02d}.f32"
                (reference / name).write_bytes(one_layer)
                candidate_bytes = changed_layer if layer == 7 else one_layer
                (candidate / name).write_bytes(candidate_bytes)

            (reference / "embd.f32").write_bytes(one_layer)
            (candidate / "embd.f32").write_bytes(one_layer)
            for stage, _, width in LAYER0_TARGETS:
                name = f"layer-00-{stage}.f32"
                checkpoint = struct.pack("<f", 1.0) * (width * TOKENS)
                candidate_bytes = checkpoint
                if stage == "ffn-out":
                    candidate_bytes = bytearray(checkpoint)
                    struct.pack_into(
                        "<f", candidate_bytes, (TOKENS - 1) * width * 4, 2.0
                    )
                (candidate / name).write_bytes(candidate_bytes)
                (reference / name).write_bytes(checkpoint)

            result = subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--format",
                    "json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["shape"], {"width": WIDTH, "tokens": TOKENS})
            self.assertEqual(len(report["layers"]), LAYERS)
            self.assertTrue(report["embedding"]["exact_hash"])
            self.assertEqual(
                report["first_divergence"],
                {"stage": "ffn-out", "layer": 0},
            )
            self.assertEqual(
                set(report["layer0_checkpoints"]), set(LAYER0_STAGES)
            )
            for stage, _, width in LAYER0_TARGETS:
                self.assertEqual(report["layer0_checkpoints"][stage]["width"], width)
            for stage in LAYER0_STAGES[:-1]:
                self.assertTrue(report["layer0_checkpoints"][stage]["exact_hash"])
            self.assertFalse(
                report["layer0_checkpoints"]["ffn-out"]["exact_hash"]
            )

            changed = report["layers"][7]
            expected_rms = 1.0 / math.sqrt(VALUES_PER_LAYER)
            self.assertAlmostEqual(changed["rms"], expected_rms, places=12)
            self.assertAlmostEqual(changed["relative_rms"], expected_rms, places=12)
            self.assertEqual(changed["max_abs"], 1.0)
            self.assertLess(changed["cosine"], 1.0)
            self.assertFalse(changed["exact_hash"])
            self.assertAlmostEqual(
                changed["last_token"]["rms"],
                1.0 / math.sqrt(WIDTH),
                places=12,
            )
            self.assertEqual(
                report["largest_last_token_relative_rms_increase"]["layer"], 7
            )
            self.assertEqual(
                report["largest_last_token_relative_rms_increase"]["previous_stage"],
                "l_out-6",
            )

            for layer in report["layers"][:7] + report["layers"][8:]:
                self.assertEqual(layer["rms"], 0.0)
                self.assertEqual(layer["relative_rms"], 0.0)
                self.assertEqual(layer["max_abs"], 0.0)
                self.assertEqual(layer["cosine"], 1.0)
                self.assertTrue(layer["exact_hash"])
                self.assertEqual(layer["last_token"]["rms"], 0.0)

    def test_comparator_rejects_noncanonical_layer_size(self):
        self.assertTrue(COMPARATOR.is_file(), f"missing comparator: {COMPARATOR}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            (reference / "layer-00.f32").write_bytes(b"short")
            (candidate / "layer-00.f32").write_bytes(b"short")

            result = subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                f"layer-00.f32: expected {BYTES_PER_LAYER} bytes",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
