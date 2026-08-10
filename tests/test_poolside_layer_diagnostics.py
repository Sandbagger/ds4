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
WIDTH = 3072
TOKENS = 22
LAYERS = 48
VALUES_PER_LAYER = WIDTH * TOKENS
BYTES_PER_LAYER = VALUES_PER_LAYER * 4


class PoolsideLayerDiagnosticsTest(unittest.TestCase):
    def test_probe_source_exposes_the_exact_short_prompt_callback_contract(self):
        self.assertTrue(PROBE.is_file(), f"missing diagnostic probe: {PROBE}")
        source = PROBE.read_text(encoding="utf-8")

        for required in (
            "--model",
            "--tokens",
            "--out",
            "cb_eval",
            "ggml_backend_tensor_get",
            "ggml_is_contiguous",
            "GGML_TYPE_F32",
            '"embd"',
            '"l_out-"',
            '"result_output"',
            "layer-%02d.f32",
            "embd.f32",
            "logits.f32",
            "duplicate",
        ):
            self.assertIn(required, source)

        self.assertRegex(source, r"kWidth\s*=\s*3072")
        self.assertRegex(source, r"kTokens\s*=\s*22")
        self.assertRegex(source, r"kLayers\s*=\s*48")

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
            struct.pack_into("<f", changed_layer, 0, 2.0)

            for layer in range(LAYERS):
                name = f"layer-{layer:02d}.f32"
                (reference / name).write_bytes(one_layer)
                candidate_bytes = changed_layer if layer == 7 else one_layer
                (candidate / name).write_bytes(candidate_bytes)

            (reference / "embd.f32").write_bytes(one_layer)
            (candidate / "embd.f32").write_bytes(one_layer)

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
                {"stage": "l_out", "layer": 7},
            )

            changed = report["layers"][7]
            expected_rms = 1.0 / math.sqrt(VALUES_PER_LAYER)
            self.assertAlmostEqual(changed["rms"], expected_rms, places=12)
            self.assertAlmostEqual(changed["relative_rms"], expected_rms, places=12)
            self.assertEqual(changed["max_abs"], 1.0)
            self.assertLess(changed["cosine"], 1.0)
            self.assertFalse(changed["exact_hash"])

            for layer in report["layers"][:7] + report["layers"][8:]:
                self.assertEqual(layer["rms"], 0.0)
                self.assertEqual(layer["relative_rms"], 0.0)
                self.assertEqual(layer["max_abs"], 0.0)
                self.assertEqual(layer["cosine"], 1.0)
                self.assertTrue(layer["exact_hash"])

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
