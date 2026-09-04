#!/usr/bin/env python3
"""Host-only RED tests for Laguna layer mismatch localization.

These tests use tiny synthetic float32 files.  They validate diagnostic
localization only; they make no model-parity or acceptance claim and never
launch a GPU, model, network, or service.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "gguf-tools/quality-testing/compare_laguna_layers.py"


_SPEC = importlib.util.spec_from_file_location(
    "compare_laguna_layers_under_test", COMPARATOR
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load comparator: {COMPARATOR}")
_COMPARATOR: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPARATOR)


def _payload(bits: list[int]) -> bytes:
    return b"".join(struct.pack("<I", value) for value in bits)


def _metric(
    reference_bits: list[int],
    candidate_bits: list[int],
    **kwargs: Any,
) -> dict[str, Any]:
    if len(reference_bits) != len(candidate_bits):
        raise AssertionError("synthetic metric files must have equal length")
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        reference = root / "reference.f32"
        candidate = root / "candidate.f32"
        reference.write_bytes(_payload(reference_bits))
        candidate.write_bytes(_payload(candidate_bits))
        return _COMPARATOR.metrics(
            reference,
            candidate,
            len(reference_bits) * 4,
            **kwargs,
        )


class LagunaLayerMismatchLocalizationTest(unittest.TestCase):
    def test_exact_finite_match_reports_no_first_mismatch(self) -> None:
        result = _metric(
            [0x3F800000, 0x40000000, 0x40400000],
            [0x3F800000, 0x40000000, 0x40400000],
            row_width=3,
        )

        self.assertTrue(result["exact_hash"])
        self.assertIn("first_mismatch", result)
        self.assertIsNone(result["first_mismatch"])

    def test_first_bitwise_difference_reports_absolute_token_and_channel(self) -> None:
        reference = [0x3F800000] * 52
        candidate = reference.copy()
        candidate[49] = 0x40000000
        result = _metric(reference, candidate, row_width=48)

        self.assertEqual(
            result["first_mismatch"],
            {
                "flat_index": 49,
                "token_index": 1,
                "element_index": 1,
                "reference_bits": "0x3f800000",
                "candidate_bits": "0x40000000",
                "reference_value": 1.0,
                "candidate_value": 2.0,
            },
        )

        # A one-ULP difference before a larger difference must still win.  The
        # locator follows original file order, not a tolerance threshold.
        candidate = reference.copy()
        candidate[0] = 0x3F800001
        candidate[49] = 0x40000000
        result = _metric(reference, candidate, row_width=48)
        self.assertEqual(result["first_mismatch"]["flat_index"], 0)
        self.assertEqual(result["first_mismatch"]["token_index"], 0)
        self.assertEqual(result["first_mismatch"]["element_index"], 0)
        self.assertEqual(result["first_mismatch"]["candidate_bits"], "0x3f800001")

    def test_slice_mismatch_uses_global_index_for_last_token(self) -> None:
        reference = [0x00000000] * 144
        candidate = reference.copy()
        candidate[101] = 0x40500000  # 3.25, in the selected final token
        result = _metric(
            reference,
            candidate,
            value_start=96,
            value_count=48,
            row_width=48,
        )

        self.assertEqual(
            result["first_mismatch"],
            {
                "flat_index": 101,
                "token_index": 2,
                "element_index": 5,
                "reference_bits": "0x00000000",
                "candidate_bits": "0x40500000",
                "reference_value": 0.0,
                "candidate_value": 3.25,
            },
        )

    def test_unknown_width_logits_report_flat_coordinate_only(self) -> None:
        reference = [0x3F800000] * 4
        candidate = reference.copy()
        candidate[3] = 0x40400000
        result = _metric(reference, candidate, row_width=None)

        self.assertEqual(
            result["first_mismatch"],
            {
                "flat_index": 3,
                "token_index": None,
                "element_index": None,
                "reference_bits": "0x3f800000",
                "candidate_bits": "0x40400000",
                "reference_value": 1.0,
                "candidate_value": 3.0,
            },
        )

    def test_signed_zero_reports_bitwise_difference_with_zero_error(self) -> None:
        result = _metric([0x00000000], [0x80000000], row_width=1)

        self.assertFalse(result["exact_hash"])
        self.assertEqual(result["rms"], 0.0)
        self.assertEqual(result["relative_rms"], 0.0)
        self.assertEqual(result["cosine"], 1.0)
        self.assertEqual(
            result["first_mismatch"],
            {
                "flat_index": 0,
                "token_index": 0,
                "element_index": 0,
                "reference_bits": "0x00000000",
                "candidate_bits": "0x80000000",
                "reference_value": 0.0,
                "candidate_value": -0.0,
            },
        )

    def test_nonfinite_inputs_are_rejected_for_identical_and_differing_bytes(
        self,
    ) -> None:
        cases = (
            ("identical_nan", 0x7FC00000, 0x7FC00000),
            ("different_nan_payload", 0x7FC00000, 0x7FC00001),
            ("identical_positive_infinity", 0x7F800000, 0x7F800000),
            ("different_infinity_sign", 0x7F800000, 0xFF800000),
        )
        for name, reference_bits, candidate_bits in cases:
            with self.subTest(case=name):
                with self.assertRaises(_COMPARATOR.DiagnosticError):
                    _metric([reference_bits], [candidate_bits], row_width=1)

    def test_invalid_row_width_is_rejected(self) -> None:
        for row_width in (0, -1):
            with self.subTest(row_width=row_width):
                with self.assertRaises(_COMPARATOR.DiagnosticError):
                    _metric([0x3F800000], [0x3F800000], row_width=row_width)

    def test_cli_json_includes_logits_first_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            # The existing comparator CLI expects the canonical 48 layer files.
            # Sparse zero-filled files keep this integration test host-only.
            for layer in range(_COMPARATOR.LAYER_COUNT):
                for directory in (reference, candidate):
                    layer_path = directory / f"layer-{layer:02d}.f32"
                    with layer_path.open("wb") as handle:
                        handle.truncate(_COMPARATOR.LAYER_BYTES)
            (reference / "logits.f32").write_bytes(
                _payload([0x3F800000, 0x40000000])
            )
            (candidate / "logits.f32").write_bytes(
                _payload([0x3F800000, 0x40400000])
            )
            completed = subprocess.run(
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

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(
            report["logits"]["first_mismatch"],
            {
                "flat_index": 1,
                "token_index": None,
                "element_index": None,
                "reference_bits": "0x40000000",
                "candidate_bits": "0x40400000",
                "reference_value": 2.0,
                "candidate_value": 3.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
