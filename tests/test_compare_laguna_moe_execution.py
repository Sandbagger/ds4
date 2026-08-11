#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = (
    ROOT
    / "tests/oracle-producers/laguna-c7/compare_laguna_moe_execution.py"
)

F32_ONE = 0x3F800000
F32_TWO = 0x40000000
F32_THREE = 0x40400000
F32_SIX = 0x40C00000
F32_TEN = 0x41200000
F32_FIFTEEN = 0x41700000

STAGES = {
    "router_logits": ("layer-01-router-logits.f32", 256),
    "router_weights": ("layer-01-router-weights.f32", 10),
    "gate": ("layer-01-ffn-moe-gate.f32", 10 * 1024),
    "up": ("layer-01-ffn-moe-up.f32", 10 * 1024),
    "swiglu": ("layer-01-ffn-moe-swiglu.f32", 10 * 1024),
    "down": ("layer-01-ffn-moe-down.f32", 10 * 3072),
    "weighted": ("layer-01-ffn-moe-weighted.f32", 10 * 3072),
    "routed_sum": ("layer-01-ffn-moe-out.f32", 3072),
    "shared": ("layer-01-ffn-shared-out.f32", 3072),
    "combined": ("layer-01-ffn-out.f32", 3072),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def f32_payload(count: int, bits: int = F32_ONE) -> bytes:
    return struct.pack("<I", bits) * count


def replace_f32(path: Path, index: int, bits: int) -> None:
    payload = bytearray(path.read_bytes())
    struct.pack_into("<I", payload, index * 4, bits)
    path.write_bytes(payload)


def replace_f32_range(path: Path, start: int, count: int, bits: int) -> None:
    payload = bytearray(path.read_bytes())
    payload[start * 4 : (start + count) * 4] = struct.pack("<I", bits) * count
    path.write_bytes(payload)


class SyntheticCapture:
    def __init__(self, root: Path) -> None:
        self.poolside = root / "poolside"
        self.ds4 = root / "ds4"
        self.microscope = root / "microscope"
        self.poolside_ffn_norm = root / "poolside-ffn-norm.f32"
        self.poolside.mkdir()
        self.ds4.mkdir()
        self.microscope.mkdir()

        ffn_norm = f32_payload(3072)
        self.poolside_ffn_norm.write_bytes(ffn_norm)
        (self.ds4 / "layer-01-ffn-norm.f32").write_bytes(ffn_norm)

        selected = list(range(144, 154))
        selected_payload = struct.pack("<10i", *selected)
        for directory in (self.poolside, self.ds4):
            (directory / "layer-01-router-selected.i32").write_bytes(
                selected_payload
            )
            for _, (filename, count) in STAGES.items():
                (directory / filename).write_bytes(f32_payload(count))
            (directory / STAGES["routed_sum"][0]).write_bytes(
                f32_payload(3072, F32_TEN)
            )

        # Semantic execution order is slot-major. Up in slot 0 must beat gate
        # in slot 1, even though gate precedes up in the per-stage list.
        replace_f32(
            self.ds4 / STAGES["up"][0],
            2,
            F32_TWO,
        )
        replace_f32(
            self.ds4 / STAGES["gate"][0],
            1024,
            F32_ONE + 1,
        )
        replace_f32(self.ds4 / STAGES["router_weights"][0], 0, F32_TWO)
        replace_f32_range(
            self.ds4 / STAGES["down"][0], 0, 3072, F32_THREE
        )
        replace_f32_range(
            self.ds4 / STAGES["weighted"][0], 0, 3072, F32_SIX
        )
        (self.ds4 / STAGES["routed_sum"][0]).write_bytes(
            f32_payload(3072, F32_FIFTEEN)
        )

        input_q8 = bytes((index * 29 + 7) & 0xFF for index in range(3456))
        (self.ds4 / "layer-01-ffn-moe-input.q8_1").write_bytes(input_q8)
        weight_row = bytes((index * 17 + 3) & 0xFF for index in range(1728))
        poolside_output = struct.pack("<I", F32_ONE)

        fixture_files = {
            "input.f32": ffn_norm,
            "input.q8_1": input_q8,
            "weight-row.q4k": weight_row,
            "poolside-output.f32": poolside_output,
        }
        for name, payload in fixture_files.items():
            (self.microscope / name).write_bytes(payload)

        manifest = {
            "schema": "q4k-mmvq-microscope-fixture/v1",
            "purpose": "synthetic comparator contract",
            "poolside_commit": "synthetic",
            "poolside_sources": {},
            "model": {"bytes": 1, "sha256": "00" * 32},
            "origin": {
                "token": 513,
                "layer": 1,
                "projection": "up",
                "selected_slot": 0,
                "expert": 144,
                "row": 2,
                "tensor": "synthetic.up",
                "tensor_absolute_model_offset": 0,
                "row_absolute_model_offset": 0,
                "activation_callback": "ffn_norm-1",
                "output_callback": "ffn_moe_gate-1",
            },
            "shape": {
                "input_elements": 3072,
                "q4_k_blocks": 12,
                "q4_k_block_bytes": 144,
                "row_bytes": 1728,
            },
            "oracle": {
                "value": 1.0,
                "poolside_float32_bits": "0x3f800000",
                "ds4_serial_float32_bits": "0x40000000",
                "quantized_operands_fp64": 1.0,
            },
            "files": {
                name: {"bytes": len(payload), "sha256": sha256(payload)}
                for name, payload in fixture_files.items()
            },
        }
        (self.microscope / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def command(self, json_out: Path) -> list[str]:
        return [
            sys.executable,
            str(COMPARATOR),
            "--poolside-moe",
            str(self.poolside),
            "--poolside-ffn-norm",
            str(self.poolside_ffn_norm),
            "--ds4",
            str(self.ds4),
            "--microscope-fixture",
            str(self.microscope),
            "--json-out",
            str(json_out),
        ]


class CompareLagunaMoeExecutionTest(unittest.TestCase):
    def test_reports_slot_major_first_mismatch_and_microscope_binding(self) -> None:
        self.assertTrue(COMPARATOR.is_file(), f"missing comparator: {COMPARATOR}")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = SyntheticCapture(root)
            first_json = root / "first.json"
            second_json = root / "second.json"

            first = subprocess.run(
                capture.command(first_json),
                check=False,
                capture_output=True,
                text=True,
            )
            second = subprocess.run(
                capture.command(second_json),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            report = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(
                report["schema"], "laguna-moe-execution-comparison/v1"
            )
            self.assertEqual(
                report["semantic_order"],
                [
                    "ffn_norm",
                    "router_logits",
                    "selected",
                    "router_weights",
                    "for each slot: gate, up, swiglu, down, weighted",
                    "routed_sum",
                    "shared",
                    "combined",
                ],
            )
            expected_first_overall = {
                "stage": "router_weights",
                "slot": 0,
                "expert": 144,
                "poolside_expert": 144,
                "ds4_expert": 144,
                "poolside_bits": "0x3f800000",
                "ds4_bits": "0x40000000",
            }
            expected_first_routed = {
                "stage": "up",
                "slot": 0,
                "expert": 144,
                "poolside_expert": 144,
                "ds4_expert": 144,
                "row": 2,
                "poolside_bits": "0x3f800000",
                "ds4_bits": "0x40000000",
            }
            self.assertEqual(
                report["first_overall_mismatch"], expected_first_overall
            )
            self.assertEqual(
                report["first_routed_expert_mismatch"], expected_first_routed
            )

            gate = report["stages"]["gate"]
            self.assertEqual(gate["elements"], 10 * 1024)
            self.assertEqual(gate["equal_count"], 10 * 1024 - 1)
            self.assertEqual(gate["mismatch_count"], 1)
            self.assertEqual(gate["first_mismatch"]["slot"], 1)
            self.assertEqual(gate["first_mismatch"]["expert"], 145)
            self.assertEqual(gate["first_mismatch"]["row"], 0)

            up = report["stages"]["up"]
            self.assertEqual(up["equal_count"], 10 * 1024 - 1)
            self.assertEqual(up["mismatch_count"], 1)
            self.assertEqual(
                report["stages"]["router_weights"]["mismatch_count"], 1
            )
            for stage in ("down", "weighted", "routed_sum"):
                self.assertEqual(
                    report["stages"][stage]["mismatch_count"], 3072
                )
            for stage in (
                "ffn_norm",
                "router_logits",
                "selected",
                "swiglu",
                "shared",
                "combined",
            ):
                self.assertEqual(report["stages"][stage]["mismatch_count"], 0)

            microscope = report["microscope"]
            self.assertEqual(microscope["origin"]["selected_slot"], 0)
            self.assertEqual(microscope["origin"]["expert"], 144)
            self.assertEqual(
                microscope["direct_output_binding"],
                {
                    "poolside_bits": "0x3f800000",
                    "ds4_bits": "0x40000000",
                    "poolside_matches_oracle": True,
                    "ds4_matches_oracle": True,
                    "poolside_matches_fixture_output": True,
                },
            )
            self.assertTrue(microscope["input_binding"]["ffn_norm_exact"])
            self.assertTrue(microscope["input_binding"]["q8_1_exact"])
            decomposition = report["counterfactual_decomposition"]
            self.assertTrue(decomposition["poolside_replay"]["weighted_exact"])
            self.assertTrue(decomposition["poolside_replay"]["routed_sum_exact"])
            self.assertTrue(decomposition["ds4_replay"]["weighted_exact"])
            self.assertTrue(decomposition["ds4_replay"]["routed_sum_exact"])
            self.assertEqual(
                decomposition["routing_only"],
                {
                    "unequal_rows": 3072,
                    "rows": 3072,
                    "max_absolute_delta": 1.0,
                    "rms_delta": 1.0,
                    "l2_delta": math.sqrt(3072),
                },
            )
            self.assertEqual(
                decomposition["expert_only"],
                {
                    "unequal_rows": 3072,
                    "rows": 3072,
                    "max_absolute_delta": 2.0,
                    "rms_delta": 2.0,
                    "l2_delta": math.sqrt(4.0 * 3072),
                },
            )
            self.assertEqual(
                decomposition["combined"],
                {
                    "unequal_rows": 3072,
                    "rows": 3072,
                    "max_absolute_delta": 5.0,
                    "rms_delta": 5.0,
                    "l2_delta": math.sqrt(25.0 * 3072),
                },
            )
            self.assertIn(
                "first_routed_expert_mismatch stage=up slot=0 expert=144 row=2",
                first.stdout,
            )
            self.assertNotIn("largest", first.stdout.lower())

    def test_rejects_direct_inputs_not_bound_to_microscope(self) -> None:
        self.assertTrue(COMPARATOR.is_file(), f"missing comparator: {COMPARATOR}")
        for artifact in ("ffn_norm", "q8_1"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                capture = SyntheticCapture(root)
                if artifact == "ffn_norm":
                    replace_f32(
                        capture.ds4 / "layer-01-ffn-norm.f32", 0, F32_TWO
                    )
                else:
                    q8_path = capture.ds4 / "layer-01-ffn-moe-input.q8_1"
                    payload = bytearray(q8_path.read_bytes())
                    payload[0] ^= 0xFF
                    q8_path.write_bytes(payload)

                json_out = root / "report.json"
                result = subprocess.run(
                    capture.command(json_out),
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("does not match microscope", result.stderr)
                self.assertFalse(json_out.exists())

    def test_rejects_microscope_selected_expert_not_bound_to_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = SyntheticCapture(root)
            selected_path = capture.ds4 / "layer-01-router-selected.i32"
            selected = list(struct.unpack("<10i", selected_path.read_bytes()))
            selected[0] = 143
            selected_path.write_bytes(struct.pack("<10i", *selected))

            json_out = root / "report.json"
            result = subprocess.run(
                capture.command(json_out),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("selected expert", result.stderr)
            self.assertFalse(json_out.exists())

    def test_rejects_direct_output_not_bound_to_microscope_oracle(self) -> None:
        for runtime in ("poolside", "ds4"):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                capture = SyntheticCapture(root)
                directory = getattr(capture, runtime)
                replace_f32(directory / STAGES["up"][0], 2, F32_ONE + 7)

                json_out = root / "report.json"
                result = subprocess.run(
                    capture.command(json_out),
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("direct output", result.stderr)
                self.assertFalse(json_out.exists())

    def test_rejects_microscope_coordinate_after_first_routed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = SyntheticCapture(root)
            replace_f32(capture.ds4 / STAGES["up"][0], 3, F32_TWO)
            manifest_path = capture.microscope / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["origin"]["row"] = 3
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            json_out = root / "report.json"
            result = subprocess.run(
                capture.command(json_out),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("first routed mismatch", result.stderr)
            self.assertFalse(json_out.exists())

    def test_rejects_capture_not_reproduced_by_f32_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = SyntheticCapture(root)
            replace_f32(capture.ds4 / STAGES["weighted"][0], 7, F32_ONE)

            json_out = root / "report.json"
            result = subprocess.run(
                capture.command(json_out),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("F32 replay", result.stderr)
            self.assertFalse(json_out.exists())


if __name__ == "__main__":
    unittest.main()
