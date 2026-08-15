#!/usr/bin/env python3
"""Contracts for the streamed GGUF artifact comparator."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "gguf-tools/quality-testing/compare_gguf_artifacts.py"
ALIGNMENT = 32


def _gguf_string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def _metadata_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _metadata_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<II", 4, value)


def _metadata_string_array(key: str, values: tuple[str, ...]) -> bytes:
    payload = b"".join(_gguf_string(value) for value in values)
    return (
        _gguf_string(key)
        + struct.pack("<IIQ", 9, 8, len(values))
        + payload
    )


def _tensor_info(
    name: str,
    dimensions: tuple[int, ...],
    ggml_type: int,
    relative_offset: int,
) -> bytes:
    return (
        _gguf_string(name)
        + struct.pack("<I", len(dimensions))
        + b"".join(struct.pack("<Q", value) for value in dimensions)
        + struct.pack("<IQ", ggml_type, relative_offset)
    )


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


TENSORS = (
    ("token_embd.weight", (4,), 0, bytes(range(16))),
    ("blk.0.ffn_gate_exps.weight", (256,), 12, bytes(range(144))),
    ("blk.0.ffn_down_exps.weight", (256,), 14, bytes(range(210))),
    ("output_norm.weight", (4,), 30, b"\x11\x22" * 4),
)


def _write_gguf(
    path: Path,
    *,
    chat_template: str,
    changed_tensor: str | None = None,
) -> None:
    metadata = (
        _metadata_u32("general.alignment", ALIGNMENT),
        _metadata_string("general.name", "synthetic Laguna"),
        _metadata_string("tokenizer.chat_template", chat_template),
        _metadata_string_array("tokenizer.ggml.tokens", ("zero", "one")),
    )

    relative_offset = 0
    tensor_directory: list[bytes] = []
    payloads: list[bytes] = []
    for name, dimensions, ggml_type, original_payload in TENSORS:
        payload = original_payload
        if name == changed_tensor:
            payload = bytes((original_payload[0] ^ 0xFF,)) + original_payload[1:]
        tensor_directory.append(
            _tensor_info(name, dimensions, ggml_type, relative_offset)
        )
        payloads.append(payload)
        relative_offset = _align(relative_offset + len(payload))

    header = struct.pack("<4sIQQ", b"GGUF", 3, len(TENSORS), len(metadata))
    prefix = header + b"".join(metadata) + b"".join(tensor_directory)
    output = bytearray(prefix)
    output.extend(b"\0" * (_align(len(output)) - len(output)))
    for payload in payloads:
        output.extend(payload)
        output.extend(b"\0" * (_align(len(output)) - len(output)))
    path.write_bytes(output)


class CompareGgufArtifactsTests(unittest.TestCase):
    def _run(self, base: Path, candidate: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--chunk-bytes",
                "17",
                str(base),
                str(candidate),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_metadata_growth_moves_but_does_not_change_tensor_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.gguf"
            candidate = Path(tmp) / "candidate.gguf"
            _write_gguf(base, chat_template="old")
            _write_gguf(candidate, chat_template="old" + "x" * 416)

            completed = self._run(base, candidate)

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema"], "ds4.gguf-artifact-comparison/v1")
        self.assertTrue(report["compatible"])
        self.assertFalse(report["metadata_equal"])
        self.assertTrue(report["tensor_layout_equal"])
        self.assertTrue(report["tensor_payloads_equal"])
        self.assertEqual(
            report["metadata_differences"], ["tokenizer.chat_template"]
        )
        self.assertEqual(report["layout_differences"], [])
        self.assertEqual(report["payload_differences"], [])
        self.assertEqual(
            int(report["candidate"]["size_bytes"])
            - int(report["base"]["size_bytes"]),
            416,
        )
        self.assertNotEqual(report["base"]["sha256"], report["candidate"]["sha256"])

        expected_types = ["f32", "q4_k", "q6_k", "bf16"]
        self.assertEqual(
            [entry["base"]["type_name"] for entry in report["tensors"]],
            expected_types,
        )
        for entry in report["tensors"]:
            self.assertTrue(entry["layout_equal"], entry["name"])
            self.assertTrue(entry["payload_equal"], entry["name"])
            self.assertEqual(entry["base"]["sha256"], entry["candidate"]["sha256"])
            self.assertEqual(
                int(entry["candidate"]["absolute_offset"])
                - int(entry["base"]["absolute_offset"]),
                416,
            )

    def test_changed_tensor_is_named_and_returns_nonzero(self) -> None:
        changed = "blk.0.ffn_gate_exps.weight"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.gguf"
            candidate = Path(tmp) / "candidate.gguf"
            _write_gguf(base, chat_template="old")
            _write_gguf(
                candidate,
                chat_template="old" + "x" * 416,
                changed_tensor=changed,
            )

            completed = self._run(base, candidate)

        self.assertEqual(
            completed.returncode,
            1,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        report = json.loads(completed.stdout)
        self.assertFalse(report["compatible"])
        self.assertTrue(report["tensor_layout_equal"])
        self.assertFalse(report["tensor_payloads_equal"])
        self.assertEqual(report["payload_differences"], [changed])
        changed_entry = next(
            entry for entry in report["tensors"] if entry["name"] == changed
        )
        self.assertFalse(changed_entry["payload_equal"])
        self.assertNotEqual(
            changed_entry["base"]["sha256"],
            changed_entry["candidate"]["sha256"],
        )

    def test_truncated_tensor_payload_fails_closed_without_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.gguf"
            candidate = Path(tmp) / "candidate.gguf"
            _write_gguf(base, chat_template="old")
            _write_gguf(candidate, chat_template="old" + "x" * 416)
            candidate.write_bytes(candidate.read_bytes()[:-25])

            completed = self._run(base, candidate)

        self.assertEqual(
            completed.returncode,
            2,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertEqual(completed.stdout, "")
        self.assertIn("extends beyond the GGUF file", completed.stderr)


if __name__ == "__main__":
    unittest.main()
