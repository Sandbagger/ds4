#!/usr/bin/env python3
"""Contract tests for the deterministic Laguna benchmark prompt."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests/test-vectors/laguna-resident/generate_benchmark_prompt.py"
LINE = b"The ring buffer stores each item at a position modulo its fixed capacity.\n"
EXPECTED_BYTES = LINE * 4096
EXPECTED_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"


class GenerateBenchmarkPromptTest(unittest.TestCase):
    def test_cli_writes_exact_stable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.txt"
            completed = subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, f"wrote={output} bytes=303104 lines=4096\n")
            actual = output.read_bytes()
            self.assertEqual(actual, EXPECTED_BYTES)
            self.assertEqual(hashlib.sha256(actual).hexdigest(), EXPECTED_SHA256)

    def test_cli_rejects_unknown_arguments_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.txt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--output",
                    str(output),
                    "--unknown",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("unrecognized arguments: --unknown", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
