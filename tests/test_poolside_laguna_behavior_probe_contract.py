#!/usr/bin/env python3
"""Contract tests for the frozen Poolside Laguna behavior oracle producer."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = (
    ROOT
    / "tests/oracle-producers/laguna-c7/probe_poolside_laguna_behavior.cpp"
)


class PoolsideLagunaBehaviorProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PRODUCER.read_text(encoding="utf-8")

    def test_cli_and_frozen_case_are_explicit(self) -> None:
        self.assertIn("--model MODEL --tokens TOKENS.i32 --out DIR --steps 1..32", self.source)
        self.assertIn("static constexpr int kPrefixTokens = 512;", self.source)
        self.assertIn("static constexpr int32_t kResumeToken = 3612;", self.source)
        self.assertIn("static constexpr int kMaxSteps = 32;", self.source)
        self.assertIn("--steps must be an integer from 1 through 32", self.source)

    def test_prefix_is_exact_little_endian_i32(self) -> None:
        self.assertIn("token file must contain exactly 512 little-endian int32 IDs", self.source)
        self.assertIn("kPrefixPattern", self.source)
        self.assertIn("token file does not contain the frozen Laguna prefix", self.source)
        self.assertRegex(
            self.source,
            r"(?s)static_cast<uint32_t>\(bytes\[offset \+ 0\]\).*"
            r"static_cast<uint32_t>\(bytes\[offset \+ 3\]\) << 24",
        )

    def test_context_and_decode_boundary_are_fixed(self) -> None:
        self.assertIn("context_params.n_ctx = 1024;", self.source)
        self.assertIn("context_params.n_batch = 1024;", self.source)
        self.assertIn("context_params.n_ubatch = 512;", self.source)
        self.assertIn("batch.n_tokens = kPrefixTokens;", self.source)
        self.assertIn("batch.token[0] = kResumeToken;", self.source)
        self.assertIn("batch.pos[0] = kPrefixTokens;", self.source)
        self.assertIn("batch.logits[0] = true;", self.source)

    def test_each_step_writes_logits_then_greedy_decodes(self) -> None:
        self.assertIn('"behavior-step-%02d.logits.f32"', self.source)
        self.assertIn('"behavior-continuation.i32"', self.source)
        self.assertIn("llama_get_logits_ith(context, 0)", self.source)
        self.assertIn("if (logits[token] > logits[best_token])", self.source)

        loop = re.search(
            r"for \(int step = 0; step < steps; step\+\+\) \{(?P<body>.*?)\n    \}",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(loop)
        body = loop.group("body")
        self.assertLess(body.index("write_f32_little_endian"), body.index("greedy_argmax"))
        self.assertLess(body.index("greedy_argmax"), body.index("continuation.push_back"))
        self.assertLess(body.index("continuation.push_back"), body.index("decode_one"))

    def test_output_is_empty_and_files_are_never_overwritten(self) -> None:
        self.assertIn("output directory must be empty", self.source)
        self.assertIn("refusing to overwrite behavior file", self.source)


if __name__ == "__main__":
    unittest.main()
