#!/usr/bin/env python3
"""Static contract for the frozen Laguna continuation behavior probe."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = (
    ROOT
    / "tests/oracle-producers/laguna-c7/probe_ds4_laguna_behavior.c"
)
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


class LagunaBehaviorProbeContractTests(unittest.TestCase):
    def test_release_linked_probe_target_is_isolated_from_test_hooks(self) -> None:
        self.assertTrue(PRODUCER.is_file(), "missing Laguna behavior producer")
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertNotIn("DS4_TEST_HOOKS", source)

        link = re.search(
            r"^tests/probe_ds4_laguna_behavior:\s*(?P<objects>.*)$",
            MAKEFILE,
            re.MULTILINE,
        )
        self.assertIsNotNone(link, "missing behavior-probe link target")
        objects = link.group("objects")
        self.assertIn("tests/probe_ds4_laguna_behavior.o", objects)
        self.assertRegex(objects, r"(?:^|\s)ds4\.o(?:\s|$)")
        self.assertRegex(objects, r"(?:^|\s)ds4_cuda\.o(?:\s|$)")
        self.assertNotIn("test_hooks", objects)

        compile_rule = re.search(
            r"^tests/probe_ds4_laguna_behavior\.o:.*\n(?P<recipe>\t.*)$",
            MAKEFILE,
            re.MULTILINE,
        )
        self.assertIsNotNone(compile_rule)
        self.assertIn("-fno-fast-math", compile_rule.group("recipe"))
        self.assertNotIn("DS4_TEST_HOOKS", compile_rule.group("recipe"))

        clean = re.search(r"^clean:\n(?P<body>(?:\t.*\n?)+)", MAKEFILE, re.MULTILINE)
        self.assertIsNotNone(clean)
        self.assertIn("tests/probe_ds4_laguna_behavior", clean.group("body"))
        self.assertIn("tests/probe_ds4_laguna_behavior.o", clean.group("body"))

    def test_cli_and_frozen_512_plus_1_boundary_are_explicit(self) -> None:
        source = PRODUCER.read_text(encoding="utf-8")
        for flag in ("--model", "--tokens", "--reference", "--steps", "--out"):
            self.assertIn(flag, source)
        self.assertRegex(source, r"PREFIX_TOKENS\s*=\s*512")
        self.assertRegex(source, r"RESUME_TOKEN\s*=\s*3612")
        self.assertRegex(source, r"MAX_STEPS\s*=\s*32")
        self.assertIn("ds4_session_sync(", source)
        self.assertRegex(
            source,
            r"ds4_session_eval\([^,]+,\s*RESUME_TOKEN",
        )
        self.assertIn("PREFIX_TOKENS + 1", source)
        self.assertIn("FROZEN_PREFIX_PATTERN", source)
        self.assertIn("validate_frozen_prefix(prefix_ids)", source)
        self.assertRegex(
            source,
            r"open\(opt->out,\s*O_WRONLY \| O_CREAT \| O_EXCL",
        )

    def test_both_trajectories_use_full_logits_and_deterministic_top20(self) -> None:
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertIn("ds4_session_copy_logits(", source)
        self.assertIn("VOCAB_SIZE", source)
        self.assertIn("ranks_before", source)
        self.assertIn("vector_top20", source)
        self.assertIn("run_greedy_trajectory", source)
        self.assertIn("run_teacher_trajectory", source)
        self.assertIn("isfinite", source)
        self.assertIn("logsumexp_binary64", source)
        self.assertIn("exp(", source)
        self.assertIn("log(", source)

    def test_json_contract_carries_ordering_and_teacher_nll(self) -> None:
        source = PRODUCER.read_text(encoding="utf-8")
        for literal in (
            '"laguna-ds4-behavior-probe/v1"',
            '\\"steps\\"',
            '\\"greedy_tokens\\"',
            '\\"greedy_steps\\"',
            '\\"greedy_matching_prefix\\"',
            '\\"top20\\"',
            '\\"teacher_steps\\"',
            '\\"target_logprob\\"',
            '\\"target_nll\\"',
            '\\"teacher_nll_total\\"',
            '\\"teacher_nll_avg\\"',
        ):
            self.assertIn(literal, source)


if __name__ == "__main__":
    unittest.main()
