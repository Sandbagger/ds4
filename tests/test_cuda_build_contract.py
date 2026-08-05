#!/usr/bin/env python3
"""Host-only source contracts for standalone CUDA links and cleanup policy."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")

STANDALONE_CUDA_TARGETS = (
    "tests/cuda_long_context_smoke",
    "tests/test_cuda_laguna_kernels",
    "tests/test_gpu_xdev",
    "tests/test_gpu_model_cache",
    "tests/test_gpu_lookup_cache_strict",
)


def rule_prerequisites(target: str) -> str:
    match = re.search(rf"(?m)^{re.escape(target)}:\s*([^\n]*)$", MAKEFILE)
    if match is None:
        raise AssertionError(f"missing Makefile rule for {target}")
    return match.group(1)


def function_body(signature: str) -> str:
    start = CUDA_SOURCE.find(signature)
    if start < 0:
        raise AssertionError(f"missing CUDA function {signature}")
    brace = CUDA_SOURCE.find("{", start)
    depth = 0
    for index in range(brace, len(CUDA_SOURCE)):
        if CUDA_SOURCE[index] == "{":
            depth += 1
        elif CUDA_SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return CUDA_SOURCE[brace + 1 : index]
    raise AssertionError(f"unterminated CUDA function {signature}")


class CudaBuildContractTest(unittest.TestCase):
    def test_standalone_cuda_links_include_runtime_tracker(self) -> None:
        for target in STANDALONE_CUDA_TARGETS:
            with self.subTest(target=target):
                prerequisites = rule_prerequisites(target)
                self.assertIn("ds4_cuda.o", prerequisites)
                self.assertIn("ds4_runtime.o", prerequisites)

    def test_noncompact_cleanup_keeps_best_effort_sync_policy(self) -> None:
        body = function_body('extern "C" void ds4_gpu_cleanup(void)')
        preamble, separator, _ = body.partition("g_current_logical_tier = -1;")
        self.assertTrue(separator, "cleanup state reset moved or disappeared")
        self.assertIn("compact_cleanup_required", preamble)
        self.assertIn("cuda_laguna_compact_destroy_checked", preamble)
        self.assertIn("(void)cudaDeviceSynchronize();", preamble)
        self.assertNotIn("cleanup_sync_error", preamble)


if __name__ == "__main__":
    unittest.main()
