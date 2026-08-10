#!/usr/bin/env python3
"""Source contract for Poolside-compatible Laguna F16 projections."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")


def function_body(name: str) -> str:
    match = re.search(
        rf'extern\s+"C"\s+int\s+{name}\s*\(', CUDA_SOURCE
    )
    if match is None:
        return ""
    next_function = re.search(
        r'\nextern\s+"C"\s+(?:int|void|uint64_t)\s+',
        CUDA_SOURCE[match.end() :],
    )
    stop = len(CUDA_SOURCE)
    if next_function is not None:
        stop = match.end() + next_function.start()
    return CUDA_SOURCE[match.start() : stop]


class LagunaF16ProjectionContractTest(unittest.TestCase):
    def test_batched_f16_projection_matches_poolside_cublas_contract(self) -> None:
        body = function_body("ds4_gpu_matmul_f16_tensor")

        self.assertTrue(body, "missing F16 matmul wrapper")
        self.assertIn("CUBLAS_COMPUTE_32F", body)
        self.assertIn("CUBLAS_GEMM_DEFAULT_TENSOR_OP", body)


if __name__ == "__main__":
    unittest.main()
