#!/usr/bin/env python3
"""Source contract for Poolside-compatible Laguna long-prefill attention."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text()


def function_body(name: str) -> str:
    match = re.search(
        rf"(?:__global__\s+static\s+void|extern\s+\"C\"\s+int)\s+{name}\s*\(",
        CUDA_SOURCE,
    )
    if match is None:
        return ""
    next_function = re.search(
        r"\n(?:__global__\s+static\s+void|extern\s+\"C\"\s+int|static\s+int)\s+",
        CUDA_SOURCE[match.end() :],
    )
    stop = len(CUDA_SOURCE)
    if next_function is not None:
        stop = match.end() + next_function.start()
    return CUDA_SOURCE[match.start() : stop]


class LagunaLongAttentionContractTest(unittest.TestCase):
    def test_long_prefill_has_a_64_key_mma_kernel(self) -> None:
        body = function_body("laguna_attention_prefill_auto_mma64_kernel")

        self.assertTrue(body, "missing Laguna long-prefill MMA kernel")
        self.assertIn("constexpr uint32_t key_tile = 64u;", body)
        self.assertIn("nvcuda::wmma", body)
        self.assertIn("3.0f * 0.6931f", body)

    def test_long_prefill_rescales_the_half_vkq_accumulator_per_tile(self) -> None:
        body = function_body("laguna_attention_prefill_auto_mma64_kernel")

        self.assertIn("__float2half_rn(max_scale)", body)
        self.assertRegex(body, r"__hmul\([^;]*max_scale_h")

    def test_laguna_shapes_dispatch_long_prefill_away_from_scalar_fallback(self) -> None:
        body = function_body("ds4_gpu_laguna_attention_prefill_tensor")

        self.assertIn("laguna_attention_prefill_auto_mma64_kernel<<<", body)
        self.assertRegex(
            body,
            r"n_tokens\s*>\s*22u[\s\S]*"
            r"\(n_head\s*==\s*48u\s*\|\|\s*n_head\s*==\s*72u\)",
        )
        long_launch = body.index("laguna_attention_prefill_auto_mma64_kernel<<<")
        scalar_launch = body.index("laguna_attention_prefill_gqa_f16_kernel<<<")
        self.assertLess(long_launch, scalar_launch)

    def test_long_prefill_mma_does_not_cross_ring_capacity(self) -> None:
        body = function_body("ds4_gpu_laguna_attention_prefill_tensor")

        launch = "laguna_attention_prefill_auto_mma64_kernel<<<"
        prefix, marker, _ = body.partition(launch)
        self.assertTrue(marker, "missing Laguna long-prefill MMA dispatch")
        branch = prefix.rsplit("} else if (", 1)[-1]
        self.assertIn("cache_cap >= n_tokens", branch)


if __name__ == "__main__":
    unittest.main()
