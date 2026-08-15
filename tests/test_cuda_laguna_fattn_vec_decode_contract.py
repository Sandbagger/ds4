#!/usr/bin/env python3
"""Source contract for the pinned Poolside Laguna q=1 FATTN_VEC slice."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "ds4_cuda.cu").read_text()
HEADER = (ROOT / "ds4_gpu.h").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()
KERNEL_TEST = (ROOT / "tests/test_cuda_laguna_kernels.c").read_text()


def function_body(name: str) -> str:
    match = re.search(
        rf"(?:__global__\s+static\s+void|static\s+void|static\s+int|static\s+uint32_t|"
        rf"extern\s+\"C\"\s+int)\s+{name}\s*\(",
        CUDA,
    )
    if match is None:
        return ""
    next_function = re.search(
        r"\n(?:__global__\s+static\s+void|static\s+void|static\s+int|static\s+uint32_t|"
        r"extern\s+\"C\"\s+int)\s+",
        CUDA[match.end() :],
    )
    stop = len(CUDA)
    if next_function is not None:
        stop = match.end() + next_function.start()
    return CUDA[match.start() : stop]


class LagunaFattnVecDecodeContractTest(unittest.TestCase):
    def test_host_policy_is_in_the_model_free_cuda_gate(self) -> None:
        self.assertIn(
            "test-cuda-laguna-fattn-vec-policy", MAKEFILE
        )
        target = re.search(
            r"^test-cuda-build-contract:(?P<deps>[^\n]*)$",
            MAKEFILE,
            re.MULTILINE,
        )
        self.assertIsNotNone(target)
        self.assertIn(
            "test-cuda-laguna-fattn-vec-policy", target.group("deps")
        )
        self.assertIn(
            "tests/test_cuda_laguna_fattn_vec_decode_contract.py",
            MAKEFILE,
        )

    def test_kernel_keeps_poolside_partition_and_mask_topology(self) -> None:
        body = function_body("laguna_attention_decode_fattn_vec_kernel")
        self.assertTrue(body, "missing targeted FATTN_VEC kernel")
        for pinned in (
            "constexpr uint32_t threads = 128u",
            "constexpr uint32_t keys_per_partition = 128u",
            "tile += partitions * keys_per_partition",
            "3.0f * 0.6931f",
            "Q_reg[slot].x *= scale",
            "__shfl_xor_sync",
            "cache_row_pairs",
            "padded_key_count",
            "key < key_count ? 0.0f : -INFINITY",
        ):
            self.assertIn(pinned, body)
        self.assertIn("Poolside llama.cpp 04b2b72", CUDA)

    def test_partition_combine_precedes_separate_gate_boundary(self) -> None:
        combine = function_body("laguna_attention_decode_fattn_vec_combine_kernel")
        softplus = function_body("laguna_attention_gate_softplus_kernel")
        gate_mul = function_body("laguna_attention_gate_mul_kernel")
        self.assertIn("for (uint32_t part = 0u; part < partitions; part++)", combine)
        self.assertIn("expf(meta[part].x - max_score)", combine)
        self.assertIn("gate_softplus[head] =", softplus)
        self.assertRegex(
            softplus,
            r"logf\(1\.0f \+ expf\(gate_value\)\)",
        )
        self.assertIn("raw_heads[index] =", gate_mul)
        self.assertIn("gate_softplus[head]", gate_mul)

    def test_decode_dispatch_is_narrow_and_rollback_is_init_latched(self) -> None:
        refresh = function_body("cuda_decode_dispatch_env_refresh")
        self.assertEqual(
            refresh.count("DS4_CUDA_NO_LAGUNA_FATTN_VEC_DECODE"), 1
        )
        dispatch = function_body("ds4_gpu_laguna_store_attention_tensor")
        vec_launch = dispatch.index("laguna_attention_decode_fattn_vec_launch(")
        scalar_launch = dispatch.index("laguna_attention_decode_gqa_f16_kernel<<<")
        self.assertLess(vec_launch, scalar_launch)
        self.assertNotIn("getenv", dispatch)

        launch = function_body("laguna_attention_decode_fattn_vec_launch")
        for required in (
            "poolside_partitions = 3u",
            "padded_key_count = 768u",
            "const dim3 grid(1u, partitions, n_head)",
            "const dim3 combine_grid(1u, n_head, 1u)",
            "laguna_attention_decode_fattn_vec_kernel<<<",
            "laguna_attention_decode_fattn_vec_combine_kernel<<<",
            "laguna_attention_gate_softplus_kernel<<<",
            "laguna_attention_gate_mul_kernel<<<",
        ):
            self.assertIn(required, launch)
        self.assertNotIn("cudaOccupancy", launch)
        self.assertNotIn("laguna_stage_fattn_vec_kv_f16_kernel", CUDA)
        self.assertIn("scratch_bytes = partial_bytes + meta_bytes", launch)

    def test_frozen_cuda_case_owns_exact_oracle_poison_and_fresh_rollback(self) -> None:
        for required in (
            "decode-attention-fattn-vec-frozen",
            "layer-00-k-t0-t511-kv6-kv7.f32",
            "layer-00-v-t0-t511-kv6-kv7.f32",
            "layer-00-q-t512-h41-h42.f32",
            "layer-00-attn-gated-t512-h41-h42.f32",
            "memcmp(selected_actual[arm], expected",
            "masked poison changed",
            "rollback-approximate",
            "ds4_gpu_test_laguna_fattn_vec_snapshot",
        ):
            self.assertIn(required, KERNEL_TEST)
        self.assertIn(
            "DS4_CUDA_NO_LAGUNA_FATTN_VEC_DECODE=1 "
            "./tests/test_cuda_laguna_kernels --case "
            "decode-attention-fattn-vec-frozen",
            MAKEFILE,
        )
        for required in (
            "ds4_gpu_laguna_fattn_vec_test_snapshot",
            "ds4_gpu_test_laguna_fattn_vec_reset",
            "ds4_gpu_test_laguna_fattn_vec_snapshot",
        ):
            self.assertIn(required, HEADER)
            self.assertIn(required, CUDA)

    def test_frozen_cuda_exact_failure_reports_bits_and_passed_scale(self) -> None:
        for required in (
            "first-bit-mismatch",
            "actual=%a (0x%08x)",
            "reference=%a (0x%08x)",
            "scale=%a (0x%08x)",
            "laguna_f32_bits(scale)",
        ):
            self.assertIn(required, KERNEL_TEST)


if __name__ == "__main__":
    unittest.main()
