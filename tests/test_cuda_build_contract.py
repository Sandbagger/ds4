#!/usr/bin/env python3
"""Host-only source contracts for standalone CUDA links and cleanup policy."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
DS4_HEADER = (ROOT / "ds4.h").read_text(encoding="utf-8")
GPU_HEADER = (ROOT / "ds4_gpu.h").read_text(encoding="utf-8")
DS4_SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")
LAGUNA_MODEL_TEST = (ROOT / "tests/test_cuda_laguna_model.c").read_text(
    encoding="utf-8"
)
LAGUNA_STREAM_TEST = (ROOT / "tests/test_cuda_laguna_stream.c").read_text(
    encoding="utf-8"
)
LAGUNA_KERNEL_TEST = (ROOT / "tests/test_cuda_laguna_kernels.c").read_text(
    encoding="utf-8"
)
LAGUNA_ATTENTION_AUTO_FIXTURE = (
    ROOT / "tests/test-vectors/laguna-attention-auto"
)
LAGUNA_ATTENTION_AUTO_FILES = {
    "layer-00-q-proj-t21.f32": (
        24576,
        "7650de1eb2539fa059cd61d0888be41e1ce53481ffd0f761d54087fe621329be",
    ),
    "layer-00-k-proj-t21.f32": (
        4096,
        "9604b2834c48eb8eb353149da1d4de1faf84c4529d997057aaab058b39fb6b4e",
    ),
    "layer-00-q-norm-weight.f32": (
        512,
        "9540bf5aba37eb7141a144324d0ddff69ce1fbf6b62dd6f91627978ed4e82067",
    ),
    "layer-00-k-norm-weight.f32": (
        512,
        "d08339e04c733249be70c9356ae8bd273194da9c7ca569943e74e6753caf4868",
    ),
    "layer-00-q-rope.f32": (
        540672,
        "ede9c00e83a4ad953900743104178f75af9219c85f66bda8d92d12ef3753c4da",
    ),
    "layer-00-k-rope.f32": (
        90112,
        "f961fc2d42616fccfc07d8d42eb10ec953f213cda9ee0d9d00bab012559cef4f",
    ),
    "layer-00-v-proj.f32": (
        90112,
        "d3f83952d2bc88275ef1af691c73b4b1bcedff41c5c0c5823f287b15a9f78c41",
    ),
    "layer-00-gate-proj.f32": (
        4224,
        "07fe34fc9bbe9e178da601b579c23037b616e4b1278ee960466e6b308663d3f1",
    ),
    "layer-00-attn-gated.f32": (
        540672,
        "f44669c93d81bbd22edb7dab4311af71b5880556a5646d3138aae1d5c0e4e3bd",
    ),
    "layer-01-q-rope.f32": (
        811008,
        "788af1e10d67728891ad9ab529d103033db664ce56fad967dd1cebaed5f74236",
    ),
    "layer-01-k-rope.f32": (
        90112,
        "6129c7273cb32a6baedd77465a817da815255cfbf43c260a3f53536a5363425f",
    ),
    "layer-01-v-proj.f32": (
        90112,
        "4a46e7ad3b6dc183090a2d3db37aeafb91a7370b01002603c9e7d5590f9e4cda",
    ),
    "layer-01-gate-proj.f32": (
        6336,
        "a29632741e53c51905152fcf4c0be9fda9c0008c690704a5079d71538b63beb8",
    ),
    "layer-01-attn-gated.f32": (
        811008,
        "43db94735d75e338303e6632bf5c063070e6f5ab01fa797dd606906e23e7d20c",
    ),
}
LAGUNA_ROUTER_AUTO_FIXTURE = ROOT / "tests/test-vectors/laguna-router-auto"
LAGUNA_ROUTER_AUTO_FILE_SIZES = {
    "layer-01-router-logits.f32": 22 * 256 * 4,
    "layer-01-router-bias.f32": 256 * 4,
    "layer-01-router-selected.i32": 22 * 10 * 4,
    "layer-01-router-weights.f32": 22 * 10 * 4,
}

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


def rule_recipe_lines(target: str) -> list[str]:
    target_line = re.search(rf"(?m)^{re.escape(target)}:[^\n]*$", MAKEFILE)
    if target_line is None:
        raise AssertionError(f"missing Makefile rule for {target}")
    recipe_lines: list[str] = []
    for line in MAKEFILE[target_line.end() + 1 :].splitlines():
        if not line.startswith("\t"):
            break
        recipe_lines.append(line)
    return recipe_lines


def source_function_body(source: str, signature: str, source_name: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing {source_name} function {signature}")
    brace = source.find("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"unterminated {source_name} function {signature}")


def function_body(signature: str) -> str:
    return source_function_body(CUDA_SOURCE, signature, "CUDA")


class CudaBuildContractTest(unittest.TestCase):
    def test_laguna_attention_auto_fixture_is_pinned_and_wired(self) -> None:
        manifest_path = LAGUNA_ATTENTION_AUTO_FIXTURE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "laguna-attention-auto-fixture/v3")
        self.assertEqual(
            manifest["poolside_commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(manifest["flash_attention"], "AUTO")
        self.assertEqual(
            manifest["shape"],
            {
                "tokens": 22,
                "query_heads": 48,
                "kv_heads": 8,
                "head_dim": 128,
                "position_start": 0,
                "kv_rows": 256,
            },
        )
        self.assertEqual(
            manifest["qk_norm_rope"]["shape"],
            {
                "layer": 0,
                "token": 21,
                "query_heads": 48,
                "kv_heads": 8,
                "head_dim": 128,
                "rotary_dim": 64,
            },
        )
        self.assertEqual(
            manifest["qk_norm_rope"]["parameters"],
            {
                "position_start": 21,
                "original_context": 8192,
                "frequency_base": 500000.0,
                "frequency_scale": 0.03125,
                "extension_factor": 1.0,
                "attention_factor": 1.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "epsilon": 1e-06,
            },
        )
        self.assertEqual(
            manifest["qk_norm_rope"]["weight_tensors"],
            {
                "query": "blk.0.attn_q_norm.weight",
                "key": "blk.0.attn_k_norm.weight",
            },
        )
        self.assertEqual(
            manifest["h72_oracle"]["shape"],
            {
                "layer": 1,
                "tokens": 22,
                "query_heads": 72,
                "kv_heads": 8,
                "head_dim": 128,
                "position_start": 0,
                "kv_rows": 256,
            },
        )
        self.assertEqual(
            manifest["h72_oracle"]["producer_commit"],
            "551ddb2c128f8e92ef0c0ea8e1b87a5e3f557de3",
        )
        self.assertEqual(
            manifest["h72_oracle"]["determinism"],
            {"successful_runs": 1},
        )
        for limits in (manifest["oracle"], manifest["h72_oracle"]):
            self.assertEqual(limits["global_max_abs_limit"], 0.0)
            self.assertEqual(limits["global_rms_limit"], 0.0)
            self.assertEqual(limits["per_head_max_abs_limit"], 0.0)
            self.assertEqual(limits["per_head_rms_limit"], 0.0)
        self.assertEqual(
            manifest["h72_oracle"]["input_callbacks"],
            {
                "layer-01-q-rope.f32": "Qcur_rope-1",
                "layer-01-k-rope.f32": "Kcur_rope-1",
                "layer-01-v-proj.f32": "Vcur-1",
                "layer-01-gate-proj.f32": "attn_gate_proj-1",
            },
        )
        self.assertEqual(
            manifest["h72_oracle"]["output_callback"],
            {"layer-01-attn-gated.f32": "attn_gated-1"},
        )
        self.assertEqual(
            manifest["qk_norm_rope"]["public_apis"],
            [
                "ds4_gpu_laguna_qk_head_rms_norm_rope_tensor",
                "ds4_gpu_laguna_head_rms_norm_rope_tensor",
            ],
        )
        self.assertEqual(
            manifest["qk_norm_rope"]["expected_output_slices"],
            {
                "query": {
                    "file": "layer-00-q-rope.f32",
                    "offset": 516096,
                    "bytes": 24576,
                    "sha256": "a7004aa3b85922043a5a6e1a4036af1d1cd937d3b69c3cb2f85a25e83e9a7303",
                },
                "key": {
                    "file": "layer-00-k-rope.f32",
                    "offset": 86016,
                    "bytes": 4096,
                    "sha256": "5e69e125564a5cbae961ca697aceb5b974f3efb3f3bbf21e05dcfc7232ac73a1",
                },
            },
        )
        for output_slice in manifest["qk_norm_rope"][
            "expected_output_slices"
        ].values():
            payload = (LAGUNA_ATTENTION_AUTO_FIXTURE / output_slice["file"]).read_bytes()
            start = output_slice["offset"]
            end = start + output_slice["bytes"]
            self.assertEqual(len(payload[start:end]), output_slice["bytes"])
            self.assertEqual(
                hashlib.sha256(payload[start:end]).hexdigest(),
                output_slice["sha256"],
            )
        self.assertEqual(set(manifest["files"]), set(LAGUNA_ATTENTION_AUTO_FILES))
        for name, (expected_size, expected_sha256) in (
            LAGUNA_ATTENTION_AUTO_FILES.items()
        ):
            with self.subTest(name=name):
                payload = (LAGUNA_ATTENTION_AUTO_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)
                self.assertEqual(
                    manifest["files"][name],
                    {"bytes": expected_size, "sha256": expected_sha256},
                )

        for required in (
            '"prefill-attention-frozen"',
            "run_prefill_attention_frozen_case",
            "run_prefill_attention_frozen_gqa9_case",
            '"poolside-auto-layer1-t22-gqa9"',
            '"layer-01-q-rope.f32"',
            '"layer-01-attn-gated.f32"',
            '"fast-shape-wrap-guard", 0u, 22u, 16u, 48u, 8u',
            "LAGUNA_ATTENTION_AUTO_FIXTURE_DIR",
            "22u, 256u, 48u, 8u",
            '"token20-head43"',
            "run_qk_norm_rope_frozen_t21_case",
            "poolside-auto-qk-t21",
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)
        self.assertNotIn("gqa9-derived", LAGUNA_KERNEL_TEST)
        for signature in (
            "static int run_prefill_attention_frozen_case(",
            "static int run_prefill_attention_frozen_gqa9_case(",
        ):
            frozen_attention_body = source_function_body(
                LAGUNA_KERNEL_TEST, signature, "tests/test_cuda_laguna_kernels.c"
            )
            self.assertRegex(
                frozen_attention_body,
                r"max_abs_limit\s*=\s*0\.0f",
            )
            self.assertRegex(
                frozen_attention_body,
                r"rms_limit\s*=\s*0\.0f",
            )
        kernel_main = source_function_body(
            LAGUNA_KERNEL_TEST, "int main(", "tests/test_cuda_laguna_kernels.c"
        )
        norm_start = kernel_main.index("if (run_norm) {")
        norm_end = kernel_main.index("if (run_decode", norm_start)
        self.assertIn(
            "run_qk_norm_rope_frozen_t21_case()",
            kernel_main[norm_start:norm_end],
        )
        frozen_qk_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_qk_norm_rope_frozen_t21_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        paired_call = frozen_qk_body.index(
            "ds4_gpu_laguna_qk_head_rms_norm_rope_tensor("
        )
        reset = frozen_qk_body.index("ds4_gpu_tensor_write(q", paired_call)
        single_call = frozen_qk_body.index(
            "ds4_gpu_laguna_head_rms_norm_rope_tensor(", reset
        )
        self.assertLess(paired_call, reset)
        self.assertLess(reset, single_call)
        self.assertEqual(
            frozen_qk_body.count("ds4_gpu_laguna_head_rms_norm_rope_tensor("),
            2,
        )
        for required in (
            "cudaDevAttrComputeCapabilityMajor",
            "cache_cap >= n_tokens",
        ):
            self.assertIn(required, CUDA_SOURCE)

    def test_laguna_attention_auto_requires_compiled_wmma_body(self) -> None:
        body = function_body(
            'extern "C" int ds4_gpu_laguna_attention_prefill_tensor('
        )
        attribute_query = re.search(
            r"cudaFuncGetAttributes\s*\(\s*&(?P<attributes>[A-Za-z_]\w*)\s*,"
            r"\s*laguna_attention_prefill_auto_mma32_kernel\s*\)"
            r"\s*==\s*cudaSuccess",
            body,
        )
        self.assertIsNotNone(attribute_query)
        attributes = re.escape(attribute_query.group("attributes"))
        self.assertRegex(
            body,
            rf"\(\s*{attributes}\.binaryVersion\s*>=\s*70\s*\|\|\s*"
            rf"{attributes}\.ptxVersion\s*>=\s*70\s*\)",
        )

    def test_laguna_router_auto_fixture_is_pinned_and_wired(self) -> None:
        manifest_path = LAGUNA_ROUTER_AUTO_FIXTURE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "laguna-router-auto-fixture/v1")
        self.assertEqual(
            manifest["poolside_commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(
            manifest["model"],
            {
                "bytes": 68248759648,
                "sha256": (
                    "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
                ),
            },
        )
        self.assertEqual(manifest["capture"]["flash_attention"], "AUTO")
        self.assertEqual(manifest["capture"]["layer"], 1)
        self.assertEqual(manifest["capture"]["tokens"], 22)
        self.assertRegex(manifest["capture"]["probe_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            manifest["capture"]["callbacks"],
            {
                "layer-01-router-logits.f32": "ffn_moe_logits-1",
                "layer-01-router-selected.i32": "ffn_moe_topk-1",
                "layer-01-router-weights.f32": "ffn_moe_weights_scaled-1",
            },
        )
        self.assertEqual(
            manifest["capture"]["fusion_preservation"],
            {
                "router_boundary": "ffn_moe_weights_scaled-1",
                "deferred_tensor": "ffn_moe_topk-1",
                "reason": (
                    "requesting top-k as its own evaluation boundary splits the CUDA "
                    "fusion; the probe saves its tensor pointer and copies it only at "
                    "the final scaled-weight boundary"
                ),
            },
        )
        self.assertEqual(
            manifest["capture"]["determinism"], {"successful_runs": 1}
        )
        self.assertEqual(
            manifest["fused_oracle"],
            {
                "source": {
                    "router_kernel": "ggml/src/ggml-cuda/topk-moe.cu:80-254",
                    "warp_reduction": "ggml/src/ggml-cuda/common.cuh:447-452",
                    "producer_sha256": (
                        "f815b8ce87c62e1551535d3096aa65d858f06ae4eb9e94fd1ed9b5b3235c8689"
                    ),
                },
                "cuda_flags": [
                    "-std=c++17",
                    "-O3",
                    "-use_fast_math",
                    "--generate-code=arch=compute_121a,code=[sm_121a]",
                    "-extended-lambda",
                    "-compress-mode=size",
                ],
                "inputs": [
                    "layer-01-router-logits.f32",
                    "layer-01-router-bias.f32",
                ],
                "outputs": [
                    "layer-01-router-selected.i32",
                    "layer-01-router-weights.f32",
                ],
                "determinism": {
                    "runs": 2,
                    "outputs_bit_exact": True,
                    "actual_poolside_capture_matches": True,
                },
            },
        )
        self.assertEqual(
            manifest["router"],
            {
                "experts": 256,
                "experts_used": 10,
                "weight_scale": 2.5,
                "gating": "sigmoid",
                "selection": "biased top-k",
                "normalization": (
                    "Poolside CUDA tree reduction, reciprocal multiply, then scale multiply"
                ),
            },
        )
        self.assertEqual(
            manifest["bias"],
            {
                "tensor": "blk.1.exp_probs_b.bias",
                "type": "F32",
                "elements": 256,
                "absolute_model_offset": 893238112,
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {
                "selected_ids_exact": True,
                "weights_max_abs_limit": 0.0,
                "weights_rms_limit": 0.0,
            },
        )

        self.assertEqual(
            set(manifest["files"]), set(LAGUNA_ROUTER_AUTO_FILE_SIZES)
        )
        for name, expected_size in LAGUNA_ROUTER_AUTO_FILE_SIZES.items():
            with self.subTest(name=name):
                file_contract = manifest["files"][name]
                self.assertEqual(file_contract["bytes"], expected_size)
                self.assertRegex(file_contract["sha256"], r"^[0-9a-f]{64}$")
                payload = (LAGUNA_ROUTER_AUTO_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    file_contract["sha256"],
                )

        for required in (
            '"router-frozen"',
            "run_router_frozen_case",
            "LAGUNA_ROUTER_AUTO_FIXTURE_DIR",
            '"layer-01-router-logits.f32"',
            '"layer-01-router-bias.f32"',
            '"layer-01-router-selected.i32"',
            '"layer-01-router-weights.f32"',
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)

        router_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_router_frozen_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        self.assertIn("ds4_gpu_glm_router_select_batch_tensor(", router_body)
        self.assertIn("cudaDeviceSynchronize()", router_body)
        self.assertIn("ds4_gpu_tensor_read(selected", router_body)
        self.assertIn("ds4_gpu_tensor_read(weights", router_body)
        self.assertRegex(
            router_body,
            r"memcmp\(\s*selected_actual,\s*selected_reference,",
        )
        self.assertRegex(
            router_body,
            r"memcmp\(\s*weights_actual,\s*weights_reference,",
        )
        self.assertRegex(router_body, r"\b0u,\s*logits,")
        self.assertRegex(router_body, r"\b256u,\s*10u,\s*2\.5f,\s*22u")

        kernel_main = source_function_body(
            LAGUNA_KERNEL_TEST, "int main(", "tests/test_cuda_laguna_kernels.c"
        )
        self.assertIn("run_router_frozen_case()", kernel_main)

    def test_laguna_attention_auto_is_qualified_only_for_gb10(self) -> None:
        body = function_body(
            'extern "C" int ds4_gpu_laguna_attention_prefill_tensor('
        )
        self.assertIn("cudaDevAttrComputeCapabilityMinor", body)
        self.assertRegex(body, r"\bcompute_major\s*==\s*12\b")
        self.assertRegex(body, r"\bcompute_minor\s*==\s*1\b")
        self.assertNotRegex(body, r"\bcompute_major\s*>=\s*8\b")

    def test_laguna_stream_links_cuda_lifecycle_test_hooks(self) -> None:
        hook_object = "tests/ds4_cuda_laguna_stream_test_hooks.o"
        hook_prerequisites = rule_prerequisites(hook_object)
        self.assertIn("ds4_cuda.cu", hook_prerequisites)

        hook_recipe = re.search(
            rf"(?m)^{re.escape(hook_object)}:[^\n]*\n\t([^\n]+)$",
            MAKEFILE,
        )
        self.assertIsNotNone(hook_recipe)
        self.assertIn("-DDS4_TEST_HOOKS", hook_recipe.group(1))

        stream_objects = rule_prerequisites(
            "tests/test_cuda_laguna_stream"
        ).split()
        self.assertIn(hook_object, stream_objects)
        self.assertNotIn("ds4_cuda.o", stream_objects)

    def test_compact_lifecycle_compile_units_track_identity_headers(self) -> None:
        cuda_prerequisites = rule_prerequisites("ds4_cuda.o")
        self.assertIn("ds4_laguna_plan.h", cuda_prerequisites)

        ds4_test_prerequisites = rule_prerequisites("ds4_test.o")
        self.assertIn("ds4_gpu.h", ds4_test_prerequisites)

        gpu_header_targets = {
            match.group(1)
            for match in re.finditer(
                r"(?m)^([^:\s]+\.o):[^\n]*\bds4_gpu\.h\b", MAKEFILE
            )
        }
        gpu_header_targets.add("ds4_test.o")
        for target in sorted(gpu_header_targets):
            with self.subTest(target=target):
                self.assertIn(
                    "ds4_laguna_plan.h", rule_prerequisites(target)
                )

        self.assertIn(
            "ds4.h", rule_prerequisites("tests/test_runtime_cpp_link.o")
        )

    def test_standalone_cuda_links_include_runtime_tracker(self) -> None:
        for target in STANDALONE_CUDA_TARGETS:
            with self.subTest(target=target):
                prerequisites = rule_prerequisites(target)
                self.assertIn("ds4_cuda.o", prerequisites)
                self.assertIn("ds4_runtime.o", prerequisites)

    def test_poolside_mmvq_uses_the_active_configured_physical_device(
        self,
    ) -> None:
        body = function_body(
            'extern "C" int ds4_gpu_matmul_q8_0_poolside_tensor('
        )
        n1_prefix, n22_marker, _ = body.partition("n_tok == 22u")
        self.assertTrue(n22_marker, "missing Poolside n_tok=22 branch")
        self.assertNotIn("g_n_gpus > 1", n1_prefix)
        self.assertNotIn(": 0;", n1_prefix)
        device_guard = re.search(
            r"if\s*\(logical_tier >= 0 && logical_tier < g_n_gpus\)\s*\{"
            r"(?P<body>.*?)"
            r"\n\s*\}",
            n1_prefix,
            re.DOTALL,
        )
        self.assertIsNotNone(device_guard, "missing logical-tier bounds")
        guarded = device_guard.group("body")
        self.assertIn(
            "g_gpu[logical_tier].device_id", guarded
        )
        self.assertIn("cudaGetDevice(&current_device)", guarded)
        self.assertIn("current_device == physical_device", guarded)
        self.assertNotIn("cudaSetDevice", n1_prefix)

    def test_laguna_decode_routes_all_quantized_projections_through_matmul(
        self,
    ) -> None:
        body = source_function_body(
            DS4_SOURCE, "static bool laguna_graph_forward_token(", "ds4.c"
        )
        self.assertNotIn("ds4_gpu_matmul_q8_0_pair_tensor", body)
        self.assertNotIn("ds4_gpu_shared_mid_swiglu_q8_0_tensor", body)

        projections = (
            ("q", "attn_q", "attn_norm"),
            ("k", "attn_k", "attn_norm"),
            ("v", "attn_v", "attn_norm"),
            ("gate", "attn_gate", "attn_norm"),
            ("ffn_gate", "ffn_gate_shexp", "ffn_norm"),
            ("ffn_up", "ffn_up_shexp", "ffn_norm"),
        )
        for output, weight, input_tensor in projections:
            with self.subTest(output=output, weight=weight):
                self.assertRegex(
                    body,
                    rf"laguna_graph_matmul\(\s*g->{output},\s*model,"
                    rf"\s*l->{weight},\s*g->{input_tensor},\s*1\s*\)",
                )

        self.assertRegex(
            body,
            r"ds4_gpu_swiglu_tensor\(\s*g->ffn_mid,\s*g->ffn_gate,"
            r"\s*g->ffn_up,\s*DS4_N_FF_SHARED,\s*0\.0f,\s*1\.0f\s*\)",
        )

    def test_laguna_batch_layer_probe_is_test_only_and_nonperturbing(self) -> None:
        self.assertIn("#ifdef DS4_TEST_HOOKS", DS4_SOURCE)
        self.assertIn("DS4_LAGUNA_DIAG_DIR", DS4_SOURCE)
        self.assertIn("DS4_LAGUNA_DIAG_LAYER", DS4_SOURCE)
        self.assertIn("laguna_graph_diag_detail_layer", DS4_SOURCE)
        self.assertIn("laguna_graph_diag_dump_tensor", DS4_SOURCE)

        body = source_function_body(
            DS4_SOURCE, "static bool laguna_graph_forward_batch(", "ds4.c"
        )
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint\(\s*g->cur,\s*n_tokens,"
            r"\s*DS4_N_EMBD,\s*-1,\s*\"embd\"\s*\)",
        )
        layer_dump = body.find(
            'laguna_graph_diag_checkpoint(\n                    g->next,\n'
        )
        layer_swap = body.find("ds4_gpu_tensor *tmp = g->cur;")
        self.assertGreaterEqual(layer_dump, 0, "missing full layer-output probe")
        self.assertGreater(layer_swap, layer_dump, "probe must precede cur/next swap")
        self.assertIn(
            "(uint64_t)n_tokens * width * sizeof(float)", DS4_SOURCE
        )
        self.assertIn('"%s/layer-%02d.f32"', DS4_SOURCE)
        self.assertIn('"%s/layer-%02d-%s.f32"', DS4_SOURCE)
        for tensor, stage, width in (
            ("attn_norm", "attn-norm", r"DS4_N_EMBD"),
            ("q", "q-proj", r"n_head\s*\*\s*DS4_N_HEAD_DIM"),
            ("k", "k-proj", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("v", "v-proj", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("gate", "gate-proj", r"n_head"),
            ("q", "q-rope", r"n_head\s*\*\s*DS4_N_HEAD_DIM"),
            ("k", "k-rope", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("heads", "attn-gated", r"n_head\s*\*\s*DS4_N_HEAD_DIM"),
            ("attn_out", "attn-o-proj", r"DS4_N_EMBD"),
            ("after_attn", "ffn-inp", r"DS4_N_EMBD"),
            ("ffn_norm", "ffn-norm", r"DS4_N_EMBD"),
            ("ffn_out", "ffn-out", r"DS4_N_EMBD"),
        ):
            self.assertRegex(
                body,
                rf"if \(ok && il == \(uint32_t\)detail_layer\) \{{\s*failed_stage = \"{stage} diagnostic\";"
                rf"\s*ok = laguna_graph_diag_checkpoint\(\s*g->{tensor},"
                rf"\s*n_tokens,\s*{width},\s*\(int\)il,\s*\"{stage}\"\s*\);",
            )
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint\(\s*g->logits,\s*1,"
            r"\s*DS4_N_VOCAB,\s*-1,\s*\"logits\"\s*\)",
        )
        checkpoint = source_function_body(
            DS4_SOURCE, "static bool laguna_graph_diag_checkpoint(", "ds4.c"
        )
        self.assertNotIn(
            "ds4_gpu_commands_active", checkpoint,
            "CUDA intentionally reports no active command buffer",
        )
        self.assertIn("ds4_gpu_end_commands()", checkpoint)

        self.assertIn("DS4_LAGUNA_DIAG_DIR", LAGUNA_MODEL_TEST)
        self.assertIn("memcmp(baseline_logits, probed_logits, VECTOR_BYTES)",
                      LAGUNA_MODEL_TEST)
        self.assertIn("short-layer-diag PASS files=62", LAGUNA_MODEL_TEST)
        run_short = source_function_body(
            LAGUNA_MODEL_TEST, "static bool run_short(",
            "tests/test_cuda_laguna_model.c"
        )
        diagnostic_branch = run_short.split(
            'const char *diag_dir_env = getenv("DS4_LAGUNA_DIAG_DIR");', 1
        )[1].split("ds4_session *session = NULL;", 1)[0]
        self.assertNotIn("compare_session_oracle", diagnostic_branch)
        self.assertEqual(
            diagnostic_branch.count('unsetenv("DS4_LAGUNA_DIAG_DIR")'),
            2,
            "short diagnostic must consume its environment before later model cases",
        )
        self.assertGreater(
            diagnostic_branch.rfind('unsetenv("DS4_LAGUNA_DIAG_DIR")'),
            diagnostic_branch.find('"short-diag-probed"'),
            "diagnostic environment must be cleared after the probed session",
        )
        main = source_function_body(
            LAGUNA_MODEL_TEST, "int main(void)", "tests/test_cuda_laguna_model.c"
        )
        self.assertIn("const bool diagnostic_mode", main)
        diagnostic_guard = main.find("if (!diagnostic_mode) {")
        swa_case = main.find('engine, "swa-513"')
        self.assertGreaterEqual(diagnostic_guard, 0)
        self.assertGreater(swa_case, diagnostic_guard)
        suite_end = main.find("\n    }\n\n    ds4_engine_close", diagnostic_guard)
        self.assertGreater(suite_end, swa_case)
        guarded_suite = main[diagnostic_guard:suite_end]
        for call in (
            "run_raw_frontier(",
            "run_deep_exact_context(",
            "run_decode_batch(",
            "run_mixed_batch(",
        ):
            self.assertIn(call, guarded_suite)
        failed = main.find("if (!ok) return 1;")
        diagnostic_done = main.find("if (diagnostic_mode) return 0;")
        full_suite_pass = main.find('"test_cuda_laguna_model PASS oracle=poolside')
        self.assertGreater(diagnostic_done, failed)
        self.assertGreater(full_suite_pass, diagnostic_done)

    def test_noncompact_cleanup_keeps_best_effort_sync_policy(self) -> None:
        body = function_body('extern "C" void ds4_gpu_cleanup(void)')
        preamble, separator, _ = body.partition("g_current_logical_tier = -1;")
        self.assertTrue(separator, "cleanup state reset moved or disappeared")
        self.assertIn("compact_cleanup_required", preamble)
        self.assertIn("cuda_laguna_compact_destroy_checked", preamble)
        self.assertIn("(void)cudaDeviceSynchronize();", preamble)
        self.assertNotIn("cleanup_sync_error", preamble)

    def test_failed_compact_create_uses_authoritative_cuda_ownership(self) -> None:
        declaration = "ds4_gpu_laguna_compact_ownership_pending("
        self.assertIn(declaration, GPU_HEADER)
        body = function_body(
            'extern "C" bool ds4_gpu_laguna_compact_ownership_pending('
        )
        self.assertIn("g_laguna_compact_mutex", body)
        self.assertIn("DS4_LAGUNA_COMPACT_IDLE", body)
        self.assertIn("g_laguna_compact_storage.tracker == tracker", body)

        close_start = DS4_SOURCE.index("void ds4_engine_close(ds4_engine *e)")
        close_body = DS4_SOURCE[close_start:]
        ownership_query = close_body.index(declaration)
        generic_cleanup = close_body.index("ds4_gpu_cleanup();")
        engine_free = close_body.index("free(e);")
        self.assertLess(ownership_query, generic_cleanup)
        self.assertLess(ownership_query, engine_free)

    def test_laguna_engine_options_expose_only_a_hidden_qualification_fd(self) -> None:
        end = DS4_HEADER.index("} ds4_engine_options;")
        start = DS4_HEADER.rfind("typedef struct {", 0, end)
        self.assertGreaterEqual(start, 0)
        body = DS4_HEADER[start:end]
        self.assertRegex(body, r"\bint\s+qualification_model_fd\s*;")
        self.assertRegex(body, r"\bbool\s+qualification_model_fd_set\s*;")
        self.assertLess(
            body.index("ds4_tp_options tp;"), body.index("qualification_model_fd")
        )
        for frontend in ("ds4_cli.c", "ds4_server.c", "ds4_bench.c"):
            with self.subTest(frontend=frontend):
                source = (ROOT / frontend).read_text(encoding="utf-8")
                self.assertNotIn("qualification_model_fd", source)

    def test_laguna_model_loader_duplicates_the_inherited_fd_atomically(self) -> None:
        for needle in (
            "F_DUPFD_CLOEXEC",
            "qualification_model_fd_set",
            "qualification_model_fd",
        ):
            with self.subTest(needle=needle):
                if needle not in DS4_SOURCE:
                    raise AssertionError(f"ds4.c is missing {needle}")

    def test_both_laguna_model_harnesses_forward_fd_to_engine_options(self) -> None:
        for name, source in (
            ("model", LAGUNA_MODEL_TEST),
            ("stream", LAGUNA_STREAM_TEST),
        ):
            with self.subTest(name=name):
                self.assertIn('getenv("DS4_TEST_MODEL_FD")', source)
                self.assertRegex(
                    source,
                    r"\.qualification_model_fd\s*=\s*model_fd\s*,",
                )
                self.assertRegex(
                    source,
                    r"\.qualification_model_fd_set\s*=\s*model_fd_set\s*,",
                )

    def test_resident_gate_has_one_runner_recipe_and_build_only_prerequisites(
        self,
    ) -> None:
        prerequisites = rule_prerequisites("test-cuda-laguna-resident")
        for target in (
            "tests/test_cuda_laguna_kernels",
            "tests/test_cuda_laguna_model",
        ):
            self.assertIn(target, prerequisites)
        recipe_lines = rule_recipe_lines("test-cuda-laguna-resident")
        self.assertEqual(
            len(recipe_lines), 1, "resident gate must own fd 9 in one recipe shell"
        )
        recipe = recipe_lines[0].strip()
        self.assertTrue(
            recipe.endswith("tests/run_cuda_laguna_gate.sh resident"), recipe
        )

    def test_resident_gate_make_render_never_interpolates_hostile_values(self) -> None:
        model_payload = 'model"; printf MAKE_MODEL_INJECTED >&2; #.gguf'
        tokenizer_payload = "revision'; printf MAKE_TOKEN_INJECTED >&2; #"
        completed = subprocess.run(
            [
                "make",
                "-n",
                "UNAME_S=Linux",
                "-o",
                "tests/test_cuda_laguna_kernels",
                "-o",
                "tests/test_cuda_laguna_model",
                f"DS4_TEST_MODEL={model_payload}",
                f"LAGUNA_TOKENIZER_RUNTIME_COMMIT={tokenizer_payload}",
                "test-cuda-laguna-resident",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rendered = completed.stdout.strip()
        self.assertEqual(rendered, "tests/run_cuda_laguna_gate.sh resident")
        self.assertNotIn(model_payload, rendered)
        self.assertNotIn(tokenizer_payload, rendered)
        self.assertNotIn(";", rendered)

    def test_c7_gate_is_phony(self) -> None:
        phony_targets = rule_prerequisites(".PHONY").split()
        self.assertIn("test-cuda-laguna-c7", phony_targets)

    def test_c7_gate_has_exact_cuda_prerequisites_and_runner_recipe(
        self,
    ) -> None:
        prerequisites = rule_prerequisites("test-cuda-laguna-c7").split()
        self.assertEqual(
            prerequisites,
            [
                "tests/test_cuda_laguna_kernels",
                "tests/test_cuda_laguna_model",
                "tests/test_cuda_laguna_stream",
            ],
        )
        self.assertEqual(
            rule_recipe_lines("test-cuda-laguna-c7"),
            ["\ttests/run_cuda_laguna_gate.sh c7"],
            "C7 gate must retain fd 9 in one exact runner process",
        )

    def test_c7_gate_make_render_never_interpolates_hostile_values(self) -> None:
        model_payload = 'model"; printf MAKE_MODEL_INJECTED >&2; #.gguf'
        tokenizer_payload = "revision'; printf MAKE_TOKEN_INJECTED >&2; #"
        completed = subprocess.run(
            [
                "make",
                "-n",
                "UNAME_S=Linux",
                "-o",
                "tests/test_cuda_laguna_kernels",
                "-o",
                "tests/test_cuda_laguna_model",
                "-o",
                "tests/test_cuda_laguna_stream",
                f"DS4_TEST_MODEL={model_payload}",
                f"LAGUNA_TOKENIZER_RUNTIME_COMMIT={tokenizer_payload}",
                "test-cuda-laguna-c7",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rendered = completed.stdout.strip()
        self.assertEqual(rendered, "tests/run_cuda_laguna_gate.sh c7")
        self.assertNotIn(model_payload, rendered)
        self.assertNotIn(tokenizer_payload, rendered)
        self.assertNotIn(";", rendered)

    def test_gate_make_exports_do_not_expand_caller_make_syntax(self) -> None:
        gates = (
            (
                "resident",
                (
                    "tests/test_cuda_laguna_kernels",
                    "tests/test_cuda_laguna_model",
                ),
            ),
            (
                "c7",
                (
                    "tests/test_cuda_laguna_kernels",
                    "tests/test_cuda_laguna_model",
                    "tests/test_cuda_laguna_stream",
                ),
            ),
        )
        for mode, prerequisites in gates:
            with self.subTest(mode=mode):
                model_sentinel = f"LAGUNA_{mode.upper()}_MODEL_MAKE_EXPANDED"
                tokenizer_sentinel = (
                    f"LAGUNA_{mode.upper()}_TOKENIZER_MAKE_EXPANDED"
                )
                arguments = ["make", "-n", "UNAME_S=Linux"]
                for prerequisite in prerequisites:
                    arguments.extend(("-o", prerequisite))
                arguments.extend(
                    (
                        f"DS4_TEST_MODEL=$(warning {model_sentinel})",
                        "LAGUNA_TOKENIZER_RUNTIME_COMMIT="
                        f"$(warning {tokenizer_sentinel})",
                        f"test-cuda-laguna-{mode}",
                    )
                )

                completed = subprocess.run(
                    arguments,
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(
                    completed.stdout.strip(),
                    f"tests/run_cuda_laguna_gate.sh {mode}",
                )
                emitted = completed.stdout + completed.stderr
                self.assertNotIn(model_sentinel, emitted)
                self.assertNotIn(tokenizer_sentinel, emitted)

    def test_c7_gate_rejects_darwin_as_unsupported(self) -> None:
        completed = subprocess.run(
            ["make", "UNAME_S=Darwin", "test-cuda-laguna-c7"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        diagnostic = (completed.stdout + completed.stderr).lower()
        for required in (
            "test-cuda-laguna-c7",
            "unsupported",
            "cuda",
            "linux",
        ):
            with self.subTest(required=required):
                self.assertIn(required, diagnostic)


if __name__ == "__main__":
    unittest.main()
