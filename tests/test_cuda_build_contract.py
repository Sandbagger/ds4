#!/usr/bin/env python3
"""Host-only source contracts for standalone CUDA links and cleanup policy."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
CUDA_SOURCE = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
DS4_HEADER = (ROOT / "ds4.h").read_text(encoding="utf-8")
GPU_HEADER = (ROOT / "ds4_gpu.h").read_text(encoding="utf-8")
GPU_MGPU_HEADER = (ROOT / "ds4_gpu_mgpu.h").read_text(encoding="utf-8")
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
LAGUNA_Q4_MMQ_AUTO_FIXTURE = ROOT / "tests/test-vectors/laguna-q4-mmq-auto"
LAGUNA_Q4_MMQ_AUTO_FILES = {
    "input-token-00.f32": (
        12288,
        "fdf4b1ac532fc5775d1af92bcfeb37d942b64fef1a76d31ec8c0813795834605",
    ),
    "expert-246-row-000-gate.q4k": (
        1728,
        "9e20e5fe216f7738b2169a3f00eade92660274db11ab2ccb2fab6358d39a1e3a",
    ),
    "expert-246-row-000-up.q4k": (
        1728,
        "63c63676594b44ff20a5b6e78289202bbe16c0aff97594b391ca2b930b2f2a8f",
    ),
    "gate.f32": (
        4,
        "eb0bbc325acdcc0cd9ed7f16b6738a199f30b4f4267de9d6cf1edd983a533f3c",
    ),
    "up.f32": (
        4,
        "5d1643e77e5b23d36619b309cccfa933b7e84483ce75e742fbb86919b2c96bca",
    ),
    "swiglu.f32": (
        4,
        "f6ea0e74779aa61d39171891d49c64a373b41aa5f67facf23d7577be79cb5eb6",
    ),
}
Q4K_MMVQ_MICROSCOPE_TEST = (
    ROOT / "tests/test_cuda_q4k_mmvq_microscope.c"
).read_text(encoding="utf-8")
Q4K_MMVQ_MICROSCOPE_FIXTURE = (
    ROOT / "tests/test-vectors/q4k-mmvq-microscope-auto"
)
Q4K_MMVQ_MICROSCOPE_FILES = {
    "input.f32": (
        12288,
        "eabe89d1d9a4bdc660e5759c2a20d347d4dedaec1e617a44f3244cfe7985ef0e",
    ),
    "input.q8_1": (
        3456,
        "8df5b6ef5738aed267bccb6afd3b0deec21cd44e3a09908e88356cfa24d7de0a",
    ),
    "weight-row.q4k": (
        1728,
        "b42e00f452c044c1cf8679b5340a9a0738854576ad7aa294076ff4a3ed870fe5",
    ),
    "poolside-output.f32": (
        4,
        "6bea612b1933fa44f45d2731a553d098ccf4cd5e63a45ddee24f81793606244b",
    ),
}
F32_MMVF_MICROSCOPE_TEST_PATH = (
    ROOT / "tests/test_cuda_f32_mmvf_microscope.c"
)
F32_MMVF_MICROSCOPE_FIXTURE = (
    ROOT / "tests/test-vectors/f32-mmvf-microscope-auto"
)
LAGUNA_Q4_L2_AUTO_FIXTURE = ROOT / "tests/test-vectors/laguna-q4-l2-auto"
LAGUNA_Q4_L2_AUTO_FILES = {
    "mid-pair0.f32": (
        4096,
        "803af6831f6c89df4bac4d788821aded26edd61e4742f1929d3e5feaf30fdece",
    ),
    "expected-col-l2-pair0.f32": (
        4,
        "7288027f7f191e69e6832ec1bc80ff2da99012560990cfc888032496fdd35778",
    ),
    "expected-down-input-pair0.f32": (
        4096,
        "c058fcac9dc6a95dbf0b3aa59bbcc84f28ae1f8233ed273d59eecda23a30d0a1",
    ),
    "mid-decode.f32": (
        40960,
        "c43a2d48c6931726245e9a91f31cc669dd2786a3a2970626e904f48b5bc5567e",
    ),
    "expected-col-l2-decode.f32": (
        40,
        "c84da10d5bb0cf015e07a441824fb1223261ee44320ef2be3d5936e1f5bc8086",
    ),
    "expected-down-input-decode.f32": (
        40960,
        "22a986deadc4750a1760fbc72f58f233ceab87816697c97289f80688ea24e69a",
    ),
}
LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE = (
    ROOT / "tests/test-vectors/laguna-moe-residual-auto"
)
LAGUNA_MOE_RESIDUAL_AUTO_FILES = {
    "residual-token0.f32": (
        12288,
        "a164ed17c7ad1c051f3a01a2e73b58f9208656970b5eaeb854d311134dfa30e6",
    ),
    "moe-token0.f32": (
        12288,
        "a8449f64a6536c8012c43d24a558be01434e37948bff550b20e9654a11bbbfc6",
    ),
    "shared-token0.f32": (
        12288,
        "9b9ef36681633aae3897bb6a54f5f2f5fd3877eac035b72c6ab92c06ef132219",
    ),
    "expected-token0.f32": (
        12288,
        "0933abacd443f5073def7d27fb8d8040ef98e63c32f59c879eabdc6b06fe78d3",
    ),
}
LAGUNA_C7_ORACLE_PRODUCER = ROOT / "tests/oracle-producers/laguna-c7"
LAGUNA_C7_ORACLE_PRODUCER_FILES = {
    "capture_poolside_laguna_moe.sh",
    "compare_laguna_moe_execution.py",
    "poolside-token513-layer1-capture.json",
    "poolside-l2-callback.patch",
    "probe_ds4_laguna_behavior.c",
    "probe_ds4_laguna_moe.c",
    "probe_poolside_laguna_behavior.cpp",
    "probe_poolside_laguna_token513_moe.cpp",
    "probe_poolside_laguna_moe.cpp",
    "short.tokens.i32",
    "token513-poolside-mmvq-experiment.json",
    "token513-router-quality-proxies.json",
    "token513-layer1-comparison.json",
    "token513-layer1-run.json",
    "verify_poolside_laguna_moe.py",
}
LAGUNA_C7_TOOLCHAIN = {
    "cmake": {
        "path": "/usr/bin/cmake",
        "realpath": "/usr/bin/cmake",
        "sha256": "05e8a03ddf5a5075139760bcf4fc5bb73112246c5cd796dc85854589cfa502cb",
        "version": "3.28.3",
    },
    "cxx": {
        "path": "/usr/bin/c++",
        "realpath": "/usr/bin/aarch64-linux-gnu-g++-13",
        "sha256": "fc02363794280f404c6ca6f5da1c8fe469be902e9de140d35d8573bb3393f53b",
        "version": "13.3.0",
        "target": "aarch64-linux-gnu",
    },
    "cuda": {
        "path": "/usr/local/cuda/bin/nvcc",
        "realpath": "/usr/local/cuda-13.0/bin/nvcc",
        "sha256": "fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1",
        "version": "13.0.88",
        "release": "13.0",
        "build": "cuda_13.0.r13.0/compiler.36424714_0",
    },
    "make": {
        "path": "/usr/bin/gmake",
        "realpath": "/usr/bin/make",
        "sha256": "abbecc214fcadef6530d1cc137bde7ceb74cae3f6b992c7d10a8669a7314c7e0",
        "version": "4.3",
    },
}
LAGUNA_C7_GENERATED_RECIPES = {
    "CMakeCache.txt": {
        "normalized_bytes": 50493,
        "normalized_sha256": (
            "42bb0a751557d3948e838a94417b2d5895c2b9d97f73258a8c1736eddfcdf7f7"
        ),
    },
    "Makefile": {
        "normalized_bytes": 40940,
        "normalized_sha256": (
            "13525b712d239479300975201e45fe94d3431c97debdf59017b678a32f605deb"
        ),
    },
    "CMakeFiles/Makefile2": {
        "normalized_bytes": 166721,
        "normalized_sha256": (
            "b00dcc163f0c2029a700f5064d1fd330bc34b8f2fdd7b06ea9dace487c78c9cc"
        ),
    },
    "CMakeFiles/Makefile.cmake": {
        "normalized_bytes": 13717,
        "normalized_sha256": (
            "a673dc5fd8135e9cc11a1b0f1936fec4337013116c959f558da00572d62ce1f6"
        ),
    },
    "src/CMakeFiles/llama.dir/build.make": {
        "normalized_bytes": 261487,
        "normalized_sha256": (
            "7f78f598d93320951125e730d0dfc644f9a24d7faee807aadabe93cd2dca4cb2"
        ),
    },
    "src/CMakeFiles/llama.dir/flags.make": {
        "normalized_bytes": 524,
        "normalized_sha256": (
            "d15fca142d5f07f5d6e4742f78d6bebd9a57ee5ed5ba5a2ef3f557a4a3467ecc"
        ),
    },
    "ggml/src/CMakeFiles/ggml.dir/build.make": {
        "normalized_bytes": 6642,
        "normalized_sha256": (
            "b13073a9cb49e1438d4a4ae3b80262dce6eb5a50d60aa6d74bd200ac34ba3559"
        ),
    },
    "ggml/src/CMakeFiles/ggml.dir/flags.make": {
        "normalized_bytes": 539,
        "normalized_sha256": (
            "a7ca9091f27f9c8c98ab598169447f01328da0ce91de484c38f0d7f7bd9132ce"
        ),
    },
    "ggml/src/CMakeFiles/ggml.dir/link.txt": {
        "normalized_bytes": 389,
        "normalized_sha256": (
            "f7058e0c7fd0abbac30b276b0f9cc823757499cfe7bba25e228d7e30cadbb9c9"
        ),
    },
    "ggml/src/CMakeFiles/ggml-base.dir/build.make": {
        "normalized_bytes": 17792,
        "normalized_sha256": (
            "1797ff4ba4079b1291b9df1925a11ae81531069c9c9f58f05b2fe9dee02e679b"
        ),
    },
    "ggml/src/CMakeFiles/ggml-base.dir/flags.make": {
        "normalized_bytes": 1127,
        "normalized_sha256": (
            "c6f46c65d943d585294b52599cbff012316654603b84326f4f8b7cc76f133fd7"
        ),
    },
    "ggml/src/CMakeFiles/ggml-base.dir/link.txt": {
        "normalized_bytes": 596,
        "normalized_sha256": (
            "960a99e84054616ad114500c4ec131fcdfdbda9a090657831bc56be715e57885"
        ),
    },
    "ggml/src/CMakeFiles/ggml-cpu.dir/build.make": {
        "normalized_bytes": 29390,
        "normalized_sha256": (
            "0266b57e44f07d4eacaa4ba3317a8b18f6bd1838cce0a0d193ded049c1b66c75"
        ),
    },
    "ggml/src/CMakeFiles/ggml-cpu.dir/flags.make": {
        "normalized_bytes": 1293,
        "normalized_sha256": (
            "f14f27e2a27227c8f8ef4098bc3f6bdc95a3367c365cdedca7a0547d3454bf18"
        ),
    },
    "ggml/src/CMakeFiles/ggml-cpu.dir/link.txt": {
        "normalized_bytes": 1007,
        "normalized_sha256": (
            "c19ebcb7617b1c521a61a0db6255e6726ea85f7bddd7df0600f40491ffdd853d"
        ),
    },

    "src/CMakeFiles/llama.dir/link.txt": {
        "normalized_bytes": 7512,
        "normalized_sha256": (
            "87adec14a16996d263489140320dcc78e5ad6f8c032b4d6648c9d2fbd41d717c"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/build.make": {
        "normalized_bytes": 290236,
        "normalized_sha256": (
            "599e5e782d7c0f8e73c8464ad4beca16277a4d36ebfbf6693d06be6eabbd0ac3"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make": {
        "normalized_bytes": 735,
        "normalized_sha256": (
            "6b889101ab32e1bab82b6f3787aa72df75f489e707a57b0e15e7e9ab90f1cb4f"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp": {
        "normalized_bytes": 120,
        "normalized_sha256": (
            "cf57567affcd762a2f970854f259b0ac2e2a9568edb769770269190d07710f71"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/objects1.rsp": {
        "normalized_bytes": 8482,
        "normalized_sha256": (
            "abaaaefdaba0dada21c22478892d94f278b8d878c2faf7283dc4941d725aafc7"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/linkLibs.rsp": {
        "normalized_bytes": 407,
        "normalized_sha256": (
            "76af44245dae3a42d87b64e62f08d05a160218c998934d260b78974cec9c1cfd"
        ),
    },
    "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/link.txt": {
        "normalized_bytes": 266,
        "normalized_sha256": (
            "114473590c21a5cd20ca612883cd8efbb644e1f14e484347cb318ed466dce80c"
        ),
    },
}

STANDALONE_CUDA_TARGETS = (
    "tests/cuda_long_context_smoke",
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
    def test_score_official_compiles_c_before_cuda_link(self) -> None:
        object_rule = (
            "gguf-tools/quality-testing/score_official.o: "
            "gguf-tools/quality-testing/score_official.c ds4.h\n"
            "\t$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) "
            "-I. -c -o $@ $<"
        )
        self.assertIn(object_rule, MAKEFILE)
        cuda_link_rule = (
            "gguf-tools/quality-testing/score_official: "
            "gguf-tools/quality-testing/score_official.o "
            "$(CORE_OBJS) rax.o\n"
            "\t$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)"
        )
        self.assertIn(cuda_link_rule, MAKEFILE)

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

    def test_laguna_q4_mmq_auto_fixture_is_pinned_and_wired(self) -> None:
        manifest_path = LAGUNA_Q4_MMQ_AUTO_FIXTURE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "laguna-q4-mmq-auto-fixture/v1")
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
        capture = manifest["capture"]
        self.assertEqual(
            {key: capture[key] for key in (
                "flash_attention", "layer", "tokens", "token_index",
                "slot", "expert", "row", "probe_sha256",
            )},
            {
                "flash_attention": "AUTO",
                "layer": 1,
                "tokens": 22,
                "token_index": 0,
                "slot": 0,
                "expert": 246,
                "row": 0,
                "probe_sha256": (
                    "43cfc41d0d5930e060ae7cb3536b13caca11ce505718913834970aee8533a67b"
                ),
            },
        )
        self.assertEqual(
            capture["callbacks"],
            {
                "gate.f32": "ffn_moe_gate-1",
                "up.f32": "ffn_moe_up-1",
                "swiglu.f32": "ffn_moe_swiglu-1",
            },
        )
        self.assertEqual(
            capture["non_perturbation"],
            {
                "selected_sha256": (
                    "8b5f9861cc02f4578fc07714a990426c0449cdf5997e4d0588ede0c262228183"
                ),
                "weights_sha256": (
                    "ccff82b7b7f6f5550c010394ed859721c91e63a774e0ecd3b1dd9891693bd43c"
                ),
                "routed_output_sha256": (
                    "32bbcc5fa2c03c566f8425585ab8153823b9db355001129c2f9698bf0931a083"
                ),
                "layer_output_sha256": (
                    "ee837552616e9b6c535f03b9ac56b433af9fad274ab69f38462ad9075228cc2e"
                ),
                "detail_capture_matches_endpoint_only": True,
            },
        )
        self.assertEqual(
            manifest["cuda"],
            {
                "device": "NVIDIA GB10",
                "compute_capability": "12.1",
                "batch_mode": "MMQ",
                "data_layout": "MMA",
                "flags": [
                    "-std=c++17",
                    "-O3",
                    "-use_fast_math",
                    "--generate-code=arch=compute_121a,code=[sm_121a]",
                    "-extended-lambda",
                    "-compress-mode=size",
                ],
            },
        )
        self.assertEqual(
            manifest["weights"],
            {
                "type": "Q4_K",
                "input_elements": 3072,
                "blocks_per_row": 12,
                "block_bytes": 144,
                "row_bytes": 1728,
                "expert_bytes": 1769472,
                "gate": {
                    "tensor": "blk.1.ffn_gate_exps.weight",
                    "tensor_absolute_model_offset": 1349566304,
                    "row_absolute_model_offset": 1784856416,
                },
                "up": {
                    "tensor": "blk.1.ffn_up_exps.weight",
                    "tensor_absolute_model_offset": 1809051488,
                    "row_absolute_model_offset": 2244341600,
                },
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {"gate_exact": True, "up_exact": True, "swiglu_exact": True},
        )
        self.assertEqual(set(manifest["files"]), set(LAGUNA_Q4_MMQ_AUTO_FILES))
        self.assertEqual(
            {path.name for path in LAGUNA_Q4_MMQ_AUTO_FIXTURE.iterdir()},
            set(LAGUNA_Q4_MMQ_AUTO_FILES) | {"manifest.json"},
        )
        for name, (expected_size, expected_sha256) in (
            LAGUNA_Q4_MMQ_AUTO_FILES.items()
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    manifest["files"][name],
                    {"bytes": expected_size, "sha256": expected_sha256},
                )
                payload = (LAGUNA_Q4_MMQ_AUTO_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)

        for required in (
            '"q4-mmq-frozen"',
            "run_q4_mmq_frozen_case",
            "LAGUNA_Q4_MMQ_AUTO_FIXTURE_DIR",
            '"input-token-00.f32"',
            '"expert-246-row-000-gate.q4k"',
            '"expert-246-row-000-up.q4k"',
            '"gate.f32"',
            '"up.f32"',
            '"swiglu.f32"',
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)
        case_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_q4_mmq_frozen_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        self.assertIn(
            "ds4_gpu_test_glm_poolside_q4_mmq_gate_up_tensor(", case_body
        )
        self.assertIn("cudaDeviceSynchronize()", case_body)
        self.assertIn("ds4_gpu_tensor_read(", case_body)
        self.assertIn("actual_bits != expected_bits", case_body)
        self.assertRegex(case_body, r"input_elements\s*=\s*3072")

        hook_body = function_body(
            'extern "C" int ds4_gpu_test_glm_poolside_q4_mmq_gate_up_tensor('
        )
        self.assertIn("glm_poolside_q8_1_mmq_quantize_kernel<<<", hook_body)
        self.assertIn("glm_poolside_q4_mmq_gate_up_test_kernel<<<", hook_body)
        hook_kernel = function_body(
            "__global__ static void glm_poolside_q4_mmq_gate_up_test_kernel("
        )
        self.assertIn("dev_dot_q4_K_q8_1_mma_row(", hook_kernel)
        mma_body = function_body(
            "__device__ static float dev_dot_q4_K_q8_1_mma_row("
        )
        self.assertGreaterEqual(mma_body.count("__float2half_rn("), 2)
        self.assertIn("-dmin * (float)minimum", mma_body)
        self.assertRegex(mma_body, r"sum\s*\+=\s*scaled_d")
        self.assertRegex(mma_body, r"sum\s*\+=\s*scaled_min")
        gate_up_body = function_body(
            "__global__ static void glm_poolside_q4_gate_up_kernel("
        )
        self.assertIn("if (mmq)", gate_up_body)
        self.assertGreaterEqual(
            gate_up_body.count("dev_dot_q4_K_q8_1_mma_row("), 2
        )
        down_body = function_body(
            "__global__ static void glm_poolside_q4_down_kernel("
        )
        self.assertIn("if (mmq)", down_body)
        self.assertIn("dev_dot_q4_K_q8_1_mma_row(", down_body)

        kernel_main = source_function_body(
            LAGUNA_KERNEL_TEST, "int main(", "tests/test_cuda_laguna_kernels.c"
        )
        self.assertIn("run_q4_mmq_frozen_case()", kernel_main)

    def test_q4k_mmvq_microscope_is_generic_pinned_and_wired(self) -> None:
        manifest = json.loads(
            (Q4K_MMVQ_MICROSCOPE_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["schema"], "q4k-mmvq-microscope-fixture/v1"
        )
        self.assertEqual(
            manifest["poolside_commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(
            manifest["poolside_sources"],
            {
                "ggml/src/ggml-cuda/vecdotq.cuh": (
                    "69ee2488c89da7e6d31a91d2f33f3312d7c66dbcb60b4c42f154bdf775c4436a"
                ),
                "ggml/src/ggml-cuda/mmvq.cu": (
                    "7dd8df4666d524a749c0e1aafe8f6fb99a3ea565b4a2fc3ce957a406ab3d4776"
                ),
                "ggml/src/ggml-cuda/quantize.cu": (
                    "838cff1fd45ab483f3f86d24f23d997833b4af7a0b945a80f4da0e055def6565"
                ),
            },
        )
        self.assertEqual(
            manifest["shape"],
            {
                "input_elements": 3072,
                "q4_k_blocks": 12,
                "q4_k_block_bytes": 144,
                "row_bytes": 1728,
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {
                "value": -0.10056672245264053,
                "poolside_float32_bits": "0xbdcdf5ed",
                "ds4_serial_float32_bits": "0xbdcdf5ef",
                "quantized_operands_fp64": -0.10056671364048952,
            },
        )
        self.assertEqual(
            set(manifest["files"]), set(Q4K_MMVQ_MICROSCOPE_FILES)
        )
        self.assertEqual(
            {path.name for path in Q4K_MMVQ_MICROSCOPE_FIXTURE.iterdir()},
            set(Q4K_MMVQ_MICROSCOPE_FILES) | {"manifest.json"},
        )
        for name, (expected_size, expected_sha256) in (
            Q4K_MMVQ_MICROSCOPE_FILES.items()
        ):
            with self.subTest(name=name):
                payload = (Q4K_MMVQ_MICROSCOPE_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), expected_sha256
                )
                self.assertEqual(
                    manifest["files"][name],
                    {"bytes": expected_size, "sha256": expected_sha256},
                )

        self.assertNotIn("laguna", Q4K_MMVQ_MICROSCOPE_TEST.lower())
        for required in (
            "ds4_gpu_test_q4_k_mmvq_microscope_tensor(",
            "DS4_SERIAL_EXPECTED_BITS",
            '"input.q8_1"',
            "q4_k_q8_1_reference(",
        ):
            self.assertIn(required, Q4K_MMVQ_MICROSCOPE_TEST)
        self.assertIn(
            "tests/test_cuda_q4k_mmvq_microscope:", MAKEFILE
        )
        self.assertIn(
            "ds4_gpu_test_q4_k_mmvq_microscope_tensor(", GPU_HEADER
        )

        serial_kernel = function_body(
            "__global__ static void q4_k_mmvq_serial_microscope_kernel("
        )
        self.assertIn("dev_dot_q4_K_q8_1_mmvq_block(", serial_kernel)
        poolside_kernel = function_body(
            "__global__ static void q4_k_mmvq_poolside_microscope_kernel("
        )
        self.assertRegex(poolside_kernel, r"block\s*\+=\s*8u")
        self.assertIn("other_warps[3][32]", poolside_kernel)
        self.assertRegex(
            poolside_kernel,
            r"for \(uint32_t offset = 16u; offset > 0u; offset >>= 1u\)",
        )
        hook = function_body(
            'extern "C" int ds4_gpu_test_q4_k_mmvq_microscope_tensor('
        )
        self.assertIn("quantize_poolside_q8_1_f32_kernel<<<", hook)
        self.assertIn("q4_k_mmvq_serial_microscope_kernel<<<1, 1>>>", hook)
        self.assertIn(
            "q4_k_mmvq_poolside_microscope_kernel<<<1, dim3(32, 4, 1)>>>",
            hook,
        )

    def test_f32_mmvf_microscope_is_generic_pinned_and_wired(self) -> None:
        test_path = F32_MMVF_MICROSCOPE_TEST_PATH
        manifest_path = F32_MMVF_MICROSCOPE_FIXTURE / "manifest.json"
        weight_path = F32_MMVF_MICROSCOPE_FIXTURE / "weight-row.f32"
        self.assertTrue(test_path.is_file())
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(weight_path.is_file())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema"], "f32-mmvf-microscope-fixture/v1"
        )
        self.assertEqual(
            manifest["poolside_commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(
            manifest["poolside_sources"],
            {
                "ggml/src/ggml-cuda/mmvf.cu": (
                    "23b580ce14a45e71cc9be31047301d502"
                    "be74a832084c16662985f93f533ba1c"
                ),
                "ggml/src/ggml-cuda/common.cuh": (
                    "a977b50f7479df092bf3c441ba88e451"
                    "9803b5f0df5a640fd7fb0874e64b9b4c"
                ),
            },
        )
        self.assertEqual(
            manifest["shape"],
            {"input_elements": 3072, "row_bytes": 12288},
        )
        self.assertEqual(
            manifest["origin"],
            {
                "token": 513,
                "layer": 1,
                "stage": "router_logits",
                "row": 0,
                "expert": 0,
                "tensor": "blk.1.ffn_gate_inp.weight",
                "tensor_absolute_model_offset": 1802551136,
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {
                "poolside_float32_bits": "0xbea377ba",
                "ds4_serial_float32_bits": "0xbea377b8",
                "operands_fp64": -0.31927273880611085,
            },
        )

        weight = weight_path.read_bytes()
        self.assertEqual(len(weight), 12288)
        self.assertEqual(
            hashlib.sha256(weight).hexdigest(),
            "4b2e76f429c40ab67023a7500cd2eb25"
            "e0fd820de9d550025642b172452b16b1",
        )
        input_path = Q4K_MMVQ_MICROSCOPE_FIXTURE / "input.f32"
        self.assertEqual(
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "eabe89d1d9a4bdc660e5759c2a20d347"
            "d4dedaec1e617a44f3244cfe7985ef0e",
        )

        source = test_path.read_text(encoding="utf-8")
        self.assertNotIn("laguna", source.lower())
        for required in (
            "ds4_gpu_test_f32_mmvf_microscope_tensor(",
            "DS4_SERIAL_EXPECTED_BITS",
            "POOLSIDE_EXPECTED_BITS",
            '"weight-row.f32"',
            '"../q4k-mmvq-microscope-auto/input.f32"',
        ):
            self.assertIn(required, source)
        self.assertIn(
            "tests/test_cuda_f32_mmvf_microscope:", MAKEFILE
        )
        self.assertIn(
            "ds4_gpu_test_f32_mmvf_microscope_tensor(", GPU_HEADER
        )

        poolside_kernel = function_body(
            "__global__ static void f32_mmvf_poolside_microscope_kernel("
        )
        self.assertIn("const float2 *weight2", poolside_kernel)
        self.assertIn("const float2 *activation2", poolside_kernel)
        self.assertRegex(poolside_kernel, r"col2\s*\+=\s*256u")
        self.assertGreaterEqual(poolside_kernel.count("sum +="), 3)
        self.assertRegex(
            poolside_kernel,
            r"for \(uint32_t offset = 16u; offset > 0u; offset >>= 1u\)",
        )
        hook = function_body(
            'extern "C" int ds4_gpu_test_f32_mmvf_microscope_tensor('
        )
        self.assertIn("matmul_f32_kernel<<<1, 256>>>", hook)
        self.assertIn(
            "f32_mmvf_poolside_microscope_kernel<<<1, 256, ", hook
        )

    def test_laguna_decode_poolside_mmvq_is_opt_in_and_narrow(self) -> None:
        selector = function_body(
            "static bool cuda_poolside_mmvq_requested(void)"
        )
        self.assertIn("return g_cuda_poolside_mmvq != 0;", selector)
        self.assertNotIn("getenv", selector)
        refresh = function_body(
            "static void cuda_decode_dispatch_env_refresh(void)"
        )
        self.assertIn('getenv("DS4_MM_VQ_REDUCTION")', refresh)
        self.assertIn('strcmp(reduction, "poolside") == 0', refresh)
        init = function_body('extern "C" int ds4_gpu_init_multi(')
        self.assertIn("cuda_decode_dispatch_env_refresh();", init)

        fragment_signature = (
            "dev_dot_q4_K_q8_1_poolside_mmvq_fragment("
        )
        fragment_offset = CUDA_SOURCE.index(fragment_signature)
        preceding_guard = CUDA_SOURCE.rfind(
            "#ifdef DS4_TEST_HOOKS", 0, fragment_offset
        )
        self.assertGreaterEqual(preceding_guard, 0)
        self.assertLess(
            CUDA_SOURCE.find("#endif", preceding_guard), fragment_offset,
            "the production Poolside reduction fragment cannot be test-only",
        )

        gate_up = function_body(
            "__global__ static void "
            "glm_poolside_q4_gate_up_poolside_mmvq_kernel("
        )
        self.assertRegex(
            CUDA_SOURCE,
            r"__launch_bounds__\(128, 1\)\s*__global__ static void\s+"
            r"glm_poolside_q4_gate_up_poolside_mmvq_kernel\(",
        )
        self.assertIn("const uint32_t tid = warp * 32u + lane;", gate_up)
        self.assertGreaterEqual(gate_up.count(fragment_signature), 2)
        self.assertIn("gate_other_warps[3][32]", gate_up)
        self.assertIn("up_other_warps[3][32]", gate_up)
        self.assertRegex(gate_up, r"block\s*\+=\s*8u")
        self.assertRegex(
            gate_up,
            r"for \(uint32_t offset = 16u; offset > 0u; offset >>= 1u\)",
        )

        down = function_body(
            "__global__ static void "
            "glm_poolside_q4_down_poolside_mmvq_kernel("
        )
        self.assertIn("other_warps[3][4][32]", down)
        self.assertIn("for (uint32_t slot = 0; slot < n_expert; slot++)", down)
        self.assertIn(fragment_signature, down)
        self.assertRegex(down, r"block\s*\+=\s*8u")
        self.assertRegex(
            down,
            r"for \(uint32_t offset = 16u; offset > 0u; offset >>= 1u\)",
        )

        launch = function_body("static int glm_poolside_routed_moe_q4_launch(")
        self.assertIn(
            "const bool poolside_mmvq = !mmq && n_tokens == 1u &&\n"
            "        n_total_expert == 256u && n_expert == 10u &&\n"
            "        expert_in_dim == 3072u && expert_mid_dim == 1024u &&\n"
            "        out_dim == 3072u && cuda_poolside_mmvq_requested();",
            launch,
        )
        self.assertIn(
            "glm_poolside_q4_gate_up_poolside_mmvq_kernel<<<", launch
        )
        self.assertIn(
            "glm_poolside_q4_down_poolside_mmvq_kernel<<<", launch
        )
        self.assertIn("glm_poolside_q4_gate_up_kernel<<<", launch)
        self.assertIn("glm_poolside_q4_down_kernel<<<", launch)
        self.assertGreaterEqual(launch.count("dim3(32, 4, 1)"), 2)

    def test_laguna_c7_oracle_producer_is_self_contained(self) -> None:
        self.assertEqual(
            {path.name for path in LAGUNA_C7_ORACLE_PRODUCER.iterdir()},
            LAGUNA_C7_ORACLE_PRODUCER_FILES,
        )
        manifests = [
            json.loads(
                (fixture / "manifest.json").read_text(encoding="utf-8")
            )
            for fixture in (
                LAGUNA_Q4_L2_AUTO_FIXTURE,
                LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE,
            )
        ]
        self.assertEqual(manifests[0]["producer"], manifests[1]["producer"])
        producer = manifests[0]["producer"]
        self.assertEqual(
            set(producer),
            {"probe", "poolside_patch", "capture_script", "tokens", "verifier"},
        )
        expected_paths = {
            "probe": "probe_poolside_laguna_moe.cpp",
            "poolside_patch": "poolside-l2-callback.patch",
            "capture_script": "capture_poolside_laguna_moe.sh",
            "tokens": "short.tokens.i32",
            "verifier": "verify_poolside_laguna_moe.py",
        }
        for key, name in expected_paths.items():
            with self.subTest(producer=key):
                entry = producer[key]
                self.assertEqual(
                    entry["path"], f"tests/oracle-producers/laguna-c7/{name}"
                )
                producer_path = Path(entry["path"])
                self.assertFalse(producer_path.is_absolute())
                self.assertNotIn("..", producer_path.parts)
                payload = (LAGUNA_C7_ORACLE_PRODUCER / name).read_bytes()
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(
                    entry["sha256"], hashlib.sha256(payload).hexdigest()
                )
        self.assertEqual(
            producer["poolside_patch"]["sha256"],
            "36b84feca9f828f4bae0553291fa3a559bfb13ca4f8f1cfca3bd80266315f2c6",
        )
        self.assertEqual(
            producer["tokens"]["sha256"],
            "900e6342be5dc175a4fc13fafbf0fb380eaac5824079cf4a2db69b4d0a777fa8",
        )
        token_bytes = (LAGUNA_C7_ORACLE_PRODUCER / "short.tokens.i32").read_bytes()
        self.assertEqual(
            struct.unpack("<22i", token_bytes),
            (
                2, 97, 1437, 99, 53225, 3203, 330, 10068, 3612, 31063, 81,
                365, 1161, 15631, 83, 268, 532, 1437, 99, 268, 23, 19,
            ),
        )
        self.assertEqual(producer["tokens"]["format"], "little-endian-int32")
        self.assertEqual(producer["tokens"]["count"], 22)
        self.assertEqual(
            producer["tokens"]["ids"], list(struct.unpack("<22i", token_bytes))
        )
        self.assertEqual(
            producer["tokens"]["prompt_sha256"],
            "e3dc84f54d1afb86e11f782bb9d19715340855df5ef602cde682768008aef74d",
        )
        self.assertEqual(
            producer["tokens"]["tokenizer_runtime_commit"],
            "15c9b92502fed6bc26842e98d11a6347caadb08e",
        )
        self.assertEqual(manifests[0]["execution"], manifests[1]["execution"])
        self.assertEqual(
            manifests[0]["execution"],
            {
                "toolchain": LAGUNA_C7_TOOLCHAIN,
                "generated_recipes": LAGUNA_C7_GENERATED_RECIPES,
                "device": {
                    "name": "NVIDIA GB10",
                    "compute_capability": "12.1",
                    "multiprocessors": 48,
                },
                "poolside_build": {
                    "cmake_build_type": "Release",
                    "ggml_cuda": True,
                    "ggml_cuda_fa_all_quants": False,
                    "ggml_cuda_force_cublas": False,
                    "cuda_flags": [
                        "-std=c++17",
                        "--generate-code=arch=compute_121a,code=[sm_121a]",
                        "-Xcompiler=-fPIC",
                        "-use_fast_math",
                        "-extended-lambda",
                        "-compress-mode=size",
                        "-Xcompiler",
                        (
                            "-Wmissing-declarations -Wmissing-noreturn -Wall "
                            "-Wextra -Wpedantic -Wcast-qual "
                            "-Wno-unused-function -Wno-array-bounds "
                            "-Wextra-semi -Wno-pedantic"
                        ),
                    ],
                    "cxx_flags": ["-include", "cmath", "-O3", "-DNDEBUG"],
                },
                "probe_build": {
                    "compiler": "/usr/bin/c++",
                    "flags": ["-std=c++17", "-O2"],
                    "libraries": ["llama", "ggml", "ggml-base"],
                    "runtime_library_path": "poolside_build/bin",
                    "elf_runpath": "none",
                    "binary_bytes": 80456,
                    "binary_sha256": (
                        "659be562dcdc4b3563d36a559ba121675db33a1c0aeb5338f5f20916d97b7f56"
                    ),
                },
            },
        )
        probe = (LAGUNA_C7_ORACLE_PRODUCER / expected_paths["probe"]).read_text(
            encoding="utf-8"
        )
        for required in (
            "--token-count",
            '"ffn_inp"',
            '"ffn_moe_col_l2"',
            '"ffn_moe_down_input"',
        ):
            self.assertIn(required, probe)
        capture_script = (
            LAGUNA_C7_ORACLE_PRODUCER / expected_paths["capture_script"]
        ).read_text(encoding="utf-8")
        self.assertTrue(capture_script.startswith("#!/bin/bash -p\n"))
        for required in (
            "--token-count 22",
            "--token-count 1",
            "apply --check -R",
            'exec 9<"$model"',
            "verify_poolside_laguna_moe.py",
            "preflight",
            "--model-fd 9",
            'model_fd_path="/proc/self/fd/9"',
            "captured",
            "--capture-root",
            "--probe-bin",
            "--target clean",
            "assert_build_outputs_absent",
            "rebuild_llama",
            "/usr/bin/cmake --build",
            "capture_environment",
            "assert_capture_environment",
            "noncanonical capture environment",
            "/usr/bin/env -i",
            "/bin/bash -p",
            "/usr/bin/nvidia-smi",
            "PATH=/usr/bin:/bin:/usr/local/cuda/bin",
            'exec 8<>"$continuity_file"',
            "--continuity-fd 8",
            "assert_gpu_processes",
            "run_probe_exclusive",
            "--query-compute-apps=pid",
            "LD_LIBRARY_PATH=$poolside_build/bin",
            "-Wl,-rpath-link,",
            "/usr/bin/c++",
            'git -C "$script_dir" rev-parse --show-toplevel',
            '$repo_root/tests/oracle-producers/laguna-c7',
        ):
            self.assertIn(required, capture_script)
        self.assertEqual(
            capture_script.count("run_probe_exclusive capture_environment"), 2
        )
        self.assertEqual(capture_script.count("8>&-"), 2)
        self.assertEqual(capture_script.count('"$verifier" preflight'), 2)
        self.assertNotIn("--clean-first", capture_script)
        self.assertNotIn("-Wl,-rpath,", capture_script)
        self.assertNotIn("status --porcelain", capture_script)
        self.assertLess(
            capture_script.index("preflight"),
            capture_script.index('git -C "$poolside_src" apply "$patch"'),
        )
        self.assertLess(
            capture_script.index("--token-count 1"),
            capture_script.index("captured"),
        )
        verifier = (
            LAGUNA_C7_ORACLE_PRODUCER / expected_paths["verifier"]
        ).read_text(encoding="utf-8")
        self.assertTrue(verifier.startswith("#!/usr/bin/python3\n"))
        for required in (
            '"/usr/bin/git"',
            '"/usr/bin/nvidia-smi"',
            '"GIT_NO_REPLACE_OBJECTS": "1"',
            '"GIT_CONFIG_NOSYSTEM": "1"',
            '"ls-files",\n                "--others"',
            '"replace", "-l"',
            '"rev-parse", "--show-toplevel"',
            '"ls-tree",',
            '"--stage", "-z"',
            "Poolside raw tracked tree",
        ):
            self.assertIn(required, verifier)
        self.assertNotIn("dso_sha256", json.dumps(manifests[0]).lower())

    def test_laguna_c7_fail_closed_provenance_suite(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tests/test_laguna_c7_provenance.py"),
                "-v",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_laguna_q4_l2_auto_fixture_is_pinned_and_wired(self) -> None:
        manifest_path = LAGUNA_Q4_L2_AUTO_FIXTURE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "laguna-q4-l2-auto-fixture/v2")
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
        capture = manifest["capture"]
        self.assertEqual(
            {key: capture[key] for key in (
                "flash_attention", "layer", "expert_mid_dim"
            )},
            {
                "flash_attention": "AUTO",
                "layer": 1,
                "expert_mid_dim": 1024,
            },
        )
        self.assertEqual(
            capture["prefill_128"],
            {
                "capture_directory": "poolside-c7-moe-auto-22",
                "tokens": 22,
                "pair_count": 220,
                "fixture_pair_count": 1,
                "production_l2_threads": 128,
                "callbacks": {
                    "mid-pair0.f32": "ffn_moe_swiglu-1",
                    "expected-col-l2-pair0.f32": "ffn_moe_col_l2-1",
                    "expected-down-input-pair0.f32": "ffn_moe_down_input-1",
                },
            },
        )
        self.assertEqual(
            capture["decode_512"],
            {
                "capture_directory": "poolside-c7-moe-auto-1",
                "tokens": 1,
                "pair_count": 10,
                "fixture_pair_count": 10,
                "production_l2_threads": 512,
                "callbacks": {
                    "mid-decode.f32": "ffn_moe_swiglu-1",
                    "expected-col-l2-decode.f32": "ffn_moe_col_l2-1",
                    "expected-down-input-decode.f32": "ffn_moe_down_input-1",
                },
            },
        )
        self.assertEqual(
            capture["non_perturbation"],
            {
                "prefill_128": {
                    "selected_sha256": (
                        "8b5f9861cc02f4578fc07714a990426c0449cdf5997e4d0588ede0c262228183"
                    ),
                    "weights_sha256": (
                        "ccff82b7b7f6f5550c010394ed859721c91e63a774e0ecd3b1dd9891693bd43c"
                    ),
                    "routed_output_sha256": (
                        "32bbcc5fa2c03c566f8425585ab8153823b9db355001129c2f9698bf0931a083"
                    ),
                    "shared_output_sha256": (
                        "14a04388d619381400e5551794423244cc6d735d68df5dab7693d8604df32370"
                    ),
                    "ffn_output_sha256": (
                        "9d4abbb288dc57ecfdb4c30c69f4695f00eefe3a51ccb41e5db004f35013285e"
                    ),
                    "layer_output_sha256": (
                        "ee837552616e9b6c535f03b9ac56b433af9fad274ab69f38462ad9075228cc2e"
                    ),
                },
                "decode_512": {
                    "selected_sha256": (
                        "6e9904384286571bb5e428a0cc77308799c142eda49b446253a2536f5eab43cf"
                    ),
                    "weights_sha256": (
                        "5122b2548c3d60ab2b61cbd5195dd3ca0ac91a7c5da1ed9001eebcd88f2a33b2"
                    ),
                    "routed_output_sha256": (
                        "4b716c227cddbb23d1594e3e628fe2513f4a17acbfb6d953d4a9e5841b1091fb"
                    ),
                    "shared_output_sha256": (
                        "b6131ad8bd87e87c3ee0610f7a76a2e43eb3683298327bee9bbe6c333de77a31"
                    ),
                    "ffn_output_sha256": (
                        "f8a2e0ec534227afa478fbbe48630166b65e35ba71ae0ade2103a228a2fadef8"
                    ),
                    "layer_output_sha256": (
                        "bf630915ef3084ada6e9ae38dcea28c08021251551708cb6240fe22dd47e4212"
                    ),
                },
                "consolidated_capture_matches_prior_captures": True,
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {
                "column_l2_exact": True,
                "down_input_exact": True,
                "reduction_topologies": {
                    "prefill_128": "128-thread production SUM_ROWS tree",
                    "decode_512": "512-thread production SUM_ROWS tree",
                },
            },
        )
        expected_extractions = {
            "mid-pair0.f32": (
                "poolside-c7-moe-auto-22", "layer-01-ffn-moe-swiglu.f32",
                901120, "35883cb5da611931117b5d14c5a88f9dab49035d8656bbddb43cba1b914f8015",
            ),
            "expected-col-l2-pair0.f32": (
                "poolside-c7-moe-auto-22", "layer-01-ffn-moe-col-l2.f32",
                880, "98274d2ad38f471b41660c11d48a21ef0704d021886d225567bc2b9e93549848",
            ),
            "expected-down-input-pair0.f32": (
                "poolside-c7-moe-auto-22", "layer-01-ffn-moe-down-input.f32",
                901120, "ff55f3b3e49034a667b30f105ea9e16306c659a7fd2115448c183c72930ffc84",
            ),
            "mid-decode.f32": (
                "poolside-c7-moe-auto-1", "layer-01-ffn-moe-swiglu.f32",
                40960, "c43a2d48c6931726245e9a91f31cc669dd2786a3a2970626e904f48b5bc5567e",
            ),
            "expected-col-l2-decode.f32": (
                "poolside-c7-moe-auto-1", "layer-01-ffn-moe-col-l2.f32",
                40, "c84da10d5bb0cf015e07a441824fb1223261ee44320ef2be3d5936e1f5bc8086",
            ),
            "expected-down-input-decode.f32": (
                "poolside-c7-moe-auto-1", "layer-01-ffn-moe-down-input.f32",
                40960, "22a986deadc4750a1760fbc72f58f233ceab87816697c97289f80688ea24e69a",
            ),
        }
        self.assertEqual(set(manifest["extractions"]), set(expected_extractions))
        for name, (directory, source, source_bytes, source_sha256) in (
            expected_extractions.items()
        ):
            with self.subTest(extraction=name):
                extraction = manifest["extractions"][name]
                self.assertEqual(
                    {key: extraction[key] for key in (
                        "capture_directory", "source", "source_bytes",
                        "source_sha256", "offset", "bytes", "producer",
                    )},
                    {
                        "capture_directory": directory,
                        "source": source,
                        "source_bytes": source_bytes,
                        "source_sha256": source_sha256,
                        "offset": 0,
                        "bytes": LAGUNA_Q4_L2_AUTO_FILES[name][0],
                        "producer": "probe",
                    },
                )
        self.assertEqual(set(manifest["files"]), set(LAGUNA_Q4_L2_AUTO_FILES))
        self.assertEqual(
            {path.name for path in LAGUNA_Q4_L2_AUTO_FIXTURE.iterdir()},
            set(LAGUNA_Q4_L2_AUTO_FILES) | {"manifest.json"},
        )
        for name, (expected_size, expected_sha256) in (
            LAGUNA_Q4_L2_AUTO_FILES.items()
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    manifest["files"][name],
                    {"bytes": expected_size, "sha256": expected_sha256},
                )
                payload = (LAGUNA_Q4_L2_AUTO_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)

        for required in (
            '"q4-l2-frozen"',
            "run_q4_l2_frozen_case",
            "run_q4_l2_frozen_topology_case",
            '"prefill-128"',
            '"decode-512"',
            "LAGUNA_Q4_L2_AUTO_FIXTURE_DIR",
            '"mid-pair0.f32"',
            '"expected-col-l2-pair0.f32"',
            '"expected-down-input-pair0.f32"',
            '"mid-decode.f32"',
            '"expected-col-l2-decode.f32"',
            '"expected-down-input-decode.f32"',
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)
        case_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_q4_l2_frozen_topology_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        self.assertIn("ds4_gpu_test_glm_poolside_q4_l2_tensor(", case_body)
        self.assertIn("cudaDeviceSynchronize()", case_body)
        self.assertGreaterEqual(case_body.count("ds4_gpu_tensor_read("), 2)
        self.assertIn("actual_bits != expected_bits", case_body)
        self.assertRegex(case_body, r"expert_mid_dim\s*=\s*1024")
        wrapper_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_q4_l2_frozen_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        self.assertEqual(
            wrapper_body.count("run_q4_l2_frozen_topology_case("), 2
        )
        self.assertRegex(wrapper_body, r'"prefill-128"[\s\S]*?1u,\s*220u')
        self.assertRegex(wrapper_body, r'"decode-512"[\s\S]*?10u,\s*10u')

        selector_body = function_body(
            "static uint32_t glm_poolside_q4_l2_thread_count("
        )
        self.assertRegex(selector_body, r"uint32_t\s+threads\s*=\s*512u")
        self.assertRegex(
            selector_body,
            r"pair_count\s*/\s*multiprocessors\s*>=\s*2u",
        )
        self.assertIn("expert_mid_dim < 1024u ? 32u : 128u", selector_body)

        hook_body = function_body(
            'extern "C" int ds4_gpu_test_glm_poolside_q4_l2_tensor('
        )
        self.assertRegex(
            hook_body,
            r"glm_poolside_q4_l2_thread_count\(\s*expert_mid_dim,\s*"
            r"topology_pair_count,\s*\(uint32_t\)multiprocessors\s*\)",
        )
        self.assertRegex(
            hook_body,
            r"glm_poolside_q4_l2_rescale_kernel<<<\s*"
            r"\(uint32_t\)pair_count,\s*l2_threads\s*>>>",
        )
        production_body = function_body(
            "static int glm_poolside_routed_moe_q4_launch("
        )
        self.assertRegex(
            production_body,
            r"l2_threads\s*=\s*glm_poolside_q4_l2_thread_count\(\s*"
            r"expert_mid_dim,\s*pair_count,\s*"
            r"\(uint32_t\)multiprocessors\s*\)",
        )
        self.assertRegex(
            production_body,
            r"glm_poolside_q4_l2_rescale_kernel<<<\s*"
            r"\(uint32_t\)pair_count,\s*l2_threads\s*>>>",
        )
        l2_body = function_body(
            "__global__ static void glm_poolside_q4_l2_rescale_kernel("
        )
        self.assertIn("const float scaled = __fmul_rn(", l2_body)
        self.assertRegex(l2_body, r"mid\[offset\]\s*=\s*scaled\s*/\s*l2;")
        self.assertNotIn("__fdiv_rn(", l2_body)
        kernel_main = source_function_body(
            LAGUNA_KERNEL_TEST, "int main(", "tests/test_cuda_laguna_kernels.c"
        )
        self.assertIn("run_q4_l2_frozen_case()", kernel_main)

    def test_laguna_moe_residual_auto_fixture_is_pinned_and_wired(self) -> None:
        manifest_path = LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["schema"], "laguna-moe-residual-auto-fixture/v2"
        )
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
        capture = manifest["capture"]
        self.assertEqual(
            {key: capture[key] for key in (
                "flash_attention", "layer", "tokens", "token_index", "width",
                "capture_directory",
            )},
            {
                "flash_attention": "AUTO",
                "layer": 1,
                "tokens": 22,
                "token_index": 0,
                "width": 3072,
                "capture_directory": "poolside-c7-moe-auto-22",
            },
        )
        self.assertEqual(
            capture["callbacks"],
            {
                "residual-token0.f32": "ffn_inp-1",
                "moe-token0.f32": "ffn_moe_out-1",
                "shared-token0.f32": "ffn_shexp-1",
                "expected-token0.f32": "l_out-1",
            },
        )
        self.assertEqual(
            capture["non_perturbation"],
            {
                "routed_output_sha256": (
                    "32bbcc5fa2c03c566f8425585ab8153823b9db355001129c2f9698bf0931a083"
                ),
                "shared_output_sha256": (
                    "14a04388d619381400e5551794423244cc6d735d68df5dab7693d8604df32370"
                ),
                "ffn_output_sha256": (
                    "9d4abbb288dc57ecfdb4c30c69f4695f00eefe3a51ccb41e5db004f35013285e"
                ),
                "layer_output_sha256": (
                    "ee837552616e9b6c535f03b9ac56b433af9fad274ab69f38462ad9075228cc2e"
                ),
                "consolidated_capture_matches_prior_captures": True,
            },
        )
        self.assertEqual(
            manifest["oracle"],
            {
                "expression": "(moe + shared) + residual",
                "expected_exact": True,
                "intended_order_mismatches": 0,
                "legacy_order": "(residual + moe) + shared",
                "legacy_order_mismatches": 753,
                "legacy_first_mismatch": 0,
            },
        )
        expected_extractions = {
            "residual-token0.f32": (
                "layer-01-ffn-inp.f32",
                "2c6d314aeea3587bd0e5eed8b2056830bbf976a3a4afe59dc1ff46a2434e4abc",
            ),
            "moe-token0.f32": (
                "layer-01-ffn-moe-out.f32",
                "32bbcc5fa2c03c566f8425585ab8153823b9db355001129c2f9698bf0931a083",
            ),
            "shared-token0.f32": (
                "layer-01-ffn-shared-out.f32",
                "14a04388d619381400e5551794423244cc6d735d68df5dab7693d8604df32370",
            ),
            "expected-token0.f32": (
                "layer-01.f32",
                "ee837552616e9b6c535f03b9ac56b433af9fad274ab69f38462ad9075228cc2e",
            ),
        }
        self.assertEqual(set(manifest["extractions"]), set(expected_extractions))
        for name, (source, source_sha256) in expected_extractions.items():
            with self.subTest(extraction=name):
                extraction = manifest["extractions"][name]
                self.assertEqual(
                    {key: extraction[key] for key in (
                        "capture_directory", "source", "source_bytes",
                        "source_sha256", "offset", "bytes", "producer",
                    )},
                    {
                        "capture_directory": "poolside-c7-moe-auto-22",
                        "source": source,
                        "source_bytes": 270336,
                        "source_sha256": source_sha256,
                        "offset": 0,
                        "bytes": LAGUNA_MOE_RESIDUAL_AUTO_FILES[name][0],
                        "producer": "probe",
                    },
                )
        self.assertEqual(
            set(manifest["files"]), set(LAGUNA_MOE_RESIDUAL_AUTO_FILES)
        )
        self.assertEqual(
            {path.name for path in LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE.iterdir()},
            set(LAGUNA_MOE_RESIDUAL_AUTO_FILES) | {"manifest.json"},
        )
        for name, (expected_size, expected_sha256) in (
            LAGUNA_MOE_RESIDUAL_AUTO_FILES.items()
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    manifest["files"][name],
                    {"bytes": expected_size, "sha256": expected_sha256},
                )
                payload = (LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE / name).read_bytes()
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)

        for required in (
            '"moe-residual-frozen"',
            "run_moe_residual_frozen_case",
            "LAGUNA_MOE_RESIDUAL_AUTO_FIXTURE_DIR",
            '"residual-token0.f32"',
            '"moe-token0.f32"',
            '"shared-token0.f32"',
            '"expected-token0.f32"',
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)
        case_body = source_function_body(
            LAGUNA_KERNEL_TEST,
            "static int run_moe_residual_frozen_case(",
            "tests/test_cuda_laguna_kernels.c",
        )
        self.assertIn("ds4_gpu_laguna_moe_residual_tensor(", case_body)
        self.assertIn("cudaDeviceSynchronize()", case_body)
        self.assertIn("ds4_gpu_tensor_read(", case_body)
        self.assertIn("actual_bits != expected_bits", case_body)
        self.assertRegex(case_body, r"width\s*=\s*3072")

        self.assertRegex(
            GPU_HEADER,
            r"static inline int ds4_gpu_laguna_moe_residual_tensor\([\s\S]*?"
            r"return ds4_gpu_add3_tensor\(out, moe, shared, residual, n\);",
        )
        self.assertIn("ds4_gpu_laguna_moe_residual_tensor(", GPU_HEADER)
        graph_calls = {
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_token(": (
                r"ds4_gpu_laguna_moe_residual_tensor\(\s*"
                r"g->next,\s*g->after_attn,\s*g->ffn_out,\s*g->shared_out,\s*"
                r"DS4_N_EMBD\s*\)"
            ),
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_batch(": (
                r"ds4_gpu_laguna_moe_residual_tensor\(\s*"
                r"g->next,\s*g->after_attn,\s*g->ffn_out,\s*g->shared_out,\s*"
                r"\(uint64_t\)n_tokens\s*\*\s*DS4_N_EMBD\s*\)"
            ),
        }
        for function_name, expected_call in graph_calls.items():
            graph_body = source_function_body(DS4_SOURCE, function_name, "ds4.c")
            self.assertEqual(
                graph_body.count("ds4_gpu_laguna_moe_residual_tensor("), 1
            )
            self.assertRegex(graph_body, expected_call)
            self.assertNotIn("ds4_gpu_add3_tensor(", graph_body)
        kernel_main = source_function_body(
            LAGUNA_KERNEL_TEST, "int main(", "tests/test_cuda_laguna_kernels.c"
        )
        self.assertIn("run_moe_residual_frozen_case()", kernel_main)

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

    def test_laguna_kernel_links_q4_mmq_test_hooks(self) -> None:
        hook_object = "tests/ds4_cuda_laguna_kernels_test_hooks.o"
        self.assertIn("ds4_cuda.cu", rule_prerequisites(hook_object))
        self.assertIn(
            "-DDS4_TEST_HOOKS",
            "\n".join(rule_recipe_lines(hook_object)),
        )
        self.assertIn(
            "-DDS4_TEST_HOOKS",
            "\n".join(rule_recipe_lines("tests/test_cuda_laguna_kernels.o")),
        )
        kernel_objects = rule_prerequisites(
            "tests/test_cuda_laguna_kernels"
        ).split()
        self.assertIn(hook_object, kernel_objects)
        self.assertIn("ds4_runtime.o", kernel_objects)
        self.assertNotIn("ds4_cuda.o", kernel_objects)

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
        n1_prefix, multi_token_marker, _ = body.partition("n_tok > 1u")
        self.assertTrue(
            multi_token_marker, "missing Poolside multi-token Stream-K branch"
        )
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
            DS4_SOURCE,
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_token(",
            "ds4.c",
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
            DS4_SOURCE,
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_batch(",
            "ds4.c",
        )
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint\(\s*g->cur,\s*n_tokens,"
            r"\s*DS4_N_EMBD,\s*-1,\s*\"embd\"\s*\)",
        )
        layer_dump_match = re.search(
            r"laguna_graph_diag_checkpoint\(\s*g->next,\s*n_tokens,"
            r"\s*DS4_N_EMBD,\s*\(int\)il,\s*\"l_out\"\s*\)",
            body,
        )
        layer_dump = layer_dump_match.start() if layer_dump_match else -1
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

    def test_laguna_decode_layer_probe_covers_the_same_boundaries(self) -> None:
        body = source_function_body(
            DS4_SOURCE,
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_token(",
            "ds4.c",
        )
        self.assertIn("laguna_graph_diag_detail_layer()", body)
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint\(\s*g->cur,\s*1,"
            r"\s*DS4_N_EMBD,\s*-1,\s*\"embd\"\s*\)",
        )
        for tensor, stage, width in (
            ("attn_norm", "attn-norm", r"DS4_N_EMBD"),
            ("q", "q-proj", r"q_dim"),
            ("k", "k-proj", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("v", "v-proj", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("gate", "gate-proj", r"n_head"),
            ("q", "q-rope", r"q_dim"),
            ("k", "k-rope", r"DS4_N_HEAD_KV\s*\*\s*DS4_N_HEAD_DIM"),
            ("heads", "attn-gated", r"q_dim"),
            ("attn_out", "attn-o-proj", r"DS4_N_EMBD"),
            ("after_attn", "ffn-inp", r"DS4_N_EMBD"),
            ("ffn_norm", "ffn-norm", r"DS4_N_EMBD"),
            ("router_logits", "router-logits", r"DS4_N_EXPERT"),
            ("router_weights", "router-weights", r"DS4_N_EXPERT_USED"),
            ("ffn_out", "ffn-moe-out", r"DS4_N_EMBD"),
            ("shared_out", "ffn-shared-out", r"DS4_N_EMBD"),
            ("ffn_out", "ffn-out", r"DS4_N_EMBD"),
        ):
            with self.subTest(stage=stage):
                self.assertRegex(
                    body,
                    rf"laguna_graph_diag_checkpoint\(\s*g->{tensor},\s*1,"
                    rf"\s*{width},\s*\(int\)il,\s*\"{stage}\"\s*\)",
                )
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint_i32\(\s*g->router_selected,\s*1,"
            r"\s*DS4_N_EXPERT_USED,\s*\(int\)il,"
            r"\s*\"router-selected\"\s*\)",
        )
        layer_dump_match = re.search(
            r"laguna_graph_diag_checkpoint\(\s*g->next,\s*1,"
            r"\s*DS4_N_EMBD,\s*\(int\)il,\s*\"l_out\"\s*\)",
            body,
        )
        layer_dump = layer_dump_match.start() if layer_dump_match else -1
        layer_swap = body.find("ds4_gpu_tensor *tmp = g->cur;")
        self.assertGreaterEqual(layer_dump, 0, "missing decode layer-output probe")
        self.assertGreater(layer_swap, layer_dump, "probe must precede cur/next swap")
        self.assertRegex(
            body,
            r"laguna_graph_diag_checkpoint\(\s*g->logits,\s*1,"
            r"\s*DS4_N_VOCAB,\s*-1,\s*\"logits\"\s*\)",
        )

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
            LAGUNA_MODEL_TEST,
            "int main(int argc, char **argv)",
            "tests/test_cuda_laguna_model.c",
        )
        self.assertIn("const bool diagnostic_mode", main)
        diagnostic_guard = main.find(
            "if (ok && selected == MODEL_CASE_ALL && !diagnostic_mode) {"
        )
        swa_case = main.find('engine, "swa-513"')
        self.assertGreaterEqual(diagnostic_guard, 0)
        self.assertGreater(swa_case, diagnostic_guard)
        suite_end = main.find(
            "\n    }\n    if (mode == MODEL_MODE_STREAMED)",
            diagnostic_guard,
        )
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

    def test_laguna_decode_direct_moe_capture_observes_production_stages(self) -> None:
        producer = (
            ROOT
            / "tests/oracle-producers/laguna-c7/probe_ds4_laguna_moe.c"
        )
        self.assertTrue(producer.is_file(), "missing tracked DS4 512+1 producer")
        producer_source = producer.read_text(encoding="utf-8")
        self.assertIn("ds4_session_sync(", producer_source)
        self.assertIn("ds4_session_eval(", producer_source)
        self.assertIn("DS4_LAGUNA_DIAG_DIR", producer_source)
        self.assertIn("DS4_LAGUNA_DIAG_LAYER", producer_source)
        self.assertIn("tests/probe_ds4_laguna_moe:", MAKEFILE)
        probe_link = re.search(
            r"^tests/probe_ds4_laguna_moe:\s*(?P<objects>.*)$",
            MAKEFILE,
            re.MULTILINE,
        )
        self.assertIsNotNone(probe_link)
        probe_objects = probe_link.group("objects")
        self.assertIn("tests/ds4_cuda_laguna_kernels_test_hooks.o", probe_objects)
        self.assertNotRegex(probe_objects, r"(?:^|\s)ds4_cuda\.o(?:\s|$)")

        capture_hook = "ds4_gpu_test_glm_routed_moe_one_capture_tensor("
        self.assertIn(capture_hook, GPU_HEADER)
        self.assertIn(capture_hook, CUDA_SOURCE)
        self.assertIn(capture_hook, DS4_SOURCE)
        for layout_name in (
            "DS4_GPU_GLM_MOE_CAPTURE_GATE_F32",
            "DS4_GPU_GLM_MOE_CAPTURE_DOWN_Q8_OFFSET",
            "DS4_GPU_GLM_MOE_CAPTURE_BYTES",
        ):
            self.assertIn(layout_name, GPU_HEADER)
            self.assertIn(layout_name, GPU_MGPU_HEADER)

        cuda_hook_object = "tests/ds4_cuda_laguna_kernels_test_hooks.o"
        for target in (
            "tests/test_engine_mgpu_runtime",
            "tests/test_sampling",
            "tests/test_cuda_mixed_batch",
            "tests/test_cuda_laguna_model",
            "tests/probe_ds4_laguna_moe",
        ):
            with self.subTest(target=target):
                objects = rule_prerequisites(target).split()
                self.assertIn(cuda_hook_object, objects)
                self.assertNotIn("ds4_cuda.o", objects)

        decode = source_function_body(
            DS4_SOURCE,
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_token(",
            "ds4.c",
        )
        for stage in (
            "ffn-moe-gate",
            "ffn-moe-up",
            "ffn-moe-swiglu",
            "ffn-moe-col-l2",
            "ffn-moe-down-input",
            "ffn-moe-down",
            "ffn-moe-weighted",
        ):
            self.assertIn(f'"{stage}"', decode)
        self.assertRegex(
            decode,
            r"laguna_graph_diag_checkpoint_i32\(\s*g->router_selected,\s*1,"
            r"\s*DS4_N_EXPERT_USED,\s*\(int\)il,\s*\"router-selected\"\s*\)",
        )
        self.assertIn('"%s/layer-%02d-%s.%s"', DS4_SOURCE)
        self.assertIn('"ffn-moe-input"', decode)
        self.assertIn('"q8_1"', decode)

        gate_up = function_body(
            "__global__ static void glm_poolside_q4_gate_up_kernel("
        )
        self.assertIn("capture_gate", gate_up)
        self.assertIn("capture_up", gate_up)
        self.assertIn("capture_swiglu", gate_up)
        self.assertGreater(
            gate_up.rfind("capture_gate"), gate_up.find("float gate = 0.0f")
        )

        down = function_body(
            "__global__ static void glm_poolside_q4_down_kernel("
        )
        self.assertIn("capture_down", down)
        self.assertIn("capture_weighted", down)
        self.assertLess(
            down.rfind("capture_down"), down.find("output = __fadd_rn")
        )

    def test_laguna_compact_sync_checkpoint_always_has_matching_logits(
        self,
    ) -> None:
        body = source_function_body(
            DS4_SOURCE,
            "static int ds4_session_sync_internal(ds4_session *s,\n"
            "                                     const ds4_tokens *prompt,\n"
            "                                     bool allow_exact_context,\n"
            "                                     char *err,\n"
            "                                     size_t errlen) {",
            "ds4.c",
        )
        laguna_start = body.find("if (ds4_session_is_laguna(s)) {")
        glm_start = body.find("if (ds4_session_is_glm(s)) {", laguna_start)
        self.assertGreaterEqual(laguna_start, 0)
        self.assertGreater(glm_start, laguna_start)
        laguna = body[laguna_start:glm_start]

        # A compact one-token prefill may be interrupted after any successful
        # token.  The retained checkpoint must therefore either receive fresh
        # logits on every compact token, or roll back to the last checkpoint
        # whose logits were actually written.
        direct_per_token_logits = re.search(
            r"(?:last\s*\|\|\s*e->laguna_compact|"
            r"e->laguna_compact\s*\|\|\s*last)\s*"
            r"\?\s*s->logits\s*:\s*NULL",
            laguna,
        )
        logits_policy = re.search(
            r"(?:const\s+)?bool\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"(?:last\s*\|\|\s*e->laguna_compact|"
            r"e->laguna_compact\s*\|\|\s*last)\s*;",
            laguna,
        )
        named_per_token_logits = bool(
            logits_policy
            and re.search(
                rf"\b{re.escape(logits_policy.group('name'))}\s*"
                r"\?\s*s->logits\s*:\s*NULL",
                laguna,
            )
        )

        rollback = re.search(
            r"(?:const\s+)?int\s+"
            r"(?P<name>[A-Za-z_]\w*checkpoint\w*logits\w*)\s*=\s*"
            r"s->checkpoint_valid\s*\?\s*s->checkpoint\.len\s*:\s*0\s*;",
            laguna,
            re.IGNORECASE,
        )
        rollback_is_complete = False
        if rollback:
            marker = re.escape(rollback.group("name"))
            rollback_is_complete = (
                len(
                    re.findall(
                        rf"s->checkpoint\.len\s*=\s*{marker}\s*;", laguna
                    )
                )
                >= 2
                and len(
                    re.findall(
                        rf"s->checkpoint_valid\s*=\s*{marker}\s*>\s*0\s*;",
                        laguna,
                    )
                )
                >= 2
            )

        self.assertTrue(
            direct_per_token_logits
            or named_per_token_logits
            or rollback_is_complete,
            "compact cancellation can retain a checkpoint without matching logits",
        )

    def test_laguna_compact_engine_generation_uses_a_session_before_reserve(
        self,
    ) -> None:
        body = source_function_body(
            DS4_SOURCE, "int ds4_engine_generate_argmax(", "ds4.c"
        )
        manual_reserve = body.find("ds4_engine_reserve_exact_cache_session(")
        self.assertGreaterEqual(manual_reserve, 0)
        prefix = body[:manual_reserve]
        compact_guard = re.search(
            r"if\s*\([^{};]*\be->laguna_compact\b[^{};]*\)\s*\{",
            prefix,
        )
        self.assertIsNotNone(
            compact_guard,
            "compact generation must select the session path before manual reserve",
        )
        session_create = prefix.find("ds4_session_create(", compact_guard.end())
        self.assertGreater(
            session_create,
            compact_guard.end(),
            "the compact guard must route through ds4_session_create",
        )

    def test_laguna_session_eval_argmax_uses_session_eval_path(self) -> None:
        body = source_function_body(
            DS4_SOURCE, "int ds4_session_eval_argmax(", "ds4.c"
        )
        fast_path, separator, _ = body.partition("#ifdef DS4_NO_GPU")
        self.assertTrue(separator, "missing graph-backend compile guard")
        dispatch = re.search(
            r"if\s*\((?P<condition>[^{};]*)\)\s*\{\s*"
            r"if\s*\(ds4_session_eval\(s,\s*token,\s*err,\s*errlen\)\s*"
            r"!=\s*0\)\s*return\s+-1\s*;\s*"
            r"return\s+ds4_session_argmax\(s\)\s*;",
            fast_path,
        )
        self.assertIsNotNone(dispatch, "missing simple session eval/argmax path")
        condition = dispatch.group("condition")
        for predicate in (
            "ds4_session_is_cpu(s)",
            "ds4_session_is_glm(s)",
            "ds4_session_is_laguna(s)",
        ):
            with self.subTest(predicate=predicate):
                self.assertIn(predicate, condition)

    def test_laguna_compact_routed_execution_holds_a_lifecycle_lease(
        self,
    ) -> None:
        self.assertTrue(
            "active_execution_count" in CUDA_SOURCE,
            "compact context is missing its active execution reference count",
        )
        for helper in (
            "cuda_laguna_compact_exec_enter(",
            "cuda_laguna_compact_exec_leave(",
        ):
            with self.subTest(helper=helper):
                self.assertTrue(
                    helper in CUDA_SOURCE,
                    f"compact lifecycle is missing {helper}",
                )

        enter_definition = re.search(
            r"static\s+(?:int|bool)\s+cuda_laguna_compact_exec_enter\s*\(",
            CUDA_SOURCE,
        )
        self.assertIsNotNone(enter_definition)
        enter = source_function_body(
            CUDA_SOURCE, enter_definition.group(0), "CUDA"
        )
        self.assertIn("g_laguna_compact_mutex", enter)
        self.assertIn("ctx->active_execution_count", enter)
        self.assertRegex(
            enter,
            r"(?:\+\+ctx->active_execution_count|"
            r"ctx->active_execution_count\s*(?:\+\+|\+=\s*1u?))",
        )
        self.assertTrue(
            "lifecycle_epoch" in enter
            or "cuda_laguna_compact_exec_epoch_matches_locked(" in enter,
            "enter must bind its reference to the published lifecycle identity",
        )

        leave_definition = re.search(
            r"static\s+(?:int|bool)\s+cuda_laguna_compact_exec_leave\s*\(",
            CUDA_SOURCE,
        )
        self.assertIsNotNone(
            leave_definition,
            "leave must report an epoch/reference mismatch to its caller",
        )
        leave = source_function_body(
            CUDA_SOURCE, leave_definition.group(0), "CUDA"
        )
        self.assertIn("g_laguna_compact_mutex", leave)
        self.assertIn("ctx->active_execution_count", leave)
        self.assertRegex(
            leave,
            r"(?:--ctx->active_execution_count|"
            r"ctx->active_execution_count\s*(?:--|-=\s*1u?))",
        )
        self.assertTrue(
            "lifecycle_epoch" in leave
            or "cuda_laguna_compact_exec_epoch_matches_locked(" in leave,
            "leave must validate the lifecycle identity it entered",
        )

        routed = function_body(
            'extern "C" ds4_gpu_laguna_exec_result\n'
            "ds4_gpu_laguna_compact_routed_moe_one_tensor("
        )
        enter_at = routed.find("cuda_laguna_compact_exec_enter(")
        leave_at = routed.find("cuda_laguna_compact_exec_leave(", enter_at)
        core_call = re.search(
            r"(?P<name>cuda_laguna_compact_[A-Za-z0-9_]*"
            r"(?:core|entered|held|impl))\s*\(",
            routed[enter_at:],
        )
        first_ctx_dereference = routed.find("ctx->")
        self.assertGreaterEqual(enter_at, 0)
        self.assertIsNotNone(
            core_call,
            "public wrapper must put the many-return implementation behind a lease",
        )
        core_at = enter_at + core_call.start()
        self.assertGreater(core_at, enter_at)
        self.assertGreater(leave_at, core_at)
        self.assertTrue(
            first_ctx_dereference < 0 or first_ctx_dereference > enter_at,
            "public wrapper dereferences ctx before acquiring its lifecycle lease",
        )
        self.assertIn(
            "DS4_GPU_LAGUNA_EXEC_UNSAFE",
            routed[leave_at:],
            "a failed leave must make the execution result unsafe",
        )
        core_definition = re.search(
            r"static\s+ds4_gpu_laguna_exec_result\s+"
            rf"{re.escape(core_call.group('name'))}\s*\(",
            CUDA_SOURCE,
        )
        self.assertIsNotNone(core_definition)
        core = source_function_body(
            CUDA_SOURCE, core_definition.group(0), "CUDA"
        )
        self.assertIn("cuda_laguna_compact_exec_note_routes(", core)
        self.assertIn("glm_poolside_routed_moe_q4_launch(", core)

        destroy = function_body(
            "static ds4_gpu_laguna_destroy_status "
            "cuda_laguna_compact_destroy_checked("
        )
        self.assertRegex(
            destroy,
            r"if\s*\(\s*ctx->active_execution_count\s*"
            r"(?:!=\s*0u?|>\s*0u?)\s*\)\s*\{?\s*"
            r"return\s+DS4_GPU_LAGUNA_DESTROY_RECOVERABLE\s*;",
        )

    def test_laguna_compact_routed_live_ranges_are_isolated(self) -> None:
        routed = function_body(
            'extern "C" ds4_gpu_laguna_exec_result\n'
            "ds4_gpu_laguna_compact_routed_moe_one_tensor("
        )
        core_call = re.search(
            r"(?P<name>cuda_laguna_compact_[A-Za-z0-9_]*"
            r"(?:core|entered|held|impl))\s*\(",
            routed,
        )
        if core_call:
            core_definition = re.search(
                r"static\s+ds4_gpu_laguna_exec_result\s+"
                rf"{re.escape(core_call.group('name'))}\s*\(",
                CUDA_SOURCE,
            )
            self.assertIsNotNone(core_definition)
            routed = source_function_body(
                CUDA_SOURCE, core_definition.group(0), "CUDA"
            )
        writable = re.search(
            r"(?:const\s+)?cuda_laguna_compact_exec_range\s+"
            r"(?P<name>(?=[A-Za-z0-9_]*writable)"
            r"[A-Za-z_][A-Za-z0-9_]*)\s*\[\]\s*=\s*\{"
            r"(?P<entries>.*?)\n\s*\};",
            routed,
            re.DOTALL | re.IGNORECASE,
        )
        read_only = re.search(
            r"(?:const\s+)?cuda_laguna_compact_exec_range\s+"
            r"(?P<name>(?=[A-Za-z0-9_]*(?:read_only|readonly))"
            r"[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\[\]\s*=\s*\{(?P<entries>.*?)\n\s*\};",
            routed,
            re.DOTALL | re.IGNORECASE,
        )
        self.assertIsNotNone(writable, "missing declarative writable range table")
        self.assertIsNotNone(read_only, "missing declarative read-only range table")
        for pointer in (
            "out->ptr",
            "mid->ptr",
            "input_q8_scratch->ptr",
            "mid_q8_scratch->ptr",
            "aux_scratch->ptr",
        ):
            with self.subTest(writable=pointer):
                self.assertIn(pointer, writable.group("entries"))
        for pointer in ("selected->ptr", "weights->ptr", "x->ptr"):
            with self.subTest(read_only=pointer):
                self.assertIn(pointer, read_only.group("entries"))

        validators = re.finditer(
            r"static\s+(?:int|bool)\s+"
            r"(?P<name>cuda_laguna_compact_exec_[A-Za-z0-9_]*range[A-Za-z0-9_]*)"
            r"\s*\(",
            CUDA_SOURCE,
        )
        validator_name = None
        validator = None
        for candidate in validators:
            candidate_body = source_function_body(
                CUDA_SOURCE, candidate.group(0), "CUDA"
            )
            if all(
                needle in candidate_body
                for needle in ("writable_count", "read_only_count")
            ):
                validator_name = candidate.group("name")
                validator = candidate_body
                break
        self.assertIsNotNone(
            validator, "missing generic writable/read-only range validator"
        )
        self.assertIn(f"{validator_name}(", routed)
        for backing in (
            "ctx->cache_payload",
            "ctx->static_slab",
            "ctx->device_entry_to_slot",
        ):
            with self.subTest(backing=backing):
                self.assertIn(backing, validator)
        self.assertGreaterEqual(
            validator.count("cuda_laguna_compact_exec_ranges_overlap("),
            3,
        )
        self.assertRegex(
            validator,
            r"for\s*\([^)]*\bi\b[^)]*writable_count[^)]*\)",
        )
        self.assertRegex(
            validator,
            r"for\s*\([^)]*\bj\b\s*=\s*i\s*\+\s*1u?"
            r"[^)]*writable_count[^)]*\)",
        )
        self.assertRegex(
            validator,
            r"for\s*\([^)]*\bj\b[^)]*read_only_count[^)]*\)",
        )

    def test_laguna_compact_prefill_uses_the_shared_batch_kernel(self) -> None:
        self.assertIn(
            "ds4_gpu_laguna_compact_routed_moe_batch_tensor(", GPU_HEADER
        )

        batch = source_function_body(
            DS4_SOURCE,
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_batch(",
            "ds4.c",
        )
        self.assertIn("if (compact)", batch)
        self.assertIn(
            "ds4_gpu_laguna_compact_routed_moe_batch_tensor(", batch
        )

        sync = source_function_body(
            DS4_SOURCE,
            "static int ds4_session_sync_internal(ds4_session *s,\n"
            "                                     const ds4_tokens *prompt,\n"
            "                                     bool allow_exact_context,\n"
            "                                     char *err,\n"
            "                                     size_t errlen) {",
            "ds4.c",
        )
        laguna_start = sync.find("if (ds4_session_is_laguna(s)) {")
        glm_start = sync.find("if (ds4_session_is_glm(s)) {", laguna_start)
        self.assertGreaterEqual(laguna_start, 0)
        self.assertGreater(glm_start, laguna_start)
        laguna = sync[laguna_start:glm_start]
        self.assertNotRegex(
            laguna,
            r"if\s*\(e->laguna_compact\)\s*\{[^{}]*\bn\s*=\s*1u\s*;",
        )

        wrapper = source_function_body(
            CUDA_SOURCE,
            'extern "C" ds4_gpu_laguna_exec_result\n'
            "ds4_gpu_laguna_compact_routed_moe_batch_tensor(",
            "CUDA",
        )
        enter_at = wrapper.find("cuda_laguna_compact_exec_enter(")
        core_at = wrapper.find("cuda_laguna_compact_routed_moe", enter_at + 1)
        leave_at = wrapper.find("cuda_laguna_compact_exec_leave(", core_at)
        self.assertGreaterEqual(enter_at, 0)
        self.assertGreater(core_at, enter_at)
        self.assertGreater(leave_at, core_at)

        launch = source_function_body(
            CUDA_SOURCE,
            "static int glm_poolside_routed_moe_q4_launch(",
            "CUDA",
        )
        self.assertNotIn("n_tokens != 1u", launch)

    def test_laguna_graph_does_not_synthesize_unlatched_unsafe(self) -> None:
        for signature in (
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_token(",
            "static ds4_gpu_laguna_exec_result laguna_graph_forward_batch(",
        ):
            with self.subTest(signature=signature):
                body = source_function_body(DS4_SOURCE, signature, "ds4.c")
                begin = body.find("ds4_gpu_begin_commands()")
                self.assertGreater(begin, 0)
                self.assertNotIn(
                    "compact ? DS4_GPU_LAGUNA_EXEC_UNSAFE",
                    body[:begin],
                    "pre-cache caller/graph failures are request-recoverable",
                )
                fallback = body.rfind(
                    "if (!ok && execution_result == "
                    "DS4_GPU_LAGUNA_EXEC_SUCCESS) {"
                )
                self.assertGreater(fallback, 0)
                tail = body[fallback:]
                self.assertIn(
                    "execution_result = DS4_GPU_LAGUNA_EXEC_RECOVERABLE;",
                    tail,
                )
                self.assertNotIn("DS4_GPU_LAGUNA_EXEC_UNSAFE", tail)
                command_end = body.rfind(
                    "if (ds4_gpu_commands_active() && "
                    "ds4_gpu_end_commands() == 0)",
                    0,
                    fallback,
                )
                self.assertGreater(command_end, 0)
                self.assertNotIn(
                    "execution_result = DS4_GPU_LAGUNA_EXEC_UNSAFE",
                    body[command_end:fallback],
                )

    def test_laguna_decode_capture_binds_release_and_hook_logits(self) -> None:
        producer = (
            ROOT
            / "tests/oracle-producers/laguna-c7/probe_ds4_laguna_moe.c"
        )
        producer_source = producer.read_text(encoding="utf-8")

        release_object = "tests/probe_ds4_laguna_moe_release.o"
        release_binary = "tests/probe_ds4_laguna_moe_release"
        release_compile = "\n".join(rule_recipe_lines(release_object))
        self.assertIn("-DDS4_LAGUNA_RELEASE_CONTROL", release_compile)
        self.assertNotIn("-DDS4_TEST_HOOKS", release_compile)

        release_objects = rule_prerequisites(release_binary).split()
        self.assertIn(release_object, release_objects)
        self.assertIn("ds4.o", release_objects)
        self.assertIn("ds4_cuda.o", release_objects)
        self.assertNotIn("ds4_cuda_test_hooks.o", release_objects)
        self.assertNotIn(
            "tests/ds4_cuda_laguna_kernels_test_hooks.o", release_objects
        )

        hook_objects = rule_prerequisites("tests/probe_ds4_laguna_moe").split()
        self.assertIn("ds4_cuda_test_hooks.o", hook_objects)
        self.assertIn(
            "tests/ds4_cuda_laguna_kernels_test_hooks.o", hook_objects
        )
        self.assertNotIn("ds4.o", hook_objects)
        self.assertNotIn("ds4_cuda.o", hook_objects)

        self.assertIn("DS4_LAGUNA_RELEASE_CONTROL", producer_source)
        self.assertIn('"--release-logits"', producer_source)
        self.assertIn("write_release_logits(", producer_source)
        self.assertIn("read_release_logits(", producer_source)
        self.assertIn(
            "memcmp(release_logits, control_logits, LOGITS_BYTES)",
            producer_source,
        )
        self.assertIn(
            "memcmp(control_logits, captured_logits, LOGITS_BYTES)",
            producer_source,
        )

        clean_recipe = "\n".join(rule_recipe_lines("clean"))
        self.assertIn(release_binary, clean_recipe)
        self.assertIn(release_object, clean_recipe)

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
