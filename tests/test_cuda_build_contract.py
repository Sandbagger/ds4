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

        self.assertEqual(manifest["schema"], "laguna-attention-auto-fixture/v1")
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
            '"poolside-auto-t22-gqa9-derived"',
            "LAGUNA_ATTENTION_AUTO_FIXTURE_DIR",
            "22u, 256u, 48u, 8u",
            "5.0e-7f",
            "1.0e-5f",
            '"token20-head43"',
        ):
            self.assertIn(required, LAGUNA_KERNEL_TEST)

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
                rf"if \(ok && il == 0\) \{{\s*failed_stage = \"{stage} diagnostic\";"
                rf"\s*ok = laguna_graph_diag_checkpoint\(\s*g->{tensor},"
                rf"\s*n_tokens,\s*{width},\s*0,\s*\"{stage}\"\s*\);",
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
