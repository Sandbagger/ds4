#!/usr/bin/env python3
"""Host-only source contracts for standalone CUDA links and cleanup policy."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
