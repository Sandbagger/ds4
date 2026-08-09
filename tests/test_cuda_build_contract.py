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
        target_line = re.search(
            r"(?m)^test-cuda-laguna-resident:[^\n]*$", MAKEFILE
        )
        self.assertIsNotNone(target_line)
        recipe_lines: list[str] = []
        for line in MAKEFILE[target_line.end() + 1 :].splitlines():
            if not line.startswith("\t"):
                break
            recipe_lines.append(line)
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


if __name__ == "__main__":
    unittest.main()
