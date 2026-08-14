#!/usr/bin/env python3
"""Host-only RED contracts for the missing 0a7ad77 ds4.c hardenings."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DS4_SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")


def function_body(signature: str) -> str:
    """Return a C function body, including nested compound statements."""
    start = DS4_SOURCE.find(signature)
    if start < 0:
        raise AssertionError(f"missing ds4.c function {signature}")
    brace = DS4_SOURCE.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing body for ds4.c function {signature}")
    depth = 0
    for index in range(brace, len(DS4_SOURCE)):
        if DS4_SOURCE[index] == "{":
            depth += 1
        elif DS4_SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return DS4_SOURCE[brace + 1 : index]
    raise AssertionError(f"unterminated ds4.c function {signature}")


def rejection_tail(body: str, diagnostic: str) -> str:
    """Return one rejection path from its unique diagnostic to its return."""
    start = body.find(diagnostic)
    if start < 0:
        raise AssertionError(f"missing rejection diagnostic {diagnostic!r}")
    end = body.find("return false;", start)
    if end < 0:
        raise AssertionError(f"rejection {diagnostic!r} has no false return")
    return body[start : end + len("return false;")]


class MainSessionRecoveryHardeningContractTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.alloc = function_body("static bool metal_graph_alloc_raw_cap(")
        cls.close = function_body("void ds4_engine_close(ds4_engine *e)")
        cls.mixed = function_body(
            "static bool metal_graph_eval_mixed_prefill_decode("
        )

    def assert_rejection_frees_graph(self, diagnostic: str) -> None:
        path = rejection_tail(self.alloc, diagnostic)
        self.assertRegex(
            path,
            r"metal_graph_free\s*\(\s*g\s*\)\s*;\s*return\s+false\s*;",
            f"{diagnostic} must release the graph state initialized before "
            "rejecting the tensor-parallel configuration",
        )

    def test_uneven_tp_partner_rejection_frees_partial_graph(self) -> None:
        self.assert_rejection_frees_graph(
            "CUDA tensor parallelism requires an even multi-GPU placement"
        )

    def test_invalid_tp_expert_shape_rejection_frees_partial_graph(self) -> None:
        self.assert_rejection_frees_graph(
            "CUDA tensor parallelism requires an even-expert DeepSeek model"
        )

    def test_upper_half_tp_placement_rejection_frees_partial_graph(self) -> None:
        self.assert_rejection_frees_graph(
            "CUDA tensor parallelism expects layer homes in lower-half"
        )

    def test_engine_close_frees_shared_prefill_before_gpu_cleanup(self) -> None:
        self.assertRegex(
            self.close,
            r"if\s*\(\s*e->shared_prefill_workspace_ready\s*\)\s*\{\s*"
            r"metal_graph_free_prefill_workspace\s*\(\s*"
            r"&e->shared_prefill_workspace\s*\)\s*;\s*"
            r"e->shared_prefill_workspace_ready\s*=\s*false\s*;\s*\}\s*"
            r"ds4_gpu_cleanup\s*\(\s*\)\s*;",
            "engine close must release the engine-owned shared prefill tensors "
            "while the GPU backend is still alive, then clear readiness before "
            "global GPU cleanup",
        )

    def test_decode_token_malloc_failure_flows_through_sync_cleanup(self) -> None:
        malloc_start = self.mixed.find("int32_t *tokens = malloc(")
        self.assertGreaterEqual(malloc_start, 0)
        next_stage = self.mixed.find(
            "metal_graph_warmup_prefill_kernels(", malloc_start
        )
        self.assertGreater(next_stage, malloc_start)
        malloc_path = self.mixed[malloc_start:next_stage]
        self.assertNotIn(
            "return false;",
            malloc_path,
            "allocation failure must preserve the common GPU cleanup path",
        )
        self.assertRegex(
            malloc_path,
            r"if\s*\(\s*!tokens\s*\)\s*\{\s*ok\s*=\s*false\s*;\s*\}"
            r"\s*else\s*\{",
            "allocation failure must become the shared ok=false state",
        )
        synchronize = re.search(
            r"if\s*\(\s*ok\s*\)\s*ok\s*=\s*"
            r"ds4_gpu_end_commands\s*\(\s*\)\s*!=\s*0\s*;\s*"
            r"else\s*\(\s*void\s*\)\s*ds4_gpu_synchronize\s*\(\s*\)\s*;",
            self.mixed,
        )
        self.assertIsNotNone(
            synchronize,
            "the shared failure path must synchronize outstanding GPU work",
        )
        self.assertGreater(synchronize.start(), malloc_start)

    def test_saved_prefill_pointer_is_bounded_validated_and_guarded(self) -> None:
        declaration = re.search(
            r"ds4_gpu_tensor\s*\*\s*saved_prefill_cur\s*=\s*"
            r"prefill_src_tier\s*>=\s*0\s*&&\s*"
            r"prefill_src_tier\s*<\s*DS4_MAX_GPUS\s*\?\s*"
            r"g->cur_hc_by_tier\s*\[\s*prefill_src_tier\s*\]\s*:\s*"
            r"NULL\s*;",
            self.mixed,
        )
        self.assertIsNotNone(
            declaration,
            "the saved pointer read must prove the active tier is in bounds",
        )
        null_gate = re.search(
            r"if\s*\(\s*ok\s*&&\s*!saved_prefill_cur\s*\)\s*"
            r"ok\s*=\s*false\s*;",
            self.mixed,
        )
        self.assertIsNotNone(
            null_gate,
            "a missing in-range prefill pointer must fail before aliasing it",
        )
        self.assertGreater(null_gate.start(), declaration.end())

        sync = self.mixed.find("ds4_gpu_synchronize()")
        self.assertGreater(sync, null_gate.end())
        free_view = self.mixed.find("ds4_gpu_tensor_free(last_prefill_hc)", sync)
        self.assertGreater(free_view, sync)
        cleanup = self.mixed[sync:free_view]
        self.assertRegex(
            cleanup,
            r"if\s*\(\s*saved_prefill_cur\s*\)\s*\{\s*"
            r"g->cur_hc_by_tier\s*\[\s*prefill_src_tier\s*\]\s*=\s*"
            r"saved_prefill_cur\s*;\s*\}",
            "common cleanup must never index the tier array to restore a null "
            "or out-of-range saved pointer",
        )


if __name__ == "__main__":
    unittest.main()
