#!/usr/bin/env python3
"""Host-runnable RED for Task 18's physical CUDA/HTTP fault gate.

The model-backed contract lives in ``test_task18_cuda_failure_contract.py``.
This file deliberately needs neither CUDA nor a model: it pins the build and
source seams that make that live contract truthful instead of allowing a
host-only classifier fixture to stand in for a compact CUDA failure.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
SERVER = (ROOT / "ds4_server.c").read_text(encoding="utf-8")
CUDA = (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8")
GPU_HEADER = (ROOT / "ds4_gpu.h").read_text(encoding="utf-8")


def _rule_prerequisites(target: str) -> list[str]:
    match = re.search(
        rf"(?m)^{re.escape(target)}:\s*([^\n]*(?:\\\n[ \t]+[^\n]*)*)$",
        MAKEFILE,
    )
    if match is None:
        return []
    return match.group(1).replace("\\\n", " ").split()


def _recipe(target: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(target)}:[^\n]*(?:\\\n[ \t]+[^\n]*)*\n"
        r"((?:\t[^\n]*\n?)*)",
        MAKEFILE,
    )
    return "" if match is None else match.group(1)


def _hook_objects_for_source(source: str) -> set[str]:
    result: set[str] = set()
    for match in re.finditer(
        rf"(?m)^([^\s:#]+\.o):[^\n]*\b{re.escape(source)}\b[^\n]*\n"
        r"((?:\t[^\n]*\n?)*)",
        MAKEFILE,
    ):
        if "-DDS4_TEST_HOOKS" in match.group(2):
            result.add(match.group(1))
    return result


def _function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        return ""
    brace = source.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    index = brace
    state = "code"
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if current == '"':
                state = "string"
            elif current == "'":
                state = "char"
            elif current == "/" and following == "/":
                state = "line_comment"
                index += 1
            elif current == "/" and following == "*":
                state = "block_comment"
                index += 1
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        elif state in {"string", "char"}:
            if current == "\\":
                index += 1
            elif (state == "string" and current == '"') or (
                state == "char" and current == "'"
            ):
                state = "code"
        elif state == "line_comment" and current == "\n":
            state = "code"
        elif state == "block_comment" and current == "*" and following == "/":
            state = "code"
            index += 1
        index += 1
    return ""


class Task18CudaFaultBuildContract(unittest.TestCase):
    def test_dedicated_hook_server_links_production_main_and_hooked_cuda(self) -> None:
        target = "ds4-server-test-hooks"
        prerequisites = _rule_prerequisites(target)
        self.assertTrue(
            prerequisites,
            "missing dedicated ds4-server-test-hooks build target",
        )
        self.assertNotIn("ds4_test.o", prerequisites)
        self.assertNotIn("ds4_test", prerequisites)

        required_sources = ("ds4_server.c", "ds4.c", "ds4_cuda.cu")
        for source in required_sources:
            with self.subTest(source=source):
                hooked = _hook_objects_for_source(source)
                self.assertTrue(
                    hooked,
                    f"no -DDS4_TEST_HOOKS object is built from {source}",
                )
                self.assertTrue(
                    hooked.intersection(prerequisites),
                    f"{target} does not link its hooked {source} object",
                )

        self.assertNotIn("ds4_server.o", prerequisites)
        self.assertNotIn("ds4.o", prerequisites)
        self.assertNotIn("ds4_cuda.o", prerequisites)
        self.assertRegex(_recipe(target), r"\$\((?:NVCC|DS4_LINK)\)|\bnvcc\b")


class Task18CudaFaultSourceContract(unittest.TestCase):
    def test_request_barrier_has_a_one_shot_post_quiescence_unsafe_hook(self) -> None:
        symbol = "DS4_GPU_LAGUNA_CACHE_FAULT_REQUEST_BARRIER_UNSAFE"
        self.assertTrue(
            symbol in GPU_HEADER, "public test enum lacks barrier fault"
        )
        self.assertTrue(symbol in CUDA, "CUDA test enum lacks barrier fault")
        self.assertTrue(
            "request_barrier_unsafe_failures" in GPU_HEADER,
            "compact test snapshot lacks a consumed barrier-fault counter",
        )

        body = _function_body(CUDA, "ds4_gpu_laguna_compact_request_barrier(")
        self.assertTrue(body, "missing production compact request barrier")
        sync = body.find("cudaDeviceSynchronize()")
        quiescent = body.find("ctx->active_load_phase")
        consume = body.find(symbol)
        self.assertGreaterEqual(sync, 0)
        self.assertGreater(quiescent, sync)
        self.assertGreater(
            consume,
            quiescent,
            "barrier fault must be consumed only after CUDA sync and quiescence",
        )
        tail = body[consume:]
        for token in (
            "cuda_laguna_compact_take_cache_fault",
            "request_barrier_unsafe_failures",
            "ctx->cache_unsafe = 1",
            "return DS4_GPU_LAGUNA_EXEC_UNSAFE",
        ):
            with self.subTest(barrier_token=token):
                haystack = body[: consume + len(symbol)] if token.startswith(
                    "cuda_laguna_compact_take"
                ) else tail
                self.assertTrue(token in haystack, f"barrier hook lacks {token}")

    def test_fault_controls_are_cli_only_and_compiled_out_of_production(self) -> None:
        required = {
            "--test-compact-fault",
            "--test-evidence-fd",
            "pread-error",
            "cuda-copy",
            "event-completion",
            "request-barrier-unsafe",
        }
        for spelling in required:
            with self.subTest(spelling=spelling):
                self.assertTrue(
                    spelling in SERVER,
                    f"server test CLI lacks {spelling}",
                )

        hook_blocks = "\n".join(
            match.group(1)
            for match in re.finditer(
                r"#if(?:def|\s+defined\()\s*DS4_TEST_HOOKS\)?"
                r"(.*?)#endif",
                SERVER,
                re.DOTALL,
            )
        )
        self.assertTrue(
            "--test-compact-fault" in hook_blocks,
            "fault selector is not inside a DS4_TEST_HOOKS block",
        )
        self.assertTrue(
            "--test-evidence-fd" in hook_blocks,
            "evidence descriptor is not inside a DS4_TEST_HOOKS block",
        )
        self.assertIsNone(
            re.search(r'getenv\("DS4_TEST_[^"\n]*FAULT', SERVER),
            "fault selection must be an explicit test-binary CLI, not ambient env",
        )

    def test_server_evidence_is_derived_from_live_compact_snapshots(self) -> None:
        locked_copy_fields = (
            "cache_slot_empty_count",
            "cache_slot_ready_count",
            "cache_slot_loading_count",
            "cache_slot_in_use_count",
            "cache_slot_total_refs",
            "cache_payload_id",
            "pinned_staging_ids",
        )
        for field in locked_copy_fields:
            with self.subTest(snapshot_field=field):
                self.assertTrue(
                    field in GPU_HEADER,
                    "fault evidence must use scalar/identity values copied "
                    f"while the CUDA compact mutex is held: missing {field}",
                )
        body = _function_body(SERVER, "server_test_cuda_fault_evidence_emit(")
        self.assertTrue(
            body,
            "missing server_test_cuda_fault_evidence_emit physical evidence seam",
        )
        for required in (
            "ds4_gpu_test_laguna_compact_active_snapshot",
            "ds4_gpu_test_laguna_compact_routed_origin_snapshot",
            *locked_copy_fields,
            "cache_payload_allocation_attempts",
            "pinned_staging_allocation_attempts",
            "model_mapping_registered_bytes",
            "whole_model_copied_bytes",
            "opportunistic_range_allocated_bytes",
            "request_barrier_unsafe_failures",
        ):
            with self.subTest(required=required):
                self.assertTrue(
                    required in body,
                    f"physical evidence emitter lacks {required}",
                )
        self.assertIsNotNone(
            re.search(r"\b(?:write|dprintf)\s*\(", body),
            "physical evidence emitter does not write its evidence FD",
        )

    def test_physical_evidence_precedes_the_http_failure_action(self) -> None:
        generate = _function_body(SERVER, "static void generate_job(")
        self.assertTrue(generate, "missing production generate_job")
        evidence = generate.find("server_test_cuda_fault_evidence_emit(")
        response = generate.find("server_emit_prepared_error_terminal(", evidence)
        self.assertGreaterEqual(evidence, 0, "failure funnel emits no CUDA evidence")
        self.assertGreater(
            response,
            evidence,
            "restore/fault evidence must be committed before the HTTP result",
        )


if __name__ == "__main__":
    unittest.main()
