#!/usr/bin/env python3
"""Host-only contracts for Task 17 effective-prompt and output ceilings."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SOURCE = (ROOT / "ds4_server.c").read_text(encoding="utf-8")


def function_body(signature: str) -> str:
    start = SERVER_SOURCE.find(signature)
    if start < 0:
        raise AssertionError(f"missing function {signature}")
    brace = SERVER_SOURCE.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing body for {signature}")
    depth = 0
    for index in range(brace, len(SERVER_SOURCE)):
        if SERVER_SOURCE[index] == "{":
            depth += 1
        elif SERVER_SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return SERVER_SOURCE[brace + 1 : index]
    raise AssertionError(f"unterminated function {signature}")


class Task17OutputCeilingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.generate = function_body(
            "static void generate_job(server *s, server_slot *slot, job *j)"
        )

    def test_admitted_output_limit_is_never_silently_clamped(self) -> None:
        clamp = re.search(r"\bmax_tokens\s*=\s*room\b", self.generate)
        self.assertIsNone(
            clamp,
            "an admitted output limit is an exact contract: a smaller live "
            "room must reject before mutation, never silently clamp decode",
        )

    def test_final_effective_prompt_is_admitted_before_state_mutation(self) -> None:
        last_prompt_selection = self.generate.rfind("prompt_for_sync =")
        self.assertGreaterEqual(last_prompt_selection, 0)

        final_admission = re.search(
            r"request_token_counts_admit\s*\(\s*"
            r"prompt_for_sync->len\s*,\s*"
            r"(?:\(uint64_t\)\s*)?j->req\.max_tokens\s*,\s*"
            r"s->ctx_size\s*\)",
            self.generate[last_prompt_selection:],
        )
        self.assertIsNotNone(
            final_admission,
            "the selected effective prompt must be re-admitted with the exact "
            "output ceiling, not only the originally rendered prompt",
        )

        final_admission_position = last_prompt_selection + final_admission.start()
        first_mutation = min(
            position
            for marker in (
                "kv_cache_store_current(",
                "kv_cache_try_load(",
                "kv_cache_slot_suppress_continued(",
                "ds4_session_set_progress(",
                "server_session_sync(",
            )
            if (position := self.generate.find(marker)) >= 0
        )
        self.assertLess(
            final_admission_position,
            first_mutation,
            "effective-prompt rejection must happen before session/cache state "
            "can be stored, loaded, suppressed, or synchronized",
        )

    def test_recovery_reuses_one_cumulative_output_count(self) -> None:
        label = self.generate.find("decode_again:")
        self.assertGreaterEqual(label, 0)
        before_retry = self.generate[:label]
        retry_body = self.generate[label:]

        request_wide_count = re.search(
            r"\bint\s+completion\s*=\s*0\s*;", before_retry
        )
        self.assertIsNotNone(
            request_wide_count,
            "the request-wide generated-token count must be initialized once "
            "before any decode attempt",
        )
        retry_reset = re.search(r"\bcompletion\s*=\s*0\s*;", retry_body)
        self.assertIsNone(
            retry_reset,
            "decode_again recovery must not reset the generated-token count "
            "and reopen the admitted output budget",
        )
        self.assertGreaterEqual(
            retry_body.count("goto decode_again;"),
            2,
            "both invalid-DSML recovery paths must remain covered by the "
            "request-wide output ceiling",
        )

    def test_think_tool_recovery_uses_the_worker_slot_session(self) -> None:
        recovery = function_body(
            "static int chat_think_tool_recovery_attributed("
        )
        self.assertEqual(
            re.search(r"\bs->session\b", recovery),
            None,
            "forced-think recovery must not mutate slot zero while another "
            "worker owns the request",
        )
        self.assertGreaterEqual(
            recovery.find("slot->session"),
            0,
            "forced-think recovery must receive and use the request's worker "
            "slot",
        )

    def test_requests_never_enter_multi_token_speculation(self) -> None:
        helper = function_body("static bool request_allows_speculative_decode")
        self.assertRegex(
            helper,
            r"\breturn\s+false\s*;",
            "server request decoding must remain one-token-at-a-time until "
            "the speculative API can roll back at protocol stop conditions",
        )
        speculative = self.generate.find("ds4_session_eval_speculative_argmax(")
        self.assertGreaterEqual(speculative, 0)
        guard_start = self.generate.rfind("if (", 0, speculative)
        self.assertGreaterEqual(guard_start, 0)
        guard = self.generate[guard_start:speculative]
        self.assertIn(
            "request_allows_speculative_decode(&j->req)",
            guard,
            "request serving must stay on one-token coordinated decode so "
            "logical stops cannot leave a hidden precommitted tail",
        )


if __name__ == "__main__":
    unittest.main()
