#!/usr/bin/env python3
"""Source contract for Poolside-compatible resident Laguna batch boundaries."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (ROOT / "ds4.c").read_text()
ORACLE_PRODUCER = (
    ROOT / "gguf-tools" / "quality-testing" / "dump_llama_logits.cpp"
).read_text()


def function_body(signature: str) -> str:
    match = re.search(re.escape(signature) + r"[^;]*?\)\s*\{", RUNTIME_SOURCE)
    if match is None:
        return ""
    start = match.start()
    opening = match.end() - 1
    depth = 0
    for offset in range(opening, len(RUNTIME_SOURCE)):
        char = RUNTIME_SOURCE[offset]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return RUNTIME_SOURCE[start : offset + 1]
    return ""


class LagunaResidentBatchContractTest(unittest.TestCase):
    def test_resident_batches_follow_pinned_poolside_ubatch_ceiling(self) -> None:
        self.assertRegex(
            ORACLE_PRODUCER,
            r"params\.n_ubatch\s*=.*512",
            "pinned Poolside producer no longer declares a 512-row ubatch",
        )
        body = function_body("static int ds4_session_sync_internal(")
        self.assertTrue(body, "missing ds4_session_sync_internal definition")
        self.assertRegex(
            body,
            r"if\s*\(\s*!e->laguna_compact\s*&&\s*n\s*>\s*512u\s*\)"
            r"\s*(?:\{\s*)?n\s*=\s*512u\s*;",
            "resident Laguna can cross the pinned Poolside 512-row ubatch",
        )


if __name__ == "__main__":
    unittest.main()
