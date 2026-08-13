#!/usr/bin/env python3
"""Source-order contract for the private qualification control channel."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DS4_C = (ROOT / "ds4.c").read_text(encoding="utf-8")
DS4_H = (ROOT / "ds4.h").read_text(encoding="utf-8")


def braced_body(source: str, signature: str) -> str:
    """Return the first function/struct definition matching signature."""
    for match in re.finditer(signature, source, re.MULTILINE | re.DOTALL):
        brace = source.find("{", match.start(), match.end())
        if brace < 0:
            continue
        depth = 0
        for index in range(brace, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[brace + 1 : index]
    raise AssertionError(f"definition not found: {signature}")


def named_typedef_body(source: str, name: str) -> str:
    """Return the anonymous-struct body ending in one exact typedef name."""
    end_match = re.search(r"\}\s*" + re.escape(name) + r"\s*;", source)
    if not end_match:
        raise AssertionError(f"typedef not found: {name}")
    close = source.find("}", end_match.start())
    depth = 0
    for index in range(close, -1, -1):
        if source[index] == "}":
            depth += 1
        elif source[index] == "{":
            depth -= 1
            if depth == 0:
                prefix = source[max(0, index - 64) : index]
                if not re.search(r"typedef\s+struct\s*$", prefix):
                    raise AssertionError(
                        f"{name} is not an anonymous struct typedef"
                    )
                return source[index + 1 : close]
    raise AssertionError(f"opening brace not found: {name}")


def ordered(body: str, *needles: str) -> None:
    cursor = -1
    for needle in needles:
        found = body.find(needle, cursor + 1)
        if found < 0:
            raise AssertionError(f"missing ordered source seam: {needle}")
        if found <= cursor:
            raise AssertionError(f"source seam out of order: {needle}")
        cursor = found


class QualificationControlIntegrationContract(unittest.TestCase):
    def test_engine_options_expose_fd_with_explicit_presence_bit(self) -> None:
        options = named_typedef_body(DS4_H, "ds4_engine_options")
        self.assertRegex(options, r"\bint\s+qualification_control_fd\s*;")
        self.assertRegex(
            options, r"\bbool\s+qualification_control_fd_set\s*;"
        )

    def test_engine_owns_the_live_control(self) -> None:
        engine = braced_body(
            DS4_C,
            r"struct\s+ds4_engine\s*\{.*?\}\s*;",
        )
        self.assertRegex(
            engine,
            r"ds4_qualification_control\s*\*\s*qualification_control\s*;",
            "engine must own the duplicated live qualification control",
        )

    def test_open_sends_retained_model_fd_before_validation_or_allocation(
        self,
    ) -> None:
        body = braced_body(
            DS4_C,
            r"static\s+int\s+ds4_engine_open_internal\s*\([^;]*?\)\s*\{",
        )
        ordered(
            body,
            "ds4_qualification_control_open(",
            "model_open(&e->model",
            "ds4_qualification_control_send_model_fd(",
            "config_validate_model(&e->model)",
            "weights_bind(&e->weights",
            "ds4_gpu_init()",
        )
        send = body.index("ds4_qualification_control_send_model_fd(")
        window = body[send : send + 500]
        self.assertIn("e->model.fd", window)
        self.assertIn("e->model.identity", window)
        self.assertIn("ds4_engine_close(e)", body[send : send + 900])

    def test_external_checkpoint_is_fully_bracketed(self) -> None:
        body = braced_body(
            DS4_C,
            r"ds4_runtime_status\s+"
            r"ds4_engine_laguna_external_checkpoint\s*\([^;]*?\)\s*\{",
        )
        ordered(
            body,
            "ds4_qualification_control_begin_sample(",
            "ds4_gpu_laguna_compact_external_checkpoint(",
            "ds4_qualification_control_finish_sample(",
        )
        self.assertNotIn(
            "return ds4_gpu_laguna_compact_external_checkpoint(",
            body,
            "the GPU sample must not return past the RESULT/ACK barrier",
        )
        finish = body.index("ds4_qualification_control_finish_sample(")
        self.assertIn("engine->model.fd", body[finish : finish + 500])
        self.assertGreaterEqual(
            body.count("DS4_RUNTIME_STATUS_UNSAFE"),
            3,
            "input, begin-barrier, and finish-barrier failures must fail safe",
        )

    def test_successful_engine_teardown_closes_the_owned_control(self) -> None:
        body = braced_body(
            DS4_C,
            r"void\s+ds4_engine_close\s*\([^;]*?\)\s*\{",
        )
        ordered(
            body,
            "ds4_qualification_control_close(",
            "free(e)",
        )
        close = body.index("ds4_qualification_control_close(")
        self.assertIn("e->qualification_control", body[close : close + 300])


if __name__ == "__main__":
    unittest.main()
