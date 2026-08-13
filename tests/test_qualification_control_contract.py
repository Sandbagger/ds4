#!/usr/bin/env python3
"""Source-order contract for the private qualification control channel."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DS4_C = (ROOT / "ds4.c").read_text(encoding="utf-8")
DS4_H = (ROOT / "ds4.h").read_text(encoding="utf-8")
GPU_H = (ROOT / "ds4_gpu.h").read_text(encoding="utf-8")


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

    def test_engine_consumes_the_inherited_control_endpoint_after_duplication(
        self,
    ) -> None:
        internal = braced_body(
            DS4_C,
            r"static\s+int\s+ds4_engine_open_internal\s*\([^;]*?\)\s*\{",
        )
        self.assertNotIn(
            "close(opt->qualification_control_fd)",
            internal,
            "the shared wrapper owns the endpoint on every internal exit",
        )
        for signature in (
            r"int\s+ds4_engine_open\s*\([^;]*?\)\s*\{",
            r"int\s+ds4_engine_create_with_gpu_config\s*\([^;]*?\)\s*\{",
        ):
            with self.subTest(signature=signature):
                body = braced_body(DS4_C, signature)
                ordered(
                    body,
                    "ds4_engine_open_internal(",
                    "close(opt->qualification_control_fd)",
                    "return rc",
                )
                self.assertIn("opt && opt->qualification_control_fd_set", body)

        body = internal
        ordered(
            body,
            "ds4_qualification_control_open(",
            "model_open(&e->model",
        )

    def test_external_checkpoint_delegates_barrier_to_gpu_quiescence(self) -> None:
        body = braced_body(
            DS4_C,
            r"ds4_runtime_status\s+"
            r"ds4_engine_laguna_external_checkpoint\s*\([^;]*?\)\s*\{",
        )
        self.assertIn("ds4_gpu_laguna_external_checkpoint_barrier", body)
        self.assertIn("ds4_gpu_laguna_compact_external_checkpoint(", body)
        self.assertIn("barrier_ptr", body)
        self.assertNotIn(
            "ds4_qualification_control_begin_sample(",
            body,
            "READY cannot precede the CUDA execution/quiescence lock",
        )
        self.assertNotIn(
            "ds4_qualification_control_finish_sample(",
            body,
            "RESULT cannot follow release of the CUDA execution lock",
        )
        self.assertGreaterEqual(
            body.count("DS4_RUNTIME_STATUS_UNSAFE"),
            2,
            "invalid inputs and a failed delegated barrier must fail safe",
        )

    def test_gpu_checkpoint_api_accepts_synchronous_barrier_callbacks(
        self,
    ) -> None:
        barrier = braced_body(
            GPU_H,
            r"typedef\s+struct\s*\{[^}]*?\}\s*"
            r"ds4_gpu_laguna_external_checkpoint_barrier\s*;",
        )
        self.assertIn("sample_ready", barrier)
        self.assertIn("sample_result", barrier)
        self.assertIn("userdata", barrier)
        self.assertRegex(
            GPU_H,
            r"ds4_gpu_laguna_compact_external_checkpoint\s*\([^;]*"
            r"const\s+ds4_gpu_laguna_external_checkpoint_barrier\s*\*"
            r"\s*barrier[^;]*\)\s*;",
        )

    def test_engine_teardown_preserves_control_until_final_disposal(
        self,
    ) -> None:
        body = braced_body(
            DS4_C,
            r"void\s+ds4_engine_close\s*\([^;]*?\)\s*\{",
        )
        last_destroy = body.rfind("ds4_gpu_laguna_compact_destroy(")
        last_retained_return = body.rfind("return;")
        close = body.index("ds4_qualification_control_close(")
        final_free = body.rfind("free(e)")
        self.assertGreaterEqual(last_destroy, 0)
        self.assertGreater(
            close,
            last_destroy,
            "the control must outlive compact CUDA quiescence",
        )
        self.assertGreater(
            close,
            last_retained_return,
            "every retained-engine return must preserve the live control",
        )
        self.assertGreater(final_free, close)
        self.assertNotIn(
            "return;",
            body[close:final_free],
            "control close must be committed only on the final free path",
        )
        self.assertIn("e->qualification_control", body[close : close + 300])
        self.assertIn("e->qualification_control = NULL", body[close : close + 300])


if __name__ == "__main__":
    unittest.main()
