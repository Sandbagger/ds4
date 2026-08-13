#!/usr/bin/env python3
"""Host-only contract tests for Laguna server request preparation.

The production runner is intentionally not scraped for implementation strings.
Instead, the DS4_SERVER_TEST binary must expose this narrow stdio seam:

    ds4_test --server-token-admission-stdio --context-tokens N

The child reads one JSON object per line and writes one JSON object per line.
Supported operations are:

* ``{"op":"ping"}`` returns ``{"ok":true}`` without touching state.
* ``{"op":"request", "method":"POST", "path":..., "body":...}`` calls
  the production HTTP parse/render/admit path without opening a socket.  It
  returns ``{"http_status":N, "body":{...}}``.
* ``{"op":"snapshot"}`` returns a read-only state fingerprint containing
  session count and positions plus memory/disk KV-cache facts.
* ``{"op":"quit"}`` returns ``{"ok":true}`` and exits.

The test-only tokenizer may be deterministic and synthetic, but the endpoint
and inference requests must share the same production prepare function.  This
keeps the fast test host-only while the model-backed HTTP acceptance test can
later verify exact Laguna vocabulary counts on DGX.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ds4.token-admission/v1"
MODEL = "laguna-s-2.1"
TEMPLATE_REVISION = "poolside-laguna-s-2.1-native-nothink-v1"
CONTEXT_TOKENS = 2048
RESULT_KEYS = {
    "schema",
    "model",
    "template_revision",
    "templated_input_tokens",
    "requested_output_tokens",
    "context_tokens",
    "fits",
    "rejection_code",
}
SNAPSHOT_KEYS = {
    "session_count",
    "session_positions",
    "session_token_hashes",
    "memory_cache_entries",
    "disk_cache_entries",
    "disk_cache_bytes",
}


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer using the supplied tool."},
            {"role": "user", "content": "What is the weather in Brussels?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    body.update(overrides)
    return body


class Driver:
    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self.proc = subprocess.Popen(
            [
                str(executable),
                "--server-token-admission-stdio",
                "--context-tokens",
                str(CONTEXT_TOKENS),
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            response = self.rpc({"op": "ping"})
            if response != {"ok": True}:
                raise AssertionError(
                    f"unexpected token-admission driver handshake: {response!r}"
                )
        except BaseException:
            self._terminate()
            raise

    def rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise AssertionError("token-admission driver pipes are unavailable")
        try:
            self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self._raise_missing_seam()
        line = self.proc.stdout.readline()
        if not line:
            self._raise_missing_seam()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"driver returned non-JSON output: {line!r}") from exc
        if not isinstance(response, dict):
            raise AssertionError(f"driver response must be an object: {response!r}")
        return response

    def _raise_missing_seam(self) -> None:
        self.proc.wait(timeout=5)
        stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
        raise AssertionError(
            "missing host-only DS4_SERVER_TEST seam: expected "
            "--server-token-admission-stdio; child stderr was " + repr(stderr)
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                response = self.rpc({"op": "quit"})
                if response != {"ok": True}:
                    raise AssertionError(f"unexpected driver shutdown response: {response!r}")
            finally:
                if self.proc.stdin is not None:
                    try:
                        self.proc.stdin.close()
                    except BrokenPipeError:
                        pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def _terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()


class AdmissionContract(unittest.TestCase):
    driver: Driver

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = Driver(SERVER)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "driver"):
            cls.driver.close()

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.driver.rpc({"op": "snapshot"})
        self.assertEqual(set(snapshot), SNAPSHOT_KEYS)
        self.assertIsInstance(snapshot["session_count"], int)
        self.assertIsInstance(snapshot["session_positions"], list)
        self.assertIsInstance(snapshot["session_token_hashes"], list)
        self.assertIsInstance(snapshot["memory_cache_entries"], int)
        self.assertIsInstance(snapshot["disk_cache_entries"], int)
        self.assertIsInstance(snapshot["disk_cache_bytes"], int)
        return snapshot

    def request(
        self, path: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        response = self.driver.rpc(
            {"op": "request", "method": "POST", "path": path, "body": body}
        )
        self.assertEqual(set(response), {"http_status", "body"})
        self.assertIsInstance(response["http_status"], int)
        self.assertIsInstance(response["body"], dict)
        return response["http_status"], response["body"]

    def assert_result_shape(self, result: dict[str, Any]) -> None:
        self.assertEqual(set(result), RESULT_KEYS)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["template_revision"], TEMPLATE_REVISION)
        self.assertIs(type(result["templated_input_tokens"]), int)
        self.assertIs(type(result["requested_output_tokens"]), int)
        self.assertIs(type(result["context_tokens"]), int)
        self.assertIs(type(result["fits"]), bool)
        self.assertGreaterEqual(result["templated_input_tokens"], 0)
        self.assertGreaterEqual(result["requested_output_tokens"], 0)
        self.assertEqual(result["context_tokens"], CONTEXT_TOKENS)
        self.assertEqual(result["rejection_code"] is None, result["fits"])

    def admission(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, result = self.request("/v1/token-admission", body)
        self.assert_result_shape(result)
        return status, result

    def assert_rejected(
        self, body: dict[str, Any], code: str
    ) -> dict[str, Any]:
        before = self.snapshot()
        status, result = self.admission(body)
        after = self.snapshot()
        self.assertGreaterEqual(status, 400)
        self.assertLess(status, 500)
        self.assertFalse(result["fits"])
        self.assertEqual(result["rejection_code"], code)
        self.assertEqual(after, before, "admission mutated session or KV/cache state")
        return result

    def test_exact_fit_and_one_token_overflow_are_not_clamped(self) -> None:
        probe_body = _chat_body(requested_output_tokens=1)
        initial = self.snapshot()
        status, probe = self.admission(probe_body)
        self.assertEqual(self.snapshot(), initial)
        self.assertEqual(status, 200)
        self.assertTrue(probe["fits"])
        remaining = CONTEXT_TOKENS - probe["templated_input_tokens"]
        self.assertGreater(remaining, 0)

        before = self.snapshot()
        status, exact = self.admission(
            _chat_body(requested_output_tokens=remaining)
        )
        middle = self.snapshot()
        self.assertEqual(status, 200)
        self.assertTrue(exact["fits"])
        self.assertIsNone(exact["rejection_code"])
        self.assertEqual(exact["templated_input_tokens"], probe["templated_input_tokens"])
        self.assertEqual(exact["requested_output_tokens"], remaining)
        self.assertEqual(
            exact["templated_input_tokens"] + exact["requested_output_tokens"],
            CONTEXT_TOKENS,
        )
        self.assertEqual(middle, before)

        overflow = self.assert_rejected(
            _chat_body(requested_output_tokens=remaining + 1),
            "context_overflow",
        )
        self.assertEqual(overflow["requested_output_tokens"], remaining + 1)
        self.assertEqual(overflow["templated_input_tokens"], probe["templated_input_tokens"])

    def test_admission_and_inference_share_prepare_result(self) -> None:
        output_tokens = 17
        admission_status, admission = self.admission(
            _chat_body(requested_output_tokens=output_tokens)
        )
        inference_status, inference = self.request(
            "/v1/chat/completions", _chat_body(max_tokens=output_tokens)
        )
        self.assertEqual(admission_status, 200)
        self.assertEqual(inference_status, 200)
        self.assert_result_shape(inference)
        self.assertEqual(inference, admission)

    def test_invalid_output_token_numbers_have_stable_rejection(self) -> None:
        for value in (0, -1, 1.5):
            with self.subTest(value=value):
                result = self.assert_rejected(
                    _chat_body(requested_output_tokens=value),
                    "invalid_output_tokens",
                )
                self.assertEqual(result["requested_output_tokens"], 0)

    def test_malformed_tools_are_invalid_request(self) -> None:
        malformed = [
            {"type": "function", "function": {"parameters": {"type": "object"}}},
            {"type": "function", "function": "get_weather"},
        ]
        for tool in malformed:
            with self.subTest(tool=tool):
                self.assert_rejected(
                    _chat_body(tools=[tool], requested_output_tokens=8),
                    "invalid_request",
                )

    def test_required_tool_choice_is_explicitly_unsupported(self) -> None:
        self.assert_rejected(
            _chat_body(tool_choice="required", requested_output_tokens=8),
            "unsupported_tool_choice",
        )

    def test_mismatched_model_family_is_rejected(self) -> None:
        result = self.assert_rejected(
            _chat_body(model="deepseek-v4-flash", requested_output_tokens=8),
            "model_mismatch",
        )
        self.assertEqual(result["model"], MODEL)

    def test_unknown_request_field_is_rejected(self) -> None:
        self.assert_rejected(
            _chat_body(requested_output_tokens=8, unexpected_extension=True),
            "invalid_request",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "ds4_test",
        help="DS4_SERVER_TEST binary exposing --server-token-admission-stdio",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=("admission",),
        default=[],
        help="contract group to run (currently admission only)",
    )
    args, remaining = parser.parse_known_args()
    if args.case and "admission" not in args.case:
        parser.error("no selected contract cases")
    args.unittest_args = [sys.argv[0], *remaining]
    return args


if __name__ == "__main__":
    ARGS = _parse_args()
    SERVER = ARGS.server.resolve()
    if not SERVER.is_file():
        raise SystemExit(f"server test binary not found: {SERVER}")
    unittest.main(argv=ARGS.unittest_args)
else:
    SERVER = ROOT / "ds4_test"
