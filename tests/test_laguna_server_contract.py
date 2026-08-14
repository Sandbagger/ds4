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
* ``{"op":"metrics", "body":{...}}`` drives the production terminal
  response emitters with a deterministic request-metrics lifecycle.  The body
  contains ``protocol`` (``openai_chat``, ``responses``, or ``anthropic``),
  ``stream``, ``fixture`` (``visible`` or ``buffered_no_emit``), and
  ``terminal_status``.  OpenAI also accepts ``include_usage``; Responses
  accepts ``native_terminal`` (``completed``, ``incomplete``, or ``failed``).
  It returns the protocol body, without HTTP headers, as
  ``{"protocol_request_id":"...", "wire":"..."}``.
* ``{"op":"snapshot"}`` returns a read-only state fingerprint containing
  sessions, live/tool replay state, and memory/disk cache facts including LRU
  order.
* ``{"op":"seed_tool_replay", ...}`` seeds one in-memory sampled tool replay.
* ``{"op":"seed_disk_tool_replay", ...}`` seeds one disk-backed sampled tool
  replay without warming the in-memory replay table.
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
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ds4.token-admission/v1"
MODEL = "laguna-s-2.1"
INPUT_MODEL = "laguna-s-2.1-chat"
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
    "tool_memory_entries",
    "tool_memory_fingerprint",
    "tool_memory_lru_fingerprint",
    "live_tool_state_fingerprint",
    "memory_cache_entries",
    "memory_cache_fingerprint",
    "memory_cache_lru_fingerprint",
    "disk_cache_entries",
    "disk_cache_bytes",
    "disk_cache_fingerprint",
    "disk_cache_lru_fingerprint",
}
REQUEST_METRICS_SCHEMA = "ds4.runtime.request/v1"
PROTOCOL_VALIDATION_KEYS = {
    "http_status",
    "code",
    "tools_enabled",
    "normalized_model",
    "mutation_fingerprint_before",
    "mutation_fingerprint_after",
}
ACCEPTED_MODEL_ALIASES = (
    "laguna-s-2.1",
    "laguna-s-2.1-chat",
    "laguna-s-2.1-no-think",
    "laguna-s-2.1-nothink",
    "laguna-s-2.1-reasoner",
    "poolside/laguna-s-2.1",
)
REQUEST_METRICS_KEYS = {
    "schema",
    "request_id",
    "instance_id",
    "snapshot_seq",
    "prompt_tokens",
    "generated_tokens",
    "ttft_ns",
    "prefill_tokens_per_second",
    "visible_decode_tokens_per_second",
    "wall_time_ns",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "model_file_read_operations",
    "model_file_read_bytes",
    "model_file_read_ns",
    "host_to_device_bytes",
    "host_to_device_ns",
    "page_advice_attempts",
    "page_advice_bytes",
    "page_advice_failures",
    "page_advice_complete_monotonic_ns",
    "terminal_status",
}
TERMINAL_STATUSES = {
    "completed",
    "cancelled",
    "rejected",
    "recoverable_error",
    "unsafe_error",
}
VISIBLE_COUNTERS = {
    "cache_hits": "1",
    "cache_misses": "2",
    "cache_evictions": "3",
    "model_file_read_operations": "4",
    "model_file_read_bytes": "5",
    "model_file_read_ns": "6",
    "host_to_device_bytes": "7",
    "host_to_device_ns": "8",
    "page_advice_attempts": "9",
    "page_advice_bytes": "10",
    "page_advice_failures": "11",
}
SELECTED_CASES: frozenset[str] = frozenset()


def _schema_objects(value: Any, schema: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("schema") == schema:
            found.append(value)
        for child in value.values():
            found.extend(_schema_objects(child, schema))
    elif isinstance(value, list):
        for child in value:
            found.extend(_schema_objects(child, schema))
    return found


def _object_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            found.add(value["id"])
        for child in value.values():
            found.update(_object_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_object_ids(child))
    return found


def _parse_sse_wire(wire: str) -> list[tuple[str | None, Any]]:
    frames: list[tuple[str | None, Any]] = []
    normalized = wire.replace("\r\n", "\n")
    for raw_frame in normalized.split("\n\n"):
        if not raw_frame.strip():
            continue
        event: str | None = None
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
            elif line.startswith(":"):
                continue
            else:
                raise AssertionError(f"invalid SSE line in metrics wire: {line!r}")
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            frames.append((event, data))
        else:
            try:
                frames.append((event, json.loads(data)))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"metrics SSE frame contains invalid JSON: {data!r}"
                ) from exc
    return frames


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": INPUT_MODEL,
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


def _protocol_body(
    protocol: str,
    *,
    model: str = INPUT_MODEL,
    tool_choice: str = "auto",
    stream: bool = False,
) -> tuple[str, dict[str, Any]]:
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    if protocol == "openai_chat":
        return "/v1/chat/completions", _chat_body(
            model=model,
            tool_choice=tool_choice,
            max_tokens=8,
            stream=stream,
        )
    if protocol == "responses":
        return "/v1/responses", {
            "model": model,
            "input": [
                {"role": "user", "content": "What is the weather in Brussels?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "parameters": parameters,
                }
            ],
            "tool_choice": tool_choice,
            "max_output_tokens": 8,
            "stream": stream,
        }
    if protocol == "anthropic":
        anthropic_choice = {
            "auto": "auto",
            "none": "none",
            # Anthropic's `any` is the protocol-native required-tool mode.
            "required": "any",
        }[tool_choice]
        return "/v1/messages", {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the weather in Brussels?"}
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "input_schema": parameters,
                }
            ],
            "tool_choice": {"type": anthropic_choice},
            "max_tokens": 8,
            "stream": stream,
        }
    raise AssertionError(f"unknown protocol fixture: {protocol}")


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
        return self.rpc_line(json.dumps(payload, separators=(",", ":")))

    def rpc_line(self, payload: str) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise AssertionError("token-admission driver pipes are unavailable")
        try:
            self.proc.stdin.write(payload + "\n")
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

    def metrics(
        self,
        protocol: str,
        *,
        stream: bool,
        fixture: str = "visible",
        terminal_status: str = "completed",
        include_usage: bool | None = None,
        native_terminal: str | None = None,
    ) -> tuple[str, str]:
        body: dict[str, Any] = {
            "protocol": protocol,
            "stream": stream,
            "fixture": fixture,
            "terminal_status": terminal_status,
        }
        if include_usage is not None:
            body["include_usage"] = include_usage
        if native_terminal is not None:
            body["native_terminal"] = native_terminal
        response = self.rpc({"op": "metrics", "body": body})
        if set(response) != {"protocol_request_id", "wire"}:
            raise AssertionError(
                "unexpected protocol-metrics driver response: " + repr(response)
            )
        protocol_request_id = response["protocol_request_id"]
        wire = response["wire"]
        if not isinstance(protocol_request_id, str) or not protocol_request_id:
            raise AssertionError("protocol request ID must be a non-empty string")
        if not isinstance(wire, str) or not wire:
            raise AssertionError("protocol metrics wire must be a non-empty string")
        return protocol_request_id, wire

    def protocol_validate(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.rpc(
            {
                "op": "protocol_validate",
                "method": "POST",
                "path": path,
                "body": body,
            }
        )
        if set(response) != PROTOCOL_VALIDATION_KEYS:
            raise AssertionError(
                "unexpected protocol-validation driver response: "
                + repr(response)
            )
        if type(response["http_status"]) is not int:
            raise AssertionError("protocol-validation status must be an integer")
        if response["code"] is not None and not isinstance(
            response["code"], str
        ):
            raise AssertionError("protocol-validation code must be null or a string")
        if type(response["tools_enabled"]) is not bool:
            raise AssertionError(
                "protocol-validation tools_enabled must be a boolean"
            )
        if response["normalized_model"] is not None and not isinstance(
            response["normalized_model"], str
        ):
            raise AssertionError(
                "protocol-validation normalized_model must be null or a string"
            )
        for key in (
            "mutation_fingerprint_before",
            "mutation_fingerprint_after",
        ):
            if not isinstance(response[key], str) or not response[key]:
                raise AssertionError(
                    f"protocol-validation {key} must be a non-empty string"
                )
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
        if SELECTED_CASES and "admission" not in SELECTED_CASES:
            raise unittest.SkipTest("admission contract not selected")
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
        self.assertIsInstance(snapshot["tool_memory_entries"], int)
        self.assertIsInstance(snapshot["tool_memory_fingerprint"], str)
        self.assertIsInstance(snapshot["tool_memory_lru_fingerprint"], str)
        self.assertIsInstance(snapshot["live_tool_state_fingerprint"], str)
        self.assertIsInstance(snapshot["memory_cache_entries"], int)
        self.assertIsInstance(snapshot["memory_cache_fingerprint"], str)
        self.assertIsInstance(snapshot["memory_cache_lru_fingerprint"], str)
        self.assertIsInstance(snapshot["disk_cache_entries"], int)
        self.assertIsInstance(snapshot["disk_cache_bytes"], int)
        self.assertIsInstance(snapshot["disk_cache_fingerprint"], str)
        self.assertIsInstance(snapshot["disk_cache_lru_fingerprint"], str)
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

    def request_raw(
        self, path: str, body_json: str
    ) -> tuple[int, dict[str, Any]]:
        wire = (
            '{"op":"request","method":"POST","path":'
            + json.dumps(path, separators=(",", ":"))
            + ',"body":'
            + body_json
            + "}"
        )
        response = self.driver.rpc_line(wire)
        self.assertEqual(set(response), {"http_status", "body"})
        self.assertIsInstance(response["http_status"], int)
        self.assertIsInstance(response["body"], dict)
        return response["http_status"], response["body"]

    @staticmethod
    def raw_object(fields: list[tuple[str, str]]) -> str:
        return "{" + ",".join(
            json.dumps(key) + ":" + value for key, value in fields
        ) + "}"

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

    @staticmethod
    def inference_body(body: dict[str, Any]) -> dict[str, Any]:
        inference = dict(body)
        if "requested_output_tokens" in inference:
            inference["max_tokens"] = inference.pop("requested_output_tokens")
        return inference

    def inference(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, result = self.request(
            "/v1/chat/completions", self.inference_body(body)
        )
        self.assert_result_shape(result)
        return status, result

    def assert_accepted_with_parity(self, body: dict[str, Any]) -> dict[str, Any]:
        before = self.snapshot()
        admission_status, admission = self.admission(body)
        after_admission = self.snapshot()
        inference_status, inference = self.inference(body)
        after_inference = self.snapshot()
        self.assertEqual(admission_status, 200)
        self.assertEqual(inference_status, 200)
        self.assertTrue(admission["fits"])
        self.assertIsNone(admission["rejection_code"])
        self.assertEqual(inference, admission)
        self.assertEqual(after_admission, before)
        self.assertEqual(after_inference, before)
        return admission

    def assert_rejected(
        self, body: dict[str, Any], code: str
    ) -> dict[str, Any]:
        before = self.snapshot()
        admission_status, admission = self.admission(body)
        after_admission = self.snapshot()
        inference_status, inference = self.inference(body)
        after_inference = self.snapshot()
        for status, result in (
            (admission_status, admission),
            (inference_status, inference),
        ):
            self.assertGreaterEqual(status, 400)
            self.assertLess(status, 500)
            self.assertFalse(result["fits"])
            self.assertEqual(result["rejection_code"], code)
        self.assertEqual(inference, admission)
        self.assertEqual(
            after_admission,
            before,
            "admission mutated session, replay, or cache state",
        )
        self.assertEqual(
            after_inference,
            before,
            "dry-run inference mutated session, replay, or cache state",
        )
        return admission

    def assert_one_endpoint_rejected(
        self, path: str, body: dict[str, Any], code: str
    ) -> dict[str, Any]:
        before = self.snapshot()
        status, result = self.request(path, body)
        after = self.snapshot()
        self.assertGreaterEqual(status, 400)
        self.assertLess(status, 500)
        self.assert_result_shape(result)
        self.assertFalse(result["fits"])
        self.assertEqual(result["rejection_code"], code)
        self.assertEqual(after, before)
        return result

    def assert_raw_rejected(
        self,
        admission_body: str,
        inference_body: str,
        code: str,
    ) -> dict[str, Any]:
        before = self.snapshot()
        admission_status, admission = self.request_raw(
            "/v1/token-admission", admission_body
        )
        after_admission = self.snapshot()
        inference_status, inference = self.request_raw(
            "/v1/chat/completions", inference_body
        )
        after_inference = self.snapshot()
        for status, result in (
            (admission_status, admission),
            (inference_status, inference),
        ):
            self.assertGreaterEqual(status, 400)
            self.assertLess(status, 500)
            self.assert_result_shape(result)
            self.assertFalse(result["fits"])
            self.assertEqual(result["rejection_code"], code)
        self.assertEqual(inference, admission)
        self.assertEqual(after_admission, before)
        self.assertEqual(after_inference, before)
        return admission

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
        inference_status, inference = self.inference(
            _chat_body(requested_output_tokens=output_tokens)
        )
        self.assertEqual(admission_status, 200)
        self.assertEqual(inference_status, 200)
        self.assertEqual(inference, admission)

    def test_same_family_chat_alias_is_canonicalized(self) -> None:
        status, admission = self.admission(
            _chat_body(model=INPUT_MODEL, requested_output_tokens=8)
        )
        inference_status, inference = self.inference(
            _chat_body(model=INPUT_MODEL, requested_output_tokens=8)
        )
        self.assertEqual(status, 200)
        self.assertEqual(inference_status, 200)
        self.assertEqual(admission["model"], MODEL)
        self.assertEqual(admission["template_revision"], TEMPLATE_REVISION)
        self.assertEqual(inference, admission)

    def test_inference_omitted_limit_resolves_exact_policy_count(self) -> None:
        before = self.snapshot()
        status, result = self.request(
            "/v1/chat/completions",
            _chat_body(model=INPUT_MODEL),
        )
        self.assertEqual(self.snapshot(), before)
        self.assert_result_shape(result)
        self.assertEqual(status, 200)
        self.assertTrue(result["fits"])
        self.assertIsNone(result["rejection_code"])
        self.assertGreater(result["requested_output_tokens"], 0)
        self.assertEqual(
            result["requested_output_tokens"],
            CONTEXT_TOKENS - result["templated_input_tokens"],
            "omitted inference limit was not resolved to one exact policy count",
        )

    def test_native_nothink_template_alias_policy_is_explicit(self) -> None:
        accepted_aliases = (
            INPUT_MODEL,
            "laguna-s-2.1-no-think",
            "laguna-s-2.1-nothink",
        )
        for alias in accepted_aliases:
            with self.subTest(alias=alias, policy="accepted"):
                admission_status, admission = self.admission(
                    _chat_body(model=alias, requested_output_tokens=8)
                )
                inference_status, inference = self.inference(
                    _chat_body(model=alias, requested_output_tokens=8)
                )
                self.assertEqual(admission_status, 200)
                self.assertEqual(inference_status, 200)
                self.assertTrue(admission["fits"])
                self.assertEqual(admission["model"], MODEL)
                self.assertEqual(
                    admission["template_revision"], TEMPLATE_REVISION
                )
                self.assertEqual(inference, admission)

        semantic_changing_aliases = (
            MODEL,
            "laguna-s-2.1-reasoner",
            "poolside/laguna-s-2.1",
        )
        for alias in semantic_changing_aliases:
            with self.subTest(alias=alias, policy="rejected"):
                result = self.assert_rejected(
                    _chat_body(model=alias, requested_output_tokens=8),
                    "invalid_request",
                )
                self.assertEqual(result["model"], MODEL)
                self.assertEqual(
                    result["template_revision"], TEMPLATE_REVISION
                )

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

    def test_developer_role_is_accepted_with_endpoint_parity(self) -> None:
        result = self.assert_accepted_with_parity(
            _chat_body(
                messages=[
                    {
                        "role": "developer",
                        "content": "Answer with one short sentence.",
                    },
                    {"role": "user", "content": "Say hello."},
                ],
                requested_output_tokens=8,
            )
        )
        self.assertGreater(result["templated_input_tokens"], 0)

    def test_optional_message_name_is_accepted_without_token_difference(self) -> None:
        unnamed = self.assert_accepted_with_parity(
            _chat_body(
                messages=[{"role": "user", "content": "Say hello."}],
                requested_output_tokens=8,
            )
        )
        named = self.assert_accepted_with_parity(
            _chat_body(
                messages=[
                    {"role": "user", "name": "operator", "content": "Say hello."}
                ],
                requested_output_tokens=8,
            )
        )
        self.assertEqual(
            named["templated_input_tokens"],
            unnamed["templated_input_tokens"],
        )

    def test_strict_text_content_array_is_accepted_with_endpoint_parity(self) -> None:
        structured = self.assert_accepted_with_parity(
            _chat_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Say"},
                            {"type": "text", "text": " hello."},
                        ],
                    }
                ],
                requested_output_tokens=8,
            )
        )
        flattened = self.assert_accepted_with_parity(
            _chat_body(
                messages=[{"role": "user", "content": "Say hello."}],
                requested_output_tokens=8,
            )
        )
        self.assertEqual(
            structured["templated_input_tokens"],
            flattened["templated_input_tokens"],
        )

        malformed_or_nontext = (
            [{"type": "text"}],
            [{"type": "text", "text": 7}],
            [{"text": "missing type"}],
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/image.png"},
                }
            ],
            ["bare string part"],
            [7],
            [{"type": "text", "text": "extra field", "extra": True}],
        )
        for content in malformed_or_nontext:
            with self.subTest(content=content):
                self.assert_rejected(
                    _chat_body(
                        messages=[{"role": "user", "content": content}],
                        requested_output_tokens=8,
                    ),
                    "invalid_request",
                )

    def test_tool_call_arguments_object_matches_string_form(self) -> None:
        def body_with_arguments(arguments: str | dict[str, str]) -> dict[str, Any]:
            return _chat_body(
                messages=[
                    {"role": "user", "content": "Check Brussels weather."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_arguments_object",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_arguments_object",
                        "content": "12 C and cloudy",
                    },
                ],
                requested_output_tokens=8,
            )

        object_result = self.assert_accepted_with_parity(
            body_with_arguments({"city": "Brussels"})
        )
        string_result = self.assert_accepted_with_parity(
            body_with_arguments('{"city":"Brussels"}')
        )
        self.assertEqual(
            object_result["templated_input_tokens"],
            string_result["templated_input_tokens"],
        )

    def test_null_tools_are_treated_as_an_empty_tool_list(self) -> None:
        null_tools = self.assert_accepted_with_parity(
            _chat_body(tools=None, requested_output_tokens=8)
        )
        empty_tools = self.assert_accepted_with_parity(
            _chat_body(tools=[], requested_output_tokens=8)
        )
        self.assertEqual(null_tools, empty_tools)

    def test_nested_message_content_must_have_a_supported_shape(self) -> None:
        malformed_content = (True, 7, {"text": "not a content block"})
        for content in malformed_content:
            with self.subTest(content=content):
                self.assert_rejected(
                    _chat_body(
                        messages=[{"role": "user", "content": content}],
                        requested_output_tokens=8,
                    ),
                    "invalid_request",
                )

    def test_message_role_is_required_unique_and_known(self) -> None:
        for message in (
            {"content": "missing role"},
            {"role": "moderator", "content": "bogus role"},
        ):
            with self.subTest(message=message):
                self.assert_rejected(
                    _chat_body(messages=[message], requested_output_tokens=8),
                    "invalid_request",
                )

        duplicate_role_messages = (
            '[{"role":"user","role":"assistant","content":"duplicate"}]'
        )
        admission = self.raw_object(
            [
                ("model", json.dumps(INPUT_MODEL)),
                ("messages", duplicate_role_messages),
                ("requested_output_tokens", "8"),
            ]
        )
        inference = self.raw_object(
            [
                ("model", json.dumps(INPUT_MODEL)),
                ("messages", duplicate_role_messages),
                ("max_tokens", "8"),
            ]
        )
        self.assert_raw_rejected(admission, inference, "invalid_request")

    def test_assistant_tool_calls_must_be_complete_and_unique(self) -> None:
        incomplete_calls = (
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            },
            {
                "id": "call_missing_type",
                "function": {"name": "get_weather", "arguments": "{}"},
            },
            {
                "id": "call_missing_name",
                "type": "function",
                "function": {"arguments": "{}"},
            },
            {
                "id": "call_missing_arguments",
                "type": "function",
                "function": {"name": "get_weather"},
            },
        )
        for tool_call in incomplete_calls:
            with self.subTest(tool_call=tool_call):
                self.assert_rejected(
                    _chat_body(
                        messages=[
                            {"role": "user", "content": "Use the tool."},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [tool_call],
                            },
                        ],
                        requested_output_tokens=8,
                    ),
                    "invalid_request",
                )

        complete_call = (
            '[{"id":"call_duplicate","type":"function",'
            '"function":{"name":"get_weather","arguments":"{}"}}]'
        )
        duplicate_tool_calls_messages = (
            '[{"role":"assistant","content":"",'
            '"tool_calls":'
            + complete_call
            + ',"tool_calls":'
            + complete_call
            + "}]"
        )
        admission = self.raw_object(
            [
                ("model", json.dumps(INPUT_MODEL)),
                ("messages", duplicate_tool_calls_messages),
                ("requested_output_tokens", "8"),
            ]
        )
        inference = self.raw_object(
            [
                ("model", json.dumps(INPUT_MODEL)),
                ("messages", duplicate_tool_calls_messages),
                ("max_tokens", "8"),
            ]
        )
        self.assert_raw_rejected(admission, inference, "invalid_request")

    def test_rejection_precedence_is_independent_of_field_order(self) -> None:
        normal_messages = '[{"role":"user","content":"hello"}]'
        oversized_messages = json.dumps(
            [{"role": "user", "content": "overflow " * CONTEXT_TOKENS}],
            separators=(",", ":"),
        )
        precedence_cases: tuple[
            tuple[str, list[tuple[str, str]], str], ...
        ] = (
            (
                "model_mismatch_over_invalid_output_tokens",
                [("model", '"deepseek-v4-flash"'), ("__output__", "0")],
                "model_mismatch",
            ),
            (
                "model_mismatch_over_invalid_request",
                [("model", '"deepseek-v4-flash"'), ("unexpected", "true")],
                "model_mismatch",
            ),
            (
                "invalid_request_over_unsupported_tool_choice",
                [("unexpected", "true"), ("tool_choice", '"required"')],
                "invalid_request",
            ),
            (
                "invalid_request_over_invalid_output_tokens",
                [("unexpected", "true"), ("__output__", "0")],
                "invalid_request",
            ),
            (
                "unsupported_tool_choice_over_invalid_output_tokens",
                [("tool_choice", '"required"'), ("__output__", "0")],
                "unsupported_tool_choice",
            ),
            (
                "unsupported_tool_choice_over_context_overflow",
                [
                    ("tool_choice", '"required"'),
                    ("__output__", str(CONTEXT_TOKENS)),
                ],
                "unsupported_tool_choice",
            ),
            (
                "invalid_output_tokens_over_context_overflow",
                [("__output__", "0"), ("messages", oversized_messages)],
                "invalid_output_tokens",
            ),
        )

        def body_for(
            conflicts: list[tuple[str, str]], output_field: str
        ) -> str:
            fields = [
                (output_field if key == "__output__" else key, value)
                for key, value in conflicts
            ]
            keys = {key for key, _ in fields}
            if "model" not in keys:
                fields.append(("model", json.dumps(INPUT_MODEL)))
            if "messages" not in keys:
                fields.append(("messages", normal_messages))
            if output_field not in keys:
                fields.append((output_field, "8"))
            return self.raw_object(fields)

        for name, conflicts, expected in precedence_cases:
            for order, ordered_conflicts in (
                ("forward", conflicts),
                ("reverse", list(reversed(conflicts))),
            ):
                with self.subTest(case=name, order=order):
                    self.assert_raw_rejected(
                        body_for(
                            ordered_conflicts, "requested_output_tokens"
                        ),
                        body_for(ordered_conflicts, "max_tokens"),
                        expected,
                    )

    def test_output_field_mapping_is_endpoint_specific(self) -> None:
        admission_wrong = _chat_body(max_tokens=8)
        admission_wrong_result = self.assert_one_endpoint_rejected(
            "/v1/token-admission", admission_wrong, "invalid_request"
        )
        admission_both = _chat_body(requested_output_tokens=8, max_tokens=8)
        admission_both_result = self.assert_one_endpoint_rejected(
            "/v1/token-admission", admission_both, "invalid_request"
        )

        inference_wrong = _chat_body(requested_output_tokens=8)
        inference_wrong_result = self.assert_one_endpoint_rejected(
            "/v1/chat/completions", inference_wrong, "invalid_request"
        )
        inference_both = _chat_body(requested_output_tokens=8, max_tokens=8)
        inference_both_result = self.assert_one_endpoint_rejected(
            "/v1/chat/completions", inference_both, "invalid_request"
        )
        self.assertEqual(inference_wrong_result, admission_wrong_result)
        self.assertEqual(inference_both_result, admission_both_result)

    def test_seeded_tool_replay_state_is_read_only_for_prepare(self) -> None:
        fixtures = (
            ("seed_tool_replay", "call_memory_seed"),
            ("seed_disk_tool_replay", "call_disk_seed"),
        )
        for operation, call_id in fixtures:
            with self.subTest(operation=operation):
                before_seed = self.snapshot()
                seeded_response = self.driver.rpc(
                    {
                        "op": operation,
                        "call_id": call_id,
                        "name": "get_weather",
                        "arguments": {"city": "Brussels"},
                        "sampled_text": (
                            "<｜DSML｜tool_calls>\n"
                            "<｜DSML｜invoke name=\"get_weather\">\n"
                            "<｜DSML｜parameter name=\"city\" string=\"true\">"
                            "Brussels</｜DSML｜parameter>\n"
                            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                        ),
                    }
                )
                self.assertEqual(seeded_response, {"ok": True})
                seeded = self.snapshot()
                self.assertNotEqual(seeded, before_seed)
                if operation == "seed_tool_replay":
                    self.assertGreater(
                        seeded["tool_memory_entries"],
                        before_seed["tool_memory_entries"],
                    )
                    self.assertNotEqual(
                        seeded["tool_memory_fingerprint"],
                        before_seed["tool_memory_fingerprint"],
                    )
                else:
                    self.assertGreater(
                        seeded["disk_cache_entries"],
                        before_seed["disk_cache_entries"],
                    )
                    self.assertNotEqual(
                        seeded["disk_cache_fingerprint"],
                        before_seed["disk_cache_fingerprint"],
                    )
                    self.assertEqual(
                        seeded["tool_memory_fingerprint"],
                        before_seed["tool_memory_fingerprint"],
                        "disk seed unexpectedly warmed in-memory replay state",
                    )

                replay = _chat_body(
                    requested_output_tokens=8,
                    messages=[
                        {"role": "user", "content": "Check Brussels weather."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Brussels"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "12 C and cloudy",
                        },
                    ],
                )
                admission_status, admission = self.admission(replay)
                self.assertEqual(self.snapshot(), seeded)
                inference_status, inference = self.inference(replay)
                self.assertEqual(self.snapshot(), seeded)
                self.assertEqual(admission_status, 200)
                self.assertEqual(inference_status, 200)
                self.assertEqual(inference, admission)


class ProtocolValidationContract(unittest.TestCase):
    """Freeze protocol validation before any session or replay mutation.

    This is deliberately narrower than stream/non-stream output equivalence.
    It exercises only the production protocol parsers' semantic validation and
    normalization boundary through a host-only driver operation.
    """

    driver: Driver
    protocols = ("openai_chat", "responses", "anthropic")

    @classmethod
    def setUpClass(cls) -> None:
        if SELECTED_CASES and "protocol" not in SELECTED_CASES:
            raise unittest.SkipTest("protocol validation contract not selected")
        cls.driver = Driver(SERVER)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "driver"):
            cls.driver.close()

    def validate(
        self,
        protocol: str,
        *,
        model: str = INPUT_MODEL,
        tool_choice: str = "auto",
        stream: bool = False,
    ) -> dict[str, Any]:
        path, body = _protocol_body(
            protocol,
            model=model,
            tool_choice=tool_choice,
            stream=stream,
        )
        return self.driver.protocol_validate(path, body)

    def assert_accepted(
        self, result: dict[str, Any], *, tools_enabled: bool
    ) -> None:
        self.assertEqual(result["http_status"], 200)
        self.assertIsNone(result["code"])
        self.assertIs(result["tools_enabled"], tools_enabled)
        self.assertEqual(result["normalized_model"], MODEL)

    def assert_rejected(
        self,
        result: dict[str, Any],
        *,
        code: str,
        normalized_model: str | None,
    ) -> None:
        self.assertEqual(result["http_status"], 400)
        self.assertEqual(result["code"], code)
        self.assertIs(result["tools_enabled"], False)
        self.assertEqual(result["normalized_model"], normalized_model)
        self.assertEqual(
            result["mutation_fingerprint_after"],
            result["mutation_fingerprint_before"],
            "protocol rejection mutated session, replay, or cache state",
        )

    def test_auto_and_none_are_supported_for_both_stream_modes(self) -> None:
        for protocol in self.protocols:
            with self.subTest(protocol=protocol):
                for stream in (False, True):
                    self.assert_accepted(
                        self.validate(
                            protocol, tool_choice="auto", stream=stream
                        ),
                        tools_enabled=True,
                    )
                    self.assert_accepted(
                        self.validate(
                            protocol, tool_choice="none", stream=stream
                        ),
                        tools_enabled=False,
                    )

    def test_required_is_stably_rejected_for_both_stream_modes(self) -> None:
        for protocol in self.protocols:
            with self.subTest(protocol=protocol):
                for stream in (False, True):
                    self.assert_rejected(
                        self.validate(
                            protocol, tool_choice="required", stream=stream
                        ),
                        code="unsupported_tool_choice",
                        normalized_model=MODEL,
                    )

    def test_wrong_model_family_is_stably_rejected_for_both_stream_modes(
        self,
    ) -> None:
        for protocol in self.protocols:
            with self.subTest(protocol=protocol):
                for stream in (False, True):
                    self.assert_rejected(
                        self.validate(
                            protocol,
                            model="deepseek-v4-flash",
                            tool_choice="auto",
                            stream=stream,
                        ),
                        code="model_mismatch",
                        normalized_model=None,
                    )

    def test_laguna_model_aliases_normalize_for_every_protocol(self) -> None:
        for protocol in self.protocols:
            with self.subTest(protocol=protocol):
                for alias in ACCEPTED_MODEL_ALIASES:
                    for stream in (False, True):
                        self.assert_accepted(
                            self.validate(
                                protocol,
                                model=alias,
                                tool_choice="auto",
                                stream=stream,
                            ),
                            tools_enabled=True,
                        )


class MetricsContract(unittest.TestCase):
    """Freeze request metrics placement without requiring a model or GPU.

    The ``visible`` driver fixture uses these deterministic lifecycle facts:
    acceptance=0.9s, prefill=1.1..1.3s, visible decode=1.5..2.0s,
    page advice=2.7s, first successful visible emission=2.8s, finish=3.0s,
    22 prompt tokens, 8 total generated tokens, and 6 visible decoded tokens.
    The deliberately late emission makes TTFT (1.9s) observably different
    from first internal decode (0.6s), while generated count and visible rate
    are independently checkable.
    """

    driver: Driver

    @classmethod
    def setUpClass(cls) -> None:
        if SELECTED_CASES and "metrics" not in SELECTED_CASES:
            raise unittest.SkipTest("metrics contract not selected")
        cls.driver = Driver(SERVER)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "driver"):
            cls.driver.close()

    def assert_metric_shape(
        self, metric: dict[str, Any], protocol_request_id: str
    ) -> None:
        self.assertEqual(set(metric), REQUEST_METRICS_KEYS)
        self.assertEqual(metric["schema"], REQUEST_METRICS_SCHEMA)
        for key in ("request_id", "instance_id"):
            self.assertIsInstance(metric[key], str)
            parsed = uuid.UUID(metric[key])
            self.assertEqual(str(parsed), metric[key].lower())
        self.assertNotEqual(metric["request_id"], metric["instance_id"])
        self.assertNotEqual(metric["request_id"], protocol_request_id)
        self.assertNotEqual(metric["instance_id"], protocol_request_id)
        for key in (
            "snapshot_seq",
            "wall_time_ns",
            *VISIBLE_COUNTERS,
        ):
            self.assertIsInstance(metric[key], str, key)
            self.assertTrue(metric[key].isdigit(), key)
        self.assertGreater(int(metric["snapshot_seq"]), 0)
        for key in ("prompt_tokens", "generated_tokens"):
            self.assertIs(type(metric[key]), int, key)
            self.assertGreaterEqual(metric[key], 0, key)
        for key in (
            "prefill_tokens_per_second",
            "visible_decode_tokens_per_second",
        ):
            self.assertIsInstance(metric[key], (int, float), key)
            self.assertGreaterEqual(metric[key], 0, key)
        for key in ("ttft_ns", "page_advice_complete_monotonic_ns"):
            if metric[key] is not None:
                self.assertIsInstance(metric[key], str, key)
                self.assertTrue(metric[key].isdigit(), key)
        self.assertIn(metric["terminal_status"], TERMINAL_STATUSES)

    def assert_visible_fixture(
        self,
        metric: dict[str, Any],
        protocol_request_id: str,
        *,
        terminal_status: str = "completed",
    ) -> None:
        self.assert_metric_shape(metric, protocol_request_id)
        self.assertEqual(metric["prompt_tokens"], 22)
        self.assertEqual(
            metric["generated_tokens"],
            8,
            "hidden generation was collapsed into the six-token visible subset",
        )
        self.assertEqual(
            metric["ttft_ns"],
            "1900000000",
            "TTFT used internal decode instead of first successful emission",
        )
        self.assertEqual(float(metric["prefill_tokens_per_second"]), 110.0)
        self.assertEqual(
            float(metric["visible_decode_tokens_per_second"]),
            10.0,
            "visible decode rate used all eight generated tokens",
        )
        self.assertEqual(metric["wall_time_ns"], "2100000000")
        for key, expected in VISIBLE_COUNTERS.items():
            self.assertEqual(metric[key], expected, key)
        self.assertEqual(
            metric["page_advice_complete_monotonic_ns"], "2700000000"
        )
        self.assertEqual(metric["terminal_status"], terminal_status)

    def assert_one_metric(
        self, payloads: list[Any], protocol_request_id: str
    ) -> dict[str, Any]:
        metrics: list[dict[str, Any]] = []
        ids: set[str] = set()
        for payload in payloads:
            metrics.extend(_schema_objects(payload, REQUEST_METRICS_SCHEMA))
            ids.update(_object_ids(payload))
        self.assertEqual(
            len(metrics),
            1,
            "a protocol response must contain exactly one request metrics object",
        )
        self.assertIn(
            protocol_request_id,
            ids,
            "driver protocol request ID is absent from the native envelope",
        )
        self.assert_metric_shape(metrics[0], protocol_request_id)
        return metrics[0]

    def test_nonstream_protocols_contain_one_metrics_object(self) -> None:
        request_ids: set[str] = set()
        instance_ids: set[str] = set()
        for protocol in ("openai_chat", "responses", "anthropic"):
            with self.subTest(protocol=protocol):
                protocol_id, wire = self.driver.metrics(protocol, stream=False)
                try:
                    payload = json.loads(wire)
                except json.JSONDecodeError as exc:
                    self.fail(f"{protocol} nonstream wire is not JSON: {exc}")
                self.assertIsInstance(payload, dict)
                metric = self.assert_one_metric([payload], protocol_id)
                self.assert_visible_fixture(metric, protocol_id)
                request_ids.add(metric["request_id"])
                instance_ids.add(metric["instance_id"])
        self.assertEqual(
            len(request_ids), 3, "request metrics IDs were reused across requests"
        )
        self.assertEqual(
            len(instance_ids), 1, "one driver process exposed multiple instance IDs"
        )

    def test_openai_stream_metrics_precede_done_even_without_usage(self) -> None:
        for include_usage in (False, True):
            with self.subTest(include_usage=include_usage):
                protocol_id, wire = self.driver.metrics(
                    "openai_chat",
                    stream=True,
                    include_usage=include_usage,
                )
                frames = _parse_sse_wire(wire)
                self.assertGreaterEqual(len(frames), 2)
                self.assertEqual(frames[-1][1], "[DONE]")
                terminal_payload = frames[-2][1]
                self.assertIsInstance(terminal_payload, dict)
                self.assertEqual(
                    terminal_payload.get("choices"),
                    [],
                    "request metrics did not use the final OpenAI usage event",
                )
                self.assertEqual(
                    "usage" in terminal_payload,
                    include_usage,
                    "native include_usage policy changed while adding metrics",
                )
                json_payloads = [
                    payload for _event, payload in frames if isinstance(payload, dict)
                ]
                metric = self.assert_one_metric(json_payloads, protocol_id)
                self.assert_visible_fixture(metric, protocol_id)
                self.assertEqual(
                    _schema_objects(terminal_payload, REQUEST_METRICS_SCHEMA),
                    [metric],
                    "metrics were not in the event immediately before [DONE]",
                )

    def test_responses_metrics_exist_only_on_terminal_events(self) -> None:
        status_by_terminal = {
            "completed": "completed",
            "incomplete": "completed",
            "failed": "recoverable_error",
        }
        for native_terminal, runtime_status in status_by_terminal.items():
            with self.subTest(native_terminal=native_terminal):
                protocol_id, wire = self.driver.metrics(
                    "responses",
                    stream=True,
                    native_terminal=native_terminal,
                    terminal_status=runtime_status,
                )
                frames = _parse_sse_wire(wire)
                self.assertGreaterEqual(len(frames), 2)
                json_payloads = [
                    payload for _event, payload in frames if isinstance(payload, dict)
                ]
                metric = self.assert_one_metric(json_payloads, protocol_id)
                self.assert_visible_fixture(
                    metric, protocol_id, terminal_status=runtime_status
                )
                expected_type = f"response.{native_terminal}"
                terminal_payload = json_payloads[-1]
                self.assertEqual(terminal_payload.get("type"), expected_type)
                self.assertEqual(
                    _schema_objects(terminal_payload, REQUEST_METRICS_SCHEMA),
                    [metric],
                )
                for payload in json_payloads[:-1]:
                    self.assertEqual(
                        _schema_objects(payload, REQUEST_METRICS_SCHEMA),
                        [],
                        "Responses metrics leaked into a non-terminal event",
                    )

    def test_anthropic_metrics_are_in_delta_before_message_stop(self) -> None:
        protocol_id, wire = self.driver.metrics("anthropic", stream=True)
        frames = _parse_sse_wire(wire)
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(frames[-1][0], "message_stop")
        self.assertEqual(frames[-1][1], {"type": "message_stop"})
        self.assertEqual(frames[-2][0], "message_delta")
        self.assertIsInstance(frames[-2][1], dict)
        self.assertEqual(frames[-2][1].get("type"), "message_delta")
        json_payloads = [
            payload for _event, payload in frames if isinstance(payload, dict)
        ]
        metric = self.assert_one_metric(json_payloads, protocol_id)
        self.assert_visible_fixture(metric, protocol_id)
        self.assertEqual(
            _schema_objects(frames[-2][1], REQUEST_METRICS_SCHEMA), [metric]
        )

    def test_buffered_decode_without_emission_has_null_ttft(self) -> None:
        protocol_id, wire = self.driver.metrics(
            "openai_chat",
            stream=False,
            fixture="buffered_no_emit",
            terminal_status="cancelled",
        )
        payload = json.loads(wire)
        metric = self.assert_one_metric([payload], protocol_id)
        self.assertEqual(metric["prompt_tokens"], 2)
        self.assertEqual(metric["generated_tokens"], 2)
        self.assertIsNone(metric["ttft_ns"])
        self.assertGreater(
            float(metric["visible_decode_tokens_per_second"]),
            0.0,
            "fixture did not prove decoded output can remain un-emitted",
        )
        self.assertEqual(metric["wall_time_ns"], "5")
        self.assertIsNone(metric["page_advice_complete_monotonic_ns"])
        self.assertEqual(metric["terminal_status"], "cancelled")

    def test_all_terminal_statuses_survive_protocol_serialization(self) -> None:
        request_ids: set[str] = set()
        for terminal_status in sorted(TERMINAL_STATUSES):
            with self.subTest(terminal_status=terminal_status):
                protocol_id, wire = self.driver.metrics(
                    "openai_chat",
                    stream=False,
                    terminal_status=terminal_status,
                )
                metric = self.assert_one_metric([json.loads(wire)], protocol_id)
                if terminal_status == "rejected":
                    self.assert_metric_shape(metric, protocol_id)
                    self.assertEqual(metric["prompt_tokens"], 22)
                    self.assertEqual(metric["generated_tokens"], 0)
                    self.assertIsNone(metric["ttft_ns"])
                    self.assertEqual(metric["prefill_tokens_per_second"], 0)
                    self.assertEqual(
                        metric["visible_decode_tokens_per_second"], 0
                    )
                    self.assertIsNone(
                        metric["page_advice_complete_monotonic_ns"]
                    )
                    for key in VISIBLE_COUNTERS:
                        self.assertEqual(metric[key], "0", key)
                    self.assertEqual(metric["terminal_status"], "rejected")
                else:
                    self.assert_visible_fixture(
                        metric, protocol_id, terminal_status=terminal_status
                    )
                request_ids.add(metric["request_id"])
        self.assertEqual(len(request_ids), len(TERMINAL_STATUSES))


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
        choices=("admission", "metrics", "protocol"),
        default=[],
        help="contract group to run (default: all)",
    )
    args, remaining = parser.parse_known_args()
    args.unittest_args = [sys.argv[0], *remaining]
    return args


if __name__ == "__main__":
    ARGS = _parse_args()
    SELECTED_CASES = frozenset(ARGS.case)
    SERVER = ARGS.server.resolve()
    if not SERVER.is_file():
        raise SystemExit(f"server test binary not found: {SERVER}")
    unittest.main(argv=ARGS.unittest_args)
else:
    SERVER = ROOT / "ds4_test"
