#!/usr/bin/env python3
"""Opt-in real CUDA/HTTP fault-boundary contract for Task 18.

Each scenario launches a fresh ``ds4-server-test-hooks`` process with the
production server main and compact CUDA engine.  The child receives an exact
retained model descriptor, one explicit one-shot fault selector, and a
dedicated evidence descriptor.  The evidence record is accepted only when it
contains locked-copy cache aggregates and stable allocation IDs from the real
compact context.

Without ``DS4_TEST_MODEL`` every model-backed case skips clearly.  A guarded
DGX invocation supplies both the path and its already-open descriptor::

    env -u DS4_LOCK_FILE \
      DS4_TEST_MODEL=/absolute/path/to/laguna-s-2.1-Q4_K_M.gguf \
      DS4_TEST_MODEL_FD=9 \
      python3 tests/test_task18_cuda_failure_contract.py \
        --server ./ds4-server-test-hooks -v
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_TOKENS = 32768
PREFILL_TOKENS = 4096
CACHE_BYTES = 8 * 1024 * 1024 * 1024
MAX_OUTPUT_TOKENS = 8
MODEL_ID = "laguna-s-2.1"
EVIDENCE_SCHEMA = "ds4.test.cuda-fault/v1"
REQUEST_METRICS_SCHEMA = "ds4.runtime.request/v1"

IDENTITY_KEYS = {"device", "inode", "size_bytes", "mtime_ns"}
COMPACT_FIELDS = {
    "lifecycle",
    "model_fd_live",
    "cache_unsafe",
    "cache_slot_count",
    "cache_slot_empty_count",
    "cache_slot_ready_count",
    "cache_slot_loading_count",
    "cache_slot_in_use_count",
    "cache_slot_total_refs",
    "cache_payload_id",
    "pinned_staging_ids",
    "cache_payload_allocation_attempts",
    "pinned_staging_allocation_attempts",
    "cache_load_failures",
    "pread_error_failures",
    "cuda_copy_failures",
    "event_completion_failures",
    "request_barrier_unsafe_failures",
    "model_mapping_registered_bytes",
    "whole_model_copied_bytes",
    "opportunistic_range_allocated_bytes",
    "legacy_model_range_count",
    "legacy_model_arena_count",
}
ROUTED_ORIGIN_FIELDS = {
    "routed_projection_requests",
    "engine_slot_resolutions",
    "static_slab_resolutions",
    "model_mapping_resolutions",
    "managed_resolutions",
    "per_request_resolutions",
    "unknown_resolutions",
}
EVIDENCE_FIELDS = {
    "schema",
    "fault",
    "pid",
    "request_id",
    "execution_result",
    "headers_sent",
    "fault_armed_count",
    "fault_consumed_count",
    "restore_sequence",
    "evidence_sequence",
    "response_sequence",
    "executable",
    "model",
    "model_fd",
    "compact_before",
    "compact_after",
    "routed_origin",
}
FALLBACK_FIELDS = (
    "model_mapping_registered_bytes",
    "whole_model_copied_bytes",
    "opportunistic_range_allocated_bytes",
    "legacy_model_range_count",
    "legacy_model_arena_count",
)


def _extract_server_option(argv: list[str]) -> tuple[str, list[str]]:
    result = list(argv)
    positions = [index for index, value in enumerate(result) if value == "--server"]
    if not positions:
        return "./ds4-server-test-hooks", result
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise SystemExit("--server requires exactly one executable path")
    position = positions[0]
    server = result[position + 1]
    if not server or server.startswith("--"):
        raise SystemExit("--server requires an executable path")
    del result[position : position + 2]
    return server, result


SERVER_OPTION, UNITTEST_ARGV = _extract_server_option(sys.argv)


def _strict_json(text: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _identity(path: os.PathLike[str] | str) -> dict[str, str]:
    status = os.stat(path)
    return {
        "device": str(status.st_dev),
        "inode": str(status.st_ino),
        "size_bytes": str(status.st_size),
        "mtime_ns": str(status.st_mtime_ns),
    }


def _wire_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not IDENTITY_KEYS.issubset(value):
        raise AssertionError(f"invalid file identity: {value!r}")
    return {key: str(value[key]) for key in IDENTITY_KEYS}


def _u64(value: Any, name: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise AssertionError(f"{name} is not a canonical uint64 string: {value!r}")
    return int(value)


def _timeout(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise AssertionError(f"{name} must be finite and positive")
    return value


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request_body(*, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Reply with exactly OK."},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def _http_wire(port: int, method: str, path: str, body: Any = None) -> bytes:
    payload = b"" if body is None else json.dumps(
        body, separators=(",", ":")
    ).encode("utf-8")
    headers = [
        f"{method} {path} HTTP/1.1",
        "Host: 127.0.0.1",
        "Accept: application/json",
        "Connection: close",
    ]
    if body is not None:
        headers.extend(
            ["Content-Type: application/json", f"Content-Length: {len(payload)}"]
        )
    request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + payload
    timeout = _timeout("DS4_LAGUNA_FAULT_REQUEST_TIMEOUT", "900")
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall(request)
        response = bytearray()
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return bytes(response)
            response.extend(chunk)


def _split_http(wire: bytes) -> tuple[int, dict[str, str], bytes]:
    head, separator, body = wire.partition(b"\r\n\r\n")
    if not separator:
        raise AssertionError(f"HTTP response has no header terminator: {wire[:300]!r}")
    lines = head.decode("iso-8859-1").split("\r\n")
    match = re.fullmatch(r"HTTP/1\.[01] ([0-9]{3}) .+", lines[0])
    if match is None:
        raise AssertionError(f"invalid HTTP status line: {lines[0]!r}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, marker, value = line.partition(":")
        if not marker:
            raise AssertionError(f"invalid HTTP header: {line!r}")
        lowered = key.strip().lower()
        if lowered in headers:
            raise AssertionError(f"duplicate HTTP header: {lowered}")
        headers[lowered] = value.strip()
    return int(match.group(1)), headers, body


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


def _live_inputs() -> tuple[Path, Path, int, dict[str, str]]:
    if "DS4_LOCK_FILE" in os.environ:
        raise AssertionError("physical CUDA fault gate requires DS4_LOCK_FILE unset")
    model_text = os.environ.get("DS4_TEST_MODEL", "")
    model = Path(model_text)
    if not model.is_absolute() or model.is_symlink() or not model.is_file():
        raise AssertionError("DS4_TEST_MODEL must be an absolute nonsymlink file")
    candidate = Path(SERVER_OPTION).resolve(strict=True)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise AssertionError("--server must name the executable hook server")
    fd_text = os.environ.get("DS4_TEST_MODEL_FD", "")
    if re.fullmatch(r"[1-9][0-9]*", fd_text) is None or int(fd_text) < 3:
        raise AssertionError("DS4_TEST_MODEL_FD must name an open descriptor >= 3")
    model_fd = int(fd_text)
    try:
        descriptor_identity = _identity(f"/proc/self/fd/{model_fd}")
    except OSError as error:
        raise AssertionError("DS4_TEST_MODEL_FD is not open") from error
    model_identity = _identity(model)
    if descriptor_identity != model_identity:
        raise AssertionError("DS4_TEST_MODEL_FD identity differs from DS4_TEST_MODEL")
    return candidate, model, model_fd, model_identity


class FaultServer:
    def __init__(self, fault: str) -> None:
        self.fault = fault
        self.candidate, self.model, self.model_fd, self.model_identity = _live_inputs()
        self.candidate_identity = _identity(self.candidate)
        self.port = _loopback_port()
        self.log = tempfile.TemporaryFile(mode="w+b")
        self.evidence = tempfile.TemporaryFile(mode="w+b")
        # Omit --session-slots deliberately: the default is one non-batched
        # session, while spelling even `--session-slots 1` selects the batched
        # coordinator and adds an unrelated concurrency surface to this gate.
        command = [
            str(self.candidate),
            "--model", f"/proc/self/fd/{self.model_fd}",
            "--cuda",
            "--ctx", str(CONTEXT_TOKENS),
            "--prefill-chunk", str(PREFILL_TOKENS),
            "--ssd-streaming",
            "--ssd-streaming-cache-bytes", str(CACHE_BYTES),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--test-compact-fault", fault,
            "--test-evidence-fd", str(self.evidence.fileno()),
        ]
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            pass_fds=(self.model_fd, self.evidence.fileno()),
            start_new_session=True,
        )
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + _timeout(
            "DS4_LAGUNA_SERVER_START_TIMEOUT", "900"
        )
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"hook server exited during startup with {self.proc.returncode}:\n"
                    + self.diagnostic()
                )
            try:
                status, _headers, body = _split_http(
                    _http_wire(self.port, "GET", "/v1/runtime")
                )
                snapshot = _strict_json(body.decode("utf-8"))
                if status == 200 and snapshot.get("state") == "ready":
                    self._assert_identity(snapshot)
                    return
            except (OSError, ValueError, AssertionError) as error:
                last_error = error
            time.sleep(0.25)
        raise AssertionError("hook server did not become ready") from last_error

    def _assert_identity(self, runtime: dict[str, Any]) -> None:
        if _identity(f"/proc/{self.proc.pid}/exe") != self.candidate_identity:
            raise AssertionError("running executable identity differs from candidate")
        if _identity(f"/proc/{self.proc.pid}/fd/{self.model_fd}") != self.model_identity:
            raise AssertionError("child did not retain the exact supplied model FD")
        if _wire_identity(runtime.get("executable")) != self.candidate_identity:
            raise AssertionError("runtime executable identity differs from candidate")
        if _wire_identity(runtime.get("model")) != self.model_identity:
            raise AssertionError("runtime model identity differs from retained FD")
        if runtime.get("model", {}).get("id") != MODEL_ID:
            raise AssertionError("runtime served model ID is not canonical Laguna")

    def request(self, *, stream: bool) -> bytes:
        return _http_wire(
            self.port, "POST", "/v1/chat/completions", _request_body(stream=stream)
        )

    def records(self) -> list[dict[str, Any]]:
        deadline = time.monotonic() + 5.0
        while True:
            self.evidence.seek(0)
            lines = [line for line in self.evidence.read().splitlines() if line.strip()]
            if lines or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        return [_strict_json(line.decode("utf-8")) for line in lines]

    def wait(self, expected: int) -> None:
        status = self.proc.wait(timeout=30)
        if status != expected:
            raise AssertionError(
                f"hook server exit={status}, expected={expected}:\n{self.diagnostic()}"
            )

    def stop_clean(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
        self.wait(0)

    def diagnostic(self) -> str:
        self.log.flush()
        self.log.seek(0)
        return self.log.read().decode("utf-8", errors="replace")[-12000:]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        self.evidence.close()
        self.log.close()


@unittest.skipUnless(
    os.environ.get("DS4_TEST_MODEL"),
    "DS4_TEST_MODEL is unset; skipping real compact CUDA fault processes",
)
class Task18CudaFailureContract(unittest.TestCase):
    def _server(self, fault: str) -> FaultServer:
        server = FaultServer(fault)
        self.addCleanup(server.close)
        return server

    def _assert_compact_snapshot(self, value: Any, label: str) -> dict[str, Any]:
        self.assertIsInstance(value, dict)
        self.assertTrue(COMPACT_FIELDS.issubset(value), f"{label} fields incomplete")
        self.assertEqual(value["lifecycle"], "active")
        self.assertIs(value["model_fd_live"], True)
        slots = _u64(value["cache_slot_count"], f"{label}.cache_slot_count")
        self.assertGreater(slots, 0)
        state_total = sum(
            _u64(value[name], f"{label}.{name}")
            for name in (
                "cache_slot_empty_count",
                "cache_slot_ready_count",
                "cache_slot_loading_count",
                "cache_slot_in_use_count",
            )
        )
        self.assertEqual(state_total, slots)
        for name in COMPACT_FIELDS - {
            "lifecycle", "model_fd_live", "cache_unsafe", "pinned_staging_ids"
        }:
            _u64(value[name], f"{label}.{name}")
        staging_ids = value["pinned_staging_ids"]
        self.assertIsInstance(staging_ids, list)
        self.assertEqual(len(staging_ids), 4)
        parsed_staging = [
            _u64(item, f"{label}.pinned_staging_ids[{index}]")
            for index, item in enumerate(staging_ids)
        ]
        self.assertEqual(len(set(parsed_staging)), 4)
        self.assertNotIn(0, parsed_staging)
        self.assertGreater(_u64(value["cache_payload_id"], f"{label}.cache_payload_id"), 0)
        self.assertEqual(
            _u64(value["cache_payload_allocation_attempts"], f"{label}.payload_attempts"),
            1,
        )
        self.assertEqual(
            _u64(value["pinned_staging_allocation_attempts"], f"{label}.staging_attempts"),
            4,
        )
        for name in FALLBACK_FIELDS:
            self.assertEqual(_u64(value[name], f"{label}.{name}"), 0)
        return value

    def _assert_evidence(
        self,
        server: FaultServer,
        *,
        expected_result: str,
        headers_sent: bool,
    ) -> dict[str, Any]:
        records = server.records()
        self.assertEqual(len(records), 1, "fault must emit exactly one evidence record")
        record = records[0]
        self.assertEqual(set(record), EVIDENCE_FIELDS)
        self.assertEqual(record["schema"], EVIDENCE_SCHEMA)
        self.assertEqual(record["fault"], server.fault)
        self.assertEqual(record["pid"], server.proc.pid)
        self.assertEqual(str(uuid.UUID(record["request_id"])), record["request_id"])
        self.assertEqual(record["execution_result"], expected_result)
        self.assertIs(record["headers_sent"], headers_sent)
        self.assertEqual(_u64(record["fault_armed_count"], "fault_armed_count"), 1)
        self.assertEqual(_u64(record["fault_consumed_count"], "fault_consumed_count"), 1)
        self.assertEqual(_wire_identity(record["executable"]), server.candidate_identity)
        self.assertEqual(_wire_identity(record["model"]), server.model_identity)
        self.assertIs(type(record["model_fd"]), int)
        self.assertGreaterEqual(record["model_fd"], 3)

        before = self._assert_compact_snapshot(record["compact_before"], "before")
        after = self._assert_compact_snapshot(record["compact_after"], "after")
        for name in (
            "cache_payload_id",
            "pinned_staging_ids",
            "cache_payload_allocation_attempts",
            "pinned_staging_allocation_attempts",
        ):
            self.assertEqual(after[name], before[name], f"fixed owner changed: {name}")

        routed = record["routed_origin"]
        self.assertIsInstance(routed, dict)
        self.assertEqual(set(routed), ROUTED_ORIGIN_FIELDS)
        routed_counts = {name: _u64(routed[name], f"routed_origin.{name}") for name in routed}
        self.assertEqual(
            routed_counts["routed_projection_requests"],
            routed_counts["engine_slot_resolutions"],
        )
        for name in ROUTED_ORIGIN_FIELDS - {
            "routed_projection_requests", "engine_slot_resolutions"
        }:
            self.assertEqual(routed_counts[name], 0, f"fallback observed: {name}")
        return record

    def _assert_json_failure(
        self, wire: bytes, expected_status: int, terminal_status: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        status, headers, body = _split_http(wire)
        self.assertEqual(status, expected_status)
        self.assertIn("application/json", headers.get("content-type", ""))
        payload = _strict_json(body.decode("utf-8"))
        self.assertIsInstance(payload.get("error"), dict)
        self.assertEqual(payload["error"].get("type"), "server_error")
        self.assertEqual(payload["error"].get("code"), "server_error")
        metrics = _schema_objects(payload, REQUEST_METRICS_SCHEMA)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].get("terminal_status"), terminal_status)
        return payload, metrics[0]

    def _assert_followup_evidence(
        self,
        server: FaultServer,
        first: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        records = server.records()
        self.assertEqual(
            len(records), 2,
            "same-process follow-up must emit a second physical snapshot",
        )
        record = records[1]
        self.assertEqual(set(record), EVIDENCE_FIELDS)
        self.assertEqual(record["schema"], EVIDENCE_SCHEMA)
        self.assertEqual(record["fault"], server.fault)
        self.assertEqual(record["pid"], server.proc.pid)
        self.assertEqual(record["execution_result"], "completed")
        self.assertEqual(record["request_id"], metrics["request_id"])
        self.assertNotEqual(record["request_id"], first["request_id"])
        self.assertEqual(str(uuid.UUID(record["request_id"])), record["request_id"])
        self.assertEqual(_u64(record["fault_armed_count"], "fault_armed_count"), 1)
        self.assertEqual(
            _u64(record["fault_consumed_count"], "fault_consumed_count"), 1
        )
        self.assertEqual(
            _wire_identity(record["executable"]), server.candidate_identity
        )
        self.assertEqual(_wire_identity(record["model"]), server.model_identity)

        after = self._assert_compact_snapshot(
            record["compact_after"], "followup.after"
        )
        self.assertIs(after["cache_unsafe"], False)
        self.assertEqual(_u64(after["cache_slot_loading_count"], "loading"), 0)
        self.assertEqual(_u64(after["cache_slot_in_use_count"], "in_use"), 0)
        self.assertEqual(_u64(after["cache_slot_total_refs"], "refs"), 0)
        baseline = first["compact_before"]
        for name in (
            "cache_payload_id",
            "pinned_staging_ids",
            "cache_payload_allocation_attempts",
            "pinned_staging_allocation_attempts",
        ):
            self.assertEqual(
                after[name], baseline[name], f"follow-up changed fixed owner: {name}"
            )

        first_routed = {
            name: _u64(first["routed_origin"][name], f"first.routed.{name}")
            for name in ROUTED_ORIGIN_FIELDS
        }
        routed = record["routed_origin"]
        self.assertIsInstance(routed, dict)
        self.assertEqual(set(routed), ROUTED_ORIGIN_FIELDS)
        routed_counts = {
            name: _u64(routed[name], f"followup.routed.{name}") for name in routed
        }
        self.assertGreater(
            routed_counts["routed_projection_requests"],
            first_routed["routed_projection_requests"],
            "follow-up did not traverse the compact routed projection",
        )
        self.assertEqual(
            routed_counts["routed_projection_requests"],
            routed_counts["engine_slot_resolutions"],
        )
        for name in ROUTED_ORIGIN_FIELDS - {
            "routed_projection_requests", "engine_slot_resolutions"
        }:
            self.assertEqual(routed_counts[name], 0, f"follow-up fallback: {name}")

    def _recoverable(self, fault: str, counter: str) -> None:
        server = self._server(fault)
        pid = server.proc.pid
        _payload, metrics = self._assert_json_failure(
            server.request(stream=False), 503, "recoverable_error"
        )
        self.assertIsNone(server.proc.poll())
        record = self._assert_evidence(
            server, expected_result="recoverable", headers_sent=False
        )
        self.assertEqual(record["request_id"], metrics["request_id"])
        restore = _u64(record["restore_sequence"], "restore_sequence")
        evidence = _u64(record["evidence_sequence"], "evidence_sequence")
        response = _u64(record["response_sequence"], "response_sequence")
        self.assertLess(restore, evidence)
        self.assertLess(evidence, response)
        before, after = record["compact_before"], record["compact_after"]
        self.assertIs(before["cache_unsafe"], False)
        self.assertIs(after["cache_unsafe"], False)
        self.assertEqual(_u64(after[counter], counter) - _u64(before[counter], counter), 1)
        self.assertEqual(_u64(after["cache_slot_loading_count"], "loading"), 0)
        self.assertEqual(_u64(after["cache_slot_in_use_count"], "in_use"), 0)
        self.assertEqual(_u64(after["cache_slot_total_refs"], "refs"), 0)

        status, _headers, body = _split_http(server.request(stream=False))
        self.assertEqual(status, 200)
        followup = _strict_json(body.decode("utf-8"))
        choices = followup.get("choices")
        self.assertIsInstance(choices, list)
        self.assertTrue(
            any(
                isinstance(choice, dict)
                and isinstance(choice.get("message"), dict)
                and str(choice["message"].get("content", "")).strip()
                for choice in choices
            ),
            "same-process follow-up did not perform a real visible inference",
        )
        followup_metrics = _schema_objects(followup, REQUEST_METRICS_SCHEMA)
        self.assertEqual(len(followup_metrics), 1)
        self.assertEqual(followup_metrics[0].get("terminal_status"), "completed")
        self._assert_followup_evidence(server, record, followup_metrics[0])
        self.assertEqual(server.proc.pid, pid)
        self.assertIsNone(server.proc.poll())
        server.stop_clean()

    def test_pread_error_restores_then_503_and_same_pid_infers(self) -> None:
        self._recoverable("pread-error", "pread_error_failures")

    def test_cuda_copy_restores_then_503_and_same_pid_infers(self) -> None:
        self._recoverable("cuda-copy", "cuda_copy_failures")

    def test_event_completion_returns_structured_500_then_exits_one(self) -> None:
        server = self._server("event-completion")
        _payload, metrics = self._assert_json_failure(
            server.request(stream=False), 500, "unsafe_error"
        )
        record = self._assert_evidence(
            server, expected_result="unsafe", headers_sent=False
        )
        self.assertEqual(record["request_id"], metrics["request_id"])
        self.assertIsNone(record["restore_sequence"])
        evidence = _u64(record["evidence_sequence"], "evidence_sequence")
        response = _u64(record["response_sequence"], "response_sequence")
        self.assertLess(evidence, response)
        before, after = record["compact_before"], record["compact_after"]
        self.assertIs(before["cache_unsafe"], False)
        self.assertIs(after["cache_unsafe"], True)
        self.assertEqual(
            _u64(after["event_completion_failures"], "event_completion_failures")
            - _u64(before["event_completion_failures"], "event_completion_failures"),
            1,
        )
        self.assertEqual(_u64(after["cache_slot_loading_count"], "loading"), 1)
        self.assertGreater(_u64(after["cache_slot_total_refs"], "refs"), 0)
        server.wait(1)

    def test_request_barrier_unsafe_abrupts_after_visible_frame(self) -> None:
        server = self._server("request-barrier-unsafe")
        wire = server.request(stream=True)
        status, headers, body = _split_http(wire)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("content-type", ""))
        text = body.decode("utf-8", errors="strict")
        visible = re.findall(
            r'"delta":\s*\{[^{}]*"content":\s*"((?:\\.|[^"\\])*)"',
            text,
        )
        self.assertTrue(any(_strict_json(f'"{item}"').strip() for item in visible))
        self.assertNotIn("data: [DONE]", text)
        self.assertNotIn("response.completed", text)
        self.assertNotIn("message_stop", text)
        self.assertNotIn(REQUEST_METRICS_SCHEMA, text)

        record = self._assert_evidence(
            server, expected_result="unsafe", headers_sent=True
        )
        self.assertIsNone(record["restore_sequence"])
        self.assertIsNone(record["response_sequence"])
        _u64(record["evidence_sequence"], "evidence_sequence")
        before, after = record["compact_before"], record["compact_after"]
        self.assertIs(before["cache_unsafe"], False)
        self.assertIs(after["cache_unsafe"], True)
        self.assertEqual(
            _u64(after["request_barrier_unsafe_failures"], "barrier_failures")
            - _u64(before["request_barrier_unsafe_failures"], "barrier_failures"),
            1,
        )
        self.assertEqual(_u64(after["cache_slot_loading_count"], "loading"), 0)
        self.assertEqual(_u64(after["cache_slot_in_use_count"], "in_use"), 0)
        self.assertEqual(_u64(after["cache_slot_total_refs"], "refs"), 0)
        server.wait(1)


if __name__ == "__main__":
    unittest.main(argv=UNITTEST_ARGV)
