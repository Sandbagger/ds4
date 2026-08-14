#!/usr/bin/env python3
"""Task 18 real-process server lifecycle contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import unittest


class LifecycleChild:
    def __init__(self, server: str, scenario: str):
        self.proc = subprocess.Popen(
            [server, "--server-lifecycle-stdio", "--scenario", scenario],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            self.proc.wait(timeout=5)
            raise AssertionError(
                f"lifecycle child did not become ready: rc={self.proc.returncode} "
                f"stderr={stderr!r}"
            )
        try:
            self.ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            self.proc.kill()
            self.proc.wait(timeout=2)
            self._close_pipes()
            raise AssertionError(
                "lifecycle child mode did not produce its production-ready "
                f"record: stdout={ready_line!r} stderr={stderr!r}"
            ) from exc
        if self.ready.get("state") != "accepting":
            raise AssertionError(f"invalid lifecycle ready record: {self.ready!r}")
        self.port = int(self.ready["port"])

    def connect(self) -> socket.socket:
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=2)
        conn.settimeout(3)
        return conn

    def command(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def signal(self, sig: int) -> None:
        os.kill(self.proc.pid, sig)

    def wait(self, expected: int, timeout: float = 5.0) -> str:
        try:
            actual = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise AssertionError(f"lifecycle child hung; stderr={stderr!r}")
        stderr = self.proc.stderr.read() if self.proc.stderr else ""
        if actual != expected:
            raise AssertionError(
                f"lifecycle child rc={actual}, expected={expected}; "
                f"stderr={stderr!r}"
            )
        self._close_pipes()
        return stderr

    def _close_pipes(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self._close_pipes()


class Task18LifecycleContract(unittest.TestCase):
    server = "./ds4_test"

    def child(self, scenario: str) -> LifecycleChild:
        child = LifecycleChild(self.server, scenario)
        self.addCleanup(child.close)
        return child

    def test_idle_term_and_sigint_have_exact_exit_statuses(self) -> None:
        term = self.child("idle")
        term.signal(signal.SIGTERM)
        term.wait(0)

        interrupt = self.child("idle")
        interrupt.signal(signal.SIGINT)
        interrupt.wait(130)

    def test_term_rejects_a_preaccepted_request_and_safe_drain_exits_zero(self) -> None:
        child = self.child("active-safe")
        conn = child.connect()
        self.addCleanup(conn.close)
        time.sleep(0.05)  # let the production accept loop own this connection
        child.signal(signal.SIGTERM)
        conn.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Length: 2\r\n\r\n{}"
        )
        response = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response += chunk
        self.assertTrue(response.startswith(b"HTTP/1.1 503 "), response)
        self.assertIn(b'"error"', response)
        child.wait(0)

    def test_term_before_a_safe_terminal_exits_143(self) -> None:
        child = self.child("active-incomplete")
        child.signal(signal.SIGTERM)
        child.wait(143)

    def test_second_term_forces_143(self) -> None:
        child = self.child("forced")
        child.signal(signal.SIGTERM)
        time.sleep(0.1)
        self.assertIsNone(child.proc.poll())
        child.signal(signal.SIGTERM)
        child.wait(143)

    def test_second_signal_uses_the_actual_forcing_signal(self) -> None:
        term_then_int = self.child("forced")
        term_then_int.signal(signal.SIGTERM)
        time.sleep(0.1)
        term_then_int.signal(signal.SIGINT)
        term_then_int.wait(130)

        int_then_term = self.child("forced")
        int_then_term.signal(signal.SIGINT)
        time.sleep(0.1)
        int_then_term.signal(signal.SIGTERM)
        int_then_term.wait(143)

    def test_internal_unsafe_wakes_blocked_accept_and_exits_one(self) -> None:
        child = self.child("unsafe")
        child.command("unsafe")
        child.wait(1)

    def test_term_received_before_unsafe_completion_retains_143(self) -> None:
        child = self.child("unsafe")
        child.signal(signal.SIGTERM)
        time.sleep(0.05)
        child.command("unsafe")
        child.wait(143)

    def test_partial_pre_admission_client_cannot_hold_drain_open(self) -> None:
        child = self.child("idle")
        conn = child.connect()
        self.addCleanup(conn.close)
        conn.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n")
        time.sleep(0.05)
        child.signal(signal.SIGTERM)
        response = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response += chunk
        self.assertTrue(response.startswith(b"HTTP/1.1 503 "), response)
        child.wait(0)


class Task18LifecycleSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (Path(__file__).resolve().parents[1] / "ds4_server.c").read_text()

    def function_body(self, name: str, next_name: str) -> str:
        start = self.source.index(name)
        end = self.source.index(next_name, start)
        return self.source[start:end]

    def test_prefill_rechecks_shutdown_after_inference_lock(self) -> None:
        body = self.function_body(
            "static bool server_prefill_enter(",
            "static void server_prefill_leave(",
        )
        model_owned = body.index("s->model_busy = true")
        inference_lock = body.index(
            "pthread_mutex_lock(&s->inference_mu);", model_owned
        )
        return_true = body.index("return true;", inference_lock)
        handoff = body[inference_lock:return_true]
        self.assertIn("if (g_stop_requested)", handoff)
        self.assertIn("s->model_busy = false", handoff)

    def test_decode_rechecks_shutdown_after_model_wait(self) -> None:
        body = self.function_body(
            "static void *decode_worker_main(",
            "static uint64_t server_next_sequence(",
        )
        wait = body.index("while (s->model_busy")
        collect = body.index("int count = 0", wait)
        self.assertIn("if (g_stop_requested)", body[wait:collect])

    def test_decode_rechecks_shutdown_at_inference_dispatch(self) -> None:
        body = self.function_body(
            "static void *decode_worker_main(",
            "static uint64_t server_next_sequence(",
        )
        busy = body.index("s->model_busy = true")
        inference_lock = body.index("pthread_mutex_lock(&s->inference_mu);", busy)
        dispatch = body.index("ds4_sessions_eval_batch_attributed(", inference_lock)
        self.assertIn("g_stop_requested", body[inference_lock:dispatch])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="./ds4_test")
    args, remaining = parser.parse_known_args()
    Task18LifecycleContract.server = args.server
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
