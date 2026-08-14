#!/usr/bin/env python3
"""Host-only RED contract for Task 18's HTTP failure boundary.

The existing ``ds4_test`` server driver is expected to grow one operation:

.. code-block:: json

    {
      "op": "failure_boundary",
      "body": {
        "typed_outcome": "recoverable",
        "headers_sent": false,
        "error_text": "...",
        "fixture": {"slot_state": "loading", "pin_count": 2, "...": "..."},
        "fixture_fingerprint": "<sha256 of canonical fixture JSON>"
      }
    }

The operation must drive the production HTTP-boundary classifier over a real
mutable cache fixture.  It reports the observed fixture before and after the
boundary, together with monotonic transition sequence numbers.  The test
recomputes both fingerprints; an adapter cannot satisfy the contract with an
unrelated, inert snapshot.

Unsafe operations flush their control response and then terminate the driver
with status 1.  Recoverable operations leave the same driver accepting RPCs.
This makes process outcome an observed fact rather than a self-reported field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_TOKENS = 2048
ADVERSARIAL_ERROR = (
    "unsafe invariant accounting failed; temporary compact read may be retried"
)
BOUNDARY_KEYS = {
    "typed_outcome",
    "headers_sent",
    "http_status",
    "abrupt",
    "terminal_metrics_emitted",
    "process_accepting",
    "fallback_calls",
    "restore_seq",
    "response_seq",
    "fixture_before",
    "fixture_before_fingerprint",
    "fixture_after",
    "fixture_after_fingerprint",
}
FIXTURE_KEYS = {
    "fixture_id",
    "slot_state",
    "generation",
    "pin_count",
    "capacity_reserved_bytes",
    "published",
    "key",
}


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class Driver:
    def __init__(self, executable: Path) -> None:
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
            if self.rpc({"op": "ping"}) != {"ok": True}:
                raise AssertionError("unexpected server-driver handshake")
        except BaseException:
            self.cleanup()
            raise

    def rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise AssertionError("server-driver pipes are unavailable")
        if self.proc.poll() is not None:
            raise AssertionError(
                f"server driver already exited with status {self.proc.returncode}"
            )
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            self.proc.wait(timeout=5)
            raise AssertionError(
                "server driver exited before returning an RPC response: "
                f"status={self.proc.returncode}, stderr={stderr!r}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"server driver returned non-JSON output: {line!r}"
            ) from exc
        if not isinstance(response, dict):
            raise AssertionError(f"server driver response is not an object: {response!r}")
        return response

    def wait_for_exit(self) -> int:
        return self.proc.wait(timeout=5)

    def close_cleanly(self) -> None:
        if self.proc.poll() is None:
            response = self.rpc({"op": "quit"})
            if response != {"ok": True}:
                raise AssertionError(f"unexpected driver shutdown response: {response!r}")
        status = self.proc.wait(timeout=5)
        if status != 0:
            raise AssertionError(f"server driver clean shutdown returned {status}")

    def cleanup(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None and self.proc.stdout is not None:
                    self.proc.stdin.write('{"op":"quit"}\n')
                    self.proc.stdin.flush()
                    self.proc.stdout.readline()
            except (BrokenPipeError, OSError):
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None:
                pipe.close()


class FailureBoundaryContract(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = Driver(SERVER)
        self.addCleanup(self.driver.cleanup)

    def fixture(self) -> dict[str, Any]:
        return {
            "fixture_id": str(uuid.uuid4()),
            "slot_state": "loading",
            "generation": self.driver.proc.pid,
            "pin_count": 2,
            "capacity_reserved_bytes": "8192",
            "published": False,
            "key": {"layer_id": 7, "expert_id": 13},
        }

    def failure_boundary(
        self, typed_outcome: str, *, headers_sent: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        fixture = self.fixture()
        response = self.driver.rpc(
            {
                "op": "failure_boundary",
                "body": {
                    "typed_outcome": typed_outcome,
                    "headers_sent": headers_sent,
                    "error_text": ADVERSARIAL_ERROR,
                    "fixture": fixture,
                    "fixture_fingerprint": _fingerprint(fixture),
                },
            }
        )
        self.assertEqual(
            set(response),
            BOUNDARY_KEYS,
            "missing production failure_boundary RPC contract",
        )
        self.assertEqual(response["typed_outcome"], typed_outcome)
        self.assertIs(response["headers_sent"], headers_sent)
        self.assertEqual(response["fallback_calls"], 0)

        before = response["fixture_before"]
        after = response["fixture_after"]
        self.assertIsInstance(before, dict)
        self.assertIsInstance(after, dict)
        self.assertEqual(set(before), FIXTURE_KEYS)
        self.assertEqual(set(after), FIXTURE_KEYS)
        self.assertEqual(before, fixture)
        self.assertEqual(response["fixture_before_fingerprint"], _fingerprint(before))
        self.assertEqual(response["fixture_after_fingerprint"], _fingerprint(after))
        self.assertNotEqual(
            response["fixture_after_fingerprint"],
            response["fixture_before_fingerprint"],
            "failure fixture was not mutated through a real transition",
        )
        return response, fixture

    def test_recoverable_restores_before_503_and_remains_accepting(self) -> None:
        response, fixture = self.failure_boundary(
            "recoverable", headers_sent=False
        )
        self.assertEqual(response["http_status"], 503)
        self.assertIs(response["abrupt"], False)
        self.assertIs(response["terminal_metrics_emitted"], True)
        self.assertIs(response["process_accepting"], True)
        self.assertIsInstance(response["restore_seq"], int)
        self.assertIsInstance(response["response_seq"], int)
        self.assertGreater(response["restore_seq"], 0)
        self.assertGreater(response["response_seq"], response["restore_seq"])

        restored = response["fixture_after"]
        self.assertEqual(restored["fixture_id"], fixture["fixture_id"])
        self.assertEqual(restored["slot_state"], "empty")
        self.assertGreaterEqual(restored["generation"], fixture["generation"])
        self.assertEqual(restored["pin_count"], 0)
        self.assertEqual(restored["capacity_reserved_bytes"], "0")
        self.assertIs(restored["published"], False)
        self.assertIsNone(restored["key"])

        self.assertEqual(
            self.driver.rpc({"op": "ping"}),
            {"ok": True},
            "recoverable 503 stopped the process from accepting work",
        )
        self.driver.close_cleanly()

    def test_unsafe_before_headers_returns_500_then_exits_one(self) -> None:
        response, _fixture = self.failure_boundary("unsafe", headers_sent=False)
        self.assertEqual(response["http_status"], 500)
        self.assertIs(response["abrupt"], False)
        self.assertIs(response["terminal_metrics_emitted"], True)
        self.assertIs(response["process_accepting"], False)
        self.assertIsInstance(response["response_seq"], int)
        self.assertGreater(response["response_seq"], 0)
        self.assertEqual(self.driver.wait_for_exit(), 1)

    def test_unsafe_after_headers_abrupts_without_terminal_then_exits_one(self) -> None:
        response, _fixture = self.failure_boundary("unsafe", headers_sent=True)
        self.assertIsNone(response["http_status"])
        self.assertIs(response["abrupt"], True)
        self.assertIs(response["terminal_metrics_emitted"], False)
        self.assertIs(response["process_accepting"], False)
        self.assertIsNone(response["response_seq"])
        self.assertEqual(self.driver.wait_for_exit(), 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "ds4_test",
        help="host ds4_test binary exposing the server stdio driver",
    )
    args, remaining = parser.parse_known_args()
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
