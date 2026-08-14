#!/usr/bin/env python3
"""Task 18 production CLI and foreground-process contract."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


def _listener_pids(port: int) -> set[int]:
    """Return the processes owning a TCP listener, if lsof is available."""

    lsof = Path("/usr/sbin/lsof")
    if not lsof.exists():
        lsof = Path("/usr/bin/lsof")
    if not lsof.exists():
        raise unittest.SkipTest("lsof is required for listener ownership")
    result = subprocess.run(
        [
            str(lsof),
            "-nP",
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
            "-Fp",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"lsof failed: {result.stderr!r}")
    return {
        int(line[1:])
        for line in result.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }


def _child_pids(parent_pid: int) -> set[int]:
    """Read process parentage through lsof without relying on ps/pgrep."""

    lsof = Path("/usr/sbin/lsof")
    if not lsof.exists():
        lsof = Path("/usr/bin/lsof")
    if not lsof.exists():
        raise unittest.SkipTest("lsof is required for process ownership")
    result = subprocess.run(
        [str(lsof), "-nP", "-F", "pR"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"lsof process scan failed: {result.stderr!r}")

    children: set[int] = set()
    current_pid: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current_pid = int(line[1:])
        elif (
            current_pid is not None
            and line.startswith("R")
            and line[1:].isdigit()
            and int(line[1:]) == parent_pid
        ):
            children.add(current_pid)
    return children


class ModelFreeServer:
    def __init__(self, binary: str):
        argv = [binary, "--server-lifecycle-stdio", "--scenario", "idle"]
        self.proc = subprocess.Popen(
            argv,
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
            self._close_pipes()
            raise AssertionError(
                f"model-free server did not listen: rc={self.proc.returncode} "
                f"stderr={stderr!r}"
            )
        self.ready = json.loads(ready_line)
        self.port = int(self.ready["port"])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


class Task18CliContract(unittest.TestCase):
    live_server = "./ds4-server"
    test_server = "./ds4_test"

    def run_live(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.live_server, *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_live_cli_does_not_inherit_an_occupied_ambient_lock(self) -> None:
        missing_model = "/definitely/missing/task18-model.gguf"
        with tempfile.NamedTemporaryFile() as ambient_lock:
            fcntl.flock(ambient_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.dict(
                os.environ, {"DS4_LOCK_FILE": ambient_lock.name}
            ):
                result = self.run_live("--model", missing_model, "--cpu")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("cannot open model", result.stderr)
        self.assertNotIn("another ds4 process", result.stderr)

    def test_invalid_invocation_exits_two_before_model_open(self) -> None:
        missing_model = "/definitely/missing/task18-model.gguf"
        cases = (
            (["--task18-unknown-option", "--model", missing_model], "unknown option"),
            (["--port", "0", "--model", missing_model], "invalid value for --port"),
            (["--port", "65536", "--model", missing_model], "invalid value for --port"),
        )
        for argv, error in cases:
            with self.subTest(argv=argv):
                result = self.run_live(*argv, "--cpu")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(error, result.stderr)
                self.assertNotIn("cannot open model", result.stderr)

    def test_model_and_pre_engine_startup_failures_exit_one(self) -> None:
        missing_model = "/definitely/missing/task18-model.gguf"
        missing = self.run_live("--model", missing_model, "--cpu")
        self.assertEqual(missing.returncode, 1, missing.stderr)
        self.assertIn("cannot open model", missing.stderr)

        with tempfile.NamedTemporaryFile() as malformed_model:
            malformed_model.write(b"not a gguf model")
            malformed_model.flush()
            malformed = self.run_live(
                "--model", malformed_model.name, "--cpu"
            )
        self.assertEqual(malformed.returncode, 1, malformed.stderr)

        missing_chdir = self.run_live(
            "--chdir",
            "/definitely/missing/task18-startup-directory",
            "--model",
            missing_model,
            "--cpu",
        )
        self.assertEqual(missing_chdir.returncode, 1, missing_chdir.stderr)
        self.assertIn("failed to chdir", missing_chdir.stderr)
        self.assertNotIn("cannot open model", missing_chdir.stderr)

    def test_occupied_requested_port_exits_one_without_alternate_bind(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            requested_port = int(occupied.getsockname()[1])
            result = subprocess.run(
                [
                    self.test_server,
                    "--server-lifecycle-stdio",
                    "--scenario",
                    "idle",
                    "--port",
                    str(requested_port),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                1,
                "missing model-free requested-port startup seam: "
                f"stderr={result.stderr!r}",
            )
            self.assertEqual(result.stdout, "")
            self.assertEqual(_listener_pids(requested_port), {os.getpid()})

    def test_foreground_child_owns_listener_and_does_not_signal_peer(self) -> None:
        server = ModelFreeServer(self.test_server)
        self.addCleanup(server.close)
        self.assertEqual(_listener_pids(server.port), {server.proc.pid})
        self.assertEqual(_child_pids(server.proc.pid), set())

        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "peer-signaled"
            peer_program = (
                "import pathlib,signal,time,sys; "
                "p=pathlib.Path(sys.argv[1]); "
                "h=lambda s,f:p.write_text(str(s)); "
                "signal.signal(signal.SIGINT,h); "
                "signal.signal(signal.SIGTERM,h); "
                "print('ready',flush=True); "
                "time.sleep(30)"
            )
            peer = subprocess.Popen(
                [sys.executable, "-c", peer_program, str(sentinel)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert peer.stdout is not None
                self.assertEqual(peer.stdout.readline().strip(), "ready")
                os.kill(server.proc.pid, signal.SIGTERM)
                self.assertEqual(server.proc.wait(timeout=5), 0)
                time.sleep(0.05)
                self.assertIsNone(peer.poll())
                self.assertFalse(sentinel.exists())
                self.assertEqual(_listener_pids(server.port), set())
                self.assertEqual(_child_pids(server.proc.pid), set())
            finally:
                if peer.poll() is None:
                    peer.kill()
                    peer.wait(timeout=2)
                for pipe in (peer.stdout, peer.stderr):
                    if pipe is not None and not pipe.closed:
                        pipe.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-server", default="./ds4-server")
    parser.add_argument("--test-server", default="./ds4_test")
    args, remaining = parser.parse_known_args()
    Task18CliContract.live_server = args.live_server
    Task18CliContract.test_server = args.test_server
    unittest.main(argv=[__file__, *remaining])


if __name__ == "__main__":
    main()
