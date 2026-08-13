#!/usr/bin/env python3
"""RED publication contract for DS4 engine and HTTP runtime snapshots."""

from __future__ import annotations

import importlib.util
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DS4_HEADER = (ROOT / "ds4.h").read_text(encoding="utf-8")
SERVER_SOURCE = (ROOT / "ds4_server.c").read_text(encoding="utf-8")
SCHEMA_PROFILE = (
    ROOT / "gguf-tools" / "quality-testing" / "compact_runtime_schema.py"
)
RUNTIME_SCHEMA = ROOT / "schemas/ds4-runtime-v1.schema.json"
LIVE_SERVER_URL = os.environ.get("DS4_RUNTIME_SERVER_URL", "").rstrip("/")


def _extract_live_option(argv: list[str]) -> tuple[str | None, list[str]]:
    """Remove the one launcher option before handing argv to unittest."""

    result = list(argv)
    positions = [index for index, value in enumerate(result) if value == "--live"]
    if not positions:
        return None, result
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise SystemExit("usage: test_runtime_endpoint_contract.py [unittest args] --live SERVER")
    position = positions[0]
    server = result[position + 1]
    if not server or server.startswith("--"):
        raise SystemExit("--live requires a ds4-server executable")
    del result[position : position + 2]
    return server, result


LIVE_SERVER_BINARY, UNITTEST_ARGV = _extract_live_option(sys.argv)


def _live_server_command(server: str, model: str, port: int) -> list[str]:
    return [
        server,
        "--model",
        model,
        "--cuda",
        "--ctx",
        "32768",
        "--prefill-chunk",
        "4096",
        "--session-slots",
        "1",
        "--ssd-streaming",
        "--ssd-streaming-cache-bytes",
        "8589934592",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _positive_timeout(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("DS4_RUNTIME_SERVER_START_TIMEOUT must be finite and positive")
    return value


def _require_default_lock_environment(environment: dict[str, str]) -> None:
    if "DS4_LOCK_FILE" in environment:
        raise ValueError(
            "live runtime endpoint qualification requires DS4_LOCK_FILE to be absent"
        )


def _frontend_version(server: str) -> dict[str, Any]:
    completed = subprocess.run(
        [server, "--version-json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(
            f"{server} --version-json failed with status {completed.returncode}:\n"
            f"{completed.stderr}"
        )
    payload = PROFILE.loads_strict(completed.stdout)
    if set(payload) != {"schema", "revision", "dirty", "backend", "features"}:
        raise AssertionError(f"{server} emitted a noncanonical version object")
    if payload["schema"] != "ds4.version/v1":
        raise AssertionError(f"{server} emitted an unexpected version schema")
    return payload


def _file_identity(path: os.PathLike[str] | str) -> dict[str, str]:
    info = os.stat(path)
    return {
        "device": str(info.st_dev),
        "inode": str(info.st_ino),
        "size_bytes": str(info.st_size),
        "mtime_ns": str(info.st_mtime_ns),
    }


def _wire_file_identity(payload: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(payload[key])
        for key in ("device", "inode", "size_bytes", "mtime_ns")
    }


def _opened_file_identity(pid: int, expected: dict[str, str]) -> dict[str, str]:
    descriptor_dir = Path(f"/proc/{pid}/fd")
    if not descriptor_dir.is_dir():
        raise AssertionError(f"live endpoint identity checks require {descriptor_dir}")
    for descriptor in descriptor_dir.iterdir():
        try:
            identity = _file_identity(descriptor)
        except (FileNotFoundError, PermissionError):
            continue
        if identity == expected:
            return identity
    raise AssertionError(
        f"server process {pid} has no open descriptor with model identity {expected}"
    )


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status != 200:
            raise AssertionError(f"GET {url} returned HTTP {response.status}")
        if response.headers.get_content_type() != "application/json":
            raise AssertionError(f"GET {url} did not return application/json")
        return PROFILE.loads_strict(response.read().decode("utf-8"))


def _load_schema_profile() -> Any:
    spec = importlib.util.spec_from_file_location(
        "compact_runtime_schema", SCHEMA_PROFILE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCHEMA_PROFILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = _load_schema_profile()
RUNTIME_VALIDATOR = PROFILE.validator_for(
    PROFILE.loads_strict(RUNTIME_SCHEMA.read_text(encoding="utf-8"))
)


def _c_function_body(source: str, signature: str) -> str:
    """Extract one C body while ignoring braces in strings and comments."""

    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing C function {signature}")
    brace = source.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing body for C function {signature}")

    depth = 0
    state = "code"
    escaped = False
    index = brace
    while index < len(source):
        byte = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state in {"string", "char"}:
            if escaped:
                escaped = False
            elif byte == "\\":
                escaped = True
            elif (state == "string" and byte == '"') or (
                state == "char" and byte == "'"
            ):
                state = "code"
        elif state == "line_comment":
            if byte == "\n":
                state = "code"
        elif state == "block_comment":
            if byte == "*" and following == "/":
                state = "code"
                index += 1
        elif byte == '"':
            state = "string"
        elif byte == "'":
            state = "char"
        elif byte == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif byte == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif byte == "{":
            depth += 1
        elif byte == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
        index += 1
    raise AssertionError(f"unterminated C function {signature}")


def _assert_order(test: unittest.TestCase, text: str, *needles: str) -> None:
    positions = []
    for needle in needles:
        position = text.find(needle)
        test.assertGreaterEqual(position, 0, f"missing {needle!r}")
        positions.append(position)
    test.assertEqual(positions, sorted(positions), f"wrong order: {needles!r}")


class RuntimeEndpointSourceContractTest(unittest.TestCase):
    maxDiff = None

    def test_engine_exposes_one_public_runtime_snapshot_api(self) -> None:
        harness = (
            '#include "ds4.h"\n'
            "static bool (*const snapshot_api)(\n"
            "    ds4_engine *, ds4_runtime_wire_snapshot *) =\n"
            "    ds4_engine_runtime_snapshot;\n"
            "int main(void) { return snapshot_api == 0; }\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "runtime_snapshot_api.c"
            output = Path(tmp) / "runtime_snapshot_api.o"
            source.write_text(harness, encoding="utf-8")
            completed = subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-std=c99",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT),
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.returncode,
            0,
            "ds4.h must declare bool ds4_engine_runtime_snapshot("
            "ds4_engine *, ds4_runtime_wire_snapshot *):\n"
            + completed.stderr,
        )

    def test_runtime_capture_helper_owns_the_inference_quiescence_boundary(self) -> None:
        body = _c_function_body(
            SERVER_SOURCE,
            "static bool server_runtime_snapshot(",
        )
        _assert_order(
            self,
            body,
            "pthread_mutex_lock(&s->inference_mu)",
            "ds4_engine_runtime_snapshot(s->engine",
            "pthread_mutex_unlock(&s->inference_mu)",
        )
        self.assertEqual(
            body.count("ds4_engine_runtime_snapshot("),
            1,
            "one HTTP observation must advance snapshot_seq exactly once",
        )

    def test_get_runtime_uses_one_captured_value_and_wire_serializer(self) -> None:
        handler = _c_function_body(SERVER_SOURCE, "static void *client_main(")
        self.assertIn('!strcmp(hr.path, "/v1/runtime")', handler)
        self.assertIn("send_runtime(s, fd)", handler)

        body = _c_function_body(SERVER_SOURCE, "static bool send_runtime(")
        _assert_order(
            self,
            body,
            "server_runtime_snapshot(s, &snapshot)",
            "ds4_runtime_wire_snapshot_json(&snapshot",
            'http_response(fd, s->enable_cors, 200, "application/json"',
        )
        self.assertEqual(body.count("server_runtime_snapshot("), 1)
        self.assertEqual(body.count("ds4_runtime_wire_snapshot_json("), 1)

    def test_models_reuse_one_runtime_snapshot_for_canonical_laguna_identity(self) -> None:
        body = _c_function_body(SERVER_SOURCE, "static bool send_models(")
        self.assertEqual(
            body.count("server_runtime_snapshot("),
            1,
            "/v1/models must take one coherent identity snapshot",
        )
        self.assertIn("&snapshot", body)
        self.assertIn('"laguna-s-2.1"', body)
        self.assertIn("s->runtime_snapshot_required", body)
        self.assertIn("!have_snapshot", body)
        detail = _c_function_body(SERVER_SOURCE, "static bool send_model(")
        self.assertIn("s->runtime_snapshot_required", detail)
        self.assertIn("!have_snapshot", detail)

        rendering = "\n".join(
            (
                _c_function_body(
                    SERVER_SOURCE, "static void append_model_json_values("
                ),
                _c_function_body(SERVER_SOURCE, "static void append_model_json("),
                body,
            )
        )
        for json_field in (
            "family",
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
        ):
            with self.subTest(field=json_field):
                self.assertIn(f'\\"{json_field}\\"', rendering)
        for snapshot_field in (
            "model_id",
            "model_family",
            "model.device",
            "model.inode",
            "model.size_bytes",
            "model.mtime_ns",
        ):
            with self.subTest(snapshot_field=snapshot_field):
                self.assertIn(snapshot_field, rendering)


class RuntimeEndpointLauncherContractTest(unittest.TestCase):
    def test_live_option_is_removed_before_unittest_parses_argv(self) -> None:
        server, unittest_argv = _extract_live_option(
            ["test_runtime_endpoint_contract.py", "-v", "--live", "./ds4-server"]
        )
        self.assertEqual(server, "./ds4-server")
        self.assertEqual(unittest_argv, ["test_runtime_endpoint_contract.py", "-v"])

    def test_launcher_uses_exact_compact_cuda_profile(self) -> None:
        self.assertEqual(
            _live_server_command("./ds4-server", "/models/laguna.gguf", 49152),
            [
                "./ds4-server",
                "--model",
                "/models/laguna.gguf",
                "--cuda",
                "--ctx",
                "32768",
                "--prefill-chunk",
                "4096",
                "--session-slots",
                "1",
                "--ssd-streaming",
                "--ssd-streaming-cache-bytes",
                "8589934592",
                "--host",
                "127.0.0.1",
                "--port",
                "49152",
            ],
        )

    def test_start_timeout_must_be_finite_and_positive(self) -> None:
        self.assertEqual(_positive_timeout("1.25"), 1.25)
        for invalid in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _positive_timeout(invalid)

    def test_live_launcher_rejects_lock_override_by_presence(self) -> None:
        _require_default_lock_environment({})
        for value in ("", "/tmp/alternate-ds4.lock"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "DS4_LOCK_FILE"):
                    _require_default_lock_environment({"DS4_LOCK_FILE": value})


@unittest.skipUnless(
    LIVE_SERVER_URL or LIVE_SERVER_BINARY,
    "set DS4_RUNTIME_SERVER_URL or pass --live SERVER for a Laguna endpoint",
)
class RuntimeEndpointLiveTest(unittest.TestCase):
    server_url = LIVE_SERVER_URL
    server_process: subprocess.Popen[bytes] | None = None
    server_log: Any = None
    model_path: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if not LIVE_SERVER_BINARY:
            return

        _require_default_lock_environment(dict(os.environ))
        model = os.environ.get("DS4_TEST_MODEL", "")
        if not model:
            raise AssertionError("--live requires DS4_TEST_MODEL to name the Laguna GGUF")
        cls.model_path = os.path.abspath(model)
        if not os.path.isfile(cls.model_path):
            raise AssertionError(f"DS4_TEST_MODEL is not a regular file: {cls.model_path}")

        port = _loopback_port()
        cls.server_url = f"http://127.0.0.1:{port}"
        cls.server_log = tempfile.TemporaryFile(mode="w+b")
        try:
            cls.server_process = subprocess.Popen(
                _live_server_command(LIVE_SERVER_BINARY, cls.model_path, port),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=cls.server_log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            timeout_text = os.environ.get("DS4_RUNTIME_SERVER_START_TIMEOUT", "900")
            timeout_seconds = _positive_timeout(timeout_text)
            deadline = time.monotonic() + timeout_seconds
            last_error: BaseException | None = None
            while time.monotonic() < deadline:
                if cls.server_process.poll() is not None:
                    raise RuntimeError(
                        f"ds4-server exited with status {cls.server_process.returncode}"
                    )
                try:
                    snapshot = _get_json(cls.server_url + "/v1/runtime")
                    RUNTIME_VALIDATOR.validate(snapshot)
                    if snapshot["state"] == "ready":
                        return
                    if snapshot["state"] == "unsafe":
                        raise RuntimeError("ds4-server reported an unsafe runtime")
                except (OSError, ValueError, AssertionError) as error:
                    last_error = error
                time.sleep(0.25)
            raise TimeoutError(
                f"ds4-server did not publish a ready runtime within {timeout_seconds}s"
            ) from last_error
        except BaseException as error:
            cls._stop_server()
            diagnostic = cls._server_diagnostic()
            cls._close_server_log()
            raise AssertionError(
                f"unable to launch compact CUDA ds4-server: {error}\n{diagnostic}"
            ) from error

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stop_server()
        cls._close_server_log()

    @classmethod
    def _stop_server(cls) -> None:
        process = cls.server_process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)

    @classmethod
    def _server_diagnostic(cls) -> str:
        if cls.server_log is None:
            return ""
        cls.server_log.flush()
        cls.server_log.seek(0)
        return cls.server_log.read().decode("utf-8", errors="replace")[-8000:]

    @classmethod
    def _close_server_log(cls) -> None:
        if cls.server_log is not None:
            cls.server_log.close()
            cls.server_log = None

    def _get(self, path: str) -> Any:
        return _get_json(self.server_url + path)

    def test_repeated_runtime_reads_advance_and_models_share_opened_identity(self) -> None:
        first = self._get("/v1/runtime")
        models = self._get("/v1/models")
        second = self._get("/v1/runtime")
        RUNTIME_VALIDATOR.validate(first)
        RUNTIME_VALIDATOR.validate(second)

        self.assertEqual(first["instance_id"], second["instance_id"])
        self.assertGreater(int(second["snapshot_seq"]), int(first["snapshot_seq"]))
        for stable in ("build", "executable", "model"):
            with self.subTest(stable=stable):
                self.assertEqual(first[stable], second[stable])

        if self.server_process is not None:
            self.assertEqual(first["state"], "ready")
            version = _frontend_version(LIVE_SERVER_BINARY or "")
            expected_build = {
                key: version[key]
                for key in ("revision", "dirty", "backend", "features")
            }
            self.assertEqual(first["build"], expected_build)
            self.assertFalse(expected_build["dirty"])
            self.assertEqual(expected_build["backend"], "cuda")
            self.assertTrue(
                {"laguna", "ssd_streaming"}.issubset(expected_build["features"])
            )
            self.assertEqual(
                first["config"],
                {
                    "context_tokens": 32768,
                    "prefill_chunk_tokens": 4096,
                    "session_slots": 1,
                    "ssd_streaming": True,
                    "ssd_streaming_cache_bytes": "8589934592",
                },
            )
            self.assertEqual(first["violations"], [])
            self.assertEqual(
                _wire_file_identity(first["executable"]),
                _file_identity(f"/proc/{self.server_process.pid}/exe"),
            )
            self.assertIsNotNone(self.model_path)
            model_identity = _file_identity(self.model_path or "")
            self.assertEqual(
                _wire_file_identity(first["model"]),
                model_identity,
            )
            self.assertEqual(
                _opened_file_identity(self.server_process.pid, model_identity),
                model_identity,
            )

        canonical = [
            entry
            for entry in models.get("data", [])
            if entry.get("id") == first["model"]["id"]
        ]
        self.assertEqual(
            len(canonical),
            1,
            "the canonical runtime model ID must occur once in /v1/models",
        )
        for field in ("family", "device", "inode", "size_bytes", "mtime_ns"):
            with self.subTest(field=field):
                self.assertEqual(canonical[0].get(field), first["model"][field])


if __name__ == "__main__":
    sys.argv[:] = UNITTEST_ARGV
    unittest.main()
