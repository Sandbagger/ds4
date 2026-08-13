#!/usr/bin/env python3
"""Live Task 17 contract for a real compact-Laguna ``ds4-server``.

The model-backed suite is opt-in because it acquires DS4's canonical process
lock and allocates the exact two-session CUDA profile. Run it only inside the
guarded DGX maintenance window, after production has stopped and the model's
retained descriptor has been cold-prepared::

    env -u DS4_LOCK_FILE \
      DS4_TEST_MODEL=/absolute/path/to/laguna-s-2.1-Q4_K_M.gguf \
      DS4_TEST_MODEL_FD=9 \
      DS4_LAGUNA_SERVER_START_TIMEOUT=900 \
      uv run --offline --with-requirements \
        gguf-tools/quality-testing/requirements-compact-runtime.txt \
        python tests/test_laguna_server_live_contract.py -v \
          --live ./ds4-server

Without ``--live``, only fast launcher and wire-parser contracts run. The live
suite starts the supplied executable itself; accepting an existing URL would
lose executable/model identity and clean-shutdown evidence.
"""

from __future__ import annotations

import importlib.util
import io
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
import urllib.error
import urllib.request
import uuid
from email.message import Message
from pathlib import Path
from typing import Any, Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PROFILE = (
    ROOT / "gguf-tools" / "quality-testing" / "compact_runtime_schema.py"
)
RUNTIME_SCHEMA = ROOT / "schemas" / "ds4-runtime-v1.schema.json"
REQUEST_SCHEMA = ROOT / "schemas" / "ds4-runtime-request-v1.schema.json"
ADMISSION_SCHEMA = ROOT / "schemas" / "ds4-token-admission-v1.schema.json"

CONTEXT_TOKENS = 32768
PREFILL_TOKENS = 4096
SESSION_SLOTS = 2
CACHE_BYTES = 8 * 1024 * 1024 * 1024
MAX_OUTPUT_TOKENS = 8
MODEL_ALIAS = "laguna-s-2.1-chat"
MODEL_ID = "laguna-s-2.1"
TEMPLATE_REVISION = "poolside-laguna-s-2.1-native-nothink-v1"
REQUEST_METRICS_SCHEMA = "ds4.runtime.request/v1"


def _extract_live_option(argv: list[str]) -> tuple[str | None, list[str]]:
    """Remove the single harness option before unittest parses argv."""

    result = list(argv)
    positions = [index for index, value in enumerate(result) if value == "--live"]
    if not positions:
        return None, result
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise SystemExit(
            "usage: test_laguna_server_live_contract.py "
            "[unittest args] --live SERVER"
        )
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
        str(CONTEXT_TOKENS),
        "--prefill-chunk",
        str(PREFILL_TOKENS),
        "--session-slots",
        str(SESSION_SLOTS),
        "--ssd-streaming",
        "--ssd-streaming-cache-bytes",
        str(CACHE_BYTES),
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
        raise ValueError(
            "DS4_LAGUNA_SERVER_START_TIMEOUT must be finite and positive"
        )
    return value


def _strict_json_stdlib(text: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )


def _load_schema_profile() -> Any:
    spec = importlib.util.spec_from_file_location(
        "compact_runtime_schema", SCHEMA_PROFILE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCHEMA_PROFILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_identity(path: os.PathLike[str] | str) -> dict[str, str]:
    status = os.stat(path)
    return _stat_identity(status)


def _stat_identity(status: os.stat_result) -> dict[str, str]:
    return {
        "device": str(status.st_dev),
        "inode": str(status.st_ino),
        "size_bytes": str(status.st_size),
        "mtime_ns": str(status.st_mtime_ns),
    }


def _wire_file_identity(payload: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(payload[key])
        for key in ("device", "inode", "size_bytes", "mtime_ns")
    }


def _opened_file_identity(pid: int, expected: dict[str, str]) -> dict[str, str]:
    descriptors = Path(f"/proc/{pid}/fd")
    if not descriptors.is_dir():
        raise AssertionError(
            f"live identity checks require Linux descriptor directory {descriptors}"
        )
    for descriptor in descriptors.iterdir():
        try:
            identity = _file_identity(descriptor)
        except (FileNotFoundError, PermissionError):
            continue
        if identity == expected:
            return identity
    raise AssertionError(
        f"server process {pid} has no open model descriptor {expected}"
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
            f"{server} --version-json failed with status "
            f"{completed.returncode}:\n{completed.stderr}"
        )
    payload = _strict_json_stdlib(completed.stdout)
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "revision", "dirty", "backend", "features"
    }:
        raise AssertionError("candidate emitted a noncanonical version object")
    revision = payload["revision"]
    features = payload["features"]
    if (
        payload["schema"] != "ds4.version/v1"
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or revision == "0" * 40
        or payload["dirty"] is not False
        or payload["backend"] != "cuda"
        or not isinstance(features, list)
        or features != sorted(set(features))
        or not {"laguna", "ssd_streaming"}.issubset(features)
    ):
        raise AssertionError(f"candidate build is not admissible: {payload!r}")
    return payload


def _require_live_inputs(
    server: str, environment: dict[str, str]
) -> tuple[Path, Path]:
    if "DS4_LOCK_FILE" in environment:
        raise ValueError(
            "live Laguna server qualification requires DS4_LOCK_FILE to be absent"
        )
    model_text = environment.get("DS4_TEST_MODEL", "")
    if not model_text:
        raise ValueError("--live requires DS4_TEST_MODEL")
    model = Path(model_text)
    if not model.is_absolute():
        raise ValueError("DS4_TEST_MODEL must be an absolute path")
    if model.is_symlink() or not model.is_file():
        raise ValueError("DS4_TEST_MODEL must be a nonsymlink regular file")
    candidate = Path(server).resolve(strict=True)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("--live must name an executable regular file")
    return candidate, model


def _require_model_fd(
    model: Path, environment: dict[str, str]
) -> tuple[int, dict[str, str]]:
    text = environment.get("DS4_TEST_MODEL_FD", "")
    if not text:
        raise ValueError("--live requires DS4_TEST_MODEL_FD")
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", text) is None:
        raise ValueError(
            "DS4_TEST_MODEL_FD must be a canonical integer at least 3"
        )
    descriptor = int(text)
    if descriptor < 3:
        raise ValueError(
            "DS4_TEST_MODEL_FD must be at least 3 so stdio redirection "
            "cannot replace it"
        )
    try:
        descriptor_identity = _stat_identity(os.fstat(descriptor))
    except OSError as error:
        raise ValueError(
            "DS4_TEST_MODEL_FD must name an open descriptor"
        ) from error
    model_identity = _file_identity(model)
    if descriptor_identity != model_identity:
        raise ValueError(
            "DS4_TEST_MODEL_FD identity does not equal DS4_TEST_MODEL: "
            f"fd={descriptor_identity!r}, path={model_identity!r}"
        )
    return descriptor, descriptor_identity


def _retained_model_path(descriptor: int) -> str:
    return f"/proc/self/fd/{descriptor}"


def _running_candidate_identity(
    candidate: Path, pid: int
) -> dict[str, str]:
    expected = _file_identity(candidate)
    observed = _file_identity(f"/proc/{pid}/exe")
    if observed != expected:
        raise AssertionError(
            "launched process executable differs from the candidate file: "
            f"candidate={expected!r}, process={observed!r}"
        )
    return expected


def _spawn_live_server(
    server: str,
    model_fd: int,
    port: int,
    server_log: Any,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        _live_server_command(server, _retained_model_path(model_fd), port),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        pass_fds=(model_fd,),
        start_new_session=True,
    )


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


def _parse_sse_wire(
    wire: str, loads: Callable[[str], Any] = _strict_json_stdlib
) -> list[tuple[str | None, Any]]:
    frames: list[tuple[str | None, Any]] = []
    for raw_frame in wire.replace("\r\n", "\n").split("\n\n"):
        if not raw_frame.strip():
            continue
        event: str | None = None
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
            elif line.startswith(":") or line.startswith("id:"):
                continue
            else:
                raise AssertionError(f"invalid SSE line: {line!r}")
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        frames.append((event, data if data == "[DONE]" else loads(data)))
    return frames


def _anthropic_prompt_tokens(usage: dict[str, Any]) -> int:
    return sum(
        int(usage.get(key, 0))
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def _runtime_purity_projection(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Strip only the read-generated snapshot sequence."""

    return {
        key: snapshot[key]
        for key in (
            "instance_id", "state", "build", "executable", "model",
            "config", "limits", "allocations", "counters", "violations",
        )
    }


class LauncherContract(unittest.TestCase):
    def test_live_command_freezes_the_two_session_profile(self) -> None:
        self.assertEqual(
            _live_server_command("/candidate/ds4-server", "/model.gguf", 18003),
            [
                "/candidate/ds4-server", "--model", "/model.gguf", "--cuda",
                "--ctx", "32768", "--prefill-chunk", "4096",
                "--session-slots", "2", "--ssd-streaming",
                "--ssd-streaming-cache-bytes", "8589934592",
                "--host", "127.0.0.1", "--port", "18003",
            ],
        )

    def test_live_option_is_removed_before_unittest(self) -> None:
        self.assertEqual(
            _extract_live_option(["test.py", "-v", "--live", "./ds4-server"]),
            ("./ds4-server", ["test.py", "-v"]),
        )
        with self.assertRaises(SystemExit):
            _extract_live_option(["test.py", "--live"])
        with self.assertRaises(SystemExit):
            _extract_live_option(["test.py", "--live", "a", "--live", "b"])

    def test_explicit_live_run_cannot_skip_model_or_override_lock(self) -> None:
        with self.assertRaisesRegex(ValueError, "DS4_TEST_MODEL"):
            _require_live_inputs("/missing", {})
        with self.assertRaisesRegex(ValueError, "DS4_LOCK_FILE"):
            _require_live_inputs(
                "/missing", {"DS4_LOCK_FILE": "", "DS4_TEST_MODEL": "/model"}
            )

    def test_retained_model_descriptor_is_canonical_open_and_identical(self) -> None:
        with tempfile.NamedTemporaryFile() as model:
            model_path = Path(model.name)
            descriptor = model.fileno()
            expected = _file_identity(model_path)
            self.assertEqual(
                _require_model_fd(
                    model_path, {"DS4_TEST_MODEL_FD": str(descriptor)}
                ),
                (descriptor, expected),
            )
            self.assertEqual(
                _retained_model_path(descriptor),
                f"/proc/self/fd/{descriptor}",
            )
            for value in ("", "0", "1", "2", "+3", "-1", "03", " 3"):
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "DS4_TEST_MODEL_FD"
                ):
                    _require_model_fd(
                        model_path, {"DS4_TEST_MODEL_FD": value}
                    )
            with tempfile.NamedTemporaryFile() as other:
                with self.assertRaisesRegex(ValueError, "identity"):
                    _require_model_fd(
                        model_path,
                        {"DS4_TEST_MODEL_FD": str(other.fileno())},
                    )

    def test_launcher_passes_retained_descriptor_and_opens_proc_fd(self) -> None:
        server_log = mock.sentinel.server_log
        with mock.patch.object(
            subprocess, "Popen", return_value=mock.sentinel.process
        ) as popen:
            process = _spawn_live_server(
                "/candidate/ds4-server", 9, 18003, server_log
            )
        self.assertIs(process, mock.sentinel.process)
        popen.assert_called_once_with(
            _live_server_command(
                "/candidate/ds4-server", "/proc/self/fd/9", 18003
            ),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            pass_fds=(9,),
            start_new_session=True,
        )

    def test_running_process_must_match_candidate_file_identity(self) -> None:
        identity = {
            "device": "1",
            "inode": "2",
            "size_bytes": "3",
            "mtime_ns": "4",
        }
        with mock.patch(
            f"{__name__}._file_identity", side_effect=[identity, identity]
        ):
            self.assertEqual(
                _running_candidate_identity(Path("/candidate"), 17),
                identity,
            )
        with mock.patch(
            f"{__name__}._file_identity",
            side_effect=[identity, {**identity, "inode": "5"}],
        ), self.assertRaisesRegex(AssertionError, "differs"):
            _running_candidate_identity(Path("/candidate"), 17)

    def test_timeout_rejects_zero_negative_and_nonfinite_values(self) -> None:
        self.assertEqual(_positive_timeout("900"), 900.0)
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _positive_timeout(value)

    def test_sse_parser_preserves_native_terminal_order(self) -> None:
        frames = _parse_sse_wire(
            "event: message_delta\n"
            'data: {"type":"message_delta","usage":{"output_tokens":2}}\n\n'
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n\n'
        )
        self.assertEqual(frames[0][0], "message_delta")
        self.assertEqual(frames[1], ("message_stop", {"type": "message_stop"}))

    def test_http_error_preserves_three_item_response_shape(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/json; charset=utf-8"
        error = urllib.error.HTTPError(
            "http://127.0.0.1:1/rejected",
            400,
            "Bad Request",
            headers,
            io.BytesIO(b'{"error":"rejected"}'),
        )
        prior_url = LiveLagunaServerContract.server_url
        LiveLagunaServerContract.server_url = "http://127.0.0.1:1"
        try:
            with mock.patch.object(
                urllib.request, "urlopen", side_effect=error
            ):
                response = LiveLagunaServerContract._http("GET", "/rejected")
        finally:
            LiveLagunaServerContract.server_url = prior_url
        self.assertEqual(
            response,
            (400, "application/json", '{"error":"rejected"}'),
        )


# LiveLagunaServerContract is appended below so the host-only tests above stay
# importable without jsonschema or a model.


@unittest.skipUnless(
    LIVE_SERVER_BINARY,
    "pass --live SERVER to run the model-backed Laguna HTTP contract",
)
class LiveLagunaServerContract(unittest.TestCase):
    server_url = ""
    server_process: subprocess.Popen[bytes] | None = None
    server_log: Any = None
    model_path: Path | None = None
    model_fd: int | None = None
    model_identity: dict[str, str] | None = None
    candidate_path: Path | None = None
    version: dict[str, Any] = {}
    loads: Callable[[str], Any] = _strict_json_stdlib
    runtime_validator: Any = None
    request_validator: Any = None
    admission_validator: Any = None

    def tearDown(self) -> None:
        outcome = getattr(self, "_outcome", None)
        result = getattr(outcome, "result", None)
        records = [] if result is None else [
            *getattr(result, "failures", []),
            *getattr(result, "errors", []),
        ]
        failed = any(
            case is self or getattr(case, "test_case", None) is self
            for case, _traceback in records
        )
        if failed:
            sys.stderr.write(
                "\n--- ds4-server diagnostic after live contract failure ---\n"
                + self._server_diagnostic()
                + "\n--- end ds4-server diagnostic ---\n"
            )

    @classmethod
    def setUpClass(cls) -> None:
        if LIVE_SERVER_BINARY is None:
            raise unittest.SkipTest("no candidate server was supplied")

        cls.candidate_path, cls.model_path = _require_live_inputs(
            LIVE_SERVER_BINARY, dict(os.environ)
        )
        cls.model_fd, cls.model_identity = _require_model_fd(
            cls.model_path, dict(os.environ)
        )
        cls.version = _frontend_version(str(cls.candidate_path))
        profile = _load_schema_profile()
        cls.loads = profile.loads_strict
        cls.runtime_validator = profile.validator_for(
            cls.loads(RUNTIME_SCHEMA.read_text(encoding="utf-8"))
        )
        cls.request_validator = profile.validator_for(
            cls.loads(REQUEST_SCHEMA.read_text(encoding="utf-8"))
        )
        cls.admission_validator = profile.validator_for(
            cls.loads(ADMISSION_SCHEMA.read_text(encoding="utf-8"))
        )

        port = _loopback_port()
        cls.server_url = f"http://127.0.0.1:{port}"
        cls.server_log = tempfile.TemporaryFile(mode="w+b")
        try:
            cls.server_process = _spawn_live_server(
                str(cls.candidate_path),
                cls.model_fd,
                port,
                cls.server_log,
            )
            start_timeout = _positive_timeout(
                os.environ.get("DS4_LAGUNA_SERVER_START_TIMEOUT", "900")
            )
            deadline = time.monotonic() + start_timeout
            last_error: BaseException | None = None
            while time.monotonic() < deadline:
                if cls.server_process.poll() is not None:
                    raise RuntimeError(
                        "ds4-server exited during cold start with status "
                        f"{cls.server_process.returncode}"
                    )
                try:
                    snapshot = cls._get_json("/v1/runtime")
                    cls.runtime_validator.validate(snapshot)
                    if snapshot["state"] == "ready":
                        if snapshot["violations"]:
                            raise RuntimeError(
                                "ready runtime published violations: "
                                f"{snapshot['violations']!r}"
                            )
                        cls._require_launch_identity(snapshot)
                        return
                    if snapshot["state"] == "unsafe":
                        raise RuntimeError("ds4-server published unsafe state")
                except (OSError, ValueError, AssertionError) as error:
                    last_error = error
                time.sleep(0.25)
            raise TimeoutError(
                "ds4-server did not publish a ready runtime within "
                f"{start_timeout:g}s"
            ) from last_error
        except BaseException as error:
            cls._terminate_server(require_clean=False)
            diagnostic = cls._server_diagnostic()
            cls._close_server_log()
            raise AssertionError(
                f"unable to launch the live Laguna candidate: {error}\n"
                f"--- ds4-server diagnostic ---\n{diagnostic}"
            ) from error

    @classmethod
    def tearDownClass(cls) -> None:
        stop_error: BaseException | None = None
        try:
            cls._terminate_server(require_clean=True)
        except BaseException as error:
            stop_error = error
        diagnostic = cls._server_diagnostic()
        cls._close_server_log()
        if stop_error is not None:
            raise AssertionError(
                f"live ds4-server did not shut down cleanly: {stop_error}\n"
                f"--- ds4-server diagnostic ---\n{diagnostic}"
            ) from stop_error

    @classmethod
    def _require_launch_identity(cls, snapshot: dict[str, Any]) -> None:
        process = cls.server_process
        candidate = cls.candidate_path
        model_identity = cls.model_identity
        if process is None or candidate is None or model_identity is None:
            raise RuntimeError("live launch identity inputs are unavailable")
        expected_build = {
            key: cls.version[key]
            for key in ("revision", "dirty", "backend", "features")
        }
        expected_config = {
            "context_tokens": CONTEXT_TOKENS,
            "prefill_chunk_tokens": PREFILL_TOKENS,
            "session_slots": SESSION_SLOTS,
            "ssd_streaming": True,
            "ssd_streaming_cache_bytes": str(CACHE_BYTES),
        }
        expected_limits = {
            "effective_context_tokens": CONTEXT_TOKENS,
            "effective_prefill_chunk_tokens": PREFILL_TOKENS,
            "effective_session_slots": SESSION_SLOTS,
            "expert_cache_limit_bytes": str(CACHE_BYTES),
        }
        try:
            candidate_identity = _running_candidate_identity(
                candidate, process.pid
            )
        except AssertionError as error:
            raise RuntimeError(str(error)) from error
        observed = (
            snapshot["build"],
            snapshot["config"],
            snapshot["limits"],
            snapshot["model"]["id"],
            snapshot["model"]["family"],
            _wire_file_identity(snapshot["executable"]),
            _wire_file_identity(snapshot["model"]),
        )
        expected = (
            expected_build,
            expected_config,
            expected_limits,
            MODEL_ID,
            "laguna",
            candidate_identity,
            model_identity,
        )
        if observed != expected:
            raise RuntimeError(
                "ready runtime identity/profile differs from the guarded "
                f"candidate: expected={expected!r}, observed={observed!r}"
            )
        try:
            _opened_file_identity(process.pid, model_identity)
        except AssertionError as error:
            raise RuntimeError(str(error)) from error

    @classmethod
    def _terminate_server(cls, *, require_clean: bool) -> None:
        process = cls.server_process
        if process is None:
            return
        forced = False
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                forced = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        if require_clean and (forced or process.returncode != 0):
            raise RuntimeError(
                f"returncode={process.returncode}, forced_kill={forced}"
            )

    @classmethod
    def _server_diagnostic(cls) -> str:
        if cls.server_log is None:
            return ""
        cls.server_log.flush()
        cls.server_log.seek(0)
        return cls.server_log.read().decode(
            "utf-8", errors="replace"
        )[-12000:]

    @classmethod
    def _close_server_log(cls) -> None:
        if cls.server_log is not None:
            cls.server_log.close()
            cls.server_log = None

    @classmethod
    def _http(
        cls,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        data = None
        request_headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(
                body, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            cls.server_url + path,
            data=data,
            method=method,
            headers=request_headers,
        )
        timeout = _positive_timeout(
            os.environ.get("DS4_LAGUNA_REQUEST_TIMEOUT", "600")
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return (
                    int(response.status),
                    response.headers.get_content_type(),
                    response.read().decode("utf-8"),
                )
        except urllib.error.HTTPError as error:
            return (
                int(error.code),
                error.headers.get_content_type(),
                error.read().decode("utf-8"),
            )

    @classmethod
    def _get_json(cls, path: str) -> dict[str, Any]:
        status, content_type, wire = cls._http("GET", path)
        if status != 200 or content_type != "application/json":
            raise AssertionError(
                f"GET {path} returned {status} {content_type}: {wire!r}"
            )
        payload = cls.loads(wire)
        if not isinstance(payload, dict):
            raise AssertionError(f"GET {path} returned non-object JSON")
        return payload

    @classmethod
    def _post_json(
        cls,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        status, content_type, wire = cls._http(
            "POST", path, body, headers
        )
        if content_type != "application/json":
            raise AssertionError(
                f"POST {path} returned {status} {content_type}: {wire!r}"
            )
        payload = cls.loads(wire)
        if not isinstance(payload, dict):
            raise AssertionError(f"POST {path} returned non-object JSON")
        return status, payload

    @classmethod
    def _post_sse(
        cls,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> list[tuple[str | None, Any]]:
        status, content_type, wire = cls._http(
            "POST", path, body, headers
        )
        if status != 200 or content_type != "text/event-stream":
            raise AssertionError(
                f"POST {path} returned {status} {content_type}: {wire!r}"
            )
        return _parse_sse_wire(wire, cls.loads)

    def _runtime(self) -> dict[str, Any]:
        snapshot = self._get_json("/v1/runtime")
        self.runtime_validator.validate(snapshot)
        self.assertEqual(snapshot["state"], "ready")
        self.assertEqual(snapshot["violations"], [])
        return snapshot

    def _admission(
        self, requested_output_tokens: int
    ) -> tuple[int, dict[str, Any]]:
        status, result = self._post_json(
            "/v1/token-admission",
            {
                "model": MODEL_ALIAS,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly OK.",
                    }
                ],
                "requested_output_tokens": requested_output_tokens,
            },
        )
        self.admission_validator.validate(result)
        return status, result

    def test_00_runtime_identity_and_exact_two_session_profile(self) -> None:
        snapshot = self._runtime()
        expected_build = {
            key: self.version[key]
            for key in ("revision", "dirty", "backend", "features")
        }
        self.assertEqual(snapshot["build"], expected_build)
        self.assertEqual(
            snapshot["config"],
            {
                "context_tokens": CONTEXT_TOKENS,
                "prefill_chunk_tokens": PREFILL_TOKENS,
                "session_slots": SESSION_SLOTS,
                "ssd_streaming": True,
                "ssd_streaming_cache_bytes": str(CACHE_BYTES),
            },
        )
        self.assertEqual(
            snapshot["limits"],
            {
                "effective_context_tokens": CONTEXT_TOKENS,
                "effective_prefill_chunk_tokens": PREFILL_TOKENS,
                "effective_session_slots": SESSION_SLOTS,
                "expert_cache_limit_bytes": str(CACHE_BYTES),
            },
        )
        self.assertEqual(snapshot["model"]["id"], MODEL_ID)
        self.assertEqual(snapshot["model"]["family"], "laguna")

        process = self.server_process
        model_identity = self.model_identity
        self.assertIsNotNone(process)
        self.assertIsNotNone(model_identity)
        if process is None or model_identity is None:
            self.fail("live process identity was not retained")
        candidate_path = self.candidate_path
        self.assertIsNotNone(candidate_path)
        if candidate_path is None:
            self.fail("candidate path was not retained")
        candidate_identity = _file_identity(candidate_path)
        running_identity = _file_identity(f"/proc/{process.pid}/exe")
        self.assertEqual(
            running_identity,
            candidate_identity,
            "launched process executable differs from the candidate file",
        )
        self.assertEqual(
            _wire_file_identity(snapshot["executable"]),
            candidate_identity,
        )
        self.assertEqual(
            _wire_file_identity(snapshot["model"]), model_identity
        )
        self.assertEqual(
            _opened_file_identity(process.pid, model_identity), model_identity
        )

    def test_10_admission_is_pure_and_preserves_the_exact_boundary(self) -> None:
        before = self._runtime()
        status, probe = self._admission(MAX_OUTPUT_TOKENS)
        after_probe = self._runtime()
        self.assertEqual(status, 200)
        self.assertTrue(probe["fits"])
        self.assertIsNone(probe["rejection_code"])
        self.assertEqual(probe["model"], MODEL_ID)
        self.assertEqual(probe["template_revision"], TEMPLATE_REVISION)
        self.assertEqual(probe["context_tokens"], CONTEXT_TOKENS)
        self.assertEqual(probe["requested_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertGreater(probe["templated_input_tokens"], 0)
        self.assertEqual(
            _runtime_purity_projection(after_probe),
            _runtime_purity_projection(before),
            "token admission mutated live runtime state",
        )

        remaining = CONTEXT_TOKENS - probe["templated_input_tokens"]
        self.assertGreater(remaining, 0)
        status, exact = self._admission(remaining)
        after_exact = self._runtime()
        self.assertEqual(status, 200)
        self.assertTrue(exact["fits"])
        self.assertIsNone(exact["rejection_code"])
        self.assertEqual(
            exact["templated_input_tokens"], probe["templated_input_tokens"]
        )
        self.assertEqual(exact["requested_output_tokens"], remaining)
        self.assertEqual(
            exact["templated_input_tokens"]
            + exact["requested_output_tokens"],
            CONTEXT_TOKENS,
        )
        self.assertEqual(
            _runtime_purity_projection(after_exact),
            _runtime_purity_projection(before),
            "exact-fit admission mutated live runtime state",
        )

        status, overflow = self._admission(remaining + 1)
        after_overflow = self._runtime()
        self.assertEqual(status, 400)
        self.assertFalse(overflow["fits"])
        self.assertEqual(overflow["rejection_code"], "context_overflow")
        self.assertEqual(
            overflow["templated_input_tokens"],
            probe["templated_input_tokens"],
        )
        self.assertEqual(
            overflow["requested_output_tokens"], remaining + 1
        )
        self.assertEqual(
            _runtime_purity_projection(after_overflow),
            _runtime_purity_projection(before),
            "overflow admission mutated live runtime state",
        )

    def _protocol_result(
        self, protocol: str, stream: bool
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str, int, int, str]:
        prompt = "Reply with exactly OK."
        if protocol == "chat":
            body: dict[str, Any] = {
                "model": MODEL_ALIAS,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0,
                "stream": stream,
            }
            if stream:
                body["stream_options"] = {"include_usage": True}
                frames = self._post_sse("/v1/chat/completions", body)
                self.assertGreaterEqual(len(frames), 3)
                self.assertEqual(frames[-1][1], "[DONE]")
                payloads = [
                    payload
                    for _event, payload in frames
                    if isinstance(payload, dict)
                ]
                terminal = frames[-2][1]
                self.assertIsInstance(terminal, dict)
                self.assertEqual(terminal.get("choices"), [])
                self.assertIs(terminal, payloads[-1])
                visible = "".join(
                    str(choice.get("delta", {}).get("content", ""))
                    for payload in payloads[:-1]
                    for choice in payload.get("choices", [])
                    if isinstance(choice, dict)
                    and isinstance(choice.get("delta"), dict)
                )
            else:
                status, terminal = self._post_json(
                    "/v1/chat/completions", body
                )
                self.assertEqual(status, 200)
                payloads = [terminal]
                choices = terminal.get("choices")
                self.assertIsInstance(choices, list)
                visible = "".join(
                    str(choice.get("message", {}).get("content", ""))
                    for choice in choices
                    if isinstance(choice, dict)
                    and isinstance(choice.get("message"), dict)
                )
            usage = terminal.get("usage")
            self.assertIsInstance(usage, dict)
            prompt_tokens = usage["prompt_tokens"]
            generated_tokens = usage["completion_tokens"]
            protocol_id = terminal["id"]

        elif protocol == "responses":
            body = {
                "model": MODEL_ALIAS,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0,
                "reasoning": {"effort": "none"},
                "stream": stream,
            }
            if stream:
                frames = self._post_sse("/v1/responses", body)
                payloads = [
                    payload
                    for _event, payload in frames
                    if isinstance(payload, dict)
                ]
                self.assertGreaterEqual(len(payloads), 2)
                terminal_event = payloads[-1]
                self.assertIn(
                    terminal_event.get("type"),
                    {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    },
                )
                terminal = terminal_event.get("response")
                self.assertIsInstance(terminal, dict)
                visible = "".join(
                    str(payload.get("delta", ""))
                    for payload in payloads[:-1]
                    if payload.get("type") == "response.output_text.delta"
                )
                self.assertEqual(
                    _schema_objects(
                        terminal_event, REQUEST_METRICS_SCHEMA
                    ),
                    _schema_objects(terminal, REQUEST_METRICS_SCHEMA),
                )
            else:
                status, terminal = self._post_json("/v1/responses", body)
                self.assertEqual(status, 200)
                payloads = [terminal]
                visible = "".join(
                    str(part.get("text", ""))
                    for item in terminal.get("output", [])
                    if isinstance(item, dict)
                    for part in item.get("content", [])
                    if isinstance(part, dict)
                    and part.get("type") == "output_text"
                )
            usage = terminal.get("usage")
            self.assertIsInstance(usage, dict)
            prompt_tokens = usage["input_tokens"]
            generated_tokens = usage["output_tokens"]
            protocol_id = terminal["id"]

        elif protocol == "anthropic":
            body = {
                "model": MODEL_ALIAS,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0,
                "stream": stream,
            }
            headers = {"anthropic-version": "2023-06-01"}
            if stream:
                frames = self._post_sse("/v1/messages", body, headers)
                payloads = [
                    payload
                    for _event, payload in frames
                    if isinstance(payload, dict)
                ]
                self.assertGreaterEqual(len(frames), 4)
                self.assertEqual(frames[-1][0], "message_stop")
                self.assertEqual(
                    frames[-1][1], {"type": "message_stop"}
                )
                self.assertEqual(frames[-2][0], "message_delta")
                terminal = frames[-2][1]
                self.assertIsInstance(terminal, dict)
                starts = [
                    payload
                    for payload in payloads
                    if payload.get("type") == "message_start"
                ]
                self.assertEqual(len(starts), 1)
                message = starts[0].get("message")
                self.assertIsInstance(message, dict)
                initial_usage = message.get("usage")
                self.assertIsInstance(initial_usage, dict)
                prompt_tokens = _anthropic_prompt_tokens(initial_usage)
                usage = terminal.get("usage")
                self.assertIsInstance(usage, dict)
                generated_tokens = usage["output_tokens"]
                protocol_id = message["id"]
                visible = "".join(
                    str(payload.get("delta", {}).get("text", ""))
                    for payload in payloads
                    if payload.get("type") == "content_block_delta"
                    and isinstance(payload.get("delta"), dict)
                    and payload["delta"].get("type") == "text_delta"
                )
            else:
                status, terminal = self._post_json(
                    "/v1/messages", body, headers
                )
                self.assertEqual(status, 200)
                payloads = [terminal]
                usage = terminal.get("usage")
                self.assertIsInstance(usage, dict)
                prompt_tokens = _anthropic_prompt_tokens(usage)
                generated_tokens = usage["output_tokens"]
                protocol_id = terminal["id"]
                visible = "".join(
                    str(block.get("text", ""))
                    for block in terminal.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        else:
            raise AssertionError(f"unknown protocol: {protocol}")

        self.assertIs(type(prompt_tokens), int)
        self.assertIs(type(generated_tokens), int)
        self.assertIsInstance(protocol_id, str)
        self.assertTrue(visible.strip(), f"{protocol} returned no visible output")
        return (
            payloads,
            terminal,
            protocol_id,
            prompt_tokens,
            generated_tokens,
            visible,
        )

    def test_20_all_protocols_publish_native_usage_and_terminal_metrics(self) -> None:
        admission_status, admission = self._admission(MAX_OUTPUT_TOKENS)
        self.assertEqual(admission_status, 200)
        self.assertTrue(admission["fits"])
        canonical_prompt_tokens = admission["templated_input_tokens"]

        request_ids: set[str] = set()
        protocol_ids: set[str] = set()
        sequences: list[int] = []
        runtime = self._runtime()
        for protocol in ("chat", "responses", "anthropic"):
            for stream in (False, True):
                with self.subTest(protocol=protocol, stream=stream):
                    (
                        payloads,
                        terminal,
                        protocol_id,
                        native_prompt_tokens,
                        native_generated_tokens,
                        visible,
                    ) = self._protocol_result(protocol, stream)
                    metrics = _schema_objects(
                        payloads, REQUEST_METRICS_SCHEMA
                    )
                    self.assertEqual(
                        len(metrics),
                        1,
                        "each native response must expose exactly one "
                        "request-metrics object",
                    )
                    metric = metrics[0]
                    self.request_validator.validate(metric)
                    self.assertEqual(
                        _schema_objects(
                            terminal, REQUEST_METRICS_SCHEMA
                        ),
                        [metric],
                        "metrics were not placed in the native terminal",
                    )
                    self.assertEqual(metric["instance_id"], runtime["instance_id"])
                    self.assertEqual(metric["prompt_tokens"], native_prompt_tokens)
                    self.assertEqual(
                        metric["prompt_tokens"], canonical_prompt_tokens
                    )
                    self.assertEqual(
                        metric["generated_tokens"], native_generated_tokens
                    )
                    self.assertGreater(metric["generated_tokens"], 0)
                    self.assertLessEqual(
                        metric["generated_tokens"], MAX_OUTPUT_TOKENS
                    )
                    self.assertEqual(metric["terminal_status"], "completed")
                    self.assertGreater(int(metric["ttft_ns"]), 0)
                    self.assertGreater(int(metric["wall_time_ns"]), 0)
                    self.assertGreater(
                        float(metric["prefill_tokens_per_second"]), 0.0
                    )
                    self.assertGreaterEqual(
                        float(metric["visible_decode_tokens_per_second"]),
                        0.0,
                    )
                    self.assertTrue(visible.strip())

                    request_id = metric["request_id"]
                    self.assertEqual(str(uuid.UUID(request_id)), request_id)
                    self.assertNotEqual(request_id, metric["instance_id"])
                    self.assertNotEqual(request_id, protocol_id)
                    self.assertNotEqual(protocol_id, metric["instance_id"])
                    self.assertNotIn(request_id, request_ids)
                    self.assertNotIn(protocol_id, protocol_ids)
                    request_ids.add(request_id)
                    protocol_ids.add(protocol_id)
                    sequences.append(int(metric["snapshot_seq"]))

                    attempts = int(metric["page_advice_attempts"])
                    failures = int(metric["page_advice_failures"])
                    completion = metric[
                        "page_advice_complete_monotonic_ns"
                    ]
                    self.assertLessEqual(failures, attempts)
                    if attempts == 0:
                        self.assertIsNone(completion)
                    else:
                        self.assertIsNotNone(completion)
                        self.assertGreater(int(completion), 0)

        self.assertEqual(len(request_ids), 6)
        self.assertEqual(len(protocol_ids), 6)
        self.assertTrue(
            all(left < right for left, right in zip(sequences, sequences[1:])),
            f"request snapshot sequences were not strictly increasing: {sequences}",
        )
        final_runtime = self._runtime()
        self.assertGreater(
            int(final_runtime["snapshot_seq"]), sequences[-1]
        )
        for stable in (
            "instance_id",
            "build",
            "executable",
            "model",
            "config",
            "limits",
        ):
            with self.subTest(post_requests_stable=stable):
                self.assertEqual(final_runtime[stable], runtime[stable])


if __name__ == "__main__":
    sys.argv[:] = UNITTEST_ARGV
    unittest.main()
