#!/usr/bin/env python3
"""RED publication contract for DS4 engine and HTTP runtime snapshots."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
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


@unittest.skipUnless(
    LIVE_SERVER_URL,
    "set DS4_RUNTIME_SERVER_URL to an already-running Laguna ds4-server",
)
class RuntimeEndpointLiveTest(unittest.TestCase):
    def _get(self, path: str) -> Any:
        request = urllib.request.Request(
            LIVE_SERVER_URL + path,
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.headers.get_content_type(),
                "application/json",
            )
            return PROFILE.loads_strict(response.read().decode("utf-8"))

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
    unittest.main()
