#!/usr/bin/env python3
"""RED contract for reproducible DS4 build identity on every frontend."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/ds4-version-v1.schema.json"
SCHEMA_PROFILE = (
    ROOT / "gguf-tools" / "quality-testing" / "compact_runtime_schema.py"
)
FRONTENDS = ("ds4", "ds4-server", "ds4-agent", "ds4-bench", "ds4-eval")
FRONTEND_SOURCES = {
    "ds4": "ds4_cli.c",
    "ds4-server": "ds4_server.c",
    "ds4-agent": "ds4_agent.c",
    "ds4-bench": "ds4_bench.c",
    "ds4-eval": "ds4_eval.c",
}


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
VERSION_SCHEMA = PROFILE.loads_strict(SCHEMA_PATH.read_text(encoding="utf-8"))
VERSION_VALIDATOR = PROFILE.validator_for(VERSION_SCHEMA)


def _make_recipe(makefile: str, target: str) -> str:
    match = re.search(rf"(?m)^{re.escape(target)}\s*:[^\n]*$", makefile)
    if match is None:
        raise AssertionError(f"missing Makefile target {target}")
    lines: list[str] = []
    for line in makefile[match.end() + 1 :].splitlines():
        if not line.startswith("\t"):
            break
        lines.append(line)
    return "\n".join(lines)


def _make_rule(makefile: str, target: str) -> str:
    match = re.search(rf"(?m)^{re.escape(target)}\s*:[^\n]*$", makefile)
    if match is None:
        raise AssertionError(f"missing Makefile target {target}")
    return match.group(0) + "\n" + _make_recipe(makefile, target)


class VersionJsonRuntimeTest(unittest.TestCase):
    maxDiff = None

    def _run_frontend(self, frontend: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(ROOT / frontend), *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def _version(self, frontend: str, *args: str) -> dict[str, Any]:
        completed = self._run_frontend(frontend, "--version-json", *args)
        self.assertEqual(
            completed.returncode,
            0,
            f"{frontend} --version-json failed:\n{completed.stderr}",
        )
        self.assertEqual(
            completed.stderr,
            "",
            f"{frontend} must emit build identity on stdout only",
        )
        payload = PROFILE.loads_strict(completed.stdout)
        VERSION_VALIDATOR.validate(payload)
        return payload

    def test_every_inference_frontend_emits_one_identical_closed_version(self) -> None:
        versions = {frontend: self._version(frontend) for frontend in FRONTENDS}
        reference = versions[FRONTENDS[0]]
        self.assertEqual(
            set(reference),
            {"schema", "revision", "dirty", "backend", "features"},
        )
        self.assertRegex(reference["revision"], r"^[0-9a-f]{40}$")
        self.assertIn(reference["backend"], {"cpu", "metal", "cuda", "rocm"})
        self.assertEqual(reference["features"], sorted(set(reference["features"])))
        for frontend, payload in versions.items():
            with self.subTest(frontend=frontend):
                self.assertEqual(payload, reference)

    def test_version_exits_before_model_validation_or_backend_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "model-would-block-if-opened.gguf"
            os.mkfifo(fifo)
            for frontend in FRONTENDS:
                with self.subTest(frontend=frontend):
                    payload = self._version(
                        frontend,
                        "--model",
                        str(fifo),
                        "--cuda",
                    )
                    self.assertIn(payload["backend"], {"cpu", "metal", "cuda", "rocm"})


class VersionJsonSourceContractTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    def test_one_shared_build_info_implementation_is_linked_by_all_frontends(self) -> None:
        header = ROOT / "ds4_build_info.h"
        source = ROOT / "ds4_build_info.c"
        self.assertTrue(header.is_file(), "missing shared ds4_build_info.h")
        self.assertTrue(source.is_file(), "missing shared ds4_build_info.c")
        header_text = header.read_text(encoding="utf-8") if header.is_file() else ""
        source_text = source.read_text(encoding="utf-8") if source.is_file() else ""
        self.assertIn("ds4_runtime_build_info", header_text)
        self.assertIn("DS4_BUILD_REVISION", source_text)
        self.assertIn("DS4_BUILD_DIRTY", source_text)
        self.assertIn("DS4_BUILD_BACKEND", source_text)
        self.assertIn("DS4_BUILD_FEATURES", source_text)

        for frontend, source_name in FRONTEND_SOURCES.items():
            with self.subTest(frontend=frontend):
                frontend_source = (ROOT / source_name).read_text(encoding="utf-8")
                self.assertIn('"--version-json"', frontend_source)
                self.assertIn("ds4_build_info", frontend_source)
                self.assertIn(
                    "$(DS4_BUILD_INFO_OBJ)",
                    _make_rule(self.makefile, frontend),
                    f"{frontend} must link the backend-selected build-info object",
                )

        self.assertRegex(
            self.makefile,
            r"(?m)^DS4_BUILD_INFO_OBJ\s*\?=\s*ds4_build_info\.o\s*$",
        )
        cpu_rule = _make_rule(self.makefile, "cpu")
        self.assertEqual(
            cpu_rule.count("ds4_build_info_cpu.o"),
            len(FRONTENDS),
            "each CPU frontend link must use CPU-stamped build identity",
        )
        rocm_rule = _make_rule(self.makefile, "strix-halo")
        self.assertIn(
            'DS4_BUILD_INFO_OBJ="ds4_build_info_rocm.o"',
            rocm_rule,
            "the recursive ROCm build must select the ROCm-stamped object",
        )

    def test_backend_specific_build_info_objects_have_exact_compile_facts(self) -> None:
        expected = {
            "ds4_build_info.o": {"metal", "cuda"},
            "ds4_build_info_cpu.o": {"cpu"},
            "ds4_build_info_rocm.o": {"rocm"},
        }
        for target, allowed_backends in expected.items():
            with self.subTest(target=target):
                recipe = _make_recipe(self.makefile, target)
                self.assertIn("DS4_BUILD_REVISION", recipe)
                self.assertIn("DS4_BUILD_DIRTY", recipe)
                self.assertIn("DS4_BUILD_BACKEND", recipe)
                self.assertIn("DS4_BUILD_FEATURES", recipe)
                self.assertIn("laguna,ssd_streaming", recipe)
                self.assertTrue(
                    any(f'\\\"{backend}\\\"' in recipe for backend in allowed_backends),
                    f"{target} does not stamp one of {sorted(allowed_backends)}: {recipe}",
                )

    def test_shared_serializer_sorts_features_and_emits_clean_or_dirty_json(self) -> None:
        source = ROOT / "ds4_build_info.c"
        header = ROOT / "ds4_build_info.h"
        self.assertTrue(source.is_file(), "missing shared ds4_build_info.c")
        self.assertTrue(header.is_file(), "missing shared ds4_build_info.h")
        harness_source = (
            '#include "ds4_build_info.h"\n'
            '#include <stdio.h>\n'
            'int main(void) { return ds4_build_info_write_json(stdout); }\n'
        )
        revision = "1234567890abcdef1234567890abcdef12345678"
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            harness = tmpdir / "version_harness.c"
            harness.write_text(harness_source, encoding="utf-8")
            for dirty in (0, 1):
                with self.subTest(dirty=dirty):
                    binary = tmpdir / f"version-{dirty}"
                    compiled = subprocess.run(
                        [
                            os.environ.get("CC", "cc"),
                            "-std=c99",
                            "-I",
                            str(ROOT),
                            f'-DDS4_BUILD_REVISION="{revision}"',
                            f"-DDS4_BUILD_DIRTY={dirty}",
                            '-DDS4_BUILD_BACKEND="cuda"',
                            '-DDS4_BUILD_FEATURES="ssd_streaming,laguna,laguna"',
                            str(source),
                            str(harness),
                            "-o",
                            str(binary),
                        ],
                        cwd=ROOT,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    emitted = subprocess.run(
                        [str(binary)],
                        cwd=ROOT,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(emitted.returncode, 0, emitted.stderr)
                    self.assertEqual(emitted.stderr, "")
                    payload = PROFILE.loads_strict(emitted.stdout)
                    VERSION_VALIDATOR.validate(payload)
                    self.assertEqual(payload["revision"], revision)
                    self.assertIs(payload["dirty"], bool(dirty))
                    self.assertEqual(payload["backend"], "cuda")
                    self.assertEqual(payload["features"], ["laguna", "ssd_streaming"])

    def test_make_stamps_clean_and_dirty_revisions_without_using_this_worktree(self) -> None:
        source = ROOT / "ds4_build_info.c"
        header = ROOT / "ds4_build_info.h"
        self.assertTrue(source.is_file(), "missing shared ds4_build_info.c")
        self.assertTrue(header.is_file(), "missing shared ds4_build_info.h")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            shutil.copy2(ROOT / "Makefile", repo / "Makefile")
            shutil.copy2(source, repo / "ds4_build_info.c")
            shutil.copy2(header, repo / "ds4_build_info.h")
            shutil.copy2(ROOT / "ds4_runtime.h", repo / "ds4_runtime.h")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "DS4 Test"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "ds4-test@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            clean = subprocess.run(
                ["make", "-n", "ds4_build_info_cpu.o"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertIn(revision, clean.stdout)
            self.assertIn("-DDS4_BUILD_DIRTY=0", clean.stdout)
            clean_repeat = subprocess.run(
                ["make", "-n", "ds4_build_info_cpu.o"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clean_repeat.stdout, clean.stdout)

            (repo / "ds4_build_info.c").write_text(
                (repo / "ds4_build_info.c").read_text(encoding="utf-8") + "\n/* dirty */\n",
                encoding="utf-8",
            )
            dirty = subprocess.run(
                ["make", "-n", "ds4_build_info_cpu.o"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(dirty.returncode, 0, dirty.stderr)
            self.assertIn(revision, dirty.stdout)
            self.assertIn("-DDS4_BUILD_DIRTY=1", dirty.stdout)


if __name__ == "__main__":
    unittest.main()
