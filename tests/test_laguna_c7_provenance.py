#!/usr/bin/env python3
"""Focused host tests for the fail-closed Laguna C7 oracle producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path


# Every synthetic Git subprocess inherits this process environment. Keep the
# fixture independent of operator templates, hooks, filters, and system config.
os.environ.update(
    GIT_CONFIG_GLOBAL="/dev/null",
    GIT_CONFIG_NOSYSTEM="1",
    GIT_CONFIG_COUNT="1",
    GIT_CONFIG_KEY_0="core.hooksPath",
    GIT_CONFIG_VALUE_0="/dev/null",
)


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = (
    ROOT
    / "tests/oracle-producers/laguna-c7/verify_poolside_laguna_moe.py"
)
CAPTURE_SCRIPT = (
    ROOT
    / "tests/oracle-producers/laguna-c7/capture_poolside_laguna_moe.sh"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def synthetic_elf64(*dynamic_tags: int) -> bytes:
    identifier = b"\x7fELF" + bytes((2, 1, 1, 0)) + bytes(8)
    program_count = 1 if dynamic_tags else 0
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        identifier,
        2,
        183,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        program_count,
        64,
        0,
        0,
    )
    if not dynamic_tags:
        return header
    dynamic = b"".join(struct.pack("<qQ", tag, 0) for tag in dynamic_tags)
    dynamic += struct.pack("<qQ", 0, 0)
    dynamic_offset = len(header) + 56
    program = struct.pack(
        "<IIQQQQQQ",
        2,
        4,
        dynamic_offset,
        0,
        0,
        len(dynamic),
        len(dynamic),
        8,
    )
    return header + program + dynamic


def load_verifier():
    spec = importlib.util.spec_from_file_location("laguna_c7_provenance", VERIFIER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import verifier: {VERIFIER}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not hasattr(module, "Verifier"):
        raise AssertionError("verifier must expose the Verifier contract")
    return module


class SyntheticProducer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.producer = root / "tests/oracle-producers/laguna-c7"
        self.q4_fixture = root / "tests/test-vectors/laguna-q4-l2-auto"
        self.residual_fixture = root / "tests/test-vectors/laguna-moe-residual-auto"
        self.poolside_src = root / "poolside"
        self.poolside_build = root / "poolside-build"
        self.capture_root = root / "captures"
        self.model = root / "model.gguf"
        self.probe_bin = root / "probe"
        self.continuity = tempfile.TemporaryFile()
        for path in (
            self.producer,
            self.q4_fixture,
            self.residual_fixture,
            self.poolside_src / "src",
            self.poolside_build
            / "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir",
            self.capture_root / "capture-22",
            self.capture_root / "capture-1",
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.model_bytes = b"pinned synthetic model"
        self.model.write_bytes(self.model_bytes)
        self.probe_bin.write_bytes(synthetic_elf64())
        self.tokens = struct.pack("<2i", 2, 97)
        assets = {
            "probe": ("probe.cpp", b"synthetic probe source\n"),
            "poolside_patch": (
                "poolside.patch",
                (
                    b"diff --git a/src/llama-graph.cpp b/src/llama-graph.cpp\n"
                    b"--- a/src/llama-graph.cpp\n"
                    b"+++ b/src/llama-graph.cpp\n"
                    b"@@ -4,3 +4,3 @@\n"
                    b" line4\n"
                    b"-before\n"
                    b"+after\n"
                    b" line6\n"
                ),
            ),
            "capture_script": ("capture.sh", b"synthetic capture script\n"),
            "tokens": ("short.tokens.i32", self.tokens),
            "verifier": ("verify.py", b"synthetic verifier identity\n"),
        }
        producer_contract = {}
        for key, (name, payload) in assets.items():
            path = self.producer / name
            path.write_bytes(payload)
            producer_contract[key] = {
                "path": f"tests/oracle-producers/laguna-c7/{name}",
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        producer_contract["tokens"].update(
            {
                "format": "little-endian-int32",
                "count": 2,
                "prompt_sha256": sha256(b"synthetic prompt"),
                "tokenizer_runtime_commit": "1" * 40,
                "ids": [2, 97],
            }
        )

        graph = self.poolside_src / "src/llama-graph.cpp"
        readme = self.poolside_src / "README"
        self.base_graph = (
            "line1\nline2\nline3\nline4\nbefore\n"
            "line6\nline7\nline8\nline9\nline10\n"
        )
        self.patched_graph = self.base_graph.replace("before\n", "after\n")
        graph.write_text(self.base_graph, encoding="utf-8")
        readme.write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.poolside_src)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.poolside_src),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.poolside_src), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.poolside_src),
                "add",
                "README",
                "src/llama-graph.cpp",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.poolside_src), "commit", "-qm", "fixture"],
            check=True,
        )
        self.poolside_commit = subprocess.check_output(
            ["git", "-C", str(self.poolside_src), "rev-parse", "HEAD"], text=True
        ).strip()

        tool_directory = self.root / "tools"
        tool_directory.mkdir()
        tool_facts = {
            "cmake": {"version": "1.0"},
            "cxx": {"version": "2.0", "target": "synthetic-linux"},
            "cuda": {
                "version": "3.0",
                "release": "3.0",
                "build": "synthetic-cuda-build",
            },
            "make": {"version": "4.0"},
        }
        self.toolchain_contract = {}
        for name, facts in tool_facts.items():
            path = tool_directory / name
            payload = f"synthetic {name} executable\n".encode()
            path.write_bytes(payload)
            path.chmod(0o755)
            self.toolchain_contract[name] = {
                "path": str(path),
                "realpath": str(path.resolve()),
                "sha256": sha256(payload),
                **facts,
            }
        self.toolchain_observed = json.loads(
            json.dumps(self.toolchain_contract)
        )

        (self.poolside_build / "CMakeCache.txt").write_text(
            "\n".join(
                (
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={self.poolside_src}",
                    "CMAKE_BUILD_TYPE:STRING=Release",
                    "GGML_CUDA:BOOL=ON",
                    "GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF",
                    "GGML_CUDA_FORCE_CUBLAS:BOOL=OFF",
                    "CMAKE_CXX_FLAGS:STRING=",
                    "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3",
                    "CMAKE_GENERATOR:INTERNAL=Unix Makefiles",
                    (
                        "CMAKE_COMMAND:INTERNAL="
                        f"{self.toolchain_contract['cmake']['path']}"
                    ),
                    (
                        "CMAKE_MAKE_PROGRAM:FILEPATH="
                        f"{self.toolchain_contract['make']['path']}"
                    ),
                    (
                        "CMAKE_CXX_COMPILER:FILEPATH="
                        f"{self.toolchain_contract['cxx']['path']}"
                    ),
                    (
                        "CMAKE_CUDA_COMPILER:FILEPATH="
                        f"{self.toolchain_contract['cuda']['path']}"
                    ),
                    "",
                )
            ),
            encoding="utf-8",
        )
        (
            self.poolside_build
            / "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make"
        ).write_text(
            "CUDA_FLAGS = -std=c++17 -use_fast_math\n", encoding="utf-8"
        )
        recipe_path = "src/CMakeFiles/llama.dir/link.txt"
        recipe = self.poolside_build / recipe_path
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe_payload = (
            f"source={self.poolside_src.resolve()}\n"
            f"build={self.poolside_build.resolve()}\nlink\n"
        ).encode()
        recipe.write_bytes(recipe_payload)
        normalized_recipe = recipe_payload.replace(
            str(self.poolside_build.resolve()).encode(), b"$BUILD"
        ).replace(str(self.poolside_src.resolve()).encode(), b"$SOURCE")
        self.recipe_contract = {
            recipe_path: {
                "normalized_bytes": len(normalized_recipe),
                "normalized_sha256": sha256(normalized_recipe),
            }
        }

        q4_payload = b"q4-source"
        (self.capture_root / "capture-22/q4-source.bin").write_bytes(q4_payload)
        (self.q4_fixture / "q4.bin").write_bytes(q4_payload)

        residual = struct.pack("<f", 1.0)
        moe = struct.pack("<f", 1.0e20)
        shared = struct.pack("<f", -1.0e20)
        expected = struct.pack("<f", 1.0)
        residual_payloads = {
            "residual-token0.f32": ("layer-01-ffn-inp.f32", residual),
            "moe-token0.f32": ("layer-01-ffn-moe-out.f32", moe),
            "shared-token0.f32": ("layer-01-ffn-shared-out.f32", shared),
            "expected-token0.f32": ("layer-01.f32", expected),
        }
        for fixture_name, (capture_name, payload) in residual_payloads.items():
            (self.capture_root / "capture-22" / capture_name).write_bytes(payload)
            (self.residual_fixture / fixture_name).write_bytes(payload)

        non_perturbation_payloads = {
            "prefill_128": {
                "selected_sha256": (
                    "layer-01-router-selected.i32",
                    struct.pack("<i", 1),
                ),
                "weights_sha256": (
                    "layer-01-router-weights.f32",
                    struct.pack("<f", 0.5),
                ),
                "routed_output_sha256": (
                    "layer-01-ffn-moe-out.f32",
                    moe,
                ),
                "shared_output_sha256": (
                    "layer-01-ffn-shared-out.f32",
                    shared,
                ),
                "ffn_output_sha256": (
                    "layer-01-ffn-out.f32",
                    struct.pack("<f", 2.0),
                ),
                "layer_output_sha256": ("layer-01.f32", expected),
            },
            "decode_512": {
                "selected_sha256": (
                    "layer-01-router-selected.i32",
                    struct.pack("<i", 2),
                ),
                "weights_sha256": (
                    "layer-01-router-weights.f32",
                    struct.pack("<f", 0.25),
                ),
                "routed_output_sha256": (
                    "layer-01-ffn-moe-out.f32",
                    struct.pack("<f", 3.0),
                ),
                "shared_output_sha256": (
                    "layer-01-ffn-shared-out.f32",
                    struct.pack("<f", 4.0),
                ),
                "ffn_output_sha256": (
                    "layer-01-ffn-out.f32",
                    struct.pack("<f", 5.0),
                ),
                "layer_output_sha256": (
                    "layer-01.f32",
                    struct.pack("<f", 6.0),
                ),
            },
        }
        non_perturbation = {}
        capture_directories = {
            "prefill_128": "capture-22",
            "decode_512": "capture-1",
        }
        for mode, payloads in non_perturbation_payloads.items():
            non_perturbation[mode] = {}
            for key, (filename, payload) in payloads.items():
                path = self.capture_root / capture_directories[mode] / filename
                path.write_bytes(payload)
                non_perturbation[mode][key] = sha256(payload)

        execution = {
            "toolchain": self.toolchain_contract,
            "generated_recipes": self.recipe_contract,
            "device": {
                "name": "Synthetic GPU",
                "compute_capability": "1.0",
                "multiprocessors": 2,
            },
            "poolside_build": {
                "cmake_build_type": "Release",
                "ggml_cuda": True,
                "ggml_cuda_fa_all_quants": False,
                "ggml_cuda_force_cublas": False,
                "cuda_flags": ["-std=c++17", "-use_fast_math"],
                "cxx_flags": ["-O3"],
            },
            "probe_build": {
                "compiler": self.toolchain_contract["cxx"]["path"],
                "flags": ["-std=c++17", "-O2"],
                "libraries": ["llama", "ggml", "ggml-base"],
                "runtime_library_path": "poolside_build/bin",
                "elf_runpath": "none",
                "binary_bytes": self.probe_bin.stat().st_size,
                "binary_sha256": sha256(self.probe_bin.read_bytes()),
            },
        }
        common = {
            "poolside_commit": self.poolside_commit,
            "model": {
                "bytes": len(self.model_bytes),
                "sha256": sha256(self.model_bytes),
            },
            "producer": producer_contract,
            "execution": execution,
        }
        q4_manifest = {
            "schema": "laguna-q4-l2-auto-fixture/v2",
            **common,
            "capture": {
                "flash_attention": "AUTO",
                "prefill_128": {"capture_directory": "capture-22"},
                "decode_512": {"capture_directory": "capture-1"},
                "non_perturbation": {
                    **non_perturbation,
                    "consolidated_capture_matches_prior_captures": True,
                },
            },
            "oracle": {},
            "extractions": {
                "q4.bin": {
                    "capture_directory": "capture-22",
                    "source": "q4-source.bin",
                    "source_bytes": len(q4_payload),
                    "source_sha256": sha256(q4_payload),
                    "offset": 0,
                    "bytes": len(q4_payload),
                    "producer": "probe",
                }
            },
            "files": {
                "q4.bin": {"bytes": len(q4_payload), "sha256": sha256(q4_payload)}
            },
        }
        residual_extractions = {}
        residual_files = {}
        for fixture_name, (capture_name, payload) in residual_payloads.items():
            residual_extractions[fixture_name] = {
                "capture_directory": "capture-22",
                "source": capture_name,
                "source_bytes": len(payload),
                "source_sha256": sha256(payload),
                "offset": 0,
                "bytes": len(payload),
                "producer": "probe",
            }
            residual_files[fixture_name] = {
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        residual_manifest = {
            "schema": "laguna-moe-residual-auto-fixture/v2",
            **common,
            "capture": {
                "flash_attention": "AUTO",
                "capture_directory": "capture-22",
                "non_perturbation": {
                    "routed_output_sha256": non_perturbation["prefill_128"][
                        "routed_output_sha256"
                    ],
                    "shared_output_sha256": non_perturbation["prefill_128"][
                        "shared_output_sha256"
                    ],
                    "ffn_output_sha256": non_perturbation["prefill_128"][
                        "ffn_output_sha256"
                    ],
                    "layer_output_sha256": non_perturbation["prefill_128"][
                        "layer_output_sha256"
                    ],
                    "consolidated_capture_matches_prior_captures": True,
                },
            },
            "oracle": {
                "expression": "(moe + shared) + residual",
                "expected_exact": True,
                "intended_order_mismatches": 0,
                "legacy_order": "(residual + moe) + shared",
                "legacy_order_mismatches": 1,
                "legacy_first_mismatch": 0,
            },
            "extractions": residual_extractions,
            "files": residual_files,
        }
        (self.q4_fixture / "manifest.json").write_text(
            json.dumps(q4_manifest), encoding="utf-8"
        )
        (self.residual_fixture / "manifest.json").write_text(
            json.dumps(residual_manifest), encoding="utf-8"
        )

    def device(self):
        return {
            "name": "Synthetic GPU",
            "compute_capability": "1.0",
            "multiprocessors": 2,
        }

    def toolchain(self, name: str, path: Path):
        if str(path) != self.toolchain_contract[name]["path"]:
            raise AssertionError(f"unexpected tool path for {name}: {path}")
        return self.toolchain_observed[name]

    def relocate_build(self, destination: Path) -> None:
        """Move and regenerate the synthetic build at its new absolute path."""
        self.poolside_build.rename(destination)
        self.poolside_build = destination

        recipe_path = next(iter(self.recipe_contract))
        recipe_payload = (
            f"source={self.poolside_src.resolve()}\n"
            f"build={self.poolside_build.resolve()}\nlink\n"
        ).encode()
        (self.poolside_build / recipe_path).write_bytes(recipe_payload)
        normalized_recipe = recipe_payload.replace(
            str(self.poolside_build.resolve()).encode(), b"$BUILD"
        ).replace(str(self.poolside_src.resolve()).encode(), b"$SOURCE")
        self.recipe_contract[recipe_path] = {
            "normalized_bytes": len(normalized_recipe),
            "normalized_sha256": sha256(normalized_recipe),
        }

        for manifest_path in (
            self.q4_fixture / "manifest.json",
            self.residual_fixture / "manifest.json",
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["execution"]["generated_recipes"] = self.recipe_contract
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def apply_patch(self) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(self.poolside_src),
                "apply",
                str(self.producer / "poolside.patch"),
            ],
            check=True,
        )


class LagunaC7ProvenanceTest(unittest.TestCase):
    def test_verifier_exposes_only_fixed_preflight_and_captured_phases(self) -> None:
        self.assertTrue(VERIFIER.is_file(), f"missing verifier: {VERIFIER}")
        result = subprocess.run(
            [sys.executable, str(VERIFIER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{preflight,captured}", result.stdout)
        self.assertNotIn("--manifest", result.stdout)

        verifier_module = load_verifier()
        self.assertEqual(verifier_module.discover_repo_root(VERIFIER), ROOT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            shallow = Path(temporary_directory) / VERIFIER.name
            shallow.write_bytes(VERIFIER.read_bytes())
            with self.assertRaisesRegex(
                verifier_module.ContractError, "full DS4 checkout"
            ):
                verifier_module.discover_repo_root(shallow)

    def test_gpu_pid_guard_is_sourceable_strict_and_fail_closed(self) -> None:
        harness = r'''
source "$1"
gpu_compute_processes() {
    printf '%s' "$GPU_OUTPUT"
    return "$GPU_STATUS"
}
assert_gpu_processes "$GPU_ALLOWED"
'''
        cases = (
            ("idle", "", "", "0", True),
            ("probe only", "773\n", "773", "0", True),
            ("foreign alongside probe", "773\n991\n", "773", "0", False),
            ("malformed", "not-a-pid\n", "", "0", False),
            ("query failure", "", "", "17", False),
        )
        for name, output, allowed, status, expected in cases:
            with self.subTest(name=name):
                environment = os.environ.copy()
                environment.update(
                    GPU_OUTPUT=output,
                    GPU_ALLOWED=allowed,
                    GPU_STATUS=status,
                )
                result = subprocess.run(
                    ["bash", "-c", harness, "bash", str(CAPTURE_SCRIPT)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode == 0, expected, result.stderr)

    def test_gpu_guard_terminates_probe_when_foreign_process_appears(self) -> None:
        harness = r'''
source "$1"
gpu_compute_processes() {
    if mkdir "$GPU_FIRST_QUERY" 2>/dev/null; then
        return 0
    fi
    printf '991\n'
}
run_probe_exclusive bash -c 'sleep 5'
'''
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = os.environ.copy()
            environment["GPU_FIRST_QUERY"] = str(
                Path(temporary_directory) / "first-query"
            )
            started = time.monotonic()
            result = subprocess.run(
                ["bash", "-c", harness, "bash", str(CAPTURE_SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=3,
            )
            elapsed = time.monotonic() - started
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(elapsed, 2.0)
        self.assertIn("foreign GPU compute-process PID", result.stderr)

    def test_capture_environment_drops_build_and_loader_injection(self) -> None:
        harness = r'''
source "$1"
capture_environment /usr/bin/env
'''
        environment = os.environ.copy()
        environment.update(
            NVCC_PREPEND_FLAGS="--hostile-nvcc",
            MAKEFLAGS="--eval=hostile",
            CPATH="/hostile/include",
            LIBRARY_PATH="/hostile/lib",
            COMPILER_PATH="/hostile/compiler",
            GCC_EXEC_PREFIX="/hostile/gcc",
            LD_PRELOAD="/hostile/preload.so",
            LD_AUDIT="/hostile/audit.so",
        )
        result = subprocess.run(
            ["bash", "-c", harness, "bash", str(CAPTURE_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        self.assertEqual(
            observed,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin:/usr/local/cuda/bin",
                "TMPDIR": "/tmp",
            },
        )

    def test_capture_main_rejects_a_spoofed_sanitization_marker(self) -> None:
        environment = os.environ.copy()
        environment.update(
            LAGUNA_C7_SANITIZED="1",
            NVCC_PREPEND_FLAGS="--hostile-nvcc",
        )
        result = subprocess.run(
            ["bash", str(CAPTURE_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("noncanonical capture environment", result.stderr)
        self.assertNotIn("usage:", result.stderr)

    def test_capture_shebang_ignores_bash_env_before_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            sentinel = temporary / "bash-env-sourced"
            bash_env = temporary / "hostile-bash-env"
            bash_env.write_text(': > "$BASH_ENV_SENTINEL"\n', encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                BASH_ENV=str(bash_env),
                BASH_ENV_SENTINEL=str(sentinel),
            )
            result = subprocess.run(
                [str(CAPTURE_SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("usage:", result.stderr)
            self.assertFalse(sentinel.exists(), "BASH_ENV ran before env -i")

    def test_clean_build_guard_rejects_any_stale_target_output(self) -> None:
        harness = r'''
source "$1"
poolside_build=$2
assert_build_outputs_absent
'''
        with tempfile.TemporaryDirectory() as temporary_directory:
            build = Path(temporary_directory)
            object_directory = (
                build / "ggml/src/CMakeFiles/ggml-base.dir/src"
            )
            object_directory.mkdir(parents=True)
            stale = object_directory / "stale.o"
            stale.write_bytes(b"stale")
            stale_result = subprocess.run(
                [
                    "bash",
                    "-c",
                    harness,
                    "bash",
                    str(CAPTURE_SCRIPT),
                    str(build),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(stale_result.returncode, 0)
            stale.unlink()

            stale.symlink_to("missing-object")
            stale_symlink_result = subprocess.run(
                [
                    "bash",
                    "-c",
                    harness,
                    "bash",
                    str(CAPTURE_SCRIPT),
                    str(build),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(stale_symlink_result.returncode, 0)
            stale.unlink()

            clean_result = subprocess.run(
                [
                    "bash",
                    "-c",
                    harness,
                    "bash",
                    str(CAPTURE_SCRIPT),
                    str(build),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(clean_result.returncode, 0, clean_result.stderr)

    def test_source_pin_rejects_ignored_files_and_allows_only_nested_build(
        self,
    ) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            nested_build = fixture.poolside_src / "build-c7-diag"
            fixture.relocate_build(nested_build)
            recipe_path = next(iter(fixture.recipe_contract))
            normalized_recipe = (
                (fixture.poolside_build / recipe_path)
                .read_bytes()
                .replace(
                    str(fixture.poolside_build.resolve()).encode(), b"$BUILD"
                )
                .replace(
                    str(fixture.poolside_src.resolve()).encode(), b"$SOURCE"
                )
            )
            self.assertEqual(
                normalized_recipe,
                b"source=$SOURCE\nbuild=$BUILD\nlink\n",
            )
            exclude = fixture.poolside_src / ".git/info/exclude"
            exclude.write_text(
                f"{exclude.read_text(encoding='utf-8')}build-c7-diag/\nignored.h\n",
                encoding="utf-8",
            )
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            ignored = fixture.poolside_src / "ignored.h"
            try:
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )

                ignored.write_text("hostile ignored header\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "untracked files"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                ignored.unlink()

                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                fixture.apply_patch()
                ignored.write_text("captured hostile header\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "untracked files"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_source_pin_rejects_parent_repositories_and_replace_refs(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                nested_source = fixture.poolside_src / "nested-source"
                nested_source.mkdir()
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "repository root"
                ):
                    verifier.preflight(
                        nested_source,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                nested_source.rmdir()

                readme = fixture.poolside_src / "README"
                readme.write_text("replacement tree\n", encoding="utf-8")
                subprocess.run(
                    ["git", "-C", str(fixture.poolside_src), "add", "README"],
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "commit",
                        "-qm",
                        "replacement",
                    ],
                    check=True,
                )
                replacement = subprocess.check_output(
                    ["git", "-C", str(fixture.poolside_src), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "reset",
                        "--hard",
                        fixture.poolside_commit,
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "replace",
                        fixture.poolside_commit,
                        replacement,
                    ],
                    check=True,
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "replacement refs"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_source_pin_hashes_raw_nonpatch_bytes_despite_clean_filters(
        self,
    ) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            readme = fixture.poolside_src / "README"
            attributes = fixture.poolside_src / ".git/info/attributes"
            try:
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                attributes.write_text("README filter=hide-drift\n", encoding="utf-8")
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "config",
                        "filter.hide-drift.clean",
                        "/usr/bin/sed 's/.*/base/'",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "add",
                        "--renormalize",
                        "README",
                    ],
                    check=True,
                )
                readme.write_text("evil\n", encoding="utf-8")
                attribute = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "check-attr",
                        "filter",
                        "--",
                        "README",
                    ],
                    text=True,
                ).strip()
                self.assertEqual(attribute, "README: filter: hide-drift")
                filtered_hash = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "hash-object",
                        "--path=README",
                        "README",
                    ],
                    text=True,
                ).strip()
                expected_hash = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "rev-parse",
                        "HEAD:README",
                    ],
                    text=True,
                ).strip()
                self.assertEqual(filtered_hash, expected_hash)
                hidden_status = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "status",
                        "--porcelain",
                    ],
                    text=True,
                )
                self.assertEqual(hidden_status, "", "clean filter did not hide drift")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "raw tracked tree"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )

                readme.write_text("base\n", encoding="utf-8")
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                fixture.apply_patch()
                readme.write_text("pwn!\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "raw tracked tree"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_raw_tree_hash_rejects_atomic_path_replacement(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            tracked = temporary / "tracked.cpp"
            replacement = temporary / "replacement.cpp"
            tracked.write_bytes(b"a" * (9 * 1024 * 1024))
            replacement.write_bytes(b"b" * (9 * 1024 * 1024))
            verifier = verifier_module.Verifier(temporary)
            original_read = os.read
            replaced = False

            def replace_after_first_read(descriptor: int, count: int) -> bytes:
                nonlocal replaced
                chunk = original_read(descriptor, count)
                if not replaced:
                    replacement.replace(tracked)
                    replaced = True
                return chunk

            def stable_open_identity(identity: os.stat_result) -> dict[str, int]:
                return {
                    "device": identity.st_dev,
                    "inode": identity.st_ino,
                    "size": identity.st_size,
                    "mtime_ns": identity.st_mtime_ns,
                }

            with (
                mock.patch.object(
                    verifier_module.os,
                    "read",
                    side_effect=replace_after_first_read,
                ),
                mock.patch.object(
                    verifier,
                    "_model_identity",
                    side_effect=stable_open_identity,
                ),
            ):
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "path changed during hash"
                ):
                    verifier._raw_blob_id(tracked, "100644")

    def test_preflight_binds_the_held_model_and_rejects_external_drift(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                replacement = fixture.root / "replacement.gguf"
                fixture.model.rename(replacement)
                fixture.model.write_bytes(b"path replacement must not be read")
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )

                readme = fixture.poolside_src / "README"
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "update-index",
                        "--assume-unchanged",
                        "README",
                    ],
                    check=True,
                )
                readme.write_text("hidden preflight drift\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "hidden index flag"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                readme.write_text("base\n", encoding="utf-8")
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "update-index",
                        "--no-assume-unchanged",
                        "README",
                    ],
                    check=True,
                )

                cache = fixture.poolside_build / "CMakeCache.txt"
                original_cache = cache.read_text(encoding="utf-8")
                cache.write_text(
                    original_cache.replace(
                        str(fixture.poolside_src), str(fixture.root / "wrong-source")
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "CMAKE_HOME_DIRECTORY"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                cache.write_text(original_cache, encoding="utf-8")

                wrong_device = verifier_module.Verifier(
                    fixture.root,
                    device_query=lambda: {
                        "name": "Wrong GPU",
                        "compute_capability": "1.0",
                        "multiprocessors": 2,
                    },
                    compute_process_query=lambda: [],
                    toolchain_query=fixture.toolchain,
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "device"
                ):
                    wrong_device.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_preflight_rejects_fsmonitor_hidden_index_state(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            index = fixture.poolside_src / ".git/index"
            index_payload = index.read_bytes()
            index_body = index_payload[:-20] + b"FSMN" + struct.pack(">I", 0)
            index.write_bytes(index_body + hashlib.sha1(index_body).digest())
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "hidden index flag"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_git_subprocesses_disable_repository_fsmonitor_hooks(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            sentinel = fixture.root / "fsmonitor-hook-ran"
            hook = fixture.root / "hostile-fsmonitor"
            hook.write_text(
                "#!/bin/sh\n"
                f"/usr/bin/touch '{sentinel.as_posix()}'\n"
                "printf 'token\\n'\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture.poolside_src),
                    "config",
                    "core.fsmonitor",
                    str(hook),
                ],
                check=True,
            )
            verifier = verifier_module.Verifier(fixture.root)
            try:
                try:
                    verifier._git(fixture.poolside_src, "status", "--porcelain")
                except verifier_module.ContractError:
                    pass
                self.assertFalse(
                    sentinel.exists(), "repository fsmonitor hook executed"
                )
            finally:
                fixture.continuity.close()

    def test_captured_phase_checks_raw_slices_fixtures_and_residual_order(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                fixture.apply_patch()
                verifier.captured(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    fixture.capture_root,
                    fixture.probe_bin,
                    continuity_fd=fixture.continuity.fileno(),
                )

                graph = fixture.poolside_src / "src/llama-graph.cpp"
                graph.write_text(f"{fixture.patched_graph}extra\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "pinned patch"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                graph.write_text(fixture.patched_graph, encoding="utf-8")

                readme = fixture.poolside_src / "README"
                readme.write_text("staged drift\n", encoding="utf-8")
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "add",
                        "README",
                    ],
                    check=True,
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "pinned patch"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(fixture.poolside_src),
                        "restore",
                        "--staged",
                        "README",
                    ],
                    check=True,
                )
                readme.write_text("base\n", encoding="utf-8")

                for hidden_flag, clear_flag in (
                    ("--assume-unchanged", "--no-assume-unchanged"),
                    ("--skip-worktree", "--no-skip-worktree"),
                ):
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(fixture.poolside_src),
                            "update-index",
                            hidden_flag,
                            "README",
                        ],
                        check=True,
                    )
                    readme.write_text("hidden drift\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        verifier_module.ContractError, "hidden index flag"
                    ):
                        verifier.captured(
                            fixture.poolside_src,
                            fixture.poolside_build,
                            model_fd,
                            fixture.capture_root,
                            fixture.probe_bin,
                            continuity_fd=fixture.continuity.fileno(),
                        )
                    readme.write_text("base\n", encoding="utf-8")
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(fixture.poolside_src),
                            "update-index",
                            clear_flag,
                            "README",
                        ],
                        check=True,
                    )

                probe_original = fixture.probe_bin.read_bytes()
                manifest_paths = (
                    fixture.q4_fixture / "manifest.json",
                    fixture.residual_fixture / "manifest.json",
                )
                manifest_originals = {
                    path: path.read_text(encoding="utf-8")
                    for path in manifest_paths
                }
                runpath_probe = synthetic_elf64(29)
                fixture.probe_bin.write_bytes(runpath_probe)
                for path, original in manifest_originals.items():
                    payload = json.loads(original)
                    payload["execution"]["probe_build"].update(
                        binary_bytes=len(runpath_probe),
                        binary_sha256=sha256(runpath_probe),
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "RPATH|RUNPATH"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                fixture.probe_bin.write_bytes(probe_original)
                for path, original in manifest_originals.items():
                    path.write_text(original, encoding="utf-8")

                raw = fixture.capture_root / "capture-22/q4-source.bin"
                original_raw = raw.read_bytes()
                raw.write_bytes(b"corrupt")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "source.*q4-source.bin"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                raw.write_bytes(original_raw)

                residual_manifest = fixture.residual_fixture / "manifest.json"
                residual_payload = json.loads(
                    residual_manifest.read_text(encoding="utf-8")
                )
                residual_payload["oracle"] = {
                    "expression": "(residual + moe) + shared",
                    "expected_exact": True,
                    "intended_order_mismatches": 1,
                    "legacy_order": "(moe + shared) + residual",
                    "legacy_order_mismatches": 0,
                    "legacy_first_mismatch": -1,
                }
                residual_manifest.write_text(
                    json.dumps(residual_payload), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError,
                    "residual association oracle mismatch",
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_non_perturbation_is_exact_for_each_declared_capture_mode(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                fixture.apply_patch()
                verifier.captured(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    fixture.capture_root,
                    fixture.probe_bin,
                    continuity_fd=fixture.continuity.fileno(),
                )

                q4_manifest = fixture.q4_fixture / "manifest.json"
                q4_original = q4_manifest.read_text(encoding="utf-8")
                for mode in ("prefill_128", "decode_512"):
                    payload = json.loads(q4_original)
                    del payload["capture"]["non_perturbation"][mode][
                        "layer_output_sha256"
                    ]
                    q4_manifest.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        verifier_module.ContractError,
                        f"non-perturbation.*{mode}",
                    ):
                        verifier.captured(
                            fixture.poolside_src,
                            fixture.poolside_build,
                            model_fd,
                            fixture.capture_root,
                            fixture.probe_bin,
                            continuity_fd=fixture.continuity.fileno(),
                        )
                    q4_manifest.write_text(q4_original, encoding="utf-8")

                decode_layer = fixture.capture_root / "capture-1/layer-01.f32"
                decode_original = decode_layer.read_bytes()
                decode_layer.write_bytes(b"wrong decode layer")
                with self.assertRaisesRegex(
                    verifier_module.ContractError,
                    "decode_512.*layer_output_sha256",
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                decode_layer.write_bytes(decode_original)

                residual_manifest = fixture.residual_fixture / "manifest.json"
                residual_payload = json.loads(
                    residual_manifest.read_text(encoding="utf-8")
                )
                del residual_payload["capture"]["non_perturbation"][
                    "shared_output_sha256"
                ]
                residual_manifest.write_text(
                    json.dumps(residual_payload), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError,
                    "residual.*non-perturbation",
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_model_fd_continuity_rejects_transient_in_place_mutation(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                with tempfile.TemporaryFile() as continuity:
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=continuity.fileno(),
                    )
                    lock_check = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import fcntl, os, sys; "
                                "fd = os.open(sys.argv[1], os.O_RDONLY); "
                                "\ntry: fcntl.flock(fd, fcntl.LOCK_EX | "
                                "fcntl.LOCK_NB)\nexcept BlockingIOError: "
                                "raise SystemExit(0)\nraise SystemExit(1)"
                            ),
                            str(fixture.model),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(lock_check.returncode, 0, lock_check.stderr)
                    fixture.model.write_bytes(b"x" * len(fixture.model_bytes))
                    fixture.model.write_bytes(fixture.model_bytes)
                    fixture.apply_patch()
                    with self.assertRaisesRegex(
                        verifier_module.ContractError, "model continuity"
                    ):
                        verifier.captured(
                            fixture.poolside_src,
                            fixture.poolside_build,
                            model_fd,
                            fixture.capture_root,
                            fixture.probe_bin,
                            continuity_fd=continuity.fileno(),
                        )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_gpu_query_failures_and_foreign_processes_fail_closed(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                def failed_query():
                    raise OSError("synthetic nvidia-smi failure")

                failed = verifier_module.Verifier(
                    fixture.root,
                    device_query=fixture.device,
                    compute_process_query=failed_query,
                    toolchain_query=fixture.toolchain,
                )
                with self.assertRaisesRegex(
                    verifier_module.ContractError,
                    "GPU compute-process query failed",
                ):
                    failed.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )

                active_processes: list[int] = []
                verifier = verifier_module.Verifier(
                    fixture.root,
                    device_query=fixture.device,
                    compute_process_query=lambda: list(active_processes),
                    toolchain_query=fixture.toolchain,
                )
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )
                fixture.apply_patch()
                active_processes.append(4242)
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "GPU is not exclusive"
                ):
                    verifier.captured(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        fixture.capture_root,
                        fixture.probe_bin,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()

    def test_build_provenance_rejects_toolchain_and_recipe_drift(self) -> None:
        verifier_module = load_verifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = SyntheticProducer(Path(temporary_directory))
            verifier = verifier_module.Verifier(
                fixture.root,
                device_query=fixture.device,
                compute_process_query=lambda: [],
                toolchain_query=fixture.toolchain,
            )
            model_fd = os.open(fixture.model, os.O_RDONLY)
            try:
                verifier.preflight(
                    fixture.poolside_src,
                    fixture.poolside_build,
                    model_fd,
                    continuity_fd=fixture.continuity.fileno(),
                )

                original_cuda = fixture.toolchain_observed["cuda"]
                fixture.toolchain_observed["cuda"] = {
                    **original_cuda,
                    "sha256": "0" * 64,
                }
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "toolchain cuda"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
                fixture.toolchain_observed["cuda"] = original_cuda

                recipe_relative = next(iter(fixture.recipe_contract))
                recipe = fixture.poolside_build / recipe_relative
                recipe.write_bytes(recipe.read_bytes() + b"tampered\n")
                with self.assertRaisesRegex(
                    verifier_module.ContractError, "generated recipe"
                ):
                    verifier.preflight(
                        fixture.poolside_src,
                        fixture.poolside_build,
                        model_fd,
                        continuity_fd=fixture.continuity.fileno(),
                    )
            finally:
                os.close(model_fd)
                fixture.continuity.close()


if __name__ == "__main__":
    unittest.main()
