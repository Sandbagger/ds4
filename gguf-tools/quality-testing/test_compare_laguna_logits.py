#!/usr/bin/env python3
"""Fail-closed contract tests for Laguna oracle promotion and verification."""

from __future__ import annotations

import array
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOOL = HERE / "compare_laguna_logits.py"
ROOT = HERE.parents[1]
FIXTURE_SOURCE = ROOT / "tests/test-vectors/laguna-resident"
FIXTURE_INPUTS = {
    "cases.json",
    "short.txt",
    "generate_benchmark_prompt.py",
    "benchmark-32768.txt",
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
}
VOCAB_SIZE = 100352
VECTOR_BYTES = VOCAB_SIZE * 4
CASES = (
    ("short", "laguna-ds4", 3, 1024),
    ("swa-513", "raw", 513, 1024),
    ("yarn-8193", "raw", 8193, 8202),
    ("deep-32768", "raw", 32768, 32768),
)
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
DWARFSTAR_COMMIT = "0123456789abcdef0123456789abcdef01234567"
GGUF_REVISION = "706fa69799926b6afde1af9e24ca2a4923f110a1"
GGUF_SIZE = 68248759648
GGUF_SHA256 = "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
CONTINUATION = [5, 42, 43, 44, 45, 46, 47, 48]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32_bytes(argmax: int) -> bytes:
    values = array.array("f", [0.0]) * VOCAB_SIZE
    values[argmax] = 2.0
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    assert len(payload) == VECTOR_BYTES
    return payload


def i32_bytes(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def fixed_model() -> dict[str, object]:
    return {
        "repository": "poolside/Laguna-S-2.1-GGUF",
        "revision": GGUF_REVISION,
        "file": "laguna-s-2.1-Q4_K_M.gguf",
        "size": GGUF_SIZE,
        "sha256": GGUF_SHA256,
    }


def case_tokens(case_id: str, frontier: int) -> list[int]:
    if case_id == "short":
        return [100000, 17, 100001]
    return [index % VOCAB_SIZE for index in range(frontier)]


def populate_fixture_inputs(root: Path, capture: Path | None = None) -> None:
    for name in (
        "cases.json",
        "short.txt",
        "generate_benchmark_prompt.py",
        "benchmark-32768.txt",
    ):
        shutil.copy2(FIXTURE_SOURCE / name, root / name)
    for case_id in ("swa-513", "yarn-8193", "deep-32768"):
        source = capture / f"{case_id}.prompt" if capture else None
        payload = source.read_bytes() if source else f"prompt:{case_id}\n".encode()
        (root / f"{case_id}.prompt").write_bytes(payload)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_promoted_fixture(root: Path) -> None:
    populate_fixture_inputs(root)
    cases = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = (
            b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
            + (root / "short.txt").read_bytes()
            + b"</user>\n<assistant></think>"
            if case_id == "short"
            else (root / f"{case_id}.prompt").read_bytes()
        )
        tokens = case_tokens(case_id, frontier)
        oracle_entries: dict[str, dict[str, object]] = {}
        for oracle in ("metal", "llama"):
            name = f"{case_id}.{oracle}.f32"
            payload = f32_bytes(case_index + 3)
            (root / name).write_bytes(payload)
            oracle_entries[oracle] = {
                "file": name,
                "sha256": sha256(payload),
                "argmax": case_index + 3,
            }
        cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt_hex": prompt.hex(),
                "prompt_sha256": sha256(prompt),
                "metal_tokens": tokens,
                "llama_tokens": tokens,
                "frontier": frontier,
                "context": context,
                "oracles": oracle_entries,
                "metrics": {
                    "centered_rms": 0.0,
                    "centered_max_abs": 0.0,
                    "top20_overlap": 20,
                    "argmax_equal": True,
                },
            }
        )

    continuation_name = "yarn-8193.continuation.i32"
    continuation_payload = i32_bytes(CONTINUATION)
    (root / continuation_name).write_bytes(continuation_payload)
    manifest = {
        "schema": "laguna-resident-promoted-v1",
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": 8,
        "model": fixed_model(),
        "runtimes": {
            "metal_commit": DWARFSTAR_COMMIT,
            "llama_commit": LLAMA_COMMIT,
        },
        "dwarfstar_commit": DWARFSTAR_COMMIT,
        "thresholds": {
            "promotion": {
                "centered_rms": 0.02,
                "centered_max_abs": 0.10,
                "top20_overlap": 18,
                "argmax_equal": True,
                "continuation_equal": True,
            },
            "cuda_admission": {
                "centered_rms": 0.04,
                "centered_max_abs": 0.20,
                "top20_overlap": 18,
                "argmax_equal": True,
                "teacher_forced_ids_equal": True,
            },
        },
        "seed": {
            "generator_sha256": "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d",
            "benchmark_sha256": "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206",
            "poolside_token_count": 40000,
            "dwarfstar_token_count": 40000,
        },
        "cases": cases,
        "continuation": {
            "case": "yarn-8193",
            "file": continuation_name,
            "sha256": sha256(continuation_payload),
            "metal_argmax": CONTINUATION,
            "llama_argmax": CONTINUATION,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_capture(root: Path, oracle: str) -> None:
    files: dict[str, str] = {}
    capture_cases = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = (
            b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
            + (FIXTURE_SOURCE / "short.txt").read_bytes()
            + b"</user>\n<assistant></think>"
            if case_id == "short"
            else f"prompt:{case_id}\n".encode()
        )
        tokens = case_tokens(case_id, frontier)
        prompt_name = f"{case_id}.prompt"
        tokens_name = f"{case_id}.tokens.i32"
        logits_name = f"{case_id}.logits.f32"
        payloads = {
            prompt_name: prompt,
            tokens_name: i32_bytes(tokens),
            logits_name: f32_bytes(case_index + 3),
        }
        for name, payload in payloads.items():
            (root / name).write_bytes(payload)
            files[name] = sha256(payload)
        capture_cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt": "short.txt" if case_id == "short" else f"{case_id}.prompt",
                "frontier": frontier,
                "context": context,
                "prompt_file": prompt_name,
                "tokens_file": tokens_name,
                "logits_file": logits_name,
                "token_count": len(tokens),
                "argmax": case_index + 3,
            }
        )

    step_files = []
    for step, token in enumerate(CONTINUATION):
        name = f"yarn-8193.step-{step:02d}.logits.f32"
        payload = f32_bytes(token)
        (root / name).write_bytes(payload)
        files[name] = sha256(payload)
        step_files.append(name)
    continuation_name = "yarn-8193.continuation.i32"
    continuation_payload = i32_bytes(CONTINUATION)
    (root / continuation_name).write_bytes(continuation_payload)
    files[continuation_name] = sha256(continuation_payload)

    capture = {
        "schema": "laguna-resident-capture-v1",
        "oracle": oracle,
        "runtime_commit": LLAMA_COMMIT if oracle == "llama" else DWARFSTAR_COMMIT,
        "vocab_size": VOCAB_SIZE,
        "seed_token_count": 40000,
        "model": fixed_model(),
        "cases": capture_cases,
        "continuation": {
            "case": "yarn-8193",
            "tokens_file": continuation_name,
            "logits_files": step_files,
            "argmax": CONTINUATION,
        },
        "files": files,
    }
    if oracle == "metal":
        capture["dwarfstar_commit"] = DWARFSTAR_COMMIT
    (root / "capture.json").write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_native_metal_capture(root: Path, model: Path) -> None:
    for case_index, (case_id, _, frontier, context) in enumerate(CASES):
        tokens = case_tokens(case_id, frontier)
        logits = [0.0] * VOCAB_SIZE
        logits[case_index + 3] = 2.0
        payload = {
            "source": "ds4",
            "model": str(model),
            "backend": "metal",
            "quant_bits": 4,
            "prompt_tokens": len(tokens),
            "ctx": context,
            "vocab": VOCAB_SIZE,
            "argmax_token": {"id": case_index + 3, "text": "", "bytes": []},
            "argmax_logit": 2.0,
            "logits": logits,
        }
        (root / f"{case_id}.logits.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )

    continuation = {
        "source": "ds4",
        "prompt_tokens": 8193,
        "ctx": 8202,
        "top_k": 20,
        "steps": [
            {
                "step": step,
                "selected": {"id": token, "text": "", "bytes": []},
                "top_logprobs": [],
            }
            for step, token in enumerate(CONTINUATION)
        ],
    }
    (root / "yarn-8193.continuation.json").write_text(
        json.dumps(continuation, separators=(",", ":")) + "\n", encoding="utf-8"
    )


class CompareLagunaLogitsTest(unittest.TestCase):
    def run_verify(self, fixture: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--verify-promoted",
                str(fixture),
                "--dwarfstar-commit",
                DWARFSTAR_COMMIT,
                "--llama-commit",
                LLAMA_COMMIT,
                "--gguf-size",
                str(GGUF_SIZE),
                "--gguf-sha256",
                GGUF_SHA256,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_valid_fixture_verifies_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted"
            fixture.mkdir()
            write_promoted_fixture(fixture)
            cache = fixture / "__pycache__"
            cache.mkdir()
            (cache / "ignored.pyc").write_bytes(b"ignored cache")
            before = tree_digest(fixture)

            completed = self.run_verify(fixture)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, f"verified={fixture} cases=4 vectors=8\n")
            self.assertEqual(tree_digest(fixture), before)

    def test_verify_fails_closed_for_representative_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            write_promoted_fixture(base)

            def edit_manifest(root: Path, change) -> None:  # type: ignore[no-untyped-def]
                path = root / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                change(manifest)
                path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            tamperers = {
                "schema": lambda root: edit_manifest(
                    root, lambda data: data.__setitem__("schema", "future-schema")
                ),
                "missing file": lambda root: (root / "short.metal.f32").unlink(),
                "extra file": lambda root: (root / "unexpected.bin").write_bytes(b"x"),
                "size": lambda root: (root / "swa-513.llama.f32").write_bytes(b"short"),
                "hash": lambda root: (root / "short.llama.f32").write_bytes(
                    b"X" + (root / "short.llama.f32").read_bytes()[1:]
                ),
                "identity": lambda root: edit_manifest(
                    root,
                    lambda data: data["runtimes"].__setitem__("llama_commit", "bad"),
                ),
                "tokens": lambda root: edit_manifest(
                    root, lambda data: data["cases"][1]["metal_tokens"].pop()
                ),
                "prompt hex": lambda root: edit_manifest(
                    root, lambda data: data["cases"][0].__setitem__("prompt_hex", "ff")
                ),
                "metrics": lambda root: edit_manifest(
                    root,
                    lambda data: data["cases"][0]["metrics"].__setitem__(
                        "centered_rms", 0.020001
                    ),
                ),
                "continuation": lambda root: edit_manifest(
                    root,
                    lambda data: data["continuation"]["metal_argmax"].__setitem__(0, 9),
                ),
            }
            for label, tamper in tamperers.items():
                with self.subTest(label=label):
                    fixture = Path(tmp) / f"tampered-{label.replace(' ', '-')}"
                    shutil.copytree(base, fixture)
                    tamper(fixture)
                    before = tree_digest(fixture)

                    completed = self.run_verify(fixture)

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("verification failed:", completed.stderr)
                    self.assertEqual(tree_digest(fixture), before)

    def test_verify_rejects_duplicate_keys_and_nonfinite_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            duplicate = Path(tmp) / "duplicate"
            duplicate.mkdir()
            write_promoted_fixture(duplicate)
            manifest_path = duplicate / "manifest.json"
            text = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                text.replace(
                    '"schema": "laguna-resident-promoted-v1",',
                    '"schema": "laguna-resident-promoted-v1",\n'
                    '  "schema": "laguna-resident-promoted-v1",',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(self.run_verify(duplicate).returncode, 0)

            nonfinite = Path(tmp) / "nonfinite"
            nonfinite.mkdir()
            write_promoted_fixture(nonfinite)
            vector = nonfinite / "deep-32768.metal.f32"
            payload = bytearray(vector.read_bytes())
            payload[:4] = struct.pack("<f", float("nan"))
            vector.write_bytes(payload)
            manifest_path = nonfinite / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][3]["oracles"]["metal"]["sha256"] = sha256(payload)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            completed = self.run_verify(nonfinite)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("non-finite", completed.stderr)

    def test_capture_promotes_atomically_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metal = root / "metal"
            llama = root / "llama"
            metal.mkdir()
            llama.mkdir()
            write_capture(metal, "metal")
            write_capture(llama, "llama")
            ds4 = root / "fake-ds4"
            ds4.write_text(
                "#!/usr/bin/env python3\n"
                "import struct, sys\n"
                "from pathlib import Path\n"
                "p = Path(sys.argv[sys.argv.index('--prompt-file') + 1])\n"
                "if p.name == 'benchmark-32768.txt':\n"
                "    print([i % 100352 for i in range(32768)])\n"
                "    raise SystemExit\n"
                "b = p.with_suffix('.tokens.i32').read_bytes()\n"
                "print(list(struct.unpack(f'<{len(b) // 4}i', b)))\n",
                encoding="utf-8",
            )
            ds4.chmod(0o755)
            promoted = root / "promoted"
            promoted.mkdir()
            populate_fixture_inputs(promoted, llama)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--ds4",
                    str(ds4),
                    "--metal",
                    str(metal),
                    "--llama",
                    str(llama),
                    "--promote",
                    str(promoted),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, f"promoted={promoted} cases=4 vectors=8\n")
            self.assertEqual(
                {path.name for path in promoted.iterdir()},
                FIXTURE_INPUTS
                | {
                    "manifest.json",
                    "yarn-8193.continuation.i32",
                    *(f"{case_id}.{oracle}.f32" for case_id, _, _, _ in CASES for oracle in ("metal", "llama")),
                },
            )
            verified = self.run_verify(promoted)
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_native_dwarfstar_metal_outputs_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metal = root / "metal"
            llama = root / "llama"
            repo = root / "dwarfstar"
            metal.mkdir()
            llama.mkdir()
            repo.mkdir()
            model = root / "laguna-s-2.1-Q4_K_M.gguf"
            write_native_metal_capture(metal, model)
            write_capture(llama, "llama")

            ds4 = repo / "ds4"
            ds4.write_text(
                "#!/usr/bin/env python3\n"
                "import struct, sys\n"
                "from pathlib import Path\n"
                "p = Path(sys.argv[sys.argv.index('--prompt-file') + 1])\n"
                "if p.name == 'benchmark-32768.txt':\n"
                "    print([i % 100352 for i in range(32768)])\n"
                "    raise SystemExit\n"
                "b = p.with_suffix('.tokens.i32').read_bytes()\n"
                "print(list(struct.unpack(f'<{len(b) // 4}i', b)))\n",
                encoding="utf-8",
            )
            ds4.chmod(0o755)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Oracle Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "oracle@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "ds4"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "test oracle"], check=True)
            dwarfstar_commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            promoted = root / "promoted"
            promoted.mkdir()
            populate_fixture_inputs(promoted, llama)
            env = os.environ.copy()
            env["LAGUNA_MODEL"] = str(model)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--ds4",
                    str(ds4),
                    "--metal",
                    str(metal),
                    "--llama",
                    str(llama),
                    "--promote",
                    str(promoted),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            verified = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--verify-promoted",
                    str(promoted),
                    "--dwarfstar-commit",
                    dwarfstar_commit,
                    "--llama-commit",
                    LLAMA_COMMIT,
                    "--gguf-size",
                    str(GGUF_SIZE),
                    "--gguf-sha256",
                    GGUF_SHA256,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)


CONTRACT_COMMIT = "a250e43722945e293f6044bc7254c4806d5a7912"
CAPTURE_MANIFEST_SHA256 = (
    "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
)


def fixed_model_v2() -> dict[str, object]:
    model = fixed_model()
    return {
        "repository": model["repository"],
        "revision": model["revision"],
        "filename": model["file"],
        "size": model["size"],
        "sha256": model["sha256"],
    }


def write_promoted_fixture_v2(root: Path) -> None:
    """Normative laguna-resident-promoted-v2 fixture (single Poolside oracle).

    Mirrors docs/superpowers/specs/2026-08-01-laguna-single-poolside-oracle-
    design.md § Fixture schema: exactly four vectors, no Metal artifacts,
    provenance block, cuda_admission-only thresholds.
    """
    populate_fixture_inputs(root)
    cases = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = (
            b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
            + (root / "short.txt").read_bytes()
            + b"</user>\n<assistant></think>"
            if case_id == "short"
            else (root / f"{case_id}.prompt").read_bytes()
        )
        tokens = case_tokens(case_id, frontier)
        name = f"{case_id}.llama.f32"
        payload = f32_bytes(case_index + 3)
        (root / name).write_bytes(payload)
        cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt_hex": prompt.hex(),
                "prompt_sha256": sha256(prompt),
                "poolside_tokens": tokens,
                "ds4_tokens": tokens,
                "frontier": frontier,
                "context": context,
                "vector": {
                    "file": name,
                    "sha256": sha256(payload),
                    "argmax": case_index + 3,
                },
            }
        )

    continuation_name = "yarn-8193.continuation.i32"
    continuation_payload = i32_bytes(CONTINUATION)
    (root / continuation_name).write_bytes(continuation_payload)

    manifest = {
        "schema": "laguna-resident-promoted-v2",
        "oracle_policy": "single-poolside-v1",
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": 8,
        "model": fixed_model_v2(),
        "oracle": {
            "name": "poolside-llama",
            "runtime_commit": LLAMA_COMMIT,
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        },
        "provenance": {
            "contract_commit": CONTRACT_COMMIT,
            "tokenizer_runtime_commit": "0123456789abcdef0123456789abcdef01234567",
            "generator_sha256": "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d",
            "benchmark_sha256": "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206",
            "poolside_seed_token_count": 61440,
            "ds4_seed_token_count": 32768,
        },
        "thresholds": {
            "cuda_admission": {
                "centered_rms": 0.04,
                "centered_max_abs": 0.20,
                "top20_overlap": 18,
                "argmax_equal": True,
                "continuation_equal": True,
            }
        },
        "cases": cases,
        "continuation": {
            "case": "yarn-8193",
            "file": continuation_name,
            "sha256": sha256(continuation_payload),
            "argmax": CONTINUATION,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class SinglePoolsideV2ContractTests(unittest.TestCase):
    """RED SPEC for the single-Poolside oracle migration (2026-08-01 design).

    Step 1 of the migration sequencing: these tests define the v2 promoted
    contract and fail against the current v1 dual-oracle tool. The comparator
    migration (step 2) turns them green without editing them.
    """

    def run_verify_v2(self, fixture: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--verify-promoted",
                str(fixture),
                "--contract-commit",
                CONTRACT_COMMIT,
                "--tokenizer-runtime-commit",
                "0123456789abcdef0123456789abcdef01234567",
                "--llama-commit",
                LLAMA_COMMIT,
                "--capture-manifest-sha256",
                CAPTURE_MANIFEST_SHA256,
                "--gguf-size",
                str(GGUF_SIZE),
                "--gguf-sha256",
                GGUF_SHA256,
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_v2_fixture_verifies_with_single_oracle_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted-v2"
            fixture.mkdir()
            write_promoted_fixture_v2(fixture)
            before = tree_digest(fixture)

            completed = self.run_verify_v2(fixture)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout,
                f"verified={fixture} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(tree_digest(fixture), before)

    def test_v2_verifier_rejects_v1_dual_oracle_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy"
            legacy.mkdir()
            write_promoted_fixture(legacy)

            completed = self.run_verify_v2(legacy)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("verification failed:", completed.stderr)

    def test_v2_verifier_rejects_legacy_metal_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted-v2"
            fixture.mkdir()
            write_promoted_fixture_v2(fixture)

            completed = self.run_verify_v2(
                fixture,
                "--dwarfstar-commit",
                DWARFSTAR_COMMIT,
            )

            self.assertNotEqual(completed.returncode, 0)

    def test_v2_manifest_extra_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "extra-key"
            fixture.mkdir()
            write_promoted_fixture_v2(fixture)
            path = fixture / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["metal_metrics"] = {}
            path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            completed = self.run_verify_v2(fixture)

            self.assertNotEqual(completed.returncode, 0)

    def test_v2_tampered_vector_fails_closed_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            write_promoted_fixture_v2(base)
            fixture = Path(tmp) / "tampered"
            shutil.copytree(base, fixture)
            victim = fixture / "swa-513.llama.f32"
            victim.write_bytes(b"X" + victim.read_bytes()[1:])
            before = tree_digest(fixture)

            completed = self.run_verify_v2(fixture)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("verification failed:", completed.stderr)
            self.assertEqual(tree_digest(fixture), before)

    def test_v2_promote_requires_pinned_capture_trust_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture.mkdir()
            write_capture(capture, "llama")
            ds4 = root / "fake-ds4"
            ds4.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
            ds4.chmod(0o755)
            promoted = root / "promoted"
            promoted.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--ds4",
                    str(ds4),
                    "--llama",
                    str(capture),
                    "--promote",
                    str(promoted),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("trust anchor", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
