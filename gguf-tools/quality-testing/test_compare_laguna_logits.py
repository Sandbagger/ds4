#!/usr/bin/env python3
"""Fail-closed contract tests for single-Poolside Laguna fixture promotion."""

from __future__ import annotations

import array
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
from unittest import mock


HERE = Path(__file__).resolve().parent
TOOL = HERE / "compare_laguna_logits.py"
ROOT = HERE.parents[1]
FIXTURE_SOURCE = ROOT / "tests/test-vectors/laguna-resident"

VOCAB_SIZE = 100352
VECTOR_BYTES = VOCAB_SIZE * 4
CASES = (
    ("short", "laguna-ds4", 3, 1024),
    ("swa-513", "raw", 513, 1024),
    ("yarn-8193", "raw", 8193, 8202),
    ("deep-32768", "raw", 32768, 32768),
)
CONTINUATION = [5, 42, 43, 44, 45, 46, 47, 48]

CONTRACT_COMMIT = "a250e43722945e293f6044bc7254c4806d5a7912"
TOKENIZER_RUNTIME_COMMIT = "0123456789abcdef0123456789abcdef01234567"
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
CAPTURE_MANIFEST_SHA256 = (
    "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
)
ORACLE_POLICY = "single-poolside-v1"
PROMOTED_SCHEMA = "laguna-resident-promoted-v2"

GGUF_REVISION = "706fa69799926b6afde1af9e24ca2a4923f110a1"
GGUF_SIZE = 68248759648
GGUF_SHA256 = "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
GGUF_FILENAME = "laguna-s-2.1-Q4_K_M.gguf"
CORRECTED_GGUF_REVISION = "e2ccc0579fc18e6ea2362fa25fccbcd470f0e332"
CORRECTED_GGUF_SIZE = 68248760064
CORRECTED_GGUF_SHA256 = "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff"
TEST_CHAT_TEMPLATE = b"""{%- set enable_thinking = enable_thinking | default(true) -%}
{%- if add_generation_prompt -%}
{{- "<assistant>" -}}
{%- if enable_thinking -%}
{{- '<think>' -}}
{%- else -%}
{{- '</think>' -}}
{%- endif -%}
{%- endif -%}
"""
TEST_CHAT_TEMPLATE_BYTES = len(TEST_CHAT_TEMPLATE)
TEST_CHAT_TEMPLATE_SHA256 = hashlib.sha256(TEST_CHAT_TEMPLATE).hexdigest()
GENERATOR_SHA256 = "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"
BENCHMARK_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"

FIXED_INPUT_FILES = {
    "cases.json",
    "short.txt",
    "generate_benchmark_prompt.py",
    "benchmark-32768.txt",
}
MATERIALIZED_PROMPT_FILES = {
    "swa-513.prompt",
    "yarn-8193.prompt",
    "deep-32768.prompt",
}
PROMOTED_VECTOR_FILES = {f"{case_id}.llama.f32" for case_id, _, _, _ in CASES}
PROMOTED_FILES = (
    FIXED_INPUT_FILES
    | MATERIALIZED_PROMPT_FILES
    | PROMOTED_VECTOR_FILES
    | {"yarn-8193.continuation.i32", "manifest.json"}
)
CAPTURE_ARTIFACT_FILES = (
    {
        artifact
        for case_id, _, _, _ in CASES
        for artifact in (
            f"{case_id}.prompt",
            f"{case_id}.tokens.i32",
            f"{case_id}.logits.f32",
        )
    }
    | {f"yarn-8193.step-{step:02d}.logits.f32" for step in range(8)}
    | {"yarn-8193.continuation.i32"}
)
assert len(CAPTURE_ARTIFACT_FILES) == 21

PROMPT_PREFIX_BYTES = {
    "swa-513": 4096,
    "yarn-8193": 65536,
    "deep-32768": 180000,
}


FAKE_DS4_TEMPLATE = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

argv = sys.argv[1:]
if (
    len(argv) != 6
    or argv[0:3] != ["--dump-tokens", "--raw-prompt", "-m"]
    or argv[4] != "--prompt-file"
    or not os.environ.get("FAKE_DS4_EXPECTED_MODEL")
    or argv[3] != os.environ["FAKE_DS4_EXPECTED_MODEL"]
):
    raise SystemExit(f"invalid synthetic ds4 argv: {argv!r}")
path = Path(argv[5])
payload = path.read_bytes()
payload_sha = hashlib.sha256(payload).hexdigest()
token_map = __TOKEN_MAP__
if payload_sha not in token_map:
    raise SystemExit(f"unrecognized synthetic prompt bytes: {path} sha256={payload_sha}")
record = {"filename": path.name, "sha256": payload_sha, "argv": argv}
call_number = 0
if os.environ.get("FAKE_DS4_LOG"):
    log_path = Path(os.environ["FAKE_DS4_LOG"])
    prior = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    call_number = len(prior) + 1
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
if call_number == int(os.environ.get("FAKE_DS4_MUTATE_ON_CALL", "0")):
    if os.environ.get("FAKE_DS4_DIRTY_REPO"):
        with (Path(os.environ["FAKE_DS4_DIRTY_REPO"]) / "README.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("tokenizer race dirty\n")
    if os.environ.get("FAKE_DS4_ADVANCE_HEAD_REPO"):
        repo = os.environ["FAKE_DS4_ADVANCE_HEAD_REPO"]
        environment = os.environ.copy()
        environment.update({
            "GIT_AUTHOR_NAME": "Laguna Race",
            "GIT_AUTHOR_EMAIL": "laguna-race@example.invalid",
            "GIT_COMMITTER_NAME": "Laguna Race",
            "GIT_COMMITTER_EMAIL": "laguna-race@example.invalid",
        })
        tree = subprocess.run(
            ["git", "-C", repo, "write-tree"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", repo, "commit-tree", tree, "-p", parent, "-m", "tokenizer race"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", repo, "update-ref", "HEAD", commit, parent],
            check=True,
            env=environment,
        )
kind, value = token_map[payload_sha]
tokens = value if kind == "tokens" else [index % 100352 for index in range(value)]
if os.environ.get("FAKE_DS4_EXTRA_SHA256") == payload_sha:
    tokens.append(23)
if os.environ.get("FAKE_DS4_MISMATCH_SHA256") == payload_sha:
    tokens[0] = (tokens[0] + 1) % 100352
print(tokens)
'''


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


def promoted_model() -> dict[str, object]:
    return {
        "repository": "poolside/Laguna-S-2.1-GGUF",
        "revision": GGUF_REVISION,
        "filename": GGUF_FILENAME,
        "size": GGUF_SIZE,
        "sha256": GGUF_SHA256,
    }


def capture_model() -> dict[str, object]:
    model = promoted_model()
    model["file"] = model.pop("filename")
    return model


def case_tokens(case_id: str, frontier: int) -> list[int]:
    if case_id == "short":
        return [100000, 17, 100001]
    return [index % VOCAB_SIZE for index in range(frontier)]


def short_rendered_prompt() -> bytes:
    return (
        b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
        + (FIXTURE_SOURCE / "short.txt").read_bytes()
        + b"</user>\n<assistant></think>"
    )


def template_reconciliation_prompt(*, think: bool) -> bytes:
    return (
        b"\xe3\x80\x88|EOS|\xe3\x80\x89"
        b"<system>Laguna template reconciliation probe.</system>\n<user>"
        + (FIXTURE_SOURCE / "short.txt").read_bytes()
        + b"</user>\n<assistant>"
        + (b"<think>" if think else b"</think>")
    )


def materialized_prompt(case_id: str) -> bytes:
    benchmark = (FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()
    return benchmark[: PROMPT_PREFIX_BYTES[case_id]]


def populate_fixed_inputs(root: Path) -> None:
    for name in sorted(FIXED_INPUT_FILES):
        shutil.copy2(FIXTURE_SOURCE / name, root / name)


def tree_digest(root: Path) -> str:
    """Hash names, entry types, link targets, and bytes without following links."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        mode = path.lstat().st_mode
        digest.update(relative)
        digest.update(b"\0")
        digest.update(stat.S_IMODE(mode).to_bytes(4, "little"))
        if stat.S_ISLNK(mode):
            digest.update(b"L")
            digest.update(os.readlink(path).encode())
        elif stat.S_ISDIR(mode):
            digest.update(b"D")
        elif stat.S_ISREG(mode):
            digest.update(b"F")
            digest.update(path.read_bytes())
        else:
            digest.update(b"X")
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def edit_manifest(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    write_json(path, manifest)


def write_promoted_fixture(root: Path) -> None:
    populate_fixed_inputs(root)
    for case_id in ("swa-513", "yarn-8193", "deep-32768"):
        (root / f"{case_id}.prompt").write_bytes(materialized_prompt(case_id))

    cases: list[dict[str, Any]] = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = short_rendered_prompt() if case_id == "short" else materialized_prompt(case_id)
        tokens = case_tokens(case_id, frontier)
        vector_name = f"{case_id}.llama.f32"
        vector_payload = f32_bytes(case_index + 3)
        (root / vector_name).write_bytes(vector_payload)
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
                    "file": vector_name,
                    "sha256": sha256(vector_payload),
                    "argmax": case_index + 3,
                },
            }
        )

    continuation_payload = i32_bytes(CONTINUATION)
    (root / "yarn-8193.continuation.i32").write_bytes(continuation_payload)
    manifest = {
        "schema": PROMOTED_SCHEMA,
        "oracle_policy": ORACLE_POLICY,
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": 8,
        "model": promoted_model(),
        "oracle": {
            "name": "poolside-llama",
            "runtime_commit": LLAMA_COMMIT,
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        },
        "provenance": {
            "contract_commit": CONTRACT_COMMIT,
            "tokenizer_runtime_commit": TOKENIZER_RUNTIME_COMMIT,
            "generator_sha256": GENERATOR_SHA256,
            "benchmark_sha256": BENCHMARK_SHA256,
            "poolside_seed_token_count": 61440,
            "ds4_seed_token_count": 61440,
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
            "file": "yarn-8193.continuation.i32",
            "sha256": sha256(continuation_payload),
            "argmax": CONTINUATION,
        },
    }
    write_json(root / "manifest.json", manifest)


def write_capture(root: Path) -> str:
    files: dict[str, str] = {}
    capture_cases: list[dict[str, Any]] = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = short_rendered_prompt() if case_id == "short" else materialized_prompt(case_id)
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
                "prompt": "short.txt" if case_id == "short" else prompt_name,
                "frontier": frontier,
                "context": context,
                "prompt_file": prompt_name,
                "tokens_file": tokens_name,
                "logits_file": logits_name,
                "token_count": frontier,
                "argmax": case_index + 3,
            }
        )

    step_files: list[str] = []
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

    assert set(files) == CAPTURE_ARTIFACT_FILES
    capture = {
        "schema": "laguna-resident-capture-v1",
        "oracle": "llama",
        "runtime_commit": LLAMA_COMMIT,
        "vocab_size": VOCAB_SIZE,
        "seed_token_count": 61440,
        "model": capture_model(),
        "cases": capture_cases,
        "continuation": {
            "case": "yarn-8193",
            "tokens_file": continuation_name,
            "logits_files": step_files,
            "argmax": CONTINUATION,
        },
        "files": files,
    }
    payload = (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode()
    (root / "capture.json").write_bytes(payload)
    return sha256(payload)


def rewrite_capture(root: Path, change: Callable[[dict[str, Any]], None]) -> str:
    path = root / "capture.json"
    capture = json.loads(path.read_text(encoding="utf-8"))
    change(capture)
    payload = (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return sha256(payload)


def upgrade_capture_to_v2(root: Path) -> str:
    template = TEST_CHAT_TEMPLATE
    artifacts = {
        "tokenizer.chat_template.jinja": template,
        "chat-template-think.prompt": template_reconciliation_prompt(think=True),
        "chat-template-nothink.prompt": template_reconciliation_prompt(think=False),
    }
    for name, payload in artifacts.items():
        (root / name).write_bytes(payload)

    def upgrade(capture: dict[str, Any]) -> None:
        capture["schema"] = "laguna-resident-capture-v2"
        capture["model"] = {
            "repository": "poolside/Laguna-S-2.1-GGUF",
            "revision": CORRECTED_GGUF_REVISION,
            "file": GGUF_FILENAME,
            "size": CORRECTED_GGUF_SIZE,
            "sha256": CORRECTED_GGUF_SHA256,
        }
        capture["chat_template"] = {
            "file": "tokenizer.chat_template.jinja",
            "bytes": len(template),
            "sha256": sha256(template),
            "render_contract": "pinned-template-semantics-v1",
            "reconciliation_system": "Laguna template reconciliation probe.",
            "think_prompt_file": "chat-template-think.prompt",
            "think_prompt_sha256": sha256(artifacts["chat-template-think.prompt"]),
            "nothink_prompt_file": "chat-template-nothink.prompt",
            "nothink_prompt_sha256": sha256(artifacts["chat-template-nothink.prompt"]),
        }
        capture["files"].update(
            {name: sha256(payload) for name, payload in artifacts.items()}
        )

    return rewrite_capture(root, upgrade)


def fake_ds4_source() -> str:
    token_map = {
        sha256(short_rendered_prompt()): ["tokens", case_tokens("short", 3)],
        sha256(materialized_prompt("swa-513")): ["count", 513],
        sha256(materialized_prompt("yarn-8193")): ["count", 8193],
        sha256(materialized_prompt("deep-32768")): ["count", 32768],
        sha256((FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()): ["count", 61440],
    }
    assert len(token_map) == 5
    return FAKE_DS4_TEMPLATE.replace(
        "__TOKEN_MAP__", json.dumps(token_map, sort_keys=True)
    )


def clean_ds4_repository(root: Path) -> tuple[Path, Path, str]:
    repo = root / "ds4-repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(repo)],
        check=True,
    )
    ds4 = repo / "fake-ds4"
    ds4.write_text(fake_ds4_source(), encoding="utf-8")
    ds4.chmod(0o755)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(head) == 40
    subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{CONTRACT_COMMIT}^{{commit}}"],
        check=True,
    )
    return repo, ds4, head


def load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_laguna_logits_under_test", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_promote(
    module: ModuleType,
    ds4: Path,
    capture: Path,
    destination: Path,
    capture_digest: str,
) -> None:
    signature = inspect.signature(module.promote)
    if "expected_capture_sha256" not in signature.parameters:
        raise AssertionError("v2 promote API lacks expected_capture_sha256 trust-anchor injection")
    model_argument = str(destination.parent / GGUF_FILENAME)
    with mock.patch.dict(
        os.environ,
        {
            "LAGUNA_MODEL": model_argument,
            "FAKE_DS4_EXPECTED_MODEL": model_argument,
        },
    ), mock.patch.object(
        module, "CHAT_TEMPLATE_BYTES", TEST_CHAT_TEMPLATE_BYTES
    ), mock.patch.object(
        module, "CHAT_TEMPLATE_SHA256", TEST_CHAT_TEMPLATE_SHA256
    ):
        module.promote(
            ds4,
            capture,
            destination,
            expected_capture_sha256=capture_digest,
        )


def run_promote_cli(
    module: ModuleType,
    ds4: Path,
    capture: Path,
    destination: Path,
    capture_digest: str,
    token_log: Path | None = None,
) -> SimpleNamespace:
    stdout = io.StringIO()
    stderr = io.StringIO()
    argv = [
        str(TOOL),
        "--ds4",
        str(ds4),
        "--llama",
        str(capture),
        "--promote",
        str(destination),
    ]
    model_argument = str(destination.parent / GGUF_FILENAME)
    environment = {
        "LAGUNA_MODEL": model_argument,
        "FAKE_DS4_EXPECTED_MODEL": model_argument,
    }
    if token_log is not None:
        environment["FAKE_DS4_LOG"] = str(token_log)
    with (
        mock.patch.object(module, "CAPTURE_MANIFEST_SHA256", capture_digest, create=True),
        mock.patch.object(module, "CHAT_TEMPLATE_BYTES", TEST_CHAT_TEMPLATE_BYTES),
        mock.patch.object(module, "CHAT_TEMPLATE_SHA256", TEST_CHAT_TEMPLATE_SHA256),
        mock.patch.object(sys, "argv", argv),
        mock.patch.dict(os.environ, environment),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            returncode = module.main()
        except SystemExit as exc:
            returncode = int(exc.code or 0)
    return SimpleNamespace(returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def make_promotion_workspace(
    root: Path, *, preexisting_prompts: bool = False
) -> tuple[Path, str, Path, Path, str, Path]:
    capture = root / "capture"
    capture.mkdir()
    write_capture(capture)
    capture_digest = upgrade_capture_to_v2(capture)
    repo, ds4, head = clean_ds4_repository(root)
    destination = root / "promoted"
    destination.mkdir()
    (root / GGUF_FILENAME).write_bytes(b"synthetic model argument placeholder")
    populate_fixed_inputs(destination)
    if preexisting_prompts:
        for name in sorted(MATERIALIZED_PROMPT_FILES):
            shutil.copy2(capture / name, destination / name)
    return capture, capture_digest, repo, ds4, head, destination


def make_unrelated_head(repo: Path) -> None:
    tree = subprocess.run(
        ["git", "-C", str(repo), "mktree"],
        input="",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Laguna Test",
            "GIT_AUTHOR_EMAIL": "laguna@example.invalid",
            "GIT_COMMITTER_NAME": "Laguna Test",
            "GIT_COMMITTER_EMAIL": "laguna@example.invalid",
        }
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "commit-tree", tree, "-m", "unrelated root"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach", commit], check=True)


class CompareLagunaLogitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool_module = load_tool_module()

    def test_gguf_fd_hash_uses_exact_bytes_without_moving_the_offset(self) -> None:
        payload = b"retained Laguna artifact\n" * 257
        with tempfile.TemporaryFile() as model:
            model.write(payload)
            model.flush()
            os.lseek(model.fileno(), 19, os.SEEK_SET)
            before = os.fstat(model.fileno())

            self.tool_module.verify_gguf_fd(
                model.fileno(), len(payload), sha256(payload)
            )

            after = os.fstat(model.fileno())
            self.assertEqual(os.lseek(model.fileno(), 0, os.SEEK_CUR), 19)
            self.assertEqual(
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
            )

    def test_gguf_fd_hash_rejects_size_and_digest_mismatch_without_moving_offset(
        self,
    ) -> None:
        payload = b"one opened model"
        cases = (
            (len(payload) + 1, sha256(payload), "size"),
            (len(payload), "0" * 64, "SHA-256"),
        )
        for expected_size, expected_digest, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), tempfile.TemporaryFile() as model:
                model.write(payload)
                model.flush()
                os.lseek(model.fileno(), 7, os.SEEK_SET)

                with self.assertRaises(self.tool_module.ContractError) as raised:
                    self.tool_module.verify_gguf_fd(
                        model.fileno(), expected_size, expected_digest
                    )

                self.assertIn(diagnostic, str(raised.exception))
                self.assertEqual(os.lseek(model.fileno(), 0, os.SEEK_CUR), 7)

    def test_gguf_fd_hash_rejects_non_regular_descriptors(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"not a model")
            with self.assertRaises(self.tool_module.ContractError) as raised:
                self.tool_module.verify_gguf_fd(
                    read_fd, len(b"not a model"), sha256(b"not a model")
                )
            self.assertIn("regular", str(raised.exception))
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_gguf_fd_hash_detects_identity_mutation_and_preserves_offset(self) -> None:
        payload = b"identity must remain stable"
        with tempfile.TemporaryFile() as model:
            model.write(payload)
            model.flush()
            os.lseek(model.fileno(), 5, os.SEEK_SET)
            real_pread = os.pread
            mutated = False

            def mutate_after_read(fd: int, count: int, offset: int) -> bytes:
                nonlocal mutated
                result = real_pread(fd, count, offset)
                if not mutated:
                    mutated = True
                    os.ftruncate(fd, len(payload) + 1)
                return result

            with (
                mock.patch.object(
                    self.tool_module.os, "pread", side_effect=mutate_after_read
                ),
                self.assertRaises(self.tool_module.ContractError) as raised,
            ):
                self.tool_module.verify_gguf_fd(
                    model.fileno(), len(payload), sha256(payload)
                )

            self.assertIn("identity changed", str(raised.exception))
            self.assertEqual(os.lseek(model.fileno(), 0, os.SEEK_CUR), 5)

    def verify_command(
        self,
        fixture: Path,
        tokenizer_commit: str = TOKENIZER_RUNTIME_COMMIT,
        capture_digest: str = CAPTURE_MANIFEST_SHA256,
        gguf_size: int = GGUF_SIZE,
        gguf_sha256: str = GGUF_SHA256,
    ) -> list[str]:
        return [
            sys.executable,
            str(TOOL),
            "--verify-promoted",
            str(fixture),
            "--contract-commit",
            CONTRACT_COMMIT,
            "--tokenizer-runtime-commit",
            tokenizer_commit,
            "--llama-commit",
            LLAMA_COMMIT,
            "--capture-manifest-sha256",
            capture_digest,
            "--gguf-size",
            str(gguf_size),
            "--gguf-sha256",
            gguf_sha256,
        ]

    def run_verify(
        self,
        fixture: Path,
        tokenizer_commit: str = TOKENIZER_RUNTIME_COMMIT,
        capture_digest: str = CAPTURE_MANIFEST_SHA256,
        gguf_size: int = GGUF_SIZE,
        gguf_sha256: str = GGUF_SHA256,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.verify_command(
                fixture,
                tokenizer_commit,
                capture_digest,
                gguf_size,
                gguf_sha256,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_verification_failure_without_mutation(self, fixture: Path) -> None:
        before = tree_digest(fixture)
        completed = self.run_verify(fixture)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("verification failed:", completed.stderr)
        self.assertEqual(tree_digest(fixture), before)

    def test_tokenizer_parser_accepts_the_real_ds4_detail_rows(self) -> None:
        output = (
            "[2, 97, 1437]\n"
            "     2  \u3008|EOS|\u3009\n"
            "    97  <\n"
            "  1437  user\n"
        )

        self.assertEqual(
            self.tool_module.parse_tokenizer_output(output, "short prompt"),
            [2, 97, 1437],
        )

    def test_capture_v2_binds_corrected_model_template_and_explicit_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture.mkdir()
            write_capture(capture)
            digest = upgrade_capture_to_v2(capture)

            with mock.patch.object(
                self.tool_module,
                "CHAT_TEMPLATE_BYTES",
                TEST_CHAT_TEMPLATE_BYTES,
            ), mock.patch.object(
                self.tool_module,
                "CHAT_TEMPLATE_SHA256",
                TEST_CHAT_TEMPLATE_SHA256,
            ):
                validated = self.tool_module.validate_capture(capture, digest)
            fixed_root = root / "fixed"
            fixed_root.mkdir()
            populate_fixed_inputs(fixed_root)
            self.tool_module.validate_capture_relationships(
                validated, self.tool_module.validate_fixture_inputs(fixed_root)
            )
            promoted = self.tool_module.build_manifest(
                validated, digest, TOKENIZER_RUNTIME_COMMIT, 61440
            )

            self.assertEqual(validated["manifest"]["model"]["revision"], CORRECTED_GGUF_REVISION)
            self.assertEqual(promoted["model"]["revision"], CORRECTED_GGUF_REVISION)
            self.assertEqual(
                validated["chat_template"]["embedded"],
                (capture / "tokenizer.chat_template.jinja").read_bytes(),
            )
            self.assertEqual(
                validated["chat_template"]["nothink_prompt"],
                template_reconciliation_prompt(think=False),
            )
            self.assertEqual(
                validated["chat_template"]["think_prompt"],
                template_reconciliation_prompt(think=True),
            )

    def test_capture_v2_rejects_legacy_identity_and_semantic_mode_tamper(self) -> None:
        mutations: tuple[tuple[str, Callable[[Path], str]], ...] = (
            (
                "legacy model",
                lambda capture: rewrite_capture(
                    capture,
                    lambda data: data.__setitem__("model", capture_model()),
                ),
            ),
            (
                "think rendering",
                lambda capture: self._rewrite_v2_prompt(
                    capture, "chat-template-think.prompt", b"wrong think prompt"
                ),
            ),
            (
                "no-think rendering",
                lambda capture: self._rewrite_v2_prompt(
                    capture, "chat-template-nothink.prompt", b"wrong no-think prompt"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                capture = Path(tmp)
                write_capture(capture)
                upgrade_capture_to_v2(capture)
                digest = mutate(capture)

                with (
                    mock.patch.object(
                        self.tool_module,
                        "CHAT_TEMPLATE_BYTES",
                        TEST_CHAT_TEMPLATE_BYTES,
                    ),
                    mock.patch.object(
                        self.tool_module,
                        "CHAT_TEMPLATE_SHA256",
                        TEST_CHAT_TEMPLATE_SHA256,
                    ),
                    self.assertRaises(self.tool_module.ContractError),
                ):
                    self.tool_module.validate_capture(capture, digest)

    def test_capture_v2_rejects_self_consistent_unpinned_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp)
            write_capture(capture)
            upgrade_capture_to_v2(capture)
            wrong = TEST_CHAT_TEMPLATE + b" " * (
                self.tool_module.CHAT_TEMPLATE_BYTES - len(TEST_CHAT_TEMPLATE)
            )
            digest = self._rewrite_v2_template(capture, wrong)

            with self.assertRaises(self.tool_module.ContractError) as raised:
                self.tool_module.validate_capture(capture, digest)

            self.assertIn("identity", str(raised.exception))

    def test_capture_v1_is_historical_only_and_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture-v1"
            destination = root / "destination"
            capture.mkdir()
            destination.mkdir()
            digest = write_capture(capture)
            before = tree_digest(destination)

            with self.assertRaises(self.tool_module.ContractError) as raised:
                call_promote(
                    self.tool_module,
                    root / "unused-ds4",
                    capture,
                    destination,
                    digest,
                )

            self.assertIn("capture-v2", str(raised.exception))
            self.assertEqual(tree_digest(destination), before)

    @staticmethod
    def _rewrite_v2_prompt(capture: Path, name: str, payload: bytes) -> str:
        (capture / name).write_bytes(payload)
        field = (
            "think_prompt_sha256"
            if name == "chat-template-think.prompt"
            else "nothink_prompt_sha256"
        )

        def update(data: dict[str, Any]) -> None:
            digest = sha256(payload)
            data["files"][name] = digest
            data["chat_template"][field] = digest

        return rewrite_capture(capture, update)

    @staticmethod
    def _rewrite_v2_template(capture: Path, payload: bytes) -> str:
        name = "tokenizer.chat_template.jinja"
        (capture / name).write_bytes(payload)

        def update(data: dict[str, Any]) -> None:
            digest = sha256(payload)
            data["files"][name] = digest
            data["chat_template"]["bytes"] = len(payload)
            data["chat_template"]["sha256"] = digest

        return rewrite_capture(capture, update)

    def test_valid_fixture_verifies_without_mutation_and_has_exact_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted"
            fixture.mkdir()
            write_promoted_fixture(fixture)
            self.assertEqual({path.name for path in fixture.iterdir()}, PROMOTED_FILES)
            self.assertEqual(len(PROMOTED_FILES), 13)
            self.assertTrue(all(path.is_file() and not path.is_symlink() for path in fixture.iterdir()))
            before = tree_digest(fixture)

            completed = self.run_verify(fixture)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout,
                f"verified={fixture} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(completed.stderr, "")
            self.assertEqual(tree_digest(fixture), before)

    def test_verify_fails_closed_for_every_v2_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            write_promoted_fixture(base)

            def manifest_change(change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
                return lambda root: edit_manifest(root, change)

            def nonfinite_vector(root: Path) -> None:
                vector = root / "deep-32768.llama.f32"
                payload = bytearray(vector.read_bytes())
                payload[400:404] = struct.pack("<f", float("nan"))
                vector.write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["cases"][3]["vector"].__setitem__("sha256", sha256(payload)),
                )

            def wrong_vector_size(root: Path) -> None:
                payload = b"short"
                (root / "swa-513.llama.f32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["cases"][1]["vector"].__setitem__("sha256", sha256(payload)),
                )

            def wrong_continuation_size(root: Path) -> None:
                payload = b"short"
                (root / "yarn-8193.continuation.i32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["continuation"].__setitem__("sha256", sha256(payload)),
                )

            def nonprefix_prompt(root: Path) -> None:
                payload = b"valid utf8 but not the deterministic benchmark prefix\n"
                (root / "swa-513.prompt").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: (
                        data["cases"][1].__setitem__("prompt_hex", payload.hex()),
                        data["cases"][1].__setitem__("prompt_sha256", sha256(payload)),
                    ),
                )

            def invalid_utf8_prompt(root: Path) -> None:
                payload = b"\xff"
                (root / "swa-513.prompt").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: (
                        data["cases"][1].__setitem__("prompt_hex", payload.hex()),
                        data["cases"][1].__setitem__("prompt_sha256", sha256(payload)),
                    ),
                )

            def step_zero_parity(root: Path) -> None:
                changed = [9, *CONTINUATION[1:]]
                payload = i32_bytes(changed)
                (root / "yarn-8193.continuation.i32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: (
                        data["continuation"].__setitem__("argmax", changed),
                        data["continuation"].__setitem__("sha256", sha256(payload)),
                    ),
                )

            def wrong_cases_input(root: Path) -> None:
                path = root / "cases.json"
                cases = json.loads(path.read_text(encoding="utf-8"))
                cases["schema"] = "wrong-schema"
                write_json(path, cases)

            def duplicate_key(root: Path) -> None:
                path = root / "manifest.json"
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    text.replace(
                        f'"schema": "{PROMOTED_SCHEMA}",',
                        f'"schema": "{PROMOTED_SCHEMA}",\n  "schema": "{PROMOTED_SCHEMA}",',
                        1,
                    ),
                    encoding="utf-8",
                )

            tamperers: list[tuple[str, Callable[[Path], None]]] = [
                ("promoted-v1 schema", manifest_change(lambda d: d.__setitem__("schema", "laguna-resident-promoted-v1"))),
                ("oracle policy", manifest_change(lambda d: d.__setitem__("oracle_policy", "dual-oracle-v1"))),
                ("missing file", lambda root: (root / "short.llama.f32").unlink()),
                ("extra file", lambda root: (root / "unexpected.bin").write_bytes(b"x")),
                ("directory", lambda root: (root / "unexpected-directory").mkdir()),
                ("pycache", lambda root: (root / "__pycache__").mkdir()),
                ("vector size", wrong_vector_size),
                ("vector hash", lambda root: (root / "short.llama.f32").write_bytes(b"X" + (root / "short.llama.f32").read_bytes()[1:])),
                ("vector argmax", manifest_change(lambda d: d["cases"][0]["vector"].__setitem__("argmax", 99))),
                ("vector filename", manifest_change(lambda d: d["cases"][0]["vector"].__setitem__("file", "other.f32"))),
                ("nonfinite vector", nonfinite_vector),
                ("model revision", manifest_change(lambda d: d["model"].__setitem__("revision", "0" * 40))),
                ("model filename", manifest_change(lambda d: d["model"].__setitem__("filename", "other.gguf"))),
                ("model size", manifest_change(lambda d: d["model"].__setitem__("size", GGUF_SIZE + 1))),
                ("model hash", manifest_change(lambda d: d["model"].__setitem__("sha256", "0" * 64))),
                ("runtime", manifest_change(lambda d: d["oracle"].__setitem__("runtime_commit", "0" * 40))),
                ("capture digest", manifest_change(lambda d: d["oracle"].__setitem__("capture_manifest_sha256", "0" * 64))),
                ("contract commit", manifest_change(lambda d: d["provenance"].__setitem__("contract_commit", "0" * 40))),
                ("tokenizer commit", manifest_change(lambda d: d["provenance"].__setitem__("tokenizer_runtime_commit", "f" * 40))),
                ("generator digest", manifest_change(lambda d: d["provenance"].__setitem__("generator_sha256", "0" * 64))),
                ("benchmark digest", manifest_change(lambda d: d["provenance"].__setitem__("benchmark_sha256", "0" * 64))),
                ("generator file bytes", lambda root: (root / "generate_benchmark_prompt.py").write_bytes((root / "generate_benchmark_prompt.py").read_bytes() + b"\n# tampered\n")),
                ("benchmark file bytes", lambda root: (root / "benchmark-32768.txt").write_bytes((root / "benchmark-32768.txt").read_bytes() + b"tampered")),
                ("cases input", wrong_cases_input),
                ("short input bytes", lambda root: (root / "short.txt").write_bytes((root / "short.txt").read_bytes() + b"tampered")),
                ("poolside seed count", manifest_change(lambda d: d["provenance"].__setitem__("poolside_seed_token_count", 61439))),
                ("ds4 seed count", manifest_change(lambda d: d["provenance"].__setitem__("ds4_seed_token_count", 32767))),
                ("prompt hex", manifest_change(lambda d: d["cases"][1].__setitem__("prompt_hex", "00"))),
                ("prompt hash", manifest_change(lambda d: d["cases"][1].__setitem__("prompt_sha256", "0" * 64))),
                ("prompt file bytes", lambda root: (root / "swa-513.prompt").write_bytes((root / "swa-513.prompt").read_bytes() + b"x")),
                ("invalid UTF-8 prompt", invalid_utf8_prompt),
                ("prompt not benchmark prefix", nonprefix_prompt),
                ("token mismatch", manifest_change(lambda d: d["cases"][1]["ds4_tokens"].__setitem__(0, 99))),
                ("out-of-range token", manifest_change(lambda d: (d["cases"][1]["poolside_tokens"].__setitem__(0, VOCAB_SIZE), d["cases"][1]["ds4_tokens"].__setitem__(0, VOCAB_SIZE)))),
                ("case order", manifest_change(lambda d: d["cases"].__setitem__(slice(0, 2), [d["cases"][1], d["cases"][0]]))),
                ("frontier", manifest_change(lambda d: d["cases"][1].__setitem__("frontier", 514))),
                ("context", manifest_change(lambda d: d["cases"][2].__setitem__("context", 8203))),
                ("continuation size", wrong_continuation_size),
                ("continuation hash", manifest_change(lambda d: d["continuation"].__setitem__("sha256", "0" * 64))),
                ("continuation argmax", manifest_change(lambda d: d["continuation"]["argmax"].__setitem__(1, 99))),
                ("continuation step zero", step_zero_parity),
                ("widened CUDA RMS", manifest_change(lambda d: d["thresholds"]["cuda_admission"].__setitem__("centered_rms", 0.041))),
                ("wrong threshold type", manifest_change(lambda d: d["thresholds"]["cuda_admission"].__setitem__("argmax_equal", 1))),
                ("extra manifest key", manifest_change(lambda d: d.__setitem__("extra", True))),
                ("duplicate JSON key", duplicate_key),
            ]

            for index, (label, tamper) in enumerate(tamperers):
                with self.subTest(label=label):
                    fixture = Path(tmp) / f"tampered-{index:02d}"
                    shutil.copytree(base, fixture)
                    tamper(fixture)
                    self.assert_verification_failure_without_mutation(fixture)

    def test_verify_rejects_allowlisted_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "promoted"
            fixture.mkdir()
            write_promoted_fixture(fixture)
            vector = fixture / "short.llama.f32"
            payload = vector.read_bytes()
            external = root / "external-short.llama.f32"
            external.write_bytes(payload)
            vector.unlink()
            vector.symlink_to(external)
            fixture_before = tree_digest(fixture)

            completed = self.run_verify(fixture)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("verification failed:", completed.stderr)
            self.assertEqual(tree_digest(fixture), fixture_before)
            self.assertTrue(external.is_file())
            self.assertEqual(external.read_bytes(), payload)

    def test_promotion_publishes_missing_prompts_and_round_trips_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, digest, _, ds4, head, destination = make_promotion_workspace(root)
            token_log = root / "tokenizer-invocations.log"

            completed = run_promote_cli(
                self.tool_module, ds4, capture, destination, digest, token_log
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout,
                f"promoted={destination} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(completed.stderr, "")
            self.assertEqual({path.name for path in destination.iterdir()}, PROMOTED_FILES)
            for name in MATERIALIZED_PROMPT_FILES:
                self.assertEqual((destination / name).read_bytes(), (capture / name).read_bytes())
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["provenance"]["tokenizer_runtime_commit"], head)
            invocations = [
                json.loads(line)
                for line in token_log.read_text(encoding="utf-8").splitlines()
            ]
            expected_invocations = {
                "short.prompt": sha256(short_rendered_prompt()),
                "swa-513.prompt": sha256(materialized_prompt("swa-513")),
                "yarn-8193.prompt": sha256(materialized_prompt("yarn-8193")),
                "deep-32768.prompt": sha256(materialized_prompt("deep-32768")),
                "benchmark-32768.txt": sha256(
                    (FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()
                ),
            }
            self.assertEqual(
                {(record["filename"], record["sha256"]) for record in invocations},
                set(expected_invocations.items()),
            )
            self.assertEqual(len(invocations), len(expected_invocations))
            expected_model_argument = str(root / GGUF_FILENAME)
            for record in invocations:
                argv = record["argv"]
                self.assertEqual(argv[:3], ["--dump-tokens", "--raw-prompt", "-m"])
                self.assertEqual(argv[3], expected_model_argument)
                self.assertEqual(argv[4], "--prompt-file")
                self.assertEqual(Path(argv[5]).name, record["filename"])

            before = tree_digest(destination)
            verified = self.run_verify(
                destination,
                head,
                digest,
                CORRECTED_GGUF_SIZE,
                CORRECTED_GGUF_SHA256,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(
                verified.stdout,
                f"verified={destination} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(tree_digest(destination), before)

    def test_promotion_rejects_ds4_token_mismatch_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            before = tree_digest(destination)

            with (
                mock.patch.dict(
                    os.environ,
                    {"FAKE_DS4_MISMATCH_SHA256": sha256(materialized_prompt("swa-513"))},
                ),
                self.assertRaises(self.tool_module.ContractError),
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(tree_digest(destination), before)

    def test_promotion_derives_the_short_frontier_from_the_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, _, _, ds4, head, destination = make_promotion_workspace(Path(tmp))
            short_tokens = [*case_tokens("short", 3), 23]
            short_tokens_payload = i32_bytes(short_tokens)
            (capture / "short.tokens.i32").write_bytes(short_tokens_payload)

            def extend_short_frontier(data: dict[str, Any]) -> None:
                data["cases"][0]["frontier"] = len(short_tokens)
                data["cases"][0]["token_count"] = len(short_tokens)
                data["files"]["short.tokens.i32"] = sha256(short_tokens_payload)

            digest = rewrite_capture(capture, extend_short_frontier)
            with mock.patch.dict(
                os.environ,
                {"FAKE_DS4_EXTRA_SHA256": sha256(short_rendered_prompt())},
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cases"][0]["frontier"], len(short_tokens))
            self.assertEqual(manifest["cases"][0]["poolside_tokens"], short_tokens)
            verified = self.run_verify(
                destination,
                head,
                digest,
                CORRECTED_GGUF_SIZE,
                CORRECTED_GGUF_SHA256,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_promotion_rejects_surplus_ds4_tokens_for_an_exact_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            before = tree_digest(destination)

            with (
                mock.patch.dict(
                    os.environ,
                    {"FAKE_DS4_EXTRA_SHA256": sha256(short_rendered_prompt())},
                ),
                self.assertRaises(self.tool_module.ContractError),
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(tree_digest(destination), before)

    def test_promotion_accepts_byte_exact_existing_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(
                Path(tmp), preexisting_prompts=True
            )
            before_prompts = {
                name: (destination / name).read_bytes() for name in MATERIALIZED_PROMPT_FILES
            }

            call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(
                {name: (destination / name).read_bytes() for name in MATERIALIZED_PROMPT_FILES},
                before_prompts,
            )

    def test_promotion_rejects_mismatched_existing_prompt_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(
                Path(tmp), preexisting_prompts=True
            )
            (destination / "swa-513.prompt").write_bytes(b"mismatch")
            before = tree_digest(destination)

            with self.assertRaises(self.tool_module.ContractError) as raised:
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertIn("prompt", str(raised.exception).lower())
            self.assertEqual(tree_digest(destination), before)

    def test_capture_trust_boundary_failures_do_not_publish(self) -> None:
        def extra_capture_key(capture: Path, digest: str) -> str:
            return rewrite_capture(capture, lambda data: data.__setitem__("extra", True))

        def extra_case_key(capture: Path, digest: str) -> str:
            return rewrite_capture(capture, lambda data: data["cases"][0].__setitem__("extra", True))

        def unsafe_filename(capture: Path, digest: str) -> str:
            return rewrite_capture(
                capture,
                lambda data: data["files"].__setitem__(
                    "../unsafe", data["files"].pop("short.prompt")
                ),
            )

        def self_consistent_extra_artifact(capture: Path, digest: str) -> str:
            payload = b"extra but internally hashed"
            (capture / "unexpected.bin").write_bytes(payload)
            return rewrite_capture(
                capture,
                lambda data: data["files"].__setitem__("unexpected.bin", sha256(payload)),
            )

        def changed_file(capture: Path, digest: str) -> str:
            (capture / "short.prompt").write_bytes(b"changed")
            return digest

        def wrong_runtime(capture: Path, digest: str) -> str:
            return rewrite_capture(
                capture, lambda data: data.__setitem__("runtime_commit", "0" * 40)
            )

        def wrong_model(capture: Path, digest: str) -> str:
            return rewrite_capture(
                capture, lambda data: data["model"].__setitem__("revision", "0" * 40)
            )

        def wrong_seed_count(capture: Path, digest: str) -> str:
            return rewrite_capture(capture, lambda data: data.__setitem__("seed_token_count", 61439))

        def deterministic_prompt_mismatch(capture: Path, digest: str) -> str:
            path = capture / "swa-513.prompt"
            payload = b"x" * len(path.read_bytes())
            path.write_bytes(payload)
            return rewrite_capture(
                capture,
                lambda data: data["files"].__setitem__("swa-513.prompt", sha256(payload)),
            )

        def corrupted_step_logits(capture: Path, digest: str) -> str:
            name = "yarn-8193.step-00.logits.f32"
            payload = f32_bytes(99)
            (capture / name).write_bytes(payload)
            return rewrite_capture(
                capture, lambda data: data["files"].__setitem__(name, sha256(payload))
            )

        scenarios: list[
            tuple[str, Callable[[Path, str], str]]
        ] = [
            ("stale trust-anchor digest", lambda capture, digest: "0" * 64),
            ("unknown capture key", extra_capture_key),
            ("extra case key", extra_case_key),
            ("unsafe capture filename", unsafe_filename),
            ("self-consistent 22nd artifact", self_consistent_extra_artifact),
            ("changed referenced artifact", changed_file),
            ("wrong runtime pin", wrong_runtime),
            ("wrong model pin", wrong_model),
            ("wrong exact seed count", wrong_seed_count),
            ("prompt no longer deterministic prefix", deterministic_prompt_mismatch),
            ("step-00 logits disagree with recorded argmax", corrupted_step_logits),
        ]

        for index, (label, mutate) in enumerate(scenarios):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
                expected_digest = mutate(capture, digest)
                before = tree_digest(destination)

                with self.assertRaises(self.tool_module.ContractError):
                    call_promote(self.tool_module, ds4, capture, destination, expected_digest)

                self.assertEqual(tree_digest(destination), before)
                self.assertEqual({path.name for path in destination.iterdir()}, FIXED_INPUT_FILES)

    def test_capture_rejects_allowlisted_artifact_symlink_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, digest, _, ds4, _, destination = make_promotion_workspace(root)
            artifact = capture / "short.logits.f32"
            payload = artifact.read_bytes()
            external = root / "external-short.logits.f32"
            external.write_bytes(payload)
            artifact.unlink()
            artifact.symlink_to(external)
            capture_before = tree_digest(capture)

            with self.assertRaises(self.tool_module.ContractError):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(tree_digest(capture), capture_before)
            self.assertTrue(external.is_file())
            self.assertEqual(external.read_bytes(), payload)
            self.assertEqual({path.name for path in destination.iterdir()}, FIXED_INPUT_FILES)

    def test_capture_digest_is_checked_before_malformed_json_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, _, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            (capture / "capture.json").write_bytes(b"{ definitely not JSON")
            before = tree_digest(destination)

            with self.assertRaises(self.tool_module.ContractError) as raised:
                call_promote(self.tool_module, ds4, capture, destination, "0" * 64)

            self.assertEqual(
                str(raised.exception),
                "Poolside capture.json trust-anchor SHA-256 mismatch",
            )
            self.assertEqual(tree_digest(destination), before)

    def test_capture_manifest_and_artifacts_are_each_consumed_from_one_read(self) -> None:
        def semantically_changed_capture(original: bytes) -> bytes:
            capture = json.loads(original.decode("utf-8"))
            capture["seed_token_count"] = 61439
            return (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode()

        scenarios = (
            ("capture.json", semantically_changed_capture),
            ("swa-513.logits.f32", lambda original: f32_bytes(99)),
        )
        for target_name, replacement_fn in scenarios:
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture, digest, _, ds4, head, destination = make_promotion_workspace(root)
                target = capture / target_name
                original = target.read_bytes()
                original_vector = (capture / "swa-513.logits.f32").read_bytes()
                replacement = replacement_fn(original)
                real_read_bytes = self.tool_module.read_bytes
                target_reads = 0

                def swapping_read_bytes(path: Path) -> bytes:
                    nonlocal target_reads
                    payload = real_read_bytes(path)
                    if Path(path) == target:
                        target_reads += 1
                        if target_reads == 1:
                            target.write_bytes(replacement)
                    return payload

                with mock.patch.object(
                    self.tool_module, "read_bytes", side_effect=swapping_read_bytes
                ):
                    call_promote(self.tool_module, ds4, capture, destination, digest)

                self.assertEqual(target_reads, 1)
                self.assertEqual({path.name for path in destination.iterdir()}, PROMOTED_FILES)
                self.assertEqual(
                    (destination / "swa-513.llama.f32").read_bytes(), original_vector
                )
                verified = self.run_verify(
                    destination,
                    head,
                    digest,
                    CORRECTED_GGUF_SIZE,
                    CORRECTED_GGUF_SHA256,
                )
                self.assertEqual(verified.returncode, 0, verified.stderr)
                self.assertFalse(destination.with_name(f".{destination.name}.lock").exists())
                self.assertFalse(
                    any(
                        path.name.startswith(f".{destination.name}.tmp-")
                        for path in destination.parent.iterdir()
                    )
                )

    def test_repository_provenance_failures_do_not_publish(self) -> None:
        with self.subTest(label="dirty tracked checkout"), tempfile.TemporaryDirectory() as tmp:
            capture, digest, repo, ds4, _, destination = make_promotion_workspace(Path(tmp))
            (repo / "README.md").write_text("dirty\n", encoding="utf-8")
            before = tree_digest(destination)
            with self.assertRaises(self.tool_module.ContractError):
                call_promote(self.tool_module, ds4, capture, destination, digest)
            self.assertEqual(tree_digest(destination), before)

        with self.subTest(label="unresolved contract"), tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            before = tree_digest(destination)
            with (
                mock.patch.object(self.tool_module, "CONTRACT_COMMIT", "f" * 40, create=True),
                self.assertRaises(self.tool_module.ContractError),
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)
            self.assertEqual(tree_digest(destination), before)

        with self.subTest(label="contract not ancestor"), tempfile.TemporaryDirectory() as tmp:
            capture, digest, repo, ds4, _, destination = make_promotion_workspace(Path(tmp))
            make_unrelated_head(repo)
            before = tree_digest(destination)
            with self.assertRaises(self.tool_module.ContractError):
                call_promote(self.tool_module, ds4, capture, destination, digest)
            self.assertEqual(tree_digest(destination), before)

        with self.subTest(label="malformed non-full HEAD"), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, digest, _, ds4, _, destination = make_promotion_workspace(root)
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            quoted_git = shlex.quote(str(real_git))
            git = fake_bin / "git"
            git.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *\"rev-parse HEAD\"*) printf 'abc\\n'; exit 0 ;;\n"
                "esac\n"
                f"exec {quoted_git} \"$@\"\n",
                encoding="utf-8",
            )
            git.chmod(0o755)
            before = tree_digest(destination)
            with (
                mock.patch.dict(os.environ, {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}),
                self.assertRaises(self.tool_module.ContractError),
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)
            self.assertEqual(tree_digest(destination), before)

    def test_repository_provenance_is_rechecked_after_tokenization(self) -> None:
        scenarios = (
            ("tracked file dirtied", "FAKE_DS4_DIRTY_REPO"),
            ("HEAD advanced", "FAKE_DS4_ADVANCE_HEAD_REPO"),
        )
        for label, mutation_variable in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture, digest, repo, ds4, original_head, destination = (
                    make_promotion_workspace(root)
                )
                token_log = root / "race-tokenizer.log"
                environment = {
                    "FAKE_DS4_LOG": str(token_log),
                    "FAKE_DS4_MUTATE_ON_CALL": "5",
                    mutation_variable: str(repo),
                }

                with (
                    mock.patch.dict(os.environ, environment),
                    self.assertRaises(self.tool_module.ContractError),
                ):
                    call_promote(self.tool_module, ds4, capture, destination, digest)

                self.assertEqual(len(token_log.read_text(encoding="utf-8").splitlines()), 5)
                current_head = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                tracked_status = subprocess.run(
                    ["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                if mutation_variable == "FAKE_DS4_DIRTY_REPO":
                    self.assertEqual(current_head, original_head)
                    self.assertNotEqual(tracked_status, "")
                else:
                    self.assertNotEqual(current_head, original_head)
                    self.assertEqual(tracked_status, "")
                self.assertEqual(
                    {path.name for path in destination.iterdir()}, FIXED_INPUT_FILES
                )
                self.assertFalse(destination.with_name(f".{destination.name}.lock").exists())
                self.assertFalse(
                    any(
                        path.name.startswith(f".{destination.name}.tmp-")
                        for path in destination.parent.iterdir()
                    )
                )

    def test_no_clobber_and_stale_partial_publication(self) -> None:
        for name in ("short.llama.f32", "yarn-8193.continuation.i32", "manifest.json"):
            with self.subTest(existing=name), tempfile.TemporaryDirectory() as tmp:
                capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
                (destination / name).write_bytes(b"sentinel")
                before = tree_digest(destination)
                with self.assertRaises(self.tool_module.ContractError):
                    call_promote(self.tool_module, ds4, capture, destination, digest)
                self.assertEqual(tree_digest(destination), before)

        with self.subTest(existing="stale partial set"), tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            stale = ["short.llama.f32", "yarn-8193.continuation.i32"]
            for name in stale:
                (destination / name).write_bytes(b"stale")
            before = tree_digest(destination)
            completed = run_promote_cli(self.tool_module, ds4, capture, destination, digest)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout, "")
            self.assertIn("promotion failed:", completed.stderr)
            for name in stale:
                self.assertIn(str(destination / name), completed.stderr)
            self.assertEqual(tree_digest(destination), before)

    def test_held_sibling_lock_rejects_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, _, destination = make_promotion_workspace(Path(tmp))
            lock = destination.with_name(f".{destination.name}.lock")
            lock.mkdir()
            before = tree_digest(destination)

            with self.assertRaises(self.tool_module.ContractError):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(tree_digest(destination), before)
            self.assertTrue(lock.is_dir())

    def test_two_concurrent_promoters_have_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture, digest, _, ds4, head, destination = make_promotion_workspace(Path(tmp))
            sibling_lock = destination.with_name(f".{destination.name}.lock")
            acquisition_barrier = threading.Barrier(2)
            attempts_guard = threading.Lock()
            lock_attempts = 0
            real_mkdir = os.mkdir

            def synchronized_mkdir(path: os.PathLike[str], *args: Any, **kwargs: Any) -> None:
                nonlocal lock_attempts
                if Path(path) == sibling_lock:
                    with attempts_guard:
                        lock_attempts += 1
                    acquisition_barrier.wait(timeout=5)
                real_mkdir(path, *args, **kwargs)

            def attempt() -> tuple[str, str]:
                try:
                    call_promote(self.tool_module, ds4, capture, destination, digest)
                    return "winner", ""
                except self.tool_module.ContractError as exc:
                    return "loser", str(exc)

            with (
                mock.patch.object(self.tool_module.os, "mkdir", side_effect=synchronized_mkdir),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                results = list(pool.map(lambda _: attempt(), range(2)))

            self.assertEqual(lock_attempts, 2)
            self.assertEqual(sorted(result for result, _ in results), ["loser", "winner"])
            loser_message = next(message for result, message in results if result == "loser")
            self.assertIn("lock", loser_message.lower())
            self.assertEqual({path.name for path in destination.iterdir()}, PROMOTED_FILES)
            verified = self.run_verify(
                destination,
                head,
                digest,
                CORRECTED_GGUF_SIZE,
                CORRECTED_GGUF_SHA256,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_link_failure_cleans_only_invocation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, digest, _, ds4, _, destination = make_promotion_workspace(
                root, preexisting_prompts=True
            )
            parent_entries = {path.name for path in root.iterdir()}
            preserved = {
                name: (destination / name).read_bytes()
                for name in FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
            }
            real_link = os.link
            link_count = 0
            published_targets: list[Path] = []
            external_sentinel = b"external replacement must survive cleanup"

            def failing_link(
                source: os.PathLike[str],
                target: os.PathLike[str],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                nonlocal link_count
                target_path = Path(target)
                if target_path.parent != destination:
                    real_link(source, target, *args, **kwargs)
                    return
                link_count += 1
                if link_count == 3:
                    raise OSError("forced test link failure")
                real_link(source, target, *args, **kwargs)
                published_targets.append(target_path)
                if link_count == 2:
                    published_targets[0].unlink()
                    published_targets[0].write_bytes(external_sentinel)

            try:
                with mock.patch.object(self.tool_module.os, "link", side_effect=failing_link):
                    call_promote(self.tool_module, ds4, capture, destination, digest)
            except AssertionError:
                raise
            except Exception:
                pass
            else:
                self.fail("forced os.link failure did not abort promotion")

            self.assertEqual(link_count, 3)
            self.assertEqual(len(published_targets), 2)
            self.assertEqual(
                {
                    name: (destination / name).read_bytes()
                    for name in FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
                },
                preserved,
            )
            self.assertEqual(published_targets[0].read_bytes(), external_sentinel)
            self.assertFalse(published_targets[1].exists())
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                FIXED_INPUT_FILES
                | MATERIALIZED_PROMPT_FILES
                | {published_targets[0].name},
            )
            self.assertFalse(destination.with_name(f".{destination.name}.lock").exists())
            self.assertEqual({path.name for path in root.iterdir()}, parent_entries)

    def test_link_cleanup_never_claims_a_pre_return_external_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, digest, _, ds4, _, destination = make_promotion_workspace(
                root, preexisting_prompts=True
            )
            real_link = os.link
            link_count = 0
            replaced_target: Path | None = None
            external_sentinel = b"replacement installed before os.link returns"

            def replacing_link(
                source: os.PathLike[str],
                target: os.PathLike[str],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                nonlocal link_count, replaced_target
                target_path = Path(target)
                if target_path.parent != destination:
                    real_link(source, target, *args, **kwargs)
                    return
                link_count += 1
                if link_count == 2:
                    raise OSError("forced failure after external replacement")
                real_link(source, target, *args, **kwargs)
                replaced_target = target_path
                target_path.unlink()
                target_path.write_bytes(external_sentinel)

            with (
                mock.patch.object(self.tool_module.os, "link", side_effect=replacing_link),
                self.assertRaises(self.tool_module.ContractError),
            ):
                call_promote(self.tool_module, ds4, capture, destination, digest)

            self.assertEqual(link_count, 2)
            self.assertIsNotNone(replaced_target)
            assert replaced_target is not None
            self.assertEqual(replaced_target.read_bytes(), external_sentinel)
            self.assertFalse(destination.with_name(f".{destination.name}.lock").exists())
            self.assertFalse(
                any(
                    path.name.startswith(f".{destination.name}.tmp-")
                    for path in destination.parent.iterdir()
                )
            )

    def test_legacy_cli_arguments_are_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            write_promoted_fixture(fixture)
            for flag in ("--metal", "--dwarfstar-commit", "--ds4-commit"):
                with self.subTest(flag=flag):
                    before = tree_digest(fixture)
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(TOOL),
                            "--promote",
                            str(fixture),
                            flag,
                            "legacy-value",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(
                        completed.stderr.splitlines()[-1],
                        "compare_laguna_logits.py: error: unrecognized arguments: "
                        f"{flag} legacy-value",
                    )
                    self.assertEqual(tree_digest(fixture), before)


if __name__ == "__main__":
    unittest.main()
