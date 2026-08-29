#!/usr/bin/env python3
"""RED contract tests for the single-Poolside Laguna fixture migration."""

from __future__ import annotations

import array
import contextlib
from collections import Counter
import errno
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


HERE = Path(__file__).resolve().parent
TOOL = HERE / "compare_laguna_logits.py"
ROOT = HERE.parents[1]
FIXTURE_SOURCE = ROOT / "tests/test-vectors/laguna-resident"

FIXED_INPUT_FILES = frozenset(
    {
        "cases.json",
        "short.txt",
        "generate_benchmark_prompt.py",
        "benchmark-32768.txt",
    }
)
MATERIALIZED_PROMPT_FILES = frozenset(
    {"swa-513.prompt", "yarn-8193.prompt", "deep-32768.prompt"}
)
CASES = (
    ("short", "laguna-ds4", 3, 1024),
    ("swa-513", "raw", 513, 1024),
    ("yarn-8193", "raw", 8193, 8202),
    ("deep-32768", "raw", 32768, 32768),
)
PROMOTED_VECTOR_FILES = frozenset(
    {
        "short.llama.f32",
        "swa-513.llama.f32",
        "yarn-8193.llama.f32",
        "deep-32768.llama.f32",
    }
)
PROMOTED_OUTPUT_FILES = frozenset(
    MATERIALIZED_PROMPT_FILES
    | PROMOTED_VECTOR_FILES
    | {"yarn-8193.continuation.i32", "manifest.json"}
)
PROMOTED_FILES = frozenset(FIXED_INPUT_FILES | PROMOTED_OUTPUT_FILES)

VOCAB_SIZE = 100352
VECTOR_BYTES = VOCAB_SIZE * 4
CONTINUATION_TOKENS = 8
CONTINUATION_BYTES = CONTINUATION_TOKENS * 4
CONTRACT_COMMIT = "a250e43722945e293f6044bc7254c4806d5a7912"
TOKENIZER_RUNTIME_COMMIT = "0123456789abcdef0123456789abcdef01234567"
LLAMA_COMMIT = "04b2b72cb54048ead292884adbe11f284e3ec950"
CAPTURE_MANIFEST_SHA256 = (
    "cc4fb338556c0895ff985edb5435ae7801be7dcb98c2958dc96a56d34f2c848e"
)
ORACLE_POLICY = "single-poolside-v1"
PROMOTED_SCHEMA = "laguna-resident-promoted-v2"
GENERATOR_SHA256 = "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"
BENCHMARK_SHA256 = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"
GGUF_REVISION = "706fa69799926b6afde1af9e24ca2a4923f110a1"
GGUF_SIZE = 68248759648
GGUF_SHA256 = "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a"
CONTINUATION = [5, 42, 43, 44, 45, 46, 47, 48]
CUDA_LIMITS = {
    "centered_rms": 0.04,
    "centered_max_abs": 0.20,
    "top20_overlap": 18,
    "argmax_equal": True,
    "continuation_equal": True,
}

# The capture-v1 shape has four rows per case (prompt, tokens, logits) and
# nine continuation artifacts.  capture.json itself is the trust anchor and
# is deliberately not part of this 21-artifact allowlist.
CAPTURE_ARTIFACT_FILES = frozenset(
    {
        "short.prompt",
        "swa-513.prompt",
        "yarn-8193.prompt",
        "deep-32768.prompt",
        "short.tokens.i32",
        "swa-513.tokens.i32",
        "yarn-8193.tokens.i32",
        "deep-32768.tokens.i32",
        "short.logits.f32",
        "swa-513.logits.f32",
        "yarn-8193.logits.f32",
        "deep-32768.logits.f32",
        "yarn-8193.continuation.i32",
        "yarn-8193.step-00.logits.f32",
        "yarn-8193.step-01.logits.f32",
        "yarn-8193.step-02.logits.f32",
        "yarn-8193.step-03.logits.f32",
        "yarn-8193.step-04.logits.f32",
        "yarn-8193.step-05.logits.f32",
        "yarn-8193.step-06.logits.f32",
        "yarn-8193.step-07.logits.f32",
    }
)

# Byte lengths are intentionally small synthetic prefixes.  They are still
# deterministic and are exact prefixes of the immutable benchmark seed.
PROMPT_PREFIX_BYTES = {"swa-513": 128, "yarn-8193": 256, "deep-32768": 512}
SHORT_RENDERED_PROMPT = b"\xe3\x80\x88|EOS|\xe3\x80\x89<user>"
SHORT_RENDERED_SUFFIX = b"</user>\n<assistant></think>"

FAKE_DS4 = r"""#!/usr/bin/env python3
import hashlib
import os
import sys
from pathlib import Path

VOCAB_SIZE = 100352
# Dispatch is by the exact rendered payload digest.  Basenames are deliberately
# ignored so promotion cannot accidentally tokenize a different byte payload.
PROMPT_COUNTS = {
    "e3dc84f54d1afb86e11f782bb9d19715340855df5ef602cde682768008aef74d": 3,
    "17fe521315d24902e6b34efb26a04c5dcaa395f56a0572078a4db1f40e84a490": 513,
    "92f6c7f1817119f7a390e55b5ce667db688924aa4d181c14a4921bb166ae28b5": 8193,
    "99fdb470b5b2afe3a23f24e7868383a020b5238da618bf2495dda327d866f568": 32768,
}
BENCHMARK_DIGEST = "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"


def log_call(digest, payload, values):
    log_name = os.environ.get("FAKE_DS4_LOG")
    if not log_name:
        return
    with open(log_name, "a", encoding="utf-8") as handle:
        handle.write(f"{digest} {len(payload)} {len(values)} {values[0]} {values[-1]}\n")


def retarget_destination_after_tokenization(prompt_digest):
    trigger_digest = os.environ.get("FAKE_DS4_RETARGET_ON_DIGEST")
    destination_name = os.environ.get("FAKE_DS4_RETARGET_DESTINATION")
    replacement_name = os.environ.get("FAKE_DS4_RETARGET_REPLACEMENT")
    original_name = os.environ.get("FAKE_DS4_RETARGET_ORIGINAL")
    if (
        not trigger_digest
        or prompt_digest != trigger_digest
        or not destination_name
        or not replacement_name
        or not original_name
    ):
        return
    destination = Path(destination_name)
    replacement = Path(replacement_name)
    original = Path(original_name)
    # The destination and replacement start as two equal directories.  The
    # first matching DS4 call atomically retargets the requested path; later
    # calls see `original` and no longer repeat the operation.
    if destination.exists() and replacement.exists() and not original.exists():
        destination.rename(original)
        replacement.rename(destination)


def dirty_tracked_checkout_after_tokenization(prompt_digest):
    target_name = os.environ.get("FAKE_DS4_DIRTY_TRACKED")
    if not target_name or prompt_digest != BENCHMARK_DIGEST:
        return
    target = Path(target_name)
    marker = b"\n# fake-ds4 late tracked mutation\n"
    payload = target.read_bytes()
    if marker not in payload:
        target.write_bytes(payload + marker)


def main():
    args = sys.argv[1:]
    try:
        prompt_path = Path(args[args.index("--prompt-file") + 1])
    except (ValueError, IndexError):
        print("missing --prompt-file", file=sys.stderr)
        return 2
    payload = prompt_path.read_bytes()
    prompt_digest = hashlib.sha256(payload).hexdigest()
    if prompt_digest == os.environ.get("FAKE_DS4_FAIL_PROMPT_DIGEST"):
        print("deterministic fake ds4 stderr", file=sys.stderr)
        return 29
    if prompt_digest == os.environ.get("FAKE_DS4_MALFORMED_PROMPT_DIGEST"):
        print("{\"not\": \"a token list\"}")
        return 0
    if prompt_digest == BENCHMARK_DIGEST:
        values = [index % VOCAB_SIZE for index in range(32768)]
        if os.environ.get("FAKE_DS4_BAD_BENCHMARK") == "1":
            values[0] = 999
    elif prompt_digest in PROMPT_COUNTS:
        count = PROMPT_COUNTS[prompt_digest]
        values = [100000, 17, 100001] if count == 3 else [index % VOCAB_SIZE for index in range(count)]
    else:
        print(f"unexpected prompt content digest: {prompt_digest}", file=sys.stderr)
        return 3
    retarget_destination_after_tokenization(prompt_digest)
    dirty_tracked_checkout_after_tokenization(prompt_digest)
    log_call(prompt_digest, payload, values)
    print(values)
    return 0

raise SystemExit(main())
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32_bytes(argmax: int) -> bytes:
    values = array.array("f", [0.0]) * VOCAB_SIZE
    values[argmax] = 2.0
    if values.itemsize != 4:
        raise AssertionError("test host does not expose four-byte float32")
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    assert len(payload) == VECTOR_BYTES
    return payload


def i32_bytes(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def capture_model() -> dict[str, object]:
    return {
        "repository": "poolside/Laguna-S-2.1-GGUF",
        "revision": GGUF_REVISION,
        "file": "laguna-s-2.1-Q4_K_M.gguf",
        "size": GGUF_SIZE,
        "sha256": GGUF_SHA256,
    }


def promoted_model() -> dict[str, object]:
    return {
        "repository": "poolside/Laguna-S-2.1-GGUF",
        "revision": GGUF_REVISION,
        "filename": "laguna-s-2.1-Q4_K_M.gguf",
        "size": GGUF_SIZE,
        "sha256": GGUF_SHA256,
    }


def case_tokens(case_id: str, frontier: int) -> list[int]:
    if case_id == "short":
        return [100000, 17, 100001]
    return [index % VOCAB_SIZE for index in range(frontier)]


def rendered_prompt(case_id: str, root: Path) -> bytes:
    if case_id == "short":
        return SHORT_RENDERED_PROMPT + (root / "short.txt").read_bytes() + SHORT_RENDERED_SUFFIX
    benchmark = (root / "benchmark-32768.txt").read_bytes()
    return benchmark[:PROMPT_PREFIX_BYTES[case_id]]


def source_rendered_prompt(case_id: str) -> bytes:
    if case_id == "short":
        return (
            SHORT_RENDERED_PROMPT
            + (FIXTURE_SOURCE / "short.txt").read_bytes()
            + SHORT_RENDERED_SUFFIX
        )
    benchmark = (FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()
    return benchmark[:PROMPT_PREFIX_BYTES[case_id]]


def populate_fixed_inputs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in FIXED_INPUT_FILES:
        shutil.copy2(FIXTURE_SOURCE / name, root / name)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def edit_manifest(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / "manifest.json"
    manifest = read_manifest(root)
    change(manifest)
    write_json(path, manifest)


def write_promoted_fixture(root: Path) -> None:
    """Write exactly the normative v2 fixture, without any Metal artifacts."""
    populate_fixed_inputs(root)
    for case_id in ("swa-513", "yarn-8193", "deep-32768"):
        (root / f"{case_id}.prompt").write_bytes(rendered_prompt(case_id, root))

    cases: list[dict[str, object]] = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = rendered_prompt(case_id, root)
        tokens = case_tokens(case_id, frontier)
        payload = f32_bytes(case_index + 3)
        vector_name = f"{case_id}.llama.f32"
        (root / vector_name).write_bytes(payload)
        cases.append(
            {
                "id": case_id,
                "render": render,
                "prompt_hex": prompt.hex(),
                "prompt_sha256": sha256(prompt),
                "poolside_tokens": tokens,
                "ds4_tokens": list(tokens),
                "frontier": frontier,
                "context": context,
                "vector": {
                    "file": vector_name,
                    "sha256": sha256(payload),
                    "argmax": case_index + 3,
                },
            }
        )

    continuation_payload = i32_bytes(CONTINUATION)
    continuation_name = "yarn-8193.continuation.i32"
    (root / continuation_name).write_bytes(continuation_payload)
    manifest = {
        "schema": PROMOTED_SCHEMA,
        "oracle_policy": ORACLE_POLICY,
        "vocab_size": VOCAB_SIZE,
        "continuation_case": "yarn-8193",
        "continuation_tokens": CONTINUATION_TOKENS,
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
            "ds4_seed_token_count": 32768,
        },
        "thresholds": {"cuda_admission": dict(CUDA_LIMITS)},
        "cases": cases,
        "continuation": {
            "case": "yarn-8193",
            "file": continuation_name,
            "sha256": sha256(continuation_payload),
            "argmax": list(CONTINUATION),
        },
    }
    write_json(root / "manifest.json", manifest)
    assert {path.name for path in root.iterdir()} == set(PROMOTED_FILES)


def write_capture(root: Path) -> str:
    """Write one llama capture and return its capture.json trust-anchor hash."""
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    capture_cases: list[dict[str, object]] = []
    for case_index, (case_id, render, frontier, context) in enumerate(CASES):
        prompt = source_rendered_prompt(case_id)
        prompt_name = f"{case_id}.prompt"
        tokens_name = f"{case_id}.tokens.i32"
        logits_name = f"{case_id}.logits.f32"
        tokens = case_tokens(case_id, frontier)
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
                "token_count": len(tokens),
                "argmax": case_index + 3,
            }
        )

    continuation_name = "yarn-8193.continuation.i32"
    continuation_payload = i32_bytes(CONTINUATION)
    (root / continuation_name).write_bytes(continuation_payload)
    files[continuation_name] = sha256(continuation_payload)
    step_names: list[str] = []
    for step, token in enumerate(CONTINUATION):
        name = f"yarn-8193.step-{step:02d}.logits.f32"
        payload = f32_bytes(token)
        (root / name).write_bytes(payload)
        files[name] = sha256(payload)
        step_names.append(name)

    assert set(files) == set(CAPTURE_ARTIFACT_FILES)
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
            "logits_files": step_names,
            "argmax": list(CONTINUATION),
        },
        "files": files,
    }
    write_json(root / "capture.json", capture)
    return sha256((root / "capture.json").read_bytes())


def fixture_entries(root: Path) -> set[str]:
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise AssertionError(f"non-regular fixture entry: {entry}")
    return {entry.name for entry in entries}


def tree_digest(root: Path) -> str:
    """Hash paths, entry kinds, symlink targets, and file bytes without mutation."""
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.readlink(path).encode() + b"\0")
        elif path.is_dir():
            digest.update(b"directory\0")
        elif path.is_file():
            digest.update(b"file\0" + path.read_bytes() + b"\0")
        else:
            digest.update(b"special\0")
    return digest.hexdigest()


def populate_destination(root: Path, *, prompts: bool = False) -> None:
    populate_fixed_inputs(root)
    if prompts:
        for case_id in ("swa-513", "yarn-8193", "deep-32768"):
            (root / f"{case_id}.prompt").write_bytes(rendered_prompt(case_id, root))


@dataclass(frozen=True)
class DirectOutcome:
    module: Any
    value: object | None = None
    error: BaseException | None = None


def load_tool_module() -> Any:
    module_name = f"compare_laguna_logits_red_{id(object())}"
    spec = importlib.util.spec_from_file_location(module_name, TOOL)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_attributes(module: Any, values: dict[str, object]) -> Iterator[None]:
    marker = object()
    old: dict[str, object] = {}
    for name, value in values.items():
        old[name] = getattr(module, name, marker)
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, previous in old.items():
            if previous is marker:
                delattr(module, name)
            else:
                setattr(module, name, previous)


@contextlib.contextmanager
def patched_environment(values: dict[str, str] | None) -> Iterator[None]:
    if not values:
        yield
        return
    marker = object()
    old = {name: os.environ.get(name, marker) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, previous in old.items():
            if previous is marker:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def direct_promote(
    ds4: Path,
    capture: Path,
    destination: Path,
    capture_sha256: str,
    *,
    module: Any | None = None,
    module_patches: dict[str, object] | None = None,
    environment: dict[str, str] | None = None,
) -> DirectOutcome:
    """Call promotion with a test-only injected capture trust anchor.

    The production CLI has no capture-digest override and must retain its
    hard-coded pin.  The returned module is used only to require its own
    ContractError class for negative outcomes.
    """
    module = module or load_tool_module()
    values: dict[str, object] = {"CAPTURE_MANIFEST_SHA256": capture_sha256}
    values.update(module_patches or {})
    with patched_attributes(module, values), patched_environment(environment):
        try:
            value = module.promote(Path(ds4), Path(capture), Path(destination))
        except Exception as exc:  # the assertion below requires module.ContractError
            return DirectOutcome(module=module, error=exc)
    return DirectOutcome(module=module, value=value)


def run_main_in_process(
    ds4: Path,
    capture: Path,
    destination: Path,
    capture_sha256: str,
    *,
    module: Any | None = None,
    module_patches: dict[str, object] | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[int | None, str, str, BaseException | None]:
    module = module or load_tool_module()
    values: dict[str, object] = {"CAPTURE_MANIFEST_SHA256": capture_sha256}
    values.update(module_patches or {})
    argv = [
        str(TOOL),
        "--ds4",
        str(ds4),
        "--llama",
        str(capture),
        "--promote",
        str(destination),
    ]
    import io

    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = argv
        with (
            patched_attributes(module, values),
            patched_environment(environment),
            contextlib.redirect_stdout(out_buffer),
            contextlib.redirect_stderr(err_buffer),
        ):
            try:
                code = module.main()
            except BaseException as exc:  # report as data so RED tests do not become ERRORs
                return None, out_buffer.getvalue(), err_buffer.getvalue(), exc
    finally:
        sys.argv = old_argv
    return int(code), out_buffer.getvalue(), err_buffer.getvalue(), None


def run_subprocess(
    argv: list[str], *, environment: dict[str, str] | None = None, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=stdout,
            stderr=stderr + "\nprocess timed out",
        )


def build_clean_ds4_repo(root: Path) -> tuple[Path, str, Path]:
    """Clone this repository and return (untracked fake executable, full HEAD, repo)."""
    ds4_repo = root / "ds4-repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(ds4_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    ds4 = ds4_repo / "fake-ds4"
    ds4.write_text(FAKE_DS4, encoding="utf-8")
    ds4.chmod(0o755)
    head = subprocess.run(
        ["git", "-C", str(ds4_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return ds4, head, ds4_repo


def refresh_capture_hash(root: Path, artifact: str) -> None:
    capture = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    capture["files"][artifact] = sha256((root / artifact).read_bytes())
    write_json(root / "capture.json", capture)


def rewrite_capture(root: Path, change: Callable[[dict[str, Any]], None]) -> str:
    path = root / "capture.json"
    capture = json.loads(path.read_text(encoding="utf-8"))
    change(capture)
    write_json(path, capture)
    return sha256(path.read_bytes())


def write_fake_git_for_malformed_head(root: Path) -> Path:
    real_git = shutil.which("git")
    if real_git is None:
        raise AssertionError("git is required for provenance tests")
    shim = root / "git"
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            saw_rev_parse=0
            for arg in "$@"; do
                if [ "$arg" = "rev-parse" ]; then saw_rev_parse=1; fi
                if [ "$arg" = "HEAD" ] && [ "$saw_rev_parse" = 1 ]; then
                    printf '%s\\n' 'not-a-full-head'
                    exit 0
                fi
            done
            exec {shlex.quote(real_git)} "$@"
            """
        ),
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def write_fake_git_for_failure(root: Path) -> Path:
    real_git = shutil.which("git")
    if real_git is None:
        raise AssertionError("git is required for provenance tests")
    shim = root / "git"
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            saw_rev_parse=0
            for arg in "$@"; do
                if [ "$arg" = "rev-parse" ]; then saw_rev_parse=1; fi
                if [ "$arg" = "--show-toplevel" ] && [ "$saw_rev_parse" = 1 ]; then
                    printf '%s\\n' 'deterministic git shim stderr' >&2
                    exit 29
                fi
            done
            exec {shlex.quote(real_git)} "$@"
            """
        ),
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def unrelated_commit(ds4_repo: Path) -> str:
    tree = subprocess.run(
        ["git", "-C", str(ds4_repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Unrelated Test",
            "GIT_AUTHOR_EMAIL": "unrelated@example.invalid",
            "GIT_COMMITTER_NAME": "Unrelated Test",
            "GIT_COMMITTER_EMAIL": "unrelated@example.invalid",
        }
    )
    return subprocess.run(
        ["git", "-C", str(ds4_repo), "commit-tree", tree, "-m", "unrelated contract"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


class SinglePoolsideV2ContractTests(unittest.TestCase):
    def run_verify_v2(
        self,
        fixture: Path,
        *extra: str,
        contract_commit: str = CONTRACT_COMMIT,
        tokenizer_runtime_commit: str = TOKENIZER_RUNTIME_COMMIT,
        llama_commit: str = LLAMA_COMMIT,
        capture_sha256: str = CAPTURE_MANIFEST_SHA256,
        gguf_size: int = GGUF_SIZE,
        gguf_sha256: str = GGUF_SHA256,
    ) -> subprocess.CompletedProcess[str]:
        return run_subprocess(
            [
                sys.executable,
                str(TOOL),
                "--verify-promoted",
                str(fixture),
                "--contract-commit",
                contract_commit,
                "--tokenizer-runtime-commit",
                tokenizer_runtime_commit,
                "--llama-commit",
                llama_commit,
                "--capture-manifest-sha256",
                capture_sha256,
                "--gguf-size",
                str(gguf_size),
                "--gguf-sha256",
                gguf_sha256,
                *extra,
            ]
        )

    def run_promote_cli_subprocess(
        self,
        ds4: Path,
        capture: Path,
        destination: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess[str]:
        return run_subprocess(
            [
                sys.executable,
                str(TOOL),
                "--ds4",
                str(ds4),
                "--llama",
                str(capture),
                "--promote",
                str(destination),
                *extra,
            ]
        )

    def require_module_callable(self, module: Any, name: str) -> Callable[..., Any]:
        candidate = getattr(module, name, None)
        self.assertTrue(
            callable(candidate),
            f"security test prerequisite missing: module.{name} callable",
        )
        return candidate

    def assert_direct_rejection(
        self, outcome: DirectOutcome, label: str, reason: str
    ) -> None:
        self.assertIsNotNone(outcome.error, f"{label}: promotion unexpectedly succeeded")
        self.assertIsInstance(
            outcome.error,
            outcome.module.ContractError,
            f"{label}: unexpected exception type {type(outcome.error).__name__}: {outcome.error}",
        )
        self.assertRegex(
            str(outcome.error),
            reason,
            f"{label}: ContractError lacks a useful reason: {outcome.error}",
        )

    def assert_main_failure(
        self,
        result: tuple[int | None, str, str, BaseException | None],
        label: str,
        reason: str,
    ) -> tuple[str, str]:
        code, stdout, stderr, error = result
        self.assertIsNone(error, f"{label}: CLI runner raised {error!r}")
        self.assertIsNotNone(code)
        self.assertNotEqual(code, 0, f"{label}: CLI unexpectedly succeeded")
        self.assertRegex(stderr, reason, f"{label}: missing useful failure reason: {stderr}")
        return stdout, stderr

    def test_valid_v2_fixture_has_exact_13_regular_files_and_no_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted"
            write_promoted_fixture(fixture)
            self.assertEqual(fixture_entries(fixture), set(PROMOTED_FILES))
            self.assertEqual(len(fixture_entries(fixture)), 13)
            before = tree_digest(fixture)

            completed = self.run_verify_v2(fixture)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout,
                f"verified={fixture} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(tree_digest(fixture), before)

    def test_v2_verifier_rejects_promoted_v1_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "legacy-schema"
            write_promoted_fixture(fixture)
            edit_manifest(fixture, lambda data: data.__setitem__("schema", "laguna-resident-promoted-v1"))
            before = tree_digest(fixture)
            completed = self.run_verify_v2(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertRegex(completed.stderr, r"verification failed:.*(?:schema|manifest|key)")
            self.assertEqual(tree_digest(fixture), before)

    def test_v2_verifier_fails_closed_for_isolated_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            write_promoted_fixture(base)

            def duplicate_schema(root: Path) -> None:
                path = root / "manifest.json"
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    text.replace(
                        "{\n",
                        '{\n  "schema": "laguna-resident-promoted-v2",\n',
                        1,
                    ),
                    encoding="utf-8",
                )

            def nonfinite_vector(root: Path) -> None:
                vector = root / "deep-32768.llama.f32"
                payload = bytearray(vector.read_bytes())
                payload[:4] = struct.pack("<f", float("nan"))
                vector.write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["cases"][3]["vector"].__setitem__("sha256", sha256(payload)),
                )

            def vector_size_only(root: Path) -> None:
                payload = b"short"
                (root / "swa-513.llama.f32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["cases"][1]["vector"].__setitem__("sha256", sha256(payload)),
                )

            def continuation_size_only(root: Path) -> None:
                payload = b"short"
                (root / "yarn-8193.continuation.i32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["continuation"].__setitem__("sha256", sha256(payload)),
                )

            def changed_continuation(root: Path) -> None:
                payload = bytearray((root / "yarn-8193.continuation.i32").read_bytes())
                payload[:4] = struct.pack("<i", 9)
                (root / "yarn-8193.continuation.i32").write_bytes(payload)

            def changed_continuation_with_matching_manifest(root: Path) -> None:
                ids = list(CONTINUATION)
                ids[0] = 9
                payload = i32_bytes(ids)
                (root / "yarn-8193.continuation.i32").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: (
                        data["continuation"].__setitem__("sha256", sha256(payload)),
                        data["continuation"].__setitem__("argmax", ids),
                    ),
                )

            def coherent_nonprefix_prompt(root: Path) -> None:
                prompt = b"not a benchmark prefix\n"
                (root / "yarn-8193.prompt").write_bytes(prompt)
                edit_manifest(
                    root,
                    lambda data: (
                        data["cases"][2].__setitem__("prompt_hex", prompt.hex()),
                        data["cases"][2].__setitem__("prompt_sha256", sha256(prompt)),
                    ),
                )

            def benchmark_bytes_with_refreshed_hash(root: Path) -> None:
                path = root / "benchmark-32768.txt"
                original = path.read_bytes()
                self.assertGreater(len(original), max(PROMPT_PREFIX_BYTES.values()))
                payload = original[:-1] + bytes([original[-1] ^ 1])
                path.write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["provenance"].__setitem__("benchmark_sha256", sha256(payload)),
                )

            def generator_bytes_with_refreshed_hash(root: Path) -> None:
                payload = b"changed generator bytes\n"
                (root / "generate_benchmark_prompt.py").write_bytes(payload)
                edit_manifest(
                    root,
                    lambda data: data["provenance"].__setitem__("generator_sha256", sha256(payload)),
                )

            def add_nested_key(container: str) -> Callable[[Path], None]:
                return lambda root: edit_manifest(
                    root, lambda data: data[container].__setitem__("unexpected", True)
                )

            def add_case_key(root: Path) -> None:
                edit_manifest(root, lambda data: data["cases"][0].__setitem__("unexpected", True))

            def add_vector_key(root: Path) -> None:
                edit_manifest(
                    root, lambda data: data["cases"][0]["vector"].__setitem__("unexpected", True)
                )

            def add_continuation_key(root: Path) -> None:
                edit_manifest(root, lambda data: data["continuation"].__setitem__("unexpected", True))

            def swap_cases(root: Path) -> None:
                edit_manifest(
                    root,
                    lambda data: data.__setitem__(
                        "cases", [data["cases"][1], data["cases"][0], data["cases"][2], data["cases"][3]]
                    ),
                )

            tamperers: dict[str, tuple[Callable[[Path], None], str]] = {
                "schema": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("schema", "laguna-resident-promoted-v1")),
                    r"schema|manifest|key",
                ),
                "oracle policy": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("oracle_policy", "dual-oracle")),
                    r"oracle|policy|manifest|key",
                ),
                "oracle name": (
                    lambda root: edit_manifest(root, lambda data: data["oracle"].__setitem__("name", "metal")),
                    r"oracle|name|poolside|manifest",
                ),
                "top-level vocabulary": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("vocab_size", VOCAB_SIZE - 1)),
                    r"vocab|size|manifest",
                ),
                "top-level continuation case": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("continuation_case", "short")),
                    r"continuation|case|manifest",
                ),
                "top-level continuation tokens": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("continuation_tokens", CONTINUATION_TOKENS - 1)),
                    r"continuation|token|length|manifest",
                ),
                "missing vector": (
                    lambda root: (root / "short.llama.f32").unlink(),
                    r"file|missing|fixture",
                ),
                "extra file": (
                    lambda root: (root / "unexpected.bin").write_bytes(b"x"),
                    r"file|extra|fixture",
                ),
                "directory entry": (
                    lambda root: (root / "__pycache__").mkdir(),
                    r"file|directory|regular|fixture",
                ),
                "symlink entry": (
                    lambda root: os.symlink(root / "short.txt", root / "extra-link"),
                    r"file|symlink|regular|fixture",
                ),
                "vector size only": (vector_size_only, r"size|bytes|vector"),
                "vector hash": (
                    lambda root: (root / "short.llama.f32").write_bytes(
                        b"X" + (root / "short.llama.f32").read_bytes()[1:]
                    ),
                    r"hash|SHA|vector",
                ),
                "vector argmax": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0]["vector"].__setitem__("argmax", 0)),
                    r"argmax|vector",
                ),
                "non-finite vector": (nonfinite_vector, r"finite|vector|logit"),
                "model repository": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("repository", "other/model")),
                    r"model|repository|pin",
                ),
                "model revision": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("revision", "0" * 40)),
                    r"model|revision|pin",
                ),
                "model legacy file": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("file", "laguna-s-2.1-Q4_K_M.gguf")),
                    r"model|file|legacy|key|pin",
                ),
                "model wrong filename": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("filename", "other.gguf")),
                    r"model|filename|pin",
                ),
                "model size": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("size", GGUF_SIZE - 1)),
                    r"model|size|pin",
                ),
                "model hash": (
                    lambda root: edit_manifest(root, lambda data: data["model"].__setitem__("sha256", "0" * 64)),
                    r"model|hash|SHA|pin",
                ),
                "model nested key": (add_nested_key("model"), r"model|key|manifest"),
                "oracle runtime": (
                    lambda root: edit_manifest(root, lambda data: data["oracle"].__setitem__("runtime_commit", "0" * 40)),
                    r"oracle|runtime|pin",
                ),
                "oracle capture digest": (
                    lambda root: edit_manifest(root, lambda data: data["oracle"].__setitem__("capture_manifest_sha256", "0" * 64)),
                    r"oracle|capture|digest|pin",
                ),
                "oracle nested key": (add_nested_key("oracle"), r"oracle|key|manifest"),
                "provenance contract": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("contract_commit", "0" * 40)),
                    r"contract|provenance|commit|pin",
                ),
                "provenance tokenizer": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("tokenizer_runtime_commit", "0" * 40)),
                    r"tokenizer|provenance|commit|pin",
                ),
                "provenance nested key": (add_nested_key("provenance"), r"provenance|key|manifest"),
                "generator digest": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("generator_sha256", "0" * 64)),
                    r"generator|digest|SHA|provenance",
                ),
                "benchmark digest": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("benchmark_sha256", "0" * 64)),
                    r"benchmark|digest|SHA|provenance",
                ),
                "poolside seed count": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("poolside_seed_token_count", 61439)),
                    r"seed|token|count|provenance",
                ),
                "ds4 seed count": (
                    lambda root: edit_manifest(root, lambda data: data["provenance"].__setitem__("ds4_seed_token_count", 32767)),
                    r"seed|token|count|provenance",
                ),
                "threshold nested key": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("unexpected", True)),
                    r"threshold|key|CUDA|manifest",
                ),
                "thresholds nested key": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"].__setitem__("unexpected", True)),
                    r"threshold|key|manifest",
                ),
                "centered RMS type": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("centered_rms", "0.04")),
                    r"threshold|number|RMS|type",
                ),
                "centered max type": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("centered_max_abs", None)),
                    r"threshold|number|max|type",
                ),
                "top20 float": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("top20_overlap", 18.0)),
                    r"threshold|top.?20|number|type",
                ),
                "top20 boolean": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("top20_overlap", True)),
                    r"threshold|top.?20|number|type",
                ),
                "argmax boolean integer": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("argmax_equal", 1)),
                    r"threshold|boolean|argmax|type",
                ),
                "continuation boolean integer": (
                    lambda root: edit_manifest(root, lambda data: data["thresholds"]["cuda_admission"].__setitem__("continuation_equal", 1)),
                    r"threshold|boolean|continuation|type",
                ),
                "case nested key": (add_case_key, r"case|key|manifest"),
                "case id": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("id", "wrong")),
                    r"case|id|identity|manifest",
                ),
                "case render": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("render", "raw")),
                    r"case|render|identity|manifest",
                ),
                "case legacy metal_tokens": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("metal_tokens", list(data["cases"][0]["poolside_tokens"]))),
                    r"case|metal|legacy|key",
                ),
                "case legacy oracles": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("oracles", {})),
                    r"case|oracle|legacy|key",
                ),
                "case legacy metrics": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("metrics", {})),
                    r"case|metric|legacy|key",
                ),
                "case order": (swap_cases, r"case|order|identity"),
                "frontier": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][1].__setitem__("frontier", 512)),
                    r"frontier|case|context",
                ),
                "context": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][2].__setitem__("context", 8201)),
                    r"context|case|frontier",
                ),
                "prompt hex": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("prompt_hex", "00")),
                    r"prompt|hex|UTF|bytes",
                ),
                "prompt hash": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][0].__setitem__("prompt_sha256", "0" * 64)),
                    r"prompt|hash|SHA|bytes",
                ),
                "prompt UTF-8": (
                    lambda root: edit_manifest(root, lambda data: (data["cases"][0].__setitem__("prompt_hex", "ff"), data["cases"][0].__setitem__("prompt_sha256", sha256(b"\xff")))),
                    r"prompt|UTF|encoding|bytes",
                ),
                "prompt file bytes": (
                    lambda root: (root / "yarn-8193.prompt").write_bytes(b"changed\n"),
                    r"prompt|bytes|hash|benchmark",
                ),
                "prompt coherent non-prefix": (coherent_nonprefix_prompt, r"prompt|prefix|benchmark"),
                "benchmark actual bytes": (benchmark_bytes_with_refreshed_hash, r"benchmark|fixed|seed|bytes|hash"),
                "generator actual bytes": (generator_bytes_with_refreshed_hash, r"generator|fixed|bytes|hash"),
                "poolside token mismatch": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][1]["poolside_tokens"].__setitem__(0, 999)),
                    r"token|Poolside|DS4|vocab",
                ),
                "ds4 token mismatch": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][1]["ds4_tokens"].__setitem__(0, 999)),
                    r"token|Poolside|DS4|vocab",
                ),
                "poolside token out of range": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][1]["poolside_tokens"].__setitem__(0, VOCAB_SIZE)),
                    r"token|vocab|range",
                ),
                "ds4 token out of range": (
                    lambda root: edit_manifest(root, lambda data: data["cases"][1]["ds4_tokens"].__setitem__(0, VOCAB_SIZE)),
                    r"token|vocab|range",
                ),
                "vector nested key": (add_vector_key, r"vector|key|manifest"),
                "continuation size only": (continuation_size_only, r"continuation|size|bytes"),
                "continuation hash": (changed_continuation, r"continuation|hash|SHA|bytes"),
                "continuation argmax": (
                    lambda root: edit_manifest(root, lambda data: data["continuation"]["argmax"].__setitem__(0, 9)),
                    r"continuation|argmax|step",
                ),
                "continuation step-zero parity": (changed_continuation_with_matching_manifest, r"continuation|argmax|step|vector"),
                "continuation nested key": (add_continuation_key, r"continuation|key|manifest"),
                "continuation case": (
                    lambda root: edit_manifest(root, lambda data: data["continuation"].__setitem__("case", "short")),
                    r"continuation|case|manifest",
                ),
                "legacy runtimes": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("runtimes", {})),
                    r"legacy|runtime|key|manifest",
                ),
                "legacy dwarfstar commit": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("dwarfstar_commit", "0" * 40)),
                    r"legacy|dwarf|commit|key|manifest",
                ),
                "extra manifest key": (
                    lambda root: edit_manifest(root, lambda data: data.__setitem__("metal_metrics", {})),
                    r"manifest|key|legacy|extra",
                ),
                "duplicate JSON key": (duplicate_schema, r"duplicate|JSON|key|parse"),
            }
            for label, (tamper, reason) in tamperers.items():
                with self.subTest(label=label):
                    fixture = Path(tmp) / f"tampered-{label.replace(' ', '-')}"
                    shutil.copytree(base, fixture)
                    tamper(fixture)
                    before = tree_digest(fixture)
                    completed = self.run_verify_v2(fixture)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertRegex(
                        completed.stderr,
                        rf"(?:verification|promotion) failed:.*(?:{reason})",
                    )
                    self.assertEqual(tree_digest(fixture), before)

    def test_supplied_verification_pins_are_each_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "promoted"
            write_promoted_fixture(fixture)
            cases: dict[str, tuple[dict[str, object], str]] = {
                "contract": ({"contract_commit": "0" * 40}, r"contract|commit|pin"),
                "tokenizer runtime": (
                    {"tokenizer_runtime_commit": "0" * 40},
                    r"tokenizer|runtime|commit|pin",
                ),
                "llama runtime": ({"llama_commit": "0" * 40}, r"llama|runtime|commit|pin"),
                "capture digest": ({"capture_sha256": "0" * 64}, r"capture|digest|SHA|pin"),
                "GGUF size": ({"gguf_size": GGUF_SIZE - 1}, r"GGUF|model|size|pin"),
                "GGUF hash": ({"gguf_sha256": "0" * 64}, r"GGUF|model|hash|SHA|pin"),
            }
            for label, (overrides, reason) in cases.items():
                with self.subTest(pin=label):
                    before = tree_digest(fixture)
                    completed = self.run_verify_v2(fixture, **overrides)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertRegex(
                        completed.stderr,
                        rf"(?:verification|promotion) failed:.*(?:{reason})",
                    )
                    self.assertEqual(tree_digest(fixture), before)

    def test_capture_hash_is_checked_before_malformed_duplicate_or_hostile_json_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds4, _, _ = build_clean_ds4_repo(root)
            payloads = {
                "malformed": b"{ this is not JSON",
                "duplicate": b'{"schema":"laguna-resident-capture-v1","schema":"hostile"}',
                "hostile": b'{"files":{"../../escape":"' + b"0" * 64 + b'"},"oracle":"metal"}',
            }
            for label, payload in payloads.items():
                with self.subTest(payload=label):
                    capture = root / f"capture-{label}"
                    capture.mkdir()
                    (capture / "capture.json").write_bytes(payload)
                    destination = root / f"destination-{label}"
                    populate_destination(destination)
                    before = tree_digest(destination)
                    outcome = direct_promote(
                        ds4,
                        capture,
                        destination,
                        "0" * 64,
                    )
                    self.assert_direct_rejection(outcome, "wrong trust anchor " + label, r"trust.?anchor")
                    self.assertEqual(tree_digest(destination), before)

    def test_capture_helper_is_single_llama_with_explicit_21_artifacts_and_trust_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_root = Path(tmp) / "capture"
            digest = write_capture(capture_root)
            capture_payload = (capture_root / "capture.json").read_bytes()
            capture = json.loads(capture_payload)
            self.assertEqual(digest, sha256(capture_payload))
            self.assertEqual(capture["schema"], "laguna-resident-capture-v1")
            self.assertEqual(capture["oracle"], "llama")
            self.assertEqual(capture["seed_token_count"], 61440)
            self.assertEqual(set(capture["files"]), set(CAPTURE_ARTIFACT_FILES))
            self.assertEqual(
                {path.name for path in capture_root.iterdir()},
                set(CAPTURE_ARTIFACT_FILES) | {"capture.json"},
            )
            self.assertEqual(len(CAPTURE_ARTIFACT_FILES), 21)
            self.assertNotIn("metal", capture_payload.decode("utf-8"))
            self.assertNotIn("dwarfstar", capture_payload.decode("utf-8"))

    def test_capture_helper_proves_deterministic_prompt_and_token_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_root = Path(tmp) / "capture"
            write_capture(capture_root)
            capture = json.loads((capture_root / "capture.json").read_text(encoding="utf-8"))
            benchmark = (FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()
            deep = capture["cases"][3]
            deep_tokens = list(struct.unpack("<32768i", (capture_root / deep["tokens_file"]).read_bytes()))
            self.assertEqual(len(deep_tokens), 32768)
            for index, (case_id, _, frontier, _) in enumerate(CASES[1:], start=1):
                case = capture["cases"][index]
                prompt = (capture_root / case["prompt_file"]).read_bytes()
                self.assertEqual(prompt, benchmark[: len(prompt)])
                tokens = list(
                    struct.unpack(
                        f"<{frontier}i", (capture_root / case["tokens_file"]).read_bytes()
                    )
                )
                self.assertEqual(tokens, deep_tokens[:frontier])
                self.assertEqual(case["id"], case_id)

    def test_clean_ds4_helper_clones_real_contract_ancestor_and_keeps_fake_untracked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds4, head, ds4_repo = build_clean_ds4_repo(Path(tmp))
            self.assertEqual(len(head), 40)
            self.assertEqual(
                subprocess.run(
                    ["git", "-C", str(ds4_repo), "merge-base", "--is-ancestor", CONTRACT_COMMIT, head],
                    check=False,
                ).returncode,
                0,
            )
            status = subprocess.run(
                ["git", "-C", str(ds4_repo), "status", "--short", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(status.stdout, "")
            self.assertTrue(ds4.is_file())
            self.assertTrue(ds4.stat().st_mode & 0o111)
            self.assertIn("?? fake-ds4", subprocess.run(
                ["git", "-C", str(ds4_repo), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout)

    def test_promotion_publishes_prompts_vectors_and_verifies_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, head, _ = build_clean_ds4_repo(root)
            self.assertRegex(head, r"^[0-9a-f]{40}$")
            destination = root / "promoted"
            populate_destination(destination)
            before_inputs = {name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES}
            call_log = root / "ds4-calls.log"

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                environment={"FAKE_DS4_LOG": str(call_log)},
            )

            self.assertIsNone(outcome.error, f"v2 promotion failed: {outcome.error!r}")
            self.assertEqual(fixture_entries(destination), set(PROMOTED_FILES))
            manifest = read_manifest(destination)
            self.assertEqual(manifest["provenance"]["tokenizer_runtime_commit"], head)
            self.assertEqual(manifest["oracle"]["capture_manifest_sha256"], capture_digest)

            capture_json = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
            for index, (case_id, render, frontier, context) in enumerate(CASES):
                captured_case = capture_json["cases"][index]
                promoted_case = manifest["cases"][index]
                self.assertEqual(promoted_case["id"], captured_case["id"])
                self.assertEqual(promoted_case["id"], case_id)
                self.assertEqual(promoted_case["render"], render)
                self.assertEqual(promoted_case["frontier"], frontier)
                self.assertEqual(promoted_case["context"], context)
                captured_prompt = (capture / captured_case["prompt_file"]).read_bytes()
                self.assertEqual(bytes.fromhex(promoted_case["prompt_hex"]), captured_prompt)
                self.assertEqual(promoted_case["prompt_sha256"], sha256(captured_prompt))
                if case_id != "short":
                    self.assertEqual(
                        (destination / f"{case_id}.prompt").read_bytes(),
                        captured_prompt,
                    )
                captured_tokens = list(
                    struct.unpack(
                        f"<{frontier}i", (capture / captured_case["tokens_file"]).read_bytes()
                    )
                )
                self.assertEqual(promoted_case["poolside_tokens"], captured_tokens)
                self.assertEqual(promoted_case["ds4_tokens"], captured_tokens)
                captured_logits = (capture / captured_case["logits_file"]).read_bytes()
                promoted_vector = promoted_case["vector"]
                self.assertEqual(promoted_vector["file"], f"{case_id}.llama.f32")
                self.assertEqual(
                    (destination / promoted_vector["file"]).read_bytes(),
                    captured_logits,
                )
                self.assertEqual(promoted_vector["sha256"], sha256(captured_logits))
                self.assertEqual(promoted_vector["argmax"], captured_case["argmax"])

            captured_continuation = (capture / "yarn-8193.continuation.i32").read_bytes()
            promoted_continuation = destination / "yarn-8193.continuation.i32"
            self.assertEqual(promoted_continuation.read_bytes(), captured_continuation)
            self.assertEqual(manifest["continuation"]["sha256"], sha256(captured_continuation))
            self.assertEqual(
                manifest["continuation"]["argmax"], capture_json["continuation"]["argmax"]
            )
            self.assertEqual(manifest["continuation"]["argmax"], CONTINUATION)
            self.assertEqual(
                {name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES},
                before_inputs,
            )

            expected_payloads = {
                sha256(source_rendered_prompt("short")): (source_rendered_prompt("short"), 3, 100000, 100001),
                sha256(source_rendered_prompt("swa-513")): (source_rendered_prompt("swa-513"), 513, 0, 512),
                sha256(source_rendered_prompt("yarn-8193")): (source_rendered_prompt("yarn-8193"), 8193, 0, 8192),
                sha256(source_rendered_prompt("deep-32768")): (source_rendered_prompt("deep-32768"), 32768, 0, 32767),
                sha256((FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes()): ((FIXTURE_SOURCE / "benchmark-32768.txt").read_bytes(), 32768, 0, 32767),
            }
            expected_digests = Counter(expected_payloads)
            calls = []
            for line in call_log.read_text(encoding="utf-8").splitlines():
                digest, size, count, first, last = line.split()
                calls.append((digest, int(size), int(count), int(first), int(last)))
            self.assertEqual(Counter(call[0] for call in calls), Counter({digest: 1 for digest in expected_payloads}))
            for digest, (payload, count, first, last) in expected_payloads.items():
                self.assertEqual(
                    [call for call in calls if call[0] == digest],
                    [(digest, len(payload), count, first, last)],
                )
            self.assertEqual(tree_digest(capture), capture_before)

            before = tree_digest(destination)
            verifier = load_tool_module()
            with patched_attributes(verifier, {"CAPTURE_MANIFEST_SHA256": capture_digest}):
                verifier.verify_promoted(
                    destination,
                    CONTRACT_COMMIT,
                    head,
                    LLAMA_COMMIT,
                    capture_digest,
                    GGUF_SIZE,
                    GGUF_SHA256,
                )
            self.assertEqual(tree_digest(destination), before)

    def test_promotion_rejects_wrong_ds4_benchmark_ids_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination)
            destination_before = tree_digest(destination)

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                environment={"FAKE_DS4_BAD_BENCHMARK": "1"},
            )

            self.assert_direct_rejection(
                outcome,
                "wrong benchmark token IDs",
                r"benchmark|token|parity|first.?32768|differ",
            )
            self.assertEqual(tree_digest(destination), destination_before)
            self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_rechecks_checkout_cleanliness_after_ds4_tokenization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, ds4_repo = build_clean_ds4_repo(root)
            tracked = ds4_repo / "tests/test-vectors/laguna-resident/cases.json"
            tracked_before = tracked.read_bytes()
            destination = root / "promoted"
            populate_destination(destination)
            destination_before = tree_digest(destination)

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                environment={"FAKE_DS4_DIRTY_TRACKED": str(tracked)},
            )

            self.assert_direct_rejection(
                outcome,
                "late tracked checkout mutation",
                r"tracked|dirty|clean|checkout|HEAD|provenance",
            )
            self.assertNotEqual(tracked.read_bytes(), tracked_before)
            self.assertEqual(tree_digest(destination), destination_before)
            self.assertEqual(tree_digest(capture), capture_before)
            status = subprocess.run(
                ["git", "-C", str(ds4_repo), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("cases.json", status)

    def test_promotion_rejects_fake_ds4_error_and_malformed_output(self) -> None:
        short_digest = sha256(source_rendered_prompt("short"))
        swa_digest = sha256(source_rendered_prompt("swa-513"))
        modes = {
            "nonzero stderr": (
                {"FAKE_DS4_FAIL_PROMPT_DIGEST": short_digest},
                r"DS4|execute|failed|stderr|exit",
            ),
            "malformed token output": (
                {"FAKE_DS4_MALFORMED_PROMPT_DIGEST": swa_digest},
                r"DS4|parse|list|token|output",
            ),
        }
        for label, (environment_extra, reason) in modes.items():
            with self.subTest(mode=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture = root / "capture"
                capture_digest = write_capture(capture)
                capture_before = tree_digest(capture)
                ds4, _, _ = build_clean_ds4_repo(root)
                destination = root / "promoted"
                populate_destination(destination)
                destination_before = tree_digest(destination)

                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    environment=environment_extra,
                )

                self.assert_direct_rejection(outcome, label, reason)
                self.assertEqual(tree_digest(destination), destination_before)
                self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_accepts_preexisting_byte_exact_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, head, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination, prompts=True)
            before_prompts = {
                name: (destination / name).read_bytes() for name in MATERIALIZED_PROMPT_FILES
            }
            outcome = direct_promote(ds4, capture, destination, capture_digest)
            self.assertIsNone(outcome.error, f"v2 promotion failed: {outcome.error!r}")
            self.assertEqual(read_manifest(destination)["provenance"]["tokenizer_runtime_commit"], head)
            self.assertEqual(
                {name: (destination / name).read_bytes() for name in MATERIALIZED_PROMPT_FILES},
                before_prompts,
            )
            self.assertEqual(fixture_entries(destination), set(PROMOTED_FILES))

    def test_promotion_rejects_mismatched_existing_prompt_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination, prompts=True)
            (destination / "yarn-8193.prompt").write_bytes(b"wrong existing bytes\n")
            before = tree_digest(destination)
            outcome = direct_promote(ds4, capture, destination, capture_digest)
            self.assert_direct_rejection(
                outcome, "mismatched existing prompt", r"prompt|byte|mismatch|existing"
            )
            self.assertEqual(tree_digest(destination), before)

    def test_cli_promotion_output_is_exact_v2_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, head, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination)
            code, stdout, stderr, error = run_main_in_process(
                ds4,
                capture,
                destination,
                capture_digest,
            )
            self.assertIsNone(error, f"v2 CLI runner raised {error!r}")
            self.assertEqual(code, 0, stderr)
            self.assertEqual(
                stdout,
                f"promoted={destination} cases=4 vectors=4 oracle=poolside\n",
            )
            self.assertEqual(stderr, "")
            self.assertEqual(read_manifest(destination)["provenance"]["tokenizer_runtime_commit"], head)
            self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_trust_boundary_rejects_stale_and_invalid_capture_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_capture = root / "base-capture"
            base_digest = write_capture(base_capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            mutations: dict[str, tuple[Callable[[Path], str], str]] = {
                "stale digest": (lambda capture: "0" * 64, r"trust.?anchor|SHA|digest"),
                "capture schema value": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("schema", "laguna-resident-capture-v0")
                    ),
                    r"schema|capture|version",
                ),
                "capture legacy field": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("metal", {})
                    ),
                    r"legacy|extra|key|capture|schema",
                ),
                "capture case id": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][0].__setitem__("id", "wrong")
                    ),
                    r"case|id|identity|capture",
                ),
                "capture case prompt reference": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][1].__setitem__("prompt", "short.txt")
                    ),
                    r"prompt|reference|case|capture",
                ),
                "capture prompt filename traversal": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][1].__setitem__("prompt_file", "../escape")
                    ),
                    r"prompt|filename|path|unsafe|escape|capture",
                ),
                "capture continuation case": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["continuation"].__setitem__("case", "short")
                    ),
                    r"continuation|case|capture",
                ),
                "capture continuation logits extra": (
                    lambda capture: rewrite_capture(
                        capture,
                        lambda data: data["continuation"].__setitem__(
                            "logits_files",
                            list(data["continuation"]["logits_files"]) + ["unexpected.step.f32"],
                        ),
                    ),
                    r"continuation|logit|length|step|capture",
                ),
                "unknown capture key": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("unexpected", True)
                    ),
                    r"capture|key|schema",
                ),
                "extra capture artifact": (
                    lambda capture: (
                        (capture / "unexpected.bin").write_bytes(b"extra"),
                        rewrite_capture(
                            capture,
                            lambda data: data["files"].__setitem__(
                                "unexpected.bin", sha256((capture / "unexpected.bin").read_bytes())
                            ),
                        ),
                    )[1],
                    r"capture|file|allow|extra",
                ),
                "unsafe capture filename": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["files"].__setitem__("../escape", "0" * 64)
                    ),
                    r"capture|filename|path|allow|unsafe",
                ),
                "changed referenced file": (
                    lambda capture: (
                        (capture / "short.prompt").write_bytes(b"changed capture bytes\n"),
                        base_digest,
                    )[1],
                    r"capture|hash|SHA|digest|short",
                ),
                "referenced hash mismatch": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["files"].__setitem__("short.prompt", "0" * 64)
                    ),
                    r"capture|hash|SHA|short",
                ),
                "wrong capture oracle": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("oracle", "metal")
                    ),
                    r"oracle|capture|poolside|llama",
                ),
                "capture model shape": (
                    lambda capture: rewrite_capture(
                        capture,
                        lambda data: data.__setitem__(
                            "model", {"repository": "poolside/Laguna-S-2.1-GGUF"}
                        ),
                    ),
                    r"model|key|shape|capture",
                ),
                "capture model repository": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("repository", "other/model")
                    ),
                    r"model|repository|pin|capture",
                ),
                "capture model file": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("file", "other.gguf")
                    ),
                    r"model|file|pin|capture",
                ),
                "capture model size": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("size", GGUF_SIZE + 1)
                    ),
                    r"model|size|pin|capture",
                ),
                "capture model sha256": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("sha256", "0" * 64)
                    ),
                    r"model|sha|SHA|pin|capture",
                ),
                "capture model pin": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("revision", "0" * 40)
                    ),
                    r"model|revision|pin|capture",
                ),
                "capture runtime pin": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("runtime_commit", "0" * 40)
                    ),
                    r"runtime|commit|pin|capture",
                ),
                "capture seed count": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("seed_token_count", 61439)
                    ),
                    r"seed|token|count|capture",
                ),
            }
            for label, (mutate, reason) in mutations.items():
                with self.subTest(label=label):
                    capture = root / f"capture-{label.replace(' ', '-')}"
                    shutil.copytree(base_capture, capture)
                    expected_digest = mutate(capture)
                    capture_before = tree_digest(capture)
                    destination = root / f"destination-{label.replace(' ', '-')}"
                    populate_destination(destination)
                    before = tree_digest(destination)
                    outcome = direct_promote(ds4, capture, destination, expected_digest)
                    self.assert_direct_rejection(outcome, label, reason)
                    self.assertEqual(tree_digest(destination), before)
                    self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_rejects_capture_payload_and_nested_schema_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_capture = root / "base-capture"
            write_capture(base_capture)
            ds4, _, _ = build_clean_ds4_repo(root)

            def capture_digest(capture: Path) -> str:
                return sha256((capture / "capture.json").read_bytes())

            def refresh_and_digest(capture: Path, artifact: str) -> str:
                refresh_capture_hash(capture, artifact)
                return capture_digest(capture)

            def short_logits_wrong_size(capture: Path) -> str:
                (capture / "short.logits.f32").write_bytes(b"short")
                return refresh_and_digest(capture, "short.logits.f32")

            def short_logits_nonfinite(capture: Path) -> str:
                payload = bytearray((capture / "short.logits.f32").read_bytes())
                payload[:4] = struct.pack("<f", float("nan"))
                (capture / "short.logits.f32").write_bytes(payload)
                return refresh_and_digest(capture, "short.logits.f32")

            def continuation_id_out_of_range(capture: Path) -> str:
                path = capture / "yarn-8193.continuation.i32"
                payload = bytearray(path.read_bytes())
                payload[:4] = struct.pack("<i", VOCAB_SIZE)
                path.write_bytes(payload)
                return refresh_and_digest(capture, path.name)

            def continuation_step_wrong_size(capture: Path) -> str:
                artifact = "yarn-8193.step-00.logits.f32"
                (capture / artifact).write_bytes(b"short")
                return refresh_and_digest(capture, artifact)

            def continuation_step_nonfinite(capture: Path) -> str:
                artifact = "yarn-8193.step-00.logits.f32"
                payload = bytearray((capture / artifact).read_bytes())
                payload[:4] = struct.pack("<f", float("nan"))
                (capture / artifact).write_bytes(payload)
                return refresh_and_digest(capture, artifact)

            def case_order(capture: Path) -> str:
                return rewrite_capture(
                    capture,
                    lambda data: data.__setitem__(
                        "cases", [data["cases"][1], data["cases"][0], data["cases"][2], data["cases"][3]]
                    ),
                )

            mutations: dict[str, tuple[Callable[[Path], str], str]] = {
                "capture logits wrong size": (short_logits_wrong_size, r"logit|size|bytes|vector"),
                "capture logits nonfinite": (short_logits_nonfinite, r"logit|finite|number|nan"),
                "capture continuation id range": (continuation_id_out_of_range, r"continuation|token|range|vocab"),
                "capture continuation step size": (continuation_step_wrong_size, r"continuation|logit|size|bytes"),
                "capture continuation step nonfinite": (continuation_step_nonfinite, r"continuation|logit|finite|number|nan"),
                "capture case ordering": (case_order, r"case|order|identity"),
                "capture case frontier": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][1].__setitem__("frontier", 512)
                    ),
                    r"frontier|case|context",
                ),
                "capture case context": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][2].__setitem__("context", 8201)
                    ),
                    r"context|case|frontier",
                ),
                "capture case render": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][0].__setitem__("render", "raw")
                    ),
                    r"render|case|identity",
                ),
                "capture nested case key": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["cases"][0].__setitem__("unexpected", True)
                    ),
                    r"case|key|schema",
                ),
            }
            for label, (mutate, reason) in mutations.items():
                with self.subTest(label=label):
                    capture = root / f"payload-capture-{label.replace(' ', '-')}"
                    shutil.copytree(base_capture, capture)
                    expected_digest = mutate(capture)
                    capture_before = tree_digest(capture)
                    destination = root / f"payload-destination-{label.replace(' ', '-')}"
                    populate_destination(destination)
                    destination_before = tree_digest(destination)

                    outcome = direct_promote(ds4, capture, destination, expected_digest)

                    self.assert_direct_rejection(outcome, label, reason)
                    self.assertEqual(tree_digest(destination), destination_before)
                    self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_rejects_capture_provenance_and_deterministic_generation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_capture = root / "base-capture"
            write_capture(base_capture)
            ds4, _, _ = build_clean_ds4_repo(root)

            def change_token_file(capture: Path, token: int) -> str:
                path = capture / "swa-513.tokens.i32"
                ids = list(struct.unpack("<513i", path.read_bytes()))
                ids[0] = token
                path.write_bytes(i32_bytes(ids))
                refresh_capture_hash(capture, path.name)
                return sha256((capture / "capture.json").read_bytes())

            mutations: dict[str, tuple[Callable[[Path], str], str]] = {
                "runtime pin": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("runtime_commit", "0" * 40)
                    ),
                    r"runtime|commit|pin|capture",
                ),
                "model pin": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data["model"].__setitem__("revision", "0" * 40)
                    ),
                    r"model|revision|pin|capture",
                ),
                "seed count": (
                    lambda capture: rewrite_capture(
                        capture, lambda data: data.__setitem__("seed_token_count", 61340)
                    ),
                    r"seed|token|count|capture",
                ),
                "non-prefix captured prompt": (
                    lambda capture: (
                        (capture / "yarn-8193.prompt").write_bytes(b"not a benchmark prefix\n"),
                        refresh_capture_hash(capture, "yarn-8193.prompt"),
                        sha256((capture / "capture.json").read_bytes()),
                    )[2],
                    r"prompt|prefix|benchmark",
                ),
                "token mismatch": (
                    lambda capture: change_token_file(capture, 999),
                    r"token|Poolside|DS4|vocab",
                ),
                "token out of range": (
                    lambda capture: change_token_file(capture, VOCAB_SIZE),
                    r"token|vocab|range",
                ),
            }
            for label, (mutate, reason) in mutations.items():
                with self.subTest(label=label):
                    capture = root / f"provenance-capture-{label.replace(' ', '-')}"
                    shutil.copytree(base_capture, capture)
                    expected_digest = mutate(capture)
                    capture_before = tree_digest(capture)
                    destination = root / f"provenance-destination-{label.replace(' ', '-')}"
                    populate_destination(destination)
                    before = tree_digest(destination)
                    outcome = direct_promote(ds4, capture, destination, expected_digest)
                    self.assert_direct_rejection(outcome, label, reason)
                    self.assertEqual(tree_digest(destination), before)
                    self.assertEqual(tree_digest(capture), capture_before)

    def test_promotion_rejects_dirty_unresolved_nonancestor_and_malformed_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)

            scenarios: list[tuple[str, Callable[[Path, Path], tuple[dict[str, object], dict[str, str] | None]]]] = []

            def dirty(repo: Path, _: Path) -> tuple[dict[str, object], dict[str, str] | None]:
                tracked = repo / "tests/test-vectors/laguna-resident/cases.json"
                tracked.write_bytes(tracked.read_bytes() + b"\n")
                return {}, None

            def unresolved(_: Path, __: Path) -> tuple[dict[str, object], dict[str, str] | None]:
                return {"CONTRACT_COMMIT": "deadbeef" * 5}, None

            def nonancestor(repo: Path, __: Path) -> tuple[dict[str, object], dict[str, str] | None]:
                return {"CONTRACT_COMMIT": unrelated_commit(repo)}, None

            def malformed(_: Path, root_for_shim: Path) -> tuple[dict[str, object], dict[str, str] | None]:
                shim = write_fake_git_for_malformed_head(root_for_shim)
                environment = {"PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"}
                return {}, environment

            scenarios.extend(
                [
                    ("dirty tracked checkout", dirty),
                    ("unresolved contract commit", unresolved),
                    ("contract is not an ancestor", nonancestor),
                    ("malformed non-full HEAD", malformed),
                ]
            )
            for label, setup in scenarios:
                with self.subTest(label=label):
                    scenario_root = root / label.replace(" ", "-")
                    scenario_root.mkdir()
                    ds4, _, ds4_repo = build_clean_ds4_repo(scenario_root)
                    module_patches, environment = setup(ds4_repo, scenario_root)
                    destination = scenario_root / "destination"
                    populate_destination(destination)
                    before = tree_digest(destination)
                    outcome = direct_promote(
                        ds4,
                        capture,
                        destination,
                        capture_digest,
                        module_patches=module_patches,
                        environment=environment,
                    )
                    self.assert_direct_rejection(
                        outcome,
                        label,
                        r"tracked|dirty|clean|contract|commit|ancestor|HEAD|full",
                    )
                    self.assertEqual(tree_digest(destination), before)

    def test_read_bytes_rejects_oversized_sparse_file_before_payload_read(self) -> None:
        oversized_bytes = 1 << 34
        readers = ("read_bytes", "read_bytes_with_stat")
        for reader_name in readers:
            with self.subTest(reader=reader_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                victim = root / "oversized-contract-file.bin"
                with victim.open("wb") as handle:
                    handle.truncate(oversized_bytes)
                module = load_tool_module()
                reader = self.require_module_callable(module, reader_name)
                original_os_read = module.os.read
                read_calls = 0

                def forbidden_payload_read(fd: int, size: int) -> bytes:
                    nonlocal read_calls
                    read_calls += 1
                    raise OSError("payload read must not be attempted")

                module.os.read = forbidden_payload_read
                try:
                    with self.assertRaises(module.ContractError) as raised:
                        reader(victim)
                finally:
                    module.os.read = original_os_read

                self.assertEqual(read_calls, 0)
                self.assertRegex(
                    str(raised.exception),
                    r"size|limit|max|large|oversized|resource",
                )
                self.assertTrue(victim.is_file())
                self.assertEqual(victim.stat().st_size, oversized_bytes)

    def test_read_bytes_rejects_fifo_without_blocking(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("os.mkfifo is unavailable on this platform")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "would-be-input.bin"
            victim.write_bytes(b"regular input before replacement\n")
            victim.unlink()
            try:
                os.mkfifo(victim, 0o600)
            except OSError as exc:
                unavailable = {
                    errno.ENOSYS,
                    getattr(errno, "ENOTSUP", errno.EINVAL),
                    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                }
                if exc.errno in unavailable:
                    self.skipTest(f"mkfifo is unavailable: {exc}")
                raise

            runner = root / "read-fifo.py"
            runner.write_text(
                textwrap.dedent(
                    f"""\
                    import importlib.util
                    import sys
                    from pathlib import Path

                    spec = importlib.util.spec_from_file_location("fifo_reader", {str(TOOL)!r})
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    try:
                        module.read_bytes(Path(sys.argv[1]))
                    except module.ContractError as exc:
                        print("ContractError: " + str(exc))
                        raise SystemExit(0)
                    except BaseException as exc:
                        print("unexpected exception: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
                        raise SystemExit(2)
                    print("unexpected success", file=sys.stderr)
                    raise SystemExit(3)
                    """
                ),
                encoding="utf-8",
            )
            completed = run_subprocess(
                [sys.executable, str(runner), str(victim)],
                timeout=3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("ContractError", completed.stdout)
            self.assertTrue(victim.is_fifo())

    def test_promotion_git_failure_stderr_becomes_contract_error_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            shim = write_fake_git_for_failure(root)
            destination = root / "promoted"
            populate_destination(destination)
            destination_before = tree_digest(destination)
            environment = {"PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"}

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                environment=environment,
            )

            self.assert_direct_rejection(
                outcome,
                "git command failure",
                r"git|stderr|failed|exit",
            )
            self.assertIn("deterministic git shim stderr", str(outcome.error))
            self.assertEqual(tree_digest(destination), destination_before)
            self.assertEqual(tree_digest(capture), capture_before)

    def test_read_bytes_rejects_symlink_swap_between_regular_check_and_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "victim.bin"
            external = root / "external.bin"
            victim.write_bytes(b"trusted bytes\n")
            external.write_bytes(b"external bytes\n")
            module = load_tool_module()
            original_read_bytes = module.Path.read_bytes
            original_os_open = module.os.open
            swapped = False

            def swap_victim() -> None:
                nonlocal swapped
                if not swapped:
                    victim.unlink()
                    victim.symlink_to(external)
                    swapped = True

            def swap_before_path_open(
                path: Path,
                *args: object,
                **kwargs: object,
            ) -> bytes:
                if path == victim:
                    swap_victim()
                return original_read_bytes(path, *args, **kwargs)

            def swap_before_os_open(
                path: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> int:
                if Path(path) == victim:
                    swap_victim()
                return original_os_open(path, *args, **kwargs)

            module.Path.read_bytes = swap_before_path_open
            module.os.open = swap_before_os_open
            try:
                with self.assertRaises(module.ContractError) as raised:
                    module.read_bytes(victim)
            finally:
                module.Path.read_bytes = original_read_bytes
                module.os.open = original_os_open

            self.assertTrue(swapped)
            self.assertTrue(victim.is_symlink())
            self.assertEqual(os.readlink(victim), str(external))
            self.assertEqual(external.read_bytes(), b"external bytes\n")
            self.assertRegex(str(raised.exception), r"symlink|regular|file|race")

    def test_publication_keyboard_interrupt_and_system_exit_reraise_and_clean(self) -> None:
        for signal_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(signal=signal_type.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture = root / "capture"
                capture_digest = write_capture(capture)
                capture_before = tree_digest(capture)
                ds4, _, _ = build_clean_ds4_repo(root)
                destination_parent = root / "destination-parent"
                destination_parent.mkdir()
                destination = destination_parent / "promoted"
                populate_destination(destination)
                inputs_before = {
                    name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES
                }
                module = load_tool_module()
                original_link = module.os.link
                link_calls = 0

                def interrupt_after_one_link(
                    source: str | bytes | os.PathLike[str],
                    target: str | bytes | os.PathLike[str],
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    nonlocal link_calls
                    link_calls += 1
                    if link_calls == 1:
                        original_link(source, target, *args, **kwargs)
                        return
                    raise signal_type("injected publication interruption")

                module.os.link = interrupt_after_one_link
                try:
                    with self.assertRaises(signal_type):
                        direct_promote(
                            ds4,
                            capture,
                            destination,
                            capture_digest,
                            module=module,
                        )
                finally:
                    module.os.link = original_link

                self.assertGreaterEqual(link_calls, 2)
                self.assertEqual(fixture_entries(destination), set(FIXED_INPUT_FILES))
                self.assertEqual(
                    {name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES},
                    inputs_before,
                )
                self.assertEqual(tree_digest(capture), capture_before)
                siblings = {path.name for path in destination_parent.iterdir()}
                self.assertNotIn(f".{destination.name}.lock", siblings)
                self.assertFalse(
                    any(name.startswith(f".{destination.name}.tmp-") for name in siblings)
                )

    def test_external_replacement_of_stage_directory_survives_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            module = load_tool_module()
            original_link = module.os.link
            original_open_owned_directory = self.require_module_callable(module, "open_owned_directory")
            stage_path: Path | None = None
            link_calls = 0
            replacement_stage: Path | None = None
            moved_owned_stage: Path | None = None
            replacement_before: str | None = None
            destination_before = tree_digest(destination)
            marker = b"external stage replacement\n"

            def record_owned_directory(path: Path, label: str):
                nonlocal stage_path
                result = original_open_owned_directory(path, label)
                if label == "promotion staging directory":
                    stage_path = Path(path)
                return result

            def replace_stage_after_first_link(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal link_calls, replacement_stage, moved_owned_stage, replacement_before
                link_calls += 1
                original_link(source, target, *args, **kwargs)
                if link_calls != 1:
                    return
                self.assertIsNotNone(stage_path)
                assert stage_path is not None
                moved_owned_stage = stage_path.with_name(stage_path.name + ".moved")
                stage_path.rename(moved_owned_stage)
                stage_path.mkdir()
                replacement_stage = stage_path
                (stage_path / "external-marker").write_bytes(marker)
                replacement_before = tree_digest(stage_path)

            module.os.link = replace_stage_after_first_link
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                    module_patches={"open_owned_directory": record_owned_directory},
                )
            finally:
                module.os.link = original_link

            self.assert_direct_rejection(
                outcome,
                "external stage replacement",
                r"stage|link|publish|output|removed|replaced|path",
            )
            self.assertGreaterEqual(link_calls, 1)
            self.assertIsNotNone(replacement_stage)
            assert replacement_stage is not None
            self.assertIsNotNone(replacement_before)
            assert replacement_before is not None
            self.assertTrue(replacement_stage.is_dir())
            self.assertEqual((replacement_stage / "external-marker").read_bytes(), marker)
            self.assertEqual(tree_digest(destination), destination_before)
            self.assertEqual(fixture_entries(destination), set(FIXED_INPUT_FILES))
            self.assertEqual(tree_digest(capture), capture_before)
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual(tree_digest(replacement_stage), replacement_before)
            lock = destination.with_name(f".{destination.name}.lock")
            self.assertFalse(lock.exists())
            # The moved original stage is not ours after the external rename; the
            # temporary test root owns its eventual teardown.
            self.assertIsNotNone(moved_owned_stage)

    def test_staged_source_inode_replacement_is_rejected_and_external_payload_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination)
            module = load_tool_module()
            original_link = module.os.link
            original_open_owned_directory = self.require_module_callable(module, "open_owned_directory")
            stage_fd: int | None = None
            link_calls = 0
            external_payload = root / "external-staged-source.bin"
            replacement_bytes = b"external staged source replacement\n"

            def record_owned_directory(path: Path, label: str):
                nonlocal stage_fd
                result = original_open_owned_directory(path, label)
                if label == "promotion staging directory":
                    stage_fd = result[0]
                return result

            def replace_source_before_link(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal link_calls
                link_calls += 1
                if link_calls == 1:
                    source_fd = kwargs.get("src_dir_fd")
                    self.assertIsNotNone(source_fd)
                    self.assertEqual(source_fd, stage_fd)
                    external_payload.write_bytes(replacement_bytes)
                    os.unlink(source, dir_fd=source_fd)
                    original_link(
                        external_payload,
                        source,
                        dst_dir_fd=source_fd,
                        follow_symlinks=False,
                    )
                original_link(source, target, *args, **kwargs)

            module.os.link = replace_source_before_link
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                    module_patches={"open_owned_directory": record_owned_directory},
                )
            finally:
                module.os.link = original_link

            self.assert_direct_rejection(
                outcome,
                "staged source inode replacement",
                r"source|inode|stage|changed|race|publish|link",
            )
            self.assertEqual(link_calls, 1)
            self.assertEqual(external_payload.read_bytes(), replacement_bytes)
            owned_outputs = {
                path.name for path in destination.iterdir() if path.name not in FIXED_INPUT_FILES
            }
            self.assertEqual(owned_outputs, set())
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual(tree_digest(capture), capture_before)

    def test_destination_retarget_during_ds4_tokenization_is_rejected_before_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            replacement = destination_parent / "replacement-destination"
            original_destination = destination_parent / "original-destination"
            populate_destination(destination)
            populate_destination(replacement)
            destination_before = tree_digest(destination)
            replacement_before = tree_digest(replacement)
            self.assertEqual(destination_before, replacement_before)
            trigger_digest = sha256(source_rendered_prompt("short"))
            environment = {
                "FAKE_DS4_RETARGET_ON_DIGEST": trigger_digest,
                "FAKE_DS4_RETARGET_DESTINATION": str(destination),
                "FAKE_DS4_RETARGET_REPLACEMENT": str(replacement),
                "FAKE_DS4_RETARGET_ORIGINAL": str(original_destination),
            }

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                environment=environment,
            )

            self.assertTrue(original_destination.is_dir())
            self.assertTrue(destination.is_dir())
            self.assert_direct_rejection(
                outcome,
                "destination retarget during DS4 tokenization",
                r"destination|inode|identity|changed|retarget|replacement|preflight",
            )
            self.assertTrue(original_destination.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertEqual(tree_digest(original_destination), destination_before)
            self.assertEqual(tree_digest(destination), replacement_before)
            self.assertEqual(fixture_entries(destination), set(FIXED_INPUT_FILES))
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual(tree_digest(capture), capture_before)
            lock = destination.with_name(f".{destination.name}.lock")
            self.assertFalse(lock.exists())
            self.assertEqual(
                {path.name for path in destination_parent.iterdir()},
                {destination.name, original_destination.name},
            )

    def test_existing_prompt_replacement_is_revalidated_before_manifest_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination, prompts=True)
            victim = destination / "swa-513.prompt"
            external_prompt = root / "external-prompt.bin"
            external_bytes = b"external replacement prompt\n"
            external_prompt.write_bytes(external_bytes)
            inputs_before = {
                name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES
            }
            module = load_tool_module()
            original_link = module.os.link
            link_targets: list[str] = []
            replaced = False

            def replace_prompt_after_first_output(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal replaced
                original_link(source, target, *args, **kwargs)
                name = Path(target).name
                link_targets.append(name)
                if not replaced:
                    self.assertNotEqual(name, "manifest.json")
                    victim.unlink()
                    victim.symlink_to(external_prompt)
                    replaced = True

            module.os.link = replace_prompt_after_first_output
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                )
            finally:
                module.os.link = original_link

            self.assert_direct_rejection(
                outcome,
                "existing prompt replacement at manifest boundary",
                r"prompt|destination|changed|symlink|revalidate|bytes",
            )
            self.assertTrue(replaced)
            self.assertGreaterEqual(len(link_targets), 1)
            self.assertNotIn("manifest.json", link_targets)
            self.assertTrue(victim.is_symlink())
            self.assertEqual(os.readlink(victim), str(external_prompt))
            self.assertEqual(external_prompt.read_bytes(), external_bytes)
            self.assertEqual(
                {name: (destination / name).read_bytes() for name in FIXED_INPUT_FILES},
                inputs_before,
            )
            owned_outputs = {
                path.name
                for path in destination.iterdir()
                if path.name not in FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
            }
            self.assertEqual(owned_outputs, set())
            self.assertEqual(tree_digest(capture), capture_before)
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual({path.name for path in destination_parent.iterdir()}, {"promoted"})

    def test_fixed_input_replacement_is_revalidated_before_manifest_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            victim = destination / "benchmark-32768.txt"
            original_bytes = victim.read_bytes()
            external_bytes = original_bytes[:-1] + bytes([original_bytes[-1] ^ 1])
            module = load_tool_module()
            original_link = module.os.link
            link_targets: list[str] = []
            replaced = False

            def replace_fixed_input_after_first_output(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal replaced
                original_link(source, target, *args, **kwargs)
                name = Path(target).name
                link_targets.append(name)
                if not replaced:
                    self.assertNotEqual(name, "manifest.json")
                    victim.write_bytes(external_bytes)
                    replaced = True

            module.os.link = replace_fixed_input_after_first_output
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                )
            finally:
                module.os.link = original_link

            self.assert_direct_rejection(
                outcome,
                "fixed input replacement at manifest boundary",
                r"fixed|input|benchmark|changed|bytes|manifest|revalidate",
            )
            self.assertTrue(replaced)
            self.assertGreaterEqual(len(link_targets), 1)
            self.assertNotIn("manifest.json", link_targets)
            self.assertEqual(victim.read_bytes(), external_bytes)
            owned_outputs = {
                path.name for path in destination.iterdir() if path.name not in FIXED_INPUT_FILES
            }
            self.assertEqual(owned_outputs, set())
            self.assertEqual(tree_digest(capture), capture_before)
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual({path.name for path in destination_parent.iterdir()}, {"promoted"})

    def test_lock_replacement_after_acquisition_is_rejected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            lock = destination.with_name(f".{destination.name}.lock")
            external_marker = b"external lock replacement\n"
            module = load_tool_module()
            original_validate = self.require_module_callable(module, "validate_destination_for_promotion")
            replaced = False

            def validate_then_replace_lock(*args: object, **kwargs: object):
                nonlocal replaced
                result = original_validate(*args, **kwargs)
                if not replaced:
                    self.assertTrue(lock.is_dir())
                    shutil.rmtree(lock)
                    lock.mkdir()
                    (lock / "external-marker").write_bytes(external_marker)
                    replaced = True
                return result

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                module=module,
                module_patches={
                    "validate_destination_for_promotion": validate_then_replace_lock,
                },
            )

            self.assert_direct_rejection(
                outcome,
                "lost promotion lock ownership",
                r"lock|ownership|owner|changed|replacement|publish",
            )
            self.assertTrue(replaced)
            self.assertTrue(lock.is_dir())
            self.assertEqual((lock / "external-marker").read_bytes(), external_marker)
            self.assertEqual(fixture_entries(destination), set(FIXED_INPUT_FILES))
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual(tree_digest(capture), capture_before)
            self.assertEqual(
                {path.name for path in destination_parent.iterdir()},
                {destination.name, lock.name},
            )

    def test_destination_root_replacement_is_rejected_before_linking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            original_destination = destination_parent / "external-original-destination"
            replacement_marker = b"external destination replacement\n"
            original_before = tree_digest(destination)
            replacement_before: str | None = None
            module = load_tool_module()
            original_open_owned_directory = self.require_module_callable(module, "open_owned_directory")
            replaced = False

            def open_then_replace_destination(path: Path, label: str):
                nonlocal replaced, replacement_before
                result = original_open_owned_directory(path, label)
                if label == "promotion destination" and not replaced:
                    destination.rename(original_destination)
                    destination.mkdir()
                    (destination / "external-marker").write_bytes(replacement_marker)
                    replacement_before = tree_digest(destination)
                    replaced = True
                return result

            outcome = direct_promote(
                ds4,
                capture,
                destination,
                capture_digest,
                module=module,
                module_patches={"open_owned_directory": open_then_replace_destination},
            )

            self.assert_direct_rejection(
                outcome,
                "destination root inode replacement",
                r"destination|root|inode|changed|replacement|publish|link",
            )
            self.assertTrue(replaced)
            self.assertEqual(tree_digest(original_destination), original_before)
            self.assertIsNotNone(replacement_before)
            assert replacement_before is not None
            self.assertEqual(tree_digest(destination), replacement_before)
            self.assertEqual(
                (destination / "external-marker").read_bytes(), replacement_marker
            )
            self.assertEqual(fixture_entries(destination), {"external-marker"})
            self.assertFalse((destination / "manifest.json").exists())
            self.assertEqual(tree_digest(capture), capture_before)
            lock = destination.with_name(f".{destination.name}.lock")
            self.assertFalse(lock.exists())
            self.assertEqual(
                {path.name for path in destination_parent.iterdir()},
                {destination.name, original_destination.name},
            )

    def test_promotion_rejects_existing_outputs_and_lists_stale_partial_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            existing_outputs = {
                "vector": "short.llama.f32",
                "continuation": "yarn-8193.continuation.i32",
                "manifest": "manifest.json",
            }
            for label, name in existing_outputs.items():
                with self.subTest(existing=label):
                    destination = root / f"existing-{label}"
                    populate_destination(destination, prompts=True)
                    if name.endswith(".f32"):
                        (destination / name).write_bytes(f32_bytes(3))
                    elif name.endswith(".i32"):
                        (destination / name).write_bytes(i32_bytes(CONTINUATION))
                    else:
                        (destination / name).write_bytes(b"pre-existing manifest")
                    before = tree_digest(destination)
                    outcome = direct_promote(ds4, capture, destination, capture_digest)
                    self.assert_direct_rejection(
                        outcome, f"existing {label}", r"existing|no.?clobber|published|output|manifest|stale|partial"
                    )
                    self.assertEqual(tree_digest(destination), before)

            stale = root / "stale-partial"
            populate_destination(stale)
            stale_paths = ("short.llama.f32", "yarn-8193.continuation.i32")
            (stale / stale_paths[0]).write_bytes(f32_bytes(3))
            (stale / stale_paths[1]).write_bytes(i32_bytes(CONTINUATION))
            before = tree_digest(stale)
            result = run_main_in_process(ds4, capture, stale, capture_digest)
            _, stderr = self.assert_main_failure(
                result, "stale partial publication", r"stale|partial|short\.llama\.f32|continuation"
            )
            for name in stale_paths:
                self.assertIn(name, stderr)
            self.assertEqual(tree_digest(stale), before)

    def test_promotion_rejects_preheld_sibling_lock_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination = root / "promoted"
            populate_destination(destination)
            lock = destination.with_name(f".{destination.name}.lock")
            lock.mkdir()
            (lock / "owner").write_text("external lock owner\n", encoding="utf-8")
            destination_before = tree_digest(destination)
            lock_before = tree_digest(lock)
            siblings_before = {path.name for path in root.iterdir()}

            outcome = direct_promote(ds4, capture, destination, capture_digest)

            self.assert_direct_rejection(
                outcome, "pre-held sibling lock", r"lock|held|busy|exist"
            )
            self.assertEqual(tree_digest(destination), destination_before)
            self.assertEqual(tree_digest(lock), lock_before)
            self.assertEqual({path.name for path in root.iterdir()}, siblings_before)

    def test_two_concurrent_promoters_have_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            barrier = root / "promoter-barrier"
            barrier.mkdir()
            runner = root / "promote-runner.py"
            runner.write_text(
                textwrap.dedent(
                    f"""\
                    import importlib.util
                    import sys
                    import time
                    from pathlib import Path

                    module_spec = importlib.util.spec_from_file_location("compare_runner", {str(TOOL)!r})
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    module.CAPTURE_MANIFEST_SHA256 = sys.argv[4]
                    barrier = Path(sys.argv[5])
                    ready = barrier / (sys.argv[6] + ".ready")
                    ready.write_text("ready\\n", encoding="utf-8")
                    deadline = time.monotonic() + 15
                    while len(list(barrier.glob("*.ready"))) < 2:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("concurrent promotion barrier timed out")
                        time.sleep(0.01)
                    try:
                        module.promote(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
                    except module.ContractError as exc:
                        print("ContractError: " + str(exc), file=sys.stderr)
                        raise SystemExit(1)
                    except BaseException as exc:
                        print("unexpected exception: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
                        raise SystemExit(2)
                    print("ok")
                    """
                ),
                encoding="utf-8",
            )
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(runner),
                        str(ds4),
                        str(capture),
                        str(destination),
                        capture_digest,
                        str(barrier),
                        f"promoter-{index}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for index in range(2)
            ]
            results: list[subprocess.CompletedProcess[str]] = []
            for process in processes:
                try:
                    stdout, stderr = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    self.fail(f"concurrent promoter hung: {stdout}{stderr}")
                results.append(
                    subprocess.CompletedProcess(
                        [sys.executable, str(runner)], process.returncode, stdout, stderr
                    )
                )
            self.assertEqual(sum(result.returncode == 0 for result in results), 1, results)
            self.assertEqual(sum(result.returncode != 0 for result in results), 1, results)
            loser = next(result for result in results if result.returncode != 0)
            self.assertIn("ContractError", loser.stderr)
            self.assertRegex(loser.stderr, r"lock|clobber|existing|published")
            self.assertEqual(fixture_entries(destination), set(PROMOTED_FILES))
            self.assertEqual({path.name for path in destination_parent.iterdir()}, {"promoted"})
            for marker in barrier.iterdir():
                marker.unlink()
            self.assertEqual(list(barrier.iterdir()), [])

    def test_publication_links_manifest_last_and_cleans_stage_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            capture_before = tree_digest(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination)
            module = load_tool_module()
            original_link = module.os.link
            original_open_owned_directory = self.require_module_callable(module, "open_owned_directory")
            destination_fd: int | None = None
            destination_links: list[str] = []

            def record_owned_directory(path: Path, label: str):
                nonlocal destination_fd
                result = original_open_owned_directory(path, label)
                if label == "promotion destination":
                    destination_fd = result[0]
                return result

            def record_destination_link(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                original_link(source, target, *args, **kwargs)
                target_path = Path(target)
                if target_path.parent == destination or (
                    destination_fd is not None and kwargs.get("dst_dir_fd") == destination_fd
                ):
                    destination_links.append(target_path.name)

            module.os.link = record_destination_link
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                    module_patches={"open_owned_directory": record_owned_directory},
                )
            finally:
                module.os.link = original_link

            self.assertIsNone(outcome.error, f"v2 promotion failed: {outcome.error!r}")
            self.assertEqual(set(destination_links), set(PROMOTED_OUTPUT_FILES))
            self.assertEqual(len(destination_links), len(PROMOTED_OUTPUT_FILES))
            self.assertEqual(destination_links[-1], "manifest.json")
            self.assertEqual(fixture_entries(destination), set(PROMOTED_FILES))
            self.assertEqual({path.name for path in destination_parent.iterdir()}, {"promoted"})
            self.assertEqual(tree_digest(capture), capture_before)

    def test_link_failure_cleans_only_new_links_and_preserves_inputs_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture_digest = write_capture(capture)
            ds4, _, _ = build_clean_ds4_repo(root)
            destination_parent = root / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / "promoted"
            populate_destination(destination, prompts=True)
            before_inputs = {
                name: (destination / name).read_bytes()
                for name in FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
            }
            module = load_tool_module()
            original_link = module.os.link
            original_open_owned_directory = self.require_module_callable(module, "open_owned_directory")
            destination_fd: int | None = None
            link_calls = 0
            replaced_target: Path | None = None
            replacement = b"external replacement must survive cleanup\n"

            def record_owned_directory(path: Path, label: str):
                nonlocal destination_fd
                result = original_open_owned_directory(path, label)
                if label == "promotion destination":
                    destination_fd = result[0]
                return result

            def fail_after_external_replacement(
                source: str | bytes | os.PathLike[str],
                target: str | bytes | os.PathLike[str],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal link_calls, replaced_target
                link_calls += 1
                self.assertEqual(kwargs.get("dst_dir_fd"), destination_fd)
                original_link(source, target, *args, **kwargs)
                if link_calls == 1:
                    target_fd = kwargs.get("dst_dir_fd")
                    self.assertIsNotNone(target_fd)
                    replaced_target = destination / Path(target).name
                    os.unlink(target, dir_fd=target_fd)
                    replaced_target.write_bytes(replacement)

            module.os.link = fail_after_external_replacement
            try:
                outcome = direct_promote(
                    ds4,
                    capture,
                    destination,
                    capture_digest,
                    module=module,
                    module_patches={"open_owned_directory": record_owned_directory},
                )
            finally:
                module.os.link = original_link
            self.assert_direct_rejection(outcome, "forced link failure", r"link|publish|injected|output")
            self.assertEqual(link_calls, 1)
            self.assertIsNotNone(replaced_target)
            self.assertEqual(replaced_target.read_bytes(), replacement)
            for name, payload in before_inputs.items():
                self.assertEqual((destination / name).read_bytes(), payload)
            output_names = {path.name for path in destination.iterdir()} - set(
                FIXED_INPUT_FILES | MATERIALIZED_PROMPT_FILES
            )
            self.assertEqual(output_names, {replaced_target.name})
            self.assertEqual({path.name for path in destination_parent.iterdir()}, {"promoted"})

    def test_legacy_cli_arguments_are_rejected_without_destination_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "promoted"
            write_promoted_fixture(fixture)
            capture = root / "capture"
            write_capture(capture)
            ds4 = root / "ds4"
            destination = root / "promotion-destination"
            populate_fixed_inputs(destination)
            before_verify = tree_digest(fixture)
            before_promote = tree_digest(destination)
            legacy_flags = ("--metal", "--dwarfstar-commit", "--ds4-commit")
            for flag in legacy_flags:
                with self.subTest(mode="verify", flag=flag):
                    completed = self.run_verify_v2(fixture, flag, "legacy-value")
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertRegex(completed.stderr, r"unrecognized arguments")
                    self.assertEqual(tree_digest(fixture), before_verify)

            promotion_overrides = (
                ("--metal", "legacy-value"),
                ("--dwarfstar-commit", "legacy-value"),
                ("--ds4-commit", "legacy-value"),
                ("--capture-manifest-sha256", "legacy-value"),
                ("--contract-commit", "legacy-value"),
                ("--tokenizer-runtime-commit", "legacy-value"),
                ("--llama-commit", "legacy-value"),
                ("--gguf-size", str(GGUF_SIZE)),
                ("--gguf-sha256", "legacy-value"),
            )
            for flag, value in promotion_overrides:
                with self.subTest(mode="promote", flag=flag):
                    completed = self.run_promote_cli_subprocess(
                        ds4, capture, destination, flag, value
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertRegex(
                        completed.stderr,
                        r"unrecognized arguments|promotion mode requires|verification mode|only",
                    )
                    self.assertEqual(tree_digest(destination), before_promote)


if __name__ == "__main__":
    unittest.main()
