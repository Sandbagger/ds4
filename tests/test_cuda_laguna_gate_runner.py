#!/usr/bin/env python3
"""Behavioral contract for the one-descriptor Laguna CUDA gate runner."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/run_cuda_laguna_gate.sh"
PINNED_TOKENIZER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
ISOLATED_FIXTURE_VARIABLES = (
    "DS4_LAGUNA_GATE_TEST_CHILD_DIR",
    "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE",
    "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256",
)
FAKE_ENVIRONMENT_VARIABLES = (
    "LAGUNA_FAKE_FAIL_ROLE",
    "LAGUNA_FAKE_HANG_ROLE",
    "LAGUNA_FAKE_LOG",
    "LAGUNA_FAKE_MUTATION",
    "LAGUNA_FAKE_ORIGINAL_HEX",
    "LAGUNA_FAKE_REPLACEMENT_HEX",
    "LAGUNA_FAKE_TIMEOUT_LOG",
    "LAGUNA_FAKE_TIMEOUT_ROLE",
    "LAGUNA_FAKE_UNLINK_ORIGINAL",
    "LAGUNA_TEST_REMOVE_STAGED_OFFSET_GUARD",
)

FAKE_CHILD = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

class ContractError(Exception):
    pass


def inspect_retained_fd(fd):
    before_offset = os.lseek(fd, 0, os.SEEK_CUR)
    status = os.fstat(fd)
    if not stat.S_ISREG(status.st_mode):
        raise ContractError(f"fd {fd} is not regular")
    payload = os.pread(fd, status.st_size, 0)
    after_offset = os.lseek(fd, 0, os.SEEK_CUR)
    expected = bytes.fromhex(os.environ["LAGUNA_FAKE_ORIGINAL_HEX"])
    if payload != expected or before_offset != after_offset:
        raise ContractError(f"fd {fd} bytes or offset changed")
    return before_offset, after_offset, status, payload


def verify_gguf_fd(fd, _expected_size, _expected_sha256):
    inspect_retained_fd(fd)


def main():
    child_name = Path(sys.argv[0]).name
    roles = {
        "compare_laguna_logits.py": "verifier",
        "test_cuda_laguna_kernels": "kernels",
        "test_cuda_laguna_model": "model",
        "test_cuda_laguna_stream": "stream",
    }
    if child_name not in roles:
        raise SystemExit(f"unexpected fake child name: {child_name}")
    role = roles[child_name]
    if os.environ.get("DS4_TEST_MODEL_FD") != "9":
        raise SystemExit(f"{role}: DS4_TEST_MODEL_FD was not forced to 9")
    try:
        before_offset, after_offset, status, payload = inspect_retained_fd(9)
    except ContractError as exc:
        raise SystemExit(f"{role}: {exc}") from exc

    case = None
    if role == "verifier":
        if "--gguf-fd" not in sys.argv or sys.argv[sys.argv.index("--gguf-fd") + 1] != "9":
            raise SystemExit("verifier did not receive --gguf-fd 9")
        if "--gguf-size" not in sys.argv or "--gguf-sha256" not in sys.argv:
            raise SystemExit("verifier did not receive the expected identity")
        if "--tokenizer-runtime-commit" in sys.argv:
            index = sys.argv.index("--tokenizer-runtime-commit")
            if sys.argv[index + 1] != os.environ["LAGUNA_TOKENIZER_RUNTIME_COMMIT"]:
                raise SystemExit("verifier did not receive the tokenizer commit")
    elif role == "kernels":
        if sys.argv[1:] != ["--case", "all"]:
            raise SystemExit(f"kernel argv mismatch: {sys.argv[1:]!r}")
    elif role == "model":
        if sys.argv[1:]:
            raise SystemExit(f"model argv mismatch: {sys.argv[1:]!r}")
    else:
        allowed_cases = {
            "startup",
            "teardown-unsafe",
            "model-startup",
            "model-teardown-unsafe",
        }
        if len(sys.argv) != 3 or sys.argv[1] != "--case":
            raise SystemExit(f"stream argv mismatch: {sys.argv[1:]!r}")
        case = sys.argv[2]
        if case not in allowed_cases:
            raise SystemExit(f"unexpected stream case: {case}")

    record = {
        "role": role,
        "case": case,
        "argv": sys.argv[1:],
        "pid": os.getpid(),
        "fd": 9,
        "device": status.st_dev,
        "inode": status.st_ino,
        "size": status.st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "before_offset": before_offset,
        "after_offset": after_offset,
        "offset": after_offset,
    }
    log = Path(os.environ["LAGUNA_FAKE_LOG"])
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    if role == "verifier":
        model = Path(os.environ["DS4_TEST_MODEL"])
        original = model.with_name(model.name + ".original")
        model.rename(original)
        model.write_bytes(
            bytes.fromhex(os.environ["LAGUNA_FAKE_REPLACEMENT_HEX"])
        )
        if os.environ.get("LAGUNA_FAKE_UNLINK_ORIGINAL") == "1":
            original.unlink()

    mutation = os.environ.get("LAGUNA_FAKE_MUTATION")
    original_path = Path(os.environ["DS4_TEST_MODEL"] + ".original")
    if role == "verifier" and mutation == "truncate-after-verifier":
        with original_path.open("r+b") as handle:
            handle.truncate(status.st_size - 1)
    if role == "model" and mutation == "same-size-after-model":
        changed = b"X" + payload[1:]
        with original_path.open("r+b") as handle:
            handle.seek(0)
            handle.write(changed)
            handle.flush()
        os.utime(
            original_path,
            ns=(status.st_atime_ns, status.st_mtime_ns),
        )
    if (
        role == "stream"
        and case == "startup"
        and mutation == "seek-after-stream-startup"
    ):
        os.lseek(9, 7, os.SEEK_SET)

    fail_role = os.environ.get("LAGUNA_FAKE_FAIL_ROLE")
    if fail_role in (role, f"{role}:{case}"):
        raise SystemExit(23)
    hang_role = os.environ.get("LAGUNA_FAKE_HANG_ROLE")
    if hang_role in (role, f"{role}:{case}"):
        time.sleep(60)


if __name__ == "__main__":
    main()
'''

FAKE_TIMEOUT = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

arguments = sys.argv[1:]
kill_after = None
if arguments and arguments[0].startswith("--kill-after="):
    kill_after = arguments.pop(0).split("=", 1)[1]
if len(arguments) < 2:
    raise SystemExit(f"fake timeout argv mismatch: {sys.argv[1:]!r}")
if os.environ.get("DS4_TEST_MODEL_FD") != "9":
    raise SystemExit("timeout: DS4_TEST_MODEL_FD was not forced to 9")

duration = arguments[0]
command = arguments[1:]
before_offset = os.lseek(9, 0, os.SEEK_CUR)
status = os.fstat(9)
if not stat.S_ISREG(status.st_mode):
    raise SystemExit("timeout: fd 9 is not regular")
payload = os.pread(9, status.st_size, 0)
after_offset = os.lseek(9, 0, os.SEEK_CUR)
expected = bytes.fromhex(os.environ["LAGUNA_FAKE_ORIGINAL_HEX"])
if payload != expected or before_offset != after_offset:
    raise SystemExit("timeout: fd 9 bytes or offset changed")

child_name = Path(command[0]).name
roles = {
    "test_cuda_laguna_kernels": "kernels",
    "test_cuda_laguna_model": "model",
    "test_cuda_laguna_stream": "stream",
}
role = roles.get(child_name, child_name)
case = None
if "--case" in command:
    case_index = command.index("--case")
    if case_index + 1 < len(command):
        case = command[case_index + 1]
selector = role if case is None else f"{role}:{case}"
record = {
    "selector": selector,
    "kill_after": kill_after,
    "duration": duration,
    "command": command,
    "pid": os.getpid(),
    "fd": 9,
    "device": status.st_dev,
    "inode": status.st_ino,
    "size": status.st_size,
    "sha256": hashlib.sha256(payload).hexdigest(),
    "before_offset": before_offset,
    "after_offset": after_offset,
}
with Path(os.environ["LAGUNA_FAKE_TIMEOUT_LOG"]).open(
    "a", encoding="utf-8"
) as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")

if os.environ.get("LAGUNA_FAKE_TIMEOUT_ROLE") == selector:
    raise SystemExit(124)
os.execvp(command[0], command)
'''


class LagunaGateRunnerTest(unittest.TestCase):
    def clean_gate_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        for name in (
            "DS4_TEST_MODEL",
            "DS4_TEST_MODEL_FD",
            "LAGUNA_TOKENIZER_RUNTIME_COMMIT",
            *ISOLATED_FIXTURE_VARIABLES,
            *FAKE_ENVIRONMENT_VARIABLES,
        ):
            environment.pop(name, None)
        return environment

    def stage_executable(self, path: Path, source: str = FAKE_CHILD) -> None:
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def stage_fake_children(self, child_dir: Path, names: tuple[str, ...]) -> None:
        child_dir.mkdir(parents=True)
        for name in names:
            self.stage_executable(child_dir / name)

    def stage_model_environment(
        self, root: Path, fail_role: str | None = None
    ) -> tuple[dict[str, str], Path, Path, bytes, bytes]:
        original = b"the bytes opened once by the Laguna runner\n"
        replacement = b"different bytes installed at the same pathname\n"
        model = root / "model path with spaces.gguf"
        model.write_bytes(original)
        log = root / "children.jsonl"
        environment = self.clean_gate_environment()
        environment.update(
            {
                "DS4_TEST_MODEL": str(model),
                "DS4_TEST_MODEL_FD": "3",
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT": PINNED_TOKENIZER_COMMIT,
                "LAGUNA_FAKE_ORIGINAL_HEX": original.hex(),
                "LAGUNA_FAKE_REPLACEMENT_HEX": replacement.hex(),
                "LAGUNA_FAKE_LOG": str(log),
            }
        )
        if fail_role is not None:
            environment["LAGUNA_FAKE_FAIL_ROLE"] = fail_role
        return environment, model, log, original, replacement

    def stage_fixture(
        self, root: Path, fail_role: str | None = None
    ) -> tuple[dict[str, str], Path, Path, bytes, bytes]:
        environment, model, log, original, replacement = (
            self.stage_model_environment(root, fail_role)
        )
        child_dir = root / "children with spaces"
        self.stage_fake_children(
            child_dir,
            (
                "compare_laguna_logits.py",
                "test_cuda_laguna_kernels",
                "test_cuda_laguna_model",
                "test_cuda_laguna_stream",
            ),
        )
        environment.update(
            {
                "DS4_LAGUNA_GATE_TEST_CHILD_DIR": str(child_dir),
                "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE": str(len(original)),
                "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256": hashlib.sha256(
                    original
                ).hexdigest(),
            }
        )
        return environment, model, log, original, replacement

    def stage_c7_fixture(
        self, root: Path
    ) -> tuple[dict[str, str], Path, Path, Path, bytes, bytes]:
        environment, model, log, original, replacement = (
            self.stage_model_environment(root)
        )
        staged_root = root / "staged repo"
        tests_dir = staged_root / "tests"
        self.stage_fake_children(
            tests_dir,
            (
                "test_cuda_laguna_kernels",
                "test_cuda_laguna_model",
                "test_cuda_laguna_stream",
            ),
        )
        staged_runner = tests_dir / RUNNER.name
        shutil.copy2(RUNNER, staged_runner)
        verifier_dir = staged_root / "gguf-tools" / "quality-testing"
        self.stage_fake_children(verifier_dir, ("compare_laguna_logits.py",))
        environment["LAGUNA_FAKE_UNLINK_ORIGINAL"] = "1"
        self.stage_fake_timeout(root, environment)
        if os.environ.get("LAGUNA_TEST_REMOVE_STAGED_OFFSET_GUARD") == "1":
            source = staged_runner.read_text(encoding="utf-8")
            offset_probe = "; o = os.lseek(9, 0, os.SEEK_CUR)"
            offset_field = ":{o}"
            self.assertEqual(source.count(offset_probe), 1)
            self.assertEqual(source.count(offset_field), 1)
            staged_runner.write_text(
                source.replace(offset_probe, "", 1).replace(
                    offset_field, "", 1
                ),
                encoding="utf-8",
            )
        return environment, staged_runner, model, log, original, replacement

    def stage_fake_timeout(
        self, root: Path, environment: dict[str, str]
    ) -> Path:
        fake_bin = root / "fake timeout bin"
        fake_bin.mkdir()
        self.stage_executable(fake_bin / "timeout", FAKE_TIMEOUT)
        timeout_log = root / "timeouts.jsonl"
        environment["PATH"] = (
            str(fake_bin) + os.pathsep + environment.get("PATH", "")
        )
        environment["LAGUNA_FAKE_TIMEOUT_LOG"] = str(timeout_log)
        return timeout_log

    def run_runner(
        self,
        mode: str,
        environment: dict[str, str],
        *,
        runner: Path = RUNNER,
        cwd: Path = ROOT,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(runner), mode],
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def run_runner_with_outer_timeout(
        self,
        mode: str,
        environment: dict[str, str],
        *,
        runner: Path,
        cwd: Path,
        outer_timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        arguments = ["/bin/bash", str(runner), mode]
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=outer_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            self.fail(
                f"C7 runner did not bound its hanging child within "
                f"{outer_timeout:.1f}s\nstdout={stdout!r}\nstderr={stderr!r}"
            )
        return subprocess.CompletedProcess(
            arguments, process.returncode, stdout, stderr
        )

    def assert_c7_validation_error(
        self,
        completed: subprocess.CompletedProcess[str],
        diagnostic: str,
    ) -> None:
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(diagnostic, completed.stderr)
        self.assertNotIn("unknown mode", completed.stderr.lower())

    def test_self_test_retains_original_inode_and_bytes_across_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment, model, log, original, replacement = self.stage_fixture(
                Path(tmp)
            )
            original_status = model.stat()

            completed = self.run_runner("self-test", environment)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["role"] for record in records],
                ["verifier", "kernels", "model"],
            )
            for record in records:
                self.assertEqual(record["fd"], 9)
                self.assertEqual(record["device"], original_status.st_dev)
                self.assertEqual(record["inode"], original_status.st_ino)
                self.assertEqual(record["size"], len(original))
                self.assertEqual(record["sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(record["offset"], 0)
            self.assertEqual(model.read_bytes(), replacement)
            self.assertNotEqual(model.stat().st_ino, original_status.st_ino)
            original_path = model.with_name(model.name + ".original")
            self.assertEqual(original_path.read_bytes(), original)
            self.assertEqual(original_path.stat().st_ino, original_status.st_ino)

    def test_child_failure_is_returned_and_later_children_do_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment, _, log, _, _ = self.stage_fixture(
                Path(tmp), fail_role="kernels"
            )

            completed = self.run_runner("self-test", environment)

            self.assertEqual(completed.returncode, 23, completed.stderr)
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["role"] for record in records], ["verifier", "kernels"]
            )

    def test_runner_rejects_identity_change_immediately_after_each_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment, _, log, _, _ = self.stage_fixture(Path(tmp))
            environment["LAGUNA_FAKE_MUTATION"] = "truncate-after-verifier"

            completed = self.run_runner("self-test", environment)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("identity", completed.stderr.lower())
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["role"] for record in records], ["verifier"])

    def test_runner_rehashes_after_the_final_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment, _, log, _, _ = self.stage_fixture(Path(tmp))
            environment["LAGUNA_FAKE_MUTATION"] = "same-size-after-model"

            completed = self.run_runner("self-test", environment)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("sha-256", completed.stderr.lower())
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["role"] for record in records],
                ["verifier", "kernels", "model"],
            )

    def test_c7_requires_an_explicit_existing_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_cwd = Path(tmp)
            default_model = temporary_cwd / "ds4flash.gguf"
            default_model.write_bytes(b"a default-named model exists here\n")
            relative_model = Path("relative inputs") / "explicit.gguf"
            absolute_relative_model = temporary_cwd / relative_model
            absolute_relative_model.parent.mkdir()
            absolute_relative_model.write_bytes(b"relative model bytes\n")
            missing_model = temporary_cwd / "missing" / "explicit.gguf"
            cases = (
                ("unset", None, "DS4_TEST_MODEL is required"),
                (
                    "default literal exists in cwd",
                    "ds4flash.gguf",
                    "DS4_TEST_MODEL must be an explicit path",
                ),
                (
                    "existing relative path",
                    str(relative_model),
                    "DS4_TEST_MODEL must be an absolute path",
                ),
                (
                    "nonexistent explicit path",
                    str(missing_model),
                    "DS4_TEST_MODEL is not a regular file",
                ),
            )
            for label, model, diagnostic in cases:
                with self.subTest(label=label):
                    runner = RUNNER
                    if model == str(relative_model):
                        environment, runner, _, _, _, _ = (
                            self.stage_c7_fixture(temporary_cwd)
                        )
                    else:
                        environment = self.clean_gate_environment()
                        environment["LAGUNA_TOKENIZER_RUNTIME_COMMIT"] = (
                            PINNED_TOKENIZER_COMMIT
                        )
                    if model is not None:
                        environment["DS4_TEST_MODEL"] = model

                    completed = self.run_runner(
                        "c7",
                        environment,
                        runner=runner,
                        cwd=temporary_cwd,
                    )

                    self.assert_c7_validation_error(completed, diagnostic)
                    if model == "ds4flash.gguf":
                        self.assertIn("ds4flash.gguf", completed.stderr)
                    if model == str(relative_model):
                        self.assertNotIn(
                            "no such file", completed.stderr.lower()
                        )
                    if model == str(missing_model):
                        self.assertIn(str(missing_model), completed.stderr)

    def test_c7_requires_a_canonical_tokenizer_runtime_commit(self) -> None:
        invalid_values = (
            ("unset", None, "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required"),
            ("empty", "", "LAGUNA_TOKENIZER_RUNTIME_COMMIT is required"),
            (
                "uppercase",
                PINNED_TOKENIZER_COMMIT.upper(),
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase "
                "hexadecimal characters",
            ),
            (
                "short",
                "0" * 39,
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase "
                "hexadecimal characters",
            ),
            (
                "long",
                "0" * 41,
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase "
                "hexadecimal characters",
            ),
            (
                "nonhex",
                "g" + "0" * 39,
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT must be 40 lowercase "
                "hexadecimal characters",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "explicit-model.gguf"
            model.write_bytes(b"model bytes\n")
            for label, value, diagnostic in invalid_values:
                with self.subTest(label=label):
                    environment = self.clean_gate_environment()
                    environment["DS4_TEST_MODEL"] = str(model)
                    if value is not None:
                        environment["LAGUNA_TOKENIZER_RUNTIME_COMMIT"] = value

                    completed = self.run_runner("c7", environment)

                    self.assert_c7_validation_error(completed, diagnostic)

    def test_c7_rejects_fixture_overrides_before_opening_the_model(self) -> None:
        override_sets = [
            (name, {name: ""}) for name in ISOLATED_FIXTURE_VARIABLES
        ]
        override_sets.append(
            (
                "all",
                {
                    "DS4_LAGUNA_GATE_TEST_CHILD_DIR": "/fixture/children",
                    "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE": "1",
                    "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256": "0" * 64,
                },
            )
        )
        missing_model = "/definitely/missing/laguna-c7-model.gguf"
        for label, overrides in override_sets:
            with self.subTest(label=label):
                environment = self.clean_gate_environment()
                environment.update(
                    {
                        "DS4_TEST_MODEL": missing_model,
                        "LAGUNA_TOKENIZER_RUNTIME_COMMIT": (
                            PINNED_TOKENIZER_COMMIT
                        ),
                        **overrides,
                    }
                )

                completed = self.run_runner("c7", environment)

                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn("forbidden in c7 mode", completed.stderr)
                self.assertNotIn("unknown mode", completed.stderr.lower())
                if label != "all":
                    self.assertIn(label, completed.stderr)
                else:
                    self.assertTrue(
                        any(
                            name in completed.stderr
                            for name in ISOLATED_FIXTURE_VARIABLES
                        ),
                        completed.stderr,
                    )
                self.assertNotIn(missing_model, completed.stderr)
                self.assertNotIn("DS4_TEST_MODEL", completed.stderr)
                self.assertNotIn("regular file", completed.stderr.lower())
                self.assertNotIn("cannot open", completed.stderr.lower())
                self.assertNotIn("no such file", completed.stderr.lower())

    def test_c7_runs_every_child_in_order_on_one_retained_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_root = Path(tmp)
            (
                environment,
                staged_runner,
                model,
                log,
                original,
                replacement,
            ) = self.stage_c7_fixture(temporary_root)
            original_status = model.stat()

            completed = self.run_runner(
                "c7",
                environment,
                runner=staged_runner,
                cwd=temporary_root,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(record["role"], record["case"]) for record in records],
                [
                    ("verifier", None),
                    ("kernels", None),
                    ("model", None),
                    ("stream", "startup"),
                    ("stream", "teardown-unsafe"),
                    ("stream", "model-startup"),
                    ("stream", "model-teardown-unsafe"),
                ],
            )
            verifier_argv = records[0]["argv"]
            self.assertIn("--tokenizer-runtime-commit", verifier_argv)
            tokenizer_index = verifier_argv.index("--tokenizer-runtime-commit")
            self.assertEqual(
                verifier_argv[tokenizer_index + 1], PINNED_TOKENIZER_COMMIT
            )
            self.assertEqual(records[1]["argv"], ["--case", "all"])
            self.assertEqual(records[2]["argv"], [])
            for record in records[3:]:
                self.assertEqual(
                    record["argv"], ["--case", record["case"]]
                )
            self.assertEqual(
                records[-1]["case"],
                "model-teardown-unsafe",
                "model teardown unsafe must be the final gate child",
            )
            self.assertEqual(
                len({record["pid"] for record in records}),
                len(records),
                "every gate child, including each unsafe case, needs its own process",
            )
            for record in records:
                with self.subTest(role=record["role"], case=record["case"]):
                    self.assertEqual(record["fd"], 9)
                    self.assertEqual(record["device"], original_status.st_dev)
                    self.assertEqual(record["inode"], original_status.st_ino)
                    self.assertEqual(record["size"], len(original))
                    self.assertEqual(
                        record["sha256"], hashlib.sha256(original).hexdigest()
                    )
                    self.assertEqual(record["before_offset"], 0)
                    self.assertEqual(record["after_offset"], 0)
            self.assertEqual(model.read_bytes(), replacement)
            self.assertNotEqual(model.stat().st_ino, original_status.st_ino)
            original_path = model.with_name(model.name + ".original")
            self.assertFalse(
                original_path.exists(),
                "the retained descriptor must be the only reference to original bytes",
            )

    def test_c7_budgets_the_verifier_separately_from_cuda_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_root = Path(tmp)
            environment, staged_runner, _, _, _, _ = self.stage_c7_fixture(
                temporary_root
            )
            timeout_log = Path(environment["LAGUNA_FAKE_TIMEOUT_LOG"])

            completed = self.run_runner(
                "c7",
                environment,
                runner=staged_runner,
                cwd=temporary_root,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in timeout_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 7)
            verifier = records[0]
            self.assertEqual(verifier["kill_after"], "5s")
            self.assertEqual(verifier["duration"], "180s")
            self.assertEqual(Path(verifier["command"][0]).name, "python3")
            self.assertEqual(
                Path(verifier["command"][1]).name,
                "compare_laguna_logits.py",
            )
            for record in records[1:]:
                with self.subTest(selector=record["selector"]):
                    self.assertEqual(record["kill_after"], "5s")
                    self.assertEqual(record["duration"], "60s")

    def test_c7_bounds_a_hanging_child_and_stops_before_later_children(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_root = Path(tmp)
            (
                environment,
                staged_runner,
                model,
                log,
                original,
                _,
            ) = self.stage_c7_fixture(temporary_root)
            original_status = model.stat()
            timeout_log = Path(environment["LAGUNA_FAKE_TIMEOUT_LOG"])
            hanging_role = "stream:teardown-unsafe"
            environment["LAGUNA_FAKE_HANG_ROLE"] = hanging_role
            environment["LAGUNA_FAKE_TIMEOUT_ROLE"] = hanging_role

            completed = self.run_runner_with_outer_timeout(
                "c7",
                environment,
                runner=staged_runner,
                cwd=temporary_root,
                outer_timeout=5.0,
            )

            self.assertEqual(completed.returncode, 124, completed.stderr)
            child_records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(record["role"], record["case"]) for record in child_records],
                [
                    ("verifier", None),
                    ("kernels", None),
                    ("model", None),
                    ("stream", "startup"),
                ],
                "the timed-out child and every later child must not complete",
            )
            timeout_records = [
                json.loads(line)
                for line in timeout_log.read_text(encoding="utf-8").splitlines()
            ]
            selected = [
                record
                for record in timeout_records
                if record["selector"] == hanging_role
            ]
            self.assertEqual(
                len(selected),
                1,
                "the runner must invoke the hanging case through timeout",
            )
            timeout_record = selected[0]
            self.assertEqual(timeout_record["kill_after"], "5s")
            self.assertEqual(timeout_record["duration"], "60s")
            self.assertEqual(
                Path(timeout_record["command"][0]).name,
                "test_cuda_laguna_stream",
            )
            self.assertEqual(
                timeout_record["command"][1:],
                ["--case", "teardown-unsafe"],
            )
            self.assertEqual(timeout_record["fd"], 9)
            self.assertEqual(timeout_record["device"], original_status.st_dev)
            self.assertEqual(timeout_record["inode"], original_status.st_ino)
            self.assertEqual(timeout_record["size"], len(original))
            self.assertEqual(
                timeout_record["sha256"], hashlib.sha256(original).hexdigest()
            )
            self.assertEqual(timeout_record["before_offset"], 0)
            self.assertEqual(timeout_record["after_offset"], 0)

    def test_c7_rejects_child_offset_change_before_the_next_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_root = Path(tmp)
            environment, staged_runner, _, log, _, _ = self.stage_c7_fixture(
                temporary_root
            )
            environment["LAGUNA_FAKE_MUTATION"] = (
                "seek-after-stream-startup"
            )

            completed = self.run_runner(
                "c7",
                environment,
                runner=staged_runner,
                cwd=temporary_root,
            )

            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn(
                "retained descriptor identity changed after stream-startup",
                completed.stderr,
            )
            records = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(record["role"], record["case"]) for record in records],
                [
                    ("verifier", None),
                    ("kernels", None),
                    ("model", None),
                    ("stream", "startup"),
                ],
                "offset drift must stop the gate before unsafe teardown",
            )

    def test_resident_rejects_each_self_test_variable_before_opening_fd9(self) -> None:
        names = (
            "DS4_LAGUNA_GATE_TEST_CHILD_DIR",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256",
        )
        for name in names:
            with self.subTest(name=name):
                environment = os.environ.copy()
                environment.update(
                    {
                        "DS4_TEST_MODEL": "/definitely/missing/laguna.gguf",
                        "LAGUNA_TOKENIZER_RUNTIME_COMMIT": PINNED_TOKENIZER_COMMIT,
                        name: "",
                    }
                )
                for other in names:
                    if other != name:
                        environment.pop(other, None)

                completed = self.run_runner("resident", environment)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(name, completed.stderr)
                self.assertNotIn("cannot open", completed.stderr.lower())

    def test_relative_invocation_ignores_hostile_cdpath(self) -> None:
        environment = os.environ.copy()
        environment["CDPATH"] = str(ROOT)
        environment.pop("DS4_TEST_MODEL", None)
        environment.pop("LAGUNA_TOKENIZER_RUNTIME_COMMIT", None)
        for name in (
            "DS4_LAGUNA_GATE_TEST_CHILD_DIR",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256",
        ):
            environment.pop(name, None)

        completed = subprocess.run(
            ["/bin/bash", "tests/run_cuda_laguna_gate.sh", "resident"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("DS4_TEST_MODEL is required", completed.stderr)
        self.assertNotIn("No such file or directory", completed.stderr)

    def test_self_test_rejects_noncanonical_expected_identity(self) -> None:
        invalid = (
            ("DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE", "0"),
            ("DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE", "01"),
            ("DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE", "+1"),
            ("DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256", "A" * 64),
            ("DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256", "0" * 63),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    environment, _, _, _, _ = self.stage_fixture(Path(tmp))
                    environment[name] = value
                    completed = self.run_runner("self-test", environment)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(name, completed.stderr)

    def test_self_test_requires_each_isolated_fixture_variable(self) -> None:
        required = (
            "DS4_LAGUNA_GATE_TEST_CHILD_DIR",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE",
            "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256",
        )
        for name in required:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                environment, _, _, _, _ = self.stage_fixture(Path(tmp))
                environment.pop(name)
                completed = self.run_runner("self-test", environment)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(name, completed.stderr)


if __name__ == "__main__":
    unittest.main()
