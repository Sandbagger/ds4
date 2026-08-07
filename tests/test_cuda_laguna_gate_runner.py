#!/usr/bin/env python3
"""Behavioral contract for the one-descriptor Laguna CUDA gate runner."""

from __future__ import annotations

import hashlib
import json
import os
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

FAKE_CHILD = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

role = Path(sys.argv[0]).name
roles = {
    "compare_laguna_logits.py": "verifier",
    "test_cuda_laguna_kernels": "kernels",
    "test_cuda_laguna_model": "model",
}
if role not in roles:
    raise SystemExit(f"unexpected fake child name: {role}")
role = roles[role]
if os.environ.get("DS4_TEST_MODEL_FD") != "9":
    raise SystemExit(f"{role}: DS4_TEST_MODEL_FD was not forced to 9")
fd = 9
before_offset = os.lseek(fd, 0, os.SEEK_CUR)
status = os.fstat(fd)
if not stat.S_ISREG(status.st_mode):
    raise SystemExit(f"{role}: fd 9 is not regular")
payload = os.pread(fd, status.st_size, 0)
after_offset = os.lseek(fd, 0, os.SEEK_CUR)
expected = bytes.fromhex(os.environ["DS4_LAGUNA_GATE_TEST_ORIGINAL_HEX"])
if payload != expected or before_offset != after_offset:
    raise SystemExit(f"{role}: fd 9 bytes or offset changed")
if role == "verifier":
    if "--gguf-fd" not in sys.argv or sys.argv[sys.argv.index("--gguf-fd") + 1] != "9":
        raise SystemExit("verifier did not receive --gguf-fd 9")
    if "--gguf-size" not in sys.argv or "--gguf-sha256" not in sys.argv:
        raise SystemExit("verifier did not receive the expected identity")
elif role == "kernels" and sys.argv[1:] != ["--case", "all"]:
    raise SystemExit(f"kernel argv mismatch: {sys.argv[1:]!r}")
elif role == "model" and sys.argv[1:]:
    raise SystemExit(f"model argv mismatch: {sys.argv[1:]!r}")

record = {
    "role": role,
    "fd": fd,
    "device": status.st_dev,
    "inode": status.st_ino,
    "size": status.st_size,
    "sha256": hashlib.sha256(payload).hexdigest(),
    "offset": after_offset,
}
with Path(os.environ["DS4_LAGUNA_GATE_TEST_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")

if role == "verifier":
    model = Path(os.environ["DS4_TEST_MODEL"])
    original = model.with_name(model.name + ".original")
    model.rename(original)
    model.write_bytes(bytes.fromhex(os.environ["DS4_LAGUNA_GATE_TEST_REPLACEMENT_HEX"]))

mutation = os.environ.get("DS4_LAGUNA_GATE_TEST_MUTATION")
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

if os.environ.get("DS4_LAGUNA_GATE_TEST_FAIL_ROLE") == role:
    raise SystemExit(23)
'''


class LagunaGateRunnerTest(unittest.TestCase):
    def stage_fixture(
        self, root: Path, fail_role: str | None = None
    ) -> tuple[dict[str, str], Path, Path, bytes, bytes]:
        child_dir = root / "children with spaces"
        child_dir.mkdir()
        for name in (
            "compare_laguna_logits.py",
            "test_cuda_laguna_kernels",
            "test_cuda_laguna_model",
        ):
            child = child_dir / name
            child.write_text(textwrap.dedent(FAKE_CHILD), encoding="utf-8")
            child.chmod(child.stat().st_mode | stat.S_IXUSR)

        original = b"the bytes opened once by the Laguna runner\n"
        replacement = b"different bytes installed at the same pathname\n"
        model = root / "model path with spaces.gguf"
        model.write_bytes(original)
        log = root / "children.jsonl"
        environment = os.environ.copy()
        environment.update(
            {
                "DS4_TEST_MODEL": str(model),
                "DS4_TEST_MODEL_FD": "3",
                "LAGUNA_TOKENIZER_RUNTIME_COMMIT": PINNED_TOKENIZER_COMMIT,
                "DS4_LAGUNA_GATE_TEST_CHILD_DIR": str(child_dir),
                "DS4_LAGUNA_GATE_TEST_EXPECTED_SIZE": str(len(original)),
                "DS4_LAGUNA_GATE_TEST_EXPECTED_SHA256": hashlib.sha256(
                    original
                ).hexdigest(),
                "DS4_LAGUNA_GATE_TEST_ORIGINAL_HEX": original.hex(),
                "DS4_LAGUNA_GATE_TEST_REPLACEMENT_HEX": replacement.hex(),
                "DS4_LAGUNA_GATE_TEST_LOG": str(log),
            }
        )
        if fail_role is not None:
            environment["DS4_LAGUNA_GATE_TEST_FAIL_ROLE"] = fail_role
        return environment, model, log, original, replacement

    def run_runner(
        self, mode: str, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(RUNNER), mode],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

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
            environment["DS4_LAGUNA_GATE_TEST_MUTATION"] = (
                "truncate-after-verifier"
            )

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
            environment["DS4_LAGUNA_GATE_TEST_MUTATION"] = (
                "same-size-after-model"
            )

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
