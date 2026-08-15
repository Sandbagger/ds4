#!/usr/bin/env python3
"""Offline contract tests for revision-qualified Laguna downloads."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = ROOT / "download_model.sh"
LAGUNA_REVISION = "e2ccc0579fc18e6ea2362fa25fccbcd470f0e332"
LAGUNA_FILENAME = "laguna-s-2.1-Q4_K_M.gguf"
LAGUNA_SIZE_BYTES = 68_248_760_064
LAGUNA_SHA256 = "a34c74e46688122bef83122f4133031bababbefcf57436dde97048c91e2cc6ff"


class LagunaDownloadContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.script = self.root / "download_model.sh"
        shutil.copy2(DOWNLOAD_SCRIPT, self.script)
        self.out_dir = self.root / "models"
        self.home = self.root / "home"
        self.home.mkdir()
        self.fake_bin = self.root / "fake-bin"
        self.fake_bin.mkdir()
        self.hf_log = self.root / "hf.log"
        self.active_model = self.root / "active-old-revision.gguf"
        self.active_model.write_bytes(b"active-old-revision")
        (self.root / "ds4flash.gguf").symlink_to(self.active_model.name)

        self._write_executable(
            self.fake_bin / "hf",
            """#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_HF_LOG"
[ "$1" = download ]
file=$3
local_dir=
revision=
shift 3
while [ "$#" -gt 0 ]; do
    case "$1" in
        --local-dir)
            shift
            local_dir=$1
            ;;
        --revision)
            shift
            revision=$1
            ;;
    esac
    shift
done
[ "$revision" = "$FAKE_EXPECTED_REVISION" ]
[ -n "$local_dir" ]
mkdir -p "$local_dir"
"$FAKE_PYTHON" -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.touch(); p.open("r+b").truncate(int(sys.argv[2]))' "$local_dir/$file" "$FAKE_HF_SIZE"
""",
        )
        self._write_executable(
            self.fake_bin / "sha256sum",
            """#!/bin/sh
set -eu
printf '%s  %s\n' "$FAKE_SHA256" "$1"
""",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_executable(path: Path, payload: str) -> None:
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o755)

    @property
    def staged_path(self) -> Path:
        return (
            self.out_dir
            / "poolside"
            / "Laguna-S-2.1-GGUF"
            / LAGUNA_REVISION
            / LAGUNA_FILENAME
        )

    def _run(self, *, size: int, sha256: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "DS4_GGUF_DIR": str(self.out_dir),
                "FAKE_EXPECTED_REVISION": LAGUNA_REVISION,
                "FAKE_HF_LOG": str(self.hf_log),
                "FAKE_HF_SIZE": str(size),
                "FAKE_PYTHON": sys.executable,
                "FAKE_SHA256": sha256,
                "HOME": str(self.home),
                "PATH": f"{self.fake_bin}{os.pathsep}{environment['PATH']}",
            }
        )
        return subprocess.run(
            [str(self.script), "laguna-q4"],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_old_flat_artifact_is_not_reused_and_verified_stage_is_reused(self) -> None:
        self.out_dir.mkdir()
        flat = self.out_dir / LAGUNA_FILENAME
        flat.write_bytes(b"old-same-name-artifact")
        active_before = os.readlink(self.root / "ds4flash.gguf")

        first = self._run(size=LAGUNA_SIZE_BYTES, sha256=LAGUNA_SHA256)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(flat.read_bytes(), b"old-same-name-artifact")
        self.assertEqual(self.staged_path.stat().st_size, LAGUNA_SIZE_BYTES)
        self.assertEqual(os.readlink(self.root / "ds4flash.gguf"), active_before)
        self.assertNotIn(f"Already downloaded: {flat}", first.stdout)
        self.assertIn(str(self.staged_path), first.stdout)
        self.assertEqual(len(self.hf_log.read_text(encoding="utf-8").splitlines()), 1)

        rejected_reuse = self._run(size=1, sha256="b" * 64)
        self.assertNotEqual(rejected_reuse.returncode, 0)
        self.assertIn("sha-256", rejected_reuse.stderr.lower())
        self.assertEqual(len(self.hf_log.read_text(encoding="utf-8").splitlines()), 1)

        second = self._run(size=1, sha256=LAGUNA_SHA256)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("Reusing verified", second.stdout)
        self.assertEqual(len(self.hf_log.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(os.readlink(self.root / "ds4flash.gguf"), active_before)

    def test_wrong_downloaded_size_is_rejected_before_publish(self) -> None:
        result = self._run(size=LAGUNA_SIZE_BYTES - 1, sha256=LAGUNA_SHA256)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("size", result.stderr.lower())
        self.assertFalse(self.staged_path.exists())

    def test_wrong_downloaded_sha256_is_rejected_before_publish(self) -> None:
        result = self._run(size=LAGUNA_SIZE_BYTES, sha256="b" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha-256", result.stderr.lower())
        self.assertFalse(self.staged_path.exists())


if __name__ == "__main__":
    unittest.main()
