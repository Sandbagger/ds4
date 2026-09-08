#!/usr/bin/env python3
"""Model-free frontend rejection through the native fake-backend harness."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tests/test_bench_qualification_lifecycle"


class ResidentArgumentContract(unittest.TestCase):
    def argv(self):
        return ["--qualification-resident-sequence", "/literal/resident-sequence.txt",
                "--qualification-manifest-sha256", "d" * 64,
                "--qualification-sequence-sha256", "e" * 64,
                "--model", "/literal/fake.gguf", "--backend", "cuda",
                "--qualification-control-fd", "9"]

    def rejected(self, args):
        result = subprocess.run(
            [str(TOOL), "--probe-argv-rejection", *args], cwd=ROOT,
            capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("argument rejection reached parser/backend", result.stderr)

    def test_conflicting_and_duplicate_sequence_owners(self):
        for extra in (["--qualification-sequence", "/literal/sequence.txt"],
                      ["--qualification-resident-sequence", "/literal/resident-sequence.txt"],
                      ["--qualification-manifest-sha256", "d" * 64],
                      ["--qualification-sequence-sha256", "e" * 64]):
            with self.subTest(extra=extra):
                self.rejected(self.argv() + extra)

    def test_incomplete_triplets(self):
        for offset in (0, 2, 4):
            with self.subTest(offset=offset):
                args = self.argv()
                del args[offset:offset + 2]
                self.rejected(args)

    def test_duplicate_model_and_backend_selectors(self):
        for extra in (["--model", "/literal/fake.gguf"],
                      ["-m", "/literal/fake.gguf"],
                      ["--backend", "cuda"], ["--cuda"]):
            with self.subTest(extra=extra):
                self.rejected(self.argv() + extra)

    def test_makefile_keeps_resident_checks_in_aggregate_and_cleanup(self):
        text = (ROOT / "Makefile").read_text().replace("\\\n", " ")
        aggregate = next(line for line in text.splitlines() if line.startswith("test:"))
        self.assertIn("test-laguna-resident-path", aggregate.split())
        cleanup = text.split("\nclean:\n", 1)[1]
        for name in ("tests/test_bench_resident_frontend", "tests/test_laguna_resident_plan"):
            self.assertIn(name, cleanup)

    def test_production_refuses_unimplemented_resident_accounting(self):
        # A real, separately named CPU-only frontend, without fake APIs or the
        # test-backend macro. Its common production gate also covers CUDA.
        payload = ("schema=ds4.resident-qualification-sequence/v1\n"
                   f"manifest_sha256={'a' * 64}\n"
                   "profile_id=resident\ncache_bytes=0\nprompt_order_index=0\n"
                   "prompt_id=native-512\nprompt_tokens=512\nmode=resident\n"
                   "input_size_bytes=5\n"
                   "input_sha256=2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\n"
                   "input_base64=aGVsbG8=\nmax_generated_tokens=512\n"
                   "temperature=0\ntop_k=0\ntop_p=1\nmin_p=0.05\nseed=1\n"
                   "stop_sequences_count=0\nstop_token_policy=model-native\n"
                   "repetition_count=4\nrepetition=0:cold\nrepetition=1:warm-1\n"
                   "repetition=2:warm-2\nrepetition=3:warm-3\n").encode()
        with tempfile.TemporaryDirectory(prefix="resident-frontend-") as name:
            directory = Path(name)
            sequence = directory / "resident.txt"
            sequence.write_bytes(payload)
            lock = directory / "must-not-open.lock"
            environment = dict(os.environ, DS4_LOCK_FILE=str(lock))
            result = subprocess.run(
                [str(ROOT / "tests/test_bench_resident_frontend"),
                 "--qualification-resident-sequence", str(sequence),
                 "--qualification-manifest-sha256", "a" * 64,
                 "--qualification-sequence-sha256", hashlib.sha256(payload).hexdigest(),
                 "--model", str(directory / "missing.gguf"), "--backend", "cuda",
                 "--qualification-control-fd", "9"], cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=5, check=False)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("resident runtime accounting is not implemented", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(lock.exists())

    def test_resident_forbids_even_matching_or_ignored_overrides(self):
        extras = (["--ssd-streaming"], ["--ssd-streaming-cold"],
                  ["--ssd-streaming-cache-bytes", "8589934592"],
                  ["--ssd-streaming-cache-experts", "1"],
                  ["--ssd-streaming-full-layers", "0"],
                  ["--ssd-streaming-preload-experts", "1"],
                  ["--prefill-chunk", "4096"], ["--ctx-alloc", "32768"],
                  ["--ctx-start", "512"], ["--gen-tokens", "1"],
                  ["--quality"], ["--warm-weights"], ["--threads", "1"],
                  ["--gpu-vram", "1"], ["--gpu-devices", "0"],
                  ["--cuda-tensor-parallel"], ["--csv", "/literal/out.csv"],
                  ["--show-output"], ["--backend", "cpu"])
        for extra in extras:
            with self.subTest(extra=extra):
                self.rejected(self.argv() + extra)


if __name__ == "__main__":
    unittest.main()
