#!/usr/bin/env python3
"""Fail-closed contract for the immutable Laguna 706 resident oracle fixture."""

from __future__ import annotations

import hashlib
import json
import stat
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "706fa69799926b6afde1af9e24ca2a4923f110a1"
HISTORY = ROOT / "tests/test-vectors/laguna-resident-history" / REVISION
EXPECTED = {
    "benchmark-32768.txt": (303104, "aa352ad2890413cf112abc10d7349db3ef4be4c3722e2276f943e7b413a59206"),
    "cases.json": (676, "b2deb66881e4a0fec1f86a8faa26ae065f23075738e09bcbbf75ba97996f8383"),
    "deep-32768.llama.f32": (401408, "f95b44b84a2dc00580cc31d87591b47c6a795d993da4c415fb8c724c9e27d3bb"),
    "deep-32768.prompt": (161653, "7548f54d4e8569bb14fed088b9c7a6326d139a8045a7ede033dbec9a8aa739cb"),
    "generate_benchmark_prompt.py": (902, "118f1223ad248f845acd0dcb69444f911a3a6843d548db866a73a1106d7c5e3d"),
    "manifest.json": (1552570, "2e7290855500b6a848a900f01aaf9d821dceb16769d9f1f16ad55229c9c77e0a"),
    "short.llama.f32": (401408, "e709da464982f858a69090e9a1d3ecca8ef945627dffc7c180a997a707259e06"),
    "short.txt": (51, "58a116e251d40c0e95bb87fb799cc69b208a03d550c0a86dec5c96c0d4dcf44d"),
    "swa-513.llama.f32": (401408, "d984734d99380848364f70fe0bb7fbb6c99d1bdf52cfc4116ecba84f6aa468d4"),
    "swa-513.prompt": (2531, "bad7b67cdacd5c619a811a5812b441f854fac978cbbd59f9dc08dd85ea9cdaf6"),
    "yarn-8193.continuation.i32": (32, "4579f73ea645998e69f72d07414b19d2d7f8c73fa52bf54d94e7891481f7b6bc"),
    "yarn-8193.llama.f32": (401408, "61f61ef87eb96f0d0af7ae7d5b947160e798ad57b588841fc385ceaecffc4687"),
    "yarn-8193.prompt": (40419, "019f2d68cf9912175876388c8cda020135ab4951c816f10d312ce5b528a27f85"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class LagunaResidentHistoryContractTest(unittest.TestCase):
    def test_706_fixture_is_an_exact_immutable_archive(self) -> None:
        self.assertTrue(HISTORY.is_dir(), f"missing historical fixture: {HISTORY}")
        found = {path.name for path in HISTORY.iterdir()}
        self.assertEqual(found, set(EXPECTED))
        for name, (expected_size, expected_sha256) in EXPECTED.items():
            with self.subTest(name=name):
                path = HISTORY / name
                mode = path.lstat().st_mode
                self.assertTrue(stat.S_ISREG(mode))
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_size, expected_size)
                self.assertEqual(sha256(path), expected_sha256)

        manifest = json.loads((HISTORY / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["model"],
            {
                "filename": "laguna-s-2.1-Q4_K_M.gguf",
                "repository": "poolside/Laguna-S-2.1-GGUF",
                "revision": REVISION,
                "sha256": "e163b2c98908809a71245d6bb68b2226994d9969cb2a438eccb72196a1c4147a",
                "size": 68248759648,
            },
        )


if __name__ == "__main__":
    unittest.main()
