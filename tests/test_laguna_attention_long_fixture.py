#!/usr/bin/env python3
"""Host contract for the compact Poolside long-attention fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/test-vectors/laguna-attention-auto-long"
MANIFEST = FIXTURE / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LagunaAttentionLongFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_pins_the_reference_identity(self) -> None:
        self.assertEqual(
            self.manifest["schema"],
            "laguna-attention-long-auto-fixture/v1",
        )
        self.assertEqual(
            self.manifest["poolside_commit"],
            "04b2b72cb54048ead292884adbe11f284e3ec950",
        )
        self.assertEqual(self.manifest["device"]["compute_capability"], "12.1")
        self.assertEqual(self.manifest["capture"]["flash_attention"], "AUTO")
        base = self.manifest["capture"]["base_probe"]
        path = ROOT / base["path"]
        self.assertEqual(path.stat().st_size, base["bytes"])
        self.assertEqual(sha256(path), base["sha256"])

    def test_every_compact_file_is_exactly_pinned(self) -> None:
        expected = self.manifest["files"]
        actual = {path.name for path in FIXTURE.glob("*.f32")}
        self.assertEqual(actual, set(expected))
        for name, contract in expected.items():
            with self.subTest(file=name):
                path = FIXTURE / name
                self.assertEqual(path.stat().st_size, contract["bytes"])
                self.assertEqual(sha256(path), contract["sha256"])

    def test_extractions_are_bounded_and_preserve_gqa_mapping(self) -> None:
        for label, case in self.manifest["cases"].items():
            topology = case["topology"]
            ratio = topology["gqa_ratio"]
            mapped = [head // ratio for head in topology["selected_query_heads"]]
            with self.subTest(case=label, contract="gqa"):
                self.assertEqual(mapped, topology["selected_kv_heads"])
                self.assertEqual(topology["nbatch_fa"], 64)
                self.assertEqual(
                    topology["ncols1"] * topology["ncols2"], 64
                )
            for name, extraction in case["extractions"].items():
                with self.subTest(case=label, extraction=name):
                    raw_bytes = case["raw"][extraction["source"]]["bytes"]
                    if "segments" in extraction:
                        end = (
                            extraction["offset"]
                            + (extraction["segments"] - 1)
                            * extraction["stride"]
                            + extraction["segment_bytes"]
                        )
                        extracted = (
                            extraction["segments"]
                            * extraction["segment_bytes"]
                        )
                    else:
                        end = extraction["offset"] + extraction["bytes"]
                        extracted = extraction["bytes"]
                    self.assertLessEqual(end, raw_bytes)
                    self.assertEqual(
                        extracted, self.manifest["files"][name]["bytes"]
                    )


if __name__ == "__main__":
    unittest.main()
