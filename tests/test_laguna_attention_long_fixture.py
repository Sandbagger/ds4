#!/usr/bin/env python3
"""Host contract for the compact Poolside long-attention fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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

    def test_tracked_producer_recreates_the_exact_capture_inputs(self) -> None:
        producer = self.manifest["capture"]["producer"]
        for asset_name in ("script", "token_prefix"):
            with self.subTest(asset=asset_name):
                contract = producer[asset_name]
                path = ROOT / contract["path"]
                self.assertEqual(path.stat().st_size, contract["bytes"])
                self.assertEqual(sha256(path), contract["sha256"])

        script = ROOT / producer["script"]["path"]
        base_probe = ROOT / self.manifest["capture"]["base_probe"]["path"]
        token_prefix = ROOT / producer["token_prefix"]["path"]
        token_specification = json.loads(token_prefix.read_text(encoding="utf-8"))
        self.assertEqual(
            token_specification["schema"], producer["token_prefix"]["schema"]
        )
        self.assertEqual(
            token_specification["tokens"], producer["token_prefix"]["tokens"]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            derived_tokens = {}
            for label, case in self.manifest["cases"].items():
                with self.subTest(case=label):
                    producer_case = case["capture_source"]["producer_case"]
                    self.assertEqual(producer_case, label)
                    self.assertEqual(
                        case["token_file"]["prefix_tokens"],
                        case["tokens"],
                    )
                    probe = temporary / f"{label}.cpp"
                    tokens = temporary / f"{label}.tokens.i32"
                    subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "--case",
                            producer_case,
                            "--base-probe",
                            str(base_probe),
                            "--token-prefix",
                            str(token_prefix),
                            "--probe-out",
                            str(probe),
                            "--tokens-out",
                            str(tokens),
                        ],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        probe.stat().st_size,
                        case["capture_source"]["bytes"],
                    )
                    self.assertEqual(
                        sha256(probe),
                        case["capture_source"]["sha256"],
                    )
                    self.assertEqual(
                        tokens.stat().st_size,
                        case["token_file"]["bytes"],
                    )
                    self.assertEqual(
                        sha256(tokens),
                        case["token_file"]["sha256"],
                    )
                    derived_tokens[label] = tokens.read_bytes()
            self.assertEqual(
                derived_tokens["layer1_gqa9_64"],
                derived_tokens["layer0_gqa6_512"][: 64 * 4],
            )

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
