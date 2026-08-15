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
            "laguna-attention-long-auto-fixture/v2",
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

    def test_token513_decode_suffix_pins_corrected_poolside_fattn_vec(self) -> None:
        self.assertEqual(
            self.manifest["corrected_model"],
            {
                "repository": "poolside/Laguna-S-2.1-GGUF",
                "revision": "e2ccc0579fc18e6ea2362fa25fccbcd470f0e332",
                "bytes": 68248760064,
                "sha256": (
                    "a34c74e46688122bef83122f4133031bababbefcf57436dde"
                    "97048c91e2cc6ff"
                ),
            },
        )
        case = self.manifest["cases"]["layer0_gqa6_token513_decode"]
        self.assertEqual(case["model_revision"], self.manifest["corrected_model"]["revision"])
        self.assertEqual(
            case["topology"],
            {
                "query_heads": 48,
                "kv_heads": 8,
                "gqa_ratio": 6,
                "head_dim": 128,
                "selected_query_heads": [41, 42],
                "selected_kv_heads": [6, 7],
                "query_position": 512,
                "key_start": 0,
                "key_count": 513,
                "padded_key_count": 768,
                "partitions": 3,
            },
        )
        self.assertEqual(
            case["raw"],
            {
                "layer-00-q-rope.f32": {
                    "bytes": 24576,
                    "sha256": "7653cd1a6b87a5f56af2f41dfb11a8bdc341280b837a6e354e51994184269833",
                },
                "layer-00-k-rope.f32": {
                    "bytes": 4096,
                    "sha256": "6f81f739d9c9229fffb46c70ae5333e6285ed95ef099b2089f0231d7cb84cf36",
                },
                "layer-00-v-proj.f32": {
                    "bytes": 4096,
                    "sha256": "3dcc8f657b7bcd41a307a3658a3f515d265da92201f8066b815f08f0c01bfc76",
                },
                "layer-00-gate-proj.f32": {
                    "bytes": 192,
                    "sha256": "b1fada6a8d2bd399e680b39087cbe9947e1fdc6f06eadae25a32c308cf09f496",
                },
                "layer-00-attn-gated.f32": {
                    "bytes": 24576,
                    "sha256": "ec2e30b5fa6b19848d0125badfa2814df4f20dfcf7bd67b17c3c753fe8e509ec",
                },
            },
        )
        self.assertEqual(
            case["extractions"],
            {
                "layer-00-q-t512-h41-h42.f32": {
                    "source": "layer-00-q-rope.f32",
                    "offset": 20992,
                    "bytes": 1024,
                },
                "layer-00-k-t512-kv6-kv7.f32": {
                    "source": "layer-00-k-rope.f32",
                    "offset": 3072,
                    "bytes": 1024,
                },
                "layer-00-v-t512-kv6-kv7.f32": {
                    "source": "layer-00-v-proj.f32",
                    "offset": 3072,
                    "bytes": 1024,
                },
                "layer-00-gate-t512-h41-h42.f32": {
                    "source": "layer-00-gate-proj.f32",
                    "offset": 164,
                    "bytes": 8,
                },
                "layer-00-attn-gated-t512-h41-h42.f32": {
                    "source": "layer-00-attn-gated.f32",
                    "offset": 20992,
                    "bytes": 1024,
                },
            },
        )

        prefix = case["prefix_provenance"]
        self.assertEqual(prefix["kind"], "tensor-payload-equivalent-reuse")
        self.assertEqual(prefix["tokens"]["count"], 512)
        self.assertEqual(
            prefix["tokens"]["little_endian_i32_sha256"],
            "569aa6394783e0f17558db421ba26480d7a530d44dd2219bcc9aac2c09a3b559",
        )
        self.assertEqual(
            prefix["reused_files"],
            {
                "key": {
                    "file": "layer-00-k-t0-t511-kv6-kv7.f32",
                    "bytes": 524288,
                    "sha256": "55b783cd0b8f132cc48f78b66910a4de717f4943c218f5f912dfd36385a1df30",
                },
                "value": {
                    "file": "layer-00-v-t0-t511-kv6-kv7.f32",
                    "bytes": 524288,
                    "sha256": "1b4d52270e650a9c48b4e4a19a400248d69e2e9a882919d22a3bb488bcd5ef7d",
                },
            },
        )
        for reused in prefix["reused_files"].values():
            tracked = self.manifest["files"][reused["file"]]
            self.assertEqual(reused["bytes"], tracked["bytes"])
            self.assertEqual(reused["sha256"], tracked["sha256"])
        attestation = prefix["tensor_payload_attestation"]
        self.assertEqual(attestation["tensor_count"], 814)
        self.assertTrue(attestation["tensor_layout_equal"])
        self.assertTrue(attestation["tensor_payloads_equal"])
        self.assertEqual(attestation["metadata_differences"], ["tokenizer.chat_template"])
        self.assertEqual(
            attestation["external_evidence"],
            {
                "path": (
                    "/home/will/.local/state/ds4-laguna-qualification/"
                    "gguf-comparison-706fa697-e2ccc057.json"
                ),
                "bytes": 462250,
                "sha256": "512bf7c899840b93c414439fed4209ae3958902d59cd284700051aef681d1f19",
            },
        )
        self.assertFalse(prefix["corrected_512_row_callback_recapture"])

        resources = case["fattn_vec_resources"]
        self.assertEqual(
            resources["kernel"],
            {"registers_per_thread": 162, "static_shared_bytes": 8448},
        )
        self.assertEqual(resources["launch"], {"block": [32, 4, 1], "threads": 128, "grid": [1, 3, 48]})
        self.assertEqual(
            resources["gb10_observed_limits"],
            {
                "compute_capability": "12.1",
                "multiprocessor_count": 48,
                "registers_per_multiprocessor": 65536,
                "shared_bytes_per_multiprocessor": 102400,
                "threads_per_multiprocessor": 1536,
                "blocks_per_multiprocessor": 24,
            },
        )
        self.assertEqual(resources["resident_blocks_per_multiprocessor"], 3)
        self.assertEqual(
            self.manifest["poolside_fattn_sources"],
            {
                "ggml/src/ggml-cuda/fattn.cu": {
                    "bytes": 23739,
                    "sha256": "dedfdb596af4ab514f2af25ed73d0faf5b9453b9b84e6fc7d7adc8cc37c4f776",
                },
                "ggml/src/ggml-cuda/fattn-common.cuh": {
                    "bytes": 47668,
                    "sha256": "47537d7980d81f7dc9daa18698f5fdbb990ef6b916fa75fed4bc0bbfd1aa08cb",
                },
                "ggml/src/ggml-cuda/fattn-vec.cuh": {
                    "bytes": 24989,
                    "sha256": "f6305c9a667f438565ee0afff0e854cce335ddbd67e30fbee466cbc5672a9577",
                },
            },
        )

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
                if "capture_source" not in case:
                    continue
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
                if "nbatch_fa" in topology:
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
