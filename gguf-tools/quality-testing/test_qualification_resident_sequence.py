#!/usr/bin/env python3
"""RED contract tests for the resident qualification sequence builder."""

from __future__ import annotations

import base64
import copy
import hashlib
import unittest

from test_compact_runtime_qualify import TOOL, build_fixture
from qualification_resident_sequence import build_resident_qualification_sequence

PROMPT_TARGETS = (512, 2048, 8192, 28672)
RESIDENT_SCHEMA = "ds4.resident-qualification-sequence/v1"


def _prompt_bytes(manifest: dict, prompt_id: str) -> bytes:
    item = next(prompt for prompt in manifest["prompts"] if prompt["id"] == prompt_id)
    return base64.b64decode(item["rendered_base64"], validate=True)


def _expected_resident_bytes(manifest: dict, prompt_id: str) -> bytes:
    """Independent 24-line oracle; do not call the implementation under test."""
    prompt_bytes = _prompt_bytes(manifest, prompt_id)
    target = int(prompt_id.removeprefix("native-"))
    order_index = PROMPT_TARGETS.index(target)
    lines = [
        f"schema={RESIDENT_SCHEMA}",
        f"manifest_sha256={TOOL.manifest_sha256(manifest)}",
        "profile_id=resident",
        "cache_bytes=0",
        f"prompt_order_index={order_index}",
        f"prompt_id={prompt_id}",
        f"prompt_tokens={target}",
        "mode=resident",
        f"input_size_bytes={len(prompt_bytes)}",
        f"input_sha256={hashlib.sha256(prompt_bytes).hexdigest()}",
        f"input_base64={base64.b64encode(prompt_bytes).decode('ascii')}",
        "max_generated_tokens=512",
        "temperature=0",
        "top_k=0",
        "top_p=1",
        "min_p=0.05",
        "seed=1",
        "stop_sequences_count=0",
        "stop_token_policy=model-native",
        "repetition_count=4",
        "repetition=0:cold",
        "repetition=1:warm-1",
        "repetition=2:warm-2",
        "repetition=3:warm-3",
    ]
    assert len(lines) == 24
    return ("\n".join(lines) + "\n").encode("ascii")


class ResidentQualificationSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = build_fixture()

    def test_byte_exact_independent_oracle_and_all_canonical_prompt_indexes(self) -> None:
        original = copy.deepcopy(self.manifest)
        for order_index, target in enumerate(PROMPT_TARGETS):
            prompt_id = f"native-{target}"
            with self.subTest(prompt_id=prompt_id):
                expected = _expected_resident_bytes(self.manifest, prompt_id)
                actual = build_resident_qualification_sequence(
                    self.manifest, prompt_id
                )
                self.assertIsInstance(actual, bytes)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.count(b"\n"), 24)
                self.assertEqual(
                    actual.splitlines()[4],
                    f"prompt_order_index={order_index}".encode("ascii"),
                )
                expected_digest = hashlib.sha256(expected).hexdigest()
                self.assertEqual(hashlib.sha256(actual).hexdigest(), expected_digest)
                self.assertEqual(actual, build_resident_qualification_sequence(
                    self.manifest, prompt_id
                ))
        self.assertEqual(self.manifest, original)

    def test_resident_constants_and_old_builder_remain_streamed(self) -> None:
        self.assertEqual(tuple(TOOL.PROMPT_TARGETS), PROMPT_TARGETS)
        self.assertTrue(all(profile[0] != "resident" for profile in TOOL.PROFILE_SPECS))
        streamed_before = TOOL.build_qualification_sequence(
            self.manifest, "cache-8gib", "native-512"
        )
        resident = build_resident_qualification_sequence(self.manifest, "native-512")
        streamed_after = TOOL.build_qualification_sequence(
            self.manifest, "cache-8gib", "native-512"
        )
        self.assertEqual(streamed_before, streamed_after)
        self.assertIn(b"schema=ds4.qualification-sequence/v1\n", streamed_before)
        self.assertIn(b"profile_id=cache-8gib\n", streamed_before)
        self.assertIn(b"cache_bytes=8589934592\n", streamed_before)
        self.assertIn(b"mode=streamed\n", streamed_before)
        self.assertNotIn(RESIDENT_SCHEMA.encode("ascii"), streamed_before)
        self.assertIn(f"schema={RESIDENT_SCHEMA}\n".encode("ascii"), resident)
        self.assertIn(b"profile_id=resident\ncache_bytes=0\n", resident)
        self.assertIn(b"mode=resident\n", resident)
        self.assertNotEqual(streamed_before, resident)

    def test_bad_prompt_ids_and_types_are_rejected(self) -> None:
        for prompt_id in ("", "native-999", "512", None, 512, b"native-512", ["native-512"]):
            with self.subTest(prompt_id=prompt_id), self.assertRaises((TypeError, ValueError)):
                build_resident_qualification_sequence(self.manifest, prompt_id)

    def test_mutated_manifests_are_rejected_without_mutating_the_original(self) -> None:
        mutations = (
            ("unknown top-level key", lambda value: value.__setitem__("extra", 1)),
            ("wrong sampling", lambda value: value["sampling"].__setitem__("temperature", 1)),
            ("stale prompt bytes", lambda value: value["prompts"][0].__setitem__(
                "rendered_base64", base64.b64encode(b"changed").decode("ascii")
            )),
            ("reordered profile", lambda value: value["profiles"].__setitem__(
                0, {**value["profiles"][0], "prompt_order": [2048, 512, 28672, 8192]}
            )),
        )
        original = copy.deepcopy(self.manifest)
        for label, mutate in mutations:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                build_resident_qualification_sequence(changed, "native-512")
            self.assertEqual(self.manifest, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
