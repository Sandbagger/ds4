#!/usr/bin/env python3
"""RED contract tests for the pure Laguna qualification schedule helper."""

from __future__ import annotations

import copy
import hashlib
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

from test_compact_runtime_qualify import TOOL, build_fixture
from qualification_resident_sequence import build_resident_qualification_sequence
import qualification_schedule as SCHEDULE
from qualification_schedule import QualificationSlice, build_qualification_schedule


RESIDENT_PROMPT_ORDER = ("native-512", "native-2048", "native-8192", "native-28672")
STREAMED_PROFILE_ORDER = (
    ("cache-8gib", 8 << 30, ("native-512", "native-2048", "native-28672", "native-8192")),
    ("cache-12gib", 12 << 30, ("native-2048", "native-8192", "native-512", "native-28672")),
    ("cache-16gib", 16 << 30, ("native-8192", "native-28672", "native-2048", "native-512")),
)


class QualificationScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = build_fixture()

    def test_exact_order_bindings_and_trusted_builder_delegation(self) -> None:
        original = copy.deepcopy(self.manifest)
        resident_calls: list[tuple[object, str]] = []
        streamed_calls: list[tuple[object, str, str]] = []

        def resident_builder(manifest: object, prompt_id: str) -> bytes:
            resident_calls.append((manifest, prompt_id))
            return f"resident:{prompt_id}".encode("ascii")

        def streamed_builder(manifest: object, profile_id: str, prompt_id: str) -> bytes:
            streamed_calls.append((manifest, profile_id, prompt_id))
            return f"streamed:{profile_id}:{prompt_id}".encode("ascii")

        with (
            mock.patch.object(
                SCHEDULE,
                "build_resident_qualification_sequence",
                side_effect=resident_builder,
            ),
            mock.patch.object(
                SCHEDULE,
                "build_qualification_sequence",
                side_effect=streamed_builder,
            ),
        ):
            actual = build_qualification_schedule(self.manifest)

        expected_bindings = [
            ("resident", "resident", prompt_id, order_index)
            for order_index, prompt_id in enumerate(RESIDENT_PROMPT_ORDER)
        ]
        expected_bindings.extend(
            ("streamed", profile_id, prompt_id, order_index)
            for profile_id, _cache_bytes, prompt_order in STREAMED_PROFILE_ORDER
            for order_index, prompt_id in enumerate(prompt_order)
        )
        self.assertEqual(
            [
                (item.record_kind, item.profile_id, item.prompt_id, item.prompt_order_index)
                for item in actual
            ],
            expected_bindings,
        )
        self.assertEqual(
            [prompt_id for _manifest, prompt_id in resident_calls],
            list(RESIDENT_PROMPT_ORDER),
        )
        self.assertEqual(
            [
                (profile_id, prompt_id)
                for _manifest, profile_id, prompt_id in streamed_calls
            ],
            [
                (profile_id, prompt_id)
                for profile_id, _cache_bytes, prompt_order in STREAMED_PROFILE_ORDER
                for prompt_id in prompt_order
            ],
        )
        self.assertTrue(all(manifest is not self.manifest for manifest, _ in resident_calls))
        self.assertTrue(
            all(manifest is not self.manifest for manifest, _profile, _prompt in streamed_calls)
        )
        manifest_digest = TOOL.manifest_sha256(self.manifest)
        self.assertTrue(all(item.manifest_sha256 == manifest_digest for item in actual))
        self.assertEqual(self.manifest, original)

    def test_canonical_resident_zero_cache_and_distinct_sequence_digests(self) -> None:
        actual = build_qualification_schedule(self.manifest)
        self.assertEqual(len(actual), 16)
        self.assertEqual(
            [item.record_kind for item in actual[:4]], ["resident"] * 4
        )
        self.assertEqual(
            [item.profile_id for item in actual[:4]], ["resident"] * 4
        )
        self.assertEqual(
            [item.prompt_id for item in actual[:4]], list(RESIDENT_PROMPT_ORDER)
        )
        self.assertEqual([item.prompt_order_index for item in actual[:4]], [0, 1, 2, 3])
        for item in actual[:4]:
            self.assertIn(b"schema=ds4.resident-qualification-sequence/v1\n", item.sequence_bytes)
            self.assertIn(b"profile_id=resident\n", item.sequence_bytes)
            self.assertIn(b"cache_bytes=0\n", item.sequence_bytes)
            self.assertIn(b"mode=resident\n", item.sequence_bytes)

        self.assertEqual(
            [item.record_kind for item in actual[4:]], ["streamed"] * 12
        )
        self.assertEqual(
            [item.profile_id for item in actual[4:]],
            [profile_id for profile_id, _cache, prompt_order in STREAMED_PROFILE_ORDER for _ in prompt_order],
        )
        self.assertEqual(
            len({item.sequence_sha256 for item in actual}),
            16,
        )
        for item in actual:
            self.assertEqual(
                item.sequence_sha256,
                hashlib.sha256(item.sequence_bytes).hexdigest(),
            )
            self.assertIs(type(item.sequence_bytes), bytes)

    def test_records_and_payloads_are_immutable(self) -> None:
        actual = build_qualification_schedule(self.manifest)
        self.assertIs(type(actual), tuple)
        self.assertTrue(all(isinstance(item, QualificationSlice) for item in actual))
        with self.assertRaises(TypeError):
            actual[0] = actual[1]  # type: ignore[index]
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            actual[0].profile_id = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            actual[0].sequence_bytes[0] = 0  # type: ignore[index]

    def test_manifest_is_detached_before_builder_delegation(self) -> None:
        original = copy.deepcopy(self.manifest)
        seen: list[object] = []

        def resident_builder(manifest: dict, _prompt_id: str) -> bytes:
            seen.append(manifest)
            manifest["sampling"]["temperature"] = 99
            return b"resident"

        def streamed_builder(manifest: dict, _profile_id: str, _prompt_id: str) -> bytes:
            seen.append(manifest)
            manifest["sampling"]["temperature"] = 98
            return b"streamed"

        with (
            mock.patch.object(
                SCHEDULE,
                "build_resident_qualification_sequence",
                side_effect=resident_builder,
            ),
            mock.patch.object(
                SCHEDULE,
                "build_qualification_sequence",
                side_effect=streamed_builder,
            ),
        ):
            build_qualification_schedule(self.manifest)

        self.assertEqual(self.manifest, original)
        self.assertEqual(len(seen), 16)
        self.assertTrue(all(manifest is not self.manifest for manifest in seen))

    def test_invalid_manifest_is_rejected_before_any_sequence_build(self) -> None:
        mutations = (
            lambda value: value.__setitem__("extra", 1),
            lambda value: value.__delitem__("sampling"),
            lambda value: value["profiles"].__setitem__(
                0, {**value["profiles"][0], "prompt_order": [2048, 512, 28672, 8192]}
            ),
            lambda value: value["prompts"][0].__setitem__(
                "rendered_base64", "Y2hhbmdlZA=="
            ),
        )
        with (
            mock.patch.object(SCHEDULE, "build_resident_qualification_sequence") as resident,
            mock.patch.object(SCHEDULE, "build_qualification_sequence") as streamed,
        ):
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    invalid = copy.deepcopy(self.manifest)
                    mutate(invalid)
                    with self.assertRaises(ValueError):
                        build_qualification_schedule(invalid)
            resident.assert_not_called()
            streamed.assert_not_called()

    def test_legacy_sequence_builders_are_unchanged(self) -> None:
        resident_before = build_resident_qualification_sequence(self.manifest, "native-512")
        streamed_before = TOOL.build_qualification_sequence(
            self.manifest, "cache-8gib", "native-512"
        )
        build_qualification_schedule(self.manifest)
        self.assertEqual(
            build_resident_qualification_sequence(self.manifest, "native-512"),
            resident_before,
        )
        self.assertEqual(
            TOOL.build_qualification_sequence(self.manifest, "cache-8gib", "native-512"),
            streamed_before,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
