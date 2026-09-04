#!/usr/bin/env python3
"""RED contract tests for the canonical Laguna qualification evidence index."""

from __future__ import annotations

import copy
import hashlib
import unittest

from qualification_evidence import build_evidence_index, evidence_root_sha256


EMPTY_SHA = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
LARGE_SHA = "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100"
SHARED_SHA = "1111222233334444555566667777888811112222333344445555666677778888"
OTHER_SHA = "9999aaaabbbbccccddddeeeeffff00009999aaaabbbbccccddddeeeeffff0000"
E000_SHA = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
U10000_SHA = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
NFC_SHA = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
NFD_SHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _reference(path: str, digest: str = SHARED_SHA) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _observation(
    path: str, size_bytes: str = "1", digest: str = SHARED_SHA
) -> dict[str, str]:
    return {"path": path, "size_bytes": size_bytes, "sha256": digest}


class QualificationEvidenceTests(unittest.TestCase):
    def _build(self, references, observations, bundle_path: str = "bundle.json") -> bytes:
        return build_evidence_index(
            references, observations, bundle_path=bundle_path
        )

    def _assert_rejected(self, call) -> None:
        with self.assertRaises((TypeError, ValueError)):
            call()

    def test_builds_independent_canonical_bytes_and_hash(self) -> None:
        references = [
            _reference("large.bin", LARGE_SHA),
            _reference("empty.bin", EMPTY_SHA),
        ]
        observations = (
            _observation("empty.bin", "0", EMPTY_SHA),
            _observation("large.bin", "9007199254740993", LARGE_SHA),
        )
        expected = (
            b'[{"path":"empty.bin","sha256":"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff","size_bytes":"0"},'
            b'{"path":"large.bin","sha256":"ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100","size_bytes":"9007199254740993"}]'
        )

        actual = self._build(references, observations)

        self.assertIs(type(actual), bytes)
        self.assertEqual(actual, expected)
        self.assertNotIn(b"\n", actual)
        self.assertEqual(
            evidence_root_sha256(actual), hashlib.sha256(expected).hexdigest()
        )

    def test_shuffled_inputs_deduplicate_same_digest_references_and_do_not_mutate(self) -> None:
        references = [
            _reference("shared.bin"),
            _reference("other.bin", OTHER_SHA),
            _reference("shared.bin"),
        ]
        observations = (
            _observation("shared.bin", "4"),
            _observation("other.bin", "8", OTHER_SHA),
        )
        references_before = copy.deepcopy(references)
        observations_before = copy.deepcopy(observations)
        expected = (
            b'[{"path":"other.bin","sha256":"9999aaaabbbbccccddddeeeeffff00009999aaaabbbbccccddddeeeeffff0000","size_bytes":"8"},'
            b'{"path":"shared.bin","sha256":"1111222233334444555566667777888811112222333344445555666677778888","size_bytes":"4"}]'
        )

        first = self._build(references, observations)
        self.assertEqual(references, references_before)
        self.assertEqual(observations, observations_before)
        second = self._build(tuple(reversed(references)), tuple(reversed(observations)))

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(references, references_before)
        self.assertEqual(observations, observations_before)

    def test_sorts_paths_by_unsigned_utf8_bytes_not_utf16(self) -> None:
        references = [
            _reference("\U00010000", U10000_SHA),
            _reference("\ue000", E000_SHA),
            _reference("ascii", OTHER_SHA),
        ]
        observations = [
            _observation("ascii", "1", OTHER_SHA),
            _observation("\ue000", "2", E000_SHA),
            _observation("\U00010000", "3", U10000_SHA),
        ]
        expected = (
            b'[{"path":"ascii","sha256":"9999aaaabbbbccccddddeeeeffff00009999aaaabbbbccccddddeeeeffff0000","size_bytes":"1"},'
            b'{"path":"\xee\x80\x80","sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","size_bytes":"2"},'
            b'{"path":"\xf0\x90\x80\x80","sha256":"fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210","size_bytes":"3"}]'
        )

        self.assertEqual(self._build(references, observations), expected)

    def test_preserves_distinct_unicode_spellings_without_normalization(self) -> None:
        nfc = "\u00e9.txt"
        nfd = "e\u0301.txt"
        references = [_reference(nfd, NFD_SHA), _reference(nfc, NFC_SHA)]
        observations = [
            _observation(nfc, "2", NFC_SHA),
            _observation(nfd, "3", NFD_SHA),
        ]
        # The decomposed spelling starts with ASCII e (0x65), before 0xc3.
        expected = (
            b'[{"path":"e\xcc\x81.txt","sha256":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef","size_bytes":"3"},'
            b'{"path":"\xc3\xa9.txt","sha256":"1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef","size_bytes":"2"}]'
        )

        self.assertEqual(self._build(references, observations), expected)

    def test_evidence_root_hashes_exact_bytes(self) -> None:
        index_bytes = (
            b'[{"path":"a.bin","sha256":"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff","size_bytes":"0"}]'
        )
        expected_hash = hashlib.sha256(index_bytes).hexdigest()

        self.assertEqual(evidence_root_sha256(index_bytes), expected_hash)
        self.assertNotEqual(
            evidence_root_sha256(index_bytes + b"\n"), expected_hash
        )

    def test_accepts_absent_files_and_full_uint64_declarations(self) -> None:
        references = [_reference("never-created/evidence.bin", LARGE_SHA)]
        observations = [
            _observation(
                "never-created/evidence.bin",
                "18446744073709551615",
                LARGE_SHA,
            )
        ]
        expected = (
            b'[{"path":"never-created/evidence.bin","sha256":"ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100","size_bytes":"18446744073709551615"}]'
        )

        self.assertEqual(
            self._build(references, observations, "also-never-created/bundle.json"),
            expected,
        )

    def test_rejects_empty_missing_extra_and_digest_mismatch_sets(self) -> None:
        ref = _reference("a.bin", EMPTY_SHA)
        obs = _observation("a.bin", "1", EMPTY_SHA)

        cases = [
            ([], []),
            ([ref], []),
            ([], [obs]),
            ([ref], [_observation("b.bin", "1", EMPTY_SHA)]),
            ([ref], [_observation("a.bin", "1", LARGE_SHA)]),
        ]
        for references, observations in cases:
            with self.subTest(references=references, observations=observations):
                self._assert_rejected(
                    lambda references=references, observations=observations: self._build(
                        references, observations
                    )
                )

    def test_rejects_conflicting_references_and_duplicate_observations(self) -> None:
        self._assert_rejected(
            lambda: self._build(
                [_reference("same.bin", EMPTY_SHA), _reference("same.bin", LARGE_SHA)],
                [_observation("same.bin", "1", EMPTY_SHA)],
            )
        )
        self._assert_rejected(
            lambda: self._build(
                [_reference("same.bin", EMPTY_SHA)],
                [
                    _observation("same.bin", "1", EMPTY_SHA),
                    _observation("same.bin", "1", EMPTY_SHA),
                ],
            )
        )

    def test_rejects_non_closed_shapes_and_wrong_types(self) -> None:
        valid_reference = _reference("a.bin", EMPTY_SHA)
        valid_observation = _observation("a.bin", "1", EMPTY_SHA)
        cases = [
            ({"path": "a.bin", "sha256": EMPTY_SHA}, [valid_observation]),
            ([valid_reference], {"path": "a.bin", "size_bytes": "1", "sha256": EMPTY_SHA}),
            ([None], [valid_observation]),
            ([valid_reference], [None]),
            ([{"path": "a.bin"}], [valid_observation]),
            ([{"path": "a.bin", "sha256": EMPTY_SHA, "extra": "no"}], [valid_observation]),
            ([valid_reference], [{"path": "a.bin", "sha256": EMPTY_SHA}]),
            ([valid_reference], [{"path": "a.bin", "size_bytes": "1", "sha256": EMPTY_SHA, "extra": "no"}]),
            ([{"path": 7, "sha256": EMPTY_SHA}], [valid_observation]),
            ([{"path": "a.bin", "sha256": 7}], [valid_observation]),
            ([valid_reference], [{"path": "a.bin", "size_bytes": 1, "sha256": EMPTY_SHA}]),
            ([valid_reference], [{"path": "a.bin", "size_bytes": True, "sha256": EMPTY_SHA}]),
        ]
        for references, observations in cases:
            with self.subTest(references=references, observations=observations):
                self._assert_rejected(
                    lambda references=references, observations=observations: self._build(
                        references, observations
                    )
                )

    def test_rejects_noncanonical_size_and_sha256_strings(self) -> None:
        for size_bytes in (
            "",
            "00",
            "01",
            "+1",
            "-1",
            "1.0",
            " 1",
            "18446744073709551616",
        ):
            with self.subTest(size_bytes=size_bytes):
                self._assert_rejected(
                    lambda size_bytes=size_bytes: self._build(
                        [_reference("a.bin", EMPTY_SHA)],
                        [_observation("a.bin", size_bytes, EMPTY_SHA)],
                    )
                )

        for digest in ("", "0" * 63, "0" * 65, "g" * 64, "A" * 64):
            with self.subTest(digest=digest):
                self._assert_rejected(
                    lambda digest=digest: self._build(
                        [_reference("a.bin", digest)],
                        [_observation("a.bin", "1", digest)],
                    )
                )

    def test_rejects_invalid_reference_paths(self) -> None:
        invalid_paths = (
            "",
            "/absolute",
            "trailing/",
            "double//slash",
            ".",
            "a/./b",
            "..",
            "a/../b",
            "a\x00b",
            "a\x1fb",
            "a\x7fb",
            "a\u0085b",
            "a\ud800b",
        )
        for path in invalid_paths:
            with self.subTest(path=repr(path)):
                self._assert_rejected(
                    lambda path=path: self._build(
                        [_reference(path, EMPTY_SHA)],
                        [_observation(path, "1", EMPTY_SHA)],
                    )
                )

    def test_rejects_invalid_bundle_paths_and_reserved_input_paths(self) -> None:
        valid_references = [_reference("evidence.bin", EMPTY_SHA)]
        valid_observations = [_observation("evidence.bin", "1", EMPTY_SHA)]
        invalid_bundle_paths = (
            "",
            "/absolute",
            "trailing/",
            "double//slash",
            ".",
            "a/./b",
            "..",
            "a/../b",
            "a\x00b",
            "a\x7fb",
            "a\u0085b",
            "a\ud800b",
        )
        for bundle_path in invalid_bundle_paths:
            with self.subTest(bundle_path=repr(bundle_path)):
                self._assert_rejected(
                    lambda bundle_path=bundle_path: self._build(
                        valid_references, valid_observations, bundle_path
                    )
                )

        for bundle_path in (None, b"bundle.json", 7):
            with self.subTest(bundle_path=repr(bundle_path)):
                self._assert_rejected(
                    lambda bundle_path=bundle_path: self._build(
                        valid_references, valid_observations, bundle_path
                    )
                )

        for reserved in (
            "bundle.json",
            "bundle.json.sha256",
            "evidence-index.json",
        ):
            with self.subTest(reserved=reserved):
                self._assert_rejected(
                    lambda reserved=reserved: self._build(
                        [_reference(reserved, EMPTY_SHA)],
                        [_observation(reserved, "1", EMPTY_SHA)],
                        "bundle.json",
                    )
                )
        self._assert_rejected(
            lambda: self._build(
                valid_references, valid_observations, "evidence-index.json"
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
