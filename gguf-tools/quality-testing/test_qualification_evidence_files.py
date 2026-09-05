#!/usr/bin/env python3
"""Contract tests for the root-owned qualification evidence verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qualification_evidence import build_evidence_index
import qualification_evidence_files as qef
from qualification_evidence_files import verify_evidence_files


BUNDLE_PATH = "bundle.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reference(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _observation(path: str, size_bytes: str, digest: str) -> dict[str, str]:
    return {"path": path, "size_bytes": size_bytes, "sha256": digest}


def _write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _index_for(
    root: Path,
    references: list[dict[str, str]] | tuple[dict[str, str], ...],
    *,
    bundle_path: str = BUNDLE_PATH,
) -> bytes:
    """Build a canonical index from the bytes in *root* for test fixtures."""

    observations: list[dict[str, str]] = []
    seen: set[str] = set()
    for reference in references:
        path = reference["path"]
        if path in seen:
            continue
        seen.add(path)
        data = (root / path).read_bytes()
        observations.append(
            _observation(path, str(len(data)), _sha256(data))
        )
    return build_evidence_index(
        references, observations, bundle_path=bundle_path
    )


def _index_from_entries(entries: list[dict[str, object]]) -> bytes:
    """Return deliberately non-canonical JSON for malformed-index tests."""

    return json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


class QualificationEvidenceFilesTests(unittest.TestCase):
    def _assert_rejected(self, call) -> None:
        with self.assertRaises((TypeError, ValueError, OSError)):
            call()

    def _verify(
        self,
        root: Path | str,
        references: list[dict[str, str]] | tuple[dict[str, str], ...],
        index_bytes: bytes,
        *,
        bundle_path: str = BUNDLE_PATH,
    ) -> str:
        return verify_evidence_files(
            root, references, index_bytes, bundle_path=bundle_path
        )

    def test_make_runs_file_verifier_in_pinned_environment_and_aggregate(self) -> None:
        import shlex
        import subprocess

        root = Path(__file__).resolve().parents[2]
        command = ["uv", "run", "--with-requirements",
                   "gguf-tools/quality-testing/requirements-compact-runtime.txt",
                   "python", "gguf-tools/quality-testing/test_qualification_evidence_files.py",
                   "-v"]
        for target in ("test-qualification-evidence-files", "test-laguna-compact-python"):
            with self.subTest(target=target):
                result = subprocess.run(
                    ["make", "--no-print-directory", "--dry-run", target],
                    cwd=root, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [shlex.split(line) for line in
                            result.stdout.replace("\\\n", " ").splitlines()
                            if line.strip()]
                self.assertIn(command, commands)

    def test_exports_all_bounded_verifier_limits(self) -> None:
        names = (
            "MAX_INDEX_BYTES",
            "MAX_EVIDENCE_FILES",
            "MAX_FILE_BYTES",
            "MAX_TOTAL_BYTES",
            "MAX_DIRECTORY_ENTRIES",
            "MAX_PATH_DEPTH",
        )
        for name in names:
            with self.subTest(name=name):
                value = getattr(qef, name)
                self.assertIs(type(value), int)
                self.assertGreater(value, 0)

    def test_accepts_nested_bytes_empty_file_repeated_digest_and_reserved_regular_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            same = b"same bytes\x00\xff"
            _write(root, "nested/z.bin", same)
            _write(root, "nested/a.bin", same)
            _write(root, "empty.bin", b"")

            bundle_path = "artifacts/result.json"
            _write(root, bundle_path, b"not evidence")
            _write(root, f"{bundle_path}.sha256", b"not evidence either")
            _write(root, "evidence-index.json", b"old index bytes")

            digest = _sha256(same)
            references = [
                _reference("nested/z.bin", digest),
                _reference("empty.bin", _sha256(b"")),
                _reference("nested/a.bin", digest),
                _reference("nested/a.bin", digest),
            ]
            index_bytes = _index_for(
                root, references, bundle_path=bundle_path
            )

            result = self._verify(
                root,
                tuple(references),
                index_bytes,
                bundle_path=bundle_path,
            )

            self.assertEqual(result, hashlib.sha256(index_bytes).hexdigest())

    def test_accepts_string_root_and_does_not_mutate_references_or_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"immutable fixture"
            _write(root, "data/evidence.bin", data)
            references = [_reference("data/evidence.bin", _sha256(data))]
            index_bytes = _index_for(root, references)
            references_before = copy.deepcopy(references)
            index_before = bytes(index_bytes)

            result = self._verify(str(root), references, index_bytes)

            self.assertEqual(result, hashlib.sha256(index_before).hexdigest())
            self.assertEqual(references, references_before)
            self.assertEqual(index_bytes, index_before)

    def test_requires_unsigned_utf8_path_sorting_in_the_canonical_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "ascii.bin": b"a",
                "\ue000.bin": b"e",
                "\U00010000.bin": b"u",
                "empty.bin": b"",
            }
            for path, data in contents.items():
                _write(root, path, data)
            references = [
                _reference("\U00010000.bin", _sha256(contents["\U00010000.bin"])),
                _reference("empty.bin", _sha256(b"")),
                _reference("\ue000.bin", _sha256(contents["\ue000.bin"])),
                _reference("ascii.bin", _sha256(contents["ascii.bin"])),
            ]
            index_bytes = _index_for(root, references)
            expected_prefix = b'[{"path":"ascii.bin"'

            self.assertTrue(index_bytes.startswith(expected_prefix))
            self.assertEqual(
                self._verify(root, references, index_bytes),
                hashlib.sha256(index_bytes).hexdigest(),
            )

            entries = json.loads(index_bytes.decode("utf-8"))
            unsorted = _index_from_entries(list(reversed(entries)))
            self._assert_rejected(
                lambda: self._verify(root, references, unsorted)
            )

    def test_requires_exact_canonical_index_bytes_without_whitespace_or_newline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"canonical"
            _write(root, "a.bin", data)
            references = [_reference("a.bin", _sha256(data))]
            index_bytes = _index_for(root, references)

            self._verify(root, references, index_bytes)
            for noncanonical in (
                index_bytes + b"\n",
                b" " + index_bytes,
                json.dumps(
                    json.loads(index_bytes.decode("utf-8")), indent=2
                ).encode("utf-8"),
            ):
                with self.subTest(noncanonical=noncanonical[:20]):
                    self._assert_rejected(
                        lambda noncanonical=noncanonical: self._verify(
                            root, references, noncanonical
                        )
                    )

    def test_rejects_missing_and_extra_regular_evidence_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = b"first"
            second = b"second"
            _write(root, "first.bin", first)
            _write(root, "second.bin", second)
            references = [
                _reference("first.bin", _sha256(first)),
                _reference("second.bin", _sha256(second)),
            ]
            index_bytes = _index_for(root, references)

            (root / "second.bin").unlink()
            self._assert_rejected(
                lambda: self._verify(root, references, index_bytes)
            )

            _write(root, "second.bin", second)
            _write(root, "extra.bin", b"unlisted")
            self._assert_rejected(
                lambda: self._verify(root, references, index_bytes)
            )

    def test_rejects_wrong_file_sha256_and_wrong_index_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"right bytes"
            _write(root, "a.bin", data)
            actual_digest = _sha256(data)
            references = [_reference("a.bin", actual_digest)]
            good_index = _index_for(root, references)

            _write(root, "a.bin", b"wrong bytes")
            self._assert_rejected(
                lambda: self._verify(root, references, good_index)
            )

            _write(root, "a.bin", data)
            wrong_size = build_evidence_index(
                references,
                [_observation("a.bin", "999", actual_digest)],
                bundle_path=BUNDLE_PATH,
            )
            self._assert_rejected(
                lambda: self._verify(root, references, wrong_size)
            )

            wrong_digest = "0" * 64
            wrong_sha = build_evidence_index(
                [_reference("a.bin", wrong_digest)],
                [_observation("a.bin", str(len(data)), wrong_digest)],
                bundle_path=BUNDLE_PATH,
            )
            self._assert_rejected(
                lambda: self._verify(
                    root, [_reference("a.bin", wrong_digest)], wrong_sha
                )
            )

    def test_rejects_conflicting_reference_declarations_but_accepts_same_digest_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"one"
            _write(root, "same.bin", data)
            digest = _sha256(data)
            same_digest = [
                _reference("same.bin", digest),
                _reference("same.bin", digest),
            ]
            index_bytes = _index_for(root, same_digest)
            self._verify(root, tuple(same_digest), index_bytes)

            conflicting = [
                _reference("same.bin", digest),
                _reference("same.bin", "0" * 64),
            ]
            self._assert_rejected(
                lambda: self._verify(root, conflicting, index_bytes)
            )

            with_status = [
                {"path": "same.bin", "sha256": digest, "status": "qualified"}
            ]
            self._assert_rejected(
                lambda: self._verify(root, with_status, index_bytes)
            )

    def test_rejects_duplicate_index_entries_and_duplicate_json_object_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"entry"
            _write(root, "a.bin", data)
            digest = _sha256(data)
            references = [_reference("a.bin", digest)]
            index_bytes = _index_for(root, references)
            entry = {
                "path": "a.bin",
                "size_bytes": str(len(data)),
                "sha256": digest,
            }

            self._assert_rejected(
                lambda: self._verify(
                    root, references, _index_from_entries([entry, entry])
                )
            )
            duplicate_key = (
                '[{"path":"a.bin","path":"a.bin","size_bytes":"5",'
                f'"sha256":"{digest}"}}]'
            ).encode("ascii")
            self.assertNotEqual(duplicate_key, index_bytes)
            self._assert_rejected(
                lambda: self._verify(root, references, duplicate_key)
            )

    def test_rejects_invalid_index_shapes_values_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"entry"
            _write(root, "a.bin", data)
            digest = _sha256(data)
            references = [_reference("a.bin", digest)]
            valid_entry = {
                "path": "a.bin",
                "size_bytes": str(len(data)),
                "sha256": digest,
            }
            invalid_indexes = [
                b"null",
                b"{}",
                b'{"observations":[]}',
                b"[]",
                _index_from_entries(
                    [{"path": "missing.bin", "size_bytes": "5", "sha256": digest}]
                ),
                _index_from_entries(
                    [{"path": "a.bin", "size_bytes": 5, "sha256": digest}]
                ),
                _index_from_entries(
                    [{"path": "a.bin", "size_bytes": "05", "sha256": digest}]
                ),
                _index_from_entries(
                    [{"path": "a.bin", "size_bytes": "5", "sha256": "A" * 64}]
                ),
                _index_from_entries(
                    [{**valid_entry, "qualification": "passed"}]
                ),
            ]
            for invalid in invalid_indexes:
                with self.subTest(invalid=invalid):
                    self._assert_rejected(
                        lambda invalid=invalid: self._verify(
                            root, references, invalid
                        )
                    )

            self._assert_rejected(
                lambda: self._verify(root, references, b'[{"path":"a.bin"')
            )

    def test_rejects_invalid_utf8_and_control_components_in_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"entry"
            _write(root, "a.bin", data)
            digest = _sha256(data)
            references = [_reference("a.bin", digest)]
            invalid_indexes = (
                b"\xff",
                b'[{"path":"a\x00b","size_bytes":"5","sha256":"'
                + digest.encode("ascii")
                + b'"}]',
                b'[{"path":"a\\u0000b","size_bytes":"5","sha256":"'
                + digest.encode("ascii")
                + b'"}]',
                b'[{"path":"a\\ud800b","size_bytes":"5","sha256":"'
                + digest.encode("ascii")
                + b'"}]',
            )
            for invalid in invalid_indexes:
                with self.subTest(invalid=invalid):
                    self._assert_rejected(
                        lambda invalid=invalid: self._verify(
                            root, references, invalid
                        )
                    )

    def test_rejects_invalid_reference_and_bundle_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"entry"
            _write(root, "a.bin", data)
            digest = _sha256(data)
            valid_references = [_reference("a.bin", digest)]
            valid_index = _index_for(root, valid_references)

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
                with self.subTest(reference_path=repr(path)):
                    self._assert_rejected(
                        lambda path=path: self._verify(
                            root,
                            [_reference(path, digest)],
                            b"[]",
                        )
                    )

            for bundle_path in invalid_paths + ("evidence-index.json",):
                with self.subTest(bundle_path=repr(bundle_path)):
                    self._assert_rejected(
                        lambda bundle_path=bundle_path: self._verify(
                            root,
                            valid_references,
                            valid_index,
                            bundle_path=bundle_path,
                        )
                    )

            for invalid_root in (None, 7, b"bytes-root"):
                with self.subTest(root=repr(invalid_root)):
                    self._assert_rejected(
                        lambda invalid_root=invalid_root: self._verify(
                            invalid_root, valid_references, valid_index
                        )
                    )

            for invalid_references in (
                None,
                {"path": "a.bin", "sha256": digest},
                [{"path": "a.bin"}],
                [{"path": "a.bin", "sha256": digest, "extra": True}],
                [{"path": "a.bin", "sha256": 7}],
                [{"path": "a.bin", "sha256": "0" * 63}],
            ):
                with self.subTest(references=repr(invalid_references)):
                    self._assert_rejected(
                        lambda invalid_references=invalid_references: self._verify(
                            root, invalid_references, valid_index
                        )
                    )

            self._assert_rejected(
                lambda: self._verify(
                    root, valid_references, valid_index, bundle_path=None
                )
            )

    def test_rejects_root_symlink_with_trailing_separator_or_dot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            actual = base / "actual"
            actual.mkdir()
            data = b"root identity"
            _write(actual, "a.bin", data)
            refs = [_reference("a.bin", _sha256(data))]
            index = _index_for(actual, refs)
            link = base / "link"
            link.symlink_to(actual, target_is_directory=True)
            for spelling in (str(link) + "/", str(link) + "/."):
                with self.subTest(spelling=spelling):
                    self._assert_rejected(
                        lambda spelling=spelling: self._verify(spelling, refs, index)
                    )

    def test_rejects_symlink_root_parent_file_and_reserved_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            actual_root = base / "actual-root"
            actual_root.mkdir()
            data = b"safe"
            _write(actual_root, "a.bin", data)
            refs = [_reference("a.bin", _sha256(data))]
            index = _index_for(actual_root, refs)
            root_link = base / "root-link"
            os.symlink(actual_root, root_link, target_is_directory=True)
            self._assert_rejected(
                lambda: self._verify(root_link, refs, index)
            )

            root = base / "root"
            root.mkdir()
            outside = base / "outside"
            outside.mkdir()
            outside_data = b"outside"
            _write(outside, "a.bin", outside_data)
            parent_link = root / "parent"
            os.symlink(outside, parent_link, target_is_directory=True)
            parent_refs = [
                _reference("parent/a.bin", _sha256(outside_data))
            ]
            parent_index = build_evidence_index(
                parent_refs,
                [_observation("parent/a.bin", str(len(outside_data)), _sha256(outside_data))],
                bundle_path=BUNDLE_PATH,
            )
            self._assert_rejected(
                lambda: self._verify(root, parent_refs, parent_index)
            )

            file_root = base / "file-root"
            file_root.mkdir()
            outside_file = base / "outside-file.bin"
            outside_file.write_bytes(outside_data)
            os.symlink(outside_file, file_root / "a.bin")
            file_refs = [_reference("a.bin", _sha256(outside_data))]
            file_index = build_evidence_index(
                file_refs,
                [_observation("a.bin", str(len(outside_data)), _sha256(outside_data))],
                bundle_path=BUNDLE_PATH,
            )
            self._assert_rejected(
                lambda: self._verify(file_root, file_refs, file_index)
            )

            for reserved in (
                "bundle.json",
                "bundle.json.sha256",
                "evidence-index.json",
            ):
                reserved_root = base / f"reserved-{reserved.replace('/', '_')}"
                reserved_root.mkdir()
                _write(reserved_root, "a.bin", data)
                reserved_target = base / f"target-{reserved.replace('/', '_')}"
                reserved_target.write_bytes(b"reserved target")
                os.symlink(reserved_target, reserved_root / reserved)
                reserved_refs = [_reference("a.bin", _sha256(data))]
                reserved_index = _index_for(reserved_root, reserved_refs)
                with self.subTest(reserved=reserved):
                    self._assert_rejected(
                        lambda reserved_root=reserved_root, reserved_refs=reserved_refs, reserved_index=reserved_index: self._verify(
                            reserved_root, reserved_refs, reserved_index
                        )
                    )

    def _assert_structural_entry_rejected_without_opening(
        self, root: Path, entry: str, *, fifo: bool
    ) -> None:
        digest = _sha256(b"declared but not read")
        references = [_reference(entry, digest)]
        index = build_evidence_index(
            references,
            [_observation(entry, "17", digest)],
            bundle_path=BUNDLE_PATH,
        )
        target = root / entry
        target_attempts: list[object] = []
        real_os_open = qef.os.open
        real_builtin_open = open

        def is_target(path: object) -> bool:
            try:
                raw = os.fsdecode(os.fspath(path))
            except TypeError:
                return False
            candidate = Path(raw)
            if candidate.is_absolute():
                return candidate == target
            try:
                return candidate == target.relative_to(root)
            except ValueError:
                return candidate == Path(target.name)

        def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
            if is_target(path):
                target_attempts.append(("os.open", path))
                raise OSError("content open blocked by structural test")
            return real_os_open(path, flags, mode, dir_fd=dir_fd)

        def guarded_builtin_open(file, *args, **kwargs):
            if is_target(file):
                target_attempts.append(("open", file))
                raise OSError("content open blocked by structural test")
            return real_builtin_open(file, *args, **kwargs)

        with mock.patch.object(qef.os, "open", side_effect=guarded_os_open), mock.patch(
            "builtins.open", side_effect=guarded_builtin_open
        ), mock.patch.object(qef.os, "read", wraps=qef.os.read) as read_mock:
            self._assert_rejected(
                lambda: self._verify(root, references, index)
            )

        self.assertEqual(
            target_attempts,
            [],
            "FIFO/directory must be rejected from metadata before opening content",
        )
        self.assertEqual(read_mock.call_count, 0)

    def test_rejects_fifo_before_any_content_open_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root / "evidence")
            self._assert_structural_entry_rejected_without_opening(
                root, "evidence", fifo=True
            )

    def test_rejects_directory_in_file_place_before_any_content_open_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            self._assert_structural_entry_rejected_without_opening(
                root, "evidence", fifo=False
            )

    def test_rejects_input_index_larger_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"bounded"
            _write(root, "a.bin", data)
            refs = [_reference("a.bin", _sha256(data))]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_INDEX_BYTES", len(index) - 1):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_rejects_more_evidence_files_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {"a.bin": b"a", "b.bin": b"b"}
            for path, data in contents.items():
                _write(root, path, data)
            refs = [_reference(path, _sha256(data)) for path, data in contents.items()]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_EVIDENCE_FILES", 1):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_rejects_one_file_larger_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"1234"
            _write(root, "a.bin", data)
            refs = [_reference("a.bin", _sha256(data))]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_FILE_BYTES", 3):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_rejects_total_evidence_bytes_larger_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {"a.bin": b"12", "b.bin": b"345"}
            for path, data in contents.items():
                _write(root, path, data)
            refs = [_reference(path, _sha256(data)) for path, data in contents.items()]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_TOTAL_BYTES", 4):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_rejects_directory_entry_fanout_larger_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {"a.bin": b"a", "b.bin": b"b"}
            for path, data in contents.items():
                _write(root, path, data)
            refs = [_reference(path, _sha256(data)) for path, data in contents.items()]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_DIRECTORY_ENTRIES", 1):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_rejects_path_depth_larger_than_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "one/two/evidence.bin"
            data = b"deep"
            _write(root, relative, data)
            refs = [_reference(relative, _sha256(data))]
            index = _index_for(root, refs)
            with mock.patch.object(qef, "MAX_PATH_DEPTH", 2):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )

    def test_detects_same_length_content_mutation_during_os_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b"A" * 8193
            replacement = b"B" * len(original)
            path = _write(root, "a.bin", original)
            refs = [_reference("a.bin", _sha256(original))]
            index = _index_for(root, refs)
            real_read = qef.os.read
            mutated = False

            def racing_read(fd: int, count: int) -> bytes:
                nonlocal mutated
                result = real_read(fd, count)
                if result and not mutated:
                    path.write_bytes(replacement)
                    mutated = True
                return result

            with mock.patch.object(qef.os, "read", side_effect=racing_read):
                with self.assertRaisesRegex(
                    ValueError, r"^evidence identity changed: a\.bin$"
                ):
                    self._verify(root, refs, index)
            self.assertTrue(mutated)

    def test_detects_in_place_identical_rewrite_during_final_rescan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b"stable evidence"
            path = _write(root, "a.bin", original)
            refs = [_reference("a.bin", _sha256(original))]
            index = _index_for(root, refs)
            initial = os.stat(path)
            real_scan = qef._scan
            scan_modes: list[bool] = []

            def scanning(
                root_fd: int,
                expected: dict[str, dict[str, str]],
                directories: set[str],
                reserved: set[str],
                *,
                authenticate: bool,
            ) -> dict[str, tuple[int, ...]]:
                scan_modes.append(authenticate)
                snapshots = real_scan(
                    root_fd,
                    expected,
                    directories,
                    reserved,
                    authenticate=authenticate,
                )
                if len(scan_modes) == 1:
                    with path.open("r+b") as stream:
                        stream.seek(0)
                        stream.write(original)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.utime(
                        path,
                        ns=(
                            initial.st_atime_ns,
                            initial.st_mtime_ns + 1_000_000_000,
                        ),
                    )
                return snapshots

            with mock.patch.object(qef, "_scan", side_effect=scanning) as scan_mock:
                with self.assertRaisesRegex(
                    ValueError, r"^evidence tree changed during verification$"
                ):
                    self._verify(root, refs, index)

            self.assertEqual(scan_mock.call_count, 2)
            self.assertEqual(scan_modes, [True, False])
            after = os.stat(path)
            self.assertEqual(after.st_dev, initial.st_dev)
            self.assertEqual(after.st_ino, initial.st_ino)
            self.assertNotEqual(after.st_mtime_ns, initial.st_mtime_ns)
            self.assertEqual(path.read_bytes(), original)

    def test_releases_nested_descriptors_when_nested_scan_or_read_raises(
        self,
    ) -> None:
        for fault in ("read", "scandir"):
            with self.subTest(fault=fault):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    data = b"nested evidence"
                    _write(root, "nested/a.bin", data)
                    refs = [_reference("nested/a.bin", _sha256(data))]
                    index = _index_for(root, refs)
                    opened: list[int] = []
                    closed: list[int] = []
                    read_fds: list[int] = []
                    scandir_calls = 0
                    real_open = qef.os.open
                    real_close = qef.os.close
                    real_scandir = qef.os.scandir

                    def tracking_open(
                        name: object,
                        flags: int,
                        mode: int = 0o777,
                        *,
                        dir_fd: int | None = None,
                    ) -> int:
                        fd = real_open(name, flags, mode, dir_fd=dir_fd)
                        opened.append(fd)
                        return fd

                    def tracking_close(fd: int) -> None:
                        closed.append(fd)
                        real_close(fd)

                    def failing_read(fd: int, count: int) -> bytes:
                        read_fds.append(fd)
                        raise OSError("injected read failure")

                    def failing_scandir(name: object):
                        nonlocal scandir_calls
                        scandir_calls += 1
                        if fault == "scandir" and scandir_calls == 2:
                            raise OSError("injected scandir failure")
                        return real_scandir(name)

                    if fault == "read":
                        fault_patch = mock.patch.object(
                            qef.os, "read", side_effect=failing_read
                        )
                    else:
                        fault_patch = mock.patch.object(
                            qef.os, "scandir", side_effect=failing_scandir
                        )

                    with mock.patch.object(
                        qef.os, "open", side_effect=tracking_open
                    ), mock.patch.object(
                        qef.os, "close", side_effect=tracking_close
                    ), fault_patch:
                        with self.assertRaises(OSError) as raised:
                            self._verify(root, refs, index)

                    self.assertEqual(str(raised.exception), f"injected {fault} failure")
                    self.assertGreaterEqual(len(opened), 2)
                    self.assertEqual(len(closed), len(opened))
                    self.assertEqual(set(closed), set(opened))
                    if fault == "read":
                        self.assertEqual(len(read_fds), 1)
                        self.assertIn(read_fds[0], opened)
                    else:
                        self.assertEqual(scandir_calls, 2)

    def test_detects_replacement_of_earlier_evidence_while_another_is_hashed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            root.mkdir()
            first = b"first evidence" * 512
            second = b"second evidence" * 512
            first_path = _write(root, "a.bin", first)
            second_path = _write(root, "b.bin", second)
            refs = [
                _reference("a.bin", _sha256(first)),
                _reference("b.bin", _sha256(second)),
            ]
            index = _index_for(root, refs)
            first_stat = os.stat(first_path)
            second_stat = os.stat(second_path)
            replacement_path = root.parent / "replacement-a.bin"
            real_read = qef.os.read
            real_fstat = qef.os.fstat
            replaced = False

            def racing_read(fd: int, count: int) -> bytes:
                nonlocal replaced
                result = real_read(fd, count)
                descriptor_stat = real_fstat(fd)
                if (
                    result
                    and not replaced
                    and descriptor_stat.st_dev == second_stat.st_dev
                    and descriptor_stat.st_ino == second_stat.st_ino
                ):
                    replacement_path.write_bytes(first)
                    os.replace(replacement_path, first_path)
                    replaced = True
                return result

            with mock.patch.object(qef.os, "read", side_effect=racing_read):
                self._assert_rejected(
                    lambda: self._verify(root, refs, index)
                )
            self.assertTrue(replaced)
            self.assertNotEqual(os.stat(first_path).st_ino, first_stat.st_ino)

    def test_rejects_non_bytes_index_and_invalid_reserved_bundle_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"entry"
            _write(root, "a.bin", data)
            refs = [_reference("a.bin", _sha256(data))]
            index = _index_for(root, refs)
            for bad_index in (bytearray(index), memoryview(index), index.decode("utf-8")):
                with self.subTest(index_type=type(bad_index).__name__):
                    self._assert_rejected(
                        lambda bad_index=bad_index: self._verify(
                            root, refs, bad_index
                        )
                    )
            for bad_bundle in (b"bundle.json", 7):
                with self.subTest(bundle_type=type(bad_bundle).__name__):
                    self._assert_rejected(
                        lambda bad_bundle=bad_bundle: self._verify(
                            root, refs, index, bundle_path=bad_bundle
                        )
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
