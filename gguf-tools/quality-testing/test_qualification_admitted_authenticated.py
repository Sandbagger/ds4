#!/usr/bin/env python3
"""Host-only RED E2E tests for admitted record-schema binding.

The executable is the existing generated control fake and the version probe is
an explicit host-only fixture.  The simulated proc tree is test data, not
Linux-native executable-origin evidence.  These tests never contact DS4, a
model, a GPU, a service, or a network.
"""

from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import io
import inspect
import json
import os
import tempfile
import unittest
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import compact_runtime_qualify as COMPACT
import qualification_admission as ADMISSION
import qualification_authenticated as AUTH
import qualification_process as PROCESS
import qualification_records as STREAMED_RECORDS
import qualification_resident_records as RESIDENT_RECORDS
import test_qualification_admission as ADMISSION_TEST
import test_qualification_authenticated as AUTH_TEST
import test_qualification_resident_authenticated as RESIDENT_AUTH_TEST
import test_qualification_resident_records as RESIDENT_TEST
from qualification_records import QualificationRecordStream
from qualification_resident_records import QualificationResidentRecordStream
from test_qualification_records import _expected as streamed_expected
from test_qualification_records import _json_line as streamed_json_line
from test_qualification_records import _lifecycle_records as streamed_lifecycle


POSIX_PROCESS_GROUPS = AUTH_TEST.POSIX_PROCESS_GROUPS
SCHEMA_NAMES = ADMISSION_TEST.SCHEMA_NAMES
RECORD_SCHEMA_DOCUMENT_NAMES = (
    "ds4-bench-qualification-v1.schema.json",
    "ds4-bench-resident-qualification-v1.schema.json",
    "ds4-runtime-v1.schema.json",
    "ds4-runtime-request-v1.schema.json",
)


@dataclass
class _AdmissionFixture:
    admission: Any
    schema_root: Path
    schema_bytes: dict[str, bytes]
    manifest_path: Path
    model: Path
    fake: Path
    allow_closed_exit: bool = False


@contextlib.contextmanager
def _schema_path_reads_fail(root: Path, reads: list[Path]) -> Iterator[None]:
    """Reject parser schema reads after admission, but allow owner identity opens."""

    admitted_paths = {(root / name).resolve() for name in SCHEMA_NAMES}
    canonical_root = (ADMISSION_TEST.ROOT / "schemas").resolve()
    canonical_paths = {(canonical_root / name).resolve() for name in SCHEMA_NAMES}
    poison_root = root / "__schema-selector-poison__"
    poison_paths = {
        "streamed": poison_root / "streamed.schema.json",
        "resident": poison_root / "resident.schema.json",
        "runtime": poison_root / "runtime.schema.json",
        "request": poison_root / "request.schema.json",
    }
    paths = admitted_paths | canonical_paths | {
        path.resolve() for path in poison_paths.values()
    }

    def is_schema_path(value: Any) -> bool:
        try:
            return Path(os.fspath(value)).resolve() in paths
        except (TypeError, ValueError, OSError):
            return False

    def guard_open(original: Any) -> Callable[..., Any]:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            mode = args[0] if args else kwargs.get("mode", "r")
            writing = any(flag in str(mode) for flag in ("w", "a", "x"))
            if is_schema_path(file) and not writing:
                reads.append(Path(file))
                raise AssertionError(f"schema content was reopened: {file}")
            return original(file, *args, **kwargs)
        return guarded

    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_path_open = Path.open
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_os_open(file: Any, *args: Any, **kwargs: Any) -> int:
        flags = args[0] if args else kwargs.get("flags", os.O_RDONLY)
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR))
        if is_schema_path(file) and not writing:
            frame = inspect.currentframe()
            artifact_verify = False
            while frame is not None:
                filename = Path(frame.f_code.co_filename).name
                if filename == "qualification_artifacts.py" and frame.f_code.co_name == "verify":
                    artifact_verify = True
                    break
                frame = frame.f_back
            if not artifact_verify:
                reads.append(Path(file))
                raise AssertionError(f"schema content was reopened: {file}")
            # QualificationArtifact.verify may open the admitted path only to
            # check retained identity/change-time; it must not read schema bytes.
        return original_os_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        writing = any(flag in str(mode) for flag in ("w", "a", "x"))
        if is_schema_path(self) and not writing:
            reads.append(self)
            raise AssertionError(f"schema content was reopened: {self}")
        return original_path_open(self, *args, **kwargs)

    def guarded_read_bytes(self: Path) -> bytes:
        if is_schema_path(self):
            reads.append(self)
            raise AssertionError(f"schema content was reopened: {self}")
        return original_read_bytes(self)

    def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if is_schema_path(self):
            reads.append(self)
            raise AssertionError(f"schema content was reopened: {self}")
        return original_read_text(self, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        # Poison every legacy selector after the real admission is complete.
        # A parser that ignores input_admission therefore fails visibly rather
        # than silently falling back to a canonical schema path.
        stack.enter_context(mock.patch.object(
            STREAMED_RECORDS, "SCHEMA_PATH", poison_paths["streamed"]
        ))
        stack.enter_context(mock.patch.object(
            STREAMED_RECORDS, "RUNTIME_SCHEMA_PATH", poison_paths["runtime"]
        ))
        stack.enter_context(mock.patch.object(
            STREAMED_RECORDS, "REQUEST_SCHEMA_PATH", poison_paths["request"]
        ))
        stack.enter_context(mock.patch.object(
            RESIDENT_RECORDS, "SCHEMA_PATH", poison_paths["resident"]
        ))
        stack.enter_context(mock.patch.object(
            builtins, "open", side_effect=guard_open(original_open)
        ))
        stack.enter_context(mock.patch.object(
            io, "open", side_effect=guard_open(original_io_open)
        ))
        stack.enter_context(mock.patch.object(
            os, "open", side_effect=guarded_os_open
        ))
        stack.enter_context(mock.patch.object(Path, "open", new=guarded_path_open))
        stack.enter_context(mock.patch.object(Path, "read_bytes", new=guarded_read_bytes))
        stack.enter_context(mock.patch.object(Path, "read_text", new=guarded_read_text))
        yield


@contextlib.contextmanager
def _real_admission_fixture(fake: Path, model: Path) -> Iterator[_AdmissionFixture]:
    """Admit the already-generated fake/model with the checked-in helpers."""

    with tempfile.TemporaryDirectory(prefix="qualification-schema-bound-admission-") as name:
        directory = Path(name)
        manifest_path = directory / "manifest.json"
        schema_root = directory / "schemas"
        observed = model.stat()
        payload = model.read_bytes()
        model_digest = hashlib.sha256(payload).hexdigest()
        identity = copy.deepcopy(ADMISSION_TEST.FIXTURE.MODEL_IDENTITY)
        identity.update(
            {
                "path": str(model),
                "size_bytes": str(observed.st_size),
                "sha256": model_digest,
                "device": str(observed.st_dev),
                "inode": str(observed.st_ino),
                "mtime_ns": str(observed.st_mtime_ns),
            }
        )
        old_identity = copy.deepcopy(ADMISSION_TEST.FIXTURE.MODEL_IDENTITY)
        ADMISSION_TEST.FIXTURE.MODEL_IDENTITY.clear()
        ADMISSION_TEST.FIXTURE.MODEL_IDENTITY.update(identity)
        patches = []
        for module in ADMISSION_TEST._compact_modules():
            patches.extend(
                (
                    mock.patch.object(module, "MODEL_SIZE", observed.st_size),
                    mock.patch.object(module, "MODEL_SHA256", model_digest),
                )
            )
        fixture: _AdmissionFixture | None = None
        try:
            # Keep the tiny-model constants patched through manifest creation,
            # admission validation, and the entire yielded live context.  This
            # mirrors _tiny_inputs and prevents restored production constants
            # from invalidating the test manifest before it is yielded.
            with ADMISSION_TEST._patches(patches):
                manifest = ADMISSION_TEST.FIXTURE.build_fixture()
                manifest_path.write_bytes(
                    ADMISSION_TEST.COMPACT.canonical_json_bytes(manifest) + b"\n"
                )
                schema_bytes = ADMISSION_TEST._write_schema_bundle(
                    schema_root,
                    model_size=observed.st_size,
                    model_sha256=model_digest,
                )
                revision = str(
                    manifest["prompt_source"]["tokenizer_runtime"]["source_revision"]
                )
                version_bytes = {
                    role: ADMISSION_TEST._version_bytes(revision)
                    for role in ADMISSION_TEST.ROLES
                }

                def host_only_version_probe(role: str, artifact: Any) -> bytes:
                    # This is a host-only claim fixture.  It is not native
                    # process-origin evidence; the authenticated runner still
                    # must prove the child executable separately.
                    del artifact
                    return version_bytes[role]

                # All three role paths intentionally name the generated fake.
                # The runner must retain the exact admitted ``bench`` owner,
                # rather than selecting an equal-looking path or role later.
                binaries = {role: fake for role in ADMISSION_TEST.ROLES}
                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", schema_root):
                    admission_context = ADMISSION.admit_qualification_inputs(
                        manifest_path,
                        model_path=model,
                        binaries=binaries,
                        version_probe=host_only_version_probe,
                    )
                    try:
                        with admission_context as admission:
                            fixture = _AdmissionFixture(
                                admission=admission,
                                schema_root=schema_root,
                                schema_bytes=schema_bytes,
                                manifest_path=manifest_path,
                                model=model,
                                fake=fake,
                            )
                            yield fixture
                    except ValueError:
                        # A deliberate schema-drift test causes the retained
                        # owner to close itself.  It has already asserted the
                        # runner's protocol_error result and only suppresses
                        # this expected second observation at context exit.
                        if fixture is None or not fixture.allow_closed_exit:
                            raise
        finally:
            ADMISSION_TEST.FIXTURE.MODEL_IDENTITY.clear()
            ADMISSION_TEST.FIXTURE.MODEL_IDENTITY.update(old_identity)

def _admitted_factory(state: dict[str, Any], *, trap_reads: bool = True) -> Callable[[Path, Path], Any]:
    """Return the test-only factory accepted by the shared-session helper."""

    @contextlib.contextmanager
    def factory(fake: Path, model: Path) -> Iterator[Any]:
        state["factory_calls"] = state.get("factory_calls", 0) + 1
        with _real_admission_fixture(fake, model) as fixture:
            state["fixture"] = fixture
            if trap_reads:
                reads: list[Path] = []
                state["schema_reads"] = reads
                with _schema_path_reads_fail(fixture.schema_root, reads):
                    yield fixture.admission
            else:
                yield fixture.admission

    return factory


@contextlib.contextmanager
def _single_admitted_session(
    records: list[dict[str, Any]],
    expected: dict[str, Any],
    record_kind: str,
) -> Iterator[dict[str, Any]]:
    """Run one generated child with a real admission and optional after hook."""

    payload = f"one admitted {record_kind} model\0".encode("utf-8") * 256
    with tempfile.TemporaryDirectory(prefix="qualification-admitted-single-") as tmp:
        directory = Path(tmp)
        model = directory / COMPACT.MODEL_FILENAME
        model.write_bytes(payload)
        with AUTH_TEST.CONTROLLED_TEST._controlled_fake_case(
            "complete", records, model
        ) as (command, metadata_path, token):
            fake = Path(command[1])
            AUTH_TEST._adapt_controlled_fake(fake)
            RESIDENT_AUTH_TEST._enable_parent_payload(fake)
            payload_path = directory / "records.json"
            with _real_admission_fixture(fake, model) as fixture:
                _set_manifest_binding(records, fixture.admission)
                expected["manifest_sha256"] = fixture.admission.manifest_sha256
                _write_payload(payload_path, records)
                with tempfile.TemporaryDirectory(
                    prefix="SIMULATED-PROC-admitted-single-"
                ) as proc:
                    executable = fixture.admission.artifacts["bench"]
                    model_artifact = fixture.admission.artifacts["model"]

                    def run_once(
                        *,
                        after_hook: Callable[[int, Path, Path], None] | None = None,
                    ) -> dict[str, Any]:
                        probe = AUTH_TEST._AuthenticationProbe(
                            executable,
                            model_artifact,
                            Path(proc),
                            running_target=fake,
                        )
                        arguments = [
                            "--profile",
                            f"single-{record_kind}",
                            "--model",
                            "ignored.gguf",
                            "--record-payload",
                            str(payload_path),
                        ]
                        return RESIDENT_AUTH_TEST._invoke_authenticated(
                            records,
                            expected,
                            record_kind,
                            executable=executable,
                            model_artifact=model_artifact,
                            probe=probe,
                            metadata_path=metadata_path,
                            token=token,
                            model=model,
                            fake=fake,
                            payload=payload,
                            arguments=arguments,
                            after_hook=after_hook,
                            input_admission=fixture.admission,
                        )

                    yield {"run": run_once, "fixture": fixture}


def _set_manifest_binding(records: list[dict[str, Any]], admission: Any) -> None:
    for record in records:
        record["manifest_sha256"] = admission.manifest_sha256


def _write_payload(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(records, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def _mutate_schema_bytes(fixture: _AdmissionFixture, filename: str) -> None:
    original = fixture.schema_bytes[filename]
    marker = b'"$id":"'
    if marker not in original:
        raise AssertionError(f"schema fixture has no identity marker: {filename}")
    fixture.schema_root.joinpath(filename).write_bytes(
        original.replace(marker, b'"$id":"schema-drift","_unused":"', 1)
    )


def _preflight_call(
    expected: dict[str, Any],
    *,
    admission: Any,
    executable: Any,
    model: Any,
) -> None:
    """Call the real authenticated boundary while forbidding child creation."""

    def never_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("callback ran during admitted preflight")

    with mock.patch.object(PROCESS.subprocess, "Popen") as popen:
        with unittest.TestCase().assertRaises((TypeError, ValueError)):
            AUTH.run_authenticated_qualification_child(
                ["--profile", "admitted-preflight"],
                expected,
                record_kind="streamed",
                executable_artifact=executable,
                model_artifact=model,
                prepare_descriptor=never_called,
                capture_before=never_called,
                capture_after=never_called,
                input_admission=admission,
            )
        popen.assert_not_called()


def _assert_admitted_success(
    test: unittest.TestCase,
    run: dict[str, Any],
    records: list[dict[str, Any]],
    expected: dict[str, Any],
    record_kind: str,
    admission: Any,
) -> None:
    """Assert a successful run without old two-owner-only assumptions."""

    result = run["result"]
    AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(test, result)
    test.assertEqual(result.transport.reason, "complete")
    test.assertEqual(result.transport.returncode, 0)
    test.assertTrue(result.transport.cleanup_complete)
    test.assertIsNone(result.control_error)
    test.assertEqual(len(result.transport.records), 12)
    test.assertEqual(result.transport.stdout.count(b"\n"), 12)
    test.assertEqual(len(result.checkpoints), 12)
    test.assertTrue(run["received_fds"])
    test.assertEqual(len(run["received_fds"]), 1)
    parser = (
        QualificationResidentRecordStream
        if record_kind == "resident" else QualificationRecordStream
    )(expected, input_admission=admission)
    parser.feed(result.transport.stdout)
    parsed = parser.finish()
    test.assertEqual(parsed, result.transport.records)
    test.assertEqual(len(parsed), len(records))

    probe = run["probe"]
    test.assertEqual(
        [name for name, _pid in probe.callback_calls],
        ["prepare"]
        + [name for _sequence in range(1, 13) for name in ("before", "after")],
    )
    callback_pids = {pid for _name, pid in probe.callback_calls}
    test.assertEqual(len(callback_pids), 1)
    test.assertTrue(
        all(owner is run["executable"] for _pid, owner in probe.auth_calls)
    )
    test.assertTrue(all(pid == next(iter(callback_pids)) for pid, _ in probe.auth_calls))
    # Admission verification also records server/eval/schema owners as
    # ``other``.  This existing assertion remains non-vacuous for live PID
    # authentication while deliberately accepting those additional owners.
    AUTH_TEST._assert_authentication_boundaries(test, probe)

    test.assertEqual(result.preparation, run["preparation_refs"][0])
    for sequence, checkpoint in enumerate(result.checkpoints, 1):
        test.assertTrue(checkpoint["complete"])
        test.assertEqual(checkpoint["sequence"], sequence)
        test.assertEqual(checkpoint["before"], run["before_refs"][sequence - 1])
        test.assertEqual(checkpoint["after"], run["after_refs"][sequence - 1])
    for fact in [
        run["preparation_refs"][0],
        *run["before_refs"],
        *run["after_refs"],
    ]:
        fact["caller_mutated"] = True
    test.assertNotIn("caller_mutated", result.preparation)
    test.assertNotIn("caller_mutated", result.checkpoints[0]["before"])
    test.assertNotIn("caller_mutated", result.checkpoints[0]["after"])

    evidence = result.model_evidence
    test.assertIsNotNone(evidence)
    test.assertEqual(evidence.sha256, hashlib.sha256(run["payload"]).hexdigest())
    test.assertEqual(evidence.identity, run["model_artifact"].identity)
    test.assertEqual(
        tuple(json.loads(line) for line in result.transport.stdout.splitlines()),
        result.transport.records,
    )
    for fd in run["received_fds"]:
        with test.assertRaises(OSError):
            os.fstat(fd)
    test.assertEqual(os.pread(run["executable"].fd, 2, 0), b"#!")
    test.assertEqual(
        os.pread(run["model_artifact"].fd, 16, 0), run["payload"][:16]
    )
    run["executable"].verify()
    run["model_artifact"].verify()
    AUTH_TEST._assert_spawn(
        test, run["requests"], run["probe"].executable_fd, run["arguments"]
    )
    AUTH_TEST._assert_cleanup(test, result, run["metadata_path"], run["token"])
    RESIDENT_AUTH_TEST._assert_full_ack_sequence(test, run)


@unittest.skipUnless(
    POSIX_PROCESS_GROUPS,
    "admitted authenticated qualification needs POSIX process groups",
)
class QualificationAdmittedAuthenticatedHostTest(unittest.TestCase):
    def _shared_cases(self) -> tuple[tuple[str, list[dict[str, Any]]], ...]:
        return (
            ("resident", RESIDENT_TEST._fresh_slice()),
            ("streamed", streamed_lifecycle()),
        )

    def _expected(self, kind: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        if kind == "resident":
            return RESIDENT_TEST._expected(records)
        return streamed_expected(records)

    def test_resident_then_streamed_share_one_live_admission_and_schema_snapshots(self) -> None:
        cases = self._shared_cases()
        expected = {kind: self._expected(kind, records) for kind, records in cases}
        state: dict[str, Any] = {}
        factory = _admitted_factory(state)
        with RESIDENT_AUTH_TEST._shared_authenticated_session(
            cases, admission_context_factory=factory
        ) as run_once:
            resident_kind, resident_records = cases[0]
            resident = run_once(resident_kind, resident_records, expected[resident_kind])
            _assert_admitted_success(
                self, resident, resident_records, expected[resident_kind], resident_kind,
                state["fixture"].admission,
            )
            streamed_kind, streamed_records = cases[1]
            streamed = run_once(streamed_kind, streamed_records, expected[streamed_kind])
            _assert_admitted_success(
                self, streamed, streamed_records, expected[streamed_kind], streamed_kind,
                state["fixture"].admission,
            )
            admission = state["fixture"].admission
            self.assertEqual(state["factory_calls"], 1)
            self.assertEqual(
                set(admission.record_schema_documents),
                set(RECORD_SCHEMA_DOCUMENT_NAMES),
            )
            self.assertEqual(len(admission.record_schema_documents), 4)
            self.assertIs(resident["executable"], admission.artifacts["bench"])
            self.assertIs(streamed["executable"], admission.artifacts["bench"])
            self.assertIs(resident["model_artifact"], admission.artifacts["model"])
            self.assertIs(streamed["model_artifact"], admission.artifacts["model"])
            self.assertIs(resident["executable"], streamed["executable"])
            self.assertIs(resident["model_artifact"], streamed["model_artifact"])
            self.assertEqual(state["schema_reads"], [])

    def test_caller_mutated_schema_copies_cannot_weaken_either_parser(self) -> None:
        records_by_kind = {
            "resident": RESIDENT_TEST._fresh_slice(),
            "streamed": streamed_lifecycle(),
        }
        with tempfile.TemporaryDirectory(prefix="qualification-schema-copy-") as tmp:
            model = Path(tmp) / COMPACT.MODEL_FILENAME
            model.write_bytes(b"schema-copy model\0" * 256)
            fake = Path(tmp) / "fake-runner"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            fake.chmod(0o700)
            with _real_admission_fixture(fake, model) as fixture:
                copied = fixture.admission.record_schema_documents
                self.assertEqual(
                    set(copied), set(RECORD_SCHEMA_DOCUMENT_NAMES)
                )
                self.assertEqual(len(copied), 4)
                for name in RECORD_SCHEMA_DOCUMENT_NAMES:
                    copied[name]["additionalProperties"] = True
                fresh = fixture.admission.record_schema_documents
                self.assertEqual(
                    set(fresh), set(RECORD_SCHEMA_DOCUMENT_NAMES)
                )
                self.assertEqual(len(fresh), 4)
                self.assertTrue(
                    all(fresh[name].get("additionalProperties") is False
                        for name in RECORD_SCHEMA_DOCUMENT_NAMES)
                )
                for kind, records in records_by_kind.items():
                    _set_manifest_binding(records, fixture.admission)
                    expected = self._expected(kind, records)
                    candidate = copy.deepcopy(records[0])
                    candidate["caller_schema_mutation"] = True
                    if kind == "resident":
                        stream = QualificationResidentRecordStream(
                            expected, input_admission=fixture.admission
                        )
                        line = RESIDENT_TEST._json_line(candidate)
                    else:
                        stream = QualificationRecordStream(
                            expected, input_admission=fixture.admission
                        )
                        line = streamed_json_line(candidate)
                    with self.subTest(record_kind=kind):
                        with self.assertRaises((TypeError, ValueError)):
                            stream.feed(line)

    def test_manifest_closed_and_exact_retained_owner_mismatches_reject_before_popen(self) -> None:
        records = streamed_lifecycle()
        expected_records = copy.deepcopy(records)
        with tempfile.TemporaryDirectory(prefix="qualification-admitted-preflight-") as tmp:
            directory = Path(tmp)
            model = directory / COMPACT.MODEL_FILENAME
            model.write_bytes(b"preflight model\0" * 256)
            with AUTH_TEST.CONTROLLED_TEST._controlled_fake_case(
                "complete", expected_records, model
            ) as (command, _metadata, _token):
                fake = Path(command[1])
                AUTH_TEST._adapt_controlled_fake(fake)
                with _real_admission_fixture(fake, model) as first:
                    _set_manifest_binding(expected_records, first.admission)
                    wrong_manifest = dict(streamed_expected(expected_records))
                    wrong_manifest["manifest_sha256"] = "f" * 64
                    self.assertEqual(
                        Path(str(first.admission.artifacts["bench"].path)).resolve(),
                        fake.resolve(),
                    )
                    _preflight_call(
                        wrong_manifest,
                        admission=first.admission,
                        executable=first.admission.artifacts["bench"],
                        model=first.admission.artifacts["model"],
                    )
                # Keep the tested admission closed but borrow valid fresh
                # owners, so the rejection cannot be explained by old
                # executable/model artifact checks alone.
                with _real_admission_fixture(fake, model) as fresh:
                    closed_expected = streamed_expected(expected_records)
                    closed_expected["manifest_sha256"] = first.admission.manifest_sha256
                    self.assertEqual(
                        closed_expected["manifest_sha256"],
                        first.admission.manifest_sha256,
                    )
                    _preflight_call(
                        closed_expected,
                        admission=first.admission,
                        executable=fresh.admission.artifacts["bench"],
                        model=fresh.admission.artifacts["model"],
                    )
                    # A wire/caller-supplied object or callable is not an
                    # admission context.  Fresh owners ensure input_admission,
                    # not owner liveness, is the boundary under test.
                    live_expected = streamed_expected(expected_records)
                    live_expected["manifest_sha256"] = fresh.admission.manifest_sha256
                    for bad_admission in (object(), lambda: None):
                        with self.subTest(input_admission=type(bad_admission).__name__):
                            _preflight_call(
                                live_expected,
                                admission=bad_admission,
                                executable=fresh.admission.artifacts["bench"],
                                model=fresh.admission.artifacts["model"],
                            )
                with _real_admission_fixture(fake, model) as first:
                    with _real_admission_fixture(fake, model) as second:
                        expected = streamed_expected(expected_records)
                        expected["manifest_sha256"] = first.admission.manifest_sha256
                        self.assertEqual(
                            expected["manifest_sha256"],
                            first.admission.manifest_sha256,
                        )
                        self.assertEqual(
                            first.admission.manifest_sha256,
                            second.admission.manifest_sha256,
                        )
                        for role in (*ADMISSION_TEST.ROLES, "model"):
                            first_owner = first.admission.artifacts[role]
                            second_owner = second.admission.artifacts[role]
                            self.assertEqual(first_owner.path, second_owner.path)
                            self.assertEqual(first_owner.identity, second_owner.identity)
                            self.assertEqual(first_owner.sha256, second_owner.sha256)
                        self.assertIsNot(
                            first.admission.artifacts["bench"],
                            second.admission.artifacts["bench"],
                        )
                        cases = (
                            (
                                "wrong admission exact owners",
                                first.admission,
                                second.admission.artifacts["bench"],
                                second.admission.artifacts["model"],
                            ),
                            (
                                "same file wrong bench role",
                                first.admission,
                                first.admission.artifacts["server"],
                                first.admission.artifacts["model"],
                            ),
                            (
                                "same file wrong model owner",
                                first.admission,
                                first.admission.artifacts["bench"],
                                second.admission.artifacts["model"],
                            ),
                        )
                        for name, admission, executable, model_owner in cases:
                            with self.subTest(case=name):
                                _preflight_call(
                                    expected,
                                    admission=admission,
                                    executable=executable,
                                    model=model_owner,
                                )

    def test_schema_drift_after_final_callback_refuses_ack_and_keeps_prefix(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)

        def drift_after_final_capture(sequence: int, _model: Path, _fake: Path) -> None:
            if sequence == 12:
                _mutate_schema_bytes(
                    session["fixture"],
                    "ds4-bench-resident-qualification-v1.schema.json",
                )

        with _single_admitted_session(records, expected, "resident") as session:
            run = session["run"](after_hook=drift_after_final_capture)
            result = run["result"]
            AUTH_TEST.CONTROLLED_TEST._assert_controlled_shape(self, result)
            self.assertEqual(result.transport.reason, "protocol_error")
            self.assertTrue(result.control_error)
            self.assertEqual(len(result.transport.records), 11)
            self.assertEqual(len(result.checkpoints), 12)
            self.assertTrue(all(item["complete"] for item in result.checkpoints[:11]))
            self.assertFalse(result.checkpoints[-1]["complete"])
            self.assertEqual(len(result.wire_records), 49)
            self.assertFalse(any(
                wire["direction"] == "send"
                and RESIDENT_AUTH_TEST.WIRE.unpack(wire["payload"])[1]
                == RESIDENT_AUTH_TEST.RESULT_ACK
                and RESIDENT_AUTH_TEST.WIRE.unpack(wire["payload"])[4] == 12
                for wire in result.wire_records
            ))
            AUTH_TEST._assert_authentication_boundaries(self, run["probe"])
            AUTH_TEST._assert_cleanup(
                self, result, run["metadata_path"], run["token"]
            )
            session["fixture"].allow_closed_exit = True
            session["fixture"].schema_root.joinpath(
                "ds4-bench-resident-qualification-v1.schema.json"
            ).write_bytes(
                session["fixture"].schema_bytes[
                    "ds4-bench-resident-qualification-v1.schema.json"
                ]
            )

    def test_exit_tail_schema_drift_is_not_accepted_and_raw_prefix_is_retained(self) -> None:
        records = RESIDENT_TEST._fresh_slice()
        expected = RESIDENT_TEST._expected(records)
        completed: list[Any] = []
        native_controlled = AUTH.run_qualification_controlled_child

        def mutate_after_owned_exit(*args: Any, **kwargs: Any) -> Any:
            observed = native_controlled(*args, **kwargs)
            self.assertEqual(observed.transport.reason, "complete")
            completed.append(observed)
            _mutate_schema_bytes(
                session["fixture"], "ds4-bench-qualification-v1.schema.json"
            )
            return observed

        with mock.patch.object(
            AUTH, "run_qualification_controlled_child", new=mutate_after_owned_exit
        ):
            with _single_admitted_session(records, expected, "resident") as session:
                run = session["run"]()
                result = run["result"]
                self.assertEqual(len(completed), 1)
                self.assertEqual(result.transport.reason, "protocol_error")
                self.assertIn("final", result.control_error)
                self.assertEqual(result.transport.stdout, completed[0].transport.stdout)
                self.assertEqual(result.transport.records, completed[0].transport.records)
                self.assertEqual(result.wire_records, completed[0].wire_records)
                self.assertEqual(len(result.transport.records), 12)
                AUTH_TEST._assert_authentication_boundaries(self, run["probe"])
                AUTH_TEST._assert_cleanup(
                    self, result, run["metadata_path"], run["token"]
                )
                session["fixture"].allow_closed_exit = True
                session["fixture"].schema_root.joinpath(
                    "ds4-bench-qualification-v1.schema.json"
                ).write_bytes(
                    session["fixture"].schema_bytes[
                        "ds4-bench-qualification-v1.schema.json"
                    ]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
