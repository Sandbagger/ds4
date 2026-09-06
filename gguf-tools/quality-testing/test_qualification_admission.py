#!/usr/bin/env python3
"""Contract tests for Task 20 immutable qualification-input admission."""

from __future__ import annotations

# Import the missing boundary before importing any fixture.  This keeps the
# initial run genuinely RED while the production admission module is absent.
import qualification_admission as ADMISSION

import copy
import errno
import hashlib
import importlib.util
import json
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
COMPACT_PATH = ROOT / "gguf-tools/quality-testing/compact_runtime_qualify.py"
FIXTURE_PATH = ROOT / "gguf-tools/quality-testing/test_compact_runtime_qualify.py"
SCHEMA_NAMES = (
    "ds4-version-v1.schema.json",
    "ds4-runtime-v1.schema.json",
    "ds4-runtime-request-v1.schema.json",
    "ds4-token-admission-v1.schema.json",
    "ds4-laguna-compact-runtime-v1.schema.json",
    "compact-runtime-benchmark-v1.schema.json",
    "ds4-bench-qualification-v1.schema.json",
    "ds4-bench-resident-qualification-v1.schema.json",
)
SCHEMA_IDS = (
    "ds4.version/v1",
    "ds4.runtime/v1",
    "ds4.runtime.request/v1",
    "ds4.token-admission/v1",
    "ds4.laguna.compact-runtime/v1",
    "ds4.compact-runtime-benchmark/v1",
    "ds4.bench.qualification/v1",
    "ds4.bench.resident-qualification/v1",
)
CORE_SCHEMA_NAMES = SCHEMA_NAMES[:-2]
RECORD_SCHEMA_NAMES = SCHEMA_NAMES[-2:]
CORE_SCHEMA_IDS = SCHEMA_IDS[:-2]
RECORD_SCHEMA_IDS = SCHEMA_IDS[-2:]
RECORD_SCHEMA_DEPENDENCIES = frozenset(
    {
        "ds4-runtime-v1.schema.json",
        "ds4-runtime-request-v1.schema.json",
    }
)
ROLES = ("server", "bench", "eval")

# This is deliberately a second module object.  The existing manifest tests
# dynamically load their TOOL, whereas admission imports the ordinary module.
COMPACT_SPEC = importlib.util.spec_from_file_location(
    "compact_runtime_qualify_admission_test", COMPACT_PATH
)
if COMPACT_SPEC is None or COMPACT_SPEC.loader is None:
    raise RuntimeError(f"cannot load compact qualifier: {COMPACT_PATH}")
COMPACT = importlib.util.module_from_spec(COMPACT_SPEC)
COMPACT_SPEC.loader.exec_module(COMPACT)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "compact_runtime_qualify_fixture_test", FIXTURE_PATH
)
if FIXTURE_SPEC is None or FIXTURE_SPEC.loader is None:
    raise RuntimeError(f"cannot load manifest fixture helpers: {FIXTURE_PATH}")
FIXTURE = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURE)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_bytes(revision: str, **changes: object) -> bytes:
    value: dict[str, object] = {
        "schema": "ds4.version/v1",
        "revision": revision,
        "dirty": False,
        "backend": "cuda",
        "features": ["laguna", "ssd_streaming"],
    }
    value.update(changes)
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


@dataclass
class Inputs:
    directory: Path
    model: Path
    manifest_path: Path
    schema_root: Path
    binaries: dict[str, Path]
    manifest: dict[str, object]
    manifest_bytes: bytes
    manifest_sha256: str
    schema_bytes: dict[str, bytes]
    version_bytes: dict[str, bytes]


# Every admission implementation import of compact_runtime_qualify should
# resolve to the ordinary module.  Patching every module object visible from
# ADMISSION also protects this test if the implementation uses a private alias.
def _compact_modules() -> tuple[types.ModuleType, ...]:
    modules: list[types.ModuleType] = [COMPACT, FIXTURE.TOOL]
    modules.extend(
        value
        for value in vars(ADMISSION).values()
        if isinstance(value, types.ModuleType)
        and value.__name__ == "compact_runtime_qualify"
    )
    unique: list[types.ModuleType] = []
    for module in modules:
        if all(module is not existing for existing in unique):
            unique.append(module)
    return tuple(unique)


@contextmanager
def _tiny_inputs():
    """Build a real, regular tiny model and a valid eight-schema bundle."""
    with tempfile.TemporaryDirectory(prefix="task20-admission-") as name:
        directory = Path(name)
        model = directory / "laguna-s-2.1-Q4_K_M.gguf"
        model_bytes = (b"Task20 tiny regular model payload\n" * 256)[:8192]
        model.write_bytes(model_bytes)
        observed = model.stat()
        model_digest = hashlib.sha256(model_bytes).hexdigest()
        identity = copy.deepcopy(FIXTURE.MODEL_IDENTITY)
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

        # build_fixture() and _qualification_preflight_fixture() retain the
        # original MODEL_IDENTITY object as a default.  Mutate it in place so
        # both helper paths receive the real tiny-file identity.
        old_identity = copy.deepcopy(FIXTURE.MODEL_IDENTITY)
        FIXTURE.MODEL_IDENTITY.clear()
        FIXTURE.MODEL_IDENTITY.update(identity)
        patches = []
        for module in _compact_modules():
            patches.extend(
                (
                    mock.patch.object(module, "MODEL_SIZE", observed.st_size),
                    mock.patch.object(module, "MODEL_SHA256", model_digest),
                )
            )
        try:
            # Keep both the actual admission module and the dynamically loaded
            # fixture TOOL patched until the caller's admission context exits.
            with _patches(patches):
                manifest = FIXTURE.build_fixture()
                manifest_sha256 = FIXTURE.TOOL.manifest_sha256(manifest)

                manifest_path = directory / "manifest.json"
                manifest_bytes = COMPACT.canonical_json_bytes(manifest) + b"\n"
                manifest_path.write_bytes(manifest_bytes)

                schema_root = directory / "schemas"
                schema_root.mkdir()
                schema_bytes = _write_schema_bundle(
                    schema_root,
                    model_size=observed.st_size,
                    model_sha256=model_digest,
                )

                binaries: dict[str, Path] = {}
                for role in ROLES:
                    binary = directory / f"ds4-{role}"
                    binary.write_bytes(f"Task20 {role} binary\n".encode("ascii"))
                    binary.chmod(0o755)
                    binaries[role] = binary

                revision = str(
                    manifest["prompt_source"]["tokenizer_runtime"]["source_revision"]
                )
                version_bytes = {
                    role: _version_bytes(revision)
                    for role in ROLES
                }
                yield Inputs(
                    directory=directory,
                    model=model,
                    manifest_path=manifest_path,
                    schema_root=schema_root,
                    binaries=binaries,
                    manifest=manifest,
                    manifest_bytes=manifest_bytes,
                    manifest_sha256=manifest_sha256,
                    schema_bytes=schema_bytes,
                    version_bytes=version_bytes,
                )
        finally:
            FIXTURE.MODEL_IDENTITY.clear()
            FIXTURE.MODEL_IDENTITY.update(old_identity)


@contextmanager
def _patches(patches):
    entered = []
    try:
        for patcher in patches:
            entered.append(patcher)
            patcher.start()
        yield
    finally:
        for patcher in reversed(entered):
            patcher.stop()


def _write_schema_bundle(
    root: Path,
    *,
    model_size: int,
    model_sha256: str,
    mutate: dict[str, object] | None = None,
) -> dict[str, bytes]:
    """Clone the eight real schemas, changing only tiny-model consts."""
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, bytes] = {}
    for name in SCHEMA_NAMES:
        value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        if name == "compact-runtime-benchmark-v1.schema.json":
            model = value["$defs"]["model"]["properties"]
            model["size_bytes"]["const"] = str(model_size)
            model["sha256"]["const"] = model_sha256
        if mutate and name == mutate.get("name"):
            action = mutate["action"]
            if not callable(action):
                raise TypeError("schema mutation must be callable")
            action(value)
        payload = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
        path = root / name
        path.write_bytes(payload)
        result[name] = payload
    return result


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schema_references(value: object) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef") and type(item) is str:
                    found.append((key, item))
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return tuple(found)


def _replace_first_schema_reference(
    value: object,
    replacement: str,
    *,
    predicate,
    keyword: str = "$ref",
) -> None:
    def visit(node: object) -> bool:
        if isinstance(node, dict):
            for key, item in list(node.items()):
                if key == "$ref" and type(item) is str and predicate(item):
                    if keyword == "$ref":
                        node[key] = replacement
                    else:
                        del node[key]
                        node[keyword] = replacement
                    return True
                if visit(item):
                    return True
        elif isinstance(node, list):
            for item in node:
                if visit(item):
                    return True
        return False

    if not visit(value):
        raise AssertionError("schema fixture did not contain the expected reference")


def _schema_variant_root(
    inputs: Inputs,
    label: str,
    schema_name: str,
    action,
) -> Path:
    root = inputs.directory / f"schemas-{label}"
    _write_schema_bundle(
        root,
        model_size=inputs.model.stat().st_size,
        model_sha256=_sha256_file(inputs.model),
        mutate={"name": schema_name, "action": action},
    )
    return root


@contextmanager
def _spy_opened_descriptors(descriptors: list[int], paths: list[Path]):
    original_open = ADMISSION.open_qualification_artifact

    @contextmanager
    def spy(path, *args, **kwargs):
        paths.append(Path(path))
        with original_open(path, *args, **kwargs) as artifact:
            descriptors.append(artifact.fd)
            yield artifact

    with mock.patch.object(ADMISSION, "open_qualification_artifact", spy):
        yield


def _proc_targets(paths: list[Path]) -> set[str]:
    """Return open descriptor targets for only this fixture's paths."""
    proc = Path("/proc/self/fd")
    if not proc.is_dir():
        return set()
    wanted = {str(path.resolve()) for path in paths}
    found: set[str] = set()
    for entry in proc.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        target = target.removesuffix(" (deleted)")
        if target in wanted:
            found.add(target)
    return found


class QualificationAdmissionTest(unittest.TestCase):
    def _assert_fds_closed(self, descriptors: list[int]) -> None:
        for descriptor in descriptors:
            with self.subTest(fd=descriptor):
                with self.assertRaises(OSError) as raised:
                    os.fstat(descriptor)
                self.assertEqual(raised.exception.errno, errno.EBADF)

    def _assert_no_fixture_fds(self, inputs: Inputs) -> None:
        paths = [inputs.model, *inputs.binaries.values(), inputs.manifest_path]
        self.assertEqual(_proc_targets(paths), set())

    def _admit(self, inputs: Inputs, probe):
        return ADMISSION.admit_qualification_inputs(
            inputs.manifest_path,
            model_path=inputs.model,
            binaries=dict(inputs.binaries),
            version_probe=probe,
        )

    def _assert_rejected_before_probe(
        self,
        inputs: Inputs,
        *,
        schema_root: Path | None = None,
        manifest_path: Path | None = None,
        model_path: Path | str | None = None,
        binaries: dict[str, Path] | None = None,
        probe=None,
        expected_exceptions=(ValueError,),
    ) -> None:
        calls: list[str] = []

        def no_probe(role, artifact):
            calls.append(role)
            return inputs.version_bytes[role]

        before = sorted(path.name for path in inputs.directory.iterdir())
        selected_schema_root = inputs.schema_root if schema_root is None else schema_root
        selected_manifest_path = (
            inputs.manifest_path if manifest_path is None else manifest_path
        )
        selected_binaries = dict(inputs.binaries) if binaries is None else binaries
        selected_probe = no_probe if probe is None else probe
        with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", selected_schema_root):
            with self.assertRaises(expected_exceptions):
                with ADMISSION.admit_qualification_inputs(
                    selected_manifest_path,
                    model_path=model_path if model_path is not None else inputs.model,
                    binaries=selected_binaries,
                    version_probe=selected_probe,
                ):
                    self.fail("invalid admission unexpectedly yielded")
        self.assertEqual(calls, [])
        self.assertEqual(before, sorted(path.name for path in inputs.directory.iterdir()))
        self._assert_no_fixture_fds(inputs)

    def _assert_schema_owner_rejected_before_hash(
        self,
        inputs: Inputs,
        schema_root: Path,
        schema_name: str,
        *,
        max_schema_bytes: int | None = None,
    ) -> None:
        target = schema_root / schema_name
        opened: list[Path] = []
        descriptors: list[int] = []
        calls: list[str] = []
        target_stat = target.stat()
        target_identity = (target_stat.st_dev, target_stat.st_ino)
        target_descriptors: set[int] = set()
        original_open = ADMISSION.open_qualification_artifact
        original_os_open = ADMISSION.os.open
        original_pread = ADMISSION.os.pread
        original_read = ADMISSION.os.read

        @contextmanager
        def spy(path, *args, **kwargs):
            observed_path = Path(path)
            opened.append(observed_path)
            with original_open(path, *args, **kwargs) as artifact:
                descriptors.append(artifact.fd)
                yield artifact

        def recording_open(*args, **kwargs):
            descriptor = original_os_open(*args, **kwargs)
            try:
                observed = os.fstat(descriptor)
            except OSError:
                pass
            else:
                if (observed.st_dev, observed.st_ino) == target_identity:
                    target_descriptors.add(descriptor)
            return descriptor

        def no_target_pread(descriptor, size, offset):
            if descriptor in target_descriptors:
                raise AssertionError("bounded owner read a rejected schema")
            return original_pread(descriptor, size, offset)

        def no_target_read(descriptor, size):
            if descriptor in target_descriptors:
                raise AssertionError("bounded owner read a rejected schema")
            return original_read(descriptor, size)

        def no_probe(role, artifact):
            calls.append(role)
            return inputs.version_bytes[role]

        patches = [
            mock.patch.object(ADMISSION, "_SCHEMA_ROOT", schema_root),
            mock.patch.object(ADMISSION, "open_qualification_artifact", spy),
            mock.patch.object(ADMISSION.os, "open", side_effect=recording_open),
            mock.patch.object(ADMISSION.os, "pread", side_effect=no_target_pread),
            mock.patch.object(ADMISSION.os, "read", side_effect=no_target_read),
        ]
        if max_schema_bytes is not None:
            patches.append(
                mock.patch.object(ADMISSION, "MAX_SCHEMA_BYTES", max_schema_bytes)
            )
        with _patches(patches):
            with self.assertRaises((ValueError, OSError)):
                with ADMISSION.admit_qualification_inputs(
                    inputs.manifest_path,
                    model_path=inputs.model,
                    binaries=dict(inputs.binaries),
                    version_probe=no_probe,
                ):
                    self.fail("rejected schema unexpectedly yielded")
        self.assertIn(target, opened)
        self.assertEqual(calls, [])
        self._assert_fds_closed(descriptors)
        self._assert_fds_closed(list(target_descriptors))
        self._assert_no_fixture_fds(inputs)

    def _assert_rejected_without_external_owner(
        self,
        inputs: Inputs,
        schema_root: Path,
    ) -> None:
        descriptors: list[int] = []
        opened: list[Path] = []
        with _spy_opened_descriptors(descriptors, opened):
            self._assert_rejected_before_probe(inputs, schema_root=schema_root)
        allowed = {
            inputs.manifest_path,
            inputs.model,
            *inputs.binaries.values(),
            *(schema_root / name for name in SCHEMA_NAMES),
        }
        self.assertTrue(set(opened).issubset(allowed), opened)
        self._assert_fds_closed(descriptors)

    def test_normal_context_exit_rechecks_the_retained_manifest(self) -> None:
        with _tiny_inputs() as inputs:
            descriptors = []
            reached_body = False

            def probe(role, artifact):
                descriptors.append(artifact.fd)
                return inputs.version_bytes[role]

            with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                with self.assertRaises(ValueError):
                    with self._admit(inputs, probe):
                        reached_body = True
                        inputs.manifest_path.write_bytes(b"{}\n")
            self.assertTrue(reached_body)
            self.assertEqual(len(descriptors), 3)
            self._assert_fds_closed(descriptors)
            self._assert_no_fixture_fds(inputs)

    def test_success_has_exact_hashes_order_detachment_and_fd_closure(self) -> None:
        with _tiny_inputs() as inputs:
            seen: dict[str, object] = {}
            descriptors: list[int] = []

            def probe(role, artifact):
                self.assertIn(role, ROLES)
                expected_path = inputs.binaries[role].resolve()
                self.assertEqual(Path(str(artifact.path)).resolve(), expected_path)
                self.assertEqual(artifact.sha256, _sha256_file(expected_path))
                self.assertIsNotNone(artifact.identity)
                self.assertIsInstance(artifact.fd, int)
                self.assertTrue(callable(artifact.verify))
                seen[role] = artifact
                descriptors.append(artifact.fd)
                # The required callback receives the pinned owner object and
                # returns exact native stdout bytes, never a parsed mapping.
                return inputs.version_bytes[role]

            before = sorted(path.name for path in inputs.directory.iterdir())
            with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                with self._admit(inputs, probe) as admitted:
                    self.assertEqual(tuple(seen), ROLES)
                    self.assertEqual(
                        set(admitted.artifacts), {"model", "server", "bench", "eval"}
                    )
                    self.assertEqual(
                        Path(str(admitted.artifacts["model"].path)).resolve(),
                        inputs.model.resolve(),
                    )
                    self.assertIsInstance(admitted.artifacts["model"].fd, int)
                    descriptors.append(admitted.artifacts["model"].fd)
                    for role in ROLES:
                        self.assertIs(admitted.artifacts[role], seen[role])
                    # The tokenizer executable in prompt_source is not one of
                    # the three qualification binaries passed to this API.
                    self.assertNotIn(
                        inputs.manifest["prompt_source"]["tokenizer_runtime"]["executable_path"],
                        {str(admitted.artifacts[key].path) for key in ROLES},
                    )

                    self.assertEqual(admitted.manifest, inputs.manifest)
                    self.assertEqual(admitted.manifest_sha256, inputs.manifest_sha256)
                    self.assertEqual(
                        admitted.manifest_file_sha256,
                        _payload_hash(inputs.manifest_bytes),
                    )
                    self.assertNotEqual(
                        admitted.manifest_sha256, admitted.manifest_file_sha256
                    )

                    retained_schema_owners = {
                        owner.path.resolve(): owner
                        for owner in admitted._owners
                        if owner.path.parent == inputs.schema_root
                    }
                    self.assertEqual(
                        set(retained_schema_owners),
                        {(inputs.schema_root / name).resolve() for name in SCHEMA_NAMES},
                    )
                    for name in SCHEMA_NAMES:
                        self.assertEqual(
                            retained_schema_owners[(inputs.schema_root / name).resolve()].sha256,
                            _payload_hash(inputs.schema_bytes[name]),
                        )
                    for name in RECORD_SCHEMA_NAMES:
                        references = _schema_references(
                            json.loads(inputs.schema_bytes[name].decode("utf-8"))
                        )
                        self.assertEqual(
                            {
                                (key, value)
                                for key, value in references
                                if not value.startswith("#")
                            },
                            {
                                ("$ref", dependency)
                                for dependency in RECORD_SCHEMA_DEPENDENCIES
                            },
                        )
                        self.assertFalse(
                            any(
                                key in ("$dynamicRef", "$recursiveRef")
                                for key, _ in references
                            )
                        )

                    self.assertIs(type(admitted.schema_records), tuple)
                    self.assertEqual(
                        [record["schema_id"] for record in admitted.schema_records],
                        list(SCHEMA_IDS),
                    )
                    self.assertEqual(
                        admitted.schema_records,
                        tuple(
                            {
                                "schema_id": next(
                                    schema_id
                                    for schema_id, filename in zip(
                                        SCHEMA_IDS, SCHEMA_NAMES, strict=True
                                    )
                                    if filename == name
                                ),
                                "sha256": _payload_hash(payload),
                            }
                            for name, payload in inputs.schema_bytes.items()
                        ),
                    )
                    for record in admitted.schema_records:
                        self.assertEqual(set(record), {"schema_id", "sha256"})

                    self.assertIs(type(admitted.versions), tuple)
                    self.assertEqual(
                        [record["role"] for record in admitted.versions], list(ROLES)
                    )
                    for record in admitted.versions:
                        role = record["role"]
                        payload = inputs.version_bytes[role]
                        self.assertEqual(record["payload"], payload)
                        self.assertIs(type(record["payload"]), bytes)
                        self.assertEqual(record["sha256"], _payload_hash(payload))
                        self.assertEqual(
                            record["value"],
                            COMPACT.validate_qualification_version(
                                json.loads(payload.decode("utf-8"))
                            ),
                        )
                        self.assertEqual(
                            set(record), {"role", "payload", "sha256", "value"}
                        )

                    manifest_snapshot = copy.deepcopy(admitted.manifest)
                    schemas_snapshot = copy.deepcopy(admitted.schema_records)
                    versions_snapshot = copy.deepcopy(admitted.versions)
                    self.assertEqual(
                        sorted(path.name for path in inputs.directory.iterdir()), before
                    )

            self._assert_fds_closed(descriptors)
            self.assertEqual(admitted.manifest, manifest_snapshot)
            self.assertEqual(admitted.schema_records, schemas_snapshot)
            self.assertEqual(admitted.versions, versions_snapshot)
            inputs.manifest["prompts"].reverse()
            inputs.manifest["model"]["path"] = "/replaced/manifest-model.gguf"
            self.assertEqual(admitted.manifest, manifest_snapshot)
            admitted.schema_records[0]["schema_id"] = "detachment-check"
            self.assertEqual(admitted.schema_records, schemas_snapshot)
            self._assert_no_fixture_fds(inputs)

    def test_invalid_manifest_schema_and_bounds_fail_before_version_callbacks(self) -> None:
        with _tiny_inputs() as inputs:
            original_manifest = inputs.manifest_path.read_bytes()
            malformed = (
                b'{"schema":"ds4.compact-runtime-benchmark/v1",'
                b'"schema":"ds4.compact-runtime-benchmark/v1"}'
            )
            for payload in (malformed, b"\xff", b"{\"schema\":NaN}"):
                with self.subTest(manifest=payload[:20]):
                    inputs.manifest_path.write_bytes(payload)
                    self._assert_rejected_before_probe(inputs)
            inputs.manifest_path.write_bytes(original_manifest)

            with mock.patch.object(
                ADMISSION, "MAX_ADMISSION_JSON_BYTES", len(original_manifest) - 1
            ):
                self._assert_rejected_before_probe(inputs)

            variants = (
                (
                    "wrong schema id",
                    {
                        "name": SCHEMA_NAMES[0],
                        "action": lambda value: value.__setitem__("$id", "wrong/v1"),
                    },
                ),
                (
                    "wrong root const",
                    {
                        "name": SCHEMA_NAMES[0],
                        "action": lambda value: value["properties"]["schema"].__setitem__(
                            "const", "wrong/v1"
                        ),
                    },
                ),
                (
                    "network reference",
                    {
                        "name": SCHEMA_NAMES[0],
                        "action": lambda value: value["properties"]["revision"].__setitem__(
                            "$ref", "https://example.invalid/schema.json"
                        ),
                    },
                ),
            )
            for label, mutation in variants:
                with self.subTest(schema=label):
                    variant_root = inputs.directory / f"schemas-{label.replace(' ', '-') }"
                    _write_schema_bundle(
                        variant_root,
                        model_size=inputs.model.stat().st_size,
                        model_sha256=_sha256_file(inputs.model),
                        mutate=mutation,
                    )
                    self._assert_rejected_before_probe(inputs, schema_root=variant_root)

            malformed_root = inputs.directory / "schemas-malformed"
            malformed_bytes = _write_schema_bundle(
                malformed_root,
                model_size=inputs.model.stat().st_size,
                model_sha256=_sha256_file(inputs.model),
            )
            malformed_path = malformed_root / CORE_SCHEMA_NAMES[-1]
            malformed_path.write_bytes(
                malformed_bytes[CORE_SCHEMA_NAMES[-1]][:-1] + b'\n"'
            )
            self._assert_rejected_before_probe(inputs, schema_root=malformed_root)

            oversized_schema_root = inputs.directory / "schemas-oversized"
            oversized = _write_schema_bundle(
                oversized_schema_root,
                model_size=inputs.model.stat().st_size,
                model_sha256=_sha256_file(inputs.model),
            )
            largest = max(map(len, oversized.values()))
            with mock.patch.object(ADMISSION, "MAX_SCHEMA_BYTES", largest - 1):
                self._assert_rejected_before_probe(
                    inputs, schema_root=oversized_schema_root
                )

            other = inputs.directory / "other"
            other.mkdir()
            other_model = other / inputs.model.name
            other_model.write_bytes(inputs.model.read_bytes())
            other_model.chmod(0o644)
            self._assert_rejected_before_probe(inputs, model_path=other_model)
            self._assert_rejected_before_probe(
                inputs,
                binaries={**inputs.binaries, "extra": inputs.binaries["server"]},
            )
            self._assert_rejected_before_probe(
                inputs,
                binaries={**inputs.binaries, "server": Path("relative-server")},
            )

    def test_record_schema_files_are_required_and_identity_checked_before_version_probe(self) -> None:
        for schema_name, schema_id in zip(RECORD_SCHEMA_NAMES, RECORD_SCHEMA_IDS, strict=True):
            for kind in ("missing", "malformed", "wrong-id", "wrong-schema-const"):
                with self.subTest(schema=schema_name, failure=kind), _tiny_inputs() as inputs:
                    if kind == "wrong-id":
                        action = lambda value: value.__setitem__("$id", "wrong/v1")
                    elif kind == "wrong-schema-const":
                        action = lambda value, schema_id=schema_id: value["properties"][
                            "schema"
                        ].__setitem__("const", "wrong/v1")
                    else:
                        action = lambda value: None
                    root = _schema_variant_root(
                        inputs,
                        f"record-{schema_name.replace('.schema.json', '')}-{kind}",
                        schema_name,
                        action,
                    )
                    target = root / schema_name
                    if kind == "missing":
                        target.unlink()
                    elif kind == "malformed":
                        target.write_bytes(b'{"schema":"unterminated')
                    self._assert_rejected_before_probe(
                        inputs,
                        schema_root=root,
                        expected_exceptions=(ValueError, OSError)
                        if kind == "missing"
                        else (ValueError,),
                    )

    def test_new_schema_owner_rejects_symlinks_and_over_bound_files_before_hash_read_probe(self) -> None:
        for schema_name in RECORD_SCHEMA_NAMES:
            with self.subTest(schema=schema_name, owner_failure="symlink"), _tiny_inputs() as inputs:
                root = _schema_variant_root(
                    inputs,
                    f"symlink-{schema_name.replace('.schema.json', '')}",
                    schema_name,
                    lambda value: None,
                )
                target = root / schema_name
                outside = inputs.directory / f"outside-{schema_name}"
                outside.write_bytes(target.read_bytes())
                target.unlink()
                target.symlink_to(outside)
                self._assert_schema_owner_rejected_before_hash(
                    inputs, root, schema_name,
                )

            with self.subTest(schema=schema_name, owner_failure="size-cap"), _tiny_inputs() as inputs:
                root = _schema_variant_root(
                    inputs,
                    f"oversized-{schema_name.replace('.schema.json', '')}",
                    schema_name,
                    lambda value: None,
                )
                target = root / schema_name
                other_sizes = [
                    (root / name).stat().st_size
                    for name in SCHEMA_NAMES
                    if name != schema_name
                ]
                cap = max(other_sizes) + 1
                padding = max(0, cap - target.stat().st_size + 1)
                target.write_bytes(target.read_bytes() + b" " * padding)
                self._assert_schema_owner_rejected_before_hash(
                    inputs, root, schema_name, max_schema_bytes=cap,
                )

    def test_schema_reference_policy_allows_only_canonical_record_dependencies(self) -> None:
        invalid_targets = (
            ("url", "https://example.invalid/ds4-runtime-v1.schema.json"),
            ("file-url", "file:///tmp/ds4-runtime-v1.schema.json"),
            ("traversal", "../ds4-runtime-v1.schema.json"),
            ("relative-alias", "./ds4-runtime-v1.schema.json"),
            ("unknown", "ds4-unknown-v1.schema.json"),
        )
        with _tiny_inputs() as inputs:
            for schema_name in RECORD_SCHEMA_NAMES:
                for label, target in invalid_targets:
                    with self.subTest(schema=schema_name, reference=label):
                        root = _schema_variant_root(
                            inputs,
                            f"ref-{schema_name.replace('.schema.json', '')}-{label}",
                            schema_name,
                            lambda value, target=target: _replace_first_schema_reference(
                                value,
                                target,
                                predicate=lambda ref: ref in RECORD_SCHEMA_DEPENDENCIES,
                            ),
                        )
                        self._assert_rejected_without_external_owner(inputs, root)

                for keyword in ("$dynamicRef", "$recursiveRef"):
                    with self.subTest(schema=schema_name, reference=keyword):
                        root = _schema_variant_root(
                            inputs,
                            f"ref-{schema_name.replace('.schema.json', '')}-{keyword[1:]}",
                            schema_name,
                            lambda value, keyword=keyword: _replace_first_schema_reference(
                                value,
                                "ds4-runtime-v1.schema.json",
                                predicate=lambda ref: ref in RECORD_SCHEMA_DEPENDENCIES,
                                keyword=keyword,
                            ),
                        )
                        self._assert_rejected_without_external_owner(inputs, root)

            for schema_name in CORE_SCHEMA_NAMES:
                with self.subTest(schema=schema_name, reference="core-nonlocal"):
                    root = _schema_variant_root(
                        inputs,
                        f"core-ref-{schema_name.replace('.schema.json', '')}",
                        schema_name,
                        lambda value: _replace_first_schema_reference(
                            value,
                            "ds4-runtime-v1.schema.json",
                            predicate=lambda ref: ref.startswith("#"),
                        ),
                    )
                    self._assert_rejected_without_external_owner(inputs, root)

    def test_record_schema_mutations_fail_verify_and_probe_guards(self) -> None:
        for schema_name in RECORD_SCHEMA_NAMES:
            with self.subTest(schema=schema_name, check="admission.verify"), _tiny_inputs() as inputs:
                target = inputs.schema_root / schema_name
                descriptors: list[int] = []
                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with self.assertRaises(ValueError):
                        with self._admit(inputs, lambda role, artifact: inputs.version_bytes[role]) as admitted:
                            owners = {
                                owner.path.name: owner for owner in admitted._owners
                            }
                            self.assertIn(schema_name, owners)
                            descriptors.extend(owner.fd for owner in admitted._owners)
                            target.write_bytes(target.read_bytes() + b"\n")
                            with self.assertRaises(ValueError):
                                admitted.verify()
                self.assertTrue(
                    descriptors,
                    "admission.verify mutation did not reach the yielded admission",
                )
                self._assert_fds_closed(descriptors)
                self._assert_no_fixture_fds(inputs)

            with self.subTest(schema=schema_name, check="probe-guards"), _tiny_inputs() as inputs:
                target = inputs.schema_root / schema_name
                descriptors: list[int] = []
                opened: list[Path] = []
                seen_roles: list[str] = []

                def mutating_probe(role, artifact):
                    seen_roles.append(role)
                    if role == "server":
                        target.write_bytes(target.read_bytes() + b"\n")
                    return inputs.version_bytes[role]

                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with _spy_opened_descriptors(descriptors, opened):
                        with self.assertRaises(ValueError):
                            with ADMISSION.admit_qualification_inputs(
                                inputs.manifest_path,
                                model_path=inputs.model,
                                binaries=dict(inputs.binaries),
                                version_probe=mutating_probe,
                            ):
                                self.fail("schema mutation unexpectedly reached yield")
                self.assertEqual(seen_roles, ["server"])
                self.assertIn(target, opened)
                self._assert_fds_closed(descriptors)
                self._assert_no_fixture_fds(inputs)

    def test_schema_owner_descriptors_close_on_normal_error_and_interrupt_exit(self) -> None:
        exits = (
            ("normal", None),
            ("error", RuntimeError),
            ("interrupt", KeyboardInterrupt),
        )
        for label, exception_type in exits:
            with self.subTest(exit=label), _tiny_inputs() as inputs:
                descriptors: list[int] = []
                opened: list[Path] = []
                with _spy_opened_descriptors(descriptors, opened):
                    with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                        if exception_type is None:
                            with self._admit(
                                inputs,
                                lambda role, artifact: inputs.version_bytes[role],
                            ):
                                pass
                        else:
                            with self.assertRaises(exception_type):
                                with self._admit(
                                    inputs,
                                    lambda role, artifact: inputs.version_bytes[role],
                                ):
                                    raise exception_type(f"forced {label} exit")
                self.assertEqual(
                    {
                        path.resolve()
                        for path in opened
                        if path.parent == inputs.schema_root
                    },
                    {(inputs.schema_root / name).resolve() for name in SCHEMA_NAMES},
                )
                self._assert_fds_closed(descriptors)
                self._assert_no_fixture_fds(inputs)

    def test_dirty_version_digest_stat_drift_and_callback_failure_close_descriptors(self) -> None:
        revision = str(
            FIXTURE.RUNTIME_IDENTITY["source_revision"]
        )
        invalid_payloads = {
            "dirty": _version_bytes(revision, dirty=True),
            "non-cuda": _version_bytes(revision, backend="cpu"),
            "missing-feature": _version_bytes(revision, features=["laguna"]),
            "wrong-revision": _version_bytes(revision[:-1] + "b"),
        }

        for label, bad_payload in invalid_payloads.items():
            with self.subTest(version=label), _tiny_inputs() as inputs:
                seen_fds: list[int] = []
                seen_roles: list[str] = []

                def probe(role, artifact):
                    seen_roles.append(role)
                    seen_fds.append(artifact.fd)
                    return bad_payload if role == "server" else inputs.version_bytes[role]

                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with self.assertRaises(ValueError):
                        with self._admit(inputs, probe):
                            self.fail("invalid version unexpectedly yielded")
                self.assertEqual(seen_roles, ["server"])
                self.assertTrue(seen_fds)
                self._assert_fds_closed(seen_fds)
                self._assert_no_fixture_fds(inputs)

        mutation_cases = (
            ("binary-content", lambda inputs, artifact: inputs.binaries["server"].write_bytes(b"changed")),
            ("binary-stat", lambda inputs, artifact: os.utime(inputs.binaries["server"], ns=(1, 1))),
            ("model-content", lambda inputs, artifact: inputs.model.write_bytes(b"changed model")),
            ("callback-closes-fd", lambda inputs, artifact: os.close(artifact.fd)),
        )
        for label, mutate in mutation_cases:
            with self.subTest(drift=label), _tiny_inputs() as inputs:
                seen_fds: list[int] = []

                def probe(role, artifact):
                    seen_fds.append(artifact.fd)
                    if role == "server":
                        mutate(inputs, artifact)
                    return inputs.version_bytes[role]

                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with self.assertRaises((ValueError, OSError)):
                        with self._admit(inputs, probe):
                            self.fail("mutated artifact unexpectedly yielded")
                self.assertTrue(seen_fds)
                self._assert_fds_closed(seen_fds)
                self._assert_no_fixture_fds(inputs)

        with _tiny_inputs() as inputs:
            seen_fds: list[int] = []
            seen_roles: list[str] = []

            def callback_failure(role, artifact):
                seen_roles.append(role)
                seen_fds.append(artifact.fd)
                if role == "bench":
                    raise RuntimeError("trusted probe failed")
                return inputs.version_bytes[role]

            with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                with self.assertRaisesRegex(RuntimeError, "trusted probe failed"):
                    with self._admit(inputs, callback_failure):
                        self.fail("callback failure unexpectedly yielded")
            self.assertEqual(seen_roles, ["server", "bench"])
            self.assertTrue(seen_fds)
            self._assert_fds_closed(seen_fds)
            self._assert_no_fixture_fds(inputs)

        with _tiny_inputs() as inputs:
            seen_fds: list[int] = []

            def parsed_result(role, artifact):
                seen_fds.append(artifact.fd)
                return json.loads(inputs.version_bytes[role].decode("utf-8"))

            with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                with self.assertRaises(ValueError):
                    with self._admit(inputs, parsed_result):
                        self.fail("parsed callback result unexpectedly yielded")
            self.assertTrue(seen_fds)
            self._assert_fds_closed(seen_fds)
            self._assert_no_fixture_fds(inputs)

        with _tiny_inputs() as inputs:
            seen_fds: list[int] = []
            with mock.patch.object(
                ADMISSION, "MAX_VERSION_BYTES", len(inputs.version_bytes["server"]) - 1
            ):
                def oversized(role, artifact):
                    seen_fds.append(artifact.fd)
                    return inputs.version_bytes[role]

                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with self.assertRaises(ValueError):
                        with self._admit(inputs, oversized):
                            self.fail("oversized version unexpectedly yielded")
            self.assertTrue(seen_fds)
            self._assert_fds_closed(seen_fds)
            self._assert_no_fixture_fds(inputs)

    def test_manifest_or_file_replacement_between_probe_and_yield_fails_closed(self) -> None:
        for replacement_kind in ("manifest", "binary", "model"):
            with self.subTest(replacement=replacement_kind), _tiny_inputs() as inputs:
                before = sorted(path.name for path in inputs.directory.iterdir())
                seen_fds: list[int] = []

                def probe(role, artifact):
                    seen_fds.append(artifact.fd)
                    if role == "server":
                        if replacement_kind == "manifest":
                            replacement = inputs.directory / "manifest-replacement.json"
                            replacement.write_bytes(inputs.manifest_bytes)
                            os.replace(replacement, inputs.manifest_path)
                        elif replacement_kind == "binary":
                            replacement = inputs.directory / "server-replacement"
                            replacement.write_bytes(b"replacement server binary\n")
                            replacement.chmod(0o755)
                            os.replace(replacement, inputs.binaries["server"])
                        else:
                            replacement = inputs.directory / "model-replacement.gguf"
                            replacement.write_bytes(b"replacement model bytes")
                            os.replace(replacement, inputs.model)
                    return inputs.version_bytes[role]

                with mock.patch.object(ADMISSION, "_SCHEMA_ROOT", inputs.schema_root):
                    with self.assertRaises((ValueError, OSError)):
                        with self._admit(inputs, probe):
                            self.fail("replacement unexpectedly yielded")
                self.assertTrue(seen_fds)
                self._assert_fds_closed(seen_fds)
                self._assert_no_fixture_fds(inputs)
                self.assertEqual(
                    before, sorted(path.name for path in inputs.directory.iterdir())
                )


if __name__ == "__main__":
    unittest.main()
