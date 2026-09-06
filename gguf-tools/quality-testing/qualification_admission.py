#!/usr/bin/env python3
"""Admit pinned Task20 inputs without creating results or running a child.

The required version probe is trusted caller code.  This layer validates and
retains its exact response, but does not prove the response's process origin.
The full runner must supply an owned descriptor-bound native probe, check
runtime identity and collect real qualification evidence before any verdict.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry

import compact_runtime_qualify as COMPACT
from qualification_artifacts import (
    QualificationArtifact,
    _normal_absolute_path,
    open_qualification_artifact,
)

MAX_ADMISSION_JSON_BYTES = 32 << 20
MAX_SCHEMA_BYTES = 2 << 20
MAX_VERSION_BYTES = 65_536
_JSON_DEPTH = 64
_SCHEMA_ROOT = COMPACT.ROOT / "schemas"
_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_RECORD_SCHEMA_SPECS = (
    ("ds4.bench.qualification/v1", "ds4-bench-qualification-v1.schema.json"),
    ("ds4.bench.resident-qualification/v1", "ds4-bench-resident-qualification-v1.schema.json"),
)
_SCHEMA_SPECS = (
    ("ds4.version/v1", "ds4-version-v1.schema.json"),
    ("ds4.runtime/v1", "ds4-runtime-v1.schema.json"),
    ("ds4.runtime.request/v1", "ds4-runtime-request-v1.schema.json"),
    ("ds4.token-admission/v1", "ds4-token-admission-v1.schema.json"),
    ("ds4.laguna.compact-runtime/v1", "ds4-laguna-compact-runtime-v1.schema.json"),
    ("ds4.compact-runtime-benchmark/v1", "compact-runtime-benchmark-v1.schema.json"),
    *_RECORD_SCHEMA_SPECS,
)
_RECORD_SCHEMA_IDS = frozenset(schema_id for schema_id, _ in _RECORD_SCHEMA_SPECS)
# These literal dependencies already have required owners in _SCHEMA_SPECS.
# Admitting their names does not enable filesystem or network retrieval.
_RECORD_SCHEMA_DEPENDENCIES = frozenset({
    "ds4-runtime-v1.schema.json", "ds4-runtime-request-v1.schema.json",
})
_ROLES = ("server", "bench", "eval")


def _no_retrieval(uri: str) -> Any:
    raise ValueError("qualification schemas cannot retrieve external resources")


_LOCAL_REGISTRY = Registry(retrieve=_no_retrieval)


def _document(payload: bytes, maximum: int, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        raise ValueError(f"{label} must be nonempty bytes within its bound")
    depth = 0
    quoted = escaped = False
    for byte in payload:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > _JSON_DEPTH:
                raise ValueError(f"{label} exceeds the JSON depth bound")
        elif byte in (93, 125):
            depth -= 1
    try:
        value = COMPACT.loads_strict(payload.decode("utf-8", errors="strict"))
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict bounded UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _read_json_owner(
    stack: ExitStack, path: Path | str, maximum: int, label: str,
) -> tuple[QualificationArtifact, bytes, dict[str, Any]]:
    # The cap is enforced before any hash/read, not after loading the file.
    owner = stack.enter_context(open_qualification_artifact(path, max_bytes=maximum))
    payload = bytearray()
    while len(payload) < owner.identity.size_bytes:
        try:
            part = os.pread(
                owner.fd, min(1 << 20, owner.identity.size_bytes - len(payload)),
                len(payload),
            )
        except InterruptedError:
            continue
        if not part:
            raise ValueError(f"{label} ended before its pinned size")
        payload.extend(part)
    raw = bytes(payload)
    owner.verify()
    if hashlib.sha256(raw).hexdigest() != owner.sha256:
        raise ValueError(f"{label} bytes changed after hashing")
    return owner, raw, _document(raw, maximum, label)


def _schema_validator(value: dict[str, Any], schema_id: str) -> Draft202012Validator:
    if (value.get("$schema") != _DRAFT or value.get("$id") != schema_id or
            not isinstance(value.get("properties"), dict) or
            value["properties"].get("schema") != {"const": schema_id}):
        raise ValueError(f"qualification schema identity mismatch: {schema_id}")
    pending: list[Any] = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            for key, item in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef") and (
                    type(item) is not str or (
                        not item.startswith("#") and not (
                            key == "$ref" and schema_id in _RECORD_SCHEMA_IDS and
                            item in _RECORD_SCHEMA_DEPENDENCIES
                        )
                    )
                ):
                    raise ValueError("qualification schema has an unadmitted reference")
                pending.append(item)
        elif isinstance(node, list):
            pending.extend(node)
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ValueError(f"invalid qualification schema: {schema_id}") from exc
    return Draft202012Validator(
        value, format_checker=Draft202012Validator.FORMAT_CHECKER,
        registry=_LOCAL_REGISTRY,
    )


def _validate(validator: Draft202012Validator, value: Any, label: str) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        # Do not put an unbounded repr of the failed input into diagnostics.
        raise ValueError(f"{label} failed a {exc.validator} schema constraint") from exc


@dataclass(frozen=True)
class QualificationAdmission:
    manifest_sha256: str
    manifest_file_sha256: str
    _manifest: dict[str, Any] = field(repr=False)
    _schemas: tuple[dict[str, Any], ...] = field(repr=False)
    _versions: tuple[dict[str, Any], ...] = field(repr=False)
    _artifacts: dict[str, QualificationArtifact] = field(repr=False)
    _owners: tuple[QualificationArtifact, ...] = field(repr=False)

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def schema_records(self) -> tuple[dict[str, Any], ...]:
        return copy.deepcopy(self._schemas)

    @property
    def versions(self) -> tuple[dict[str, Any], ...]:
        return copy.deepcopy(self._versions)

    @property
    def artifacts(self) -> dict[str, QualificationArtifact]:
        return dict(self._artifacts)

    def verify(self) -> None:
        """Recheck every retained input; requires the live admission context."""
        for owner in self._owners:
            owner.verify()


@contextmanager
def admit_qualification_inputs(
    manifest_path: Path | str, *, model_path: Path | str,
    binaries: Mapping[str, Path | str],
    version_probe: Callable[[str, QualificationArtifact], bytes],
) -> Iterator[QualificationAdmission]:
    """Validate the immutable inputs before the enclosing runner creates results.

    No output directory or process is created here.  All descriptors survive
    through the yielded context and close on normal, error or interrupt exit.
    The three version responses are claims until an owned native probe binds
    their origin; no running-executable or served-model identity is inferred.
    """
    if not isinstance(binaries, Mapping) or set(binaries) != set(_ROLES):
        raise ValueError("qualification binaries must contain server, bench and eval")
    if not callable(version_probe):
        raise ValueError("qualification version probe must be callable")
    source = _normal_absolute_path(manifest_path)
    model_source = _normal_absolute_path(model_path)
    binary_paths = {role: _normal_absolute_path(binaries[role]) for role in _ROLES}
    for limit in (MAX_ADMISSION_JSON_BYTES, MAX_SCHEMA_BYTES, MAX_VERSION_BYTES):
        if type(limit) is not int or limit <= 0:
            raise ValueError("qualification admission limits must be positive integers")

    with ExitStack() as stack:
        manifest_owner, manifest_raw, manifest = _read_json_owner(
            stack, source, MAX_ADMISSION_JSON_BYTES, "qualification manifest",
        )
        validators = {}
        schema_records = []
        owners = [manifest_owner]
        for schema_id, filename in _SCHEMA_SPECS:
            owner, _, document = _read_json_owner(
                stack, _SCHEMA_ROOT / filename, MAX_SCHEMA_BYTES, "qualification schema",
            )
            owners.append(owner)
            validators[schema_id] = _schema_validator(document, schema_id)
            schema_records.append({"schema_id": schema_id, "sha256": owner.sha256})
        _validate(validators[COMPACT.SCHEMA_ID], manifest, "qualification manifest")
        COMPACT.validate_manifest(manifest)
        canonical_digest = COMPACT.manifest_sha256(manifest)
        if model_source != _normal_absolute_path(manifest["model"]["path"]):
            raise ValueError("qualification model path differs from the immutable manifest")
        expected_model = {
            key: manifest["model"][key]
            for key in ("device", "inode", "size_bytes", "mtime_ns", "sha256")
        }
        artifacts = {
            "model": stack.enter_context(open_qualification_artifact(
                model_source, expected=expected_model,
            )),
        }
        for role in _ROLES:
            artifacts[role] = stack.enter_context(open_qualification_artifact(
                binary_paths[role], executable=True,
            ))
        owners.extend(artifacts.values())
        versions = []
        expected_revision = manifest["prompt_source"]["tokenizer_runtime"]["source_revision"]
        for role in _ROLES:
            for owner in owners:
                owner.verify()
            payload = version_probe(role, artifacts[role])
            for owner in owners:
                owner.verify()
            value = _document(payload, MAX_VERSION_BYTES, "qualification version response")
            _validate(validators["ds4.version/v1"], value, "qualification version response")
            version = COMPACT.validate_qualification_version(value)
            if version["revision"] != expected_revision:
                raise ValueError("qualification binary revision differs from the manifest")
            versions.append({
                "role": role, "payload": payload,
                "sha256": hashlib.sha256(payload).hexdigest(), "value": version,
            })
        admission = QualificationAdmission(
            canonical_digest, hashlib.sha256(manifest_raw).hexdigest(),
            copy.deepcopy(manifest), tuple(schema_records), tuple(versions),
            artifacts, tuple(owners),
        )
        admission.verify()
        yield admission
        admission.verify()
