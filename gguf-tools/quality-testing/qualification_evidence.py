"""Build the canonical qualification evidence index from declarations only.

Observations are untrusted declarations until a later descriptor reader checks
file identity, bytes, and authentication.  This module does not inspect files,
symlinks, bundles, gates, or qualification status.  ``evidence_root_sha256``
only hashes the bytes it receives; it does not assert canonicality or
authenticity.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import rfc8785


_EVIDENCE_INDEX_NAME = "evidence-index.json"
_MAX_UINT64 = "18446744073709551615"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UINT64_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_REFERENCE_KEYS = frozenset(("path", "sha256"))
_OBSERVATION_KEYS = frozenset(("path", "size_bytes", "sha256"))


def _validate_path(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8") from exc
    if not value or value.startswith("/") or value.endswith("/"):
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    if any(component in ("", ".", "..") for component in value.split("/")):
        raise ValueError(f"{field} contains a non-normal POSIX component")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field} contains a control character")
    return value


def _validate_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase hexadecimal SHA-256")
    return value


def _validate_size_bytes(value: Any, *, field: str) -> str:
    if type(value) is not str or len(value) > len(_MAX_UINT64):
        raise ValueError(f"{field} must be a canonical uint64 decimal string")
    if _UINT64_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical uint64 decimal string")
    # Compare fixed-width decimal strings without converting unbounded input.
    if len(value) == len(_MAX_UINT64) and value > _MAX_UINT64:
        raise ValueError(f"{field} exceeds uint64")
    return value


def _validate_container(value: Any, *, field: str) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{field} must be a list or tuple")
    return value


def _validate_record(record: Any, *, keys: frozenset[str], field: str) -> dict[str, Any]:
    if type(record) is not dict:
        raise TypeError(f"{field} must be an object")
    if frozenset(record) != keys:
        raise ValueError(f"{field} has the wrong closed shape")
    return record


def build_evidence_index(
    references: Any,
    observations: Any,
    *,
    bundle_path: Any,
) -> bytes:
    """Return canonical evidence-index bytes for matching declarations.

    ``references`` and ``observations`` are declarations only.  A later
    descriptor reader must authenticate the observed files; this function never
    opens, stats, hashes, or otherwise accesses any path.
    """

    bundle_path = _validate_path(bundle_path, field="bundle_path")
    if bundle_path == _EVIDENCE_INDEX_NAME:
        raise ValueError("bundle_path cannot be evidence-index.json")
    reserved_paths = frozenset(
        (bundle_path, f"{bundle_path}.sha256", _EVIDENCE_INDEX_NAME)
    )

    references = _validate_container(references, field="references")
    observations = _validate_container(observations, field="observations")

    reference_digests: dict[str, str] = {}
    for index, record in enumerate(references):
        record = _validate_record(
            record, keys=_REFERENCE_KEYS, field=f"references[{index}]"
        )
        path = _validate_path(record["path"], field=f"references[{index}].path")
        digest = _validate_sha256(
            record["sha256"], field=f"references[{index}].sha256"
        )
        if path in reserved_paths:
            raise ValueError(f"reserved evidence path: {path}")
        previous = reference_digests.get(path)
        if previous is not None and previous != digest:
            raise ValueError(f"conflicting reference digest for path: {path}")
        reference_digests[path] = digest

    observed_values: dict[str, tuple[str, str]] = {}
    for index, record in enumerate(observations):
        record = _validate_record(
            record, keys=_OBSERVATION_KEYS, field=f"observations[{index}]"
        )
        path = _validate_path(record["path"], field=f"observations[{index}].path")
        size_bytes = _validate_size_bytes(
            record["size_bytes"], field=f"observations[{index}].size_bytes"
        )
        digest = _validate_sha256(
            record["sha256"], field=f"observations[{index}].sha256"
        )
        if path in reserved_paths:
            raise ValueError(f"reserved evidence path: {path}")
        if path in observed_values:
            raise ValueError(f"duplicate observation path: {path}")
        observed_values[path] = (size_bytes, digest)

    if not reference_digests:
        raise ValueError("evidence set must not be empty")
    if set(reference_digests) != set(observed_values):
        raise ValueError("references and observations must have the exact same paths")

    entries: list[dict[str, str]] = []
    for path in sorted(observed_values, key=lambda item: item.encode("utf-8")):
        size_bytes, digest = observed_values[path]
        if reference_digests[path] != digest:
            raise ValueError(f"reference and observation digest differ for path: {path}")
        entries.append(
            {"path": path, "size_bytes": size_bytes, "sha256": digest}
        )
    return rfc8785.dumps(entries)


def evidence_root_sha256(index_bytes: bytes) -> str:
    """Hash exactly ``index_bytes`` without parsing or authenticating them."""

    return hashlib.sha256(index_bytes).hexdigest()


__all__ = ["build_evidence_index", "evidence_root_sha256"]
