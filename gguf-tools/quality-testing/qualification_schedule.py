#!/usr/bin/env python3
"""Build the pure, ordered qualification-slice schedule.

This module only prepares deterministic sequence inputs.  Admission,
authentication, execution, evidence, and status decisions belong to the
caller that consumes this schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from compact_runtime_qualify import (
    PROFILE_SPECS,
    PROMPT_TARGETS,
    build_qualification_sequence,
    manifest_sha256,
    validate_manifest,
)
from qualification_resident_sequence import build_resident_qualification_sequence


@dataclass(frozen=True, slots=True)
class QualificationSlice:
    """One immutable sequence binding for a qualification run."""

    record_kind: str
    profile_id: str
    prompt_id: str
    prompt_order_index: int
    manifest_sha256: str
    sequence_sha256: str
    sequence_bytes: bytes


def _slice(
    *,
    record_kind: str,
    profile_id: str,
    prompt_id: str,
    prompt_order_index: int,
    manifest_digest: str,
    sequence_bytes: bytes,
) -> QualificationSlice:
    if type(sequence_bytes) is not bytes:
        raise TypeError("qualification sequence builder must return bytes")
    return QualificationSlice(
        record_kind=record_kind,
        profile_id=profile_id,
        prompt_id=prompt_id,
        prompt_order_index=prompt_order_index,
        manifest_sha256=manifest_digest,
        sequence_sha256=sha256(sequence_bytes).hexdigest(),
        sequence_bytes=sequence_bytes,
    )


def build_qualification_schedule(
    manifest: Mapping[str, Any],
) -> tuple[QualificationSlice, ...]:
    """Return resident canonical prompts followed by the streamed sweep.

    Validation happens before either trusted sequence builder is called.  The
    validated manifest is deep-copied once so sequence builders cannot mutate
    the caller's object or any of its nested values.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("qualification schedule manifest must be an object")
    validate_manifest(manifest)
    detached_manifest = deepcopy(manifest)
    validate_manifest(detached_manifest)
    manifest_digest = manifest_sha256(detached_manifest)

    records: list[QualificationSlice] = []
    for prompt_order_index, token_count in enumerate(PROMPT_TARGETS):
        prompt_id = f"native-{token_count}"
        sequence_bytes = build_resident_qualification_sequence(
            detached_manifest, prompt_id
        )
        records.append(
            _slice(
                record_kind="resident",
                profile_id="resident",
                prompt_id=prompt_id,
                prompt_order_index=prompt_order_index,
                manifest_digest=manifest_digest,
                sequence_bytes=sequence_bytes,
            )
        )

    for profile_id, _cache_bytes, prompt_order in PROFILE_SPECS:
        for prompt_order_index, token_count in enumerate(prompt_order):
            prompt_id = f"native-{token_count}"
            sequence_bytes = build_qualification_sequence(
                detached_manifest, profile_id, prompt_id
            )
            records.append(
                _slice(
                    record_kind="streamed",
                    profile_id=profile_id,
                    prompt_id=prompt_id,
                    prompt_order_index=prompt_order_index,
                    manifest_digest=manifest_digest,
                    sequence_bytes=sequence_bytes,
                )
            )

    return tuple(records)
