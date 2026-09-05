#!/usr/bin/env python3
"""Build the distinct, immutable resident qualification sequence contract.

This is an input format for an authenticated producer, not a resident runtime
snapshot, allocation plan, executed baseline, or qualification verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compact_runtime_qualify import (
    PROMPT_TARGETS,
    _qualification_sequence_bytes,
    validate_manifest,
)

RESIDENT_QUALIFICATION_SEQUENCE_SCHEMA = "ds4.resident-qualification-sequence/v1"


def build_resident_qualification_sequence(
    manifest: Mapping[str, Any], prompt_id: str,
) -> bytes:
    """Bind a canonical prompt to exactly one cold and three warm repetitions."""
    if not isinstance(manifest, Mapping):
        raise ValueError("resident qualification manifest must be an object")
    validate_manifest(manifest)
    if type(prompt_id) is not str or not prompt_id:
        raise ValueError("resident qualification prompt_id must be a nonempty string")
    prompt_ids = tuple(f"native-{target}" for target in PROMPT_TARGETS)
    if prompt_id not in prompt_ids:
        raise ValueError("unknown resident qualification prompt_id")
    order_index = prompt_ids.index(prompt_id)
    return _qualification_sequence_bytes(
        manifest, prompt_id, schema=RESIDENT_QUALIFICATION_SEQUENCE_SCHEMA,
        profile_id="resident", cache_bytes=0, prompt_order_index=order_index,
        prompt_tokens=PROMPT_TARGETS[order_index], mode="resident",
    )
