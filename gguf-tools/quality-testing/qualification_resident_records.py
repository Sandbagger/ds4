#!/usr/bin/env python3
"""Strict resident lifecycle validation, separate from the streamed boundary.

This consumes raw observations.  It does not launch a producer, authenticate
files/processes, sample native allocation state, or decide qualification gates.
"""
from __future__ import annotations

from typing import Any

import qualification_records as _records

SCHEMA_PATH = _records.ROOT / "schemas/ds4-bench-resident-qualification-v1.schema.json"
EXPECTED_BINDINGS = _records.EXPECTED_BINDINGS
MAX_JSON_NESTING = _records.MAX_JSON_NESTING
MAX_RECORD_BYTES = _records.MAX_RECORD_BYTES
MAX_RECORD_COUNT = _records.MAX_RECORD_COUNT
MAX_STREAM_BYTES = _records.MAX_STREAM_BYTES


def validate_resident_record(record: Any, *, complete: bool | None = None) -> None:
    """Validate resident schema, geometry, footprint and request bindings."""
    _records._validate_record_with_contract(
        record, schema_path=SCHEMA_PATH, resident=True, complete=complete,
    )


class QualificationResidentRecordStream(_records._QualificationRecordStreamBase):
    """Consume one resident cold/three-warm slice with copied partial evidence.

    The fixed class selects the validator.  The wire cannot change the kind.
    Framing, lifecycle, failure and drain/finish ownership use the same tested
    implementation as the streamed boundary; byte caps remain module-local.
    """

    def _record_byte_limit(self) -> int:
        return MAX_RECORD_BYTES

    def _stream_byte_limit(self) -> int:
        return MAX_STREAM_BYTES

    def _validate_record(self, record: Any) -> None:
        validate_resident_record(record)
