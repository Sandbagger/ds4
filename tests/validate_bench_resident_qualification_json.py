#!/usr/bin/env python3
"""Validate one resident fixture lifecycle using the production contract."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools/quality-testing"))
from qualification_records import _read_bounded_records, _validate_lifecycle
from qualification_resident_records import validate_resident_record


def main():
    try:
        records = _read_bounded_records(sys.stdin.buffer)
        for record in records:
            validate_resident_record(record)
        _validate_lifecycle(records)
    except (TypeError, ValueError, RecursionError) as exc:
        print(f"resident-qualification-json: {exc}", file=sys.stderr)
        return 1
    print(f"resident-qualification-json: {len(records)} strict record(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
