#!/usr/bin/env python3
"""Compatibility shim for the production qualification-record validator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_PATH = _ROOT / "gguf-tools/quality-testing/qualification_records.py"
_SPEC = importlib.util.spec_from_file_location(
    "qualification_records", _PRODUCTION_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load qualification-record validator: {_PRODUCTION_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

# Keep the historical module surface, including private helpers used by older
# local tooling, while making the production module the single implementation.
for _name, _value in vars(_MODULE).items():
    if _name not in {"__name__", "__loader__", "__package__", "__spec__", "__file__"}:
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(main())
