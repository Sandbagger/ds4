#!/usr/bin/env python3
"""Contract for the standalone Task 16 runtime-identity publication gate."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
TARGET = "test-laguna-runtime-identity"
REQUIREMENTS = "gguf-tools/quality-testing/requirements-compact-runtime.txt"


def _rule_text(target: str) -> str:
    match = re.search(rf"(?m)^{re.escape(target)}\s*:[^\n]*$", MAKEFILE)
    if match is None:
        raise AssertionError(f"missing Makefile target {target}")
    lines = [match.group(0)]
    rest = MAKEFILE[match.end() + 1 :].splitlines()
    while lines[-1].rstrip().endswith("\\"):
        if not rest:
            raise AssertionError(f"unterminated Makefile rule {target}")
        lines.append(rest.pop(0))
    return "\n".join(lines)


def _prerequisites(target: str) -> list[str]:
    text = _rule_text(target).split(":", 1)[1]
    return text.replace("\\\n", " ").split()


def _recipe(target: str) -> list[str]:
    match = re.search(rf"(?m)^{re.escape(target)}\s*:[^\n]*$", MAKEFILE)
    if match is None:
        raise AssertionError(f"missing Makefile target {target}")
    lines: list[str] = []
    for line in MAKEFILE[match.end() + 1 :].splitlines():
        if not line.startswith("\t"):
            break
        lines.append(line)
    return lines


class Task16GateContractTest(unittest.TestCase):
    maxDiff = None

    def test_runtime_identity_gate_is_standalone_and_complete(self) -> None:
        self.assertIn(TARGET, _prerequisites(".PHONY"))
        self.assertNotIn(TARGET, _prerequisites("test"))
        self.assertEqual(
            _prerequisites(TARGET),
            [
                "tests/test_runtime",
                "tests/test_qualification_control",
                "ds4",
                "ds4-server",
                "ds4-agent",
                "ds4-bench",
                "ds4-eval",
            ],
        )
        self.assertEqual(
            _recipe(TARGET),
            [
                "\t@command -v uv >/dev/null 2>&1 || { \\",
                f'\t\techo "error: {TARGET} requires uv" >&2; \\',
                "\t\texit 127; \\",
                "\t}",
                f"\t@uv run --with-requirements {REQUIREMENTS} \\",
                "\t\tpython -c 'from jsonschema import Draft202012Validator; import rfc3339_validator, rfc8785' || { \\",
                "\t\techo \"error: unable to provision pinned compact-runtime requirements with uv\" >&2; \\",
                "\t\texit 1; \\",
                "\t}",
                "\tpython3 tests/test_task16_gate_contract.py -v",
                "\t./tests/test_runtime --case external-attribution",
                "\t./tests/test_qualification_control",
                "\tpython3 tests/test_qualification_control_contract.py -v",
                "\tpython3 tests/test_qualification_control_cli_contract.py -v",
                f"\tuv run --with-requirements {REQUIREMENTS} \\",
                "\t\tpython gguf-tools/quality-testing/test_compact_runtime_qualify.py -v",
                f"\tuv run --with-requirements {REQUIREMENTS} \\",
                "\t\tpython tests/test_version_json.py -v",
                f"\tuv run --with-requirements {REQUIREMENTS} \\",
                "\t\tpython tests/test_runtime_contract.py -v",
                f"\tDS4_RUNTIME_SERVER_URL= uv run --with-requirements {REQUIREMENTS} \\",
                "\t\tpython tests/test_runtime_endpoint_contract.py -v",
            ],
        )


if __name__ == "__main__":
    unittest.main()
