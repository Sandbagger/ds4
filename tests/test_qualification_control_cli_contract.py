#!/usr/bin/env python3
"""Contract for the hidden qualification-control descriptor frontend option."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = {
    "ds4": ("ds4_cli.c", "c.engine", "parse_nonnegative_int"),
    "ds4-server": ("ds4_server.c", "c.engine", "parse_nonneg_int_arg"),
    "ds4-agent": ("ds4_agent.c", "c.engine", "parse_nonnegative_int"),
    "ds4-bench": ("ds4_bench.c", "c", "parse_nonnegative_int"),
    "ds4-eval": ("ds4_eval.c", "c", "parse_nonnegative_int_arg"),
}
OPTION = "--qualification-control-fd"


def _function(source: str, name: str) -> str:
    match = re.search(
        rf"(?m)^static\s+[^\n]+\b{re.escape(name)}\s*\([^;]*\)\s*\{{",
        source,
    )
    if match is None:
        match = re.search(
            rf"(?m)^int\s+{re.escape(name)}\s*\([^;]*\)\s*\{{",
            source,
        )
    if match is None:
        raise AssertionError(f"missing function {name}")

    depth = 0
    for index in range(match.end() - 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1]
    raise AssertionError(f"unterminated function {name}")


def _option_branch(parse_body: str) -> str:
    marker = f'!strcmp(arg, "{OPTION}")'
    start = parse_body.find(marker)
    if start < 0:
        raise AssertionError(f"parse_options does not recognize {OPTION}")
    end = parse_body.find("} else if", start)
    if end < 0:
        end = parse_body.find("} else {", start)
    if end < 0:
        raise AssertionError(f"cannot isolate {OPTION} parser branch")
    return parse_body[start:end]


class QualificationControlCliContractTest(unittest.TestCase):
    maxDiff = None

    def test_every_frontend_parses_one_nonnegative_control_fd(self) -> None:
        for frontend, (path, config, parser) in FRONTENDS.items():
            with self.subTest(frontend=frontend):
                source = (ROOT / path).read_text(encoding="utf-8")
                branch = _option_branch(_function(source, "parse_options"))
                fd = f"{config}.qualification_control_fd"
                fd_set = f"{config}.qualification_control_fd_set"
                self.assertIn(f"if ({fd_set})", branch)
                self.assertIn("may only be specified once", branch)
                self.assertRegex(
                    branch,
                    rf"{re.escape(fd)}\s*=\s*{parser}\s*\(\s*"
                    rf"need_arg\s*\(\s*&i\s*,\s*argc\s*,\s*argv\s*,\s*arg\s*\)\s*,\s*arg\s*\)\s*;",
                )
                self.assertRegex(
                    branch,
                    rf"{re.escape(fd_set)}\s*=\s*true\s*;",
                )

    def test_split_frontend_configs_forward_both_fields_to_engine(self) -> None:
        for frontend, path in (
            ("ds4-bench", "ds4_bench.c"),
            ("ds4-eval", "ds4_eval.c"),
        ):
            with self.subTest(frontend=frontend):
                source = (ROOT / path).read_text(encoding="utf-8")
                self.assertRegex(
                    source,
                    r"\.qualification_control_fd\s*=\s*"
                    r"cfg\.qualification_control_fd\s*,",
                )
                self.assertRegex(
                    source,
                    r"\.qualification_control_fd_set\s*=\s*"
                    r"cfg\.qualification_control_fd_set\s*,",
                )

    def test_version_remains_terminal_before_qualification_parsing(self) -> None:
        for frontend, (path, _, _) in FRONTENDS.items():
            with self.subTest(frontend=frontend):
                source = (ROOT / path).read_text(encoding="utf-8")
                main = _function(source, "main")
                self.assertLess(main.index('"--version-json"'),
                                main.index("ds4_qualification_args_preflight"))
                self.assertLess(main.index("ds4_qualification_args_preflight"),
                                main.index("parse_options(argc, argv)"))

    def test_control_descriptor_option_stays_hidden_from_help(self) -> None:
        help_source = (ROOT / "ds4_help.c").read_text(encoding="utf-8")
        self.assertNotIn(OPTION, help_source)


if __name__ == "__main__":
    unittest.main()
