#!/usr/bin/env python3
"""Generate the immutable long-context seed prompt for Laguna oracles."""

from __future__ import annotations

import argparse
from pathlib import Path


LINE = b"The ring buffer stores each item at a position modulo its fixed capacity.\n"
REPETITIONS = 4096
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("benchmark-32768.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = LINE * REPETITIONS
    args.output.write_bytes(payload)
    print(f"wrote={args.output} bytes={len(payload)} lines={REPETITIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
