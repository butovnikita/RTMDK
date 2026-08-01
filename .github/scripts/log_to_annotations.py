#!/usr/bin/env python3
"""Emit the tail of a log file as GitHub Actions ::error annotations.

Usage:
    python .github/scripts/log_to_annotations.py <logfile> [max_lines]
"""

import sys


def main() -> int:
    path = sys.argv[1]
    max_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        print(f"log_to_annotations: cannot read {path}: {exc}", file=sys.stderr)
        return 0
    for line in lines[-max_lines:]:
        msg = line.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")[:500]
        print(f"::error ::{msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
