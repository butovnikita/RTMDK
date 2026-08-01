#!/usr/bin/env python3
"""mypy ratchet: fail if the error count grows beyond the committed baseline.

Full cleanup of the historical debt (~1000 errors) is a long tail; this gate
guarantees the debt only shrinks, never grows.

Baseline lives in .github/mypy-baseline.txt (single integer). When you reduce
errors, regenerate it:
    python scripts/check_mypy.py --write-baseline

Note: the count can shift when the mypy version changes (requirements-dev.txt
uses a >= range). If a CI failure coincides with a mypy release, verify the
diff is from the tool, not the code, then regenerate the baseline.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = ROOT / ".github" / "mypy-baseline.txt"


def run_mypy() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "rtmdk", "--config-file", "mypy.ini"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"Found (\d+) errors? in (\d+) files?", output)
    if not match:
        print(output)
        print("check_mypy: could not parse mypy summary line", file=sys.stderr)
        sys.exit(2)
    print(output.strip().splitlines()[-1])
    return int(match.group(1))


def main() -> int:
    count = run_mypy()
    if "--write-baseline" in sys.argv:
        BASELINE_FILE.write_text(f"{count}\n", encoding="utf-8")
        print(f"baseline updated: {count}")
        return 0

    if not BASELINE_FILE.exists():
        print(f"check_mypy: baseline file missing: {BASELINE_FILE}", file=sys.stderr)
        return 2
    baseline = int(BASELINE_FILE.read_text(encoding="utf-8").strip())

    if count > baseline:
        print(f"check_mypy: FAIL — {count} errors > baseline {baseline} (+{count - baseline})")
        return 1
    if count < baseline:
        print(f"check_mypy: OK — {count} errors < baseline {baseline} (regenerate baseline!)")
    else:
        print(f"check_mypy: OK — {count} errors == baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
