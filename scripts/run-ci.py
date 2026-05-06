#!/usr/bin/env python3
"""Local CI script — runs the same checks as GitHub Actions."""

import subprocess
import sys


def run(cmd, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {label}")
        sys.exit(result.returncode)
    print(f"OK: {label}")


def main():
    # Fast tests
    run("pytest tests/ -q -m 'not slow'", "Fast tests")

    # Coverage
    run("pytest tests/ --cov=rtmdk --cov-report=term-missing --cov-fail-under=50 -m 'not slow'", "Coverage")

    # Build
    run("python -m build", "Build wheel + sdist")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
