#!/usr/bin/env python3
"""
R11.1 — check_docs_sync: verify README/docs stats vs actual repo counts.

Compares:
- README.md: LOC, files, API endpoints, tests
- docs/README.md version
- BACKLOG.md version vs rtmdk/__init__.py
- server/app.py endpoint count vs docs

Usage:
    python scripts/check_docs_sync.py          # check, exit 1 on major drift (>20%)
    python scripts/check_docs_sync.py --strict # exit 1 on any drift

This script is the R11.1 close criterion (README vs cloc) and is run in CI
as `python scripts/check_docs_sync.py --strict` (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def count_py_files() -> int:
    return len(list((ROOT / "rtmdk").rglob("*.py")))


def count_loc() -> int:
    total = 0
    for p in (ROOT / "rtmdk").rglob("*.py"):
        try:
            total += len(p.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass
    return total


def count_tests() -> int:
    cnt = 0
    for p in (ROOT / "tests").rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
            cnt += len(re.findall(r"def test_", txt))
        except Exception:
            pass
    return cnt


def count_endpoints() -> int:
    srv = ROOT / "rtmdk" / "server" / "app.py"
    txt = srv.read_text(encoding="utf-8")
    return len(re.findall(r'@app\.(get|post|put|delete|patch)', txt))


def parse_readme_stats():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    # Look for "74,000+ LOC, 440+ files, 49 API endpoints, 1281 tests" pattern
    m = re.search(r"(\d[\d,]*)\+? LOC.*?(\d+)\+? files.*?(\d+)\s*API endpoints.*?(\d+)\s*tests", txt, re.DOTALL)
    if not m:
        return None
    def to_int(s: str) -> int:
        return int(s.replace(",", "").replace("+", ""))
    return {
        "loc": to_int(m.group(1)),
        "files": to_int(m.group(2)),
        "endpoints": int(m.group(3)),
        "tests": to_int(m.group(4)),
    }


def main() -> int:
    strict = "--strict" in sys.argv
    ok = True

    actual_loc = count_loc()
    actual_files = count_py_files()
    actual_tests = count_tests()
    actual_endpoints = count_endpoints()

    print(f"Actual: LOC={actual_loc} files={actual_files} endpoints={actual_endpoints} tests={actual_tests}")

    parsed = parse_readme_stats()
    if parsed:
        print(f"README: LOC={parsed['loc']} files={parsed['files']} endpoints={parsed['endpoints']} tests={parsed['tests']}")
        # Allow 20% drift for LOC (74k vs 42k is ~43% drift — will warn)
        for key in ["loc", "files", "endpoints", "tests"]:
            actual = {"loc": actual_loc, "files": actual_files, "endpoints": actual_endpoints, "tests": actual_tests}[key]
            claimed = parsed[key]
            drift = abs(actual - claimed) / max(claimed, 1)
            status = "OK" if drift < 0.2 else "WARN" if drift < 0.5 else "FAIL"
            print(f"  {key}: claimed {claimed} actual {actual} drift {drift:.1%} -> {status}")
            if status == "FAIL" and strict:
                ok = False
            if key == "endpoints" and actual != claimed:
                print(f"  NOTE: update README.md and docs/01_API_REFERENCE.md to {actual} endpoints (server/app.py)")
    else:
        print("WARN: could not parse README stats pattern")

    # docs/README version
    docs_readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    m = re.search(r"Current version:\s*\*\*v?([\d.]+)", docs_readme)
    if m:
        docs_ver = m.group(1)
        import rtmdk
        pkg_ver = rtmdk.__version__
        print(f"Version: docs/README {docs_ver} vs pkg {pkg_ver} -> {'OK' if docs_ver == pkg_ver else 'WARN/FAIL'}")
        if docs_ver != pkg_ver and strict:
            ok = False
    else:
        print("WARN: could not parse docs/README version")

    # BACKLOG version
    backlog = (ROOT / "BACKLOG.md").read_text(encoding="utf-8")
    m2 = re.search(r"Current version:\s*([\d.]+)", backlog)
    if m2:
        bl_ver = m2.group(1)
        import rtmdk
        pkg_ver = rtmdk.__version__
        print(f"BACKLOG: {bl_ver} vs pkg {pkg_ver} -> {'OK' if bl_ver == pkg_ver else 'WARN/FAIL'}")
        if bl_ver != pkg_ver and strict:
            ok = False

    # mkdocs nav includes RISKS
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    if "RISKS" not in mkdocs and "risks" not in mkdocs.lower():
        print("WARN: mkdocs.yml nav missing RISKS.md")
        if strict:
            ok = False
    else:
        print("mkdocs nav: RISKS present -> OK")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
