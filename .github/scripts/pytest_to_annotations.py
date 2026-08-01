#!/usr/bin/env python3
"""Convert pytest JUnit XML failures into GitHub Actions ::error annotations.

Makes test failures visible in the Checks API / PR "Files changed" view
without needing to download raw logs.

Usage:
    pytest ... --junitxml=pytest.xml
    python .github/scripts/pytest_to_annotations.py pytest.xml
"""

import sys
import xml.etree.ElementTree as ET


def main(path: str) -> int:
    tree = ET.parse(path)
    root = tree.getroot()
    count = 0
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        classname = case.get("classname", "")
        name = case.get("name", "unknown")
        # classname like tests.test_module.TestClass -> tests/test_module.py
        parts = classname.split(".")
        if parts and parts[-1][:1].isupper():
            parts = parts[:-1]  # drop TestClass component
        file_path = "/".join(parts) + ".py"
        message = (failure.get("message") or "test failed").splitlines()[0][:300]
        # Escape for workflow commands
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error file={file_path},title=FAILED {name}::{message}")
        count += 1
    print(f"pytest-annotations: {count} failure(s) annotated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pytest.xml"))
