#!/usr/bin/env python3
"""Performance regression test for RTMDK Pipeline.

Compares current benchmark results against a baseline and fails
if latency or recall regresses beyond thresholds.

Usage:
    python scripts/check_regression.py --baseline baseline.json --current current.json
    python scripts/check_regression.py --run-benchmark --baseline baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_result(results: List[Dict], dataset: str, method: str) -> Dict:
    for r in results:
        if r.get("dataset") == dataset and r.get("method") == method:
            return r
    return {}


def check_regression(
    baseline: List[Dict],
    current: List[Dict],
    latency_threshold_pct: float = 20.0,
    recall_threshold_pct: float = -5.0,
) -> bool:
    """Returns True if no regression detected."""
    passed = True

    for curr in current:
        dataset = curr.get("dataset", "unknown")
        method = curr.get("method", "unknown")
        base = find_result(baseline, dataset, method)

        if not base:
            print(f"  [WARN] No baseline for {dataset}/{method}")
            continue

        # Check latency regression
        base_latency = base.get("p95_latency_ms", 0)
        curr_latency = curr.get("p95_latency_ms", 0)
        if base_latency > 0:
            latency_change = ((curr_latency - base_latency) / base_latency) * 100
            if latency_change > latency_threshold_pct:
                print(
                    f"  [FAIL] {dataset}/{method}: latency regressed "
                    f"{latency_change:.1f}% (p95: {base_latency:.1f}ms -> {curr_latency:.1f}ms)"
                )
                passed = False
            else:
                print(
                    f"  [OK] {dataset}/{method}: latency {latency_change:+.1f}% "
                    f"(p95: {base_latency:.1f}ms -> {curr_latency:.1f}ms)"
                )

        # Check recall regression
        base_recall = base.get("recall_at_k", 0)
        curr_recall = curr.get("recall_at_k", 0)
        if base_recall > 0:
            recall_change = ((curr_recall - base_recall) / base_recall) * 100
            if recall_change < recall_threshold_pct:
                print(
                    f"  [FAIL] {dataset}/{method}: recall regressed "
                    f"{recall_change:.1f}% ({base_recall:.3f} -> {curr_recall:.3f})"
                )
                passed = False
            else:
                print(
                    f"  [OK] {dataset}/{method}: recall {recall_change:+.1f}% "
                    f"({base_recall:.3f} -> {curr_recall:.3f})"
                )

    return passed


def run_benchmark() -> str:
    """Run production benchmark and return output path."""
    import subprocess

    output = "/tmp/rtmdk_regression_current.json"
    cmd = [
        sys.executable,
        "scripts/bench_pipeline_production.py",
        "--dataset",
        "datasets/qa_1000_en.json",
        "--output",
        output,
    ]
    print(f"Running benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Performance regression test")
    parser.add_argument("--baseline", "-b", type=str, required=True, help="Path to baseline JSON")
    parser.add_argument("--current", "-c", type=str, default="", help="Path to current results JSON")
    parser.add_argument("--run-benchmark", action="store_true", help="Run benchmark before comparison")
    parser.add_argument("--latency-threshold", type=float, default=20.0, help="Max acceptable latency regression %")
    parser.add_argument("--recall-threshold", type=float, default=-5.0, help="Max acceptable recall regression %")

    args = parser.parse_args()

    if args.run_benchmark:
        current_path = run_benchmark()
    elif args.current:
        current_path = args.current
    else:
        print("Error: --current or --run-benchmark required")
        sys.exit(1)

    print(f"\nLoading baseline: {args.baseline}")
    baseline = load_results(args.baseline)

    print(f"Loading current: {current_path}")
    current = load_results(current_path)

    print(
        f"\nChecking regression (latency threshold: +{args.latency_threshold}%, "
        f"recall threshold: {args.recall_threshold}%)"
    )
    print("=" * 60)

    passed = check_regression(
        baseline,
        current,
        latency_threshold_pct=args.latency_threshold,
        recall_threshold_pct=args.recall_threshold,
    )

    print("=" * 60)
    if passed:
        print("[PASS] No performance regression detected")
        sys.exit(0)
    else:
        print("[FAIL] Performance regression detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
