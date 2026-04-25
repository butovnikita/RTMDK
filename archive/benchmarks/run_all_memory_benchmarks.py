"""
run_all_memory_benchmarks.py — Master Runner for RTMDK Long-Term Memory Benchmarks.

Runs all specialized memory benchmarks and produces a consolidated report.

Benchmarks:
1. Forgetting Curve — retention over time
2. Interference — catastrophic forgetting & topic bleed
3. Capacity — scaling at N=500/1000/2000/5000
4. Consolidation — quality of merging
5. Cross-Session — recall across session boundaries

Usage:
    python run_all_memory_benchmarks.py [--report memory_benchmark_report.json]
    python run_all_memory_benchmarks.py --select forgetting,capacity
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


ALL_BENCHMARKS = {
    "forgetting": ("benchmark_forgetting", ["--n_facts", "200", "--max_steps", "500"]),
    "interference": ("benchmark_interference", ["--n_facts_per_topic", "50"]),
    "capacity": ("benchmark_capacity", []),
    "consolidation": ("benchmark_consolidation", ["--n_facts", "200", "--n_steps", "50"]),
    "cross_session": ("benchmark_cross_session", ["--n_sessions", "5", "--facts_per_session", "20"]),
}


def run_benchmark(name: str, args: list) -> dict:
    """Run a single benchmark by importing and executing its main function."""
    module_name = ALL_BENCHMARKS[name][0]
    try:
        module = __import__(module_name)
        # Override sys.argv for argparse
        orig_argv = sys.argv
        sys.argv = [module_name] + args
        module.main()
        sys.argv = orig_argv
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def load_report(filename: str) -> dict:
    """Load a benchmark report JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def print_summary(report: dict):
    """Print a consolidated summary of all benchmark results."""
    print("\n" + "=" * 70)
    print("  RTMDK LONG-TERM MEMORY BENCHMARK — CONSOLIDATED SUMMARY")
    print("=" * 70)
    print(f"  Date: {report.get('date', 'N/A')}")
    print(f"  Duration: {report.get('duration_seconds', 0):.1f}s")
    print()

    # Forgetting
    fg = report.get("forgetting", {})
    if fg and "initial_recall" in fg:
        print(f"  FORGETTING CURVE:")
        print(f"    Initial recall:        {fg.get('initial_recall', 0):.2%}")
        print(f"    Half-life (steps):     {fg.get('half_life_steps', 'N/A')}")
        print(f"    Retention @ 500 steps: {fg.get('retention_at_500_steps', 0):.2%}" if fg.get('retention_at_500_steps') is not None else "    Retention @ 500 steps: N/A")

    # Interference
    inf = report.get("interference", {})
    if inf and "avg_interference_ratio" in inf:
        print(f"\n  INTERFERENCE:")
        print(f"    Avg interference:      {inf.get('avg_interference_ratio', 0):.2%}")
        print(f"    Catastrophic forget:   {'YES' if inf.get('catastrophic_forgetting') else 'No'}")

    # Capacity
    cap = report.get("capacity", {})
    if isinstance(cap, list):
        print(f"\n  CAPACITY SCALING:")
        print(f"    {'Nodes':>8} {'Recall':>10} {'Latency(ms)':>14}")
        for r in cap:
            print(f"    {r.get('actual_nodes', 0):>8} {r.get('recall_rate', 0):>10.2%} {r.get('avg_query_latency_ms', 0):>14.2f}")

    # Consolidation
    cons = report.get("consolidation", {})
    if cons and "consolidation_gain" in cons:
        print(f"\n  CONSOLIDATION:")
        print(f"    Recall before:         {cons.get('recall_before_consolidation', 0):.2%}")
        print(f"    Recall after:          {cons.get('recall_after_consolidation', 0):.2%}")
        print(f"    Consolidation gain:    {cons.get('consolidation_gain', 0):+.2%}")
        print(f"    Compression ratio:     {cons.get('compression_ratio', 0):.2f}x")

    # Cross-Session
    xs = report.get("cross_session", {})
    if xs and "avg_same_session_recall" in xs:
        print(f"\n  CROSS-SESSION:")
        print(f"    Same-session recall:   {xs.get('avg_same_session_recall', 0):.2%}")
        print(f"    Cross-session recall:  {xs.get('avg_cross_session_recall', 0):.2%}")
        print(f"    Isolation score:       {xs.get('session_isolation_score', 0):.2%}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RTMDK All Memory Benchmarks Runner")
    parser.add_argument("--report", type=str, default="memory_benchmark_report.json")
    parser.add_argument("--select", type=str, default=None,
                        help="Comma-separated list: forgetting,interference,capacity,consolidation,cross_session")
    args = parser.parse_args()

    # Determine which benchmarks to run
    if args.select:
        selected = [s.strip() for s in args.select.split(",")]
    else:
        selected = list(ALL_BENCHMARKS.keys())

    print("=" * 60)
    print("  RTMDK Long-Term Memory Benchmarks")
    print("=" * 60)
    print(f"  Benchmarks: {', '.join(selected)}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "benchmarks_run": selected,
    }

    t_start = time.time()

    for name in selected:
        if name not in ALL_BENCHMARKS:
            print(f"\n  Skipping unknown benchmark: {name}")
            continue

        report_file = f"{name}_report.json"
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        bench_args = list(ALL_BENCHMARKS[name][1]) + ["--report", report_file]
        result = run_benchmark(name, bench_args)

        # Load the benchmark's report
        bench_report = load_report(report_file)
        report[name] = bench_report

        if result.get("status") == "error":
            report[f"{name}_error"] = result.get("error", "unknown")
            print(f"  ERROR: {result.get('error', 'unknown')}")

    report["duration_seconds"] = round(time.time() - t_start, 1)

    # Print summary
    print_summary(report)

    # Save consolidated report
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Consolidated report saved to {args.report}")


if __name__ == "__main__":
    main()
