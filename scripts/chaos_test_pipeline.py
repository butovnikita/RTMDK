#!/usr/bin/env python3
"""Chaos engineering test for RTMDK pipeline.

Purposefully injects failures into pipeline stages and verifies:
- Graceful degradation (fallbacks work)
- Circuit breaker opens under sustained failure
- System remains available (some results still returned)
- Metrics correctly report degraded stages

Usage:
    python scripts/chaos_test_pipeline.py --mode all
    python scripts/chaos_test_pipeline.py --mode stage_failure --stage rerank
    python scripts/chaos_test_pipeline.py --mode latency_spike --stage embed --latency 5000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "..")

from rtmdk.pipeline.base import PipelineContext, PipelineStage
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.circuit_breaker import CircuitBreaker


class ChaosStage(PipelineStage):
    """A stage that can be configured to fail or spike latency."""

    name = "chaos"

    def __init__(
        self,
        inner: PipelineStage,
        fail_rate: float = 0.0,
        latency_spike_ms: float = 0.0,
        fail_after_n: Optional[int] = None,
    ):
        self.inner = inner
        self._fail_rate = fail_rate
        self._latency_spike_ms = latency_spike_ms
        self._fail_after_n = fail_after_n
        self._call_count = 0
        self.name = inner.name

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self._call_count += 1
        if self._latency_spike_ms > 0:
            time.sleep(self._latency_spike_ms / 1000.0)
        if self._fail_after_n and self._call_count >= self._fail_after_n:
            raise RuntimeError(f"Injected failure after {self._call_count} calls")
        if random.random() < self._fail_rate:
            raise RuntimeError(f"Injected random failure (rate={self._fail_rate})")
        return self.inner.process(ctx)

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        return self.inner.fallback(ctx, exc)


def make_test_pipeline(
    failure_config: Optional[Dict[str, dict]] = None,
) -> PipelineExecutor:
    """Build a pipeline with chaos injection."""
    from rtmdk.pipeline.stages import EmbedStage, RouteStage, RetrieveStage, RerankStage, CalibrateStage, ExplainStage

    # Dummy embedder
    def dummy_embed(text: str) -> np.ndarray:
        return np.random.randn(384).astype(np.float32)

    # Dummy field
    class DummyField:
        def query(self, embedding, top_k=5, session_id=None, modality="text", query_text=None, **kwargs):
            return [(f"node_{i}", 0.9 - i * 0.05, None) for i in range(top_k)]

    stages = [
        ChaosStage(EmbedStage(dummy_embed), **failure_config.get("embed", {})),
        ChaosStage(RouteStage(), **failure_config.get("route", {})),
        ChaosStage(RetrieveStage(DummyField()), **failure_config.get("retrieve", {})),
        ChaosStage(RerankStage(), **failure_config.get("rerank", {})),
        ChaosStage(CalibrateStage(), **failure_config.get("calibrate", {})),
        ChaosStage(ExplainStage(), **failure_config.get("explain", {})),
    ]

    # Attach circuit breakers
    from rtmdk.pipeline.health import PipelineHealthMonitor

    monitor = PipelineHealthMonitor()
    for s in stages:
        s.circuit_breaker = monitor.get_breaker(s.name)

    return PipelineExecutor(stages)


def test_stage_failure_resilience(stage_name: str, fail_rate: float = 1.0) -> bool:
    """Verify pipeline survives a stage failure."""
    print(f"\n[TEST] Stage failure: {stage_name} @ {fail_rate*100:.0f}% fail rate")
    config = {stage_name: {"fail_rate": fail_rate}}
    pipeline = make_test_pipeline(config)

    ctx = pipeline.run("test query", top_k=5)

    # Should still return results (except embed/retrieve which are unrecoverable)
    if stage_name in ("embed", "retrieve"):
        print(f"  Expected: embed/retrieve failures are unrecoverable")
        return ctx.metrics[-1].error is not None if ctx.metrics else False

    success = len(ctx.results) > 0
    degraded = stage_name in ctx.degraded_stages
    print(f"  Results returned: {success} ({len(ctx.results)} items)")
    print(f"  Stage degraded: {degraded}")
    print(f"  Breaker state: {ctx.breaker_states.get(stage_name)}")
    return success and degraded


def test_circuit_breaker_opens(stage_name: str, n_calls: int = 10) -> bool:
    """Verify breaker opens after sustained failures."""
    print(f"\n[TEST] Circuit breaker: {stage_name} failing {n_calls}x")
    config = {stage_name: {"fail_after_n": 1}}
    pipeline = make_test_pipeline(config)

    open_count = 0
    for i in range(n_calls):
        ctx = pipeline.run(f"query {i}", top_k=5)
        state = ctx.breaker_states.get(stage_name)
        if state == "open":
            open_count += 1

    print(f"  Breaker open in {open_count}/{n_calls} calls")
    return open_count > n_calls // 2


def test_latency_spike_degradation(stage_name: str, spike_ms: float = 2000) -> bool:
    """Verify high latency triggers degradation alert."""
    print(f"\n[TEST] Latency spike: {stage_name} +{spike_ms}ms")
    config = {stage_name: {"latency_spike_ms": spike_ms}}
    pipeline = make_test_pipeline(config)

    ctx = pipeline.run("test query", top_k=5)
    total_latency = sum(m.latency_ms for m in ctx.metrics)

    from rtmdk.pipeline.health import PipelineHealthMonitor

    monitor = PipelineHealthMonitor()
    alerts = monitor.check_alerts(ctx, latency_threshold_ms=1000.0)

    print(f"  Total latency: {total_latency:.1f}ms")
    print(f"  Alerts: {len(alerts)}")
    for a in alerts:
        print(f"    - {a['type']}: {a['message']}")
    return len(alerts) > 0


def test_mixed_failure_scenario() -> bool:
    """Complex scenario: multiple stages failing at different rates."""
    print(f"\n[TEST] Mixed failure scenario")
    config = {
        "route": {"fail_rate": 0.5},
        "rerank": {"fail_rate": 0.3},
        "explain": {"fail_rate": 0.2},
    }
    pipeline = make_test_pipeline(config)

    ok_count = 0
    for i in range(20):
        ctx = pipeline.run(f"query {i}", top_k=5)
        if len(ctx.results) > 0:
            ok_count += 1

    success_rate = ok_count / 20
    print(f"  Success rate: {success_rate*100:.1f}% ({ok_count}/20)")
    return success_rate >= 0.8


def run_all_tests() -> Dict[str, bool]:
    """Run full chaos test suite."""
    results = {}

    # Test 1: Each non-critical stage failing
    for stage in ("route", "rerank", "calibrate", "explain"):
        results[f"stage_failure_{stage}"] = test_stage_failure_resilience(stage)

    # Test 2: Circuit breaker opens
    for stage in ("rerank", "explain"):
        results[f"breaker_opens_{stage}"] = test_circuit_breaker_opens(stage, n_calls=10)

    # Test 3: Latency spike
    results["latency_spike"] = test_latency_spike_degradation("embed", spike_ms=2000)

    # Test 4: Mixed scenario
    results["mixed_failure"] = test_mixed_failure_scenario()

    return results


def main():
    parser = argparse.ArgumentParser(description="Chaos engineering tests for RTMDK pipeline")
    parser.add_argument(
        "--mode", default="all", choices=["all", "stage_failure", "breaker", "latency", "mixed"], help="Test mode"
    )
    parser.add_argument("--stage", default="rerank", help="Target stage for single-stage tests")
    parser.add_argument("--latency", type=float, default=2000, help="Latency spike in ms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "all":
        results = run_all_tests()
    elif args.mode == "stage_failure":
        results = {f"stage_failure_{args.stage}": test_stage_failure_resilience(args.stage)}
    elif args.mode == "breaker":
        results = {f"breaker_{args.stage}": test_circuit_breaker_opens(args.stage)}
    elif args.mode == "latency":
        results = {"latency_spike": test_latency_spike_degradation(args.stage, args.latency)}
    elif args.mode == "mixed":
        results = {"mixed_failure": test_mixed_failure_scenario()}

    print(f"\n{'='*60}")
    print("CHAOS TEST SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")

    if passed < len(results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
