#!/usr/bin/env python3
"""Load test script for RTMDK pipeline endpoints.

Usage:
    python scripts/load_test_pipeline.py --endpoint query_pipeline --rps 10 --duration 30
    python scripts/load_test_pipeline.py --endpoint stream --rps 5 --duration 60
    python scripts/load_test_pipeline.py --endpoint health --rps 50 --duration 10

Outputs:
    - Requests per second (RPS)
    - Latency percentiles (p50, p95, p99)
    - Error rate
    - Pipeline stage breakdown (if available)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)


@dataclass
class LoadTestResult:
    endpoint: str
    total_requests: int
    successful: int
    failed: int
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def rps(self) -> float:
        duration = sum(self.latencies) if self.latencies else 1
        return self.total_requests / max(duration, 0.001)

    @property
    def error_rate(self) -> float:
        return self.failed / max(self.total_requests, 1)

    def latency_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"Load Test Results: {self.endpoint}")
        print(f"{'=' * 60}")
        print(f"  Total requests:   {self.total_requests}")
        print(f"  Successful:       {self.successful}")
        print(f"  Failed:           {self.failed}")
        print(f"  Error rate:       {self.error_rate * 100:.2f}%")
        if self.latencies:
            print(f"  RPS (observed):   {len(self.latencies) / max(sum(self.latencies), 0.001):.2f}")
            print(f"  Latency p50:      {self.latency_percentile(0.50):.2f} ms")
            print(f"  Latency p95:      {self.latency_percentile(0.95):.2f} ms")
            print(f"  Latency p99:      {self.latency_percentile(0.99):.2f} ms")
            print(f"  Latency min:      {min(self.latencies):.2f} ms")
            print(f"  Latency max:      {max(self.latencies):.2f} ms")
        if self.errors:
            print(f"  Sample errors:")
            for err in self.errors[:5]:
                print(f"    - {err}")


async def _run_query_pipeline(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    top_k: int,
) -> tuple[bool, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/v1/memory/query_pipeline",
            json={"query": query, "top_k": top_k},
            timeout=30.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return True, latency, None
        return False, latency, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return False, latency, str(exc)


async def _run_pipeline_stream(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    top_k: int,
) -> tuple[bool, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        resp = await client.get(
            f"{base_url}/v1/memory/pipeline/stream",
            params={"query": query, "top_k": top_k},
            timeout=30.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return True, latency, None
        return False, latency, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return False, latency, str(exc)


async def _run_pipeline_health(
    client: httpx.AsyncClient,
    base_url: str,
) -> tuple[bool, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        resp = await client.get(
            f"{base_url}/v1/memory/pipeline/health",
            timeout=10.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return True, latency, None
        return False, latency, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return False, latency, str(exc)


async def _worker(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    queries: List[str],
    top_k: int,
    result: LoadTestResult,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event,
):
    query_idx = 0
    while not stop_event.is_set():
        async with semaphore:
            query = queries[query_idx % len(queries)]
            query_idx += 1

            if endpoint == "query_pipeline":
                ok, lat, err = await _run_query_pipeline(client, base_url, query, top_k)
            elif endpoint == "stream":
                ok, lat, err = await _run_pipeline_stream(client, base_url, query, top_k)
            elif endpoint == "health":
                ok, lat, err = await _run_pipeline_health(client, base_url)
            else:
                ok, lat, err = False, 0.0, f"Unknown endpoint: {endpoint}"

            result.total_requests += 1
            if ok:
                result.successful += 1
                result.latencies.append(lat)
            else:
                result.failed += 1
                if err:
                    result.errors.append(err)


async def run_load_test(
    base_url: str,
    endpoint: str,
    target_rps: int,
    duration_sec: int,
    queries: List[str],
    top_k: int,
) -> LoadTestResult:
    result = LoadTestResult(endpoint=endpoint, total_requests=0, successful=0, failed=0)
    semaphore = asyncio.Semaphore(target_rps)
    stop_event = asyncio.Event()

    async with httpx.AsyncClient() as client:
        # Quick health check
        try:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"WARNING: Server health check returned {resp.status_code}")
        except Exception as exc:
            print(f"WARNING: Could not reach server: {exc}")

        # Spawn workers
        workers = [
            asyncio.create_task(_worker(client, base_url, endpoint, queries, top_k, result, semaphore, stop_event))
            for _ in range(target_rps)
        ]

        print(f"Running load test: {endpoint} @ {target_rps} RPS for {duration_sec}s...")
        await asyncio.sleep(duration_sec)
        stop_event.set()

        for w in workers:
            w.cancel()

    return result


def main():
    parser = argparse.ArgumentParser(description="Load test RTMDK pipeline endpoints")
    parser.add_argument(
        "--endpoint",
        "-e",
        default="query_pipeline",
        choices=["query_pipeline", "stream", "health"],
        help="Endpoint to test",
    )
    parser.add_argument("--rps", "-r", type=int, default=10, help="Target requests per second (concurrency)")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--url", "-u", default="http://localhost:8080", help="Base URL of RTMDK server")
    parser.add_argument(
        "--queries", "-q", default="hello,world,test,query,search", help="Comma-separated query strings"
    )
    parser.add_argument("--top-k", "-k", type=int, default=5, help="top_k parameter for queries")

    args = parser.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    if not queries:
        queries = ["hello", "world", "test"]

    result = asyncio.run(
        run_load_test(
            base_url=args.url,
            endpoint=args.endpoint,
            target_rps=args.rps,
            duration_sec=args.duration,
            queries=queries,
            top_k=args.top_k,
        )
    )

    result.print_summary()

    if result.error_rate > 0.05:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
