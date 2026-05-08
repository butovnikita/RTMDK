"""Benchmark GraphQL and WebSocket endpoints.

Usage:
    python scripts/bench_graphql_websocket.py --url http://localhost:8080

Requires: httpx, websockets (pip install httpx websockets)
"""
import argparse
import asyncio
import json
import sys
import time
from urllib.parse import urljoin, urlparse

try:
    import httpx
except ImportError:
    httpx = None

try:
    import websockets
except ImportError:
    websockets = None


async def bench_graphql_health(client, url: str, n: int = 100):
    payload = {"query": "{ health { status version memoryNodes } }"}
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        resp = await client.post(urljoin(url, "/graphql"), json=payload)
        await resp.aread()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


async def bench_graphql_nodes(client, url: str, n: int = 50):
    payload = {"query": "{ nodes(limit: 10, offset: 0) { id content salience } }"}
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        resp = await client.post(urljoin(url, "/graphql"), json=payload)
        await resp.aread()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


async def bench_websocket_ping(ws_url: str, n: int = 100):
    latencies = []
    async with websockets.connect(ws_url) as ws:
        for _ in range(n):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"action": "ping"}))
            await ws.recv()
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


async def bench_websocket_query(ws_url: str, n: int = 50):
    latencies = []
    async with websockets.connect(ws_url) as ws:
        for _ in range(n):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"action": "query", "query": "hello", "top_k": 5}))
            await ws.recv()
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def fmt(latencies: list) -> str:
    if not latencies:
        return "no data"
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_idx, len(latencies) - 1)]
    return (
        f"min={min(latencies):.2f}ms, "
        f"median={p50:.2f}ms, "
        f"p95={p95:.2f}ms, "
        f"max={max(latencies):.2f}ms"
    )


async def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphQL + WebSocket")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--graphql-n", type=int, default=100)
    parser.add_argument("--ws-n", type=int, default=100)
    args = parser.parse_args()

    if httpx is None:
        print("ERROR: httpx not installed. Run: pip install httpx")
        return 1
    if websockets is None:
        print("ERROR: websockets not installed. Run: pip install websockets")
        return 1

    parsed = urlparse(args.url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws/memory"

    print(f"Benchmarking {args.url}")
    print(f"WebSocket: {ws_url}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # GraphQL health
        lat = await bench_graphql_health(client, args.url, args.graphql_n)
        print(f"GraphQL health  ({args.graphql_n} req): {fmt(lat)}")

        # GraphQL nodes
        lat = await bench_graphql_nodes(client, args.url, args.graphql_n // 2)
        print(f"GraphQL nodes   ({args.graphql_n // 2} req): {fmt(lat)}")

    # WebSocket ping
    lat = await bench_websocket_ping(ws_url, args.ws_n)
    print(f"WebSocket ping  ({args.ws_n} msg): {fmt(lat)}")

    # WebSocket query
    lat = await bench_websocket_query(ws_url, args.ws_n // 2)
    print(f"WebSocket query ({args.ws_n // 2} msg): {fmt(lat)}")

    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
