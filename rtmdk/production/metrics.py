"""rtmdk/production/metrics.py — Prometheus metrics for RTMDK server.

Exposes:
  - rtmdk_query_latency_seconds (Histogram)
  - rtmdk_nodes_total (Gauge)
  - rtmdk_requests_total (Counter)
  - rtmdk_replication_lag (Gauge)
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

QUERY_LATENCY = Histogram(
    "rtmdk_query_latency_seconds",
    "Latency of memory queries",
    ["endpoint", "cache"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

NODES_TOTAL = Gauge(
    "rtmdk_nodes_total",
    "Total number of memory nodes",
    ["tier"],
)

REQUESTS_TOTAL = Counter(
    "rtmdk_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REPLICATION_LAG = Gauge(
    "rtmdk_replication_lag",
    "Replication lag in entries behind peers",
    ["peer"],
)

EMBEDDING_LATENCY = Histogram(
    "rtmdk_embedding_latency_seconds",
    "Latency of embedding generation",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ---------------------------------------------------------------------------
# Middleware helpers
# ---------------------------------------------------------------------------


class MetricsMiddleware:
    """Starlette/FastAPI middleware that records request counts and latencies."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "unknown")
        start_time = __import__("time").time()
        status_code = 500

        async def wrapped_send(message: Any) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            REQUESTS_TOTAL.labels(
                method=method,
                endpoint=path,
                status=str(status_code),
            ).inc()
            # Record query latency for memory endpoints
            if path.startswith("/v1/memory/query"):
                latency = __import__("time").time() - start_time
                QUERY_LATENCY.labels(endpoint="query", cache="unknown").observe(latency)


def update_node_count(count: int, tier: str = "hot") -> None:
    """Update the nodes_total gauge."""
    NODES_TOTAL.labels(tier=tier).set(count)


def get_metrics_payload() -> bytes:
    """Return Prometheus exposition format."""
    return generate_latest()
