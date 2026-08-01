"""Prometheus-compatible metrics export for RTMDK pipeline."""

from __future__ import annotations
from typing import Any, Dict


def to_prometheus_format(ctx_dict: Dict[str, Any]) -> str:
    """Convert PipelineContext.to_dict() to Prometheus exposition format.

    Example output:
        # HELP rtmdk_query_latency_ms Total query latency in milliseconds
        # TYPE rtmdk_query_latency_ms gauge
        rtmdk_query_latency_ms{query="..."} 14.9

        # HELP rtmdk_stage_latency_ms Per-stage latency in milliseconds
        # TYPE rtmdk_stage_latency_ms gauge
        rtmdk_stage_latency_ms{stage="embed"} 12.5
        rtmdk_stage_latency_ms{stage="route"} 0.1
        ...
    """
    lines = []
    query_label = ctx_dict.get("query_text", "")[:50].replace('"', '\\"')

    # Total latency
    total = ctx_dict.get("total_latency_ms", 0.0)
    lines.append("# HELP rtmdk_query_latency_ms Total query latency")
    lines.append("# TYPE rtmdk_query_latency_ms gauge")
    lines.append(f'rtmdk_query_latency_ms{{query="{query_label}"}} {total}')

    # Results count
    rc = ctx_dict.get("results_count", 0)
    lines.append("# HELP rtmdk_query_results_count Number of results returned")
    lines.append("# TYPE rtmdk_query_results_count gauge")
    lines.append(f'rtmdk_query_results_count{{query="{query_label}"}} {rc}')

    # Per-stage latency
    lines.append("# HELP rtmdk_stage_latency_ms Per-stage latency")
    lines.append("# TYPE rtmdk_stage_latency_ms gauge")
    for stage in ctx_dict.get("stages", []):
        name = stage.get("stage", "unknown")
        latency = stage.get("latency_ms", 0.0)
        error = "1" if stage.get("error") else "0"
        degraded = "1" if stage.get("degraded") else "0"
        lines.append(f'rtmdk_stage_latency_ms{{stage="{name}",error="{error}",degraded="{degraded}"}} {latency}')

    return "\n".join(lines) + "\n"
