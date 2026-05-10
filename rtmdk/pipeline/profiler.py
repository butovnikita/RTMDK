"""Memory profiling for pipeline stages.

Tracks peak RAM usage per stage to identify memory bottlenecks.
"""
from __future__ import annotations
import tracemalloc
from typing import Any, Dict, List, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage


class PipelineMemoryProfiler:
    """Profile peak memory usage per pipeline stage.

    Usage:
        profiler = PipelineMemoryProfiler()
        ctx = stage.run(ctx)  # profiler attached to stage
        print(profiler.get_summary())
    """

    def __init__(self):
        self._snapshots: Dict[str, List[int]] = {}
        self._enabled = False

    def start(self) -> None:
        """Start global tracemalloc tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._enabled = True

    def stop(self) -> None:
        """Stop global tracemalloc tracking."""
        if self._enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._enabled = False

    def profile_stage(self, stage: PipelineStage, ctx: PipelineContext) -> PipelineContext:
        """Run a stage and record its peak memory usage."""
        if not tracemalloc.is_tracing():
            self.start()

        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()

        ctx = stage.run(ctx)

        after, peak = tracemalloc.get_traced_memory()
        peak_bytes = peak - before
        stage_name = stage.name

        self._snapshots.setdefault(stage_name, []).append(max(0, peak_bytes))

        # Attach to context for immediate inspection
        if "memory_usage" not in ctx.__dict__:
            ctx.memory_usage = {}  # type: ignore
        ctx.memory_usage[stage_name] = {  # type: ignore
            "peak_bytes": peak_bytes,
            "current_bytes": after - before,
        }
        return ctx

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregate memory stats per stage."""
        summary = {}
        for name, values in self._snapshots.items():
            if not values:
                continue
            summary[name] = {
                "runs": len(values),
                "peak_bytes_mean": sum(values) // max(len(values), 1),
                "peak_bytes_max": max(values),
                "peak_bytes_min": min(values),
                "peak_mb_mean": round(sum(values) / max(len(values), 1) / (1024 * 1024), 3),
                "peak_mb_max": round(max(values) / (1024 * 1024), 3),
            }
        return summary

    def reset(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
