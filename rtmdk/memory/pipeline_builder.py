"""PipelineBuilder — extracts build_pipeline from core.py."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from rtmdk.pipeline import (
    PipelineExecutor,
    PlannedPipelineExecutor,
    EmbedStage,
    RouteStage,
    RetrieveStage,
    RerankStage,
    CalibrateStage,
    ExplainStage,
)

if TYPE_CHECKING:
    from rtmdk.memory.core import RTMDKMemory
    from rtmdk.pipeline import PipelineExecutor as PipelineExecutorType


class PipelineBuilder:
    """Encapsulates pipeline stage assembly for RTMDKMemory."""

    def __init__(self, memory: "RTMDKMemory") -> None:
        """Create a pipeline builder bound to an RTMDKMemory instance.

        Args:
            memory: The RTMDKMemory facade that owns the field and config.
        """
        self._mem = memory

    def build(self) -> "PipelineExecutorType":
        """Build an explicit stage-based pipeline for retrieval."""
        from rtmdk.production.cascade_router import AdaptiveCascadeRouter
        from rtmdk.pipeline.health import PipelineHealthMonitor
        from rtmdk.pipeline.cache_stages import (
            QueryCacheCheckStage, QueryCacheSaveStage)
        from rtmdk.pipeline.lock_stages import (
            DistributedLockStage, DistributedLockReleaseStage)

        stages: List = []
        monitor = PipelineHealthMonitor()

        cfg = self._mem.config
        breaker_enabled = getattr(cfg, "pipeline_breaker_enabled", True)
        breaker_thresholds = getattr(
            cfg, "pipeline_breaker_thresholds",
            {"embed": 5000.0, "route": 100.0, "retrieve": 500.0,
             "rerank": 1000.0, "calibrate": 200.0, "explain": 100.0})
        failure_thresh = getattr(
            cfg, "pipeline_breaker_failure_threshold", 5)
        latency_violation_thresh = getattr(
            cfg, "pipeline_breaker_latency_violation_threshold", 3)
        recovery_ms = getattr(
            cfg, "pipeline_breaker_recovery_timeout_ms", 30_000.0)
        half_open_calls = getattr(
            cfg, "pipeline_breaker_half_open_max_calls", 3)

        for stage_name, threshold in breaker_thresholds.items():
            monitor.set_threshold(stage_name, threshold)

        def _attach_breaker(stage):
            if breaker_enabled:
                stage.circuit_breaker = monitor.get_breaker(
                    stage.name,
                    failure_threshold=failure_thresh,
                    latency_violation_threshold=latency_violation_thresh,
                    recovery_timeout_ms=recovery_ms,
                )
                stage.circuit_breaker.half_open_max_calls = half_open_calls
            return stage

        # Stage -1: Distributed lock acquire (if configured)
        if self._mem._distributed_lock is not None:
            stages.append(_attach_breaker(
                DistributedLockStage(self._mem._distributed_lock)))

        # Stage 0: Query cache check
        if getattr(self._mem.field, "query_cache", None) is not None:
            stages.append(_attach_breaker(
                QueryCacheCheckStage(self._mem.field, self._mem)))

        # Stage 1: Embed
        stages.append(_attach_breaker(EmbedStage(self._mem.embedder)))

        # Stage 2: Route
        router = None
        if getattr(cfg, "cascade_enabled", False):
            router = AdaptiveCascadeRouter()
        stages.append(_attach_breaker(RouteStage(router)))

        # Stage 3: Retrieve
        stages.append(_attach_breaker(RetrieveStage(self._mem.field)))

        # Stage 4: Rerank
        stages.append(_attach_breaker(RerankStage(self._mem._sentence_reranker)))

        # Stage 5: Calibrate
        calibrator = getattr(self._mem.field, "conformal_calibrator", None)
        stages.append(_attach_breaker(CalibrateStage(calibrator)))

        # Stage 6: Explain
        stages.append(_attach_breaker(ExplainStage(self._mem._result_explainer)))

        # Stage 7: Query cache save
        if getattr(self._mem.field, "query_cache", None) is not None:
            stages.append(_attach_breaker(
                QueryCacheSaveStage(self._mem.field, self._mem)))

        # Stage 8: Distributed lock release
        if self._mem._distributed_lock is not None:
            stages.append(_attach_breaker(
                DistributedLockReleaseStage(self._mem._distributed_lock)))

        # Use planned executor if query planner is enabled
        if getattr(cfg, "pipeline_planner_enabled", False):
            from rtmdk.pipeline.planner import QueryPlanner
            planner = QueryPlanner()
            return PlannedPipelineExecutor(stages, planner=planner)
        return PipelineExecutor(stages)
