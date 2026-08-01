"""Pipeline executor: compose stages and run them in order."""

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage
from rtmdk.pipeline.profiler import PipelineMemoryProfiler


class PipelineExecutor:
    """Run a sequence of PipelineStages with full observability.

    Example:
        pipeline = PipelineExecutor([EmbedStage(embedder), RetrieveStage(field)])
        ctx = pipeline.run("What is the capital of France?", top_k=5)
        print(ctx.results)
        print(ctx.to_dict()["stages"])  # per-stage latency breakdown
    """

    def __init__(self, stages: List[PipelineStage], webhook_manager: Optional[Any] = None):
        self.stages = stages
        self.webhook_manager = webhook_manager

    def _dispatch_stage_event(self, ctx: PipelineContext, stage: PipelineStage) -> None:
        """Dispatch webhook events for degraded stages or breaker state changes."""
        if self.webhook_manager is None:
            return
        latest = ctx.metrics[-1] if ctx.metrics else None
        if latest and latest.degraded:
            self.webhook_manager.dispatch(
                "pipeline_stage_degraded",
                {
                    "stage": stage.name,
                    "query": ctx.query_text[:100],
                    "session_id": ctx.session_id,
                    "error": latest.error,
                    "latency_ms": latest.latency_ms,
                    "breaker_state": ctx.breaker_states.get(stage.name),
                },
            )

    def run(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> PipelineContext:
        ctx = PipelineContext(
            query_text=query_text,
            top_k=top_k,
            session_id=session_id,
            embedding=embedding,
        )
        for stage in self.stages:
            ctx = stage.run(ctx)
            self._dispatch_stage_event(ctx, stage)
            if ctx.skip_remaining:
                break
        return ctx

    def run_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[PipelineContext]:
        """Run pipeline on multiple queries sequentially."""
        return [self.run(q, top_k=top_k, session_id=session_id) for q in queries]

    async def run_async(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> PipelineContext:
        """Async version of run() — executes sync stages in thread pool.

        Useful for FastAPI endpoints to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query_text, top_k, session_id, embedding)

    def run_with_profiler(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> tuple[PipelineContext, PipelineMemoryProfiler]:
        """Run pipeline with memory profiling per stage.

        Returns:
            (ctx, profiler) — context and profiler with memory summary
        """
        profiler = PipelineMemoryProfiler()
        profiler.start()
        ctx = PipelineContext(
            query_text=query_text,
            top_k=top_k,
            session_id=session_id,
            embedding=embedding,
        )
        for stage in self.stages:
            ctx = profiler.profile_stage(stage, ctx)
            if ctx.skip_remaining:
                break
        profiler.stop()
        return ctx, profiler

    async def run_batch_async(
        self,
        queries: List[str],
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[PipelineContext]:
        """Async batch execution — runs queries concurrently."""
        tasks = [self.run_async(q, top_k=top_k, session_id=session_id) for q in queries]
        return await asyncio.gather(*tasks)

    def get_metrics(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Return aggregated metrics for a completed run."""
        return ctx.to_dict()
