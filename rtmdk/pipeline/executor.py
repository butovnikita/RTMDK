"""Pipeline executor: compose stages and run them in order."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage


class PipelineExecutor:
    """Run a sequence of PipelineStages with full observability.

    Example:
        pipeline = PipelineExecutor([EmbedStage(embedder), RetrieveStage(field)])
        ctx = pipeline.run("What is the capital of France?", top_k=5)
        print(ctx.results)
        print(ctx.to_dict()["stages"])  # per-stage latency breakdown
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

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
        return ctx

    def run_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[PipelineContext]:
        """Run pipeline on multiple queries sequentially."""
        return [
            self.run(q, top_k=top_k, session_id=session_id)
            for q in queries
        ]

    def get_metrics(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Return aggregated metrics for a completed run."""
        return ctx.to_dict()
