"""Batch pipeline execution for high-throughput retrieval."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

from rtmdk.pipeline.base import PipelineContext, PipelineStage


class BatchEmbedStage(PipelineStage):
    """Batch variant of EmbedStage: embed multiple texts at once."""

    name = "batch_embed"

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def process(self, ctx: PipelineContext) -> PipelineContext:
        # BatchEmbedStage expects ctx.query_text to be a list in batch mode
        # For single-query compatibility, delegate to standard EmbedStage
        if isinstance(ctx.query_text, list):
            if hasattr(self.embedder, "embed_batch"):
                ctx.embedding = self.embedder.embed_batch(ctx.query_text)
            else:
                ctx.embedding = [self.embedder(t) for t in ctx.query_text]
        elif ctx.embedding is None:
            ctx.embedding = self.embedder(ctx.query_text)
        return ctx


class BatchPipelineExecutor:
    """Execute pipeline on multiple queries with vectorized retrieval.

    Usage:
        executor = BatchPipelineExecutor(memory.build_pipeline().stages)
        outputs = executor.run_batch(queries, top_k=5)
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        session_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run pipeline on multiple queries sequentially.

        Future: vectorized retrieval stage for true batching.
        """
        results = []
        for i, query in enumerate(queries):
            sid = session_ids[i] if session_ids else None
            ctx = PipelineContext(
                query_text=query,
                top_k=top_k,
                session_id=sid,
            )
            for stage in self.stages:
                ctx = stage.run(ctx)
            results.append(ctx.to_dict())
        return results
