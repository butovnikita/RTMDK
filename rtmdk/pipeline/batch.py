"""Batch pipeline execution for high-throughput retrieval."""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

import numpy as np

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


class BatchRetrieveStage(PipelineStage):
    """Batch variant of RetrieveStage using field.query_batch()."""

    name = "batch_retrieve"

    def __init__(self, field: Any, modality: str = "text"):
        self.field = field
        self.modality = modality

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.embedding is None:
            raise RuntimeError("BatchRetrieveStage requires embeddings.")
        if isinstance(ctx.embedding, list):
            embeddings = np.array(ctx.embedding)
        else:
            embeddings = ctx.embedding
        ctx.results = self.field.query_batch(
            embeddings,
            top_k=ctx.top_k,
            session_id=ctx.session_id,
            modality=self.modality,
        )
        return ctx


class BatchPipelineExecutor:
    """Execute pipeline on multiple queries with true vectorized retrieval.

    Usage:
        executor = BatchPipelineExecutor(memory.build_pipeline().stages, memory.field)
        outputs = executor.run_batch(queries, top_k=5)

    Backward-compatible: if field is omitted, falls back to sequential execution.
    """

    def __init__(self, stages: List[PipelineStage], field: Optional[Any] = None):
        self.stages = stages
        self.field = field

    def run_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        session_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run pipeline on multiple queries with batch embed + batch retrieve.

        Stages:
          1. batch_embed — all queries embedded at once
          2. route — per-query routing
          3. batch_retrieve — vectorized resonance across all queries
          4+ rerank/calibrate/explain — per-query (if present)
        """
        if not queries:
            return []

        # Backward-compatible fallback: no field → sequential execution
        if self.field is None:
            results: List[Dict[str, Any]] = []
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

        t0 = time.perf_counter()
        results = []

        # Stage 1: Batch embed — bypass stage circuit breaker for batch safety
        embedder = None
        for s in self.stages:
            if s.name in ("embed", "batch_embed"):
                embedder = getattr(s, "embedder", None)
                break

        if embedder is None:
            raise RuntimeError("No embed stage found in pipeline")

        te0 = time.perf_counter()
        if hasattr(embedder, "embed_batch"):
            embeddings = embedder.embed_batch(queries)
        else:
            embeddings = np.array([embedder(q) for q in queries])
        embed_time_ms = (time.perf_counter() - te0) * 1000

        # Stage 2: Route (per-query, usually fast)
        routes = []
        route_stage = None
        for s in self.stages:
            if s.name == "route":
                route_stage = s
                break

        if route_stage:
            for q in queries:
                ctx = PipelineContext(query_text=q, top_k=top_k)
                ctx = route_stage.run(ctx)
                routes.append(ctx.route or "standard")
        else:
            routes = ["standard"] * len(queries)

        # Stage 3: Batch retrieve
        tr0 = time.perf_counter()
        batch_results = self.field.query_batch(
            embeddings,
            top_k=top_k,
            session_id=session_ids[0] if session_ids else None,
        )
        retrieve_time_ms = (time.perf_counter() - tr0) * 1000

        # Build per-query outputs
        for i, query in enumerate(queries):
            per_query_results = batch_results[i] if i < len(batch_results) else []
            formatted = []
            for nid, score, node in per_query_results:
                content = (
                    node.content.get("content", node.content) if isinstance(node.content, dict) else str(node.content)
                )
                formatted.append(
                    {
                        "id": nid,
                        "content": content,
                        "score": round(float(score), 4),
                    }
                )

            total_time_ms = (time.perf_counter() - t0) * 1000
            results.append(
                {
                    "query": query,
                    "results": formatted,
                    "route": routes[i],
                    "total": len(formatted),
                    "metrics": {
                        "stages": [
                            {"stage": "embed", "latency_ms": round(embed_time_ms / len(queries), 2)},
                            {"stage": "retrieve", "latency_ms": round(retrieve_time_ms / len(queries), 2)},
                            {"stage": "total", "latency_ms": round(total_time_ms / len(queries), 2)},
                        ],
                    },
                }
            )

        return results
