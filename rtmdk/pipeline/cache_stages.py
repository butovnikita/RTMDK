"""Pipeline stages for query caching.

Separates cache check/save from the monolithic retrieve_nodes() logic.
"""
from __future__ import annotations
from typing import Any, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage


class QueryCacheCheckStage(PipelineStage):
    """Check query cache before retrieval.

    If cache hit: populate ctx.results, set ctx.skip_remaining=True.
    If cache miss: proceed to next stages.
    """

    name = "query_cache_check"

    def __init__(self, field: Any, memory: Any):
        self.field = field
        self.memory = memory

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self.field.query_cache is None or ctx.embedding is None:
            return ctx

        query_latent = self.field._project(ctx.embedding)
        phase = self.memory._get_phase(ctx.session_id, ctx.embedding)
        top_k = ctx.top_k or self.field.cfg.top_k
        cache_key = self.field._query_cache_key(
            query_latent, phase, top_k, "text", ctx.session_id
        )
        cached = self.field.query_cache.get_raw(cache_key)
        if cached is not None:
            ctx.results = cached
            ctx.skip_remaining = True
        return ctx


class QueryCacheSaveStage(PipelineStage):
    """Save query results to cache after retrieval."""

    name = "query_cache_save"

    def __init__(self, field: Any, memory: Any):
        self.field = field
        self.memory = memory

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self.field.query_cache is None or ctx.embedding is None:
            return ctx

        query_latent = self.field._project(ctx.embedding)
        phase = self.memory._get_phase(ctx.session_id, ctx.embedding)
        top_k = ctx.top_k or self.field.cfg.top_k
        cache_key = self.field._query_cache_key(
            query_latent, phase, top_k, "text", ctx.session_id
        )
        self.field.query_cache.put_raw(cache_key, ctx.results)
        return ctx
