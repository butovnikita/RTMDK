"""Explicit retrieval pipeline for RTMDK.

Replaces the monolithic retrieve_nodes() with a composable stage-based pipeline.
Each stage has a uniform interface and can be independently measured, swapped,
or disabled.

Stages (in order):
    1. EmbedStage         — query text → embedding
    2. RouteStage         — CascadeRouter (factual/exploratory/deep)
    3. RetrieveStage      — resonance / HNSW / BM25 hybrid
    4. RerankStage        — sentence-level + cross-encoder reranking
    5. CalibrateStage     — conformal prediction filtering
    6. ExplainStage       — ResultExplainer annotations

Usage:
    from rtmdk.pipeline import PipelineExecutor, EmbedStage, RetrieveStage
    pipeline = PipelineExecutor([
        EmbedStage(memory.embedder),
        RouteStage(memory.field.cfg),
        RetrieveStage(memory.field),
    ])
    results = pipeline.run(query_text, top_k=5)
"""
from __future__ import annotations

from rtmdk.pipeline.base import PipelineStage, PipelineContext, StageMetrics
from rtmdk.pipeline.stages import (
    EmbedStage,
    RouteStage,
    RetrieveStage,
    RerankStage,
    CalibrateStage,
    ExplainStage,
)
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.metrics import to_prometheus_format
from rtmdk.pipeline.batch import BatchEmbedStage, BatchPipelineExecutor
from rtmdk.pipeline.registry import StageRegistry, GLOBAL_REGISTRY

# Register default stages in the global registry
GLOBAL_REGISTRY.register("embed", EmbedStage)
GLOBAL_REGISTRY.register("route", RouteStage)
GLOBAL_REGISTRY.register("retrieve", RetrieveStage)
GLOBAL_REGISTRY.register("rerank", RerankStage)
GLOBAL_REGISTRY.register("calibrate", CalibrateStage)
GLOBAL_REGISTRY.register("explain", ExplainStage)

__all__ = [
    "PipelineStage",
    "PipelineContext",
    "StageMetrics",
    "EmbedStage",
    "RouteStage",
    "RetrieveStage",
    "RerankStage",
    "CalibrateStage",
    "ExplainStage",
    "PipelineExecutor",
    "to_prometheus_format",
    "BatchEmbedStage",
    "BatchPipelineExecutor",
    "StageRegistry",
    "GLOBAL_REGISTRY",
]
