"""Concrete pipeline stages for RTMDK retrieval."""

from __future__ import annotations
from typing import Any, Callable, Optional
import logging
from numpy.typing import NDArray

from rtmdk.pipeline.base import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class EmbedStage(PipelineStage):
    """Stage 1: Convert query text to embedding vector."""

    name = "embed"

    def __init__(self, embedder: Callable[[str], NDArray]):
        self.embedder = embedder

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.embedding is None:
            ctx.embedding = self.embedder(ctx.query_text)
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.warning("EmbedStage failed for query '%s...': %s", ctx.query_text[:30], exc)
        # If embedding is already present (e.g. from cache), proceed
        if ctx.embedding is None:
            raise  # Cannot recover without embedding
        return ctx


class RouteStage(PipelineStage):
    """Stage 2: Cascade routing (factual/exploratory/deep)."""

    name = "route"

    def __init__(self, router: Optional[Any] = None):
        self.router = router

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self.router is not None:
            ctx.route = self.router.route(ctx.query_text)
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.warning("RouteStage failed: %s. Defaulting to 'standard'.", exc)
        ctx.route = "standard"
        return ctx


class RetrieveStage(PipelineStage):
    """Stage 3: Core retrieval (resonance / HNSW / BM25 hybrid)."""

    name = "retrieve"

    def __init__(self, field: Any, modality: str = "text"):
        self.field = field
        self.modality = modality

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.embedding is None:
            raise RuntimeError("RetrieveStage requires embedding. Ensure EmbedStage runs first.")
        self._last_input_count = 0
        ctx.results = self.field.query(
            ctx.embedding,
            top_k=ctx.top_k,
            session_id=ctx.session_id,
            modality=self.modality,
            query_text=ctx.query_text,
        )
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.error("RetrieveStage failed: %s. This is unrecoverable.", exc)
        raise  # Core retrieval cannot be skipped


class RerankStage(PipelineStage):
    """Stage 4: Sentence-level reranking + optional cross-encoder."""

    name = "rerank"

    def __init__(
        self,
        sentence_reranker: Optional[Any] = None,
        cross_encoder: Optional[Any] = None,
    ):
        self.sentence_reranker = sentence_reranker
        self.cross_encoder = cross_encoder

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self._last_input_count = len(ctx.results)
        if self.sentence_reranker is not None and ctx.results:
            ctx.results = self.sentence_reranker.rerank(ctx.query_text, ctx.results, top_k=ctx.top_k)
        if self.cross_encoder is not None and ctx.results:
            # Placeholder for cross-encoder reranking
            pass
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.warning("RerankStage failed: %s. Returning unranked results.", exc)
        return ctx  # Skip reranking, keep original results


class CalibrateStage(PipelineStage):
    """Stage 5: Conformal prediction filtering."""

    name = "calibrate"

    def __init__(self, calibrator: Optional[Any] = None):
        self.calibrator = calibrator

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self._last_input_count = len(ctx.results)
        if self.calibrator is not None and self.calibrator.n_calibrated > 0:
            scores = [score for _, score, _ in ctx.results]
            nids = [nid for nid, _, _ in ctx.results]
            pred_set, _, threshold = self.calibrator.predict(scores, nids)
            ctx.results = [(nid, score, node) for nid, score, node in ctx.results if nid in pred_set]
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.warning("CalibrateStage failed: %s. Returning uncalibrated results.", exc)
        return ctx  # Skip calibration, keep all results


class ExplainStage(PipelineStage):
    """Stage 6: Add human-readable explanations to results."""

    name = "explain"

    def __init__(self, explainer: Optional[Any] = None):
        self.explainer = explainer

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self._last_input_count = len(ctx.results)
        if self.explainer is not None and ctx.results:
            ctx.explanations = [
                self.explainer.explain(
                    ctx.query_text,
                    nid,
                    score,
                    node,
                    ctx.session_id or "",
                )
                for nid, score, node in ctx.results
            ]
        return ctx

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        logger.warning("ExplainStage failed: %s. Returning results without explanations.", exc)
        ctx.explanations = []
        return ctx
