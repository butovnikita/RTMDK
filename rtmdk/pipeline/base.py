"""Base classes for the retrieval pipeline."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from numpy.typing import NDArray


@dataclass
class StageMetrics:
    """Latency and throughput metrics for a single stage."""
    name: str
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "latency_ms": round(self.latency_ms, 3),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "error": self.error,
        }


@dataclass
class PipelineContext:
    """Mutable context passed through every pipeline stage."""
    query_text: str
    embedding: Optional[NDArray] = None
    route: Optional[str] = None
    results: List[Tuple[str, float, Any]] = field(default_factory=list)
    top_k: int = 5
    session_id: Optional[str] = None
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[StageMetrics] = field(default_factory=list)

    def add_metric(self, name: str, latency_ms: float, input_count: int = 0, output_count: int = 0, error: Optional[str] = None):
        self.metrics.append(StageMetrics(
            name=name,
            latency_ms=latency_ms,
            input_count=input_count,
            output_count=output_count,
            error=error,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "route": self.route,
            "top_k": self.top_k,
            "results_count": len(self.results),
            "explanations_count": len(self.explanations),
            "stages": [m.to_dict() for m in self.metrics],
            "total_latency_ms": round(sum(m.latency_ms for m in self.metrics), 3),
        }


class PipelineStage(ABC):
    """Abstract base for a retrieval pipeline stage."""

    name: str = "abstract"
    enabled: bool = True

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform the context and return it.

        A stage may:
        - Read ctx.embedding / ctx.results
        - Write ctx.results / ctx.route / ctx.explanations
        - Call ctx.add_metric() before returning
        """
        ...

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Wrap process() with timing and error handling."""
        if not self.enabled:
            ctx.add_metric(self.name, 0.0, input_count=len(ctx.results), output_count=len(ctx.results))
            return ctx

        t0 = time.perf_counter()
        try:
            ctx = self.process(ctx)
            latency = (time.perf_counter() - t0) * 1000
            ctx.add_metric(
                self.name,
                latency,
                input_count=getattr(self, "_last_input_count", len(ctx.results)),
                output_count=len(ctx.results),
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            ctx.add_metric(
                self.name,
                latency,
                input_count=len(ctx.results),
                output_count=len(ctx.results),
                error=f"{type(exc).__name__}: {exc}",
            )
        return ctx
