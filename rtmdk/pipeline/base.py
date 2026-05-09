"""Base classes for the retrieval pipeline."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from numpy.typing import NDArray

from rtmdk.pipeline.circuit_breaker import CircuitBreaker


@dataclass
class StageMetrics:
    """Latency and throughput metrics for a single stage."""
    name: str
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: Optional[str] = None
    degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "latency_ms": round(self.latency_ms, 3),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "error": self.error,
            "degraded": self.degraded,
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
    degraded_stages: List[str] = field(default_factory=list)
    breaker_states: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, name: str, latency_ms: float, input_count: int = 0, output_count: int = 0, error: Optional[str] = None, degraded: bool = False):
        self.metrics.append(StageMetrics(
            name=name,
            latency_ms=latency_ms,
            input_count=input_count,
            output_count=output_count,
            error=error,
            degraded=degraded,
        ))
        if degraded:
            self.degraded_stages.append(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "route": self.route,
            "top_k": self.top_k,
            "results_count": len(self.results),
            "explanations_count": len(self.explanations),
            "degraded_stages": self.degraded_stages,
            "breaker_states": self.breaker_states,
            "stages": [m.to_dict() for m in self.metrics],
            "total_latency_ms": round(sum(m.latency_ms for m in self.metrics), 3),
        }


class PipelineStage(ABC):
    """Abstract base for a retrieval pipeline stage.

    Subclasses should override:
        - process(ctx)        — main logic
        - fallback(ctx, exc)  — optional graceful degradation
        - health_check()      — optional health probe
    """

    name: str = "abstract"
    enabled: bool = True
    circuit_breaker: Optional[CircuitBreaker] = None

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform the context and return it."""
        ...

    def fallback(self, ctx: PipelineContext, exc: Exception) -> PipelineContext:
        """Graceful degradation: return context unchanged on failure.

        Override to provide meaningful fallback (e.g. skip reranking,
        return uncalibrated results, etc.).
        """
        return ctx

    def health_check(self) -> Tuple[bool, Optional[str]]:
        """Return (healthy, reason_or_None).

        Override to implement component-specific health probes.
        """
        return True, None

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Wrap process() with timing, error handling, circuit breaker, and graceful degradation."""
        if not self.enabled:
            ctx.add_metric(self.name, 0.0, input_count=len(ctx.results), output_count=len(ctx.results))
            if self.circuit_breaker:
                ctx.breaker_states[self.name] = self.circuit_breaker.state.value
            return ctx

        # Circuit breaker check
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            ctx = self.fallback(ctx, RuntimeError(f"Circuit breaker open for {self.name}"))
            ctx.add_metric(
                self.name,
                0.0,
                input_count=len(ctx.results),
                output_count=len(ctx.results),
                error="circuit_breaker_open",
                degraded=True,
            )
            ctx.breaker_states[self.name] = self.circuit_breaker.state.value
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
            if self.circuit_breaker:
                self.circuit_breaker.record_success(latency)
                ctx.breaker_states[self.name] = self.circuit_breaker.state.value
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            ctx = self.fallback(ctx, exc)
            ctx.add_metric(
                self.name,
                latency,
                input_count=len(ctx.results),
                output_count=len(ctx.results),
                error=f"{type(exc).__name__}: {exc}",
                degraded=True,
            )
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(latency)
                ctx.breaker_states[self.name] = self.circuit_breaker.state.value
        return ctx
