"""OpenTelemetry tracing integration for pipeline stages.

Provides distributed tracing for each pipeline stage, enabling
observability via Jaeger, Zipkin, or any OTLP-compatible backend.

Usage:
    from rtmdk.pipeline.tracing import PipelineTracer

    tracer = PipelineTracer()
    ctx = tracer.trace_run(pipeline_executor, "query", top_k=5)

Optional dependency: opentelemetry-api
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _TRACING_AVAILABLE = True
except ImportError:
    _TRACING_AVAILABLE = False


class PipelineTracer:
    """Wrap pipeline execution with OpenTelemetry spans.

    Each stage becomes a child span of the parent pipeline span.
    Attributes include latency, error info, and breaker state.
    """

    def __init__(self, tracer_name: str = "rtmdk.pipeline"):
        self._tracer: Optional[Any] = None
        if _TRACING_AVAILABLE:
            self._tracer = trace.get_tracer(tracer_name)

    def _is_available(self) -> bool:
        return self._tracer is not None

    def trace_stage(self, stage: PipelineStage, ctx: PipelineContext) -> PipelineContext:
        """Run a single stage inside an OpenTelemetry span.

        Falls back to plain stage.run() if tracing is not available.
        """
        if not self._is_available():
            return stage.run(ctx)

        with self._tracer.start_as_current_span(
            f"pipeline.stage.{stage.name}",
            attributes={
                "pipeline.stage.name": stage.name,
                "pipeline.stage.enabled": stage.enabled,
                "pipeline.query": ctx.query_text[:100],
                "pipeline.session_id": ctx.session_id or "",
            },
        ) as span:
            t0 = __import__("time").perf_counter()
            try:
                ctx = stage.run(ctx)
                latency = (__import__("time").perf_counter() - t0) * 1000
                span.set_attribute("pipeline.latency_ms", round(latency, 3))

                # Add breaker state if present
                if stage.circuit_breaker is not None:
                    span.set_attribute(
                        "pipeline.breaker_state", stage.circuit_breaker.state.value
                    )

                # Check if stage degraded
                latest = ctx.metrics[-1] if ctx.metrics else None
                if latest and latest.degraded:
                    span.set_attribute("pipeline.degraded", True)
                    span.set_attribute("pipeline.error", latest.error or "")
                    span.set_status(Status(StatusCode.ERROR, latest.error))
                else:
                    span.set_status(Status(StatusCode.OK))

            except Exception as exc:
                span.set_attribute("pipeline.error", str(exc))
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            return ctx

    def trace_run(
        self,
        executor,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> PipelineContext:
        """Run a full pipeline inside a parent OpenTelemetry span.

        Each stage becomes a child span automatically.
        """
        if not self._is_available():
            return executor.run(query_text, top_k, session_id, embedding)

        with self._tracer.start_as_current_span(
            "pipeline.run",
            attributes={
                "pipeline.query": query_text[:100],
                "pipeline.top_k": top_k,
                "pipeline.session_id": session_id or "",
                "pipeline.stage_count": len(executor.stages),
            },
        ) as parent_span:
            ctx = PipelineContext(
                query_text=query_text,
                top_k=top_k,
                session_id=session_id,
                embedding=embedding,
            )
            for stage in executor.stages:
                ctx = self.trace_stage(stage, ctx)
                if ctx.skip_remaining:
                    break

            parent_span.set_attribute(
                "pipeline.total_latency_ms",
                round(sum(m.latency_ms for m in ctx.metrics), 3),
            )
            parent_span.set_attribute(
                "pipeline.degraded_stages", ",".join(ctx.degraded_stages)
            )
            return ctx
