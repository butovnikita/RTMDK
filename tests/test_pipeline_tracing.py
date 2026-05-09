"""Tests for pipeline OpenTelemetry tracing."""

import pytest

from rtmdk.pipeline.base import PipelineStage, PipelineContext
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.tracing import PipelineTracer, _TRACING_AVAILABLE


class DummyStage(PipelineStage):
    name = "dummy"

    def process(self, ctx):
        ctx.results.append(("n0", 1.0, None))
        return ctx


class FailingStage(PipelineStage):
    name = "fail"

    def process(self, ctx):
        raise RuntimeError("boom")


class TestPipelineTracer:
    def test_trace_stage_without_opentelemetry(self):
        """Tracer works even when opentelemetry is not installed."""
        tracer = PipelineTracer()
        if _TRACING_AVAILABLE:
            # If otel is available, tracer uses it — test still works
            pass
        stage = DummyStage()
        ctx = PipelineContext(query_text="hello")
        result = tracer.trace_stage(stage, ctx)
        assert result.query_text == "hello"
        assert len(result.results) == 1

    def test_trace_run_full_pipeline(self):
        tracer = PipelineTracer()
        executor = PipelineExecutor([DummyStage()])
        ctx = tracer.trace_run(executor, "hello", top_k=5)
        assert ctx.query_text == "hello"
        assert len(ctx.results) == 1

    def test_trace_run_failing_stage(self):
        tracer = PipelineTracer()
        executor = PipelineExecutor([FailingStage()])
        ctx = tracer.trace_run(executor, "hello", top_k=5)
        # Fallback should handle the error
        assert ctx.query_text == "hello"
        assert len(ctx.metrics) == 1
        assert ctx.metrics[0].degraded is True
        assert "boom" in ctx.metrics[0].error

    def test_trace_stage_degraded_sets_attributes(self):
        """Degraded stages should be captured in metrics regardless of tracing."""
        tracer = PipelineTracer()
        stage = FailingStage()
        ctx = PipelineContext(query_text="test")
        result = tracer.trace_stage(stage, ctx)
        assert len(result.metrics) == 1
        assert result.metrics[0].degraded is True

    def test_trace_run_with_skip_remaining(self):
        class SkipStage(PipelineStage):
            name = "skip"

            def process(self, ctx):
                ctx.skip_remaining = True
                return ctx

        tracer = PipelineTracer()
        executor = PipelineExecutor([SkipStage(), DummyStage()])
        ctx = tracer.trace_run(executor, "hello", top_k=5)
        assert ctx.skip_remaining is True
        # Second stage should not have executed
        assert len(ctx.metrics) == 1
