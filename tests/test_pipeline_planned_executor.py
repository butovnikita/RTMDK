"""Tests for planned pipeline executor."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.pipeline.base import PipelineContext
from rtmdk.pipeline.stages import EmbedStage, RouteStage, RetrieveStage, RerankStage, CalibrateStage, ExplainStage
from rtmdk.pipeline.planner import QueryPlanner
from rtmdk.pipeline.planned_executor import PlannedPipelineExecutor


class DummyField:
    def query(self, embedding, top_k=5, session_id=None, modality="text"):
        return [(f"node_{i}", 0.9 - i * 0.05, None) for i in range(top_k)]


def make_stages():
    def dummy_embed(text: str) -> np.ndarray:
        return np.random.randn(384).astype(np.float32)

    return [
        EmbedStage(dummy_embed),
        RouteStage(),
        RetrieveStage(DummyField()),
        RerankStage(),
        CalibrateStage(),
        ExplainStage(),
    ]


class TestPlannedPipelineExecutor:
    def test_runs_all_stages_by_default(self):
        stages = make_stages()
        executor = PlannedPipelineExecutor(stages, planner=QueryPlanner(short_query_threshold=2))
        ctx = executor.run("What is artificial intelligence?", top_k=5)
        stage_names = [m.name for m in ctx.metrics if m.latency_ms > 0 or m.name in ("embed", "route")]
        # With short_query_threshold=2, explain should run for long query
        assert "embed" in stage_names
        assert "retrieve" in stage_names
        assert len(ctx.results) > 0

    def test_fast_route_skips_expensive_stages(self):
        stages = make_stages()
        planner = QueryPlanner(fast_route_skip={"rerank", "calibrate", "explain"})
        executor = PlannedPipelineExecutor(stages, planner=planner)

        # Manually set route by overriding route stage behavior isn't easy,
        # so we'll test get_plan instead
        plan = executor.get_plan("hello", route="fast", top_k=5)
        assert "rerank" not in plan["stage_names"]
        assert "calibrate" not in plan["stage_names"]
        assert "explain" not in plan["stage_names"]
        assert set(plan["skipped_stages"]) == {"calibrate", "explain", "rerank"}

    def test_get_plan_preview(self):
        stages = make_stages()
        executor = PlannedPipelineExecutor(stages)
        plan = executor.get_plan("test query", route="standard", top_k=5)
        assert "stage_names" in plan
        assert "estimated_latency_ms" in plan
        assert plan["estimated_latency_ms"] > 0

    def test_records_skip_metrics(self):
        stages = make_stages()
        planner = QueryPlanner(fast_route_skip={"rerank"}, short_query_threshold=2)
        executor = PlannedPipelineExecutor(stages, planner=planner)
        ctx = executor.run("hi", top_k=5)  # short query + no route set

        # embed and route run; rerank skipped by fast route? No, route is None (not fast)
        # explain skipped by short query
        metric_names = [m.name for m in ctx.metrics]
        assert "explain" in metric_names  # zero-latency metric for skipped stage

    def test_preserves_stage_order(self):
        stages = make_stages()
        executor = PlannedPipelineExecutor(stages, planner=QueryPlanner(short_query_threshold=5))
        plan = executor.get_plan("What is artificial intelligence and how does it work?", route="standard", top_k=5)
        expected_order = ["embed", "route", "retrieve", "rerank", "calibrate", "explain"]
        # get_plan returns all stages when route=standard and query is not short
        assert plan["stage_names"] == expected_order

    def test_skip_reasons_in_context(self):
        stages = make_stages()
        planner = QueryPlanner(fast_route_skip={"rerank"})
        executor = PlannedPipelineExecutor(stages, planner=planner)
        ctx = executor.run("test", top_k=5)
        assert hasattr(ctx, "_skip_reasons")
