"""Tests for pipeline query planner."""
from __future__ import annotations

import pytest

from rtmdk.pipeline.planner import QueryPlanner, ExecutionPlan


class TestQueryPlanner:
    def test_plan_fast_route_skips_rerank_and_calibrate(self):
        planner = QueryPlanner()
        plan = planner.plan("What is AI?", route="fast", top_k=5)
        assert "rerank" not in plan.stage_names
        assert "calibrate" not in plan.stage_names
        assert "embed" in plan.stage_names
        assert "retrieve" in plan.stage_names
        assert plan.skip_reasons["rerank"] == "fast_route_skip:rerank"

    def test_plan_standard_route_runs_all_stages(self):
        planner = QueryPlanner(short_query_threshold=2)  # threshold=2 so "What is AI?" (3 tokens) keeps explain
        plan = planner.plan("What is AI?", route="standard", top_k=5)
        assert "embed" in plan.stage_names
        assert "route" in plan.stage_names
        assert "retrieve" in plan.stage_names
        assert "rerank" in plan.stage_names
        assert "calibrate" in plan.stage_names
        assert "explain" in plan.stage_names
        assert not plan.skipped_stages

    def test_plan_short_query_skips_explain(self):
        planner = QueryPlanner(short_query_threshold=10)
        plan = planner.plan("hi", route="standard", top_k=5)
        assert "explain" not in plan.stage_names
        assert plan.skip_reasons["explain"] == "short_query"

    def test_plan_long_query_keeps_explain(self):
        planner = QueryPlanner(short_query_threshold=10)
        plan = planner.plan("What is the capital of France and why is it historically significant?", route="standard", top_k=5)
        assert "explain" in plan.stage_names

    def test_plan_low_top_k_skips_calibrate(self):
        planner = QueryPlanner()
        plan = planner.plan("test", route="standard", top_k=3)
        assert "calibrate" not in plan.stage_names
        assert plan.skip_reasons["calibrate"] == "low_top_k"

    def test_plan_estimates_latency(self):
        planner = QueryPlanner()
        plan = planner.plan("test", route="fast", top_k=5)
        assert plan.estimated_latency_ms > 0
        assert plan.estimated_cost > 0

    def test_plan_batch(self):
        planner = QueryPlanner()
        plans = planner.plan_batch(
            queries=["hi", "What is AI?"],
            routes=["fast", "standard"],
            top_ks=[3, 5],
        )
        assert len(plans) == 2
        assert "rerank" not in plans[0].stage_names  # fast
        assert "rerank" in plans[1].stage_names  # standard
        assert "calibrate" not in plans[0].stage_names  # fast
        assert "calibrate" not in plans[0].stage_names  # low top_k

    def test_report_savings(self):
        planner = QueryPlanner()
        plan = planner.plan("test", route="fast", top_k=5)
        savings = planner.report_savings(plan)
        assert savings["latency_reduction_pct"] > 0
        assert savings["cost_reduction_pct"] > 0
        assert savings["baseline_latency_ms"] > savings["planned_latency_ms"]

    def test_plan_with_custom_stages(self):
        planner = QueryPlanner()
        plan = planner.plan("test", route="standard", top_k=5, available_stages=["embed", "retrieve"])
        assert plan.stage_names == ["embed", "retrieve"]

    def test_execution_plan_immutable(self):
        plan = ExecutionPlan(stage_names=["embed", "retrieve"])
        # frozen dataclass — field replacement should raise
        with pytest.raises(Exception):
            plan.stage_names = ["embed", "route"]
