"""Tests for pipeline cost analyzer."""

from __future__ import annotations

import pytest

from rtmdk.pipeline.cost import PipelineCostAnalyzer


class TestPipelineCostAnalyzer:
    def test_start_and_finalize(self):
        analyzer = PipelineCostAnalyzer()
        analyzer.start("What is AI?")
        analyzer.record_stage("embed", latency_ms=15.0, tokens_in=12)
        analyzer.record_stage("retrieve", latency_ms=1.0)
        cost = analyzer.finalize()
        assert cost.query_text == "What is AI?"
        assert cost.total_latency_ms == 16.0
        assert cost.total_cost > 0
        assert "embed" in cost.stage_costs
        assert "retrieve" in cost.stage_costs

    def test_start_without_finalize_raises(self):
        analyzer = PipelineCostAnalyzer()
        with pytest.raises(RuntimeError):
            analyzer.record_stage("embed", latency_ms=10.0)

    def test_finalize_without_start_raises(self):
        analyzer = PipelineCostAnalyzer()
        with pytest.raises(RuntimeError):
            analyzer.finalize()

    def test_custom_cost_rates(self):
        analyzer = PipelineCostAnalyzer(cost_rates={"embed": 1.0, "retrieve": 0.5})
        analyzer.start("test")
        analyzer.record_stage("embed", latency_ms=0.0)
        analyzer.record_stage("retrieve", latency_ms=0.0)
        cost = analyzer.finalize()
        assert cost.stage_costs["embed"] == 1.0
        assert cost.stage_costs["retrieve"] == 0.5

    def test_summary(self):
        analyzer = PipelineCostAnalyzer()
        for i in range(3):
            analyzer.start(f"query {i}")
            analyzer.record_stage("embed", latency_ms=10.0)
            analyzer.record_stage("retrieve", latency_ms=1.0)
            analyzer.finalize()
        summary = analyzer.summary()
        assert summary["queries"] == 3
        assert summary["avg_latency_ms"] == 11.0
        assert summary["cost_by_stage"]["embed"] > 0

    def test_summary_n_recent(self):
        analyzer = PipelineCostAnalyzer()
        for i in range(5):
            analyzer.start(f"query {i}")
            analyzer.record_stage("embed", latency_ms=10.0)
            analyzer.finalize()
        summary = analyzer.summary(n=2)
        assert summary["queries"] == 2

    def test_latency_adjusts_cost(self):
        analyzer = PipelineCostAnalyzer(cost_rates={"embed": 1.0})
        analyzer.start("test")
        analyzer.record_stage("embed", latency_ms=1000.0)
        cost = analyzer.finalize()
        # cost = rate * (1 + latency/1000) = 1.0 * 2.0 = 2.0
        assert cost.stage_costs["embed"] == pytest.approx(2.0, rel=0.01)

    def test_reset_clears_history(self):
        analyzer = PipelineCostAnalyzer()
        analyzer.start("test")
        analyzer.record_stage("embed", latency_ms=10.0)
        analyzer.finalize()
        analyzer.reset()
        summary = analyzer.summary()
        assert summary == {}
