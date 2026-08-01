"""Tests for rtmdk.utils.preset_recommender."""

from rtmdk.utils.preset_recommender import recommend_preset


class TestPresetRecommender:
    def test_recommend_local(self):
        result = recommend_preset(expected_nodes=1000, available_ram_mb=256)
        assert result["preset"] == "local"

    def test_recommend_agent(self):
        result = recommend_preset(expected_nodes=50000, available_ram_mb=256, max_latency_ms=50)
        assert result["preset"] == "agent"

    def test_recommend_production(self):
        result = recommend_preset(expected_nodes=80000, available_ram_mb=256, max_latency_ms=100)
        assert result["preset"] == "production"

    def test_recommend_research(self):
        result = recommend_preset(expected_nodes=200000, available_ram_mb=512, use_case="research")
        assert result["preset"] == "research"

    def test_recommend_enterprise(self):
        result = recommend_preset(expected_nodes=1_000_000, available_ram_mb=1024)
        assert result["preset"] == "enterprise"

    def test_recommend_fallback(self):
        result = recommend_preset(expected_nodes=1_000_000, available_ram_mb=16)
        assert "preset" in result
