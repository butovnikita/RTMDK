"""Tests for rtmdk.production.ab_testing."""

import pytest
from rtmdk.production.ab_testing import ABTesting


class TestABTesting:
    def test_add_variant(self):
        ab = ABTesting()
        ab.add_variant("A", {"top_k": 5})
        assert "A" in ab._variants
        assert ab._variants["A"] == {"top_k": 5}

    def test_get_variant_deterministic(self):
        ab = ABTesting()
        ab.add_variant("A", {"top_k": 5})
        ab.add_variant("B", {"top_k": 10})
        v1, c1 = ab.get_variant("user123")
        v2, c2 = ab.get_variant("user123")
        assert v1 == v2
        assert c1 == c2

    def test_get_variant_no_variants(self):
        ab = ABTesting()
        name, cfg = ab.get_variant("user123")
        assert name == "control"
        assert cfg == {}

    def test_record_metric_and_results(self):
        ab = ABTesting()
        ab.add_variant("A", {"top_k": 5})
        ab.get_variant("user1")
        ab.record_metric("user1", "recall", 0.95)
        ab.record_metric("user1", "recall", 0.90)
        results = ab.get_results()
        assert "A" in results
        assert results["A"]["metrics"]["recall"]["count"] == 2
        assert results["A"]["metrics"]["recall"]["mean"] == pytest.approx(0.925)

    def test_unknown_user_metric(self):
        ab = ABTesting()
        ab.record_metric("unknown_user", "latency", 1.0)
        results = ab.get_results()
        assert "unknown" not in results
