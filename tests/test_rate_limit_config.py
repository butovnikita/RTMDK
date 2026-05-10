"""Tests for rate_limit_nodes_per_sec config option."""

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField
from rtmdk.memory.utils import SecurityViolationError


def _make_field(rate_limit: int = 100):
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, rate_limit_nodes_per_sec=rate_limit)
    return RTMDKField(cfg)


class TestRateLimitConfig:
    def test_default_rate_limit_100(self):
        cfg = RTMDKConfig(latent_dim=16)
        assert cfg.rate_limit_nodes_per_sec == 100

    def test_rate_limit_zero_disables_limit(self):
        f = _make_field(rate_limit=0)
        # Should not raise even with many rapid calls
        for i in range(10):
            f.add_node(
                embedding=np.array([0.0] * 16),
                content={"text": f"node {i}"},
                node_id=f"n{i}",
            )
        assert len(f.nodes) == 10

    def test_rate_limit_enforced(self):
        f = _make_field(rate_limit=2)
        f.add_node(embedding=np.array([0.0] * 16), content={"a": 1}, node_id="n0")
        f.add_node(embedding=np.array([0.0] * 16), content={"a": 2}, node_id="n1")
        with pytest.raises(SecurityViolationError, match="Rate limit exceeded"):
            f.add_node(embedding=np.array([0.0] * 16), content={"a": 3}, node_id="n2")

    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("RTMDK_ADD_RATE_LIMIT", "0")
        f = _make_field(rate_limit=5)
        # Env var override should disable rate limit
        for i in range(10):
            f.add_node(
                embedding=np.array([0.0] * 16),
                content={"text": f"node {i}"},
                node_id=f"n{i}",
            )
        assert len(f.nodes) == 10

    def test_env_override_can_enable_limit(self, monkeypatch):
        monkeypatch.setenv("RTMDK_ADD_RATE_LIMIT", "1")
        f = _make_field(rate_limit=100)
        f.add_node(embedding=np.array([0.0] * 16), content={"a": 1}, node_id="n0")
        with pytest.raises(SecurityViolationError, match="Rate limit exceeded"):
            f.add_node(embedding=np.array([0.0] * 16), content={"a": 2}, node_id="n1")
