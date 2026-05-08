"""Tests for query_with_confidence API."""
import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def embedder(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) % (2 ** 32))
    emb = rng.standard_normal(768).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


@pytest.fixture
def memory_conf():
    cfg = RTMDKConfig(
        latent_dim=64,
        conformal_prediction=True,
        conformal_alpha=0.1,
        conformal_min_calib=5,
        top_k=5,
    )
    mem = RTMDKMemory(config=cfg, embedder=embedder)
    # Seed calibration
    for i in range(10):
        mem.add_node(embedder(f"doc{i}"), {"text": f"doc{i}"})
        # Simulate feedback: all are relevant
        mem.field.conformal_calibrator.add_sample(0.9)
    return mem


class TestQueryWithConfidence:
    def test_returns_dict(self, memory_conf):
        q = embedder("query")
        result = memory_conf.query_with_confidence("query", q)
        assert isinstance(result, dict)
        assert "results" in result
        assert "prediction_set" in result
        assert "confidence" in result
        assert "threshold" in result
        assert "coverage_guarantee" in result

    def test_confidence_matches_calibrator(self, memory_conf):
        q = embedder("query")
        result = memory_conf.query_with_confidence("query", q)
        # Calibrator was initialized with alpha=0.1 from config
        assert result["confidence"] == pytest.approx(0.9)

    def test_coverage_guarantee_when_calibrated(self, memory_conf):
        q = embedder("query")
        result = memory_conf.query_with_confidence("query", q)
        assert result["coverage_guarantee"] is True

    def test_insufficient_calibration(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            conformal_prediction=True,
            conformal_alpha=0.1,
            conformal_min_calib=100,  # more than we have
            top_k=5,
        )
        mem = RTMDKMemory(config=cfg, embedder=embedder)
        mem.add_node(embedder("doc"), {"text": "doc"})
        q = embedder("query")
        result = mem.query_with_confidence("query", q)
        assert result["coverage_guarantee"] is False
        assert "insufficient" in result["reason"]

    def test_disabled_conformal(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            conformal_prediction=False,
            top_k=5,
        )
        mem = RTMDKMemory(config=cfg, embedder=embedder)
        mem.add_node(embedder("doc"), {"text": "doc"})
        q = embedder("query")
        result = mem.query_with_confidence("query", q)
        assert result["coverage_guarantee"] is False
        assert "disabled" in result["reason"]
