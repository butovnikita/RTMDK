"""Tests for rtmdk.production.feedback_loop."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.feedback_loop import FeedbackLoop


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestFeedbackLoop:
    def test_apply_feedback_positive(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        fb = FeedbackLoop(mem)
        result = fb.apply_feedback("hello", quality=0.9)
        assert result["nodes_updated"] >= 0
        assert result["quality"] == 0.9
        assert result["avg_quality"] == 0.9

    def test_apply_feedback_negative(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        nid = list(mem.field.nodes.keys())[0]
        original_salience = mem.field.nodes[nid].salience
        fb = FeedbackLoop(mem)
        fb.apply_feedback("hello", quality=0.1)
        # Low quality should decrease salience slightly
        assert mem.field.nodes[nid].salience < original_salience

    def test_get_stats_empty(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        fb = FeedbackLoop(mem)
        stats = fb.get_stats()
        assert stats["total_feedback"] == 0
        assert stats["avg_quality"] is None

    def test_get_stats_with_feedback(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        fb = FeedbackLoop(mem)
        fb.apply_feedback("q1", quality=1.0)
        fb.apply_feedback("q2", quality=0.5)
        fb.apply_feedback("q3", quality=0.2)
        stats = fb.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["avg_quality"] == pytest.approx(0.567, 0.01)
        assert stats["distribution"]["excellent_0.8_plus"] == 1
        assert stats["distribution"]["good_0.5_to_0.8"] == 1
        assert stats["distribution"]["poor_below_0.5"] == 1

    def test_node_quality_and_session_quality(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        nid = list(mem.field.nodes.keys())[0]
        fb = FeedbackLoop(mem)
        fb.apply_feedback("hello", quality=0.8, session_id="s1", node_ids=[nid])
        assert fb.get_node_quality(nid) == pytest.approx(0.8, 0.01)
        assert fb.get_session_quality("s1") == pytest.approx(0.8, 0.01)

    def test_export_feedback(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        fb = FeedbackLoop(mem)
        fb.apply_feedback("q1", quality=0.9)
        path = str(tmp_path / "feedback.json")
        data = fb.export_feedback(path)
        assert len(data) == 1
        import os
        assert os.path.exists(path)

    def test_avg_quality_property(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        fb = FeedbackLoop(mem)
        assert fb.avg_quality == 0.5
        fb.apply_feedback("q1", quality=1.0)
        assert fb.avg_quality == 1.0


import pytest
