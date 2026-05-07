"""Tests for rtmdk.engines.trust_consensus."""

import numpy as np
import pytest
from rtmdk.engines.trust_consensus import TrustDAG, TrustConsensusEngine


class TestTrustDAG:
    def test_add_trust_edge(self):
        dag = TrustDAG()
        dag.add_trust_edge("a", "b", 0.8)
        assert dag.edges["a"]["b"] == pytest.approx(0.8)

    def test_get_reputation_default(self):
        dag = TrustDAG()
        assert dag.get_reputation("unknown") == pytest.approx(0.5)

    def test_update_reputation(self):
        dag = TrustDAG()
        dag.update_reputation("a", 0.2)
        assert dag.get_reputation("a") == pytest.approx(0.7)


class TestTrustConsensusEngine:
    def test_accept_update_low_reputation(self):
        engine = TrustConsensusEngine(min_reputation=0.5)
        engine.trust_dag.update_reputation("bad", -0.3)
        assert engine.accept_update("bad", "n1", {}) is False

    def test_accept_update_no_embedding(self):
        engine = TrustConsensusEngine()
        assert engine.accept_update("good", "n1", {}) is True

    def test_accept_update_similar_embedding(self):
        engine = TrustConsensusEngine()
        emb = np.array([1.0, 0.0], dtype=np.float32)
        engine.peer_embeddings["sel"] = {"n1": emb}
        assert engine.accept_update("good", "n1", {}, emb) is True

    def test_run_consensus_round(self):
        engine = TrustConsensusEngine()
        emb = np.array([1.0, 0.0], dtype=np.float32)
        engine.accept_update("a", "n1", {}, emb)
        consensus = engine.run_consensus_round()
        assert "n1" in consensus

    def test_detect_byzantine_peers(self):
        engine = TrustConsensusEngine()
        engine.trust_dag.reputation["bad"] = 0.1
        engine.peer_embeddings["bad"] = {}
        bad = engine.detect_byzantine_peers()
        assert "bad" in bad

    def test_get_trust_stats(self):
        engine = TrustConsensusEngine()
        stats = engine.get_trust_stats()
        assert "n_peers" in stats
