"""
test_rtmdk_swarm.py
Tests for Swarm Memory (Phase 14 Track 5).
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_memory import SwarmAgent, SwarmMemory
from rtmdk_memory_v8 import RTMDKConfig, SwarmConsensusProtocol


@pytest.fixture
def dummy_embedder():
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return _embed


# ============================================================================
# SWARM AGENT
# ============================================================================

class TestSwarmAgent:
    def test_creation(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "general", embedder=dummy_embedder)
        assert agent.agent_id == "agent_1"
        assert agent.specialization == "general"
        assert agent.n_exchanges == 0

    def test_learn(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "code", embedder=dummy_embedder)
        agent.learn("def hello(): pass", "Python function", "python")
        assert len(agent.memory.field.nodes) == 1
        assert "python" in agent.specialty_topics

    def test_query(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "general", embedder=dummy_embedder)
        agent.learn("test content", "test response", "general")
        result = agent.query("test")
        assert "context" in result
        assert "n_nodes" in result
        assert result["n_nodes"] >= 1

    def test_extract_attractors(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "science", embedder=dummy_embedder)
        for i in range(5):
            agent.learn(f"fact {i}", f"response {i}", "science")
        attractors = agent.extract_attractors(top_k=3)
        assert len(attractors) <= 3
        assert all("latent_pos" in a for a in attractors)
        assert all("source_agent" in a for a in attractors)

    def test_ingest_attractors(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "general", embedder=dummy_embedder)
        attractors = [
            {
                "node_id": "n0",
                "source_agent": "agent_2",
                "text": "external fact",
                "latent_pos": np.random.randn(64).astype(np.float32).tolist(),
                "phase": 0.5,
                "amplitude": 0.7,
                "salience": 0.6,
                "modality": "text",
                "tier": "semantic",
            }
        ]
        n_ingested = agent.ingest_attractors(attractors)
        assert n_ingested == 1
        assert len(agent.memory.field.nodes) == 1
        assert agent.n_exchanges == 1

    def test_get_stats(self, dummy_embedder):
        agent = SwarmAgent("agent_1", "code", embedder=dummy_embedder)
        agent.learn("def foo(): pass", "ok", "python")
        stats = agent.get_stats()
        assert stats["agent_id"] == "agent_1"
        assert stats["specialization"] == "code"
        assert stats["n_nodes"] >= 1


# ============================================================================
# SWARM MEMORY
# ============================================================================

class TestSwarmMemory:
    def test_creation(self):
        swarm = SwarmMemory(n_agents=3)
        assert len(swarm.agents) == 0
        assert swarm.consensus is not None

    def test_add_agent(self, dummy_embedder):
        swarm = SwarmMemory()
        assert swarm.add_agent("agent_1", "general", embedder=dummy_embedder) is True
        assert swarm.add_agent("agent_1", "general", embedder=dummy_embedder) is False  # duplicate
        assert "agent_1" in swarm.agents

    def test_learn_distributed(self, dummy_embedder):
        swarm = SwarmMemory()
        swarm.add_agent("agent_1", "general", embedder=dummy_embedder)
        swarm.add_agent("agent_2", "code", embedder=dummy_embedder)
        data = [
            {"input": "general topic", "output": "general response", "topic": "general"},
            {"input": "python code", "output": "code response", "topic": "python"},
        ]
        learned = swarm.learn_distributed(data)
        assert sum(learned.values()) == 2

    def test_sync_round(self, dummy_embedder):
        swarm = SwarmMemory(consensus_threshold=0.3)
        swarm.add_agent("agent_1", "general", embedder=dummy_embedder)
        swarm.add_agent("agent_2", "code", embedder=dummy_embedder)
        # Learn some data
        for i in range(3):
            swarm.agents["agent_1"].learn(f"fact {i}", f"response {i}", "general")
            swarm.agents["agent_2"].learn(f"code {i}", f"code response {i}", "python")
        result = swarm.sync_round()
        assert result["round"] == 1
        assert "exchanges" in result
        assert "consensus_events" in result
        assert "agent_stats" in result

    def test_query_swarm(self, dummy_embedder):
        swarm = SwarmMemory()
        swarm.add_agent("agent_1", "general", embedder=dummy_embedder)
        swarm.add_agent("agent_2", "science", embedder=dummy_embedder)
        swarm.agents["agent_1"].learn("general knowledge", "general answer", "general")
        swarm.agents["agent_2"].learn("quantum physics", "physics answer", "physics")
        results = swarm.query_swarm("knowledge")
        assert len(results) >= 1

    def test_get_swarm_report(self, dummy_embedder):
        swarm = SwarmMemory()
        swarm.add_agent("agent_1", "general", embedder=dummy_embedder)
        swarm.add_agent("agent_2", "code", embedder=dummy_embedder)
        swarm.agents["agent_1"].learn("test", "response", "general")
        report = swarm.get_swarm_report()
        assert report["n_agents"] == 2
        assert "total_nodes" in report
        assert "total_exchanges" in report
        assert "kuramoto_order" in report

    def test_export_swarm(self, dummy_embedder, tmp_path):
        swarm = SwarmMemory()
        swarm.add_agent("agent_1", "general", embedder=dummy_embedder)
        swarm.agents["agent_1"].learn("test", "response", "general")
        path = str(tmp_path / "swarm_test.json")
        swarm.export_swarm(path)
        assert os.path.exists(path)


# ============================================================================
# SWARM CONSENSUS PROTOCOL (already tested in v14, just verify integration)
# ============================================================================

class TestSwarmConsensusIntegration:
    def test_consensus_with_swarm(self):
        swarm = SwarmMemory(consensus_threshold=0.5, max_agents=5)
        swarm.add_agent("a1", "general")
        swarm.add_agent("a2", "code")
        swarm.add_agent("a3", "science")
        assert swarm.consensus.consensus_threshold == 0.5
        assert len(swarm.consensus.agents) == 3
        assert swarm.consensus.max_agents == 5
