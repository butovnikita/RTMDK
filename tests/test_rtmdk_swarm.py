"""
tests/test_rtmdk_swarm.py — Tests for Swarm Consensus module.

Covers:
1. Agent registration
2. Consensus proposal
3. State serialization
"""

import pytest
import numpy as np

from rtmdk.support.swarm import SwarmConsensusProtocol


class TestSwarmConsensus:
    def test_register_agent(self):
        swarm = SwarmConsensusProtocol(consensus_threshold=0.5, max_agents=3)
        assert swarm.register_agent("agent_1", "general") is True
        assert swarm.register_agent("agent_2", "legal") is True
        assert len(swarm.agents) == 2

    def test_max_agents_limit(self):
        swarm = SwarmConsensusProtocol(max_agents=2)
        assert swarm.register_agent("a1") is True
        assert swarm.register_agent("a2") is True
        assert swarm.register_agent("a3") is False

    def test_propose_attractor_requires_registered_agent(self):
        swarm = SwarmConsensusProtocol()
        result = swarm.propose_attractor("unknown", {"topic": "ai"})
        assert result is False

    def test_get_state_roundtrip(self):
        swarm = SwarmConsensusProtocol()
        swarm.register_agent("agent_1", "medical")
        state = swarm.get_state()
        assert "agents" in state
        assert "agent_1" in state["agents"]

        new_swarm = SwarmConsensusProtocol()
        new_swarm.load_state(state)
        assert "agent_1" in new_swarm.agents
        assert new_swarm.agents["agent_1"]["specialization"] == "medical"
