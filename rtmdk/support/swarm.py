"""Swarm consensus protocol for RTMDK."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    pass


class SwarmConsensusProtocol:
    """Consensus-based memory sharing for multi-agent scenarios."""

    def __init__(self, consensus_threshold: float = 0.5, max_agents: int = 10,
                 vote_weight: float = 0.3):
        self.consensus_threshold = consensus_threshold
        self.max_agents = max_agents
        self.vote_weight = vote_weight
        self.agents: Dict[str, Dict] = {}
        self._consensus_log: List[Dict] = []

    def register_agent(
            self,
            agent_id: str,
            specialization: str = "general") -> bool:
        if len(self.agents) >= self.max_agents:
            return False
        self.agents[agent_id] = {
            "specialization": specialization, "vote_weight": self.vote_weight,
            "last_sync": time.time(), "n_exchanges": 0,
        }
        return True

    def propose_attractor(self, proposer_id: str,
                          attractor: Dict[str, Any]) -> bool:
        if proposer_id not in self.agents:
            return False
        total_weight = 0
        agree_weight = 0
        votes = {}
        for agent_id, agent in self.agents.items():
            if agent_id == proposer_id:
                votes[agent_id] = True
                agree_weight += agent["vote_weight"]
                total_weight += agent["vote_weight"]
                continue
            spec_match = 1.0 if agent["specialization"] == "general" else 0.7
            vote = bool(np.random.random() < spec_match)
            votes[agent_id] = vote
            total_weight += agent["vote_weight"]
            if vote:
                agree_weight += agent["vote_weight"]
        consensus_ratio = agree_weight / max(total_weight, 1e-8)
        accepted = consensus_ratio >= self.consensus_threshold
        self._consensus_log.append({
            "proposer": proposer_id,
            "attractor_preview": str(attractor.get("text", ""))[:50],
            "accepted": accepted, "consensus_ratio": consensus_ratio,
            "votes": votes, "timestamp": time.time(),
        })
        if accepted:
            for agent_id in self.agents:
                self.agents[agent_id]["n_exchanges"] += 1
                self.agents[agent_id]["last_sync"] = time.time()
        return accepted

    def get_swarm_status(self) -> Dict:
        return {
            "n_agents": len(self.agents), "agents": dict(self.agents),
            "n_consensus_events": len(self._consensus_log),
            "recent_consensus": self._consensus_log[-5:],
        }

    def get_state(self) -> Dict:
        return {"agents": dict(self.agents),
                "consensus_log": self._consensus_log[-100:]}

    def load_state(self, state: Dict):
        self.agents = state.get("agents", {})
        self._consensus_log = state.get("consensus_log", [])
