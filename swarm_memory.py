"""
swarm_memory.py
Swarm Memory for multi-agent RTMDK scenarios.

Extends FederatedRTMDK with consensus-based memory sharing.
Agents specialize, exchange only resonant attractors via weighted voting.

Usage:
    python swarm_memory.py [--n_agents 5] [--n_rounds 10]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import (
    RTMDKConfig, RTMDKMemory,
    SwarmConsensusProtocol,
    KuramotoSync,
    MemoryNode,
)


# ============================================================================
# SWARM AGENT
# ============================================================================

class SwarmAgent:
    """Individual agent in the swarm with specialized memory."""

    def __init__(self, agent_id: str, specialization: str = "general",
                 config: Optional[RTMDKConfig] = None,
                 embedder=None):
        self.agent_id = agent_id
        self.specialization = specialization
        self.config = config or RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5, enable_async=False,
            min_response=0.01,
        )
        self.embedder = embedder or self._default_embedder
        self.memory = RTMDKMemory(config=self.config, embedder=self.embedder)
        self.n_exchanges = 0
        self.last_sync = time.time()
        self.specialty_topics: List[str] = []

    @staticmethod
    def _default_embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base

    def learn(self, text: str, response: str, topic: str = "general"):
        """Learn new information."""
        self.memory.save_context(
            {"input": text, "session_id": self.agent_id, "topic": topic},
            {"output": response}
        )
        if topic not in self.specialty_topics:
            self.specialty_topics.append(topic)

    def query(self, text: str) -> Dict[str, Any]:
        """Query local memory."""
        ctx = self.memory.load_memory_variables(
            {"input": text, "session_id": self.agent_id}
        )
        return {
            "context": ctx["rtmdk_context"],
            "n_nodes": self.memory.field.stats.get("active_nodes", 0),
            "specialization": self.specialization,
        }

    def extract_attractors(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract top resonant attractors for sharing."""
        attractors = []
        for nid, node in list(self.memory.field.nodes.items())[:top_k]:
            attractors.append({
                "node_id": f"{self.agent_id}:{nid}",
                "source_agent": self.agent_id,
                "text": node.content.get("text", ""),
                "latent_pos": node.latent_pos.tolist(),
                "phase": node.phase,
                "amplitude": node.amplitude,
                "salience": node.salience,
                "modality": node.modality,
                "tier": getattr(node, 'tier', 'semantic'),
                "specialization": self.specialization,
            })
        return attractors

    def ingest_attractors(self, attractors: List[Dict[str, Any]]) -> int:
        """Ingest attractors from other agents."""
        ingested = 0
        for attr in attractors:
            if attr["source_agent"] == self.agent_id:
                continue
            # Directly create node with latent position (no projection needed)
            pos = np.array(attr["latent_pos"], dtype=np.float32)
            nid = f"swarm_{attr['source_agent']}_{attr.get('node_id', '')}_{ingested}"
            node = MemoryNode(
                id=nid, latent_pos=pos, phase=attr["phase"],
                amplitude=attr["amplitude"], salience=attr["salience"],
                content={"text": attr["text"], "source": attr["source_agent"],
                         "modality": attr.get("modality", "text"),
                         "tier": attr.get("tier", "semantic")},
                modality=attr.get("modality", "text"),
            )
            node.tier = attr.get("tier", "semantic")
            self.memory.field.nodes[nid] = node
            self.memory.field.node_index.append(nid)
            self.memory.field.stats["total_adds"] += 1
            ingested += 1
        self.n_exchanges += 1
        self.last_sync = time.time()
        return ingested

    def get_stats(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "n_nodes": len(self.memory.field.nodes),
            "n_exchanges": self.n_exchanges,
            "specialty_topics": self.specialty_topics,
        }


# ============================================================================
# SWARM MEMORY
# ============================================================================

class SwarmMemory:
    """Multi-agent swarm memory with consensus-based sharing."""

    def __init__(self, n_agents: int = 5, consensus_threshold: float = 0.5,
                 max_agents: int = 10, vote_weight: float = 0.3):
        self.consensus = SwarmConsensusProtocol(
            consensus_threshold=consensus_threshold,
            max_agents=max_agents,
            vote_weight=vote_weight,
        )
        self.agents: Dict[str, SwarmAgent] = {}
        self.kuramoto = KuramotoSync(coupling_strength=0.3)
        self._exchange_log: List[Dict] = []
        self._round = 0

    def add_agent(self, agent_id: str, specialization: str = "general",
                  embedder=None) -> bool:
        """Add an agent to the swarm."""
        if agent_id in self.agents:
            return False
        agent = SwarmAgent(agent_id, specialization, embedder=embedder)
        self.agents[agent_id] = agent
        return self.consensus.register_agent(agent_id, specialization)

    def learn_distributed(self, data: List[Dict[str, str]]) -> Dict[str, int]:
        """Distribute learning across agents based on specialization."""
        learned = defaultdict(int)
        for item in data:
            topic = item.get("topic", "general")
            # Find best agent for topic
            best_agent = self._find_specialist(topic)
            best_agent.learn(item["input"], item["output"], topic)
            learned[best_agent.agent_id] += 1
        return dict(learned)

    def _find_specialist(self, topic: str) -> SwarmAgent:
        """Find the agent best specialized for a topic."""
        for agent in self.agents.values():
            if topic in agent.specialty_topics:
                return agent
        # Fallback: round-robin
        agent_ids = list(self.agents.keys())
        if not agent_ids:
            raise ValueError("No agents in swarm")
        return self.agents[agent_ids[self._round % len(agent_ids)]]

    def sync_round(self) -> Dict[str, Any]:
        """Execute one synchronization round."""
        self._round += 1
        results = {
            "round": self._round,
            "exchanges": 0,
            "consensus_events": 0,
            "agent_stats": {},
        }

        # Phase 1: Extract attractors from each agent
        all_attractors = {}
        for agent_id, agent in self.agents.items():
            attractors = agent.extract_attractors(top_k=3)
            all_attractors[agent_id] = attractors

        # Phase 2: Consensus-based sharing
        for source_id, attractors in all_attractors.items():
            for attr in attractors:
                # Propose attractor for consensus
                accepted = self.consensus.propose_attractor(source_id, attr)
                if accepted:
                    results["consensus_events"] += 1
                    # Share with other agents
                    for target_id, target_agent in self.agents.items():
                        if target_id != source_id:
                            n_ingested = target_agent.ingest_attractors([attr])
                            results["exchanges"] += n_ingested

        # Phase 3: Kuramoto phase synchronization
        phases = {}
        for agent_id, agent in self.agents.items():
            if agent.memory.field.nodes:
                avg_phase = np.mean([n.phase for n in agent.memory.field.nodes.values()])
                phases[agent_id] = avg_phase

        if len(phases) >= 2:
            self.kuramoto.phases = phases
            self.kuramoto.step(n_steps=5)

        # Collect stats
        for agent_id, agent in self.agents.items():
            results["agent_stats"][agent_id] = agent.get_stats()

        self._exchange_log.append(results)
        return results

    def query_swarm(self, query: str) -> List[Dict[str, Any]]:
        """Query all agents and aggregate results."""
        results = []
        for agent_id, agent in self.agents.items():
            result = agent.query(query)
            if result["context"] and result["context"] not in ("No relevant memory.", "[]"):
                results.append({
                    "agent_id": agent_id,
                    "specialization": agent.specialization,
                    "context": result["context"],
                    "n_nodes": result["n_nodes"],
                })
        return results

    def get_swarm_report(self) -> Dict[str, Any]:
        """Generate comprehensive swarm report."""
        total_nodes = sum(a.memory.field.stats.get("active_nodes", 0)
                         for a in self.agents.values())
        total_exchanges = sum(a.n_exchanges for a in self.agents.values())

        return {
            "n_agents": len(self.agents),
            "total_nodes": total_nodes,
            "total_exchanges": total_exchanges,
            "n_sync_rounds": self._round,
            "n_consensus_events": len(self.consensus._consensus_log),
            "agents": {aid: a.get_stats() for aid, a in self.agents.items()},
            "recent_exchanges": self._exchange_log[-5:],
            "kuramoto_order": self.kuramoto.compute_order_parameter() if self.kuramoto.phases else 0.0,
        }

    def export_swarm(self, path: str):
        """Export entire swarm state."""
        data = {
            "swarm_report": self.get_swarm_report(),
            "consensus": self.consensus.get_state(),
            "agents": {},
        }
        for agent_id, agent in self.agents.items():
            agent_file = f"{path.replace('.json', '')}_{agent_id}.json"
            agent.memory.export_field(agent_file)
            data["agents"][agent_id] = {
                "file": agent_file,
                "stats": agent.get_stats(),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================

def run_swarm_simulation(n_agents: int = 5, n_rounds: int = 10,
                         n_samples_per_agent: int = 10):
    """Run a full swarm memory simulation."""
    print("=" * 60)
    print("  RTMDK Swarm Memory Simulation")
    print("=" * 60)

    swarm = SwarmMemory(n_agents=n_agents, consensus_threshold=0.5)

    # Create specialized agents
    specializations = ["general", "code", "science", "history", "art",
                       "music", "sports", "travel", "food", "tech"]
    for i in range(n_agents):
        spec = specializations[i % len(specializations)]
        swarm.add_agent(f"agent_{i}", spec)
        print(f"  Added agent_{i} (specialization: {spec})")

    # Distributed learning
    print(f"\n[1] Distributed learning ({n_samples_per_agent} samples per agent)...")
    topics = ["python programming", "quantum physics", "world war 2",
              "impressionist art", "classical music", "olympic sports",
              "european travel", "italian cuisine", "machine learning",
              "renaissance history"]
    data = []
    for i in range(n_agents * n_samples_per_agent):
        topic = topics[i % len(topics)]
        data.append({
            "input": f"Tell me about {topic} fact {i}",
            "output": f"Information about {topic}.",
            "topic": topic.split()[0],
        })

    learned = swarm.learn_distributed(data)
    print(f"  Learned: {learned}")

    # Synchronization rounds
    print(f"\n[2] Running {n_rounds} sync rounds...")
    for r in range(n_rounds):
        result = swarm.sync_round()
        if r % 3 == 0 or r == n_rounds - 1:
            print(f"  Round {r+1}: {result['exchanges']} exchanges, "
                  f"{result['consensus_events']} consensus events")

    # Query swarm
    print("\n[3] Querying swarm...")
    queries = ["python", "physics", "history", "art"]
    for q in queries:
        results = swarm.query_swarm(q)
        n_responders = len(results)
        print(f"  '{q}': {n_responders} agents responded")

    # Report
    print("\n[4] Swarm Report:")
    report = swarm.get_swarm_report()
    print(f"  Agents: {report['n_agents']}")
    print(f"  Total nodes: {report['total_nodes']}")
    print(f"  Total exchanges: {report['total_exchanges']}")
    print(f"  Sync rounds: {report['n_sync_rounds']}")
    print(f"  Consensus events: {report['n_consensus_events']}")
    print(f"  Kuramoto order parameter: {report['kuramoto_order']:.3f}")

    # Export
    report_file = "swarm_report.json"
    swarm.export_swarm(report_file)
    print(f"\n  Report saved to {report_file}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMDK Swarm Memory Simulation")
    parser.add_argument("--n_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_rounds", type=int, default=10, help="Number of sync rounds")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per agent")
    args = parser.parse_args()

    run_swarm_simulation(
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_samples_per_agent=args.n_samples,
    )
