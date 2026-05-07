"""
rtmdk/engines/trust_consensus.py — Trust-based Semantic Consensus for Federation.

DAG-based trust protocol for federated RTMDK nodes.
Prevents "poisoning" of memory fields by untrusted peers.

Algorithm:
1. Each peer has a reputation score (0-1)
2. Updates are weighted by reputation
3. Byzantine fault tolerance: reject updates from peers below threshold
4. Reputation updated based on consistency with other peers

Based on: Byzantine consensus protocols + DAG trust models.
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np


class TrustDAG:
    """DAG structure for tracking trust relationships."""

    def __init__(self) -> None:
        self.edges: Dict[str, Dict[str, float]] = defaultdict(
            dict)  # from → {to: weight}
        self.reputation: Dict[str, float] = defaultdict(lambda: 0.5)
        self.update_history: List[Dict[str, Any]] = []

    def add_trust_edge(self, from_peer: str, to_peer: str, weight: float) -> None:
        """Add/update trust edge."""
        self.edges[from_peer][to_peer] = max(0.0, min(1.0, weight))

    def get_reputation(self, peer: str) -> float:
        """Get reputation score for a peer."""
        if peer not in self.reputation and peer not in self.edges:
            return 0.5  # Default for unknown peers
        return self.reputation.get(peer, 0.5)

    def update_reputation(self, peer: str, delta: float) -> None:
        """Update reputation based on consistency."""
        current = self.get_reputation(peer)
        new_rep = max(0.0, min(1.0, current + delta))
        self.reputation[peer] = new_rep


class TrustConsensusEngine:
    """Manages trust-based consensus for federated RTMDK."""

    def __init__(
        self,
        min_reputation: float = 0.3,    # Min reputation to accept updates
        consensus_threshold: float = 0.6,  # Min agreement for consensus
        reputation_decay: float = 0.99,   # How fast reputation decays
        byzantine_tolerance: float = 0.33,  # Max fraction of byzantine nodes
    ):
        self.min_reputation = min_reputation
        self.consensus_threshold = consensus_threshold
        self.reputation_decay = reputation_decay
        self.byzantine_tolerance = byzantine_tolerance

        self.trust_dag = TrustDAG()
        # peer → {node_id → emb}
        self.peer_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self._stats = {
            "updates_accepted": 0,
            "updates_rejected": 0,
            "consensus_rounds": 0,
            "byzantine_detected": 0,
        }

    def accept_update(
        self,
        peer_id: str,
        node_id: str,
        update_data: Dict,
        peer_embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """Accept or reject an update from a peer based on trust.

        Returns True if update is accepted, False if rejected.
        """
        reputation = self.trust_dag.get_reputation(peer_id)

        # Check minimum reputation
        if reputation < self.min_reputation:
            self._stats["updates_rejected"] += 1
            return False

        # If we have embedding, check consistency with our data
        if peer_embedding is not None and node_id in self.peer_embeddings.get("self", {
        }):
            local_emb = self.peer_embeddings["sel"][node_id]
            similarity = float(np.dot(peer_embedding, local_emb) /
                               (np.linalg.norm(peer_embedding) *
                                np.linalg.norm(local_emb) +
                                1e-8))

            # Low similarity → suspicious
            if similarity < 0.3:
                self.trust_dag.update_reputation(peer_id, -0.1)
                self._stats["updates_rejected"] += 1
                return False

        # Accept update (weighted by reputation)
        if peer_embedding is not None:
            self.peer_embeddings.setdefault(peer_id, {})[node_id] = peer_embedding
        self._stats["updates_accepted"] += 1
        return True

    def run_consensus_round(self) -> Dict[str, np.ndarray]:
        """Run one round of trust-weighted consensus.

        Returns aggregated embeddings weighted by peer reputation.
        """
        all_node_ids: set[str] = set()
        for peer_embs in self.peer_embeddings.values():
            all_node_ids.update(peer_embs.keys())

        consensus: Dict[str, np.ndarray] = {}

        for node_id in all_node_ids:
            weighted_sum = None
            total_weight = 0.0

            for peer_id, peer_embs in self.peer_embeddings.items():
                if node_id not in peer_embs:
                    continue

                reputation = self.trust_dag.get_reputation(peer_id)
                weight = reputation * reputation  # Quadratic weighting

                emb = peer_embs[node_id]
                if weighted_sum is None:
                    weighted_sum = weight * emb
                else:
                    weighted_sum += weight * emb
                total_weight += weight

            if total_weight > 0 and weighted_sum is not None:
                consensus[node_id] = weighted_sum / total_weight

        self._stats["consensus_rounds"] += 1

        # Decay reputations slightly
        for peer_id in list(self.trust_dag.reputation.keys()):
            self.trust_dag.reputation[peer_id] *= self.reputation_decay

        return consensus

    def detect_byzantine_peers(self) -> List[str]:
        """Detect potentially byzantine (malicious) peers.

        A peer is suspicious if its updates consistently disagree with majority.
        """
        suspicious = []

        for peer_id in self.peer_embeddings:
            if peer_id == "sel":
                continue

            reputation = self.trust_dag.get_reputation(peer_id)
            if reputation < self.byzantine_tolerance:
                suspicious.append(peer_id)
                self._stats["byzantine_detected"] += 1

        return suspicious

    def get_trust_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "n_peers": len(
                self.peer_embeddings),
            "avg_reputation": float(
                np.mean(
                    list(
                        self.trust_dag.reputation.values()))) if self.trust_dag.reputation else 0.5,
            "min_reputation": float(
                np.min(
                    list(
                        self.trust_dag.reputation.values()))) if self.trust_dag.reputation else 0.5,
        }
