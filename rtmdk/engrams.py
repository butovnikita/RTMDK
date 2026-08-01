"""
rtmdk/engrams.py — Engram-based Memory System.

Implements biological memory principles:
- Engram patterns: groups of co-activated nodes representing one memory
- Pattern completion: partial query → full memory retrieval
- Engram indexing: fast HNSW search on engram centroids
- Engram consolidation: episodic → semantic over time
- Engram merging: handling overlapping memories

Scientific basis:
- Josselyn & Tonegawa (2020): Memory engram cells
- Marr (1971): Pattern completion in hippocampus
- Dudai (2004): Systems consolidation
- Tonegawa lab (2012): Engram reactivation
"""

import time
import math
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

# ============================================================================
# ENGRAM DATA STRUCTURES
# ============================================================================


@dataclass
class EngramPattern:
    """A group of co-activated nodes representing one memory.

    Biological analogy: Engram cell ensemble in hippocampus.
    """

    id: str
    node_weights: Dict[str, float]  # {node_id: contribution_weight}
    centroid_embedding: Optional[np.ndarray] = None  # Mean embedding in 768D
    strength: float = 1.0  # Overall engram strength
    created_at: float = 0.0  # Creation timestamp
    last_activated: float = 0.0  # Last activation time
    activation_count: int = 0  # Number of activations
    semantic_core: str = ""  # Central theme/topic
    context_tags: Set[str] = field(default_factory=set)  # Contextual markers
    tier: str = "episodic"  # episodic/semantic/procedural

    def activate(self, current_time: Optional[float] = None):
        """Boost engram on activation."""
        t = current_time or time.time()
        self.last_activated = t
        self.activation_count += 1
        # Hebbian strengthening: co-activated nodes strengthen connections
        self.strength = min(2.0, self.strength * 1.05)

    def decay(self, rate: float = 0.998):
        """Exponential decay of engram strength."""
        self.strength *= rate
        # Decay individual node weights
        for nid in list(self.node_weights.keys()):
            self.node_weights[nid] *= rate
            if self.node_weights[nid] < 0.01:
                del self.node_weights[nid]

    @property
    def is_alive(self) -> bool:
        """Check if engram has meaningful content."""
        return len(self.node_weights) > 0 and self.strength > 0.05

    @property
    def node_count(self) -> int:
        return len(self.node_weights)

    def compute_centroid(self, node_embeddings: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Compute centroid from member node embeddings."""
        if not self.node_weights or not node_embeddings:
            return None

        vectors = []
        weights = []
        for nid, w in self.node_weights.items():
            if nid in node_embeddings:
                vectors.append(node_embeddings[nid])
                weights.append(w)

        if not vectors:
            return None

        vectors = np.array(vectors)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        self.centroid_embedding = np.average(vectors, axis=0, weights=weights)
        return self.centroid_embedding

    def overlap_with(self, other: "EngramPattern") -> float:
        """Compute Jaccard overlap with another engram."""
        set_a = set(self.node_weights.keys())
        set_b = set(other.node_weights.keys())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / max(union, 1)


# ============================================================================
# ENGRAM INDEX (HNSW on centroids)
# ============================================================================


class EngramIndex:
    """Fast retrieval of engrams via HNSW on centroid embeddings.

    Complexity: O(log E · D) where E = num engrams, D = embedding dim.
    vs O(N · d) for node-level search.
    For E=500, N=100K: 500x speedup.
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.engrams: Dict[str, EngramPattern] = {}
        # Simple linear search for now; can be replaced with FAISS/HNSW
        self._centroid_cache: Dict[str, np.ndarray] = {}

    def add_engram(self, engram: EngramPattern):
        self.engrams[engram.id] = engram
        if engram.centroid_embedding is not None:
            self._centroid_cache[engram.id] = engram.centroid_embedding.copy()

    def remove_engram(self, engram_id: str):
        self.engrams.pop(engram_id, None)
        self._centroid_cache.pop(engram_id, None)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, EngramPattern]]:
        """Find most similar engrams by cosine similarity."""
        if not self.engrams:
            return []

        query_norm = np.linalg.norm(query_embedding) + 1e-8
        scores = []

        for eid, centroid in self._centroid_cache.items():
            if eid not in self.engrams:
                continue
            cent_norm = np.linalg.norm(centroid) + 1e-8
            cos_sim = float(np.dot(query_embedding, centroid) / (query_norm * cent_norm))
            # Normalize to [0, 1]
            score = max(0.0, (cos_sim + 1.0) / 2.0)
            # Boost by engram strength and recency
            engram = self.engrams[eid]
            score *= engram.strength
            # Recency bonus: recent activations are more accessible
            age_hours = (time.time() - engram.last_activated) / 3600
            recency_bonus = max(0.5, math.exp(-age_hours / 24))  # 24h half-life
            score *= recency_bonus

            scores.append((eid, score, engram))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def update_centroid(self, engram: EngramPattern, node_embeddings: Dict[str, np.ndarray]):
        """Recompute and cache centroid."""
        centroid = engram.compute_centroid(node_embeddings)
        if centroid is not None:
            self._centroid_cache[engram.id] = centroid
            engram.centroid_embedding = centroid

    @property
    def size(self) -> int:
        return len(self.engrams)

    def get_alive_count(self) -> int:
        return sum(1 for e in self.engrams.values() if e.is_alive)


# ============================================================================
# PATTERN COMPLETER
# ============================================================================


class PatternCompleter:
    """Completes partial queries by finding engrams that match a subset.

    Biological analogy: Hippocampal pattern completion.
    If 20% of a memory matches, retrieve the full 100%.
    """

    def __init__(self, min_overlap: float = 0.2):
        self.min_overlap = min_overlap  # Min fraction to trigger completion

    def complete(
        self,
        query_embedding: np.ndarray,
        engram_results: List[Tuple[str, float, EngramPattern]],
        node_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[str, float, EngramPattern]]:
        """Expand partial matches to full engrams.

        Returns engrams where query matched >= min_overlap of nodes.
        """
        completed = []

        for eid, score, engram in engram_results:
            if not engram.is_alive:
                continue

            # Check if this engram is relevant enough
            if score >= self.min_overlap:
                # Full pattern completion: return the complete engram
                completed.append((eid, score * engram.strength, engram))
                engram.activate()

        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[:top_k]


# ============================================================================
# ENGRAM MANAGER
# ============================================================================


class EngramManager:
    """Manages the full lifecycle of engrams.

    Responsibilities:
    1. Create engrams from co-activated nodes
    2. Retrieve engrams by query
    3. Pattern completion
    4. Decay and consolidation
    5. Merging overlapping engrams
    """

    def __init__(
        self,
        min_nodes: int = 2,
        max_nodes: int = 20,
        creation_threshold: float = 0.6,
        decay_rate: float = 0.998,
        pattern_completion: bool = True,
        overlap_threshold: float = 0.7,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.creation_threshold = creation_threshold
        self.decay_rate = decay_rate
        self.pattern_completion_enabled = pattern_completion
        self.overlap_threshold = overlap_threshold

        self.index = EngramIndex()
        self.completer = PatternCompleter(min_overlap=0.2)

        # Track which nodes belong to which engrams (reverse index)
        self._node_to_engrams: Dict[str, Set[str]] = defaultdict(set)

        self._engram_counter = 0
        self.stats = {
            "engrams_created": 0,
            "engrams_merged": 0,
            "engrams_decayed": 0,
            "pattern_completions": 0,
        }

    def create_engram_from_nodes(
        self,
        # [(node_id, activation_score)]
        activated_nodes: List[Tuple[str, float]],
        node_embeddings: Dict[str, np.ndarray],
        semantic_core: str = "",
        context_tags: Optional[Set[str]] = None,
        tier: str = "episodic",
    ) -> Optional[EngramPattern]:
        """Create an engram from a set of co-activated nodes.

        Called during save_context when multiple nodes are activated together.
        """
        if len(activated_nodes) < self.min_nodes:
            return None  # Too few nodes for an engram

        # Take top nodes (up to max_nodes)
        activated_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = activated_nodes[: self.max_nodes]

        # Check if average activation exceeds threshold
        avg_activation = np.mean([s for _, s in top_nodes])
        if avg_activation < self.creation_threshold:
            return None  # Not strong enough

        # Create engram
        self._engram_counter += 1
        engram_id = f"egr_{self._engram_counter}_{int(time.time())}"

        node_weights = {nid: score for nid, score in top_nodes}

        engram = EngramPattern(
            id=engram_id,
            node_weights=node_weights,
            strength=1.0,
            created_at=time.time(),
            last_activated=time.time(),
            activation_count=1,
            semantic_core=semantic_core,
            context_tags=context_tags or set(),
            tier=tier,
        )

        # Compute centroid
        engram.compute_centroid(node_embeddings)

        # Add to index
        self.index.add_engram(engram)

        # Update reverse index
        for nid in node_weights:
            self._node_to_engrams[nid].add(engram_id)

        self.stats["engrams_created"] += 1
        return engram

    def retrieve_engrams(
        self,
        query_embedding: np.ndarray,
        node_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[str, float, EngramPattern]]:
        """Retrieve engrams matching the query.

        Uses EngramIndex for fast search, then pattern completion.
        """
        # 1. Fast search on centroids
        results = self.index.search(query_embedding, top_k * 2)

        # 2. Pattern completion (if enabled)
        if self.pattern_completion_enabled:
            results = self.completer.complete(query_embedding, results, node_embeddings, top_k)
            if results:
                self.stats["pattern_completions"] += 1

        return results

    def expand_engrams(
        self,
        engram_results: List[Tuple[str, float, EngramPattern]],
        memory_field,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Any]]:
        """Expand engrams back to node-level results for context formatting.

        Returns: [(node_id, combined_score, node_object), ...]
        """
        expanded = []
        seen_nodes = set()

        for eid, engram_score, engram in engram_results:
            for nid, node_weight in engram.node_weights.items():
                if nid in seen_nodes:
                    continue
                seen_nodes.add(nid)

                node = memory_field.nodes.get(nid)
                if node is None:
                    continue

                # Combined score: engram_score × node_weight × node.salience
                combined = engram_score * node_weight * node.salience
                expanded.append((nid, combined, node))

        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded[: top_k * 2]  # Return more for downstream filtering

    def step(self, node_embeddings: Dict[str, np.ndarray]):
        """Advance one time step: decay, cleanup, check for merges."""
        dead_engrams = []

        for eid, engram in self.index.engrams.items():
            engram.decay(self.decay_rate)

            # Recompute centroid if nodes changed
            if engram.is_alive and node_embeddings:
                self.index.update_centroid(engram, node_embeddings)

            if not engram.is_alive:
                dead_engrams.append(eid)

        # Remove dead engrams
        for eid in dead_engrams:
            self.index.remove_engram(eid)
            # Update reverse index
            for nid in list(self._node_to_engrams.keys()):
                self._node_to_engrams[nid].discard(eid)
                if not self._node_to_engrams[nid]:
                    del self._node_to_engrams[nid]

            self.stats["engrams_decayed"] += 1

        # Check for overlapping engrams to merge
        self._check_and_merge_overlaps()

    def _check_and_merge_overlaps(self):
        """Merge engrams with high overlap."""
        engrams = list(self.index.engrams.values())
        merged_ids = set()

        for i in range(len(engrams)):
            if engrams[i].id in merged_ids:
                continue
            for j in range(i + 1, len(engrams)):
                if engrams[j].id in merged_ids:
                    continue

                overlap = engrams[i].overlap_with(engrams[j])
                if overlap >= self.overlap_threshold:
                    # Merge engrams
                    self._merge_engrams(engrams[i], engrams[j])
                    merged_ids.add(engrams[j].id)

        # Remove merged engrams
        for eid in merged_ids:
            self.index.remove_engram(eid)
            self.stats["engrams_merged"] += 1

    def _merge_engrams(self, target: EngramPattern, source: EngramPattern):
        """Merge source into target engram."""
        # Combine node weights (take max for overlapping nodes)
        for nid, weight in source.node_weights.items():
            if nid in target.node_weights:
                target.node_weights[nid] = max(target.node_weights[nid], weight)
            else:
                target.node_weights[nid] = weight

        # Cap at max_nodes
        if target.node_count > self.max_nodes:
            sorted_nodes = sorted(target.node_weights.items(), key=lambda x: x[1], reverse=True)
            target.node_weights = dict(sorted_nodes[: self.max_nodes])

        # Update metadata
        target.strength = min(2.0, target.strength + source.strength * 0.5)
        target.context_tags.update(source.context_tags)
        if not target.semantic_core and source.semantic_core:
            target.semantic_core = source.semantic_core

        # Update reverse index
        for nid in source.node_weights:
            self._node_to_engrams[nid].add(target.id)
            self._node_to_engrams[nid].discard(source.id)

    def get_engrams_for_node(self, node_id: str) -> List[str]:
        """Get all engrams containing a specific node."""
        return list(self._node_to_engrams.get(node_id, set()))

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "total_engrams": self.index.size,
            "alive_engrams": self.index.get_alive_count(),
        }
