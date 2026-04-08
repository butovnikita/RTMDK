"""
rtmdk/production/advanced_retrieval.py — 7 Algorithmic Improvements for RTMDK.

Implements:
1. Hybrid Retrieval (Resonance + BM25 + Cosine)
2. Confidence-Aware Fallback
3. Query Expansion с RTMDK Context
4. Adaptive Retrieval Depth
5. Temporal Decay с Learning
6. Causal Graph-Augmented Retrieval
7. Meta-Retrieval Controller

Each improvement is independently toggleable for A/B testing.
"""

import os
import sys
import re
import time
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rtmdk_memory_v8 import RTMDKMemory, RTMDKConfig, MemoryNode
from rtmdk.production.bm25_fallback import BM25FallbackRetriever


# ============================================================================
# 1. HYBRID RETRIEVAL (Resonance + BM25 + Cosine)
# ============================================================================

class HybridRetriever:
    """Combines RTMDK resonance, BM25, and cosine similarity."""

    def __init__(
        self,
        memory: RTMDKMemory,
        bm25: BM25FallbackRetriever,
        resonance_weight: float = 0.40,
        bm25_weight: float = 0.35,
        cosine_weight: float = 0.25,
    ):
        self.memory = memory
        self.bm25 = bm25
        self.w_res = resonance_weight
        self.w_bm25 = bm25_weight
        self.w_cos = cosine_weight
        # Cache original embeddings for cosine
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def add_embedding(self, node_id: str, embedding: np.ndarray):
        """Store original embedding for cosine computation."""
        self._embeddings_cache[node_id] = embedding.copy()

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, MemoryNode]]:
        """Hybrid retrieval combining all three signals."""
        # Step 1: Get RTMDK resonance results
        rtmdk_results = self._get_rtmdk_results(query, query_embedding, top_k * 3)

        # Step 2: Get BM25 results
        bm25_results = self.bm25.search(query, top_k * 3)

        # Step 3: Normalize and combine scores
        all_scores: Dict[str, Dict[str, float]] = {}

        # RTMDK resonance scores
        max_rtmdk = max((r[1] for r in rtmdk_results), default=1.0)
        for nid, score, node in rtmdk_results:
            all_scores.setdefault(nid, {"rtmdk": 0.0, "bm25": 0.0, "cosine": 0.0, "node": node})
            all_scores[nid]["rtmdk"] = score / max(max_rtmdk, 1e-8)

        # BM25 scores
        max_bm25 = max((s for _, s in bm25_results), default=1.0)
        for nid, score in bm25_results:
            # BM25 returns doc_id, find corresponding node
            node = self._find_node_by_bm25_id(nid)
            if node:
                all_scores.setdefault(node.id, {"rtmdk": 0.0, "bm25": 0.0, "cosine": 0.0, "node": node})
                all_scores[node.id]["bm25"] = score / max(max_bm25, 1e-8)

        # Cosine similarity scores
        query_norm = np.linalg.norm(query_embedding) + 1e-8
        for nid in all_scores:
            if nid in self._embeddings_cache:
                node_emb = self._embeddings_cache[nid]
                cos_sim = float(np.dot(query_embedding, node_emb) / (query_norm * np.linalg.norm(node_emb) + 1e-8))
                all_scores[nid]["cosine"] = max(0.0, (cos_sim + 1.0) / 2.0)  # Normalize to [0, 1]

        # Combined score
        combined = []
        for nid, scores in all_scores.items():
            hybrid = (
                self.w_res * scores["rtmdk"] +
                self.w_bm25 * scores["bm25"] +
                self.w_cos * scores["cosine"]
            )
            combined.append((nid, hybrid, scores["node"]))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _get_rtmdk_results(self, query: str, query_embedding: np.ndarray, top_k: int):
        """Get standard RTMDK results."""
        ctx = self.memory.load_memory_variables({"input": query, "session_id": "hybrid"})
        # Parse results from context (simplified — just get from field.query)
        phase = self.memory._get_phase("hybrid", query_embedding)
        results = self.memory.field.query(query_embedding, phase, top_k=top_k)
        return results

    def _find_node_by_bm25_id(self, bm25_id: str) -> Optional[MemoryNode]:
        """Find RTMDK node by BM25 document ID."""
        # BM25 doc_id format: "doc_{index}"
        # We need to map this back to node — simplified implementation
        # In production, maintain explicit mapping
        for nid, node in self.memory.field.nodes.items():
            if bm25_id in node.content.get("text", "")[:100]:
                return node
        return None


# ============================================================================
# 2. CONFIDENCE-AWARE FALLBACK
# ============================================================================

class ConfidenceAwareFallback:
    """Returns 'I don't know' when confidence is too low."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        bm25: BM25FallbackRetriever,
        rtmdk_threshold: float = 0.3,
        bm25_threshold: float = 0.1,
    ):
        self.hybrid = hybrid_retriever
        self.bm25 = bm25
        self.rtmdk_threshold = rtmdk_threshold
        self.bm25_threshold = bm25_threshold

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float, MemoryNode]], str]:
        """Returns (results, status) where status is 'confident', 'fallback', or 'unknown'."""
        # Try hybrid retrieval
        results = self.hybrid.retrieve(query, query_embedding, top_k)

        if results and results[0][1] >= self.rtmdk_threshold:
            return results, "confident"

        # Fallback to BM25 only
        bm25_results = self.bm25.search(query, top_k)
        if bm25_results and bm25_results[0][1] >= self.bm25_threshold:
            # Convert BM25 results to standard format
            converted = []
            for doc_id, score in bm25_results:
                node = self.hybrid._find_node_by_bm25_id(doc_id)
                if node:
                    converted.append((node.id, score * 0.5, node))  # Lower weight for fallback
            return converted, "fallback"

        # Still no good result
        return [], "unknown"


# ============================================================================
# 3. QUERY EXPANSION WITH RTMDK CONTEXT
# ============================================================================

class QueryExpander:
    """Expands vague queries using RTMDK context."""

    def __init__(self, memory: RTMDKMemory):
        self.memory = memory

    def expand(self, query: str, top_k_context: int = 3) -> str:
        """Expand query with related context keywords."""
        # Get initial context
        ctx = self.memory.load_memory_variables({"input": query, "session_id": "expand"})
        context = ctx.get("rtmdk_context", "")

        if not context or context in ("No relevant memory.", "[]"):
            return query

        # Extract significant keywords from context
        context_words = self._extract_significant_words(context)

        # Expand query
        if context_words:
            expanded = f"{query} {' '.join(context_words[:5])}"
            return expanded
        return query

    @staticmethod
    def _extract_significant_words(text: str) -> List[str]:
        """Extract meaningful words from context, ignoring scores and formatting."""
        # Remove score tags like [R:0.42], [ATTN:0.5], etc.
        text = re.sub(r'\[[\w:.\s]+\]', ' ', text)
        # Tokenize
        tokens = re.findall(r'[a-zа-яё]{4,}', text.lower())
        # Remove common words
        stopwords = {'the', 'this', 'that', 'with', 'from', 'have', 'been', 'were',
                     'what', 'which', 'their', 'there', 'about', 'would', 'could',
                     'should', 'these', 'those', 'other', 'some', 'such', 'only'}
        return [t for t in tokens if t not in stopwords][:10]


# ============================================================================
# 4. ADAPTIVE RETRIEVAL DEPTH
# ============================================================================

class AdaptiveDepthRetriever:
    """Dynamically adjusts top_k based on score distribution."""

    def __init__(
        self,
        retriever,  # Any retriever with retrieve(query, embedding, top_k) interface
        min_k: int = 3,
        max_k: int = 15,
        confidence_threshold: float = 0.1,
    ):
        self.retriever = retriever
        self.min_k = min_k
        self.max_k = max_k
        self.confidence_threshold = confidence_threshold

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, MemoryNode]]:
        """Retrieve with adaptive depth."""
        # First, get a larger pool to analyze score distribution
        pool_size = min(self.max_k * 2, 50)
        pool = self.retriever.retrieve(query, query_embedding, pool_size)

        if not pool:
            return []

        # Analyze score distribution
        scores = [r[1] for r in pool]  # Extract numeric scores from tuples
        if len(scores) < 2:
            return pool[:top_k]

        # Calculate score gap between top results
        score_gap = scores[0] - scores[min(4, len(scores) - 1)]

        if score_gap < self.confidence_threshold:
            # Scores are close → ambiguous query → retrieve more
            effective_k = min(self.max_k, top_k * 2)
        else:
            # Clear winner → retrieve less
            effective_k = max(self.min_k, top_k)

        return pool[:effective_k]


# ============================================================================
# 5. TEMPORAL DECAY WITH LEARNING
# ============================================================================

class TemporalDecayLearner:
    """Adapts decay rate based on user feedback."""

    def __init__(
        self,
        base_decay: float = 0.999,
        learning_rate: float = 0.01,
        min_decay: float = 0.990,
        max_decay: float = 0.9999,
    ):
        self.base_decay = base_decay
        self.lr = learning_rate
        self.min_decay = min_decay
        self.max_decay = max_decay
        self._node_decay_rates: Dict[str, float] = {}
        self._feedback_counts: Dict[str, int] = defaultdict(int)

    def apply_feedback(self, node_id: str, quality: float):
        """Update decay rate for a node based on feedback.

        quality: 0.0 (bad) → 1.0 (excellent)
        High quality → slower decay (lower decay_rate value)
        Low quality → faster decay (higher decay_rate value)
        """
        current = self._node_decay_rates.get(node_id, self.base_decay)
        # If quality > 0.5, slow decay (decrease decay_rate)
        # If quality < 0.5, speed up decay (increase decay_rate)
        delta = self.lr * (0.5 - quality)
        new_decay = current + delta
        new_decay = max(self.min_decay, min(self.max_decay, new_decay))
        self._node_decay_rates[node_id] = new_decay
        self._feedback_counts[node_id] += 1

    def get_decay_rate(self, node_id: str) -> float:
        return self._node_decay_rates.get(node_id, self.base_decay)

    def apply_to_node(self, node: MemoryNode, node_id: str):
        """Apply learned decay rate to a node's salience update."""
        decay = self.get_decay_rate(node_id)
        node.salience *= decay

    @property
    def stats(self) -> Dict:
        if not self._node_decay_rates:
            return {"nodes_tracked": 0, "avg_decay": self.base_decay}
        return {
            "nodes_tracked": len(self._node_decay_rates),
            "avg_decay": float(np.mean(list(self._node_decay_rates.values()))),
            "min_decay": float(np.min(list(self._node_decay_rates.values()))),
            "max_decay": float(np.max(list(self._node_decay_rates.values()))),
        }


# ============================================================================
# 6. CAUSAL GRAPH-AUGMENTED RETRIEVAL
# ============================================================================

class CausalAugmentedRetriever:
    """Retrieval augmented by causal graph traversal."""

    def __init__(
        self,
        memory: RTMDKMemory,
        max_hops: int = 2,
        causal_weight: float = 0.3,
    ):
        self.memory = memory
        self.max_hops = max_hops
        self.causal_weight = causal_weight

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, MemoryNode]]:
        """Retrieve with causal graph augmentation."""
        # Step 1: Get initial results
        phase = self.memory._get_phase("causal", query_embedding)
        initial_results = self.memory.field.query(query_embedding, phase, top_k=top_k * 2)

        if not initial_results:
            return []

        # Step 2: Traverse causal graph from top results
        causal_bonus: Dict[str, float] = {}
        for nid, score, node in initial_results[:3]:  # Top 3 seed nodes
            self._traverse_causal_graph(nid, causal_bonus, depth=0, bonus=score)

        # Step 3: Combine initial scores with causal bonus
        combined = {}
        for nid, score, node in initial_results:
            combined[nid] = (nid, score, node)

        for nid, bonus in causal_bonus.items():
            if nid in combined:
                # Boost existing
                old_score = combined[nid][1]
                new_score = old_score + self.causal_weight * bonus
                combined[nid] = (nid, new_score, combined[nid][2])
            else:
                # New node from causal traversal
                node = self.memory.field.nodes.get(nid)
                if node:
                    combined[nid] = (nid, self.causal_weight * bonus, node)

        results = list(combined.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _traverse_causal_graph(
        self,
        node_id: str,
        bonuses: Dict[str, float],
        depth: int,
        bonus: float,
    ):
        """Traverse causal graph and assign bonuses to related nodes."""
        if depth >= self.max_hops:
            return

        node = self.memory.field.nodes.get(node_id)
        if not node:
            return

        # Get causal parents and effects
        causal_parents = getattr(node, 'causal_parents', [])
        causal_effects = getattr(node, 'causal_strength', {})

        # Apply bonus to causal parents
        for parent_id in causal_parents:
            strength = causal_effects.get(parent_id, 0.5)
            parent_bonus = bonus * strength * (0.5 ** depth)  # Decay with depth
            bonuses[parent_id] = bonuses.get(parent_id, 0.0) + parent_bonus
            self._traverse_causal_graph(parent_id, bonuses, depth + 1, parent_bonus)

        # Apply bonus to causal effects
        for effect_id, strength in causal_effects.items():
            effect_bonus = bonus * strength * (0.5 ** depth)
            bonuses[effect_id] = bonuses.get(effect_id, 0.0) + effect_bonus
            self._traverse_causal_graph(effect_id, bonuses, depth + 1, effect_bonus)


# ============================================================================
# 7. META-RETRIEVAL CONTROLLER
# ============================================================================

class MetaRetrievalController:
    """Selects retrieval strategy based on query characteristics."""

    # Query type patterns
    PATTERNS = {
        "factual": [
            r'\b(what|who|when|where|which|how many|how much)\b',
            r'\b(capital|president|invented|discovered|created|built)\b',
        ],
        "procedural": [
            r'\b(how to|how do|how can|steps?|tutorial|guide|configure|install)\b',
        ],
        "multi-hop": [
            r'\b(why|because|reason|cause|effect|lead to|result in)\b',
        ],
        "vague": [
            r'\b(that thing|you know|remember|we discussed|about)\b',
            r'\b(something|anything|stuff|things?)\b',
        ],
    }

    def __init__(self, strategies: Dict[str, Any]):
        """strategies: dict of query_type → retriever instance."""
        self.strategies = strategies
        self._query_log: List[Dict] = []

    def classify_query(self, query: str) -> str:
        """Classify query into type."""
        query_lower = query.lower()
        scores = {}

        for qtype, patterns in self.PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            scores[qtype] = score

        # Default to factual if no pattern matches
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "factual"
        return best_type

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float, MemoryNode]], str]:
        """Select strategy and retrieve."""
        query_type = self.classify_query(query)
        retriever = self.strategies.get(query_type, self.strategies.get("factual"))

        results = retriever.retrieve(query, query_embedding, top_k)

        self._query_log.append({
            "query": query[:100],
            "type": query_type,
            "n_results": len(results),
            "timestamp": time.time(),
        })

        return results, query_type

    @property
    def stats(self) -> Dict:
        type_counts = defaultdict(int)
        for log in self._query_log:
            type_counts[log["type"]] += 1
        return {
            "total_queries": len(self._query_log),
            "query_type_distribution": dict(type_counts),
        }


# ============================================================================
# INTEGRATION: AdvancedRTMDKRetriever
# ============================================================================

class AdvancedRTMDKRetriever:
    """Combines all 7 improvements into a single retriever."""

    def __init__(
        self,
        memory: RTMDKMemory,
        bm25: BM25FallbackRetriever,
        enable_hybrid: bool = True,
        enable_confidence_aware: bool = True,
        enable_query_expansion: bool = True,
        enable_adaptive_depth: bool = True,
        enable_temporal_decay: bool = True,
        enable_causal_augmentation: bool = True,
        enable_meta_controller: bool = True,
    ):
        self.memory = memory
        self.bm25 = bm25

        # 1. Hybrid retriever
        self.hybrid = HybridRetriever(memory, bm25) if enable_hybrid else None

        # 2. Confidence-aware fallback
        self.confidence_aware = None
        if enable_confidence_aware and self.hybrid:
            self.confidence_aware = ConfidenceAwareFallback(self.hybrid, bm25)

        # 3. Query expander
        self.query_expander = QueryExpander(memory) if enable_query_expansion else None

        # 4. Adaptive depth
        base_retriever = self.confidence_aware or self.hybrid or self
        self.adaptive = AdaptiveDepthRetriever(base_retriever) if enable_adaptive_depth else None

        # 5. Temporal decay learner
        self.temporal_decay = TemporalDecayLearner() if enable_temporal_decay else None

        # 6. Causal augmented retriever
        self.causal = CausalAugmentedRetriever(memory) if enable_causal_augmentation else None

        # 7. Meta-retrieval controller
        if enable_meta_controller:
            strategies = {}
            if self.causal:
                strategies["multi-hop"] = self.causal
            if self.hybrid:
                strategies["factual"] = self.hybrid
                strategies["procedural"] = self.hybrid
            if self.query_expander:
                strategies["vague"] = self._VagueQueryRetriever(self, self.query_expander)
            self.meta_controller = MetaRetrievalController(strategies) if strategies else None
        else:
            self.meta_controller = None

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float, MemoryNode]], str]:
        """Retrieve with all enabled improvements."""
        # 3. Query expansion (if enabled and query is vague)
        if self.query_expander:
            query = self.query_expander.expand(query)

        # 7. Meta-controller selects strategy
        if self.meta_controller:
            results, query_type = self.meta_controller.retrieve(query, query_embedding, top_k)
        # 6. Causal augmentation
        elif self.causal:
            results = self.causal.retrieve(query, query_embedding, top_k)
            query_type = "causal"
        # 2. Confidence-aware fallback
        elif self.confidence_aware:
            results, status = self.confidence_aware.retrieve(query, query_embedding, top_k)
            query_type = status
        # 1. Hybrid retrieval
        elif self.hybrid:
            results = self.hybrid.retrieve(query, query_embedding, top_k)
            query_type = "hybrid"
        else:
            # Fallback to basic RTMDK
            phase = self.memory._get_phase("basic", query_embedding)
            results = self.memory.field.query(query_embedding, phase, top_k)
            results = [(nid, score, node) for nid, score, node in results]
            query_type = "basic"

        # 4. Adaptive depth (if enabled, override top_k)
        if self.adaptive:
            results = self.adaptive.retrieve(query, query_embedding, top_k)

        # 5. Apply temporal decay to results
        if self.temporal_decay:
            for nid, score, node in results:
                self.temporal_decay.apply_to_node(node, nid)

        return results, query_type

    def apply_feedback(self, query: str, node_ids: List[str], quality: float):
        """Apply user feedback for temporal decay learning."""
        if self.temporal_decay:
            for nid in node_ids:
                self.temporal_decay.apply_feedback(nid, quality)

    def get_stats(self) -> Dict:
        """Get comprehensive stats."""
        stats = {
            "hybrid_enabled": self.hybrid is not None,
            "confidence_aware_enabled": self.confidence_aware is not None,
            "query_expansion_enabled": self.query_expander is not None,
            "adaptive_depth_enabled": self.adaptive is not None,
            "temporal_decay_enabled": self.temporal_decay is not None,
            "causal_augmentation_enabled": self.causal is not None,
            "meta_controller_enabled": self.meta_controller is not None,
        }
        if self.temporal_decay:
            stats["temporal_decay"] = self.temporal_decay.stats
        if self.meta_controller:
            stats["meta_controller"] = self.meta_controller.stats
        return stats

    class _VagueQueryRetriever:
        """Internal retriever for vague queries with expansion."""
        def __init__(self, parent, expander):
            self.parent = parent
            self.expander = expander

        def retrieve(self, query, embedding, top_k):
            expanded = self.expander.expand(query)
            if self.parent.hybrid:
                return self.parent.hybrid.retrieve(expanded, embedding, top_k)
            phase = self.parent.memory._get_phase("expanded", embedding)
            results = self.parent.memory.field.query(embedding, phase, top_k)
            return [(nid, score, node) for nid, score, node in results]
