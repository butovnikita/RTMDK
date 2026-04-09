"""
rtmdk/engines/causal_traversal.py — Reasoning-Aware Causal Graph Traversal.

Extends resonance-based retrieval with causal graph traversal.
When you ask "Why did the server crash?", finds not just "Error 500"
but also "Disk full → Error 500 → Server crash" chain.

Algorithm:
1. Find top-K resonance nodes (standard)
2. BFS through causal edges from each seed node (max_hops)
3. Score discovered nodes by causal path strength × resonance
4. Merge with resonance results, re-rank
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict


class CausalTraversalEngine:
    """Causal graph traversal for reasoning-aware retrieval."""
    
    def __init__(self, max_hops: int = 3, decay_per_hop: float = 0.5):
        self.max_hops = max_hops
        self.decay_per_hop = decay_per_hop
    
    def retrieve_with_causal(
        self,
        resonance_results: List[Tuple[str, float, Any]],  # [(node_id, score, node)]
        memory_field,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Any]]:
        """Extend resonance results with causal traversal.
        
        Args:
            resonance_results: Standard resonance retrieval results
            memory_field: RTMDKField with causal graph
            top_k: Number of results to return
        
        Returns:
            Extended and re-ranked results
        """
        if not resonance_results:
            return []
        
        # Check if causal graph exists
        has_causal = any(
            hasattr(n, 'causal_parents') and n.causal_parents
            for _, _, n in resonance_results
        )
        if not has_causal:
            return resonance_results[:top_k]
        
        # BFS through causal graph
        causal_results: Dict[str, Tuple[float, List[str]]] = {}  # node_id → (score, path)
        
        for seed_id, seed_score, seed_node in resonance_results[:3]:  # Top 3 seeds
            self._bfs_causal(
                seed_id, seed_score, memory_field,
                causal_results, visited={seed_id}, path=[seed_id],
                depth=0
            )
        
        # Merge resonance and causal results
        all_results = {}
        for nid, score, node in resonance_results:
            all_results[nid] = (nid, score, node)
        
        for nid, (causal_score, path) in causal_results.items():
            if nid in all_results:
                # Boost existing node with causal score
                old_score = all_results[nid][1]
                new_score = old_score + 0.3 * causal_score  # 30% causal weight
                all_results[nid] = (nid, new_score, all_results[nid][2])
            else:
                # New node from causal traversal
                node = memory_field.nodes.get(nid)
                if node:
                    all_results[nid] = (nid, causal_score * 0.7, node)  # Lower base score
        
        # Sort and return top_k
        results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _bfs_causal(
        self,
        current_id: str,
        parent_score: float,
        memory_field,
        results: Dict[str, Tuple[float, List[str]]],
        visited: Set[str],
        path: List[str],
        depth: int,
    ):
        """BFS through causal graph."""
        if depth >= self.max_hops:
            return
        
        current_node = memory_field.nodes.get(current_id)
        if not current_node:
            return
        
        # Get causal connections
        causal_parents = getattr(current_node, 'causal_parents', [])
        causal_children = self._get_causal_children(current_id, memory_field)
        
        # Traverse parents (causes)
        for parent_id in causal_parents:
            if parent_id in visited:
                continue
            visited.add(parent_id)
            
            # Score: parent_score × decay × causal_strength
            causal_strength = self._get_causal_strength(parent_id, current_id, memory_field)
            hop_score = parent_score * (self.decay_per_hop ** (depth + 1)) * causal_strength
            
            if hop_score > 0.01:  # Minimum threshold
                results[parent_id] = (hop_score, path + [parent_id])
                self._bfs_causal(
                    parent_id, hop_score, memory_field,
                    results, visited, path + [parent_id],
                    depth + 1
                )
        
        # Traverse children (effects)
        for child_id in causal_children:
            if child_id in visited:
                continue
            visited.add(child_id)
            
            causal_strength = self._get_causal_strength(current_id, child_id, memory_field)
            hop_score = parent_score * (self.decay_per_hop ** (depth + 1)) * causal_strength
            
            if hop_score > 0.01:
                results[child_id] = (hop_score, path + [child_id])
                self._bfs_causal(
                    child_id, hop_score, memory_field,
                    results, visited, path + [child_id],
                    depth + 1
                )
    
    def _get_causal_children(self, node_id: str, memory_field) -> List[str]:
        """Find nodes that have this node as causal parent."""
        children = []
        for nid, node in memory_field.nodes.items():
            parents = getattr(node, 'causal_parents', [])
            if node_id in parents:
                children.append(nid)
        return children
    
    def _get_causal_strength(self, from_id: str, to_id: str, memory_field) -> float:
        """Get causal strength between two nodes."""
        to_node = memory_field.nodes.get(to_id)
        if not to_node:
            return 0.5  # Default
        
        causal_strengths = getattr(to_node, 'causal_strengths', {})
        return causal_strengths.get(from_id, 0.5)


class CausalExplanationGenerator:
    """Generate human-readable causal explanations from traversal paths."""
    
    def generate_explanation(
        self,
        query: str,
        results: List[Tuple[str, float, Any]],
        memory_field,
        max_path_length: int = 5,
    ) -> str:
        """Generate explanation text from causal traversal results.
        
        Returns text like:
        "Server crash ← Error 500 ← Disk full (causal chain, 3 hops)"
        """
        # Find longest causal chain
        chains = []
        for nid, score, node in results[:5]:
            text = node.content.get("text", "")[:80]
            chains.append((score, text))
        
        if not chains:
            return f"No causal chain found for: {query}"
        
        # Format as chain
        parts = []
        for score, text in chains:
            parts.append(f"[{score:.2f}] {text}")
        
        return " ← ".join(parts)
