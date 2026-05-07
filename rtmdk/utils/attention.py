"""rtmdk/utils/attention.py"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rtmdk.nodes import MemoryNode


def apply_attention_bias(results: List[Tuple[str,
                                             float,
                                             "MemoryNode"]],
                         temperature: float = 1.0) -> List[Tuple[str,
                                                                 float,
                                                                 "MemoryNode"]]:
    if not results:
        return results
    raw_scores = np.array([r for _, r, _ in results])
    if len(raw_scores) < 2:
        return results
    weights = []
    for nid, resp, node in results:
        score = resp
        causal_boost = sum(
            node.causal_strength.values()) if hasattr(
            node, 'causal_strength') else 0
        score *= (1.0 + 0.2 * min(1.0, causal_boost))
        score *= max(0.5, 1.0 - node.tension)
        goal_rel = getattr(node, 'goal_relevance', 0.0)
        score *= (1.0 + 0.3 * goal_rel)
        weights.append(score)
    weights = np.array(weights)
    if temperature > 0:
        exp_weights = np.exp(weights / temperature)
        normalized = exp_weights / (exp_weights.sum() + 1e-8)
    else:
        normalized = weights / (weights.sum() + 1e-8)
    biased_results = []
    for i, (nid, resp, node) in enumerate(results):
        biased_results.append((nid, float(normalized[i]), node))
    biased_results.sort(key=lambda x: x[1], reverse=True)
    return biased_results


def format_cognitive_context(results: List[Tuple[str, float, "MemoryNode"]],
                             bias_applied: bool = False) -> str:
    if not results:
        return "### COGNITIVE_CONTEXT\nNo relevant structures."
    lines = ["### COGNITIVE_CONTEXT"]
    for nid, score, node in results:
        text = node.content.get("text", "unknown")[:80]
        tier = getattr(node, 'tier', 'semantic')
        causal = len(
            node.causal_strength) if hasattr(
            node, 'causal_strength') else 0
        tension = node.tension
        lineage = len(node.lineage) if node.lineage else 0
        tokens = f"[SCORE:{score:.3f}]"
        tokens += f"[TIER:{tier[0].upper()}]"
        if causal > 0:
            tokens += f"[CAUSAL:{causal}]"
        if tension > 0.3:
            tokens += f"[TENSION:{tension:.2f}]"
        if lineage > 0:
            tokens += f"[LINEAGE:{lineage}]"
        lines.append(f"{tokens} {text}")
    return "\n".join(lines)
