"""Result explainability: why was this memory retrieved?"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any


class ResultExplainer:
    """Generate human-readable explanations for retrieved memories.

    Example output:
        "Matched via: sentence similarity (score 0.87) + session context boost + causal link"
    """

    @staticmethod
    def explain(
        query: str,
        node_id: str,
        score: float,
        node: Any,
        session_id: str,
        route: str = "resonance",
    ) -> Dict[str, Any]:
        """Return structured explanation for a single result."""
        reasons = []
        details = {"score": round(score, 4)}

        # Route reason
        if route == "bm25":
            reasons.append("lexical match (BM25)")
        elif route == "hnsw":
            reasons.append("approximate nearest neighbor (HNSW)")
        elif route == "resonance":
            reasons.append("resonance-topological similarity")
        elif route == "engram":
            reasons.append("engram pattern completion")
        elif route == "causal":
            reasons.append("causal traversal")

        # Session boost
        content_session = node.content.get("session") if hasattr(node, "content") else None
        if session_id and content_session == session_id:
            reasons.append("session context boost")
            details["session_match"] = True

        # Sentence reranking
        if hasattr(node, "content") and node.content.get("text"):
            text = node.content.get("text", "")
            if len(text) > 100:
                reasons.append("sentence-level reranking applied")

        # Causal connections
        if hasattr(node, "causal_strength") and node.causal_strength:
            reasons.append(f"causal connections: {len(node.causal_strength)}")

        # Tier
        tier = node.content.get("tier", "semantic") if hasattr(node, "content") else "semantic"
        details["tier"] = tier

        explanation = "Matched via: " + "; ".join(reasons) if reasons else "Unknown match reason"
        return {
            "node_id": node_id,
            "explanation": explanation,
            "details": details,
        }


class QueryRewriter:
    """Rewrite query when retrieval quality is low.

    Uses synonym expansion, disambiguation, and clarification.
    """

    def __init__(self, embedder=None, llm_client=None):
        self.embedder = embedder
        self.llm_client = llm_client

    def should_rewrite(self, results: List[Tuple[str, float, Any]], threshold: float = 0.3) -> bool:
        """Check if top result score is below threshold."""
        if not results:
            return True
        return results[0][1] < threshold

    def rewrite(self, query: str, results: List[Tuple[str, float, Any]]) -> str:
        """Rewrite query for better retrieval.

        Tries heuristic expansion first, then LLM if available.
        """
        # Heuristic: add synonyms from top result text
        if results and self.embedder is not None:
            top_text = results[0][2].content.get("text", "") if hasattr(results[0][2], "content") else ""
            if top_text:
                # Simple keyword overlap expansion
                query_words = set(query.lower().split())
                text_words = set(top_text.lower().split())
                missing = text_words - query_words
                # Add up to 3 missing content words
                added = [w for w in missing if len(w) > 3][:3]
                if added:
                    return f"{query} ({' '.join(added)})"

        # LLM-based rewrite
        if self.llm_client is not None:
            try:
                prompt = (
                    f"Original query: {query}\n"
                    "This query retrieved poor results. Rewrite it to be more specific, "
                    "add synonyms, or disambiguate. Return ONLY the rewritten query, no explanation."
                )
                rewritten = self.llm_client.complete(prompt, max_tokens=100).strip()
                if rewritten and rewritten != query:
                    return rewritten
            except Exception:
                pass

        return query


class QueryIntentClassifier:
    """Classify query intent to tune retrieval strategy.

    Intents:
        - factual: "What is the capital of France?" → boost BM25, exact match
        - exploratory: "Tell me about quantum computing" → causal traversal, diverse results
        - conversational: "How are you?" → session boost, recent memories
        - comparative: "Compare X and Y" → decomposition, multi-hop
    """

    PATTERNS = {
        "factual": [
            r"\b(what is|who is|when did|where is|how many|define|explain)\b",
        ],
        "comparative": [
            r"\b(compare|difference between|versus|vs\.|pros and cons)\b",
        ],
        "conversational": [
            r"\b(how are you|hello|hi |hey |good morning|good evening)\b",
        ],
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        import re
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.PATTERNS.items()
        }

    def classify(self, query: str) -> str:
        """Classify query intent."""
        for intent, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(query):
                    return intent

        # LLM fallback for ambiguous queries
        if self.llm_client is not None:
            try:
                prompt = (
                    f"Classify the following query into one of: factual, exploratory, conversational, comparative.\n"
                    f"Query: {query}\n"
                    f"Return ONLY the label, no explanation."
                )
                label = self.llm_client.complete(prompt, max_tokens=20).strip().lower()
                if label in {"factual", "exploratory", "conversational", "comparative"}:
                    return label
            except Exception:
                pass

        return "exploratory"  # Default
