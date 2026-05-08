"""Adaptive Cascade Router for RTMDK retrieval pipeline.

Routes queries through the optimal sub-pipeline based on query type:
- factual → fast resonance only
- ambiguous → resonance + engram fallback + hybrid
- causal / multi-hop → full pipeline with causal traversal + reranker
"""

import re
from enum import Enum
from typing import List, Tuple, Any, Optional


class QueryType(Enum):
    FACTUAL = "factual"
    AMBIGUOUS = "ambiguous"
    CAUSAL = "causal"


class AdaptiveCascadeRouter:
    """Lightweight query classifier for cascade routing."""

    # Keywords indicating causal / explanatory intent
    CAUSAL_KEYWORDS = [
        r"\bwhy\b", r"\bbecause\b", r"\bcause\b", r"\breason\b",
        r"\blead to\b", r"\bresult\b", r"\bconsequence\b",
        r"\bhow did\b", r"\bwhat caused\b", r"\bexplain\b",
    ]

    # Keywords indicating simple factual intent
    FACTUAL_KEYWORDS = [
        r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\bwhat is\b",
        r"\bdefine\b", r"\blist\b", r"\bname\b",
    ]

    def __init__(self, causal_threshold: float = 0.3, factual_threshold: float = 0.3):
        self.causal_threshold = causal_threshold
        self.factual_threshold = factual_threshold
        self._causal_patterns = [re.compile(kw, re.IGNORECASE) for kw in self.CAUSAL_KEYWORDS]
        self._factual_patterns = [re.compile(kw, re.IGNORECASE) for kw in self.FACTUAL_KEYWORDS]

    def classify(self, query: str) -> QueryType:
        """Classify query into QueryType."""
        causal_score = sum(1 for p in self._causal_patterns if p.search(query))
        factual_score = sum(1 for p in self._factual_patterns if p.search(query))

        if causal_score >= self.causal_threshold:
            return QueryType.CAUSAL
        if factual_score >= self.factual_threshold:
            return QueryType.FACTUAL
        return QueryType.AMBIGUOUS

    def route(self, query: str) -> str:
        """Return stage name for the query."""
        qt = self.classify(query)
        if qt == QueryType.FACTUAL:
            return "fast"
        if qt == QueryType.AMBIGUOUS:
            return "standard"
        return "deep"
