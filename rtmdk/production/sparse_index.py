"""Sparse inverted index for learned sparse retrieval (BGE-M3, SPLADE, etc.).

Replaces BM25 with a token-level learned sparse index.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SparseIndex:
    """Simple inverted index for sparse vectors."""

    def __init__(self):
        self.index: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self._doc_count = 0

    def insert(self, doc_id: str, sparse_vec: Dict[int, float]) -> None:
        """Insert a document's sparse vector into the index."""
        for token_id, weight in sparse_vec.items():
            self.index[token_id].append((doc_id, weight))
        self._doc_count += 1

    def search(self, query_sparse_vec: Dict[int, float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the index with a query sparse vector."""
        if not query_sparse_vec or not self.index:
            return []
        scores: Dict[str, float] = defaultdict(float)
        for token_id, q_weight in query_sparse_vec.items():
            postings = self.index.get(token_id)
            if not postings:
                continue
            for doc_id, d_weight in postings:
                scores[doc_id] += q_weight * d_weight
        if not scores:
            return []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def clear(self) -> None:
        """Clear all index data."""
        self.index.clear()
        self._doc_count = 0

    @property
    def doc_count(self) -> int:
        return self._doc_count
