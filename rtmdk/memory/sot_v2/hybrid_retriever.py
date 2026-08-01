"""Hybrid BM25 + SIF Retriever for SOT v2.0.

Theoretical foundation:
    Dense retrieval (SIF) captures semantic similarity but fails on
    exact lexical matches (e.g. IDs, codes, rare technical terms).
    Sparse retrieval (BM25) excels at lexical matching but misses
    semantic paraphrases.

    Their combination via convex fusion:
        S_hybrid = α·S_dense + (1-α)·S_sparse
    is provably more robust than either alone under the assumption
    that their errors are conditionally independent (Clinchant &
    Gaussier 2010, "Combining BM25 and neural language models").

    The optimal α can be estimated from retrieval calibration data
    or set heuristically (typically 0.7-0.8 for semantic-heavy tasks).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BM25Index:
    """Simple in-memory BM25 index over tokenized documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[int, int] = {}
        self.doc_lengths: List[int] = []
        self.doc_token_freqs: List[Dict[int, int]] = []
        self.avgdl = 0.0
        self.N = 0

    def add_document(self, token_ids: List[int]):
        freq: Dict[int, int] = defaultdict(int)
        for t in token_ids:
            freq[t] += 1
        self.doc_token_freqs.append(dict(freq))
        self.doc_lengths.append(len(token_ids))
        for t in freq:
            self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
        self.N += 1
        self.avgdl = sum(self.doc_lengths) / self.N

    def score(self, query_tokens: List[int]) -> np.ndarray:
        """Return BM25 scores for all documents."""
        scores: np.ndarray = np.zeros(self.N, dtype=np.float32)
        for doc_idx, doc_freq in enumerate(self.doc_token_freqs):
            d_len = self.doc_lengths[doc_idx]
            score = 0.0
            for t in query_tokens:
                df = self.doc_freqs.get(t, 0)
                if df == 0:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc_freq.get(t, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * d_len / self.avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[doc_idx] = score
        return scores


class HybridSIFBM25Retriever:
    """Fuses dense SIF retrieval with sparse BM25 retrieval."""

    def __init__(
        self,
        latent_dim: int = 384,
        alpha: float = 0.7,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.k1 = k1
        self.b = b

        self.doc_embs: List[np.ndarray] = []
        self.bm25 = BM25Index(k1=k1, b=b)

    def add_document(self, token_ids: List[int], doc_emb: np.ndarray):
        """Index a document by its tokens and dense embedding."""
        self.doc_embs.append(doc_emb.astype(np.float32))
        self.bm25.add_document(token_ids)

    def query(
        self,
        query_tokens: List[int],
        query_emb: np.ndarray,
        top_k: int = 10,
        pseudo_relevance_k: int = 3,
        rocchio_beta: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """Return top-k document indices with hybrid scores.

        Optional pseudo-relevance feedback (Rocchio expansion):
        top-k documents from the first pass are used to expand the
        query embedding, then a second pass produces the final ranking.
        """
        N = len(self.doc_embs)
        if N == 0:
            return []

        def _score(q_emb: np.ndarray, q_tokens: List[int]) -> np.ndarray:
            doc_matrix = np.stack(self.doc_embs)
            q = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            dense_scores = doc_matrix @ q
            sparse_scores = self.bm25.score(q_tokens)
            d_min: float = float(dense_scores.min())
            d_max: float = float(dense_scores.max())
            s_min: float = float(sparse_scores.min())
            s_max: float = float(sparse_scores.max())
            dense_norm = (dense_scores - d_min) / (d_max - d_min + 1e-8)
            sparse_norm = (sparse_scores - s_min) / (s_max - s_min + 1e-8)
            return self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        # First pass
        hybrid = _score(query_emb, query_tokens)
        first_top = np.argsort(-hybrid)[:pseudo_relevance_k]

        # Rocchio expansion: q_new = q + beta * mean(top_docs)
        if len(first_top) > 0:
            feedback_emb = np.mean([self.doc_embs[i] for i in first_top], axis=0)
            expanded_emb = query_emb + rocchio_beta * feedback_emb
        else:
            expanded_emb = query_emb

        # Second pass with expanded query
        hybrid = _score(expanded_emb, query_tokens)
        top_idx = np.argsort(-hybrid)[:top_k]
        return [(int(i), float(hybrid[i])) for i in top_idx]
