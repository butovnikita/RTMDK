"""Spectral Self-Supervised Embedder for SOT v2.0.

Theoretical foundation:
    Given a token co-occurrence graph G = (V, E) with edge weights
    w_{ij} = max(0, PMI(i,j)), the optimal low-dimensional embedding
    that preserves local graph structure is given by the k smallest
    eigenvectors of the normalized graph Laplacian
    L_sym = I - D^{-1/2} A D^{-1/2}.

    This is the spectral relaxation of the Ratio-Cut problem, and it
    minimises the Dirichlet energy:

        E(X) = ½ Σ_{i,j} w_{ij} ||x_i - x_j||²

    subject to orthonormality constraints X^T X = I.
    (see von Luxburg 2007, "A Tutorial on Spectral Clustering")

    In practice we compute the truncated SVD of the PMI matrix
    PMI(i,j) = log( count(i,j)·N / (count(i)·count(j)) ),
    which is equivalent to spectral embedding on the pointwise-mutual-
    information graph (Levy & Goldberg 2014, "Neural Word Embedding
    as Implicit Matrix Factorization").

    The embedding for token i is:
        e_i = U_i · sqrt(Σ_i)
    where U, Σ are the left singular vectors and values of the PMI matrix.
    This factorisation is identical (up to rotation) to the optimal
    spectral embedding.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpectralEmbedder:
    """Self-supervised token embedder via spectral decomposition of PMI graph.

    Requires no teacher model. Training data = raw corpus texts.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        token_dim: Optional[int] = None,
        window_size: int = 5,
        pmi_shift: float = 0.0,
        min_count: int = 2,
    ):
        self.latent_dim = latent_dim
        self.token_dim = token_dim or latent_dim
        self.window_size = window_size
        self.pmi_shift = pmi_shift  # Shifted PMI (Levy & Goldberg): PMI - log(k)
        self.min_count = min_count

        self.embeddings: Dict[int, np.ndarray] = {}
        self.projection: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        tokenized_docs: List[List[int]],
        vocab_size: int,
    ) -> "SpectralEmbedder":
        """Fit spectral embeddings on tokenized documents.

        Args:
            tokenized_docs: List of documents, each a list of token IDs.
            vocab_size: Total number of distinct token IDs.
        """
        logger.info(
            "SpectralEmbedder: fitting on %d docs, vocab=%d, dim=%d",
            len(tokenized_docs),
            vocab_size,
            self.latent_dim,
        )

        # 1. Count unigrams and co-occurrences
        unigram_counts = np.zeros(vocab_size, dtype=np.float64)
        cooc_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        total_windows = 0

        for doc in tokenized_docs:
            if not doc:
                continue
            for t in doc:
                if 0 <= t < vocab_size:
                    unigram_counts[t] += 1.0
            # Sliding window co-occurrences
            w = self.window_size
            for i in range(len(doc)):
                for j in range(i + 1, min(i + w + 1, len(doc))):
                    a, b = doc[i], doc[j]
                    if 0 <= a < vocab_size and 0 <= b < vocab_size:
                        cooc_counts[a, b] += 1.0
                        cooc_counts[b, a] += 1.0
                        total_windows += 1

        # Filter rare tokens
        valid_mask = unigram_counts >= self.min_count
        n_valid = int(valid_mask.sum())
        logger.info(
            "SpectralEmbedder: %d / %d tokens pass min_count=%d",
            n_valid,
            vocab_size,
            self.min_count,
        )

        if n_valid == 0:
            logger.warning("No valid tokens, using random embeddings")
            rng = np.random.default_rng(42)
            for t in range(vocab_size):
                emb = rng.standard_normal(self.token_dim).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-8
                self.embeddings[t] = emb
            return self

        # Build dense sub-matrix for valid tokens
        valid_idx = np.where(valid_mask)[0]
        idx_map = {int(v): i for i, v in enumerate(valid_idx)}
        u_counts = unigram_counts[valid_idx] + 1e-8
        c_counts = cooc_counts[np.ix_(valid_idx, valid_idx)]

        N = float(c_counts.sum()) + 1e-8

        # 2. Compute PMI matrix
        # PMI(i,j) = log( P(i,j) / (P(i)P(j)) )
        #           = log( c_ij / N ) - log( c_i / N ) - log( c_j / N )
        #           = log(c_ij) + log(N) - log(c_i) - log(c_j)
        log_cij = np.log(c_counts + 1e-8)
        log_ci = np.log(u_counts)
        log_cj = np.log(u_counts)
        pmi = log_cij + math.log(N) - log_ci[:, None] - log_cj[None, :]
        if self.pmi_shift > 0:
            pmi -= math.log(self.pmi_shift)
        pmi = np.maximum(pmi, 0.0)  # SPPMI: keep only positive PMI
        pmi = pmi.astype(np.float32)

        # 3. Truncated SVD on PMI matrix
        # For n_valid <= 10000, dense SVD is fast enough
        logger.info("SpectralEmbedder: computing SVD on %dx%d PMI matrix...", n_valid, n_valid)
        U, S, Vt = np.linalg.svd(pmi, full_matrices=False)
        # Keep top-latent_dim components
        k = min(self.latent_dim, n_valid)
        Uk = U[:, :k]
        Sk = np.sqrt(S[:k])
        raw_emb = (Uk * Sk[None, :]).astype(np.float32)  # (n_valid, k)

        # Normalise
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-8
        raw_emb /= norms

        # 4. Store embeddings
        for orig_t, sub_i in idx_map.items():
            emb = raw_emb[sub_i]
            # Pad if k < latent_dim
            if k < self.latent_dim:
                pad = np.zeros(self.latent_dim - k, dtype=np.float32)
                emb = np.concatenate([emb, pad])
            self.embeddings[orig_t] = emb.astype(np.float32)

        # Random init for rare / invalid tokens
        rng = np.random.default_rng(42)
        for t in range(vocab_size):
            if t not in self.embeddings:
                emb = rng.standard_normal(self.latent_dim).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-8
                self.embeddings[t] = emb

        logger.info("SpectralEmbedder: fit complete, stored %d embeddings", len(self.embeddings))
        return self

    def embed_tokens(self, token_ids: List[int]) -> np.ndarray:
        """Return mean-pooled embedding for a sequence of token IDs."""
        if not token_ids:
            return np.zeros(self.latent_dim, dtype=np.float32)
        vecs = []
        for t in token_ids:
            if t in self.embeddings:
                vecs.append(self.embeddings[t])
        if not vecs:
            return np.zeros(self.latent_dim, dtype=np.float32)
        stacked = np.stack(vecs)
        pooled = stacked.mean(axis=0)
        norm = np.linalg.norm(pooled) + 1e-8
        return (pooled / norm).astype(np.float32)

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "token_dim": self.token_dim,
            "window_size": self.window_size,
            "pmi_shift": self.pmi_shift,
            "min_count": self.min_count,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
        }

    def load_state(self, state: dict) -> "SpectralEmbedder":
        self.latent_dim = state["latent_dim"]
        self.token_dim = state["token_dim"]
        self.window_size = state["window_size"]
        self.pmi_shift = state["pmi_shift"]
        self.min_count = state["min_count"]
        self.embeddings = {int(k): np.array(v, dtype=np.float32) for k, v in state["embeddings"].items()}
        return self
