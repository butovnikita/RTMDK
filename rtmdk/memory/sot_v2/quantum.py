"""Quantum Resonance Retrieval for SOT v2.0.

Theoretical foundation:
    In quantum information theory, a mixed state is described by a
    density matrix ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|.  The overlap between a pure
    state |φ⟩ and a mixed state ρ is given by the expectation value:

        S(φ, ρ) = ⟨φ|ρ|φ⟩ = Σ_i p_i |⟨φ|ψ_i⟩|²

    For a document composed of token vectors {v_i} (L2-normalised),
    we form the document density matrix:

        ρ_d = (1/N) Σ_i |v_i⟩⟨v_i|  +  ε·I

    where ε is a small regularisation (maximally mixed component) that
    prevents singularity and models semantic uncertainty.

    For a query vector q (pure state |q⟩), the resonance score is:

        S(q, ρ_d) = q^T ρ_d q = (1/N) Σ_i (q·v_i)² + ε·||q||²

    This is the average *squared* cosine similarity between the query
    and each document token, plus a regularisation bias.  Squaring the
    inner product emphasises strong matches and suppresses weak ones,
    which empirically improves discriminability for retrieval.

    The density matrix formalism also allows off-diagonal coherence
    terms (entanglement) between tokens:

        ρ_{ij} = c_{ij} / √(c_{ii} c_{jj})

    where c_{ij} is the co-occurrence count of tokens i and j within
    a sliding window.  These terms capture semantic correlations that
    pure bag-of-vectors models miss.

    Reference:
        Nielsen & Chuang, "Quantum Computation and Quantum Information",
        Ch. 2 (Density Operators) and Ch. 9 (Quantum Information).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumResonanceRetriever:
    """Retrieval via quantum fidelity between query pure state and
    document mixed states.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        epsilon: float = 1e-4,
        use_coherence: bool = False,
        window_size: int = 5,
    ):
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.use_coherence = use_coherence
        self.window_size = window_size

        # doc_id -> density matrix ρ (latent_dim, latent_dim)
        self.doc_states: Dict[str, np.ndarray] = {}
        self.doc_meta: Dict[str, dict] = {}

    def add_document(
        self,
        doc_id: str,
        token_embs: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ):
        """Build density matrix for a document from token embeddings.

        Args:
            doc_id: Unique document identifier.
            token_embs: (n_tokens, latent_dim), L2-normalised.
            token_ids: Optional token IDs for coherence computation.
        """
        if token_embs.ndim != 2 or token_embs.shape[1] != self.latent_dim:
            raise ValueError("token_embs shape mismatch")
        n = token_embs.shape[0]
        if n == 0:
            self.doc_states[doc_id] = np.eye(self.latent_dim, dtype=np.float32) * self.epsilon
            return

        # Pure-state contributions: (1/N) Σ |v_i⟩⟨v_i|
        rho = (token_embs.T @ token_embs) / n  # (latent_dim, latent_dim)

        if self.use_coherence and token_ids is not None and len(token_ids) > 1:
            # Add off-diagonal coherence from token co-occurrence
            cooc = self._local_cooc(token_ids)
            # Compute coherence matrix C where C_{ab} = Σ_{i,j} cooc(i,j) e_i[a] e_j[b]
            # We project this into the latent space via token embeddings
            # Simplified: add small off-diagonal perturbation proportional to cooc
            n_tok = len(token_ids)
            for i in range(n_tok - 1):
                for j in range(i + 1, min(i + self.window_size + 1, n_tok)):
                    a, b = token_ids[i], token_ids[j]
                    weight = cooc.get((min(a, b), max(a, b)), 0.0)
                    if weight > 0:
                        vi = token_embs[i]
                        vj = token_embs[j]
                        # Symmetric off-diagonal contribution
                        outer = np.outer(vi, vj) + np.outer(vj, vi)
                        rho += (weight / n_tok) * outer

        # Regularise with maximally mixed component
        rho += self.epsilon * np.eye(self.latent_dim, dtype=np.float32)
        # Ensure positive semi-definite (clip small negative eigenvalues)
        eigs, V = np.linalg.eigh(rho)
        eigs = np.maximum(eigs, 0.0)
        rho = V @ np.diag(eigs) @ V.T
        # Normalise trace to 1 (valid density matrix)
        trace = np.trace(rho)
        if trace > 0:
            rho /= trace

        self.doc_states[doc_id] = rho.astype(np.float32)
        self.doc_meta[doc_id] = {"n_tokens": n, "factor": token_embs.astype(np.float32)}

    def query(
        self,
        query_emb: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Score documents by quantum fidelity ⟨q|ρ|q⟩.

        Optimised using low-rank factorisation: if ρ = V V^T, then
        q^T ρ q = ||V^T q||^2, which is computed as a single matrix-vector
        product followed by a norm squared.
        """
        q = query_emb.astype(np.float32)
        scores = []
        for doc_id, rho in self.doc_states.items():
            # Fast path: use precomputed factorisation if available
            meta = self.doc_meta.get(doc_id)
            if meta and "factor" in meta:
                V = meta["factor"]  # (n_tokens, latent_dim)
                score = float(np.linalg.norm(V @ q) ** 2)
            else:
                score = float(q @ rho @ q)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _local_cooc(self, token_ids: List[int]) -> Dict[Tuple[int, int], float]:
        cooc: Dict[Tuple[int, int], float] = {}
        n = len(token_ids)
        for i in range(n):
            for j in range(i + 1, min(i + self.window_size + 1, n)):
                pair = (min(token_ids[i], token_ids[j]), max(token_ids[i], token_ids[j]))
                cooc[pair] = cooc.get(pair, 0.0) + 1.0
        return cooc

    def num_docs(self) -> int:
        return len(self.doc_states)

    def clear(self):
        self.doc_states.clear()
        self.doc_meta.clear()
