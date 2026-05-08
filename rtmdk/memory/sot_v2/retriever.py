"""Resonance Retriever — Late-interaction retrieval for SOT v2.0.

Theoretical foundation:
    Mean-pooling token embeddings destroys fine-grained semantic
    structure (especially for long documents).  ColBERT-style late
    interaction (Khattab & Zaharia 2020) stores per-token embeddings
    and scores via MaxSim:

        S(q,d) = Σ_{t∈q}  max_{t'∈d}  sim(e_t, e_{t'})

    This is a decomposition of the document representation into a
    bag-of-token-vectors, and the score measures how well each query
    token is "explained" by the best-matching document token.

    Resonance extension:
        Each token carries an amplitude a_t (salience) and phase φ_t
        (contextual orientation).  The resonance score between two
        tokens is:

            R(t, t') = a_t · a_{t'} · cos(φ_t - φ_{t'}) · sim(e_t, e_{t'})

        The document score becomes:

            S_R(q,d) = Σ_{t∈q}  max_{t'∈d}  R(t, t')

    When amplitudes are uniform and phases are zero, this reduces to
    standard MaxSim.  The phase term allows the same lexical token to
    have different semantic orientations in different contexts,
    analogous to polysemy in contextualised embeddings.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ResonanceRetriever:
    """Late-interaction retriever storing per-token document embeddings."""

    def __init__(
        self,
        latent_dim: int = 384,
        use_resonance_phase: bool = False,
        use_amplitude: bool = False,
    ):
        self.latent_dim = latent_dim
        self.use_resonance_phase = use_resonance_phase
        self.use_amplitude = use_amplitude

        # doc_id -> (token_ids, token_embs, amplitudes, phases)
        self.docs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_document(
        self,
        doc_id: str,
        token_ids: List[int],
        token_embs: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ):
        """Store token-level embeddings for a document.

        Args:
            doc_id: Unique document identifier.
            token_ids: List of token IDs (for debugging/sparse index).
            token_embs: Array of shape (n_tokens, latent_dim), L2-normalised.
            amplitudes: Optional array of shape (n_tokens,) — token saliences.
            phases: Optional array of shape (n_tokens,) — token phases in radians.
        """
        if token_embs.ndim != 2 or token_embs.shape[1] != self.latent_dim:
            raise ValueError(
                f"token_embs must be (n_tokens, {self.latent_dim}), got {token_embs.shape}"
            )
        n = token_embs.shape[0]
        if amplitudes is None:
            amplitudes = np.ones(n, dtype=np.float32)
        if phases is None:
            phases = np.zeros(n, dtype=np.float32)
        self.docs[doc_id] = (
            np.array(token_ids, dtype=np.int32),
            token_embs.astype(np.float32),
            amplitudes.astype(np.float32),
            phases.astype(np.float32),
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_token_embs: np.ndarray,
        query_amplitudes: Optional[np.ndarray] = None,
        query_phases: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Score all documents against a query and return top-k.

        Args:
            query_token_embs: (n_q_tokens, latent_dim), L2-normalised.
            query_amplitudes: Optional (n_q_tokens,).
            query_phases: Optional (n_q_tokens,).
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if query_token_embs.ndim != 2:
            raise ValueError("query_token_embs must be 2D")
        n_q = query_token_embs.shape[0]
        if query_amplitudes is None:
            query_amplitudes = np.ones(n_q, dtype=np.float32)
        if query_phases is None:
            query_phases = np.zeros(n_q, dtype=np.float32)

        scores: List[Tuple[str, float]] = []
        for doc_id, (_, doc_embs, doc_amp, doc_phase) in self.docs.items():
            score = self._maxsim_score(
                query_token_embs,
                query_amplitudes,
                query_phases,
                doc_embs,
                doc_amp,
                doc_phase,
            )
            scores.append((doc_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _maxsim_score(
        self,
        q_embs: np.ndarray,
        q_amp: np.ndarray,
        q_phase: np.ndarray,
        d_embs: np.ndarray,
        d_amp: np.ndarray,
        d_phase: np.ndarray,
    ) -> float:
        """Compute MaxSim resonance score between query and document."""
        # Cosine similarity matrix: (n_q, n_d)
        sims = q_embs @ d_embs.T  # assumed L2-normalised

        if self.use_resonance_phase:
            # Phase coherence: cos(Δφ)
            phase_diff = q_phase[:, None] - d_phase[None, :]
            phase_term = np.cos(phase_diff)
            sims = sims * phase_term

        if self.use_amplitude:
            # Amplitude modulation
            amp_term = q_amp[:, None] * d_amp[None, :]
            sims = sims * amp_term

        # MaxSim: for each query token, take best document token
        max_sims = sims.max(axis=1)
        return float(max_sims.sum())

    # ------------------------------------------------------------------ #
    # Batch query
    # ------------------------------------------------------------------ #

    def batch_query(
        self,
        queries: List[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple queries efficiently."""
        return [self.query(q, qa, qp, top_k) for q, qa, qp in queries]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def num_docs(self) -> int:
        return len(self.docs)

    def clear(self):
        self.docs.clear()
