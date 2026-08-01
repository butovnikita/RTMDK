"""Learned Consolidation — memory compression via small MLP.

Instead of heuristic averaging (latent_pos = 0.5*(A+B)), a tiny MLP learns
how to merge two nodes while preserving retrieval quality for queries that
matched either parent.

Mathematical model:
    Given nodes A, B with states x_A, x_B ∈ R^d
    and a set of queries Q = {q_1, ..., q_k} that retrieved A or B,
    find merged state x_M = f_θ(concat(x_A, x_B)) that maximises:

        L = Σ_q [ sim(x_M, q) - max(sim(x_A, q), sim(x_B, q)) ]_+

    where [z]_+ = max(0, z) is the hinge loss.  This pushes the merged
    state to be at least as retrievable as the better parent for every
    query.

    A secondary L2 regularisation term keeps x_M close to the heuristic
    average (trust region):

        L_reg = λ ||x_M - 0.5*(x_A + x_B)||²

Architecture:
    Input:  concat(latent_A, latent_B, phase_sin_A, phase_sin_B,
                   phase_cos_A, phase_cos_B, amp_A, amp_B, sal_A, sal_B)
    Hidden: ReLU(W1·x + b1),  W1 ∈ R^{h×in},  h = 2·latent_dim
    Output: W2·h + b2,        W2 ∈ R^{d×h}

    Total params ≈ 2·d·(2d) + 2·(2d) + d·(2d) = O(d²)
    For d=384: ~450K parameters, fits in L2 cache.

Training:
    Pure NumPy SGD (no PyTorch dependency).  Accumulates merge examples
    in a replay buffer; trains every N merges or on explicit call.

References:
- Graves et al. (2016) "Neural Turing Machines" — learned write/erase
- Mnih et al. (2014) "Neural Episodic Control" — memory compression
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LearnedConsolidator:
    """Small MLP that learns optimal node-merge states."""

    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or max(64, latent_dim)
        self._rng = np.random.default_rng(42)

        # MLP weights — Xavier-like init
        d_in = latent_dim * 2 + 8  # 2 latents + sin/cos phase ×2 + amp ×2 + sal ×2
        d_h = self.hidden_dim
        d_out = latent_dim

        scale1 = math.sqrt(2.0 / d_in)
        self.W1 = self._rng.normal(0, scale1, (d_h, d_in)).astype(np.float32)
        self.b1: np.ndarray = np.zeros(d_h, dtype=np.float32)

        scale2 = math.sqrt(2.0 / d_h)
        self.W2 = self._rng.normal(0, scale2, (d_out, d_h)).astype(np.float32)
        self.b2: np.ndarray = np.zeros(d_out, dtype=np.float32)

        # Replay buffer
        self._buffer: List[Dict] = []
        self._max_buffer = 10_000
        self._trained = False

    # ------------------------------------------------------------------ #
    # Encoding / forward pass
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_pair(
        latent_a: np.ndarray,
        latent_b: np.ndarray,
        phase_a: float,
        phase_b: float,
        amp_a: float,
        amp_b: float,
        sal_a: float,
        sal_b: float,
    ) -> np.ndarray:
        """Concatenate node pair into input vector."""
        return np.concatenate(
            [
                latent_a.astype(np.float32),
                latent_b.astype(np.float32),
                [math.sin(phase_a), math.sin(phase_b)],
                [math.cos(phase_a), math.cos(phase_b)],
                [amp_a, amp_b],
                [sal_a, sal_b],
            ],
            axis=0,
        )

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MLP forward. Returns (output, hidden_activations)."""
        h = np.maximum(self.W1 @ x + self.b1, 0)  # ReLU
        y = self.W2 @ h + self.b2
        return y, h

    def predict(
        self,
        latent_a: np.ndarray,
        latent_b: np.ndarray,
        phase_a: float = 0.0,
        phase_b: float = 0.0,
        amp_a: float = 1.0,
        amp_b: float = 1.0,
        sal_a: float = 1.0,
        sal_b: float = 1.0,
    ) -> np.ndarray:
        """Predict merged latent position."""
        x = self._encode_pair(latent_a, latent_b, phase_a, phase_b, amp_a, amp_b, sal_a, sal_b)
        y, _ = self._forward(x)
        # Normalise to unit sphere (consistent with heuristic merge)
        norm = np.linalg.norm(y) + 1e-8
        return (y / norm).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def add_example(
        self,
        latent_a: np.ndarray,
        latent_b: np.ndarray,
        queries: List[np.ndarray],
        phase_a: float = 0.0,
        phase_b: float = 0.0,
        amp_a: float = 1.0,
        amp_b: float = 1.0,
        sal_a: float = 1.0,
        sal_b: float = 1.0,
    ):
        """Add a merge example to the replay buffer.

        Args:
            latent_a, latent_b: Parent node latent positions.
            queries: List of query embeddings that retrieved A or B.
        """
        self._buffer.append(
            {
                "latent_a": latent_a.copy(),
                "latent_b": latent_b.copy(),
                "queries": [q.copy() for q in queries],
                "phase_a": phase_a,
                "phase_b": phase_b,
                "amp_a": amp_a,
                "amp_b": amp_b,
                "sal_a": sal_a,
                "sal_b": sal_b,
            }
        )
        if len(self._buffer) > self._max_buffer:
            self._buffer.pop(0)

    def train(self, epochs: int = 20, lr: float = 0.01, lambda_reg: float = 0.5, batch_size: int = 32) -> float:
        """Train MLP on replay buffer via SGD.

        Loss = hinge(merged, queries) + λ·L2_reg(merged, heuristic_avg)

        Returns:
            Final average loss.
        """
        if len(self._buffer) < 8:
            logger.info("LearnedConsolidator: buffer too small (%d), skipping train", len(self._buffer))
            return 0.0

        logger.info("LearnedConsolidator: training on %d examples, epochs=%d", len(self._buffer), epochs)

        total_loss = 0.0
        n_batches = 0

        for epoch in range(epochs):
            # Shuffle buffer
            order = self._rng.permutation(len(self._buffer))
            epoch_loss = 0.0
            epoch_n = 0

            for i in range(0, len(order), batch_size):
                batch_idx = order[i : i + batch_size]
                grad_W1 = np.zeros_like(self.W1)
                grad_b1 = np.zeros_like(self.b1)
                grad_W2 = np.zeros_like(self.W2)
                grad_b2 = np.zeros_like(self.b2)

                for idx in batch_idx:
                    ex = self._buffer[idx]
                    x = self._encode_pair(
                        ex["latent_a"],
                        ex["latent_b"],
                        ex["phase_a"],
                        ex["phase_b"],
                        ex["amp_a"],
                        ex["amp_b"],
                        ex["sal_a"],
                        ex["sal_b"],
                    )
                    y, h = self._forward(x)
                    norm_y = np.linalg.norm(y) + 1e-8
                    y_norm = y / norm_y

                    # Heuristic average (trust region target)
                    heuristic = 0.5 * (ex["latent_a"] + ex["latent_b"])
                    heuristic_norm = heuristic / (np.linalg.norm(heuristic) + 1e-8)

                    # Gradient accumulation
                    dy = np.zeros_like(y)

                    # L2 reg towards heuristic
                    dy += 2 * lambda_reg * (y_norm - heuristic_norm)

                    # Hinge: push merged to be at least as good as best parent
                    for q in ex["queries"]:
                        qn = q / (np.linalg.norm(q) + 1e-8)
                        sim_m = float(y_norm @ qn)
                        sim_a = float(ex["latent_a"] @ qn)
                        sim_b = float(ex["latent_b"] @ qn)
                        margin = max(sim_a, sim_b) + 0.05  # 0.05 margin
                        if sim_m < margin:
                            # Gradient: push y_norm towards qn
                            # d/dy (y_norm · qn) = (qn - (y_norm·qn) * y_norm) / norm_y
                            dy += -(qn - sim_m * y_norm) / norm_y

                    # Backprop through MLP
                    # dL/dW2 = dy · h^T,  dL/db2 = dy
                    grad_W2 += np.outer(dy, h)
                    grad_b2 += dy

                    # dL/dh = W2^T · dy
                    dh = self.W2.T @ dy
                    dh[h <= 0] = 0  # ReLU derivative

                    grad_W1 += np.outer(dh, x)
                    grad_b1 += dh

                    # Loss tracking (approximate)
                    reg_loss = lambda_reg * np.sum((y_norm - heuristic_norm) ** 2)
                    epoch_loss += float(reg_loss)
                    epoch_n += 1

                # Apply gradients
                bs = len(batch_idx)
                self.W2 -= lr * grad_W2 / bs
                self.b2 -= lr * grad_b2 / bs
                self.W1 -= lr * grad_W1 / bs
                self.b1 -= lr * grad_b1 / bs

            total_loss += epoch_loss / max(epoch_n, 1)
            n_batches += 1

        final_loss = total_loss / max(n_batches, 1)
        self._trained = True
        logger.info("LearnedConsolidator: training complete, loss=%.6f", final_loss)
        return final_loss

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "trained": self._trained,
        }

    def load_state(self, state: dict):
        self.latent_dim = state["latent_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.W1 = np.array(state["W1"], dtype=np.float32)
        self.b1 = np.array(state["b1"], dtype=np.float32)
        self.W2 = np.array(state["W2"], dtype=np.float32)
        self.b2 = np.array(state["b2"], dtype=np.float32)
        self._trained = state.get("trained", False)
