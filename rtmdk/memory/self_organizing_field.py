"""Self-Organizing Tokenizer + Embedding Field for RTMDK.

Replaces static external embeddings with a dynamic field that:
1. Builds vocabulary from bytes up to subtokens via field co-occurrence.
2. Learns embeddings via online contrastive Hebbian rules.
3. Synchronizes with SSM dynamics for smooth latent trajectories.

Architecture: token_dim != latent_dim supported via learnable projection.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from rtmdk.engines.ssm_dynamics import SSMDynamics

logger = logging.getLogger(__name__)


class SOTokenizer:
    """Self-organizing tokenizer: bytes -> subtokens, driven by field co-occurrence.

    Supports token_dim != latent_dim via a learnable projection matrix.
    This allows high-capacity token representations (e.g., 256d) while
    keeping the field space lightweight (e.g., 64d).
    """

    def __init__(
        self,
        latent_dim: int,
        token_dim: Optional[int] = None,
        max_vocab: int = 4096,
        initial_byte_vocab: int = 256,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.token_dim = token_dim or latent_dim
        self.max_vocab = max_vocab
        self.initial_byte_vocab = initial_byte_vocab
        self.next_token_id = initial_byte_vocab
        self._rng = np.random.default_rng(seed)

        self.token_embeddings: Dict[int, np.ndarray] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.cooccurrence: Dict[Tuple[int, int], float] = defaultdict(float)

        # Learnable projection: token_dim -> latent_dim
        if self.token_dim == self.latent_dim:
            self.projection = np.eye(self.token_dim, dtype=np.float32)
        else:
            self.projection = self._rng.standard_normal(
                (self.token_dim, self.latent_dim)
            ).astype(np.float32) * 0.1
            # Orthogonal initialization for better conditioning
            if self.token_dim >= self.latent_dim:
                q, _ = np.linalg.qr(self.projection)
                self.projection = q[:, :self.latent_dim].astype(np.float32)

        self._init_byte_embeddings()

    def _init_byte_embeddings(self):
        for i in range(self.initial_byte_vocab):
            emb = self._rng.standard_normal(self.token_dim).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self.token_embeddings[i] = emb

    def encode(self, text: str) -> List[int]:
        """Greedy longest-match encoding using current merge table."""
        if not text:
            return []
        raw_bytes = list(text.encode("utf-8"))
        if not self.merges:
            return raw_bytes

        tokens = raw_bytes[:]
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in self.merges:
                    new_tokens.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to a string."""
        if not tokens:
            return ""
        byte_seq = bytearray()
        stack = list(tokens)
        while stack:
            tid = stack.pop(0)
            if tid < self.initial_byte_vocab:
                byte_seq.append(tid)
            else:
                found = False
                for (a, b), merged_id in self.merges.items():
                    if merged_id == tid:
                        stack.insert(0, b)
                        stack.insert(0, a)
                        found = True
                        break
                if not found:
                    logger.warning(f"SOT decode: token {tid} has no merge rule")
                    byte_seq.append(tid % self.initial_byte_vocab)
        return byte_seq.decode("utf-8", errors="replace")

    def embed(self, tokens: List[int]) -> np.ndarray:
        """Mean-pool token embeddings and project to latent_dim."""
        if not tokens:
            return np.zeros(self.latent_dim, dtype=np.float32)
        vecs = [self.token_embeddings[t] for t in tokens if t in self.token_embeddings]
        if not vecs:
            return np.zeros(self.latent_dim, dtype=np.float32)
        pooled = np.mean(vecs, axis=0).astype(np.float32)  # (token_dim,)
        latent = pooled @ self.projection  # (latent_dim,)
        # Normalize latent output
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent /= norm
        return latent.astype(np.float32)

    def record_cooccurrence(self, tokens: List[int], weight: float = 1.0):
        """Record adjacent token co-occurrences for merge proposals."""
        if len(tokens) < 2:
            return
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            self.cooccurrence[(a, b)] += weight

    def propose_merges(self, n: int) -> List[Tuple[int, int]]:
        """Return top-N merge candidates by co-occurrence score."""
        if not self.cooccurrence:
            return []
        sorted_pairs = sorted(self.cooccurrence.items(), key=lambda kv: kv[1], reverse=True)
        return [pair for pair, _ in sorted_pairs[:n]]

    def merge(self, pair: Tuple[int, int]):
        """Execute a merge: create new token embedding as weighted average."""
        if len(self.token_embeddings) >= self.max_vocab:
            raise RuntimeError(f"Max vocab size {self.max_vocab} reached, cannot merge")
        a, b = pair
        if a not in self.token_embeddings or b not in self.token_embeddings:
            raise ValueError(f"Invalid pair {pair}: missing embeddings")
        weight = self.cooccurrence.get(pair, 1.0)
        new_id = self.next_token_id
        self.next_token_id += 1
        wa = weight
        wb = weight
        new_emb = (wa * self.token_embeddings[a] + wb * self.token_embeddings[b]) / (wa + wb)
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb /= norm
        self.token_embeddings[new_id] = new_emb.astype(np.float32)
        self.merges[pair] = new_id
        logger.debug(f"SOT merge: {pair} -> {new_id} (vocab={len(self.token_embeddings)})")

    def update_projection(
        self,
        positive_pairs: List[Tuple[int, int]],
        negative_pairs: List[Tuple[int, int]] = None,
        lr: float = 0.001,
    ):
        """Update projection matrix via contrastive Hebbian rule.

        Pulls projected positive pairs closer, pushes negatives apart.
        """
        if self.token_dim == self.latent_dim:
            return  # Identity projection, nothing to learn
        negative_pairs = negative_pairs or []
        delta = np.zeros_like(self.projection)
        for i, j in positive_pairs:
            if i not in self.token_embeddings or j not in self.token_embeddings:
                continue
            t_i = self.token_embeddings[i]  # (token_dim,)
            t_j = self.token_embeddings[j]  # (token_dim,)
            p_i = t_i @ self.projection     # (latent_dim,)
            p_j = t_j @ self.projection     # (latent_dim,)
            error = p_j - p_i
            # Gradient: d(loss)/dW = t_i^T * error
            delta += lr * np.outer(t_i, error)
        for i, k in negative_pairs:
            if i not in self.token_embeddings or k not in self.token_embeddings:
                continue
            t_i = self.token_embeddings[i]
            t_k = self.token_embeddings[k]
            p_i = t_i @ self.projection
            p_k = t_k @ self.projection
            error = p_k - p_i
            delta -= lr * 0.1 * np.outer(t_i, error)
        self.projection += delta
        # Normalize columns to prevent explosion
        norms = np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.projection /= np.maximum(norms, 1e-8)

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "token_dim": self.token_dim,
            "max_vocab": self.max_vocab,
            "initial_byte_vocab": self.initial_byte_vocab,
            "next_token_id": self.next_token_id,
            "token_embeddings": {k: v.tolist() for k, v in self.token_embeddings.items()},
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "cooccurrence": {f"{a},{b}": v for (a, b), v in self.cooccurrence.items()},
            "projection": self.projection.tolist(),
        }

    def load_state(self, state: dict):
        self.latent_dim = state.get("latent_dim", self.latent_dim)
        self.token_dim = state.get("token_dim", self.token_dim)
        self.max_vocab = state.get("max_vocab", self.max_vocab)
        self.initial_byte_vocab = state.get("initial_byte_vocab", self.initial_byte_vocab)
        self.next_token_id = state.get("next_token_id", self.initial_byte_vocab)
        self.token_embeddings = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in state.get("token_embeddings", {}).items()
        }
        self.merges = {}
        for k, v in state.get("merges", {}).items():
            a_str, b_str = k.split(",")
            self.merges[(int(a_str), int(b_str))] = int(v)
        self.cooccurrence = defaultdict(float)
        for k, v in state.get("cooccurrence", {}).items():
            a_str, b_str = k.split(",")
            self.cooccurrence[(int(a_str), int(b_str))] = float(v)
        if "projection" in state:
            self.projection = np.array(state["projection"], dtype=np.float32)
            assert self.projection.shape == (self.token_dim, self.latent_dim), \
                f"Projection shape mismatch: {self.projection.shape} vs ({self.token_dim}, {self.latent_dim})"


class ContrastiveHebbian:
    """Online contrastive Hebbian learning for embeddings."""

    def __init__(self, lr: float = 0.01, neg_ratio: float = 0.2, temperature: float = 0.1):
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.temperature = temperature

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def update(
        self,
        embeddings: Dict[int, np.ndarray],
        positives: List[int],
        negatives: List[int],
    ):
        """Update token embeddings in-place via contrastive Hebbian rule.

        Positives pull each other closer; negatives are pushed away from positives.
        Negatives do NOT get pulled toward positives.
        """
        if len(positives) < 2 and not negatives:
            return
        for i in positives:
            if i not in embeddings:
                continue
            delta = np.zeros_like(embeddings[i])
            for j in positives:
                if i == j or j not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[j])
                delta += self.lr * sim * (embeddings[j] - embeddings[i])
            for k in negatives:
                if k not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[k])
                delta -= self.lr * 0.1 * sim * (embeddings[k] - embeddings[i])
            embeddings[i] += delta
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        for k in negatives:
            if k not in embeddings:
                continue
            delta = np.zeros_like(embeddings[k])
            for i in positives:
                if i not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[k], embeddings[i])
                delta -= self.lr * 0.1 * sim * (embeddings[i] - embeddings[k])
            embeddings[k] += delta
            norm = np.linalg.norm(embeddings[k])
            if norm > 0:
                embeddings[k] /= norm

    def field_update(
        self,
        node_embeddings: np.ndarray,
        positives: List[int],
        negatives: List[int],
    ):
        """Update node latent positions in-place (node_embeddings is (N, latent_dim)).

        Positives pull each other closer; negatives are pushed away from positives.
        """
        if node_embeddings.ndim != 2:
            return
        n_nodes = node_embeddings.shape[0]
        if n_nodes == 0:
            return
        valid_pos = [p for p in positives if 0 <= p < n_nodes]
        valid_neg = [n for n in negatives if 0 <= n < n_nodes]
        if len(valid_pos) < 2 and not valid_neg:
            return
        for i in valid_pos:
            delta = np.zeros_like(node_embeddings[i])
            for j in valid_pos:
                if i == j:
                    continue
                sim = self._cosine_sim(node_embeddings[i], node_embeddings[j])
                delta += self.lr * sim * (node_embeddings[j] - node_embeddings[i])
            for k in valid_neg:
                sim = self._cosine_sim(node_embeddings[i], node_embeddings[k])
                delta -= self.lr * 0.1 * sim * (node_embeddings[k] - node_embeddings[i])
            node_embeddings[i] += delta
            norm = np.linalg.norm(node_embeddings[i])
            if norm > 0:
                node_embeddings[i] /= norm
        for k in valid_neg:
            delta = np.zeros_like(node_embeddings[k])
            for i in valid_pos:
                sim = self._cosine_sim(node_embeddings[k], node_embeddings[i])
                delta -= self.lr * 0.1 * sim * (node_embeddings[i] - node_embeddings[k])
            node_embeddings[k] += delta
            norm = np.linalg.norm(node_embeddings[k])
            if norm > 0:
                node_embeddings[k] /= norm


class EmbeddingFieldSSM:
    """Bridges SSMDynamics with the embedding field for smooth trajectories."""

    def __init__(self, latent_dim: int, tokenizer: SOTokenizer, diagonal: bool = True):
        self.latent_dim = latent_dim
        self.tokenizer = tokenizer
        self.ssm = SSMDynamics(
            state_dim=latent_dim,
            input_dim=latent_dim,
            output_dim=latent_dim,
            n_nodes=1,
            dt=0.1,
            learnable=False,
            diagonal=diagonal,
        )

    def step(self, token_ids: List[int], field_state: np.ndarray) -> np.ndarray:
        """Compute momentum from SSM given current token sequence and field state."""
        if field_state.ndim != 1 or field_state.shape[0] != self.latent_dim:
            field_state = np.zeros(self.latent_dim, dtype=np.float32)
        u = self.tokenizer.embed(token_ids)
        if u.ndim != 1:
            u = u.reshape(-1)
        h = field_state.reshape(1, -1)
        u = u.reshape(1, -1)
        _, y = self.ssm.step(h, u)
        if y.ndim != 1:
            y = y.reshape(-1)
        return y.astype(np.float32)

    def sync_embeddings(self, token_ids: List[int], momentum: np.ndarray):
        """Add SSM momentum to token embeddings via projection."""
        if momentum.ndim != 1 or momentum.shape[0] != self.latent_dim:
            return
        for tid in token_ids:
            if tid not in self.tokenizer.token_embeddings:
                continue
            # Apply momentum in latent space, backpropagate through projection
            # delta_token = momentum @ projection.T
            delta_token = momentum @ self.tokenizer.projection.T
            self.tokenizer.token_embeddings[tid] += delta_token * 0.01
            norm = np.linalg.norm(self.tokenizer.token_embeddings[tid])
            if norm > 0:
                self.tokenizer.token_embeddings[tid] /= norm
