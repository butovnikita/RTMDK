"""rtmdk/production/tpr.py — Tensor Product Representations (Research Mode)."""

import numpy as np
from typing import Dict


class TensorProductRepresentation:
    """Role-filler binding via tensor products (Smolensky, 1990).

    Allows binding of roles and fillers in a distributed representation:
    V = Σ (role_i ⊗ filler_i)

    Usage (research mode):
        tpr = TensorProductRepresentation(role_dim=32, filler_dim=64)
        bound = tpr.bind("subject", "Alice")
        unbound = tpr.unbind(bound, "subject")  # → "Alice"
    """

    def __init__(self, role_dim: int = 32, filler_dim: int = 64):
        self.role_dim = role_dim
        self.filler_dim = filler_dim
        self.role_vectors: Dict[str, np.ndarray] = {}

    def _get_role_vector(self, role: str) -> np.ndarray:
        if role not in self.role_vectors:
            rng = np.random.default_rng(hash(role) % 2**32)
            v = rng.standard_normal(self.role_dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            self.role_vectors[role] = v
        return self.role_vectors[role]

    def bind(self, role: str, filler_embedding: np.ndarray) -> np.ndarray:
        """Bind a role to a filler via tensor product."""
        role_vec = self._get_role_vector(role)
        # Outer product: role ⊗ filler
        return np.outer(role_vec, filler_embedding).flatten()

    def unbind(self, bound: np.ndarray, role: str) -> np.ndarray:
        """Unbind to recover the filler from a bound representation."""
        role_vec = self._get_role_vector(role)
        # Reshape and project
        matrix = bound.reshape(self.role_dim, self.filler_dim)
        return matrix.T @ role_vec
