"""rtmdk/engines/privacy.py"""
from __future__ import annotations
from typing import Dict
import math
import numpy as np
from numpy.typing import NDArray


class DifferentialPrivacy:
    """Differential privacy for federated learning."""

    def __init__(self, epsilon: float = 2.0, delta: float = 1e-5, max_norm: float = 1.0, seed: int = 42):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self._rng = np.random.default_rng(seed)
        self._privacy_spent = 0.0
        self._num_updates = 0

    def clip_update(self, update: NDArray) -> NDArray:
        """Clip update to max_norm."""
        norm = np.linalg.norm(update)
        if norm > self.max_norm:
            return (update * self.max_norm / norm).astype(np.float32)
        return update

    def add_noise(self, update: NDArray, sensitivity: float = 1.0) -> NDArray:
        """Add calibrated Gaussian noise."""
        noise_std = self.compute_noise_multiplier(sensitivity)
        noise = self._rng.standard_normal(update.shape).astype(np.float32) * noise_std
        return (update + noise).astype(np.float32)

    def compute_noise_multiplier(self, sensitivity: float = 1.0) -> float:
        """Compute noise multiplier for given privacy budget."""
        if self.epsilon <= 0:
            return float('inf')
        # Bug #10 FIX: Gaussian mechanism — sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        # The sensitivity (Delta_f) MUST be multiplied — without it, DP guarantees don't hold
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def get_privacy_spent(self) -> float:
        """Return cumulative privacy budget spent."""
        return self._privacy_spent

    def record_update(self, n_samples: int = 1):
        """Record that an update was made (track privacy budget)."""
        self._num_updates += n_samples
        # Bug #11 FIX: Advanced composition — use per-mechanism epsilon correctly
        # epsilon_total = sqrt(2 * k * ln(1/delta')) * epsilon_per_mechanism
        k = self._num_updates
        self._privacy_spent = math.sqrt(2 * k * math.log(1 / self.delta)) * self.epsilon

    def get_state(self) -> Dict:
        return {"epsilon": self.epsilon, "delta": self.delta, "max_norm": self.max_norm,
                "privacy_spent": self._privacy_spent, "num_updates": self._num_updates,
                "seed": int(self._rng.integers(0, 2**31))}

    def load_state(self, state: Dict):
        self.epsilon = state.get("epsilon", self.epsilon)
        self.delta = state.get("delta", self.delta)
        self.max_norm = state.get("max_norm", self.max_norm)
        self._privacy_spent = state.get("privacy_spent", 0.0)
        self._num_updates = state.get("num_updates", 0)
        seed = state.get("seed", 42)
        self._rng = np.random.default_rng(seed)
