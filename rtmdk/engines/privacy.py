"""rtmdk/engines/privacy.py"""
from __future__ import annotations
from typing import Dict
import math
import numpy as np
from numpy.typing import NDArray


class DifferentialPrivacy:
    """Differential privacy for federated learning."""

    def __init__(self, epsilon: float = 2.0, delta: float = 1e-5, max_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self._privacy_spent = 0.0
        self._num_updates = 0

    def clip_update(self, update: NDArray) -> NDArray:
        norm = np.linalg.norm(update)
        if norm > self.max_norm:
            return (update * self.max_norm / norm).astype(np.float32)
        return update

    def add_noise(self, update: NDArray, sensitivity: float = 1.0) -> NDArray:
        noise_std = self.compute_noise_multiplier(1) * sensitivity
        noise = np.random.randn(*update.shape).astype(np.float32) * noise_std
        return (update + noise).astype(np.float32)

    def compute_noise_multiplier(self, n_samples: int) -> float:
        if self.epsilon <= 0:
            return float('inf')
        sigma = math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma / max(math.sqrt(n_samples), 1.0)

    def get_privacy_spent(self) -> float:
        return self._privacy_spent

    def record_update(self, n_samples: int = 1):
        self._num_updates += n_samples
        k = self._num_updates
        self._privacy_spent = self.epsilon * math.sqrt(2 * k * math.log(1 / self.delta))

    def get_state(self) -> Dict:
        return {"epsilon": self.epsilon, "delta": self.delta, "max_norm": self.max_norm,
                "privacy_spent": self._privacy_spent, "num_updates": self._num_updates}

    def load_state(self, state: Dict):
        self.epsilon = state.get("epsilon", self.epsilon)
        self.delta = state.get("delta", self.delta)
        self.max_norm = state.get("max_norm", self.max_norm)
        self._privacy_spent = state.get("privacy_spent", 0.0)
        self._num_updates = state.get("num_updates", 0)
