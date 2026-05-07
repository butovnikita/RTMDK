"""rtmdk/engines/predictive.py"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray


class PredictiveCodingModel:
    """Predictive coding / active inference for field dynamics."""

    def __init__(
            self,
            latent_dim: int,
            hidden_dim: int = 128,
            lr: float = 0.01):
        self.latent_dim = latent_dim
        self.state_dim = latent_dim * 4
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.W = np.random.randn(
            self.state_dim,
            self.state_dim).astype(
            np.float32) * 0.01
        self.b = np.zeros(self.state_dim, dtype=np.float32)
        self._complexity_weight = 0.01

    def predict(self, state: NDArray) -> NDArray:
        return (state @ self.W + self.b).astype(np.float32)

    def compute_free_energy(
            self,
            state_t: NDArray,
            state_t1: NDArray) -> float:
        pred = self.predict(state_t)
        prediction_error = float(np.mean((pred - state_t1) ** 2))
        complexity = float(np.mean(self.W ** 2)) * self._complexity_weight
        return prediction_error + complexity

    def update(
            self,
            state_t: NDArray,
            state_t1: NDArray,
            lr: Optional[float] = None):
        lr = lr or self.lr
        pred = self.predict(state_t)
        error = pred - state_t1
        self.W -= lr * np.outer(state_t, error)
        self.b -= lr * error
        self.W *= (1.0 - lr * self._complexity_weight)

    def get_state(self) -> Dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "lr": self.lr}

    def load_state(self, state: Dict):
        self.W = np.array(state["W"], dtype=np.float32)
        self.b = np.array(state["b"], dtype=np.float32)
        self.lr = state.get("lr", self.lr)
