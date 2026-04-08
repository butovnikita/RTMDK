"""Incremental PCA projection for RTMDK."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class IncPCAProjection:
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.001, update_freq: int = 50, l2_reg: float = 0.0001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.update_freq = update_freq
        self.l2_reg = l2_reg
        self.projection = np.random.randn(input_dim, latent_dim).astype(np.float32) * 0.1
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.buffer: List[NDArray] = []
        self.n_samples = 0
        self._try_sklearn()

    def _try_sklearn(self):
        try:
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(n_components=self.latent_dim, batch_size=min(64, self.update_freq))
            self.use_sklearn = True
            self._ipca_fitted = False
        except ImportError:
            self.use_sklearn = False

    def update(self, embedding: NDArray) -> NDArray:
        self.n_samples += 1
        self.buffer.append(embedding.copy())
        if len(self.buffer) >= self.update_freq:
            batch = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            if self.use_sklearn:
                self.ipca.partial_fit(batch)
                self._ipca_fitted = True
                self.projection = self.ipca.components_.T.astype(np.float32)
                self.mean = self.ipca.mean_.astype(np.float32)
            else:
                alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                self.mean += alpha * (batch.mean(axis=0) - self.mean)
                for emb in batch:
                    centered = emb - self.mean
                    latent = centered @ self.projection
                    reconstructed = latent @ self.projection.T
                    error = centered - reconstructed
                    self.projection += alpha * (np.outer(centered, latent) - np.outer(error, latent))
                    self.projection -= alpha * self.l2_reg * self.projection
                    norm = np.linalg.norm(self.projection, axis=0, keepdims=True)
                    self.projection /= np.maximum(norm, 1e-8)
        return self.project(embedding)

    def project(self, embedding: NDArray) -> NDArray:
        if self.use_sklearn and self._ipca_fitted:
            return self.ipca.transform(embedding.reshape(1, -1))[0].astype(np.float32)
        return ((embedding - self.mean) @ self.projection).astype(np.float32)

    def get_state(self) -> Dict:
        return {"projection": self.projection.tolist(), "mean": self.mean.tolist(), "n_samples": self.n_samples, "use_sklearn": self.use_sklearn}

    def set_matrix(self, matrix: NDArray):
        """Set projection matrix directly (for import/initialization)."""
        assert matrix.shape == (self.input_dim, self.latent_dim), \
            f"Expected shape ({self.input_dim}, {self.latent_dim}), got {matrix.shape}"
        self.projection = matrix.astype(np.float32)
        if self.use_sklearn:
            self._ipca_fitted = True
            self.ipca = IncrementalPCA(n_components=self.latent_dim)
            self.ipca.components_ = self.projection.T
            self.ipca.mean_ = self.mean
            self.ipca.n_samples_seen_ = max(self.n_samples, 1)

    def load_state(self, state: Dict):
        self.projection = np.array(state["projection"], dtype=np.float32)
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.n_samples = state.get("n_samples", 0)
        if self.use_sklearn and state.get("use_sklearn"):
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(n_components=self.latent_dim)
            self.ipca.mean_ = self.mean
            self.ipca.components_ = self.projection.T
            self.ipca.n_samples_seen_ = self.n_samples
            self._ipca_fitted = True
