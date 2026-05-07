"""Incremental PCA projection for RTMDK."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class IncPCAProjection:
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            lr: float = 0.001,
            update_freq: int = 50,
            l2_reg: float = 0.0001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.update_freq = update_freq
        self.l2_reg = l2_reg
        self.projection = np.random.randn(
            input_dim, latent_dim).astype(
            np.float32) * 0.1
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.buffer: List[NDArray] = []
        self.n_samples = 0
        self._try_sklearn()

    def _try_sklearn(self):
        try:
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(
                n_components=self.latent_dim, batch_size=min(
                    64, self.update_freq))
            self.use_sklearn = True
            self._ipca_fitted = False
            self._ipca_error = None  # Store any sklearn errors for fallback
        except ImportError:
            self.use_sklearn = False
            self._ipca_fitted = False
            self._ipca_error = "sklearn not installed"

    def update(self, embedding: NDArray) -> NDArray:
        self.n_samples += 1
        self.buffer.append(embedding.copy())
        if len(self.buffer) >= self.update_freq:
            batch = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            if self.use_sklearn:
                try:
                    self.ipca.partial_fit(batch)
                    # Only mark as fitted if we have enough samples
                    if self.ipca.n_samples_seen_ >= self.latent_dim:
                        self._ipca_fitted = True
                        self.projection = self.ipca.components_.T.astype(
                            np.float32)
                        self.mean = self.ipca.mean_.astype(np.float32)
                    else:
                        # Not enough samples yet, use manual update
                        self._ipca_fitted = False
                        alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                        self.mean += alpha * (batch.mean(axis=0) - self.mean)
                except Exception as e:
                    self._ipca_error = str(e)
                    self._ipca_fitted = False
            else:
                alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                self.mean += alpha * (batch.mean(axis=0) - self.mean)
                for emb in batch:
                    centered = emb - self.mean
                    latent = centered @ self.projection
                    reconstructed = latent @ self.projection.T
                    error = centered - reconstructed
                    self.projection += alpha * \
                        (np.outer(centered, latent) - np.outer(error, latent))
                    self.projection -= alpha * self.l2_reg * self.projection
                    norm = np.linalg.norm(
                        self.projection, axis=0, keepdims=True)
                    self.projection /= np.maximum(norm, 1e-8)
        return self.project(embedding)

    def project(self, embedding: NDArray) -> NDArray:
        # Only use sklearn transform if properly fitted
        if self.use_sklearn and self._ipca_fitted and self._ipca_error is None:
            try:
                return self.ipca.transform(
                    embedding.reshape(1, -1))[0].astype(np.float32)
            except Exception as e:
                logger.warning(
                    f"IncrementalPCA projection failed, falling back to manual: {e}")
                self._ipca_fitted = False
        # Fallback to manual projection — track reconstruction error to detect
        # divergence
        embedding - self.mean
        proj_norm = np.linalg.norm(self.projection)
        if proj_norm < 1e-8:
            logger.warning(
                "IncPCAProjection: projection matrix ill-conditioned, may diverge")
        return ((embedding - self.mean) @ self.projection).astype(np.float32)

    def get_state(self) -> Dict:
        return {
            "projection": self.projection.tolist(),
            "mean": self.mean.tolist(),
            "n_samples": self.n_samples,
            "use_sklearn": self.use_sklearn,
            "ipca_fitted": self._ipca_fitted,
        }

    def set_matrix(self, matrix: NDArray):
        """Set projection matrix directly (for import/initialization)."""
        assert matrix.shape == (self.input_dim, self.latent_dim), \
            f"Expected shape ({self.input_dim}, {self.latent_dim}), got {matrix.shape}"
        self.projection = matrix.astype(np.float32)
        # Don't try to initialize sklearn here - it's safer to use manual
        # projection
        self._ipca_fitted = False
        self.use_sklearn = False

    def load_state(self, state: Dict):
        self.projection = np.array(state["projection"], dtype=np.float32)
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.n_samples = state.get("n_samples", 0)
        self._ipca_fitted = state.get("ipca_fitted", False)
        # Don't re-initialize sklearn from state - use manual projection
        self.use_sklearn = False
