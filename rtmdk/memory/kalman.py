"""
rtmdk/memory/kalman.py — Kalman Filtering on Manifold (Riemannian EKF).

Simplified EKF for node position uncertainty tracking.
- Prediction: add process noise Q to covariance
- Update: Kalman gain on tangent space, then exp_map back to manifold
- Diagonal approximation available to cap memory (64 floats vs 4096 per node)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from rtmdk.memory.geometry import exp_map_poincare, log_map_poincare


class KalmanFilter:
    """Simplified Kalman filter for RTMDK node positions."""

    def __init__(
        self,
        latent_dim: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        init_variance: float = 1.0,
        diagonal_approx: bool = True,
        hyperbolic: bool = False,
        ball_radius: float = 0.85,
    ):
        self.latent_dim = latent_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.init_variance = init_variance
        self.diagonal_approx = diagonal_approx
        self.hyperbolic = hyperbolic
        self.ball_radius = ball_radius
        self._I: np.ndarray = np.eye(latent_dim, dtype=np.float32)
        self._Q = np.eye(latent_dim, dtype=np.float32) * process_noise
        self._R = np.eye(latent_dim, dtype=np.float32) * measurement_noise

    def init_covariance(self) -> NDArray:
        """Return initial covariance for a new node."""
        if self.diagonal_approx:
            return np.full(self.latent_dim, self.init_variance, dtype=np.float32)
        return np.eye(self.latent_dim, dtype=np.float32) * self.init_variance

    def predict(self, covariance: NDArray) -> NDArray:
        """Prediction step: add process noise."""
        if self.diagonal_approx:
            return covariance + self.process_noise
        return covariance + self._Q

    def update(
        self,
        x: NDArray,
        z: NDArray,
        covariance: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Update step: fuse position x with measurement z.

        Returns (updated_position, updated_covariance).
        """
        if self.diagonal_approx:
            return self._update_diagonal(x, z, covariance)
        return self._update_full(x, z, covariance)

    def _update_diagonal(self, x: NDArray, z: NDArray, cov: NDArray) -> Tuple[NDArray, NDArray]:
        """Scalar Kalman update per dimension."""
        # Innovation variance per dimension
        S = cov + self.measurement_noise
        # Kalman gain
        K = cov / (S + 1e-10)
        if self.hyperbolic:
            # Innovation on tangent space
            y = log_map_poincare(z, x, self.ball_radius)
            dx = K * y
            x_new = exp_map_poincare(dx, x, self.ball_radius)
        else:
            y = z - x
            x_new = x + K * y
        cov_new = (1.0 - K) * cov
        return x_new.astype(np.float32), cov_new.astype(np.float32)

    def _update_full(self, x: NDArray, z: NDArray, cov: NDArray) -> Tuple[NDArray, NDArray]:
        """Full vector Kalman update."""
        # Kalman gain: K = Σ H^T (H Σ H^T + R)^{-1}
        # H = I (direct observation of position)
        S = cov + self._R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = cov @ S_inv
        if self.hyperbolic:
            y = log_map_poincare(z, x, self.ball_radius)
            dx = K @ y
            x_new = exp_map_poincare(dx, x, self.ball_radius)
        else:
            y = z - x
            x_new = x + K @ y
        cov_new = (self._I - K) @ cov
        return x_new.astype(np.float32), cov_new.astype(np.float32)

    def merge_covariance(self, cov_a: NDArray, cov_b: NDArray) -> NDArray:
        """Merge two covariances (information-weighted average).

        Used when two nodes are consolidated.
        """
        if self.diagonal_approx:
            # Information form: 1/Σ = 1/Σ_a + 1/Σ_b
            merged = 1.0 / (1.0 / (cov_a + 1e-10) + 1.0 / (cov_b + 1e-10))
            return merged.astype(np.float32)
        # Full matrix: information form
        try:
            inv_a = np.linalg.inv(cov_a)
        except np.linalg.LinAlgError:
            inv_a = np.linalg.pinv(cov_a)
        try:
            inv_b = np.linalg.inv(cov_b)
        except np.linalg.LinAlgError:
            inv_b = np.linalg.pinv(cov_b)
        info = inv_a + inv_b
        try:
            merged = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            merged = np.linalg.pinv(info)
        return merged.astype(np.float32)

    def uncertainty_weight(self, covariance: NDArray) -> float:
        """Return scalar weight factor for retrieval: 1 / (1 + trace(Σ))."""
        if self.diagonal_approx:
            tr = float(np.sum(covariance))
        else:
            tr = float(np.trace(covariance))
        return 1.0 / (1.0 + tr)
