"""
rtmdk/memory/conformal.py — Inductive Conformal Prediction (ICP) for retrieval confidence.

Provides statistical guarantees that the true relevant result is contained in the
prediction set with probability >= 1 - alpha.

Mathematics:
- Calibration set: {(x_i, y_i, s_i)} where s_i = resonance(x_i, y_i) for relevant y_i
- Non-conformity score: alpha_i = 1 - s_i  (lower = more conforming)
- Quantile: q_hat = quantile({alpha_i}, ceil((n+1)*(1-alpha))/n)
- Prediction set: C(x_{n+1}) = {y : s(x_{n+1}, y) >= 1 - q_hat}
"""

import numpy as np
from typing import List, Tuple


class ConformalCalibrator:
    """Inductive Conformal Prediction calibrator for retrieval scores."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        # relevance scores from calibration
        self.calibration_scores: List[float] = []

    def fit(self, scores: List[float]) -> None:
        """Replace calibration scores with a held-out set."""
        self.calibration_scores = list(scores)

    def add_sample(self, score: float) -> None:
        """Append a single relevance score to the calibration set."""
        self.calibration_scores.append(float(score))

    def get_threshold(self) -> float:
        """Compute conformal score threshold.

        Returns the minimum resonance score a result must have to be included
        in the prediction set. Guarantees coverage >= 1 - alpha on the
        calibration distribution.

        Uses exact order-statistic formula instead of np.quantile to avoid
        interpolation ambiguity for small n.
        """
        n = len(self.calibration_scores)
        if n == 0:
            return 0.0
        # Shafer-Vovk ICP: k = ceil((n+1)*(1-alpha))
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            return 0.0  # not enough calibration data: include everything
        sorted_scores = np.sort(self.calibration_scores)
        # k-th largest score = element at index n - k in ascending array
        return max(0.0, float(sorted_scores[n - k]))

    def predict(self, scores: List[float], node_ids: List[str]) -> Tuple[List[str], float, float]:
        """Return prediction set, confidence level, and threshold.

        Args:
            scores: resonance scores for candidate results
            node_ids: corresponding node identifiers

        Returns:
            (prediction_set, confidence, threshold)
        """
        threshold = self.get_threshold()
        prediction_set = [nid for nid, s in zip(node_ids, scores) if s >= threshold]
        confidence = 1.0 - self.alpha
        return prediction_set, confidence, threshold

    @property
    def n_calibrated(self) -> int:
        return len(self.calibration_scores)
