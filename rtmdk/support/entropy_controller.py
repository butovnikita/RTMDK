"""rtmdk/support/entropy_controller.py — Information-theoretic capacity management."""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np


class EntropyController:
    """Manages memory field capacity via Shannon entropy of resonance responses.

    High entropy → field is noisy → trigger consolidation.
    Low entropy → field is stagnant → trigger exploration.
    """

    def __init__(
        self,
        high_entropy_threshold: float = 3.0,
        low_entropy_threshold: float = 0.5,
        consolidation_boost: float = 0.1,
        exploration_boost: float = 0.05,
        window_size: int = 50,
    ):
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold
        self.consolidation_boost = consolidation_boost
        self.exploration_boost = exploration_boost
        self.window_size = window_size
        self._response_history: List[float] = []
        self._salience_history: List[float] = []
        self._last_entropy: float = 0.0
        self._last_state: str = "normal"  # "noisy", "stagnant", "normal"

    def record_response(self, resonance_score: float, salience: float = 1.0):
        """Record a resonance response for entropy tracking."""
        self._response_history.append(max(0.0, resonance_score))
        self._salience_history.append(max(0.0, salience))
        # Keep bounded history
        if len(self._response_history) > self.window_size:
            self._response_history = self._response_history[-self.window_size :]
            self._salience_history = self._salience_history[-self.window_size :]

    def compute_entropy(self) -> float:
        """Compute Shannon entropy of resonance response distribution.

        H = -Σ p_i * log(p_i) where p_i are normalized resonance scores.
        Higher H = more uniform distribution (noisy field).
        Lower H = concentrated distribution (stagnant/focused field).
        """
        if len(self._response_history) < 3:
            return 0.0

        scores = np.array(self._response_history, dtype=np.float64)
        scores = scores[scores > 1e-8]  # Filter zeros
        if len(scores) < 2:
            return 0.0

        # Normalize to probability distribution
        total = scores.sum()
        if total < 1e-8:
            return 0.0
        probs = scores / total

        # Shannon entropy
        entropy: float = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by max possible entropy (log2(N))
        max_entropy = np.log2(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        self._last_entropy = float(normalized_entropy)
        return self._last_entropy

    def compute_salience_entropy(self) -> float:
        """Compute entropy of salience distribution across nodes."""
        if len(self._salience_history) < 3:
            return 0.0

        saliencies = np.array(self._salience_history, dtype=np.float64)
        saliencies = saliencies[saliencies > 1e-8]
        if len(saliencies) < 2:
            return 0.0

        total = saliencies.sum()
        if total < 1e-8:
            return 0.0
        probs = saliencies / total
        entropy: float = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def get_state(self) -> str:
        """Get current field state based on entropy.

        Returns:
            "noisy" — high entropy, too many competing activations
            "stagnant" — low entropy, field is not exploring
            "normal" — balanced
        """
        entropy = self.compute_entropy()
        if entropy > self.high_entropy_threshold:
            self._last_state = "noisy"
        elif entropy < self.low_entropy_threshold:
            self._last_state = "stagnant"
        else:
            self._last_state = "normal"
        return self._last_state

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered due to noise."""
        return self.get_state() == "noisy"

    def should_explore(self) -> bool:
        """Check if exploration should be triggered due to stagnation."""
        return self.get_state() == "stagnant"

    def get_consolidation_multiplier(self) -> float:
        """Return multiplier for consolidation threshold.

        >1.0 when noisy (lower effective threshold → more consolidation).
        <1.0 when stagnant (higher effective threshold → less consolidation).
        """
        state = self.get_state()
        if state == "noisy":
            return 1.0 + self.consolidation_boost
        elif state == "stagnant":
            return 1.0 - self.consolidation_boost
        return 1.0

    def get_decay_multiplier(self) -> float:
        """Return multiplier for decay rate.

        <1.0 when stagnant (faster decay → clear space for new nodes).
        >1.0 when noisy (slower decay → preserve important nodes).
        """
        state = self.get_state()
        if state == "stagnant":
            return 1.0 - self.exploration_boost
        elif state == "noisy":
            return 1.0 + self.exploration_boost
        return 1.0

    def get_info(self) -> Dict[str, Any]:
        """Get entropy info for dashboard/stats."""
        return {
            "entropy": self._last_entropy,
            "state": self._last_state,
            "n_responses": len(self._response_history),
            "consolidation_multiplier": self.get_consolidation_multiplier(),
            "decay_multiplier": self.get_decay_multiplier(),
            "should_consolidate": self.should_consolidate(),
            "should_explore": self.should_explore(),
        }

    def get_state_dict(self) -> Dict:
        return {
            "response_history": self._response_history,
            "salience_history": self._salience_history,
            "last_entropy": self._last_entropy,
            "last_state": self._last_state,
        }

    def load_state_dict(self, data: Dict):
        self._response_history = data.get("response_history", [])
        self._salience_history = data.get("salience_history", [])
        self._last_entropy = data.get("last_entropy", 0.0)
        self._last_state = data.get("last_state", "normal")
