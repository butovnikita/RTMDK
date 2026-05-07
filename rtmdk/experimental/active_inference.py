"""rtmdk/production/active_inference.py — Active Inference & Curiosity Loop."""
import numpy as np
from typing import Dict, List, Optional, Any


class ActiveInferenceLoop:
    """Curiosity-driven exploration for autonomous RTMDK agents.

    Generates do(X) interventions to resolve uncertainty.
    Based on Friston's free energy principle.
    """

    def __init__(self, curiosity_weight: float = 0.1,
                 uncertainty_threshold: float = 0.3):
        self.curiosity_weight = curiosity_weight
        self.uncertainty_threshold = uncertainty_threshold
        self._interventions_generated = 0
        self._prediction_errors: List[float] = []

    def compute_uncertainty(self, memory_field) -> Dict[str, float]:
        """Compute uncertainty for each node (high = needs exploration)."""
        uncertainties = {}
        for nid, node in memory_field.nodes.items():
            # Uncertainty = 1 - salience (low salience = uncertain)
            uncertainties[nid] = 1.0 - node.salience
        return uncertainties

    def generate_intervention(self, memory_field) -> Optional[Dict[str, Any]]:
        """Generate a do(X) intervention to reduce uncertainty."""
        uncertainties = self.compute_uncertainty(memory_field)
        if not uncertainties:
            return None

        # Find most uncertain node
        most_uncertain = max(uncertainties, key=lambda k: uncertainties[k])
        if uncertainties[most_uncertain] < self.uncertainty_threshold:
            return None  # Uncertainty low enough, no intervention needed

        node = memory_field.nodes.get(most_uncertain)
        if not node:
            return None

        self._interventions_generated += 1
        return {
            "type": "active_query",
            "target_node": most_uncertain,
            "query": f"Tell me more about: {node.content.get('text', '')[:50]}",
            "expected_outcome": "reduce_uncertainty",
        }

    def record_prediction_error(self, error: float):
        """Record prediction error for curiosity tracking."""
        self._prediction_errors.append(error)

    def get_curiosity_drive(self) -> float:
        """Get current curiosity drive level."""
        if not self._prediction_errors:
            return 0.0
        return float(
            np.mean(self._prediction_errors[-100:])) * self.curiosity_weight
