"""Meta-memory evaluator for RTMDK."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    pass


class MetaMemoryEvaluator:
    """Evaluates recall accuracy, memory age, and self-reflection."""

    def __init__(self, recall_threshold: float = 0.6, age_factor: float = 0.001, reflection_freq: int = 100):
        self.recall_threshold = recall_threshold
        self.age_factor = age_factor
        self.reflection_freq = reflection_freq
        self._recall_history: deque = deque(maxlen=100)
        self._reflection_log: List[Dict] = []
        self._step_counter = 0

    def record_recall(self, query_text: str, result_score: float, node_age: float = 0.0) -> Dict[str, float]:
        """Record a recall event and compute accuracy metrics."""
        self._recall_history.append(result_score)
        age_penalty = 1.0 - min(1.0, node_age * self.age_factor)
        adjusted_score = result_score * age_penalty
        return {
            "raw_score": result_score,
            "age_penalty": age_penalty,
            "adjusted_score": adjusted_score,
            "node_age": node_age,
        }

    def evaluate_recall_accuracy(self) -> float:
        if not self._recall_history:
            return 1.0
        return float(np.mean(self._recall_history))

    def should_reflect(self) -> bool:
        self._step_counter += 1
        return self._step_counter % self.reflection_freq == 0

    def self_reflect(self, field: Any) -> Dict[str, Any]:
        """Introspective analysis of memory field health."""
        recall_acc = self.evaluate_recall_accuracy()
        n_nodes = len(field.nodes) if hasattr(field, "nodes") else 0
        n_consolidations = field.stats.get("consolidations", 0) if hasattr(field, "stats") else 0
        false_merges = field.stats.get("false_merges", 0) if hasattr(field, "stats") else 0

        recommendations = []
        if recall_acc < self.recall_threshold:
            recommendations.append("lower_consolidation_threshold")
        if n_consolidations > 0 and false_merges > n_consolidations * 0.2:
            recommendations.append("increase_tension_threshold")
        if n_nodes > 1000:
            recommendations.append("trigger_crystallization")

        reflection = {
            "recall_accuracy": recall_acc,
            "n_nodes": n_nodes,
            "n_consolidations": n_consolidations,
            "false_merges": false_merges,
            "false_merge_rate": false_merges / max(n_consolidations, 1),
            "recommendations": recommendations,
            "timestamp": time.time(),
        }
        self._reflection_log.append(reflection)
        return reflection

    def get_adaptive_params(self) -> Dict[str, float]:
        recall_acc = self.evaluate_recall_accuracy()
        if recall_acc < self.recall_threshold:
            return {"consolidation_multiplier": 0.8, "decay_multiplier": 1.1}
        elif recall_acc > 0.9:
            return {"consolidation_multiplier": 1.2, "decay_multiplier": 0.95}
        return {"consolidation_multiplier": 1.0, "decay_multiplier": 1.0}

    def get_state(self) -> Dict:
        return {
            "recall_history": list(self._recall_history),
            "reflection_log": self._reflection_log[-50:],
            "step_counter": self._step_counter,
        }

    def load_state(self, state: Dict):
        self._recall_history = deque(state.get("recall_history", []), maxlen=100)
        self._reflection_log = state.get("reflection_log", [])
        self._step_counter = state.get("step_counter", 0)
