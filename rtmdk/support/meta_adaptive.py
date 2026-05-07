"""Meta-adaptive kernel for RTMDK."""
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np
from scipy import stats as scipy_stats


class MetaAdaptiveKernel:
    def __init__(
            self,
            base_bandwidth: float = 1.0,
            base_phase_coupling: float = 0.3,
            adaptation_lr: float = 0.005,
            kurtosis_target_min: float = 1.5,
            kurtosis_target_max: float = 4.0):
        self.base_bandwidth = base_bandwidth
        self.base_phase_coupling = base_phase_coupling
        self.adaptation_lr = adaptation_lr
        self.kurtosis_target_min = kurtosis_target_min
        self.kurtosis_target_max = kurtosis_target_max
        self.effective_bandwidth = base_bandwidth
        self.effective_phase_coupling = base_phase_coupling
        self._response_history: deque = deque(maxlen=100)
        self._semantic_density: deque = deque(maxlen=50)
        self._uncertainty: deque = deque(maxlen=20)
        self._kurtosis_history: deque = deque(maxlen=50)

    def record_response(self, response: float):
        self._response_history.append(response)

    def record_semantic_density(self, density: float):
        self._semantic_density.append(density)

    def record_uncertainty(self, entropy: float):
        self._uncertainty.append(entropy)

    def compute_resonance_kurtosis(self) -> float:
        if len(self._response_history) < 4:
            return 3.0
        responses = np.array(self._response_history)
        if np.std(responses) < 1e-8:
            return 3.0
        return float(scipy_stats.kurtosis(responses) + 3.0)

    def adapt(self):
        kurtosis = self.compute_resonance_kurtosis()
        self._kurtosis_history.append(kurtosis)
        # Bug #9 FIX: Reversed direction was driving system AWAY from target
        # Low kurtosis (flat distribution) → WIDEN bandwidth to sharpen
        # High kurtosis (peaked distribution) → NARROW bandwidth to smooth
        if kurtosis < self.kurtosis_target_min:
            self.effective_bandwidth *= (1.0 +
                                         self.adaptation_lr)  # WIDEN to sharpen
        elif kurtosis > self.kurtosis_target_max:
            self.effective_bandwidth *= (1.0 -
                                         self.adaptation_lr)  # NARROW to smooth
        if self._semantic_density:
            density = np.mean(self._semantic_density)
            if density > 0.7:
                self.effective_phase_coupling = min(
                    0.9, self.effective_phase_coupling + self.adaptation_lr * 0.5)
            elif density < 0.2:
                self.effective_phase_coupling = max(
                    0.05, self.effective_phase_coupling - self.adaptation_lr * 0.5)
        if self._uncertainty:
            uncertainty = np.mean(self._uncertainty)
            if uncertainty > 1.5:
                self.effective_bandwidth *= (1.0 + self.adaptation_lr)
        self.effective_bandwidth = max(
            0.1, min(10.0, self.effective_bandwidth))
        self.effective_phase_coupling = max(
            0.0, min(1.0, self.effective_phase_coupling))

    def get_bandwidth(self) -> float:
        return self.effective_bandwidth

    def get_phase_coupling(self) -> float:
        return self.effective_phase_coupling

    def get_state(self) -> Dict:
        return {
            "base_bandwidth": self.base_bandwidth,
            "base_phase_coupling": self.base_phase_coupling,
            "effective_bandwidth": self.effective_bandwidth,
            "effective_phase_coupling": self.effective_phase_coupling,
            "kurtosis": self.compute_resonance_kurtosis(),
            "avg_density": float(
                np.mean(
                    self._semantic_density)) if self._semantic_density else 0,
            "avg_uncertainty": float(
                np.mean(
                    self._uncertainty)) if self._uncertainty else 0,
        }

    def load_state(self, state: Dict):
        self.base_bandwidth = state.get("base_bandwidth", self.base_bandwidth)
        self.base_phase_coupling = state.get(
            "base_phase_coupling", self.base_phase_coupling)
        self.effective_bandwidth = state.get(
            "effective_bandwidth", self.base_bandwidth)
        self.effective_phase_coupling = state.get(
            "effective_phase_coupling", self.base_phase_coupling)
