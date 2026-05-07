"""Learnable kernel and differentiable consolidation for RTMDK."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from rtmdk.nodes import MemoryNode


class LearnableKernel:
    def __init__(self, bandwidth: float = 1.0, phase_coupling: float = 0.3,
                 decay_rate: float = 0.998, gradient_clip: float = 1.0):
        self.bandwidth = bandwidth
        self.phase_coupling = phase_coupling
        self.decay_rate = decay_rate
        self.gradient_clip = gradient_clip
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0
        self._adam_state = {
            "bandwidth": {"m": 0.0, "v": 0.0, "t": 0},
            "phase_coupling": {"m": 0.0, "v": 0.0, "t": 0},
        }

    def resonance_response(
            self,
            dist: float,
            phase_diff: float,
            amplitude: float,
            salience: float) -> float:
        # Bug #1 FIX: Gaussian kernel for sharper resonance
        spatial = math.exp(-dist ** 2 / (2 * self.bandwidth ** 2))
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        return spatial * ((1 - self.phase_coupling) +
                          self.phase_coupling * phase_align) * amplitude * salience

    def compute_gradients(
            self,
            dist: float,
            phase_diff: float,
            amplitude: float,
            salience: float,
            loss_gradient: float = 1.0):
        # Bug #1 FIX: Gaussian kernel gradient
        spatial = math.exp(-dist ** 2 / (2 * self.bandwidth ** 2))
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        self._grad_bandwidth += loss_gradient * spatial * (dist ** 2 / (self.bandwidth ** 3)) * (
            (1 - self.phase_coupling) + self.phase_coupling * phase_align) * amplitude * salience
        self._grad_phase_coupling += loss_gradient * \
            spatial * (phase_align - 1.0) * amplitude * salience

    def step(self):
        for param_name, grad in [
                ("bandwidth", self._grad_bandwidth), ("phase_coupling", self._grad_phase_coupling)]:
            if abs(grad) < 1e-12:
                continue
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
            s = self._adam_state[param_name]
            s["t"] += 1
            s["m"] = 0.9 * s["m"] + 0.1 * grad
            s["v"] = 0.999 * s["v"] + 0.001 * grad ** 2
            m_hat = s["m"] / (1 - 0.9 ** s["t"])
            v_hat = s["v"] / (1 - 0.999 ** s["t"])
            lr = 0.001
            update = lr * m_hat / (math.sqrt(v_hat) + 1e-8)
            if param_name == "bandwidth":
                self.bandwidth = max(0.1, self.bandwidth - update)
            elif param_name == "phase_coupling":
                self.phase_coupling = float(
                    np.clip(self.phase_coupling - update, 0.0, 1.0))
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0

    def get_state(self) -> Dict:
        return {
            "bandwidth": self.bandwidth,
            "phase_coupling": self.phase_coupling,
            "decay_rate": self.decay_rate,
            "adam_state": {
                k: dict(v) for k,
                v in self._adam_state.items()}}

    def load_state(self, state: Dict):
        self.bandwidth = state["bandwidth"]
        self.phase_coupling = state["phase_coupling"]
        self.decay_rate = state.get("decay_rate", self.decay_rate)
        if "adam_state" in state:
            self._adam_state = state["adam_state"]


class DifferentiableConsolidation:
    def __init__(self, loss_weight: float = 0.1):
        self.loss_weight = loss_weight
        self.consolidation_loss = 0.0

    def compute_synthesis(
            self,
            node1: "MemoryNode",
            node2: "MemoryNode",
            gate: float) -> Dict:
        w1, w2 = gate, 1.0 - gate
        new_latent = w1 * node1.latent_pos + w2 * node2.latent_pos
        new_phase = np.arctan2(w1 * np.sin(node1.phase) + w2 * np.sin(
            node2.phase), w1 * np.cos(node1.phase) + w2 * np.cos(node2.phase)) % (2 * np.pi)
        new_amp = min(1.0, w1 * node1.amplitude + w2 * node2.amplitude)
        new_sal = w1 * node1.salience + w2 * node2.salience
        pos_loss: float = np.sum((new_latent - node1.latent_pos)**2) + \
            np.sum((new_latent - node2.latent_pos)**2)
        phase_loss = min(abs(new_phase - node1.phase),
                         2 * np.pi - abs(new_phase - node1.phase)) + min(abs(new_phase - node2.phase),
                                                                         2 * np.pi - abs(new_phase - node2.phase))
        self.consolidation_loss = self.loss_weight * \
            (pos_loss + phase_loss * 0.1)
        return {
            "latent_pos": new_latent,
            "phase": new_phase,
            "amplitude": new_amp,
            "salience": new_sal,
            "loss": self.consolidation_loss}
