"""rtmdk/support/safety_certifier.py — Lyapunov-based Soft Regulator for RTMDK.

Computes a Lyapunov function V = α·tension² + β·entropy(responses) + γ·causal_conflict
and applies soft regulation when dV/dt exceeds threshold.

Key design:
- Instead of binary blocking, uses soft regulator: lr *= exp(-dV/clamp)
- Three modes: monitor_only (local), soft_regulate (default), hard_block (production)
- Hooks into RTMDKField.step() before consolidation and meta-adaptation
"""
from __future__ import annotations
from typing import Dict, List, Any
from collections import deque
import math
import time
import numpy as np


class LyapunovFunction:
    """Computes V = α·tension² + β·entropy(responses) + γ·causal_conflict."""

    def __init__(
            self,
            alpha: float = 0.4,
            beta: float = 0.4,
            gamma: float = 0.2,
            window_size: int = 20):
        self.alpha = alpha    # tension weight
        self.beta = beta      # entropy weight
        self.gamma = gamma    # causal conflict weight
        self.window_size = window_size
        self._tension_history: deque = deque(maxlen=window_size)
        self._response_history: deque = deque(maxlen=window_size)
        self._conflict_history: deque = deque(maxlen=window_size)
        self._v_history: deque = deque(maxlen=window_size)

    def record_tensions(self, nodes: Dict[str, Any]):
        """Record mean tension across all nodes."""
        if not nodes:
            self._tension_history.append(0.0)
            return
        tensions = [getattr(n, 'tension', 0.0) for n in nodes.values()]
        mean_tension = float(np.mean(tensions))
        self._tension_history.append(mean_tension)

    def record_responses(self, resonance_scores: List[float]):
        """Record resonance response entropy."""
        if not resonance_scores:
            self._response_history.append(0.0)
            return
        scores = np.array(resonance_scores)
        scores = scores[scores > 1e-8]
        if len(scores) < 2:
            self._response_history.append(0.0)
            return
        total = scores.sum()
        if total < 1e-8:
            self._response_history.append(0.0)
            return
        probs = scores / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        self._response_history.append(normalized)

    def record_conflicts(self, n_contradictions: int, n_nodes: int):
        """Record causal conflict ratio."""
        if n_nodes == 0:
            self._conflict_history.append(0.0)
            return
        ratio = min(1.0, n_contradictions / max(n_nodes, 1))
        self._conflict_history.append(ratio)

    def compute_V(self) -> float:
        """Compute current Lyapunov function value."""
        if not self._tension_history or not self._response_history or not self._conflict_history:
            return 0.0
        tension = self._tension_history[-1]
        entropy = self._response_history[-1]
        conflict = self._conflict_history[-1]
        V = self.alpha * tension**2 + self.beta * entropy + self.gamma * conflict
        self._v_history.append(V)
        return V

    def compute_dV_dt(self) -> float:
        """Compute rate of change of V."""
        if len(self._v_history) < 2:
            return 0.0
        return self._v_history[-1] - self._v_history[-2]

    def get_state_summary(self) -> Dict[str, float]:
        return {
            "V": self.compute_V(),
            "dV_dt": self.compute_dV_dt(),
            "mean_tension": float(
                np.mean(
                    self._tension_history)) if self._tension_history else 0.0,
            "mean_entropy": float(
                np.mean(
                    self._response_history)) if self._response_history else 0.0,
            "mean_conflict": float(
                np.mean(
                    self._conflict_history)) if self._conflict_history else 0.0,
        }


class SafetyCertifier:
    """Lyapunov-based safety certifier with soft regulation.

    Modes:
    - monitor_only: Just log warnings, no action
    - soft_regulate: Scale learning rates by exp(-dV/clamp)
    - hard_block: Block updates entirely when dV/dt > threshold
    """

    def __init__(self, mode: str = "soft_regulate",
                 lyapunov_threshold: float = 0.1,
                 clamp_value: float = 0.5,
                 alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2,
                 window_size: int = 20):
        self.mode = mode  # "monitor_only" | "soft_regulate" | "hard_block"
        self.lyapunov_threshold = lyapunov_threshold
        self.clamp_value = clamp_value
        self.lyapunov = LyapunovFunction(alpha, beta, gamma, window_size)
        self._warnings: List[Dict] = []
        self._regulation_events: int = 0
        self._block_events: int = 0
        self._total_checks: int = 0

    def check_and_regulate(self, nodes: Dict, resonance_scores: List[float],
                           n_contradictions: int = 0) -> Dict[str, Any]:
        """Check Lyapunov stability and apply regulation if needed.

        Returns:
            Dict with regulation info:
            - V, dV_dt, safe, regulation_factor, should_block, mode
        """
        self._total_checks += 1

        # Record current state
        self.lyapunov.record_tensions(nodes)
        self.lyapunov.record_responses(resonance_scores)
        self.lyapunov.record_conflicts(n_contradictions, len(nodes))

        V = self.lyapunov.compute_V()
        dV_dt = self.lyapunov.compute_dV_dt()

        result = {
            "V": V,
            "dV_dt": dV_dt,
            "safe": True,
            "regulation_factor": 1.0,
            "should_block": False,
            "mode": self.mode,
        }

        # Check stability
        if abs(dV_dt) > self.lyapunov_threshold:
            if self.mode == "monitor_only":
                result["safe"] = True  # Don't block, just warn
                self._warnings.append({
                    "type": "lyapunov_warning",
                    "V": V, "dV_dt": dV_dt,
                    "timestamp": time.time(),
                })
            elif self.mode == "soft_regulate":
                # Soft regulation: exp(-|dV|/clamp)
                reg_factor = math.exp(-abs(dV_dt) / self.clamp_value)
                reg_factor = max(0.1, min(1.0, reg_factor)
                                 )  # Clamp to [0.1, 1.0]
                result["safe"] = True  # Still allow updates, just slower
                result["regulation_factor"] = reg_factor
                self._regulation_events += 1
                if reg_factor < 0.5:
                    self._warnings.append({
                        "type": "strong_regulation",
                        "V": V, "dV_dt": dV_dt, "factor": reg_factor,
                        "timestamp": time.time(),
                    })
            elif self.mode == "hard_block":
                result["safe"] = False
                result["should_block"] = True
                self._block_events += 1
                self._warnings.append({
                    "type": "update_blocked",
                    "V": V, "dV_dt": dV_dt,
                    "timestamp": time.time(),
                })

        return result

    def get_regulation_factor(self) -> float:
        """Get current regulation factor for scaling learning rates."""
        dV_dt = self.lyapunov.compute_dV_dt()
        if self.mode == "soft_regulate" and abs(
                dV_dt) > self.lyapunov_threshold:
            return math.exp(-abs(dV_dt) / self.clamp_value)
        return 1.0

    def should_block_updates(self) -> bool:
        """Check if updates should be blocked (hard_block mode only)."""
        if self.mode != "hard_block":
            return False
        dV_dt = self.lyapunov.compute_dV_dt()
        return abs(dV_dt) > self.lyapunov_threshold

    def get_stats(self) -> Dict:
        summary = self.lyapunov.get_state_summary()
        summary.update({
            "mode": self.mode,
            "total_checks": self._total_checks,
            "regulation_events": self._regulation_events,
            "block_events": self._block_events,
            "n_warnings": len(self._warnings),
        })
        return summary

    def get_state(self) -> Dict:
        return {
            "mode": self.mode,
            "lyapunov_threshold": self.lyapunov_threshold,
            "clamp_value": self.clamp_value,
            "tension_history": list(self.lyapunov._tension_history),
            "response_history": list(self.lyapunov._response_history),
            "conflict_history": list(self.lyapunov._conflict_history),
            "v_history": list(self.lyapunov._v_history),
            "warnings": self._warnings[-50:],
            "regulation_events": self._regulation_events,
            "block_events": self._block_events,
            "total_checks": self._total_checks,
        }

    def load_state(self, data: Dict):
        self.mode = data.get("mode", self.mode)
        self.lyapunov_threshold = data.get(
            "lyapunov_threshold", self.lyapunov_threshold)
        self.clamp_value = data.get("clamp_value", self.clamp_value)
        self.lyapunov._tension_history = deque(
            data.get("tension_history", []), maxlen=self.lyapunov.window_size)
        self.lyapunov._response_history = deque(
            data.get("response_history", []), maxlen=self.lyapunov.window_size)
        self.lyapunov._conflict_history = deque(
            data.get("conflict_history", []), maxlen=self.lyapunov.window_size)
        self.lyapunov._v_history = deque(
            data.get(
                "v_history",
                []),
            maxlen=self.lyapunov.window_size)
        self._warnings = data.get("warnings", [])
        self._regulation_events = data.get("regulation_events", 0)
        self._block_events = data.get("block_events", 0)
        self._total_checks = data.get("total_checks", 0)
