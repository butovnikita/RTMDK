"""Security validator for RTMDK."""
from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    pass


class SecurityValidator:
    """Protects against memory poisoning, prompt injection, and graph attacks."""

    def __init__(self, max_text_length: int = 10000,
                 tension_spike_threshold: float = 0.5,
                 injection_patterns: Optional[List[str]] = None):
        self.max_text_length = max_text_length
        self.tension_spike_threshold = tension_spike_threshold
        self.injection_patterns = injection_patterns or [
            "ignore previous", "system prompt", "you are now", "disregard",
            "ignore all", "new instruction", "override"
        ]
        self._violation_log: List[Dict] = []
        self._tension_history: deque = deque(maxlen=100)

    def validate_node_content(self, text: str) -> Dict[str, Any]:
        """Validate node text for injection patterns and length."""
        violations = []
        if len(text) > self.max_text_length:
            violations.append({"type": "text_too_long", "length": len(text), "max": self.max_text_length})
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if pattern in text_lower:
                violations.append({"type": "prompt_injection", "pattern": pattern})
        is_safe = len(violations) == 0
        if violations:
            self._violation_log.append({
                "type": "node_validation", "text_preview": text[:100],
                "violations": violations, "timestamp": time.time(),
            })
        return {"is_safe": is_safe, "violations": violations}

    def validate_tension_spike(self, current_tension: float) -> bool:
        """Detect anomalous tension spikes that may indicate attacks."""
        self._tension_history.append(current_tension)
        if len(self._tension_history) < 10:
            return True
        mean_t = np.mean(self._tension_history)
        std_t = np.std(self._tension_history)
        if std_t > 0 and (current_tension - mean_t) / std_t > self.tension_spike_threshold:
            self._violation_log.append({
                "type": "tension_spike", "current": current_tension,
                "mean": float(mean_t), "std": float(std_t), "timestamp": time.time(),
            })
            return False
        return True

    def validate_causal_graph_integrity(self, causal_engine: Any) -> Dict[str, Any]:
        """Check causal graph for anomalies."""
        if not causal_engine or not hasattr(causal_engine, 'causal_effects'):
            return {"is_valid": True, "issues": []}
        issues = []
        effects = causal_engine.causal_effects
        for (src, tgt), edge in effects.items():
            if src == tgt:
                issues.append({"type": "self_loop", "node": src})
            if edge.strength < 0 or edge.strength > 1.0:
                issues.append({"type": "invalid_strength", "edge": f"{src}->{tgt}", "strength": edge.strength})
        is_valid = len(issues) == 0
        if issues:
            self._violation_log.append({
                "type": "causal_graph_integrity", "issues": issues, "timestamp": time.time(),
            })
        return {"is_valid": is_valid, "issues": issues, "n_edges": len(effects)}

    def get_violation_summary(self) -> Dict:
        return {
            "total_violations": len(self._violation_log),
            "recent_violations": self._violation_log[-10:],
            "tension_spike_rate": sum(1 for v in self._violation_log if v["type"] == "tension_spike") / max(len(self._tension_history), 1),
        }

    def get_state(self) -> Dict:
        return {"violation_log": self._violation_log[-100:], "tension_history": list(self._tension_history)}

    def load_state(self, state: Dict):
        self._violation_log = state.get("violation_log", [])
        self._tension_history = deque(state.get("tension_history", []), maxlen=100)
