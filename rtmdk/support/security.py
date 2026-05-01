"""Security validator for RTMDK."""
from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    pass

# Zero-width characters to strip from text (Fix 7: Unicode bypass prevention)
_ZERO_WIDTH_CHARS = re.compile(
    r"[​‌‍﻿￼᠎⁠   ]"
)

# Confusable unicode homoglyphs (partial list — maps lookalikes to ASCII)
_HOMOGLYPHS = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x',
    'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'ɡ': 'g', 'һ': 'h',
    'ӏ': 'l', 'п': 'n', 'υ': 'u', 'ү': 'y', 'қ': 'k', 'ӣ': 'i',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Κ': 'K', 'Μ': 'M', 'Ν': 'H',
    'Ο': 'O', 'Ρ': 'P', 'Τ': 'Y', 'Χ': 'X', 'Ι': 'I', 'Ξ': 'Z',
}


class SecurityValidator:
    """Protects against memory poisoning, prompt injection, and graph attacks."""

    def __init__(self, max_text_length: int = 10000,
                 tension_spike_threshold: float = 0.5,
                 injection_patterns: Optional[List[str]] = None):
        self.max_text_length = max_text_length
        self.tension_spike_threshold = tension_spike_threshold
        # Fix 7: Compile patterns as regex for more robust matching
        self.injection_patterns = injection_patterns or [
            r"ignore\s*previous", r"system\s*prompt", r"you\s*are\s*now",
            r"disregard", r"ignore\s*all", r"new\s*instruction", r"override",
            r"forget\s*(?:all\s*)?previous", r"disregard\s*all",
            r"act\s*(?:as\s*)?(?:if|though)\s*you\s*(?:are|were)",
            r"pretend\s*(?:you\s*)?(?:are|were)", r"roleplay\s*as",
            r"sudo\s", r"admin\s*(?:mode)?",
        ]
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.injection_patterns]
        self._violation_log: List[Dict] = []
        self._tension_history: deque = deque(maxlen=100)

    def _sanitize_text(self, text: str) -> str:
        """Normalize and strip dangerous unicode characters (Fix 7)."""
        # Strip zero-width characters
        text = _ZERO_WIDTH_CHARS.sub("", text)
        # Unicode NFC normalization — combines combining characters
        text = unicodedata.normalize("NFC", text)
        # Replace confusable homoglyphs with ASCII equivalents
        result = []
        for ch in text.lower():
            result.append(_HOMOGLYPHS.get(ch, ch))
        return "".join(result)

    def validate_node_content(self, text: str) -> Dict[str, Any]:
        """Validate node text for injection patterns and length."""
        violations = []
        if len(text) > self.max_text_length:
            violations.append({"type": "text_too_long", "length": len(text), "max": self.max_text_length})
        # Fix 7: Sanitize text before pattern matching
        sanitized = self._sanitize_text(text)
        for pattern_re in self._compiled_patterns:
            if pattern_re.search(sanitized):
                violations.append({"type": "prompt_injection", "pattern": pattern_re.pattern})
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
