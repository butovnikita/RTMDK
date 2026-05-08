"""Utility functions and exceptions for RTMDK memory system."""

import re
import math
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rtmdk.nodes import MemoryNode


class SecurityViolationError(Exception):
    """Raised when a node violates security policy (prompt injection detected)."""


def detect_modality(text: str) -> str:
    """Detect content modality from text patterns."""
    code_patterns = [
        r"\b(def|class|function|import|from|return|if|else|for|while|const|let|var)\b",
        r"[{}()\[\];]",
        r"\b(async|await|lambda|yield|try|except|catch|throw)\b",
    ]
    audio_patterns = [
        r"\b(audio|sound|music|frequency|hz|waveform|sample|rate|decibel|db)\b",
        r"\b(mp3|wav|flac|aac|ogg|pcm|bitrate|spectrum)\b",
    ]
    vision_patterns = [
        r"\b(image|photo|picture|pixel|resolution|rgb|color|frame|video)\b",
        r"\b(png|jpg|jpeg|gif|bmp|tiff|width|height|crop|resize)\b",
    ]
    metrics_patterns = [
        r"\b(metric|kpi|latency|throughput|error_rate|uptime|cpu|memory|disk)\b",
        r"\b(p99|p95|p50|iops|mbps|gbps|ms|rpm)\b",
        r"\d+\s*(ms|s|mb|gb|tb|kb)",
    ]
    text_lower = text.lower()
    for pattern in code_patterns:
        if re.search(pattern, text_lower):
            return "code"
    for pattern in metrics_patterns:
        if re.search(pattern, text_lower):
            return "metrics"
    for pattern in audio_patterns:
        if re.search(pattern, text_lower):
            return "audio"
    for pattern in vision_patterns:
        if re.search(pattern, text_lower):
            return "vision"
    return "text"


def cross_modal_resonance(
    q_mod: str,
    n_mod: str,
    base_resp: float,
    modal_phase_offsets: Dict[str, float],
    cross_modal_kernel_weight: float,
) -> float:
    """Compute cross-modal resonance boost."""
    q_phase = modal_phase_offsets.get(q_mod, 0.0)
    n_phase = modal_phase_offsets.get(n_mod, 0.0)
    phase_diff = q_phase - n_phase
    modal_coupling = math.cos(phase_diff)
    boost = 1.0 + cross_modal_kernel_weight * modal_coupling
    return base_resp * boost


def apply_attention_bias(
    results: List[Tuple[str, float, "MemoryNode"]],
    temperature: float = 1.0,
) -> List[Tuple[str, float, "MemoryNode"]]:
    """Transform raw resonance scores into attention-biased scores.

    Incorporates causal_strength, tension, salience as structural signals.
    """
    if not results:
        return results

    raw_scores = np.array([r for _, r, _ in results])
    if len(raw_scores) < 2:
        return results

    weights = []
    for nid, resp, node in results:
        score = resp
        causal_boost = sum(node.causal_strength.values()) if hasattr(node, 'causal_strength') else 0
        score *= (1.0 + 0.2 * min(1.0, causal_boost))
        score *= max(0.5, 1.0 - node.tension)
        goal_rel = getattr(node, 'goal_relevance', 0.0)
        score *= (1.0 + 0.3 * goal_rel)
        weights.append(score)

    weights = np.array(weights)
    if temperature > 0:
        exp_weights = np.exp(weights / temperature)
        normalized = exp_weights / (exp_weights.sum() + 1e-8)
    else:
        normalized = weights / (weights.sum() + 1e-8)

    biased = []
    for i, (nid, _, node) in enumerate(results):
        biased.append((nid, float(normalized[i]), node))
    biased.sort(key=lambda x: x[1], reverse=True)
    return biased


def _enum_value(val, default):
    """Convert enum or enum-value to canonical value."""
    if val is None:
        return default
    if isinstance(val, Enum):
        return val
    return val.value if hasattr(val, "value") else val
