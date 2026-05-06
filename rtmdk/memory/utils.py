"""Utility functions and exceptions for RTMDK memory system."""

import re
import math
from typing import Dict


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
