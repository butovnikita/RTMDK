"""rtmdk/utils/modality.py"""

from __future__ import annotations
import re
from typing import Optional, Dict


def detect_tier(text: str, context: Optional[Dict] = None) -> str:
    """Auto-detect memory tier from content."""
    context = context or {}
    text_lower = text.lower()
    if context.get("tool_used"):
        return "procedural"
    if any(p in text_lower for p in ["how to", "how do", "how can", "steps to", "tutorial", "guide"]):
        return "procedural"
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text):
        return "episodic"
    if any(
        p in text_lower for p in ["yesterday", "last week", "last month", "ago", "вчера", "на прошлой", "неделю назад"]
    ):
        return "episodic"
    return "semantic"


def detect_modality(text: str) -> str:
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
        r"\b(metric|kpi|latency|throughput|error-rate|uptime|cpu|memory|disk)\b",
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
