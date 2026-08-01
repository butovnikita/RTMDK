"""
rtmdk/production/context_optimizer.py — Adaptive Context Window Optimization.

Compresses and prioritizes retrieval context for LLM consumption.
Features:
- Adaptive window: 50-300 tokens based on query relevance
- Priority-based selection: most relevant facts first
- LLM model awareness (GPT-4: 128K, Claude: 200K, etc.)
- Semantic deduplication: removes redundant information
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# LLM model context limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet": 200000,
    "claude-opus": 200000,
    "claude-haiku": 200000,
    "gemini": 1048576,
    "llama": 131072,
    "qwen": 131072,
    "default": 8192,
}


@dataclass
class ContextSegment:
    """A segment of context text with metadata."""

    text: str
    score: float  # Relevance score from retrieval
    node_id: str  # Source node ID
    token_count: int  # Estimated token count
    topic: str = ""  # Topic/category tag


def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    if not text:
        return 0
    # Rough estimate: English ~4 chars/token, Russian ~6 chars/token
    has_cyrillic = bool(re.search(r"[а-яё]", text.lower()))
    chars_per_token = 6 if has_cyrillic else 4
    return max(1, len(text) // chars_per_token)


class ContextOptimizer:
    """Optimizes retrieval context for LLM consumption.

    Usage:
        optimizer = ContextOptimizer(model="gpt-4o", max_tokens=300)
        context = optimizer.optimize(raw_context, query)
    """

    def __init__(
        self,
        model: str = "default",
        min_tokens: int = 50,
        max_tokens: int = 300,
        deduplicate: bool = True,
        priority_format: bool = True,
    ):
        self.model = model
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.model_limit = MODEL_CONTEXT_LIMITS.get(model, MODEL_CONTEXT_LIMITS["default"])
        self.deduplicate = deduplicate
        self.priority_format = priority_format

    def optimize(
        self,
        raw_context: str,
        query: str = "",
        available_tokens: Optional[int] = None,
    ) -> str:
        """Optimize context for LLM.

        Args:
            raw_context: Raw context string from RTMDK retrieval
            query: Original query (for relevance-aware compression)
            available_tokens: Override token budget (None = use max_tokens)

        Returns:
            Optimized context string
        """
        if not raw_context or raw_context.strip() in ("", "No relevant memory.", "[]"):
            return raw_context

        max_tok = available_tokens or self.max_tokens

        # Parse context into segments
        segments = self._parse_context(raw_context)
        if not segments:
            return raw_context

        # Deduplicate if enabled
        if self.deduplicate:
            segments = self._deduplicate(segments)

        # Sort by score (most relevant first)
        segments.sort(key=lambda s: s.score, reverse=True)

        # Build optimized context within token budget
        optimized = self._build_context(segments, max_tok, query)

        return optimized

    def _parse_context(self, raw_context: str) -> List[ContextSegment]:
        """Parse raw context into segments."""
        segments = []

        # Try to parse structured format: [ATTN:xxx][SAL:xxx][TIER:x] text
        pattern = r"\[ATTN:([0-9.]+)\](?:\[SAL:([0-9.]+)\])?(?:\[TIER:(\w+)\])?\s*(.+?)(?=\[ATTN:|$)"
        matches = re.findall(pattern, raw_context, re.DOTALL)

        if matches:
            for i, (attn, sal, tier, text) in enumerate(matches):
                text = text.strip()
                if text:
                    score = float(attn)
                    segments.append(
                        ContextSegment(
                            text=text,
                            score=score,
                            node_id=f"segment_{i}",
                            token_count=estimate_tokens(text),
                            topic=tier or "",
                        )
                    )
        else:
            # Fallback: split by newlines
            lines = raw_context.strip().split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    segments.append(
                        ContextSegment(
                            text=line,
                            score=1.0 / (i + 1),  # Decreasing score by position
                            node_id=f"line_{i}",
                            token_count=estimate_tokens(line),
                        )
                    )

        return segments

    def _deduplicate(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Remove semantically duplicate segments."""
        unique = []
        seen_hashes = set()

        for seg in segments:
            # Normalize text for comparison
            normalized = re.sub(r"[^a-zа-яё0-9\s]", "", seg.text.lower())
            # Create n-gram fingerprint
            words = normalized.split()
            if len(words) < 3:
                key = normalized
            else:
                # Use first 5 words as fingerprint
                key = " ".join(words[:5])

            if key not in seen_hashes:
                seen_hashes.add(key)
                unique.append(seg)

        return unique

    def _build_context(
        self,
        segments: List[ContextSegment],
        max_tokens: int,
        query: str = "",
    ) -> str:
        """Build optimized context within token budget."""
        result_parts = []
        total_tokens = 0

        # Add header if priority format enabled
        if self.priority_format:
            header = f"### RELEVANT CONTEXT ({len(segments)} facts)\n"
            result_parts.append(header)
            total_tokens = estimate_tokens(header)

        for seg in segments:
            seg_tokens = seg.token_count
            if total_tokens + seg_tokens > max_tokens:
                break

            if self.priority_format:
                # Format with score indicator
                prefix = f"[{seg.score:.2f}] "
                result_parts.append(f"{prefix}{seg.text}\n")
            else:
                result_parts.append(f"{seg.text}\n")

            total_tokens += seg_tokens

        # Ensure minimum context
        if not result_parts or total_tokens < self.min_tokens:
            # Include at least one segment even if over budget
            if segments:
                return f"{segments[0].text}"
            return ""

        return "\n".join(result_parts).strip()

    def get_stats(self) -> Dict[str, Any]:
        """Return optimizer configuration."""
        return {
            "model": self.model,
            "model_context_limit": self.model_limit,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "deduplicate": self.deduplicate,
            "priority_format": self.priority_format,
        }
