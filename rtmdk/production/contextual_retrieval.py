"""Contextual Retrieval — Anthropic-style chunk headers for RTMDK.

Generates a 1-sentence context header for each chunk before embedding,
improving retrieval by disambiguating identical phrases in different contexts.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ContextualHeaderGenerator:
    """Generate context headers for memory chunks."""

    def __init__(self, backend: str = "heuristic", sot_tokenizer: Optional[Any] = None):
        self.backend = backend
        self.sot_tokenizer = sot_tokenizer

    def generate(self, text: str) -> str:
        """Generate a context header for the given text."""
        if not text:
            return ""
        if self.backend == "sot" and self.sot_tokenizer is not None:
            return self._generate_sot(text)
        return self._generate_heuristic(text)

    def _generate_heuristic(self, text: str) -> str:
        """Simple heuristic: first sentence or first 12 words."""
        # Try to get first sentence
        sentences = text.split(".")
        first = sentences[0].strip() if sentences else text
        if len(first.split()) <= 15:
            return first
        # Fallback: first 12 words
        words = text.split()
        return " ".join(words[:12]) + "..."

    def _generate_sot(self, text: str) -> str:
        """Use SOT tokenizer to extract dominant tokens as pseudo-header."""
        try:
            tokens = self.sot_tokenizer.encode(text)
            if not tokens:
                return self._generate_heuristic(text)
            # Get top-5 most frequent / highest-weighted tokens
            seen = []
            for t in tokens:
                if t not in seen:
                    seen.append(t)
                if len(seen) >= 5:
                    break
            header = " ".join(str(t) for t in seen)
            return header if header else self._generate_heuristic(text)
        except Exception:
            return self._generate_heuristic(text)


class ContextualEmbedderWrapper:
    """Wraps an embedder to prepend context headers before embedding."""

    def __init__(self, embedder, header_generator: ContextualHeaderGenerator):
        self.embedder = embedder
        self.header_generator = header_generator

    def __call__(self, text: str):
        header = self.header_generator.generate(text)
        if header:
            full_text = f"{header}\n\n{text}"
        else:
            full_text = text
        return self.embedder(full_text)
