"""Causal Graph Extraction from LLM Explanations.

Instead of statistically broken co-occurrence PC-algorithm, we extract
causal edges directly from natural language explanations using pattern
matching.  When an LLM generates "A because B", we add a directed edge
B → A with confidence proportional to linguistic certainty.

Patterns:
    "X because Y"         → Y causes X
    "Y causes X"          → Y causes X
    "Y leads to X"        → Y causes X
    "due to Y, X"         → Y causes X
    "as a result of Y, X" → Y causes X
    "X is caused by Y"    → Y causes X
    "the reason for X is Y" → Y causes X

Reference:
    Das et al. (2024) "What Causes What?" — causal extraction from text
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict

_CAUSE_EFFECT_PATTERNS = [
    # "X because Y"
    (r"(.+?)\s+because\s+(.+)", 1, 2),
    # "Y causes X"
    (r"(.+?)\s+causes?\s+(.+)", 2, 1),
    # "Y leads to X"
    (r"(.+?)\s+leads?\s+to\s+(.+)", 2, 1),
    # "due to Y, X"
    (r"due\s+to\s+(.+?),\s*(.+)", 2, 1),
    # "as a result of Y, X"
    (r"as\s+a\s+result\s+of\s+(.+?),\s*(.+)", 2, 1),
    # "X is caused by Y"
    (r"(.+?)\s+is\s+caused\s+by\s+(.+)", 1, 2),
    # "the reason for X is Y"
    (r"the\s+reason\s+for\s+(.+?)\s+is\s+(.+)", 1, 2),
    # "Y, therefore X"
    (r"(.+?),\s*therefore\s+(.+)", 2, 1),
    # "Y; thus X"
    (r"(.+?);\s*thus\s+(.+)", 2, 1),
]


def extract_causal_edges(text: str) -> List[Tuple[str, str, float]]:
    """Extract (effect, cause, confidence) triples from explanation text.

    Returns:
        List of (effect_phrase, cause_phrase, confidence) tuples.
        Confidence is 1.0 for exact matches, scaled down for longer sentences.
    """
    text = text.strip()
    if not text or len(text) < 10:
        return []

    results: List[Tuple[str, str, float]] = []
    sentences = [s.strip() for s in re.split(r"[.!?;]", text) if s.strip()]

    for sentence in sentences:
        for pattern, effect_group, cause_group in _CAUSE_EFFECT_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                try:
                    effect = match.group(effect_group).strip().lower()
                    cause = match.group(cause_group).strip().lower()
                    # Clean up trailing punctuation/words
                    effect = re.sub(r"\W+$", "", effect)
                    cause = re.sub(r"\W+$", "", cause)
                    if len(effect) > 3 and len(cause) > 3:
                        # Confidence decays with sentence length
                        confidence = max(0.5, 1.0 - len(sentence) / 200.0)
                        results.append((effect, cause, confidence))
                except IndexError:
                    continue
    return results


def extract_causal_edges_from_content(content: Dict) -> List[Tuple[str, str, float]]:
    """Extract causal edges from node content dict."""
    texts = []
    for key in ["text", "input_text", "output_text", "explanation", "reasoning"]:
        val = content.get(key, "")
        if val and isinstance(val, str):
            texts.append(val)
    all_results: List[Tuple[str, str, float]] = []
    for t in texts:
        all_results.extend(extract_causal_edges(t))
    return all_results
