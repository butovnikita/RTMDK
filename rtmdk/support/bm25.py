"""BM25 index for RTMDK — optimized with pre-tokenized inverted index."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class BM25Index:
    """BM25 index with inverted term-frequency map for O(query_tokens × avg_df) search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        # Inverted index: token -> {doc_id: term_frequency}
        self.term_freqs: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        counts = Counter(tokens)
        for token, count in counts.items():
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
            self.term_freqs.setdefault(token, {})[doc_id] = count
        self.avg_doc_length = float(np.mean(list(self.doc_lengths.values()))) if self.doc_lengths else 0.0

    def remove_document(self, doc_id: str):
        if doc_id not in self.documents:
            return
        text = self.documents.pop(doc_id)
        self.doc_lengths.pop(doc_id, None)
        for token in set(self._tokenize(text)):
            self.doc_freq[token] = max(0, self.doc_freq.get(token, 1) - 1)
            if self.doc_freq[token] == 0:
                del self.doc_freq[token]
            tf_map = self.term_freqs.get(token)
            if tf_map is not None:
                tf_map.pop(doc_id, None)
                if not tf_map:
                    del self.term_freqs[token]
        if self.doc_lengths:
            self.avg_doc_length = float(np.mean(list(self.doc_lengths.values())))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        n = len(self.documents)
        scores: Dict[str, float] = {}
        avg_dl = max(self.avg_doc_length, 1.0)
        k1 = self.k1
        b = self.b

        for token in self._tokenize(query):
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            tf_map = self.term_freqs.get(token)
            if not tf_map:
                continue
            for doc_id, tf in tf_map.items():
                doc_len = max(self.doc_lengths.get(doc_id, 1), 1)
                denom = tf + k1 * (1 - b + b * doc_len / avg_dl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / max(denom, 1e-8)

        if not scores:
            return []
        return [(d, s) for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k] if s > 0]
