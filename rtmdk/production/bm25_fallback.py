"""rtmdk/production/bm25_fallback.py — BM25 Fallback when resonance is low."""

import re
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class BM25FallbackRetriever:
    """BM25 text retrieval as fallback when RTMDK resonance is too low."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, min_score: float = 0.1):
        self.k1 = k1
        self.b = b
        self.min_score = min_score
        self._documents: Dict[str, str] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._term_freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._avg_doc_length = 0.0
        self._total_docs = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                     'and', 'or', 'but', 'if', 'then', 'than', 'so', 'as', 'that',
                     'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
                     'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
                     'them', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
                     'why', 'how', 'not', 'no', 'all', 'each', 'every', 'both', 'few',
                     'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
                     'about', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'out', 'off', 'over', 'under', 'again',
                     'further', 'once', 'here', 'there', 'very', 'just', 'because'}
        tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
        return tokens

    def add_document(self, doc_id: str, text: str):
        self._documents[doc_id] = text
        tokens = self._tokenize(text)
        self._doc_lengths[doc_id] = len(tokens)
        self._total_docs += 1

        for token in set(tokens):
            self._term_freqs[doc_id][token] += 1
            if self._term_freqs[doc_id][token] == 1:
                self._doc_freqs[token] += 1

        self._avg_doc_length = sum(self._doc_lengths.values()) / max(self._total_docs, 1)

    def remove_document(self, doc_id: str):
        if doc_id not in self._documents:
            return
        text = self._documents.pop(doc_id)
        tokens = self._tokenize(text)
        for token in set(tokens):
            self._term_freqs[doc_id][token] = 0
            self._doc_freqs[token] = max(0, self._doc_freqs[token] - 1)
        del self._doc_lengths[doc_id]
        self._total_docs -= 1
        self._avg_doc_length = sum(self._doc_lengths.values()) / max(self._total_docs, 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = defaultdict(float)
        n = self._total_docs

        for token in query_tokens:
            df = self._doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in self._term_freqs.items():
                if doc_id not in self._doc_lengths:
                    continue
                term_count = tf.get(token, 0)
                if term_count == 0:
                    continue
                doc_len = self._doc_lengths[doc_id]
                numerator = term_count * (self.k1 + 1)
                denominator = term_count + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_length, 1))
                scores[doc_id] += idf * numerator / denominator

        results = [(doc_id, score) for doc_id, score in scores.items() if score >= self.min_score]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def size(self) -> int:
        return len(self._documents)
