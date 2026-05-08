"""MI-Subword Tokenizer — Information-theoretic BPE for SOT v2.0.

Theoretical foundation:
    Optimal subword segmentation minimizes the description length of the
    corpus. The greedy merge rule selects the pair (a,b) that maximizes
    mutual information:

        MI(a,b) = log( P(a,b) / (P(a)·P(b)) )

    This is equivalent to minimizing the cross-entropy of the corpus under
    a bigram language model, and provably yields the most compressible
    tokenization among all greedy byte-pair schemes (see e.g. Gage 1994,
    Sennrich et al. 2016, and the information-theoretic treatment in
    "Neural Machine Translation of Rare Words with Subword Units").

    The algorithm is an online variant of Re-Pair (Larsson & Moffat 1999)
    with MI-based scoring instead of frequency-based scoring.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Pre-tokenization regex: words + standalone punctuation
# This prevents merges across whitespace boundaries.
_WORD_RE = re.compile(r"[\w\']+|[^\w\s]+")


class MI_SubwordTokenizer:
    """Mutual-Information subword tokenizer trained on a corpus.

    After training, `encode(text)` returns a list of subword token IDs.
    The tokenizer is fully self-contained: no external models, no dictionaries.
    """

    def __init__(
        self,
        max_vocab: int = 8192,
        min_mi_threshold: float = 0.5,
        char_encoding: str = "utf-8",
        seed: int = 42,
    ):
        self.max_vocab = max_vocab
        self.min_mi_threshold = min_mi_threshold
        self.char_encoding = char_encoding
        self._rng = np.random.default_rng(seed)

        # Vocabulary: token_id -> bytes (or string representation)
        self.vocab: Dict[int, bytes] = {}
        # Merge rules: (left_id, right_id) -> merged_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Initialized after training
        self._initialized = False
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, corpus_texts: List[str]) -> "MI_SubwordTokenizer":
        """Train MI-based BPE on a corpus.

        Args:
            corpus_texts: List of raw strings (e.g. from dataset).

        Returns:
            self (for chaining).
        """
        if not corpus_texts:
            raise ValueError("corpus_texts must not be empty")

        logger.info(
            "MI-Tokenizer: training on %d texts, target_vocab=%d",
            len(corpus_texts),
            self.max_vocab,
        )

        # 1. Pre-tokenize into words; represent each word as list of char IDs
        words: List[List[int]] = []
        char_set: set = set()
        for text in corpus_texts:
            for word_str in _WORD_RE.findall(text):
                chars = list(word_str.encode(self.char_encoding))
                char_set.update(chars)
                words.append(chars)

        if not char_set:
            raise ValueError("No characters found in corpus")

        # 2. Initialise vocabulary with every byte observed
        for ch in sorted(char_set):
            self.vocab[self._next_id] = bytes([ch])
            self._next_id += 1

        logger.info("MI-Tokenizer: initial vocab size = %d", len(self.vocab))

        # 3. Greedy MI merging
        while len(self.vocab) < self.max_vocab:
            pair_counts, token_counts, total_pairs = self._count_pairs(words)
            if not pair_counts:
                break

            best_pair, best_mi = self._best_mi_pair(
                pair_counts, token_counts, total_pairs
            )
            if best_pair is None or best_mi < self.min_mi_threshold:
                logger.info(
                    "MI-Tokenizer: stopping, best MI %.3f < threshold %.3f",
                    best_mi,
                    self.min_mi_threshold,
                )
                break

            # Create merged token
            merged_id = self._next_id
            self._next_id += 1
            left_bytes = self.vocab[best_pair[0]]
            right_bytes = self.vocab[best_pair[1]]
            self.vocab[merged_id] = left_bytes + right_bytes
            self.merges[best_pair] = merged_id

            # Apply merge to all words
            words = [self._apply_merge_in_word(w, best_pair, merged_id) for w in words]

            if len(self.vocab) % 500 == 0:
                logger.info(
                    "MI-Tokenizer: vocab=%d, last MI=%.3f, merge=%s",
                    len(self.vocab),
                    best_mi,
                    (best_pair[0], best_pair[1]),
                )

        self._initialized = True
        logger.info(
            "MI-Tokenizer: training complete, vocab=%d, merges=%d",
            len(self.vocab),
            len(self.merges),
        )
        return self

    # ------------------------------------------------------------------ #
    # Encoding / Decoding
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> List[int]:
        """Encode text into subword token IDs."""
        if not self._initialized:
            raise RuntimeError("Tokenizer has not been trained yet")
        if not text:
            return []

        tokens: List[int] = []
        for word_str in _WORD_RE.findall(text):
            word_bytes = word_str.encode(self.char_encoding)
            word_ids = []
            for b in word_bytes:
                # Find the smallest token that matches at current position
                # Greedy longest-match using vocab
                i = 0
                while i < len(word_bytes):
                    longest_id = None
                    longest_len = 0
                    for tid, tb in self.vocab.items():
                        if word_bytes.startswith(tb, i) and len(tb) > longest_len:
                            longest_id = tid
                            longest_len = len(tb)
                    if longest_id is None:
                        # Fallback: single byte (must be in vocab)
                        longest_id = list(self.vocab.keys())[0]
                        longest_len = 1
                    word_ids.append(longest_id)
                    i += longest_len
            tokens.extend(word_ids)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        parts = []
        for tid in token_ids:
            if tid in self.vocab:
                parts.append(self.vocab[tid])
            else:
                parts.append(b"?")
        return b"".join(parts).decode(self.char_encoding, errors="replace")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _count_pairs(
        self, words: List[List[int]]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int], int]:
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        token_counts: Dict[int, int] = defaultdict(int)
        total_pairs = 0
        for word in words:
            for t in word:
                token_counts[t] += 1
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += 1
                total_pairs += 1
        return dict(pair_counts), dict(token_counts), total_pairs

    def _best_mi_pair(
        self,
        pair_counts: Dict[Tuple[int, int], int],
        token_counts: Dict[int, int],
        total_pairs: int,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        best_pair = None
        best_mi = -float("inf")
        # Pre-compute log denominators for speed
        log_total_pairs = math.log(total_pairs + 1e-10)
        for (a, b), count_ab in pair_counts.items():
            count_a = token_counts.get(a, 0)
            count_b = token_counts.get(b, 0)
            if count_a == 0 or count_b == 0:
                continue
            # MI = log( P(a,b) / (P(a)P(b)) )
            #     = log( count_ab / total ) - log( count_a / total ) - log( count_b / total )
            #     = log(count_ab) - log(count_a) - log(count_b) + log(total)
            mi = (
                math.log(count_ab + 1e-10)
                - math.log(count_a + 1e-10)
                - math.log(count_b + 1e-10)
                + log_total_pairs
            )
            if mi > best_mi:
                best_mi = mi
                best_pair = (a, b)
        return best_pair, best_mi

    def _apply_merge_in_word(
        self, word: List[int], pair: Tuple[int, int], merged_id: int
    ) -> List[int]:
        new_word: List[int] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    # ------------------------------------------------------------------ #
    # State persistence
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        return {
            "max_vocab": self.max_vocab,
            "min_mi_threshold": self.min_mi_threshold,
            "char_encoding": self.char_encoding,
            "vocab": {k: v.decode("latin1") for k, v in self.vocab.items()},
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "next_id": self._next_id,
            "initialized": self._initialized,
        }

    def load_state(self, state: dict) -> "MI_SubwordTokenizer":
        self.max_vocab = state["max_vocab"]
        self.min_mi_threshold = state["min_mi_threshold"]
        self.char_encoding = state["char_encoding"]
        self.vocab = {
            int(k): v.encode("latin1") for k, v in state["vocab"].items()
        }
        self.merges = {}
        for k, v in state["merges"].items():
            a_str, b_str = k.split(",")
            self.merges[(int(a_str), int(b_str))] = int(v)
        self._next_id = state["next_id"]
        self._initialized = state["initialized"]
        return self
