"""SOT v2.0 Integration — Self-contained embedder for RTMDK.

Usage:
    embedder = SOTv2Embedder(latent_dim=384)
    embedder.train(corpus_texts)          # one-time training on corpus
    emb = embedder("query text")          # returns np.ndarray (latent_dim,)
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Dict, List, Optional

import numpy as np

from .sif_embedder import SIFEmbedder

logger = logging.getLogger(__name__)


def _word_tokenize(text: str) -> List[str]:
    """Unicode-aware word tokenization."""
    text = text.lower()
    tokens = []
    current = []
    for ch in text:
        is_cjk = (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u309f"
            or "\u30a0" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        )
        if is_cjk:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for text in texts:
        for word in _word_tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class SOTv2Embedder:
    """Fully self-contained sentence embedder using SIF over PMI word vectors.

    No external model dependencies after training.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        window_size: int = 5,
        a: float = 1e-3,
        remove_pc: bool = True,
    ):
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.a = a
        self.remove_pc = remove_pc

        self._trained = False
        self._vocab: Dict[str, int] = {}
        self._embedder: Optional[SIFEmbedder] = None

    def train(self, corpus_texts: List[str]) -> "SOTv2Embedder":
        """Train on a corpus of raw texts.

        Args:
            corpus_texts: List of strings (e.g. all document texts + queries).
        """
        if not corpus_texts:
            raise ValueError("corpus_texts must not be empty")
        logger.info("SOTv2Embedder: training on %d texts", len(corpus_texts))

        self._vocab = _build_vocab(corpus_texts)
        logger.info("SOTv2Embedder: vocab size = %d", len(self._vocab))

        tokenized = [
            [self._vocab[w] for w in _word_tokenize(text)]
            for text in corpus_texts
        ]

        self._embedder = SIFEmbedder(
            latent_dim=self.latent_dim,
            window_size=self.window_size,
            min_count=1,
            a=self.a,
            remove_pc=self.remove_pc,
        )
        self._embedder.fit(tokenized, vocab_size=len(self._vocab))
        self._trained = True
        logger.info("SOTv2Embedder: training complete")
        return self

    def __call__(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not self._trained:
            raise RuntimeError("SOTv2Embedder has not been trained yet. Call .train() first.")
        tokens = [self._vocab[w] for w in _word_tokenize(text) if w in self._vocab]
        return self._embedder.embed(tokens)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.vstack([self(t) for t in texts])

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "window_size": self.window_size,
            "a": self.a,
            "remove_pc": self.remove_pc,
            "vocab": self._vocab,
            "embedder": self._embedder.get_state() if self._embedder else None,
        }

    def load_state(self, state: dict) -> "SOTv2Embedder":
        self.latent_dim = state["latent_dim"]
        self.window_size = state["window_size"]
        self.a = state["a"]
        self.remove_pc = state["remove_pc"]
        self._vocab = state["vocab"]
        if state["embedder"] is not None:
            self._embedder = SIFEmbedder().load_state(state["embedder"])
        self._trained = True
        return self
