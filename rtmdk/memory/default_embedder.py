"""Batteries-included default embedder.

New users should be able to ``pip install rtmdk`` and get a working memory
in three lines — without LM Studio, OpenAI keys, GPU, or model downloads.
This module provides a deterministic, zero-dependency embedder backed by
the built-in word-level SOT tokenizer (lexical retrieval; ~98% R@1 on the
QA benchmark vs SBERT baseline, see tests/test_sot_benchmark.py).

For production-grade semantic quality, pass any callable
``embedder(text) -> NDArray`` (LM Studio, OpenAI, SBERT, ...) or train the
built-in SOT v2 embedder via ``memory.train_sot_v2(...)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from rtmdk.memory.self_organizing_field import SOTokenizer


def create_default_embedder(dim: int = 768, seed: int = 42) -> Callable[[str], NDArray[np.float32]]:
    """Return a zero-dependency ``embedder(text) -> (dim,) float32`` callable.

    Word-level SOT tokenizer: identical tokens get identical embeddings, so
    lexical-overlap retrieval works out of the box. Deterministic (seeded).
    """
    tokenizer = SOTokenizer(
        latent_dim=dim,
        token_dim=dim,
        tokenization_mode="word",
        seed=seed,
    )

    def embed(text: str) -> NDArray[np.float32]:
        return tokenizer.embed(tokenizer.encode(text)).astype(np.float32)

    embed.tokenizer = tokenizer  # type: ignore[attr-defined]  # introspection/online learning
    return embed
