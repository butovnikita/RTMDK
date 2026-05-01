"""
embedder_lmstudio.py — Real embedding via LM Studio API.

Uses text-embedding-nomic-embed-text-v1.5 from local LM Studio.
Falls back to hash-based embedder if LM Studio is unavailable.

Usage:
    from embedder_lmstudio import get_embedder
    embedder = get_embedder()
    vec = embedder("hello world")
"""

import os
import json
import time
import hashlib
from typing import Optional, Callable
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LMStudioEmbedder:
    """Real embedding via LM Studio with caching."""

    def __init__(self, url: str = "http://127.0.0.1:12345",
                 model: str = "text-embedding-nomic-embed-text-v1.5",
                 cache_size: int = 10000):
        self.url = url
        self.model = model
        self.dim = 768  # nomic-embed-text-v1.5 outputs 768d
        self._cache: dict = {}
        self._cache_size = cache_size
        self._available = self._check_available()
        if self._available:
            print(f"  LM Studio embedder: {self.url} (model: {self.model})")
        else:
            print(f"  LM Studio NOT available at {url}, using fallback")

    def _check_available(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            resp = requests.get(f"{self.url}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def __call__(self, text: str) -> np.ndarray:
        # Check cache
        if text in self._cache:
            return self._cache[text]

        if self._available:
            try:
                resp = requests.post(
                    f"{self.url}/v1/embeddings",
                    json={"model": self.model, "input": text},
                    timeout=15,
                )
                data = resp.json()
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
                self._cache[text] = embedding
                if len(self._cache) > self._cache_size:
                    # Drop oldest entries
                    keys_to_drop = list(self._cache.keys())[:self._cache_size // 4]
                    for k in keys_to_drop:
                        del self._cache[k]
                return embedding
            except Exception as e:
                print(f"  LM Studio embed error: {e}, using fallback")
                self._available = False

        # Fallback
        return _hash_embedder(text)


def _hash_embedder(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic keyword-based fallback."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal(dim).astype(np.float32) * 0.01
    tokens = text.lower().split()
    for tok in tokens[:20]:
        tok_rng = np.random.default_rng(hash(tok + "fallback_seed") % 2**32)
        direction = tok_rng.standard_normal(dim).astype(np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        base += direction * 0.5
    return base


def get_embedder() -> Callable[[str], np.ndarray]:
    """Get embedder — tries LM Studio first, falls back to hash."""
    embedder = LMStudioEmbedder()

    def embed(text: str) -> np.ndarray:
        return embedder(text)

    embed.dim = embedder.dim
    embed.is_real = embedder._available
    return embed
