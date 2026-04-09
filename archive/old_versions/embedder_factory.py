"""
embedder_factory.py
Unified embedder factory for RTMDK.

Provides consistent embedding interface across tests, server, and CLI.
Supports:
  - DummyEmbedder: deterministic hash-based (for unit tests)
  - LmStudioEmbedder: real embeddings via LM Studio API (for integration tests)
  - SentenceTransformerEmbedder: local sentence-transformers (optional)

Usage:
    from embedder_factory import EmbedderFactory

    # Dummy (deterministic, no network)
    embedder = EmbedderFactory.create("dummy", dim=768)

    # LM Studio (requires running server)
    embedder = EmbedderFactory.create("lmstudio", url="http://localhost:12345/v1")

    # Sentence Transformers (requires pip install sentence-transformers)
    embedder = EmbedderFactory.create("sentence", model="all-MiniLM-L6-v2")
"""

import os
import hashlib
from typing import Optional, Callable
import numpy as np


class EmbedderFactory:
    """Factory for creating embedder functions."""

    @staticmethod
    def create(mode: str = "dummy", **kwargs) -> Callable[[str], np.ndarray]:
        """
        Create an embedder function.

        Args:
            mode: "dummy", "lmstudio", "sentence"
            **kwargs: mode-specific parameters

        Returns:
            Callable[[str], np.ndarray]
        """
        if mode == "dummy":
            return DummyEmbedder(
                dim=kwargs.get("dim", 768),
                seed=kwargs.get("seed", 42),
            )
        elif mode == "lmstudio":
            return LmStudioEmbedder(
                url=kwargs.get("url", "http://localhost:12345/v1"),
                model=kwargs.get("model", "nomic-ai/nomic-embed-text-v1.5-GGUF"),
                dim=kwargs.get("dim", 768),
                timeout=kwargs.get("timeout", 30),
            )
        elif mode == "sentence":
            return SentenceTransformerEmbedder(
                model=kwargs.get("model", "all-MiniLM-L6-v2"),
                dim=kwargs.get("dim", 384),
            )
        else:
            raise ValueError(f"Unknown embedder mode: {mode}. Use 'dummy', 'lmstudio', or 'sentence'.")


class DummyEmbedder:
    """Deterministic hash-based embedder for unit tests."""

    def __init__(self, dim: int = 768, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def __call__(self, text: str) -> np.ndarray:
        np.random.seed(int(hashlib.md5(f"{text}{self.seed}".encode()).hexdigest(), 16) % 2**32)
        base = np.random.randn(self.dim).astype(np.float32) * 0.1
        # Add deterministic signal for text matching
        sig = np.array([int(hashlib.md5(f"{text}{i}".encode()).hexdigest(), 16) % 1000 / 500.0 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base

    def __repr__(self):
        return f"DummyEmbedder(dim={self.dim})"


class LmStudioEmbeder:
    """Real embeddings via LM Studio API for integration tests."""

    def __init__(self, url: str = "http://localhost:12345/v1",
                 model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF",
                 dim: int = 768, timeout: int = 30):
        self.url = url
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self._cache: dict = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        import requests
        try:
            resp = requests.post(
                f"{self.url}/embeddings",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            data = resp.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            self._cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"\n  [WARN] LM Studio embedding error: {e}, using fallback")
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(self.dim).astype(np.float32) * 0.1
            self._cache[text] = emb
            return emb

    def __repr__(self):
        return f"LmStudioEmbedder(url={self.url!r}, model={self.model!r})"


# Alias for backward compatibility (typo in class name preserved)
LmStudioEmbedder = LmStudioEmbeder


class SentenceTransformerEmbedder:
    """Local sentence-transformers embeddings (optional dependency)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model
        self.dim = dim
        self._model = None
        self._cache: dict = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def __call__(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        self._cache[text] = embedding
        return embedding

    def __repr__(self):
        return f"SentenceTransformerEmbedder(model={self.model_name!r})"


# Convenience function for simple imports
def get_dummy_embedder(dim: int = 768) -> Callable[[str], np.ndarray]:
    return DummyEmbedder(dim=dim)


def get_lmstudio_embedder(url: str = "http://localhost:12345/v1") -> Callable[[str], np.ndarray]:
    return LmStudioEmbedder(url=url)
