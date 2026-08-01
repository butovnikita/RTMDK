"""Tests for rtmdk/utils/async_embedder.py — async batching wrapper."""

import numpy as np
import pytest

from rtmdk.utils.async_embedder import AsyncEmbedder

pytestmark = pytest.mark.asyncio


def hash_embedder(text: str) -> np.ndarray:
    """Deterministic local embedder."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(8).astype(np.float32)


class TestEmbed:
    async def test_single_embed_returns_vector(self):
        emb = AsyncEmbedder(hash_embedder, batch_size=16, max_wait_ms=5)
        result = await emb.embed("hello")

        np.testing.assert_array_equal(result, hash_embedder("hello"))

    async def test_batch_matches_sync(self):
        emb = AsyncEmbedder(hash_embedder, batch_size=16, max_wait_ms=5)
        texts = [f"text {i}" for i in range(5)]

        results = await emb.embed_batch(texts)

        assert len(results) == len(texts)
        for text, result in zip(texts, results):
            np.testing.assert_array_equal(result, hash_embedder(text))

    async def test_full_batch_flushes_immediately(self):
        calls = []

        def tracking_embedder(text):
            calls.append(text)
            return hash_embedder(text)

        emb = AsyncEmbedder(tracking_embedder, batch_size=2, max_wait_ms=60_000)
        # Two concurrent requests fill the batch → flush without waiting max_wait_ms
        results = await emb.embed_batch(["a", "b"])

        assert len(results) == 2
        assert sorted(calls) == ["a", "b"]

    async def test_delayed_flush_within_max_wait(self):
        emb = AsyncEmbedder(hash_embedder, batch_size=100, max_wait_ms=10)
        # Single request: must be served by the delayed flush, not hang
        result = await emb.embed("lonely")
        assert result.shape == (8,)

    async def test_embedder_exception_propagates(self):
        def failing_embedder(text):
            raise ValueError("embedder exploded")

        emb = AsyncEmbedder(failing_embedder, batch_size=16, max_wait_ms=5)
        with pytest.raises(ValueError, match="embedder exploded"):
            await emb.embed("boom")

    async def test_concurrent_load_all_served(self):
        emb = AsyncEmbedder(hash_embedder, batch_size=4, max_wait_ms=5)
        texts = [f"load {i}" for i in range(10)]

        results = await emb.embed_batch(texts)

        assert len(results) == 10
        for text, result in zip(texts, results):
            np.testing.assert_array_equal(result, hash_embedder(text))
