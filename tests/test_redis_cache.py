"""Tests for Redis cache layer."""

import numpy as np
import pytest

from rtmdk.production.redis_cache import RedisQueryCache, RedisEmbeddingCache


class FakeRedis:
    """In-memory fake Redis for testing without a real server."""

    def __init__(self):
        self._data = {}
        self._ttls = {}

    def ping(self):
        return True

    def get(self, key):
        return self._data.get(key)

    def setex(self, key, ttl, value):
        self._data[key] = value
        self._ttls[key] = ttl

    def delete(self, key):
        self._data.pop(key, None)


@pytest.fixture
def fake_query_cache():
    cache = RedisQueryCache()
    cache._client = FakeRedis()
    return cache


@pytest.fixture
def fake_emb_cache():
    cache = RedisEmbeddingCache()
    cache._client = FakeRedis()
    return cache


class TestRedisQueryCache:
    def test_get_miss(self, fake_query_cache):
        assert fake_query_cache.get("missing") is None
        assert fake_query_cache.stats()["misses"] == 1

    def test_set_and_get(self, fake_query_cache):
        fake_query_cache.set("q1", {"results": ["a", "b"]})
        val = fake_query_cache.get("q1")
        assert val == {"results": ["a", "b"]}
        assert fake_query_cache.stats()["hits"] == 1

    def test_ttl_set(self, fake_query_cache):
        fake_query_cache.set("q2", [1, 2, 3])
        assert fake_query_cache._client._ttls["rtmdk:query:q2"] == 3600

    def test_unavailable_returns_none(self):
        cache = RedisQueryCache(redis_url="redis://invalid:9999")
        assert not cache.available
        assert cache.get("anything") is None


class TestRedisEmbeddingCache:
    def test_get_miss(self, fake_emb_cache):
        assert fake_emb_cache.get("missing") is None
        assert fake_emb_cache.stats()["misses"] == 1

    def test_set_and_get(self, fake_emb_cache):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_emb_cache.set("emb1", vec)
        out = fake_emb_cache.get("emb1")
        assert out is not None
        np.testing.assert_allclose(out, vec)
        assert fake_emb_cache.stats()["hits"] == 1

    def test_unavailable_returns_none(self):
        cache = RedisEmbeddingCache(redis_url="redis://invalid:9999")
        assert not cache.available
        assert cache.get("anything") is None
