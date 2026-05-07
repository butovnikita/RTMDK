"""rtmdk/production/redis_cache.py — Optional Redis-backed cache layer.

Provides distributed caching for query results and embeddings.
Falls back to in-memory caches when Redis is unavailable.
"""

import json
import pickle
from typing import Any, Optional

import numpy as np


def _get_redis_client(url: Optional[str] = None):
    try:
        import redis
    except ImportError:
        return None
    try:
        url = url or "redis://localhost:6379/0"
        client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


class RedisQueryCache:
    """Redis-backed query result cache with JSON serialization."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600, prefix: str = "rtmdk:query:"):
        self._client = _get_redis_client(redis_url)
        self.ttl = ttl
        self.prefix = prefix
        self._hits = 0
        self._misses = 0

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, query_hash: str) -> str:
        return f"{self.prefix}{query_hash}"

    def get(self, query_hash: str) -> Optional[Any]:
        if not self.available:
            return None
        try:
            raw = self._client.get(self._key(query_hash))
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(raw)
        except Exception:
            return None

    def set(self, query_hash: str, value: Any) -> None:
        if not self.available:
            return
        try:
            self._client.setex(self._key(query_hash), self.ttl, json.dumps(value))
        except Exception:
            pass

    def stats(self):
        return {"hits": self._hits, "misses": self._misses, "available": self.available}


class RedisEmbeddingCache:
    """Redis-backed embedding cache with pickle serialization for numpy arrays."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 86400, prefix: str = "rtmdk:emb:"):
        self._client = _get_redis_client(redis_url)
        self.ttl = ttl
        self.prefix = prefix
        self._hits = 0
        self._misses = 0

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, text_hash: str) -> str:
        return f"{self.prefix}{text_hash}"

    def get(self, text_hash: str) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            raw = self._client.get(self._key(text_hash))
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return pickle.loads(raw)
        except Exception:
            return None

    def set(self, text_hash: str, embedding: np.ndarray) -> None:
        if not self.available:
            return
        try:
            self._client.setex(self._key(text_hash), self.ttl, pickle.dumps(embedding))
        except Exception:
            pass

    def stats(self):
        return {"hits": self._hits, "misses": self._misses, "available": self.available}
