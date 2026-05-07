"""
rtmdk/production/embedding_cache.py — Disk-Based Embedding Cache.

Caches embeddings on disk to avoid redundant API calls.
Features:
- Hash-based keys (MD5 of text)
- LRU eviction with max size
- TTL support (time-to-live)
- Hit/miss ratio statistics
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict
import numpy as np


class EmbeddingCache:
    """Disk and memory cache for text embeddings.

    Usage:
        cache = EmbeddingCache(cache_dir="~/.rtmdk/embedding_cache", max_size=100000)

        # Get embedding (from cache or via embedder)
        emb = cache.get_or_compute("hello world", embedder)

        # Stats
        stats = cache.get_stats()  # hit_rate, size, etc.
    """

    def __init__(
        self,
        cache_dir: str = "~/.rtmdk/embedding_cache",
        max_size: int = 100000,
        ttl_seconds: int = 86400 * 30,  # 30 days default
        memory_cache_size: int = 10000,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.memory_cache: OrderedDict[str,
                                       Tuple[np.ndarray, float]] = OrderedDict()
        self.memory_cache_size = memory_cache_size

        self._hits = 0
        self._misses = 0
        self._load_disk_index()

    def get_or_compute(self, text: str, embedder) -> np.ndarray:
        """Get embedding from cache or compute via embedder.

        Args:
            text: Text to embed
            embedder: Function that takes text and returns np.ndarray

        Returns:
            Embedding vector
        """
        key = self._make_key(text)

        # Check memory cache first
        if key in self.memory_cache:
            emb, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.ttl:
                self._hits += 1
                self.memory_cache.move_to_end(key)
                return emb
            else:
                del self.memory_cache[key]

        # Check disk cache
        emb = self._load_from_disk(key)
        if emb is not None:
            self._hits += 1
            self._save_to_memory_cache(key, emb)
            return emb

        # Compute via embedder
        self._misses += 1
        emb = embedder(text)

        # Save to caches
        self._save_to_memory_cache(key, emb)
        self._save_to_disk(key, emb)

        return emb

    def get_or_compute_batch(self,
                             texts: List[str],
                             embedder) -> List[np.ndarray]:
        """Batch version — more efficient for multiple texts."""
        results = []
        uncached_texts = []
        uncached_keys = []

        # Check cache for each text
        for text in texts:
            key = self._make_key(text)

            if key in self.memory_cache:
                emb, ts = self.memory_cache[key]
                if time.time() - ts < self.ttl:
                    self._hits += 1
                    self.memory_cache.move_to_end(key)
                    results.append(emb)
                    continue

            emb = self._load_from_disk(key)
            if emb is not None:
                self._hits += 1
                self._save_to_memory_cache(key, emb)
                results.append(emb)
                continue

            uncached_texts.append(text)
            uncached_keys.append(key)
            results.append(None)  # Placeholder

        # Compute uncached embeddings
        if uncached_texts:
            uncached_embs = [embedder(t) for t in uncached_texts]
            self._misses += len(uncached_texts)

            for i, (key, emb) in enumerate(zip(uncached_keys, uncached_embs)):
                results[results.index(None)] = emb
                self._save_to_memory_cache(key, emb)
                self._save_to_disk(key, emb)

        return results

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            index_path.unlink()
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        disk_count = len(list(self.cache_dir.glob("*.npy")))

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": disk_count,
            "total_requests": total,
            "max_size": self.max_size,
        }

    def _make_key(self, text: str) -> str:
        """Create cache key from text."""
        return hashlib.md5(
            text.encode('utf-8'),
            usedforsecurity=False).hexdigest()

    def _save_to_memory_cache(self, key: str, emb: np.ndarray):
        """Save to in-memory LRU cache."""
        if len(self.memory_cache) >= self.memory_cache_size:
            self.memory_cache.popitem(last=False)
        self.memory_cache[key] = (emb.copy(), time.time())

    def _save_to_disk(self, key: str, emb: np.ndarray):
        """Save embedding to disk."""
        filepath = self.cache_dir / f"{key}.npy"
        np.save(filepath, emb)

        # Update index
        index_path = self.cache_dir / "index.json"
        try:
            if index_path.exists():
                with open(index_path) as f:
                    index = json.load(f)
            else:
                index = {}
            index[key] = {"saved_at": time.time(), "shape": list(emb.shape)}

            # Evict oldest if over max_size
            if len(index) > self.max_size:
                oldest_key = min(index, key=lambda k: index[k]["saved_at"])
                del index[oldest_key]
                oldest_file = self.cache_dir / f"{oldest_key}.npy"
                if oldest_file.exists():
                    oldest_file.unlink()

            with open(index_path, 'w') as f:
                json.dump(index, f)
        except (json.JSONDecodeError, IOError):
            pass  # Best-effort indexing

    def _load_from_disk(self, key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache."""
        filepath = self.cache_dir / f"{key}.npy"
        if filepath.exists():
            try:
                emb = np.load(filepath)
                return emb
            except (IOError, ValueError):
                return None
        return None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    def _load_disk_index(self):
        """Load disk index (for stats)."""
        pass  # Index is loaded lazily when saving
