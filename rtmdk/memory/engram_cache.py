"""Engram Embedding Cache — hot/warm/cold tiered cache for engram retrieval.

Prevents TieredNodeStore from scanning cold disk layers during
engram similarity searches.
"""
from __future__ import annotations
import json
import os
import threading
from collections import OrderedDict
from typing import Dict, Optional, Iterator, Tuple
import numpy as np


class EngramEmbeddingCache:
    """Tiered in-memory cache of node embeddings for fast engram similarity.

    Hot tier: LRU cache for most frequently accessed embeddings.
    Warm tier: secondary cache for recently evicted from hot.
    Cold tier: not stored in memory (falls back to TieredNodeStore).
    """

    def __init__(self, max_hot: int = 10_000, max_warm: int = 90_000):
        self._hot: OrderedDict[str, np.ndarray] = OrderedDict()
        self._warm: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_hot = max_hot
        self._max_warm = max_warm
        self._lock = threading.RLock()

    def add(self, node_id: str, embedding: np.ndarray) -> None:
        """Add or update an embedding (goes to hot tier)."""
        with self._lock:
            self._hot[node_id] = embedding.copy()
            self._hot.move_to_end(node_id)
            while len(self._hot) > self._max_hot:
                oldest_nid, oldest_emb = self._hot.popitem(last=False)
                self._warm[oldest_nid] = oldest_emb
                self._warm.move_to_end(oldest_nid)
            while len(self._warm) > self._max_warm:
                self._warm.popitem(last=False)

    put = add  # Alias for compatibility

    def remove(self, node_id: str) -> None:
        """Remove an embedding from all tiers."""
        with self._lock:
            self._hot.pop(node_id, None)
            self._warm.pop(node_id, None)

    def get(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding by node id (promotes to hot on warm hit)."""
        with self._lock:
            emb = self._hot.get(node_id)
            if emb is not None:
                self._hot.move_to_end(node_id)
                return emb.copy()
            emb = self._warm.pop(node_id, None)
            if emb is not None:
                self._hot[node_id] = emb
                self._hot.move_to_end(node_id)
                if len(self._hot) > self._max_hot:
                    oldest_nid, oldest_emb = self._hot.popitem(last=False)
                    self._warm[oldest_nid] = oldest_emb
                    self._warm.move_to_end(oldest_nid)
                    while len(self._warm) > self._max_warm:
                        self._warm.popitem(last=False)
                return emb.copy()
            return None

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return a shallow copy of hot + warm caches."""
        with self._lock:
            result = dict(self._warm)
            result.update(self._hot)
            return {k: v.copy() for k, v in result.items()}

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over cached embeddings (snapshot)."""
        with self._lock:
            snapshot = list(self._hot.items()) + list(self._warm.items())
        for nid, emb in snapshot:
            yield nid, emb.copy()

    def __contains__(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._hot or node_id in self._warm

    def __len__(self) -> int:
        with self._lock:
            return len(self._hot) + len(self._warm)

    def clear(self) -> None:
        with self._lock:
            self._hot.clear()
            self._warm.clear()

    def save(self, path: str) -> None:
        """Serialize cache to NPZ file."""
        with self._lock:
            data = {k: v for k, v in {**self._warm, **self._hot}.items()}
        if not data:
            np.savez(path, _empty=True)
            return
        np.savez(path, **{k.replace("/", "_"): v for k, v in data.items()})

    def load(self, path: str) -> None:
        """Deserialize cache from NPZ file."""
        if not os.path.exists(path):
            return
        with self._lock:
            self._hot.clear()
            self._warm.clear()
            archive = np.load(path, allow_pickle=False)
            if "_empty" in archive:
                return
            for key in archive.files:
                nid = key.replace("_", "/")
                self._hot[nid] = archive[key]
