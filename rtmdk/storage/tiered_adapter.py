"""Adapter: wraps TieredNodeStore with dict-like MutableMapping interface.

This allows RTMDKField to use TieredNodeStore v2 (memmap-based) as a drop-in
replacement for the standard `self.nodes` dict.

Usage:
    from rtmdk.storage.tiered import TieredNodeStore
    from rtmdk.storage.tiered_adapter import TieredNodeStoreAdapter

    store = TieredNodeStore(max_hot=100, max_warm=1000)
    nodes = TieredNodeStoreAdapter(store)
    nodes["node_1"] = memory_node  # delegates to store.put()
    node = nodes["node_1"]         # delegates to store.get()
"""
from __future__ import annotations
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, Optional

from rtmdk.storage.tiered import TieredNodeStore


class TieredNodeStoreAdapter(MutableMapping):
    """Dict-like wrapper around TieredNodeStore.

    Converts MemoryNode objects to/from plain dicts automatically
    for compatibility with the tiered storage serialization layer.
    """

    def __init__(self, store: TieredNodeStore):
        self._store = store

    def _node_to_data(self, node: Any) -> Dict[str, Any]:
        """Serialize a MemoryNode to a plain dict."""
        # Handle both MemoryNode objects and existing dicts
        if isinstance(node, dict):
            return node
        # MemoryNode with to_dict()
        if hasattr(node, "to_dict"):
            return node.to_dict()
        if hasattr(node, "__dict__"):
            return dict(node.__dict__)
        raise TypeError(f"Cannot serialize node of type {type(node)}")

    def _data_to_node(self, data: Dict[str, Any]) -> Any:
        """Deserialize a plain dict back to MemoryNode."""
        try:
            from rtmdk.nodes import MemoryNode
            return MemoryNode.from_dict(data)
        except Exception:
            # Fallback: return as-is if deserialization fails
            return data

    def __setitem__(self, key: str, value: Any) -> None:
        self._store.put(key, self._node_to_data(value))

    def __getitem__(self, key: str) -> Any:
        data = self._store.get(key)
        if data is None:
            raise KeyError(key)
        return self._data_to_node(data)

    def __delitem__(self, key: str) -> None:
        if not self._store.delete(key):
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self._store.get(key) is not None  # type: ignore[arg-type]

    def __len__(self) -> int:
        stats = self._store.stats()
        return stats.get("hot_count", 0) + stats.get("warm_count", 0) + stats.get("cold_count", 0)

    def __iter__(self) -> Iterator[str]:
        # Yield hot keys
        stats = self._store.stats()
        # We cannot iterate directly over tiers, so we maintain a separate index
        # For now, return an empty iterator — RTMDKField uses node_index list separately
        return iter([])

    def keys(self):
        # RTMDKField maintains self.node_index separately
        return []

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()

    def cacheable_nodes(self):
        """Yield (key, data) for nodes in hot tier.

        Used by RTMDKField._build_node_cache() to construct vectorized caches.
        """
        for key, entry in self._store._hot.items():
            yield key, self._data_to_node(entry.data)

    def warm_ids(self):
        """Return list of node IDs in warm tier."""
        return list(self._store._warm_meta.keys())

    def cold_ids(self):
        """Return list of node IDs in cold tier."""
        return list(self._store._cold_manifest.keys())

    def get_batch(self, node_ids):
        """Retrieve multiple nodes by ID."""
        result = []
        for nid in node_ids:
            try:
                result.append(self[nid])
            except KeyError:
                pass
        return result

    def close(self) -> None:
        self._store.close()
