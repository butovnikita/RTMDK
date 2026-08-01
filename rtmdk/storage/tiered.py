"""Tiered node storage prototype — Hot / Warm / Cold tiers.

Goal: Support unlimited node count without proportional RAM growth.

Tiers:
    Hot  (top 1% by frequency)  → RAM (dict)
    Warm (next 9%)              → numpy memmap on SSD
    Cold (remaining 90%)        → compressed msgpack snapshots on disk

Promotion/demotion based on LFU (Least Frequently Used) access counts.

Usage:
    store = TieredNodeStore(max_hot=100, max_warm=1000, cold_dir="./cold")
    store.put("node_1", {"text": "hello", "embedding": np.array(...)})
    node = store.get("node_1")  # auto-promotes on access
"""

from __future__ import annotations
import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _TieredEntry:
    """Internal entry tracking access frequency and tier."""

    key: str
    data: Dict[str, Any]
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    tier: str = "hot"  # hot / warm / cold


class TieredNodeStore:
    """Multi-tier node store with automatic promotion/demotion.

    Args:
        max_hot: Maximum number of nodes in hot tier (RAM)
        max_warm: Maximum number of nodes in warm tier (memmap)
        cold_dir: Directory for cold-tier compressed files
        latent_dim: Dimension of embeddings for memmap allocation
    """

    def __init__(
        self,
        max_hot: int = 1000,
        max_warm: int = 10_000,
        cold_dir: str = "./tiered_cold",
        latent_dim: int = 384,
    ):
        self.max_hot = max_hot
        self.max_warm = max_warm
        self.cold_dir = Path(cold_dir)
        self.cold_dir.mkdir(parents=True, exist_ok=True)
        self.latent_dim = latent_dim

        # Hot tier: in-memory dict
        self._hot: Dict[str, _TieredEntry] = {}

        # Warm tier: memmap file for embeddings + metadata dict
        self._warm_meta: Dict[str, Dict[str, Any]] = {}
        self._warm_path = self.cold_dir / "warm_embeddings.mmap"
        self._warm_mmap: Optional[np.memmap] = None
        self._warm_capacity = max_warm
        self._warm_index: Dict[str, int] = {}  # key -> row index
        self._warm_next_idx = 0
        self._init_warm_mmap()

        # Cold tier: compressed files
        self._cold_manifest: Dict[str, Dict[str, Any]] = {}
        self._manifest_path = self.cold_dir / "manifest.json"
        self._load_manifest()

        self._lock = threading.RLock()
        self._total_puts = 0
        self._total_gets = 0
        self._promotions = 0
        self._demotions = 0

    def _init_warm_mmap(self) -> None:
        """Initialize or resize warm-tier memmap."""
        shape = (self._warm_capacity, self.latent_dim)
        if self._warm_path.exists():
            existing = np.memmap(str(self._warm_path), dtype=np.float32, mode="r+", shape=shape)
            self._warm_mmap = existing
        else:
            self._warm_mmap = np.memmap(str(self._warm_path), dtype=np.float32, mode="w+", shape=shape)

    def _load_manifest(self) -> None:
        """Load cold-tier manifest from disk."""
        if self._manifest_path.exists():
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                self._cold_manifest = json.load(f)

    def _save_manifest(self) -> None:
        """Persist cold-tier manifest."""
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._cold_manifest, f)

    def _cold_path(self, key: str) -> Path:
        """Path to compressed cold file for a key."""
        return self.cold_dir / f"{key}.msgpack.gz"

    def _write_cold(self, key: str, data: Dict[str, Any]) -> None:
        """Write node data to cold tier (compressed JSON)."""
        path = self._cold_path(key)
        # Serialize latent_pos separately, rest as JSON-serializable dict
        import json as _json

        payload = {k: v for k, v in data.items() if k != "latent_pos"}
        if data.get("latent_pos") is not None:
            lp = data["latent_pos"]
            if isinstance(lp, np.ndarray):
                payload["latent_pos"] = lp.tolist()
            else:
                payload["latent_pos"] = lp
        # Convert numpy arrays to lists for JSON serialization
        for k, v in list(payload.items()):
            if isinstance(v, np.ndarray):
                payload[k] = v.tolist()
        with gzip.open(path, "wt", encoding="utf-8") as f:
            _json.dump(payload, f)
        self._cold_manifest[key] = {"size": path.stat().st_size, "tier": "cold"}

    def _read_cold(self, key: str) -> Optional[Dict[str, Any]]:
        """Read node data from cold tier."""
        path = self._cold_path(key)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("latent_pos"):
            payload["latent_pos"] = np.array(payload["latent_pos"], dtype=np.float32)
        return payload

    def _delete_cold(self, key: str) -> None:
        """Remove node from cold tier."""
        path = self._cold_path(key)
        if path.exists():
            path.unlink()
        self._cold_manifest.pop(key, None)

    def _demote_to_warm(self, entry: _TieredEntry) -> None:
        """Move a node from hot to warm tier."""
        if len(self._warm_meta) >= self._warm_capacity or self._warm_next_idx >= self._warm_capacity:
            # Warm full: evict oldest warm to cold and reuse its slot
            idx = self._evict_warm_to_cold()
        else:
            idx = self._warm_next_idx
            self._warm_next_idx += 1
        self._warm_index[entry.key] = idx
        # Store latent_pos vector in memmap; keep remaining metadata separate
        if entry.data.get("latent_pos") is not None:
            self._warm_mmap[idx] = entry.data["latent_pos"]
        self._warm_meta[entry.key] = {k: v for k, v in entry.data.items() if k != "latent_pos"}
        entry.tier = "warm"

    def _evict_warm_to_cold(self) -> int:
        """Evict least-frequently-used warm entry to cold.

        Returns the freed slot index for reuse.
        """
        if not self._warm_meta:
            # All warm slots freed (promoted back to hot); reset counter
            self._warm_next_idx = 0
            return 0
        # Find oldest / least accessed (simplified: oldest)
        lfu_key = min(self._warm_meta.keys(), key=lambda k: self._warm_meta[k].get("_last_access", 0))
        data = dict(self._warm_meta[lfu_key])
        idx = self._warm_index[lfu_key]
        data["latent_pos"] = np.array(self._warm_mmap[idx])
        self._write_cold(lfu_key, data)
        # Clear warm slot
        del self._warm_meta[lfu_key]
        del self._warm_index[lfu_key]
        self._warm_mmap[idx] = 0
        self._demotions += 1
        return idx

    def _promote_from_warm(self, key: str) -> Dict[str, Any]:
        """Load node from warm into hot."""
        idx = self._warm_index[key]
        latent_pos = np.array(self._warm_mmap[idx])
        data = dict(self._warm_meta[key])
        data["latent_pos"] = latent_pos
        # Clear warm slot
        del self._warm_meta[key]
        del self._warm_index[key]
        self._warm_mmap[idx] = 0
        self._promotions += 1
        return data

    def _promote_from_cold(self, key: str) -> Optional[Dict[str, Any]]:
        """Load node from cold into hot."""
        data = self._read_cold(key)
        if data is None:
            return None
        self._delete_cold(key)
        self._promotions += 1
        return data

    def put(self, key: str, data: Dict[str, Any]) -> None:
        """Store a node.  Automatically places in appropriate tier."""
        with self._lock:
            self._total_puts += 1
            if len(self._hot) >= self.max_hot:
                # Demote oldest hot to warm
                lfu_key = min(self._hot.keys(), key=lambda k: self._hot[k].access_count)
                self._demote_to_warm(self._hot.pop(lfu_key))
            entry = _TieredEntry(key=key, data=data, tier="hot")
            self._hot[key] = entry

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node.  Automatically promotes on access."""
        with self._lock:
            self._total_gets += 1
            # Check hot
            if key in self._hot:
                entry = self._hot[key]
                entry.access_count += 1
                entry.last_access = time.time()
                return entry.data

            # Check warm
            if key in self._warm_meta:
                data = self._promote_from_warm(key)
                entry = _TieredEntry(key=key, data=data, tier="hot")
                entry.access_count = 1
                # Make room in hot if needed
                if len(self._hot) >= self.max_hot:
                    lfu_key = min(self._hot.keys(), key=lambda k: self._hot[k].access_count)
                    self._demote_to_warm(self._hot.pop(lfu_key))
                self._hot[key] = entry
                return data

            # Check cold
            data = self._read_cold(key)
            if data is not None:
                self._delete_cold(key)
                entry = _TieredEntry(key=key, data=data, tier="hot")
                entry.access_count = 1
                if len(self._hot) >= self.max_hot:
                    lfu_key = min(self._hot.keys(), key=lambda k: self._hot[k].access_count)
                    self._demote_to_warm(self._hot.pop(lfu_key))
                self._hot[key] = entry
                return data

            return None

    def peek(self, key: str) -> Optional[Dict[str, Any]]:
        """Read a node WITHOUT promoting — used for batch query fallback.

        Checks hot, warm, and cold tiers in order but leaves the node in
        its current tier.  Much faster than get() for bulk reads.
        """
        with self._lock:
            # Hot
            if key in self._hot:
                return self._hot[key].data
            # Warm
            if key in self._warm_meta:
                idx = self._warm_index[key]
                data = dict(self._warm_meta[key])
                data["latent_pos"] = np.array(self._warm_mmap[idx])
                return data
            # Cold
            return self._read_cold(key)

    def peek_batch(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch peek without promotion."""
        with self._lock:
            results = []
            for key in keys:
                data = None
                if key in self._hot:
                    data = self._hot[key].data
                elif key in self._warm_meta:
                    idx = self._warm_index[key]
                    data = dict(self._warm_meta[key])
                    data["latent_pos"] = np.array(self._warm_mmap[idx])
                else:
                    data = self._read_cold(key)
                if data is not None:
                    results.append(data)
            return results

    def delete(self, key: str) -> bool:
        """Remove a node from all tiers."""
        with self._lock:
            found = False
            if key in self._hot:
                del self._hot[key]
                found = True
            if key in self._warm_meta:
                idx = self._warm_index[key]
                self._warm_mmap[idx] = 0
                del self._warm_meta[key]
                del self._warm_index[key]
                found = True
            if key in self._cold_manifest:
                self._delete_cold(key)
                found = True
            return found

    def keys(self) -> List[str]:
        """Return all node IDs across all tiers."""
        with self._lock:
            result = list(self._hot.keys())
            result.extend(self._warm_meta.keys())
            result.extend(self._cold_manifest.keys())
            return result

    def values(self) -> List[Dict[str, Any]]:
        """Return all node data across all tiers."""
        with self._lock:
            result = [entry.data for entry in self._hot.values()]
            for key in self._warm_meta:
                idx = self._warm_index[key]
                data = dict(self._warm_meta[key])
                data["latent_pos"] = np.array(self._warm_mmap[idx])
                result.append(data)
            for key in list(self._cold_manifest.keys()):
                data = self._read_cold(key)
                if data is not None:
                    result.append(data)
            return result

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return all (key, data) pairs across all tiers."""
        with self._lock:
            result = [(k, entry.data) for k, entry in self._hot.items()]
            for key in self._warm_meta:
                idx = self._warm_index[key]
                data = dict(self._warm_meta[key])
                data["latent_pos"] = np.array(self._warm_mmap[idx])
                result.append((key, data))
            for key in list(self._cold_manifest.keys()):
                data = self._read_cold(key)
                if data is not None:
                    result.append((key, data))
            return result

    def __iter__(self):
        """Iterate over all keys."""
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        """Check if key exists in any tier."""
        with self._lock:
            if key in self._hot:
                return True
            if key in self._warm_meta:
                return True
            if key in self._cold_manifest:
                return True
            return False

    def stats(self) -> Dict[str, Any]:
        """Return tier statistics."""
        with self._lock:
            return {
                "hot_count": len(self._hot),
                "warm_count": len(self._warm_meta),
                "cold_count": len(self._cold_manifest),
                "total_puts": self._total_puts,
                "total_gets": self._total_gets,
                "promotions": self._promotions,
                "demotions": self._demotions,
                "hot_capacity": self.max_hot,
                "warm_capacity": self._warm_capacity,
            }

    def close(self) -> None:
        """Persist manifest and close memmap.

        Demotes all hot and warm nodes to cold so they survive restart.
        """
        with self._lock:
            # Flush hot tier to cold
            for key, entry in list(self._hot.items()):
                self._write_cold(key, entry.data)
            self._hot.clear()
            # Flush warm tier to cold
            for key in list(self._warm_meta.keys()):
                idx = self._warm_index[key]
                data = dict(self._warm_meta[key])
                data["latent_pos"] = np.array(self._warm_mmap[idx])
                self._write_cold(key, data)
            self._warm_meta.clear()
            self._warm_index.clear()
            self._warm_next_idx = 0
            self._save_manifest()
            if self._warm_mmap is not None:
                del self._warm_mmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
