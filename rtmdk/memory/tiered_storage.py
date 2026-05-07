"""
rtmdk/memory/tiered_storage.py
Track 2: Tiered Storage
"""
from __future__ import annotations
import os, time, zlib, logging, threading
from typing import Dict, List, Optional, Iterator, Tuple, Any
from collections import OrderedDict
import numpy as np
from rtmdk.nodes import MemoryNode
logger = logging.getLogger(__name__)

def _msgpack_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")

class TieredNodeStore:
    def __init__(self, hot_limit: int, warm_limit: int, cold_dir: str, latent_dim: int):
        self.hot_limit = max(1, hot_limit)
        self.warm_limit = max(1, warm_limit)
        self.cold_dir = cold_dir
        self.latent_dim = latent_dim
        os.makedirs(cold_dir, exist_ok=True)
        self._hot: OrderedDict[str, MemoryNode] = OrderedDict()
        self._warm: Dict[str, Dict[str, Any]] = {}
        self._cold_batches: Dict[str, List[str]] = {}
        self._node_id_to_batch: Dict[str, str] = {}
        self._access_count: Dict[str, int] = {}
        self._tier: Dict[str, str] = {}
        self._lock = threading.RLock()

    def __getitem__(self, node_id: str) -> MemoryNode:
        with self._lock:
            if node_id in self._hot:
                self._access_count[node_id] = self._access_count.get(node_id, 0) + 1
                return self._hot[node_id]
            if node_id in self._warm:
                self._access_count[node_id] = self._access_count.get(node_id, 0) + 1
                node = self._warm_dict_to_node(node_id)
                self._promote_to_hot(node_id, node)
                return node
            if node_id in self._node_id_to_batch:
                self._access_count[node_id] = self._access_count.get(node_id, 0) + 1
                node = self._load_from_cold(node_id)
                self._promote_to_hot(node_id, node)
                return node
            raise KeyError(node_id)

    def __setitem__(self, node_id: str, node: MemoryNode) -> None:
        with self._lock:
            if not isinstance(node, MemoryNode):
                raise TypeError(f"TieredNodeStore only accepts MemoryNode, got {type(node)}")
            self._hot[node_id] = node
            self._access_count[node_id] = self._access_count.get(node_id, 0) + 1
            self._tier[node_id] = "hot"
            self._rebalance()

    def __delitem__(self, node_id: str) -> None:
        with self._lock:
            self._tier.pop(node_id, None)
            self._access_count.pop(node_id, None)
            self._hot.pop(node_id, None)
            self._warm.pop(node_id, None)
            batch = self._node_id_to_batch.pop(node_id, None)
            if batch and batch in self._cold_batches:
                self._cold_batches[batch] = [nid for nid in self._cold_batches[batch] if nid != node_id]
                if not self._cold_batches[batch]:
                    try: os.remove(batch)
                    except OSError: pass
                    del self._cold_batches[batch]

    def __contains__(self, node_id: str) -> bool:
        with self._lock: return node_id in self._tier

    def __len__(self) -> int:
        with self._lock: return len(self._tier)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            ids = list(self._tier.keys())
        return iter(ids)

    def keys(self) -> Iterator[str]: return self.__iter__()
    def values(self) -> Iterator[MemoryNode]:
        for nid in self.keys(): yield self[nid]
    def items(self) -> Iterator[Tuple[str, MemoryNode]]:
        for nid in self.keys(): yield nid, self[nid]

    def get(self, node_id: str, default=None):
        try: return self[node_id]
        except KeyError: return default

    def pop(self, node_id: str) -> MemoryNode:
        node = self[node_id]
        del self[node_id]
        return node

    def update(self, other: Dict[str, MemoryNode]) -> None:
        for nid, node in other.items():
            self[nid] = node

    def hot_items(self) -> Iterator[Tuple[str, MemoryNode]]:
        with self._lock:
            for nid, node in list(self._hot.items()):
                yield nid, node

    def hot_values(self) -> Iterator[MemoryNode]:
        with self._lock:
            for node in list(self._hot.values()):
                yield node

    def hot_keys(self) -> List[str]:
        with self._lock: return list(self._hot.keys())

    def warm_ids(self) -> List[str]:
        with self._lock: return list(self._warm.keys())

    def cold_ids(self) -> List[str]:
        with self._lock: return list(self._node_id_to_batch.keys())

    def cacheable_nodes(self) -> Iterator[Tuple[str, MemoryNode]]:
        yield from self.hot_items()
        warm_ids = list(self._warm.keys())
        for nid in warm_ids:
            yield nid, self._warm_dict_to_node(nid)

    def get_batch(self, node_ids: List[str]) -> List[MemoryNode]:
        result: List[MemoryNode] = []
        with self._lock:
            for nid in node_ids:
                if nid in self._hot: result.append(self._hot[nid])
                elif nid in self._warm: result.append(self._warm_dict_to_node(nid))
                elif nid in self._node_id_to_batch: result.append(self._load_from_cold(nid))
        return result

    def all_node_dicts(self) -> Iterator[Dict[str, Any]]:
        with self._lock:
            for node in self._hot.values(): yield node.to_dict()
            for d in self._warm.values(): yield d
            import msgpack
            for batch_path, nids in list(self._cold_batches.items()):
                if not os.path.exists(batch_path): continue
                with open(batch_path, "rb") as f:
                    data = zlib.decompress(f.read())
                batch = msgpack.unpackb(data, raw=False)
                for d in batch: yield d

    def _warm_dict_to_node(self, node_id: str) -> MemoryNode:
        return MemoryNode.from_dict(self._warm[node_id])

    @staticmethod
    def _node_to_warm_dict(node: MemoryNode) -> Dict[str, Any]:
        return node.to_dict()

    def _promote_to_hot(self, node_id: str, node: MemoryNode) -> None:
        self._hot[node_id] = node
        self._tier[node_id] = "hot"
        self._warm.pop(node_id, None)

    def _rebalance(self) -> None:
        if len(self._hot) > self.hot_limit:
            sorted_hot = sorted(self._hot.keys(), key=lambda nid: self._access_count.get(nid, 0))
            to_demote = sorted_hot[:len(self._hot) - self.hot_limit]
            for nid in to_demote:
                node = self._hot.pop(nid)
                self._warm[nid] = self._node_to_warm_dict(node)
                self._tier[nid] = "warm"
        if len(self._warm) > self.warm_limit:
            sorted_warm = sorted(self._warm.keys(), key=lambda nid: self._access_count.get(nid, 0))
            to_freeze = sorted_warm[:len(self._warm) - self.warm_limit]
            self._freeze_to_cold(to_freeze)

    def _freeze_to_cold(self, node_ids: List[str]) -> None:
        if not node_ids: return
        import msgpack
        batch_data, batch_ids = [], []
        for nid in node_ids:
            d = self._warm.pop(nid, None)
            if d is None: continue
            batch_data.append(d)
            batch_ids.append(nid)
            self._tier[nid] = "cold"
        if not batch_data: return
        packed = msgpack.packb(batch_data, use_bin_type=True, default=_msgpack_default)
        compressed = zlib.compress(packed)
        batch_name = f"cold_{int(time.time()*1000)}_{os.urandom(4).hex()}.msgpack"
        batch_path = os.path.join(self.cold_dir, batch_name)
        with open(batch_path, "wb") as f:
            f.write(compressed)
        self._cold_batches[batch_path] = batch_ids
        for nid in batch_ids:
            self._node_id_to_batch[nid] = batch_path
        logger.debug("TieredNodeStore: froze %d nodes to cold batch %s", len(batch_ids), batch_name)

    def _load_from_cold(self, node_id: str) -> MemoryNode:
        batch_path = self._node_id_to_batch[node_id]
        import msgpack
        with open(batch_path, "rb") as f:
            data = zlib.decompress(f.read())
        batch = msgpack.unpackb(data, raw=False)
        for d in batch:
            if d.get("id") == node_id:
                return MemoryNode.from_dict(d)
        raise KeyError(f"Node {node_id} not found in cold batch {batch_path}")

    def save_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hot_limit": self.hot_limit,
                "warm_limit": self.warm_limit,
                "latent_dim": self.latent_dim,
                "access_count": dict(self._access_count),
                "tier": dict(self._tier),
                "cold_batches": {k: v for k, v in self._cold_batches.items()},
                "node_id_to_batch": dict(self._node_id_to_batch),
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self.hot_limit = state.get("hot_limit", self.hot_limit)
            self.warm_limit = state.get("warm_limit", self.warm_limit)
            self._access_count = dict(state.get("access_count", {}))
            self._tier = dict(state.get("tier", {}))
            self._cold_batches = dict(state.get("cold_batches", {}))
            self._node_id_to_batch = dict(state.get("node_id_to_batch", {}))

    def clear_cold_storage(self) -> None:
        with self._lock:
            for batch_path in list(self._cold_batches.keys()):
                try: os.remove(batch_path)
                except OSError: pass
            self._cold_batches.clear()
            self._node_id_to_batch.clear()
