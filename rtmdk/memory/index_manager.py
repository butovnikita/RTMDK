"""IndexManager — encapsulates HNSW, BM25, and sparse shard routing indices."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from rtmdk.memory.config import RTMDKConfig
from rtmdk.support.bm25 import BM25Index
from rtmdk.support.hnsw import NaiveGraphIndex

try:
    from rtmdk.support.hnsw_lib import HNSWLibIndex
    _HAS_HNSWLIB = True
except Exception:
    _HAS_HNSWLIB = False

try:
    from rtmdk.memory.async_index import AsyncIndexBuilder
except Exception:
    AsyncIndexBuilder = None  # type: ignore

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


class IndexManager:
    """Manages approximate nearest-neighbour and inverted indices."""

    def __init__(
        self,
        cfg: RTMDKConfig,
        latent_dim: int,
        rng: np.random.Generator,
        quant: Any,
    ):
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.rng = rng
        self._quant = quant

        # BM25 inverted index
        self.bm25_index: Optional[BM25Index] = None
        if cfg.bm25_fallback or getattr(cfg, "bm25_first_stage_k", 0) > 0:
            self.bm25_index = BM25Index(cfg.bm25_k1, cfg.bm25_b)

        # HNSW graph index
        self.hnsw_index: Optional[Any] = None
        if cfg.use_hnsw:
            if _HAS_HNSWLIB:
                self.hnsw_index = HNSWLibIndex(
                    dim=latent_dim,
                    m=cfg.hnsw_m,
                    ef_construction=cfg.hnsw_ef_construction,
                )
            else:
                self.hnsw_index = NaiveGraphIndex(
                    m=cfg.hnsw_m,
                    ef_construction=cfg.hnsw_ef_construction,
                )

        # Async builder wrapper
        self._async_builder: Optional[Any] = None
        if cfg.use_hnsw and getattr(cfg, "async_hnsw_build", False) and self.hnsw_index and AsyncIndexBuilder:
            self._async_builder = AsyncIndexBuilder(
                self.hnsw_index,
                interval_ms=cfg.async_hnsw_interval_ms,
                batch_size=cfg.async_hnsw_batch_size,
            )

        # Sparse shard routing
        self.shard_centers: Optional[NDArray] = None
        if cfg.sparse_routing and cfg.num_shards > 0:
            self.shard_centers = rng.standard_normal(
                (cfg.num_shards, latent_dim), dtype=np.float32)
            self.shard_centers /= np.linalg.norm(
                self.shard_centers, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # HNSW operations
    # ------------------------------------------------------------------
    def hnsw_search(self, query_latent: NDArray, top_k: int) -> List[str]:
        if self.hnsw_index is None:
            return []
        min_nodes = getattr(self.cfg, "hnsw_min_nodes", 50)
        if len(self.hnsw_index.positions) <= min_nodes:
            return []
        return self.hnsw_index.search(query_latent, top_k)

    def hnsw_insert(self, node_id: str, latent: NDArray) -> None:
        if self.hnsw_index is None:
            return
        if self._async_builder:
            self._async_builder.submit(node_id, latent)
        else:
            self.hnsw_index.insert(node_id, latent)

    def hnsw_insert_batch(self, node_ids: List[str], latents: NDArray) -> None:
        if self.hnsw_index is None:
            return
        if self._async_builder:
            self._async_builder.submit_batch(node_ids, latents)
        elif hasattr(self.hnsw_index, "insert_batch"):
            self.hnsw_index.insert_batch(node_ids, latents)
        else:
            for nid, latent in zip(node_ids, latents):
                self.hnsw_index.insert(nid, latent)

    def hnsw_remove(self, node_id: str) -> None:
        if self.hnsw_index is None:
            return
        if self._async_builder:
            self._async_builder.remove(node_id)
        else:
            self.hnsw_index.remove(node_id)

    def hnsw_count(self) -> int:
        return len(self.hnsw_index.positions) if self.hnsw_index else 0

    # ------------------------------------------------------------------
    # BM25 operations
    # ------------------------------------------------------------------
    def bm25_add(self, node_id: str, text: str) -> None:
        if self.bm25_index is not None and text:
            self.bm25_index.add_document(node_id, text)

    def bm25_remove(self, node_id: str) -> None:
        if self.bm25_index is not None:
            self.bm25_index.remove_document(node_id)

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, top_k)

    # ------------------------------------------------------------------
    # Shard routing
    # ------------------------------------------------------------------
    def route_query(self, query_latent: NDArray, top_shards: int) -> List[int]:
        if self.shard_centers is None:
            return []
        dists = np.linalg.norm(self.shard_centers - query_latent, axis=1)
        return np.argsort(dists)[:top_shards].tolist()

    def get_node_shard(self, node_id: str, node: Any) -> int:
        if self.shard_centers is None:
            return 0
        pos = getattr(node, "latent_pos", None)
        if pos is None:
            return 0
        if self._quant.mode == "int8":
            pos = self._quant.dequantize(
                pos,
                getattr(node, "latent_scale", 1.0),
                getattr(node, "latent_zero_point", 0.0),
            )
        dists = np.linalg.norm(self.shard_centers - pos, axis=1)
        return int(np.argmin(dists))

    def update_shard_centers(self, nodes: Dict[str, Any], node_index: List[str], num_shards: int) -> None:
        if self.shard_centers is None or len(nodes) < num_shards:
            return
        if not _HAS_SKLEARN:
            return
        positions = []
        for nid in node_index:
            if nid not in nodes:
                continue
            node = nodes[nid]
            pos = getattr(node, "latent_pos", None)
            if pos is None:
                continue
            if self._quant.mode == "int8":
                pos = self._quant.dequantize(
                    pos,
                    getattr(node, "latent_scale", 1.0),
                    getattr(node, "latent_zero_point", 0.0),
                )
            positions.append(pos)
        if len(positions) < num_shards:
            return
        kmeans = MiniBatchKMeans(
            n_clusters=num_shards,
            random_state=42,
            n_init=3,
            max_iter=100,
        )
        kmeans.fit(np.stack(positions, axis=0))
        self.shard_centers = kmeans.cluster_centers_.astype(np.float32)

    def update_shard_centers_bm25(self, nodes: Dict[str, Any], node_index: List[str], num_shards: int) -> None:
        if self.shard_centers is None or self.bm25_index is None or len(nodes) < num_shards:
            return
        vocab = sorted(self.bm25_index.doc_freq.keys())
        if not vocab:
            return
        vectors = []
        for nid in node_index:
            if nid not in nodes:
                continue
            vec = np.zeros(len(vocab), dtype=np.float32)
            for token, tf in self.bm25_index.inverted_index.items():
                if nid in tf:
                    idx = vocab.index(token)
                    vec[idx] = tf[nid]
            if vec.sum() > 0:
                vectors.append(vec)
        if len(vectors) < num_shards:
            return
        if not _HAS_SKLEARN:
            return
        kmeans = MiniBatchKMeans(
            n_clusters=num_shards,
            random_state=42,
            n_init=3,
            max_iter=100,
        )
        kmeans.fit(np.stack(vectors, axis=0))
        proj = self.rng.standard_normal(
            (num_shards, self.latent_dim), dtype=np.float32)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)
        self.shard_centers = (kmeans.cluster_centers_ @ proj).astype(np.float32)

    # ------------------------------------------------------------------
    # Stats / helpers
    # ------------------------------------------------------------------
    def flush_async(self) -> None:
        if self._async_builder:
            self._async_builder.flush()

    def close_async(self) -> None:
        if self._async_builder and hasattr(self._async_builder, "close"):
            self._async_builder.close()

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self.hnsw_index is not None and hasattr(self.hnsw_index, "get_state"):
            state["hnsw"] = self.hnsw_index.get_state()
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if self.hnsw_index is not None and "hnsw" in state and hasattr(self.hnsw_index, "load_state"):
            self.hnsw_index.load_state(state["hnsw"])
