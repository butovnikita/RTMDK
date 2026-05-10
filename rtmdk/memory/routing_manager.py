"""RoutingManager — shard routing and shard center updates.

Extracted from RTMDKField to reduce monolithic field.py size.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RoutingManager:
    """Handles query-to-shard routing and shard center maintenance."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    def get_node_shard(self, node_id: str) -> int:
        """Get shard assignment for a node."""
        f = self.field
        if node_id in f._node_shard_map:
            return f._node_shard_map[node_id]
        if node_id in f.nodes:
            pos = f.nodes[node_id].latent_pos
            dists = np.linalg.norm(f._index_mgr.shard_centers - pos, axis=1)
            shard = int(np.argmin(dists))
            f._node_shard_map[node_id] = shard
            return shard
        return 0

    def route_query(self, query_latent: NDArray, top_shards: int = 3) -> List[int]:
        """Route query to top_k most relevant shards (softmax-free)."""
        f = self.field
        if f._index_mgr.shard_centers is None:
            return list(range(f.cfg.num_shards))
        dists = np.linalg.norm(f._index_mgr.shard_centers - query_latent, axis=1)
        f.shard_router = 1.0 / (1.0 + dists)
        return list(np.argsort(f.shard_router)[-top_shards:])

    def update_shard_centers(self) -> None:
        """Update shard centers based on current node distribution."""
        f = self.field
        if f._index_mgr.shard_centers is None or len(f.nodes) < f.cfg.num_shards:
            return
        from sklearn.cluster import KMeans
        positions = np.array([n.latent_pos for n in f.nodes.values()])
        if len(positions) < f.cfg.num_shards:
            return
        kmeans = KMeans(
            n_clusters=f.cfg.num_shards,
            n_init=3,
            random_state=42)
        labels = kmeans.fit_predict(positions)
        f._index_mgr.shard_centers = kmeans.cluster_centers_.astype(np.float32)
        f._node_shard_map.clear()
        for i, nid in enumerate(f.node_index):
            f._node_shard_map[nid] = int(labels[i])

    def update_shard_centers_bm25(self) -> None:
        """Build topic-based shards from BM25 term vectors."""
        f = self.field
        if f._index_mgr.bm25_index is None or len(f.nodes) < f.cfg.num_shards:
            return
        from collections import Counter
        term_doc: Dict[str, List[int]] = {}
        doc_terms: List[List[str]] = []
        nids = list(f.node_index)
        for i, nid in enumerate(nids):
            node = f.nodes[nid]
            text = node.content.get("text", "")
            terms = [w for w in text.lower().split() if len(w) > 2]
            doc_terms.append(terms)
            for t in set(terms):
                term_doc.setdefault(t, []).append(i)
        doc_cluster: Dict[int, str] = {}
        for i, terms in enumerate(doc_terms):
            if not terms:
                continue
            best_term = min(terms, key=lambda t: len(term_doc.get(t, [])))
            doc_cluster[i] = best_term
        groups: Dict[str, List[int]] = {}
        for i, term in doc_cluster.items():
            groups.setdefault(term, []).append(i)
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        clusters: List[List[int]] = []
        for term, members in sorted_groups:
            clusters.append(members)
        while len(clusters) > f.cfg.num_shards:
            clusters.sort(key=len)
            clusters[1].extend(clusters[0])
            clusters.pop(0)
        f._node_shard_map.clear()
        centers: List[NDArray] = []
        for shard_id, members in enumerate(clusters):
            for idx in members:
                f._node_shard_map[nids[idx]] = shard_id
            positions = np.array([f.nodes[nids[idx]].latent_pos for idx in members])
            centers.append(positions.mean(axis=0))
        f._index_mgr.shard_centers = np.stack(centers).astype(np.float32)
        logger.info("BM25 topic shards: %d clusters from %d docs", len(clusters), len(nids))
