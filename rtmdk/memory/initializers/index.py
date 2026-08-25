"""IndexInitializer — R10.1 split from FieldInitializer.

Handles tiered storage, HNSW/BM25/shard routing, crystallization, async pipeline.
See field_initializer.py:92 and docs/RISKS.md R10.1.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.index_manager import IndexManager

logger = logging.getLogger(__name__)


class IndexInitializer:
    """Wires index and storage subsystems."""

    def __init__(self, field: "RTMDKField", config: RTMDKConfig, projection_matrix=None, wal_path=None):
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path

    def initialize(self) -> None:
        self._init_tiered_storage()
        self._init_index_manager()
        self.field._batch_resonance_fn = None  # populated by QueryManager later
        self._init_sparse_routing()
        self._init_crystallization_counters()
        self._init_async_pipeline()

    def _init_tiered_storage(self) -> None:
        f = self.field
        cfg = self.cfg
        f._tiered_store = None
        if cfg.tiered_storage_v2_enabled:
            from rtmdk.storage.tiered import TieredNodeStore
            from rtmdk.storage.tiered_adapter import TieredNodeStoreAdapter

            hot_limit = max(1, int(cfg.max_nodes * cfg.tiered_hot_pct)) if cfg.max_nodes else 100
            warm_limit = max(1, int(cfg.max_nodes * cfg.tiered_warm_pct)) if cfg.max_nodes else 1000
            cold_dir = cfg.tiered_storage_path or "./rtmdk_cold_storage_v2"
            inner = TieredNodeStore(
                max_hot=hot_limit, max_warm=warm_limit, cold_dir=cold_dir, latent_dim=cfg.latent_dim
            )
            f._tiered_store = TieredNodeStoreAdapter(inner)
            f.nodes = f._tiered_store  # type: ignore[assignment]
        elif cfg.tiered_storage_enabled:
            from rtmdk.memory.tiered_storage import TieredNodeStore as LegacyTieredNodeStore

            hot_limit = max(1, int(cfg.max_nodes * cfg.tiered_hot_pct)) if cfg.max_nodes else 100
            warm_limit = max(1, int(cfg.max_nodes * cfg.tiered_warm_pct)) if cfg.max_nodes else 1000
            cold_dir = cfg.tiered_storage_path or "./rtmdk_cold_storage"
            f._tiered_store = LegacyTieredNodeStore(hot_limit, warm_limit, cold_dir, cfg.latent_dim)
            f.nodes = f._tiered_store  # type: ignore[assignment]

    def _init_index_manager(self) -> None:
        f = self.field
        cfg = self.cfg
        f._index_mgr = IndexManager(cfg, cfg.latent_dim, f._rng, f._quant)
        f.bm25_index = f._index_mgr.bm25_index
        f.hnsw_index = f._index_mgr.hnsw_index
        f.shard_centers = f._index_mgr.shard_centers
        f._async_index_builder = f._index_mgr._async_builder

    def _init_sparse_routing(self) -> None:
        f = self.field
        cfg = self.cfg
        f.shard_router = None
        f._node_shard_map = {}
        if cfg.sparse_routing:
            f.shard_router = np.zeros(cfg.num_shards, dtype=np.float32)

    def _init_crystallization_counters(self) -> None:
        f = self.field
        f._crystallization_counter = 0
        f._crystallized_nodes = set()

    def _init_async_pipeline(self) -> None:
        f = self.field
        cfg = self.cfg
        f.query_q = None
        f.save_q = None
        f.evolve_q = None
        f._workers_started = False
        if cfg.async_pipeline:
            f.query_q = asyncio.Queue(maxsize=cfg.query_queue_size)
            f.save_q = asyncio.Queue(maxsize=cfg.save_queue_size)
            f.evolve_q = asyncio.Queue(maxsize=cfg.evolve_queue_size)
