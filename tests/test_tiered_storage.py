"""Unit tests for TieredNodeStore (hot/warm/cold tiered storage)."""

import tempfile

import numpy as np

from rtmdk.memory.tiered_storage import TieredNodeStore
from rtmdk.nodes import MemoryNode


def _make_node(node_id: str, dim: int = 64) -> MemoryNode:
    emb = np.random.randn(dim).astype(np.float32)
    return MemoryNode(
        id=node_id,
        content={"text": f"node {node_id}"},
        latent_pos=emb,
        phase=0.0,
        amplitude=1.0,
        salience=0.5,
    )


class TestTieredNodeStore:
    def test_basic_set_get(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=5, warm_limit=5, cold_dir=td, latent_dim=64)
            node = _make_node("n1")
            store["n1"] = node
            assert len(store) == 1
            assert "n1" in store
            retrieved = store["n1"]
            assert retrieved.id == "n1"

    def test_hot_to_warm_rebalance(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=2, warm_limit=5, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            assert len(store.hot_keys()) == 2
            assert len(store.warm_ids()) == 2
            assert store._tier["n0"] == "warm"
            assert store._tier["n3"] == "hot"

    def test_warm_to_cold_rebalance(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=1, warm_limit=1, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            assert len(store.hot_keys()) == 1
            assert len(store.warm_ids()) <= 1
            # At least some nodes should be in cold
            assert len(store.cold_ids()) >= 2

    def test_cold_promotion_on_get(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=1, warm_limit=1, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            cold_ids = store.cold_ids()
            assert cold_ids
            first_cold = cold_ids[0]
            assert store._tier[first_cold] == "cold"
            # Access promotes to hot
            node = store[first_cold]
            assert node.id == first_cold
            assert store._tier[first_cold] == "hot"

    def test_delete_removes_from_all_tiers(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=1, warm_limit=1, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            # n0 should be in cold after rebalance
            del store["n0"]
            assert "n0" not in store
            assert len(store) == 3

    def test_get_batch_multi_tier(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=1, warm_limit=1, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            nodes = store.get_batch(["n0", "n1", "n2", "n3"])
            assert len(nodes) == 4
            ids = {n.id for n in nodes}
            assert ids == {"n0", "n1", "n2", "n3"}

    def test_save_load_state(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=2, warm_limit=2, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            state = store.save_state()
            assert state["hot_limit"] == 2
            assert state["warm_limit"] == 2
            assert len(state["tier"]) == 4

            store2 = TieredNodeStore(hot_limit=2, warm_limit=2, cold_dir=td, latent_dim=64)
            store2.load_state(state)
            assert store2._tier == store._tier
            assert store2._access_count == store._access_count

    def test_clear_cold_storage(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=1, warm_limit=1, cold_dir=td, latent_dim=64)
            for i in range(4):
                store[f"n{i}"] = _make_node(f"n{i}")
            assert store.cold_ids()
            store.clear_cold_storage()
            assert not store.cold_ids()
            assert not store._cold_batches

    def test_items_iteration(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=5, warm_limit=5, cold_dir=td, latent_dim=64)
            for i in range(3):
                store[f"n{i}"] = _make_node(f"n{i}")
            items = list(store.items())
            assert len(items) == 3
            ids = {nid for nid, _ in items}
            assert ids == {"n0", "n1", "n2"}

    def test_pop_returns_node(self):
        with tempfile.TemporaryDirectory() as td:
            store = TieredNodeStore(hot_limit=5, warm_limit=5, cold_dir=td, latent_dim=64)
            store["n1"] = _make_node("n1")
            node = store.pop("n1")
            assert node.id == "n1"
            assert "n1" not in store
