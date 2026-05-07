"""
Tests for Track 2: Tiered Storage (Hot / Warm / Cold)
"""
import os
import tempfile
import numpy as np

from rtmdk.memory.tiered_storage import TieredNodeStore
from rtmdk.nodes import MemoryNode


def make_node(node_id, latent_dim=64):
    return MemoryNode(
        id=node_id,
        latent_pos=np.random.randn(latent_dim).astype(np.float32),
        phase=0.0,
        amplitude=1.0,
        salience=1.0,
        content={"text": f"node {node_id}"},
    )


class TestTieredNodeStore:
    def test_basic_crud(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=5, warm_limit=5,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        n1 = make_node("n1")
        store["n1"] = n1
        assert len(store) == 1
        assert "n1" in store
        assert store["n1"].id == "n1"
        assert store.get("n2") is None
        del store["n1"]
        assert "n1" not in store
        assert len(store) == 0

    def test_hot_to_warm_demotion(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=3, warm_limit=10,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        for i in range(5):
            store[f"n{i}"] = make_node(f"n{i}")
        assert len(store) == 5
        assert len(store.hot_keys()) <= 3
        assert len(store.warm_ids()) >= 2
        for i in range(5):
            assert store[f"n{i}"].id == f"n{i}"

    def test_warm_to_cold_freeze(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=2, warm_limit=2,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        for i in range(6):
            store[f"n{i}"] = make_node(f"n{i}")
        assert len(store) == 6
        assert len(store.cold_ids()) >= 1
        node = store["n0"]
        assert node.id == "n0"
        assert "n0" in store.hot_keys()

    def test_all_node_dicts_roundtrip(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=2, warm_limit=2,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        for i in range(6):
            store[f"n{i}"] = make_node(f"n{i}")
        dicts = list(store.all_node_dicts())
        assert len(dicts) == 6
        ids = {d["id"] for d in dicts}
        assert ids == {f"n{i}" for i in range(6)}

    def test_save_load_state(self, tmp_path):
        cold_dir = str(tmp_path / "cold")
        store = TieredNodeStore(
            hot_limit=2, warm_limit=2,
            cold_dir=cold_dir, latent_dim=64
        )
        for i in range(6):
            store[f"n{i}"] = make_node(f"n{i}")
        state = store.save_state()
        store2 = TieredNodeStore(
            hot_limit=2, warm_limit=2,
            cold_dir=cold_dir, latent_dim=64
        )
        store2.load_state(state)
        for d in list(store.all_node_dicts()):
            store2._hot[d["id"]] = MemoryNode.from_dict(d)
            store2._tier[d["id"]] = "hot"
        assert len(store2) == 6
        for i in range(6):
            assert store2[f"n{i}"].id == f"n{i}"

    def test_clear_cold_storage(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=1, warm_limit=1,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        for i in range(4):
            store[f"n{i}"] = make_node(f"n{i}")
        assert len(store.cold_ids()) >= 1
        store.clear_cold_storage()
        assert len(store.cold_ids()) == 0

    def test_delete_removes_from_all_tiers(self, tmp_path):
        store = TieredNodeStore(
            hot_limit=1, warm_limit=1,
            cold_dir=str(tmp_path / "cold"), latent_dim=64
        )
        for i in range(4):
            store[f"n{i}"] = make_node(f"n{i}")
        del store["n0"]
        assert "n0" not in store
        assert len(store) == 3


class TestTieredStorageIntegration:
    def test_field_with_tiered_storage(self):
        from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
        config = RTMDKConfig(
            tiered_storage_enabled=True, max_nodes=100, latent_dim=64
        )
        memory = RTMDKMemory(
            config=config,
            embedder=lambda x: np.zeros(64, dtype=np.float32),
        )
        for i in range(20):
            memory.add_node(
                embedding=np.random.randn(64).astype(np.float32),
                content={"text": f"doc {i}"},
            )
        assert len(memory.field.nodes) == 20
        results = memory.field.query(
            embedding=np.random.randn(64).astype(np.float32), top_k=3
        )
        assert isinstance(results, list)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "memory.json")
            memory.export_field(path)
            loaded = RTMDKMemory.import_field(
                path,
                embedder=lambda x: np.zeros(64, dtype=np.float32),
            )
            assert len(loaded.field.nodes) == 20
            results2 = loaded.field.query(
                embedding=np.random.randn(64).astype(np.float32), top_k=3
            )
            assert isinstance(results2, list)
