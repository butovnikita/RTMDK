"""Tests for production modules: VectorStorage, ReplicationManager, GPUBackend."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestInMemoryVectorStorage:
    def test_crud(self):
        from rtmdk.production.vector_storage import InMemoryVectorStorage

        vs = InMemoryVectorStorage(dim=4)
        assert vs.available is True
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert vs.insert("n1", vec, {"tag": "a"}) is True
        assert vs.count() == 1
        assert np.allclose(vs.get("n1"), vec)

        results = vs.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "n1"
        assert results[0][1] > 0.99

        assert vs.delete("n1") is True
        assert vs.count() == 0
        assert vs.get("n1") is None

    def test_search_cosine(self):
        from rtmdk.production.vector_storage import InMemoryVectorStorage

        vs = InMemoryVectorStorage(dim=3)
        vs.insert("a", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        vs.insert("b", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = vs.search(q, top_k=2)
        assert results[0][0] == "a"
        assert results[1][0] == "b"

    def test_dim_mismatch(self):
        from rtmdk.production.vector_storage import InMemoryVectorStorage

        vs = InMemoryVectorStorage(dim=2)
        with pytest.raises(ValueError):
            vs.insert("x", np.array([1.0, 2.0, 3.0]))


class TestSQLiteVectorStorage:
    def test_crud(self):
        from rtmdk.production.vector_storage import SQLiteVectorStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            vs = SQLiteVectorStorage(dsn=f"sqlite:///{path}", dim=4)
            vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            assert vs.insert("n1", vec, {"tag": "a"}) is True
            assert vs.count() == 1
            assert np.allclose(vs.get("n1"), vec)

            results = vs.search(vec, top_k=1)
            assert len(results) == 1
            assert results[0][0] == "n1"

            assert vs.delete("n1") is True
            assert vs.count() == 0
            vs.close()
        finally:
            os.unlink(path)

    def test_persistence(self):
        from rtmdk.production.vector_storage import SQLiteVectorStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            vs = SQLiteVectorStorage(dsn=f"sqlite:///{path}", dim=2)
            vs.insert("x", np.array([1.0, 2.0], dtype=np.float32))
            vs.close()

            vs2 = SQLiteVectorStorage(dsn=f"sqlite:///{path}", dim=2)
            assert vs2.count() == 1
            assert np.allclose(vs2.get("x"), np.array([1.0, 2.0]))
            vs2.close()
        finally:
            os.unlink(path)


class TestVectorStorageFactory:
    def test_create_memory(self):
        from rtmdk.production.vector_storage import VectorStorage

        vs = VectorStorage.create(None, dim=4)
        assert isinstance(vs, VectorStorage)
        assert vs.count() == 0

    def test_create_sqlite(self):
        from rtmdk.production.vector_storage import SQLiteVectorStorage, VectorStorage

        vs = VectorStorage.create("sqlite:///:memory:", dim=4)
        assert isinstance(vs, SQLiteVectorStorage)


class TestReplicationManager:
    def test_disabled_without_peers(self):
        from rtmdk.production.replication import ReplicationManager

        mgr = ReplicationManager()
        assert mgr.enabled is False
        assert mgr.replicate({"x": 1}) is False
        assert mgr.sync_from_peers() == []
        assert mgr.local_clock() == 0

    def test_replicate_bumps_clock_and_wal(self):
        from rtmdk.production.replication import ReplicationManager

        mgr = ReplicationManager(peers=[], node_id="n1")
        assert mgr.enabled is False

        mgr2 = ReplicationManager(peers=["http://n2:8000"], node_id="n1")
        assert mgr2.enabled is True
        # httpx may not be installed in some envs; handle gracefully
        try:
            import httpx  # noqa: F401
        except ImportError:
            pytest.skip("httpx not installed")

        with patch.object(mgr2, "_broadcast"):
            ok = mgr2.replicate({"op": "add_node"})
            assert ok is True
            assert mgr2.local_clock() == 1
            assert mgr2._wal.count() == 1

    def test_get_wal(self):
        from rtmdk.production.replication import ReplicationManager

        try:
            import httpx  # noqa: F401
        except ImportError:
            pytest.skip("httpx not installed")

        mgr = ReplicationManager(peers=["http://n2:8000"], node_id="n1")
        with patch.object(mgr, "_broadcast"):
            mgr.replicate({"op": "add_node"})
            mgr.replicate({"op": "delete"})
        entries = mgr.get_wal(since=0)
        assert len(entries) == 2
        assert entries[0]["_rep_clock"] == 1
        assert entries[1]["_rep_clock"] == 2

    def test_sync_from_peers_mock(self):
        from rtmdk.production.replication import ReplicationManager

        try:
            import httpx  # noqa: F401
        except ImportError:
            pytest.skip("httpx not installed")

        mgr = ReplicationManager(peers=["http://n2:8000"], node_id="n1")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "mutations": [
                {"_rep_clock": 5, "_rep_origin": "n2", "op": "add_node"},
            ]
        }
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
            new = mgr.sync_from_peers()
            assert len(new) == 1
            assert new[0]["_rep_clock"] == 5
            assert mgr._wal.count() == 1


class TestGPUBackendProduction:
    def test_available_delegation(self):
        from rtmdk.production.gpu_backend import GPUBackend

        g = GPUBackend(min_nodes_for_gpu=10)
        # Should not crash regardless of CUDA availability
        assert isinstance(g.available, bool)

    def test_batch_distance_fallback(self):
        from rtmdk.production.gpu_backend import GPUBackend

        g = GPUBackend(min_nodes_for_gpu=10)
        q = np.array([0.0, 1.0], dtype=np.float32)
        pos = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        dists = g.batch_distance(q, pos)
        assert dists.shape == (2,)
        np.testing.assert_allclose(dists, [np.sqrt(2), 1.0], rtol=1e-5)

    def test_project_fallback(self):
        from rtmdk.production.gpu_backend import GPUBackend

        g = GPUBackend(min_nodes_for_gpu=999999)
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = g.project(vecs, mat)
        np.testing.assert_allclose(out, vecs @ mat.T, rtol=1e-5)
