"""Tests for production stub modules."""


class TestReplicationManager:
    def test_default_disabled(self):
        from rtmdk.production.replication import ReplicationManager

        mgr = ReplicationManager()
        assert mgr.enabled is False
        assert mgr.replicate({"x": 1}) is False
        assert mgr.sync_from_peers() == []

    def test_with_peers_enabled(self):
        from rtmdk.production.replication import ReplicationManager

        mgr = ReplicationManager(peers=["http://n2:8000"], node_id="n1")
        assert mgr.enabled is True
        assert mgr.replicate({"x": 1}) is True


class TestVectorStorage:
    def test_default_unavailable(self):
        from rtmdk.production.vector_storage import VectorStorage

        vs = VectorStorage()
        assert vs.available is False
        assert vs.insert("a", __import__("numpy").array([1.0]), {}) is False
        assert vs.search(__import__("numpy").array([1.0])) == []
        assert vs.delete("a") is False

    def test_sqlite_dsn_unavailable_without_lib(self):
        from rtmdk.production.vector_storage import VectorStorage

        vs = VectorStorage(dsn="sqlite:///test.db")
        # sqlite_vss not installed in most envs
        assert vs.available is False
