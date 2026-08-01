"""Tests for rtmdk.production.session_persistence."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.session_persistence import SessionPersistence


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestSessionPersistence:
    def test_save_and_load_session(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})

        sp = SessionPersistence(mem, save_dir=str(tmp_path / "sessions"))
        path = sp.save_session("s1")
        assert path.endswith("s1.json")

        # Clear memory
        mem.field.nodes.clear()
        mem.field.node_index.clear()
        assert len(mem.field.nodes) == 0

        meta = sp.load_session("s1")
        assert meta is not None
        assert meta["session_id"] == "s1"
        assert len(mem.field.nodes) == 1

    def test_list_sessions(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        sp = SessionPersistence(mem, save_dir=str(tmp_path / "sessions"))
        sp.save_session("s1")
        sp.save_session("s2")
        sessions = sp.list_sessions()
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"s1", "s2"}

    def test_delete_session(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        sp = SessionPersistence(mem, save_dir=str(tmp_path / "sessions"))
        sp.save_session("s1")
        assert len(sp.list_sessions()) == 1
        assert sp.delete_session("s1") is True
        assert len(sp.list_sessions()) == 0
        assert sp.delete_session("s1") is False

    def test_auto_save(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        sp = SessionPersistence(mem, save_dir=str(tmp_path / "sessions"), auto_save_interval=0)
        sp.save_session("s1")
        # Manually trigger auto-save with very short interval
        sp.auto_save_interval = 1
        sp._auto_save_timer = 0
        sp.start_auto_save()
        # Should have saved again
        sessions = sp.list_sessions()
        assert any(s["session_id"] == "s1" for s in sessions)
