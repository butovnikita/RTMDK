"""Tests for rtmdk.production.backup_restore."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.backup_restore import BackupManager


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestBackupManager:
    def test_create_and_restore_backup(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        original_nids = set(mem.field.nodes.keys())

        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"), compression=False)
        path = bm.create_backup(name="test1")
        assert path.endswith("test1.json")

        # Add another node
        mem.save_context({"input": "extra", "session_id": "s1"}, {"output": ""})
        assert len(mem.field.nodes) == 2

        # Restore
        result = bm.restore(path)
        assert result["success"] is True
        assert result["nodes_restored"] == 1
        restored_nids = set(mem.field.nodes.keys())
        assert restored_nids == original_nids

    def test_create_backup_compressed(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"), compression=True)
        path = bm.create_backup(name="compressed")
        assert path.endswith(".json.gz")

    def test_list_backups(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"), compression=False)
        bm.create_backup(name="b1")
        bm.create_backup(name="b2")
        backups = bm.list_backups()
        assert len(backups) == 2
        names = {b["name"] for b in backups}
        assert names == {"b1", "b2"}

    def test_delete_backup(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"), compression=False)
        path = bm.create_backup(name="del_me")
        assert bm.delete_backup(path) is True
        assert bm.delete_backup(path) is False

    def test_restore_missing_file(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"))
        result = bm.restore(str(tmp_path / "nonexistent.json"))
        assert result["success"] is False

    def test_auto_rotate(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        bm = BackupManager(mem, backup_dir=str(tmp_path / "backups"), compression=False)
        for i in range(5):
            bm.create_backup(name=f"b{i}")
        # Rotate to max 3
        bm.create_backup(name="latest", auto_rotate=True, max_backups=3)
        backups = bm.list_backups()
        assert len(backups) == 3
