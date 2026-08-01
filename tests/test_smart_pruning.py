"""Tests for rtmdk.production.smart_pruning."""

import time

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.smart_pruning import SmartPruner


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestSmartPruner:
    def test_prune_empty(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        pruner = SmartPruner(mem, max_age_days=1, min_salience=1.0)
        result = pruner.prune()
        assert result["nodes_pruned"] == 0
        assert result["nodes_before"] == 0

    def test_prune_old_low_salience(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "old", "session_id": "s1"}, {"output": ""})
        nid = list(mem.field.nodes.keys())[0]
        node = mem.field.nodes[nid]
        node.tier = "episodic"
        node.created_at = time.time() - 86400 * 2  # 2 days old
        node.salience = 0.01
        node.amplitude = 0.01

        pruner = SmartPruner(
            mem,
            max_age_days=1,
            min_salience=0.05,
            tier_overrides={"episodic": {"max_age_days": 1, "min_salience": 0.05}},
        )
        result = pruner.prune()
        assert result["nodes_pruned"] == 1
        assert nid not in mem.field.nodes

    def test_prune_dry_run(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "old", "session_id": "s1"}, {"output": ""})
        nid = list(mem.field.nodes.keys())[0]
        node = mem.field.nodes[nid]
        node.tier = "episodic"
        node.created_at = time.time() - 86400 * 2
        node.salience = 0.01
        node.amplitude = 0.01

        pruner = SmartPruner(
            mem,
            max_age_days=1,
            min_salience=0.05,
            dry_run=True,
            tier_overrides={"episodic": {"max_age_days": 1, "min_salience": 0.05}},
        )
        result = pruner.prune()
        assert result["nodes_pruned"] == 1
        assert result["dry_run"] is True
        assert nid in mem.field.nodes  # not actually removed

    def test_prune_respects_tier_overrides(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "old semantic", "session_id": "s1"}, {"output": ""})
        nid = list(mem.field.nodes.keys())[0]
        node = mem.field.nodes[nid]
        node.tier = "semantic"
        node.created_at = time.time() - 86400 * 10
        node.salience = 0.01
        node.amplitude = 0.01

        pruner = SmartPruner(
            mem,
            max_age_days=1,
            min_salience=0.05,
            tier_overrides={"semantic": {"max_age_days": 5, "min_salience": 0.05}},
        )
        # 10 days > 5 days override, salience 0.01 < 0.05 → should prune
        result = pruner.prune()
        assert result["nodes_pruned"] == 1

    def test_prune_export_backup(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "old", "session_id": "s1"}, {"output": ""})
        nid = list(mem.field.nodes.keys())[0]
        node = mem.field.nodes[nid]
        node.tier = "episodic"
        node.created_at = time.time() - 86400 * 2
        node.salience = 0.01
        node.amplitude = 0.01

        export_dir = str(tmp_path / "pruning_backups")
        pruner = SmartPruner(
            mem,
            max_age_days=1,
            min_salience=0.05,
            export_dir=export_dir,
            tier_overrides={"episodic": {"max_age_days": 1, "min_salience": 0.05}},
        )
        result = pruner.prune()
        assert result["exported_to"] is not None
        import os

        assert os.path.exists(result["exported_to"])

    def test_get_stats(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        pruner = SmartPruner(mem)
        stats = pruner.get_stats()
        assert stats["total_prunes"] == 0
        assert "config" in stats
