"""Tests for rtmdk.production.memory_diff."""

import numpy as np
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.production.memory_diff import MemoryDiff


def _embed(text: str) -> np.ndarray:
    return np.random.randn(768).astype(np.float32)


def _make_mem():
    cfg = RTMDKConfig(latent_dim=64)
    return RTMDKMemory(config=cfg, embedder=_embed)


class TestMemoryDiff:
    def test_no_changes(self):
        mem = _make_mem()
        diff = MemoryDiff(mem, mem)
        result = diff.compute()
        assert result["added"] == []
        assert result["removed"] == []
        assert result["modified"] == []
        assert result["summary"]["nodes_before"] == result["summary"]["nodes_after"]

    def test_added_nodes(self):
        mem1 = _make_mem()
        mem2 = _make_mem()
        emb = np.random.randn(64).astype(np.float32)
        nid = mem2.field.add_node(emb, {"text": "new"})
        diff = MemoryDiff(mem1, mem2)
        result = diff.compute()
        assert result["added"] == [nid]
        assert result["summary"]["added_count"] == 1

    def test_removed_nodes(self):
        mem1 = _make_mem()
        mem2 = _make_mem()
        emb = np.random.randn(64).astype(np.float32)
        nid = mem1.field.add_node(emb, {"text": "old"})
        diff = MemoryDiff(mem1, mem2)
        result = diff.compute()
        assert result["removed"] == [nid]
        assert result["summary"]["removed_count"] == 1

    def test_modified_nodes(self):
        mem1 = _make_mem()
        mem2 = _make_mem()
        emb = np.random.randn(64).astype(np.float32)
        nid = mem1.field.add_node(emb, {"text": "same"}, node_id="test_node")
        mem2.field.add_node(emb, {"text": "same"}, node_id="test_node")
        # modify salience on mem2
        mem2.field.nodes[nid].salience = 0.99
        diff = MemoryDiff(mem1, mem2)
        result = diff.compute()
        assert len(result["modified"]) == 1
        assert "salience" in result["modified"][0]["changes"]
