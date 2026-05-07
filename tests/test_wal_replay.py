"""Tests for WAL replay / durability recovery (Track 5)."""

import json
import numpy as np
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    """Deterministic embedder for tests."""
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestWALReplay:
    def test_replay_add_node(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem1 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        mem1.save_context(
            {"input": "hello world", "session_id": "s1"}, {"output": ""})
        n1 = len(mem1.field.nodes)
        assert n1 == 1

        # Simulate restart: new memory instance with same WAL
        mem2 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        assert len(mem2.field.nodes) == n1
        nid = list(mem2.field.nodes.keys())[0]
        node = mem2.field.nodes[nid]
        assert node.content.get("input_text") == "hello world"

    def test_replay_add_nodes_batch(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem1 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        n = 5
        embeddings = np.random.randn(n, 64).astype(np.float32)
        contents = [{"text": f"batch {i}"} for i in range(n)]
        mem1.add_nodes_batch(embeddings, contents)
        assert len(mem1.field.nodes) == n

        mem2 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        assert len(mem2.field.nodes) == n

    def test_replay_delete(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem1 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        mem1.save_context(
            {"input": "keep me", "session_id": "s1"}, {"output": ""})
        mem1.save_context(
            {"input": "delete me", "session_id": "s1"}, {"output": ""})
        nids = list(mem1.field.nodes.keys())
        assert len(nids) == 2
        mem1.field.delete_nodes([nids[1]])
        assert len(mem1.field.nodes) == 1

        mem2 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        assert len(mem2.field.nodes) == 1
        assert nids[0] in mem2.field.nodes
        assert nids[1] not in mem2.field.nodes

    def test_replay_no_wal(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=None)
        assert len(mem.field.nodes) == 0

    def test_replay_corrupted_line(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem1 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        mem1.save_context(
            {"input": "valid", "session_id": "s1"}, {"output": ""})
        # Append corrupted line
        with open(wal_path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")

        mem2 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        assert len(mem2.field.nodes) == 1

    def test_replay_old_format_without_embedding(self, tmp_path):
        """WAL records without embedding should fallback to embedder."""
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        # Write old-format WAL manually
        with open(wal_path, "w", encoding="utf-8") as f:
            rec = {
                "op": "add_node",
                "ts": 0.0,
                "payload": {
                    "node_id": "old_1",
                    "content": {"text": "fallback text"},
                    "modality": "text",
                    # "embedding" intentionally omitted
                },
            }
            f.write(json.dumps(rec) + "\n")

        mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        assert "old_1" in mem.field.nodes
        assert mem.field.nodes["old_1"].content["text"] == "fallback text"

    def test_replay_with_snapshot(self, tmp_path):
        """WAL replay after snapshot load should only add post-snapshot nodes."""
        wal_path = str(tmp_path / "wal.jsonl")
        snapshot_path = str(tmp_path / "snapshot.json")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=False,
            hyperbolic=False,
            quantization="none",
            enable_engrams=False)
        embedder = _make_embedder(64)
        mem1 = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        mem1.save_context({"input": "before snapshot",
                          "session_id": "s1"}, {"output": ""})
        mem1.export_field(snapshot_path)

        # Add more nodes after snapshot
        mem1.save_context({"input": "after snapshot",
                          "session_id": "s1"}, {"output": ""})

        # Load snapshot + WAL replay
        mem2 = RTMDKMemory.import_field(
            snapshot_path, embedder, wal_path=wal_path)
        assert len(mem2.field.nodes) == 2
        texts = [n.content.get("input_text", "")
                 for n in mem2.field.nodes.values()]
        assert "before snapshot" in texts
        assert "after snapshot" in texts
