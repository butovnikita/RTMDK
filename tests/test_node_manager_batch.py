"""Unit tests for NodeManager batch operations, especially edge cases."""

import numpy as np

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestNodeManagerBatch:
    def test_add_nodes_batch_with_duplicate_node_ids(self):
        """Duplicate node_ids within a batch: last wins in nodes dict,
        node_index deduplicates via set-based lookup."""
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)

        embeddings = np.random.randn(3, 64).astype(np.float32)
        contents = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        node_ids = ["dup", "dup", "dup"]

        field.add_nodes_batch(embeddings, contents, node_ids=node_ids)

        # Last duplicate overwrites previous nodes
        assert len(field.nodes) == 1
        assert field.nodes["dup"].content["text"] == "c"
        # node_index deduplicates
        assert field.node_index.count("dup") == 1

    def test_add_nodes_batch_auto_ids_unique(self):
        """Auto-generated IDs should be unique across batches."""
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)

        for _ in range(3):
            embeddings = np.random.randn(5, 64).astype(np.float32)
            contents = [{"text": f"doc {i}"} for i in range(5)]
            field.add_nodes_batch(embeddings, contents)

        assert len(field.nodes) == 15
        assert len(field.node_index) == 15
        assert len(set(field.node_index)) == 15

    def test_add_nodes_batch_preserves_order(self):
        """node_index should preserve insertion order."""
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)

        embeddings = np.random.randn(3, 64).astype(np.float32)
        contents = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        node_ids = ["first", "second", "third"]

        field.add_nodes_batch(embeddings, contents, node_ids=node_ids)

        assert field.node_index[0] == "first"
        assert field.node_index[1] == "second"
        assert field.node_index[2] == "third"
