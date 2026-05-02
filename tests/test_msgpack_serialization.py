"""Test msgpack serialization and dirty flag behavior."""
import os
import tempfile

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKField, RTMDKConfig, RTMDKMemory


def test_msgpack_roundtrip():
    """Export/import with msgpack should preserve all nodes."""
    config = RTMDKConfig(latent_dim=64, embedding_dim=64, use_hnsw=False)
    field = RTMDKField(config)
    rng = np.random.default_rng(42)
    for i in range(50):
        emb = rng.standard_normal(64).astype(np.float32)
        content = {"text": f"node_{i}", "tier": "semantic"}
        field.add_node(emb, content, phase=rng.random(), session_id="default")

    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
        path = f.name
    try:
        field.export_field(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Dirty flag should be cleared after export
        assert not field._dirty

        embedder = lambda text: np.random.default_rng(7).standard_normal(64).astype(np.float32)
        memory = RTMDKMemory.import_field(path, embedder)
        assert len(memory.field.nodes) == 50
    finally:
        os.unlink(path)


def test_dirty_flag_set_on_add():
    """Adding a node should set dirty flag."""
    config = RTMDKConfig(latent_dim=64, embedding_dim=64)
    field = RTMDKField(config)
    assert not field._dirty

    emb = np.random.default_rng(1).standard_normal(64).astype(np.float32)
    field.add_node(emb, {"text": "hello"}, session_id="default")
    assert field._dirty


def test_dirty_flag_cleared_on_export():
    """Export should clear dirty flag."""
    config = RTMDKConfig(latent_dim=64, embedding_dim=64)
    field = RTMDKField(config)
    emb = np.random.default_rng(1).standard_normal(64).astype(np.float32)
    field.add_node(emb, {"text": "hello"}, session_id="default")
    assert field._dirty

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        field.export_field(path, fmt="json")
        assert not field._dirty
    finally:
        os.unlink(path)
