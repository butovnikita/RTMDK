"""Tests for rtmdk.production.export."""

import json

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.export import MemoryExporter


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestMemoryExporter:
    def test_to_dict(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        exporter = MemoryExporter(mem)
        data = exporter.to_dict()
        assert data["node_count"] == 1
        assert "nodes" in data
        assert "stats" in data

    def test_to_markdown(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        # Set text explicitly for exporter
        for node in mem.field.nodes.values():
            node.content["text"] = node.content.get("input_text", "")
        exporter = MemoryExporter(mem)
        md = exporter.to_markdown()
        assert "# RTMDK Memory Export" in md
        assert "hello" in md

    def test_to_text(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        for node in mem.field.nodes.values():
            node.content["text"] = node.content.get("input_text", "")
        exporter = MemoryExporter(mem)
        txt = exporter.to_text()
        assert "RTMDK Memory Export" in txt
        assert "hello" in txt

    def test_export_to_file_markdown(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        exporter = MemoryExporter(mem)
        path = str(tmp_path / "out.md")
        result = exporter.export_to_file(path, format="markdown")
        assert result == path
        with open(path, "r", encoding="utf-8") as f:
            assert "# RTMDK Memory Export" in f.read()

    def test_export_to_file_json(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        exporter = MemoryExporter(mem)
        path = str(tmp_path / "out.json")
        exporter.export_to_file(path, format="json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "node_count" in data

    def test_export_to_file_unknown_format(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        exporter = MemoryExporter(mem)
        with pytest.raises(ValueError):
            exporter.export_to_file(str(tmp_path / "out.xyz"), format="xyz")


import pytest
