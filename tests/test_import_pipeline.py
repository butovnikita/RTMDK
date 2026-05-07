"""Tests for rtmdk.production.import_pipeline."""

import json

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.import_pipeline import ImportPipeline


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestImportPipeline:
    def test_import_json(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        data = [{"text": f"doc {i}"} for i in range(5)]
        path = tmp_path / "data.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pipeline = ImportPipeline(mem)
        result = pipeline.import_json(str(path), text_field="text")
        assert result.imported_items == 5
        assert result.failed_items == 0
        assert len(mem.field.nodes) == 5

    def test_import_json_nested_records(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        data = {"records": [{"text": "nested"}]}
        path = tmp_path / "nested.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        pipeline = ImportPipeline(mem)
        result = pipeline.import_json(str(path), text_field="text")
        assert result.imported_items == 1

    def test_import_csv(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        path = tmp_path / "data.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("text,id\n")
            for i in range(3):
                f.write(f"row {i},{i}\n")

        pipeline = ImportPipeline(mem)
        result = pipeline.import_csv(str(path), text_column="text")
        assert result.imported_items == 3

    def test_import_text(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        path = tmp_path / "doc.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("Hello world this is a test document.")

        pipeline = ImportPipeline(mem)
        result = pipeline.import_text(str(path), chunk_size=3, overlap=1)
        assert result.imported_items > 0
        assert result.total_items > 0

    def test_import_empty_json(self, tmp_path):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        path = tmp_path / "empty.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)

        pipeline = ImportPipeline(mem)
        result = pipeline.import_json(str(path))
        assert result.imported_items == 0

    def test_chunk_text(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        pipeline = ImportPipeline(mem)
        chunks = pipeline._chunk_text("a b c d e f g h", chunk_size=3, overlap=1)
        assert len(chunks) > 0
        # Verify overlap
        assert chunks[0].split()[-1] == chunks[1].split()[0]
