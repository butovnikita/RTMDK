"""Unit tests for ContextManager."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.context_manager import ContextManager


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestContextManager:
    def test_save_context_creates_nodes(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        ctx = ContextManager(mem)

        ctx.save_context(
            inputs={"input": "hello world", "output": "hi there"},
            outputs={"output": "hi there"},
        )
        assert len(mem.field.nodes) > 0

    def test_retrieve_and_format_returns_string(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        ctx = ContextManager(mem)

        ctx.save_context(
            inputs={"input": "python programming", "output": "a high-level language"},
            outputs={"output": "a high-level language"},
        )

        emb = _make_embedder(64)("python")
        result = ctx.retrieve_and_format("python", emb, session_id="test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_save_context_detects_modality(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        ctx = ContextManager(mem)

        ctx.save_context(
            inputs={"input": "code: def foo(): pass", "output": "a function definition"},
            outputs={"output": "a function definition"},
        )
        # At least one node should have code-related content
        texts = [n.content.get("input_text", "") for n in mem.field.nodes.values()]
        assert any("def foo" in t for t in texts)

    def test_retrieve_and_format_with_empty_field(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        ctx = ContextManager(mem)

        emb = _make_embedder(64)("anything")
        result = ctx.retrieve_and_format("query", emb, session_id="empty")
        assert isinstance(result, str)
