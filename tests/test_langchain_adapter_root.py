"""Tests for rtmdk.langchain_adapter (root)."""

import numpy as np
from langchain_core.messages import HumanMessage, AIMessage

from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.langchain_adapter import RTMDKMemory as LangChainMemory, as_langchain


def _embed(text: str) -> np.ndarray:
    return np.random.randn(768).astype(np.float32)


def _make_core():
    cfg = RTMDKConfig(latent_dim=64)
    return RTMDKMemory(config=cfg, embedder=_embed)


class TestLangChainMemoryRoot:
    def test_messages_property(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        assert mem.messages == []

    def test_add_message(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        msg = HumanMessage(content="hello")
        mem.add_message(msg)
        assert len(mem.messages) == 1

    def test_add_user_message(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        mem.add_user_message("hi")
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], HumanMessage)

    def test_add_ai_message(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        mem.add_ai_message("hello back")
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], AIMessage)

    def test_clear(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        mem.add_user_message("x")
        mem.clear()
        assert mem.messages == []

    def test_get_context(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        ctx = mem.get_context("test")
        assert isinstance(ctx, str)

    def test_get_stats(self):
        core = _make_core()
        mem = LangChainMemory(core_memory=core)
        stats = mem.get_stats()
        assert isinstance(stats, dict)

    def test_as_langchain(self):
        core = _make_core()
        mem = as_langchain(core, session_id="s1")
        assert isinstance(mem, LangChainMemory)
        assert mem._session_id == "s1"
