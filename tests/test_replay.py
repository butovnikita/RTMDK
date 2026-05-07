"""Tests for rtmdk.production.replay."""

import numpy as np
import pytest
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.production.replay import ConversationReplay


def _embed(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(768, dtype=np.float32)


class TestConversationReplay:
    def test_record_query(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embed)
        replay = ConversationReplay(mem)
        replay.record_query("hello", "hi there")
        assert len(replay._history) == 1
        assert replay._history[0]["query"] == "hello"

    def test_replay_queries(self):
        cfg = RTMDKConfig(latent_dim=768)
        mem = RTMDKMemory(config=cfg, embedder=_embed)
        mem.save_context({"input": "coffee", "session_id": "s1"}, {"output": ""})
        replay = ConversationReplay(mem)
        results = replay.replay_queries(["coffee"], _embed)
        assert len(results) == 1
        assert results[0]["query"] == "coffee"

    def test_get_history(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embed)
        replay = ConversationReplay(mem)
        replay.record_query("q", "r")
        hist = replay.get_history()
        assert len(hist) == 1
