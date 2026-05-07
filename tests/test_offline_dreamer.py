"""Tests for rtmdk.production.offline_dreamer."""

import time

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.offline_dreamer import OfflineDreamer, DreamTask


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestOfflineDreamer:
    def test_start_stop(self):
        dreamer = OfflineDreamer()
        dreamer.start()
        assert dreamer._running is True
        dreamer.stop()
        assert dreamer._running is False

    def test_on_step_schedules_tasks(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        dreamer = OfflineDreamer(field=mem.field, dream_freq=1)
        for _ in range(101):
            dreamer.on_step()
        assert dreamer._task_queue.qsize() > 0

    def test_get_stats(self):
        dreamer = OfflineDreamer()
        stats = dreamer.get_stats()
        assert stats["cycles_completed"] == 0
        assert "enabled_tasks" in stats

    def test_task_execution(self):
        called = []
        def dummy():
            called.append(1)
        dreamer = OfflineDreamer()
        dreamer._task_queue.put((-1, DreamTask("test", dummy, priority=1)))
        dreamer.start()
        time.sleep(0.2)
        dreamer.stop()
        assert len(called) >= 1

    def test_cooldown_prevents_spam(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        dreamer = OfflineDreamer(field=mem.field, dream_freq=1)
        dreamer.start()
        for i in range(250):
            dreamer.on_step()
        time.sleep(0.3)
        dreamer.stop()
        # After processing, queue should be drained significantly
        assert dreamer._task_queue.qsize() <= 5

    def test_disabled_tasks_not_queued(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        dreamer = OfflineDreamer(
            field=mem.field,
            dream_freq=5,
            enable_tda=False,
            enable_crystallization=False,
            enable_shard_recalc=False,
            enable_engram_merge=False,
            enable_topology_repair=False,
        )
        for _ in range(5):
            dreamer.on_step()
        assert dreamer._task_queue.qsize() == 0
