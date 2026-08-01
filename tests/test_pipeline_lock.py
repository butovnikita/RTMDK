"""Tests for distributed lock pipeline stages."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.lock_stages import DistributedLockStage, DistributedLockReleaseStage
from rtmdk.pipeline.base import PipelineContext


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class MockLock:
    def __init__(self, acquire_success: bool = True):
        self.acquire_success = acquire_success
        self.acquired = False
        self.released = False

    def acquire(self, blocking: bool = True) -> bool:
        if self.acquire_success:
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        self.released = True


class TestDistributedLockStages:
    def test_lock_acquire_and_release(self):
        lock = MockLock()
        acquire_stage = DistributedLockStage(lock)
        release_stage = DistributedLockReleaseStage(lock)

        ctx = PipelineContext(query_text="q")
        ctx = acquire_stage.process(ctx)
        assert lock.acquired is True
        assert lock.released is False

        ctx = release_stage.process(ctx)
        assert lock.released is True

    def test_lock_acquire_failure(self):
        lock = MockLock(acquire_success=False)
        acquire_stage = DistributedLockStage(lock)

        ctx = PipelineContext(query_text="q")
        ctx = acquire_stage.process(ctx)
        assert lock.acquired is False

    def test_lock_release_idempotent(self):
        lock = MockLock()
        release_stage = DistributedLockReleaseStage(lock)

        ctx = PipelineContext(query_text="q")
        # Release without acquire should not crash
        ctx = release_stage.process(ctx)
        assert lock.released is True

    def test_pipeline_includes_lock_stages(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))

        # No lock by default
        pipeline = mem.build_pipeline()
        stage_names = [s.name for s in pipeline.stages]
        assert "distributed_lock_acquire" not in stage_names
        assert "distributed_lock_release" not in stage_names

    def test_pipeline_with_mock_lock(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(5):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        lock = MockLock()
        mem._distributed_lock = lock
        pipeline = mem.build_pipeline()
        stage_names = [s.name for s in pipeline.stages]
        assert "distributed_lock_acquire" in stage_names
        assert "distributed_lock_release" in stage_names

        ctx = pipeline.run("doc 2", top_k=3)
        assert lock.acquired is True
        assert lock.released is True
        assert len(ctx.results) > 0
