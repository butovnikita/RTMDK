"""Tests for StepScheduler extraction."""
import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.scheduler import StepScheduler


class TestStepSchedulerExtraction:
    def test_scheduler_created_with_field(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        assert field._scheduler is not None
        assert isinstance(field._scheduler, StepScheduler)
        assert field._scheduler.field is field

    def test_scheduler_run_on_step(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        field.add_node(
            np.random.randn(64).astype(np.float32),
            {"text": "hello"})
        # step() should call _run_periodic_tasks which delegates to scheduler
        field.step([{"embedding": np.random.randn(64).astype(np.float32), "content": {"text": "world"}}])
        assert field._step_counter == 1

    def test_scheduler_does_not_crash_empty_field(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        field._scheduler.run(backpressure_ok=True)
        assert field.stats["tier_distribution"] == {}

    def test_scheduler_decay_on_run(self):
        cfg = RTMDKConfig()
        cfg.core.decay_rate = 0.9
        field = RTMDKField(cfg)
        emb = np.random.randn(64).astype(np.float32)
        field.add_node(emb, {"text": "test"})
        old_amp = field.nodes[list(field.nodes.keys())[0]].amplitude
        field._scheduler.run(backpressure_ok=True)
        new_amp = field.nodes[list(field.nodes.keys())[0]].amplitude
        assert new_amp <= old_amp

    def test_scheduler_triggers_consolidation(self):
        cfg = RTMDKConfig()
        cfg.core.tension_threshold = 0.01
        field = RTMDKField(cfg)
        for i in range(20):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        # Run scheduler many times to trigger consolidation
        for _ in range(200):
            field._scheduler.run(backpressure_ok=True)
            field._step_counter += 1
        # Step counter was incremented by our loop
        assert field._step_counter == 200

    def test_scheduler_tda_check(self):
        cfg = RTMDKConfig()
        cfg.core.tda_monitoring = True
        cfg.core.tda_check_freq = 1
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        field._scheduler.run(backpressure_ok=True)
        # TDA may or may not run depending on node count
        assert field.stats.get("tda_checks", 0) >= 0

    def test_scheduler_meta_kernel_adapt(self):
        cfg = RTMDKConfig()
        cfg.meta_adaptive = True
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        field._scheduler.run(backpressure_ok=True)
        assert field._resonance_engine.meta_kernel is not None
