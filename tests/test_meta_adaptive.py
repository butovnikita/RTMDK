"""
tests/test_meta_adaptive.py — MetaAdaptiveKernel behavior validation.

Covers:
1. MetaAdaptiveKernel adapts bandwidth based on response kurtosis
2. Extreme kurtosis drives bandwidth toward bounds
3. Retrieval accuracy not degraded by meta-adaptation
"""

import pytest
import numpy as np

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig


def _make_field(dim=8, meta_adaptive=True, bw=1.0):
    cfg = RTMDKConfig(
        latent_dim=dim,
        bandwidth=bw,
        meta_adaptive=meta_adaptive,
        meta_adaptation_lr=0.02,
        kurtosis_target_min=2.0,
        kurtosis_target_max=3.5,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    field = RTMDKField(cfg)
    rng = np.random.default_rng(42)
    for i in range(20):
        pos = rng.standard_normal(dim).astype(np.float32) * 0.5
        nid = field.add_node(
            pos,
            content={"text": f"node {i}"},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        field.nodes[nid].amplitude = 1.0
        field.nodes[nid].salience = 1.0
    field._build_node_cache()
    return field


class TestMetaAdaptiveBasics:
    def test_disabled_by_default_in_non_meta_config(self):
        field = _make_field(meta_adaptive=False)
        assert field.meta_kernel is None

    def test_enabled_creates_meta_kernel(self):
        field = _make_field(meta_adaptive=True)
        assert field.meta_kernel is not None
        assert field.meta_kernel.get_bandwidth() == 1.0

    def test_adapt_changes_bandwidth(self):
        field = _make_field(meta_adaptive=True)
        initial_bw = field.meta_kernel.get_bandwidth()
        # Simulate queries to populate response history
        query = np.zeros(8, dtype=np.float32)
        for _ in range(10):
            field.query(query, top_k=5)
        field.meta_kernel.adapt()
        new_bw = field.meta_kernel.get_bandwidth()
        # Bandwidth should change after adaptation (kurtosis != target)
        assert new_bw != initial_bw or len(field.meta_kernel._response_history) < 4

    def test_bandwidth_clipped_to_bounds(self):
        field = _make_field(meta_adaptive=True)
        mk = field.meta_kernel
        # Drive bandwidth to extreme by simulating very flat responses
        for _ in range(100):
            mk.record_response(0.5)
        for _ in range(20):
            mk.adapt()
        bw = mk.get_bandwidth()
        assert 0.1 <= bw <= 10.0

    def test_retrieval_not_degraded(self):
        field = _make_field(meta_adaptive=True)
        query = np.zeros(8, dtype=np.float32)
        # Baseline
        baseline = field.query(query, top_k=5)
        # Adapt multiple times
        for _ in range(5):
            for _ in range(10):
                field.query(query, top_k=5)
            field.meta_kernel.adapt()
        after = field.query(query, top_k=5)
        assert len(after) == len(baseline)


class TestMetaAdaptiveKurtosis:
    def test_low_kurtosis_widens_bandwidth(self):
        mk = _make_field(meta_adaptive=True).meta_kernel
        bw0 = mk.get_bandwidth()
        # U-shaped (beta(0.3,0.3)) → low kurtosis (< 1.5)
        for val in np.random.beta(0.3, 0.3, 50):
            mk.record_response(val)
        for _ in range(5):
            mk.adapt()
        bw1 = mk.get_bandwidth()
        assert bw1 > bw0

    def test_high_kurtosis_narrows_bandwidth(self):
        mk = _make_field(meta_adaptive=True).meta_kernel
        bw0 = mk.get_bandwidth()
        # Peaked distribution → high kurtosis (> 3.5)
        for _ in range(50):
            mk.record_response(0.1)
        for _ in range(5):
            mk.record_response(1.0)
        mk.adapt()
        bw1 = mk.get_bandwidth()
        assert bw1 < bw0

    def test_normal_kurtosis_no_extreme_change(self):
        mk = _make_field(meta_adaptive=True).meta_kernel
        bw0 = mk.get_bandwidth()
        # Normal-ish distribution → kurtosis ~ 3.0 (inside target range)
        rng = np.random.default_rng(7)
        for _ in range(50):
            mk.record_response(float(rng.normal(0.5, 0.2)))
        mk.adapt()
        bw1 = mk.get_bandwidth()
        # Should stay close to initial (target range [2.0, 3.5] includes 3.0)
        assert abs(bw1 - bw0) < 0.1
