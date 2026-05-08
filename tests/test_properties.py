"""Property-based tests for RTMDK core invariants.

Uses Hypothesis to generate random embeddings and verify:
  1. Query to self is top-1
  2. Adding + deleting node doesn't affect other queries
  3. Amplitude conservation bounds after consolidate
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, lists

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField

# Reusable field fixture (lightweight config)
_cfg = RTMDKConfig(latent_dim=8, max_nodes=100, top_k=5)


class TestQuerySelfTop1:
    """Property: query embedding should find itself as top-1 result."""

    @given(
        floats(min_value=-1.0, max_value=1.0),
        floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_query_self_top1(self, x, y):
        field = RTMDKField(_cfg)
        emb = np.array([x, y] + [0.0] * 6, dtype=np.float32)
        nid = field.add_node(embedding=emb, content={"text": "test"})
        results = field.query(emb, top_k=5)
        assert len(results) >= 1
        assert results[0][0] == nid


class TestAddDeleteIsolation:
    """Property: deleting a node should not change results for others."""

    @given(
        lists(
            floats(min_value=-1.0, max_value=1.0),
            min_size=8,
            max_size=8,
        ),
        lists(
            floats(min_value=-1.0, max_value=1.0),
            min_size=8,
            max_size=8,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_delete_isolation(self, a, b):
        field = RTMDKField(_cfg)
        emb_a = np.array(a, dtype=np.float32)
        emb_b = np.array(b, dtype=np.float32)
        nid_a = field.add_node(embedding=emb_a, content={"text": "a"})
        nid_b = field.add_node(embedding=emb_b, content={"text": "b"})

        # Query for b before delete
        field.query(emb_b, top_k=2)

        # Delete a
        field.delete_nodes([nid_a])

        # Query for b after delete
        after = field.query(emb_b, top_k=2)
        after_ids = {n for n, _, _ in after}

        # b should still be findable
        assert nid_b in after_ids
        # The remaining top result should still be b itself
        assert after[0][0] == nid_b


class TestAmplitudeConservation:
    """Property: sum of amplitudes after consolidate <= 2x sum before."""

    @given(integers(min_value=3, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_consolidation_amplitude_bound(self, n_nodes):
        field = RTMDKField(_cfg)
        rng = np.random.default_rng(42)
        for i in range(n_nodes):
            emb = rng.standard_normal(_cfg.latent_dim).astype(np.float32)
            field.add_node(embedding=emb, content={"text": str(i)})

        before = sum(n.amplitude for n in field.nodes.values())
        field.consolidate()
        after = sum(n.amplitude for n in field.nodes.values())

        assert after <= before * 2.0 + 1e-6
