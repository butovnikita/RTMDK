"""Tests for rtmdk/memory/learned_consolidation.py — MLP-based node merge."""

import numpy as np
import pytest

from rtmdk.memory.learned_consolidation import LearnedConsolidator

DIM = 8


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def cons():
    return LearnedConsolidator(latent_dim=DIM)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def make_example(rng, dim=DIM):
    a = unit(rng.standard_normal(dim))
    b = unit(rng.standard_normal(dim))
    queries = [a.copy()]  # query that retrieved parent A
    return a, b, queries


class TestInit:
    def test_default_hidden_dim(self, cons):
        assert cons.hidden_dim == max(64, DIM)
        d_in = DIM * 2 + 8
        assert cons.W1.shape == (cons.hidden_dim, d_in)
        assert cons.b1.shape == (cons.hidden_dim,)
        assert cons.W2.shape == (DIM, cons.hidden_dim)
        assert cons.b2.shape == (DIM,)
        assert cons.W1.dtype == np.float32
        assert cons._trained is False

    def test_explicit_hidden_dim(self):
        cons = LearnedConsolidator(latent_dim=DIM, hidden_dim=16)
        assert cons.W1.shape == (16, DIM * 2 + 8)
        assert cons.W2.shape == (DIM, 16)


class TestEncodePair:
    def test_dimension_and_layout(self):
        d = 4
        a = np.arange(d, dtype=np.float32)
        b = np.arange(d, 2 * d, dtype=np.float32)
        x = LearnedConsolidator._encode_pair(a, b, 0.0, np.pi / 2, 2.0, 3.0, 0.5, 0.25)

        assert x.shape == (2 * d + 8,)
        np.testing.assert_array_equal(x[:d], a)
        np.testing.assert_array_equal(x[d : 2 * d], b)
        # sin(0), sin(pi/2), cos(0), cos(pi/2), amp_a, amp_b, sal_a, sal_b
        np.testing.assert_allclose(x[2 * d :], [0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 0.5, 0.25], atol=1e-7)


class TestPredict:
    def test_unit_norm_output(self, cons, rng):
        a, b, _ = make_example(rng)
        merged = cons.predict(a, b, phase_a=0.3, phase_b=1.1)

        assert merged.shape == (DIM,)
        assert merged.dtype == np.float32
        assert np.linalg.norm(merged) == pytest.approx(1.0, abs=1e-5)

    def test_deterministic(self, rng):
        c1 = LearnedConsolidator(latent_dim=DIM)
        c2 = LearnedConsolidator(latent_dim=DIM)
        a, b, _ = make_example(rng)

        np.testing.assert_array_equal(c1.predict(a, b), c2.predict(a, b))

    def test_depends_on_inputs(self, cons, rng):
        a, b, _ = make_example(rng)
        a2, b2, _ = make_example(rng)

        assert not np.allclose(cons.predict(a, b), cons.predict(a2, b2))


class TestReplayBuffer:
    def test_add_example_stores_copies(self, cons, rng):
        a, b, queries = make_example(rng)
        cons.add_example(a, b, queries)

        a[:] = 999.0
        b[:] = -999.0
        queries[0][:] = 0.0

        ex = cons._buffer[0]
        assert not np.any(ex["latent_a"] == 999.0)
        assert not np.any(ex["latent_b"] == -999.0)
        assert np.linalg.norm(ex["queries"][0]) == pytest.approx(1.0, abs=1e-6)

    def test_buffer_evicts_oldest(self, cons, rng):
        cons._max_buffer = 3
        for i in range(5):
            a = np.full(DIM, float(i), dtype=np.float32)
            cons.add_example(a, a, [])

        assert len(cons._buffer) == 3
        kept = [ex["latent_a"][0] for ex in cons._buffer]
        assert kept == [2.0, 3.0, 4.0]

    def test_scalar_metadata_stored(self, cons, rng):
        a, b, queries = make_example(rng)
        cons.add_example(a, b, queries, phase_a=0.1, phase_b=0.2, amp_a=1.5, amp_b=2.5, sal_a=0.7, sal_b=0.9)

        ex = cons._buffer[0]
        assert (ex["phase_a"], ex["phase_b"]) == (0.1, 0.2)
        assert (ex["amp_a"], ex["amp_b"]) == (1.5, 2.5)
        assert (ex["sal_a"], ex["sal_b"]) == (0.7, 0.9)


class TestTrain:
    def test_small_buffer_skips(self, cons, rng):
        for _ in range(7):
            a, b, q = make_example(rng)
            cons.add_example(a, b, q)

        assert cons.train(epochs=3) == 0.0
        assert cons._trained is False

    def test_train_returns_finite_loss(self, cons, rng):
        for _ in range(10):
            a, b, q = make_example(rng)
            cons.add_example(a, b, q)

        loss = cons.train(epochs=5, lr=0.01, batch_size=4)

        assert np.isfinite(loss)
        assert loss >= 0.0
        assert cons._trained is True

    def test_training_improves_query_similarity(self, rng):
        """After training, merged state must be at least as retrievable for parent queries."""
        cons = LearnedConsolidator(latent_dim=DIM)
        examples = [make_example(rng) for _ in range(12)]
        for a, b, q in examples:
            cons.add_example(a, b, q)

        def mean_query_sim():
            sims = []
            for a, b, q in examples:
                merged = cons.predict(a, b)
                sims.append(float(merged @ unit(q[0])))
            return float(np.mean(sims))

        before = mean_query_sim()
        cons.train(epochs=30, lr=0.01, batch_size=8)
        after = mean_query_sim()

        assert after > before

    def test_training_keeps_trust_region(self, rng):
        """L2 reg term: merged output must stay reasonably close to the heuristic average."""
        cons = LearnedConsolidator(latent_dim=DIM)
        for _ in range(10):
            a, b, q = make_example(rng)
            cons.add_example(a, b, q)
        cons.train(epochs=20, lr=0.01, lambda_reg=1.0)

        a, b, _ = make_example(rng)
        merged = cons.predict(a, b)
        heuristic = unit(0.5 * (a + b))
        assert np.linalg.norm(merged - heuristic) < 1.0


class TestPersistence:
    def test_state_roundtrip(self, rng):
        cons = LearnedConsolidator(latent_dim=DIM)
        for _ in range(9):
            a, b, q = make_example(rng)
            cons.add_example(a, b, q)
        cons.train(epochs=3)

        state = cons.get_state()
        assert state["latent_dim"] == DIM
        assert state["trained"] is True

        restored = LearnedConsolidator(latent_dim=1)  # wrong dim on purpose
        restored.load_state(state)
        assert restored.latent_dim == DIM
        assert restored._trained is True

        a, b, _ = make_example(rng)
        np.testing.assert_array_equal(cons.predict(a, b), restored.predict(a, b))

    def test_untrained_state_roundtrip(self, cons, rng):
        state = cons.get_state()
        assert state["trained"] is False

        restored = LearnedConsolidator(latent_dim=DIM)
        restored.load_state(state)

        a, b, _ = make_example(rng)
        np.testing.assert_array_equal(cons.predict(a, b), restored.predict(a, b))
