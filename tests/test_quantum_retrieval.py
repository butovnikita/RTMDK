"""Tests for rtmdk/memory/sot_v2/quantum.py — density-matrix resonance retrieval."""

import numpy as np
import pytest

from rtmdk.memory.sot_v2.quantum import QuantumResonanceRetriever

DIM = 8


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def normalized(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class TestAddDocument:
    def test_density_matrix_is_valid(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        embs = normalized(rng.standard_normal((5, DIM)))
        retriever.add_document("d1", embs)

        rho = retriever.doc_states["d1"]
        assert rho.shape == (DIM, DIM)
        # Trace normalized to 1
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-5)
        # Positive semi-definite
        eigs = np.linalg.eigvalsh(rho)
        assert eigs.min() >= -1e-6
        assert retriever.num_docs() == 1

    def test_shape_mismatch_raises(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        with pytest.raises(ValueError, match="shape mismatch"):
            retriever.add_document("bad", rng.standard_normal((3, DIM + 1)))
        with pytest.raises(ValueError, match="shape mismatch"):
            retriever.add_document("bad", rng.standard_normal(DIM))

    def test_empty_document_gets_regularized_identity(self):
        retriever = QuantumResonanceRetriever(latent_dim=DIM, epsilon=1e-3)
        retriever.add_document("empty", np.zeros((0, DIM)))

        np.testing.assert_allclose(
            retriever.doc_states["empty"],
            np.eye(DIM) * 1e-3,
            atol=1e-8,
        )

    def test_coherence_path(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM, use_coherence=True, window_size=2)
        embs = normalized(rng.standard_normal((6, DIM)))
        retriever.add_document("coh", embs, token_ids=[1, 2, 1, 3, 2, 1])

        rho = retriever.doc_states["coh"]
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-5)
        assert np.linalg.eigvalsh(rho).min() >= -1e-6
        # Coherence must change the matrix vs. no-coherence version
        plain = QuantumResonanceRetriever(latent_dim=DIM)
        plain.add_document("coh", embs)
        assert not np.allclose(rho, plain.doc_states["coh"])

    def test_local_cooc_counts_pairs(self):
        retriever = QuantumResonanceRetriever(latent_dim=DIM, window_size=1)
        cooc = retriever._local_cooc([1, 2, 1])
        # window=1: pairs (1,2) from positions (0,1) and (1,2) → counted twice
        assert cooc == {(1, 2): 2.0}


class TestQuery:
    def test_matching_document_ranks_first(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        target = normalized(rng.standard_normal((1, DIM)))[0]
        other = normalized(rng.standard_normal((4, DIM)))

        retriever.add_document("hit", normalized(np.vstack([target, rng.standard_normal((2, DIM))])))
        retriever.add_document("miss", other)

        results = retriever.query(target, top_k=2)
        assert results[0][0] == "hit"
        assert results[0][1] > results[1][1] > 0.0

    def test_top_k_truncation(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        for i in range(5):
            retriever.add_document(f"d{i}", normalized(rng.standard_normal((3, DIM))))

        results = retriever.query(normalized(rng.standard_normal((1, DIM)))[0], top_k=3)
        assert len(results) == 3
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_empty_document_slow_path(self):
        """Docs without a factor (empty docs) use the q @ rho @ q path."""
        retriever = QuantumResonanceRetriever(latent_dim=DIM, epsilon=1e-2)
        retriever.add_document("empty", np.zeros((0, DIM)))

        q = np.ones(DIM, dtype=np.float32)
        doc_id, score = retriever.query(q, top_k=1)[0]
        assert doc_id == "empty"
        # rho = eps * I → q^T rho q = eps * ||q||^2
        assert score == pytest.approx(1e-2 * DIM, rel=1e-5)

    def test_query_on_empty_store(self):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        assert retriever.query(np.ones(DIM)) == []


class TestClear:
    def test_clear_resets_state(self, rng):
        retriever = QuantumResonanceRetriever(latent_dim=DIM)
        retriever.add_document("d", normalized(rng.standard_normal((2, DIM))))
        retriever.clear()

        assert retriever.num_docs() == 0
        assert retriever.doc_meta == {}
