"""Tests for rtmdk/memory/sot_v2/procrustes.py — orthogonal Procrustes alignment."""

import numpy as np
import pytest

from rtmdk.memory.sot_v2.procrustes import ProcrustesAligner, procrustes_align_sif_to_teacher

DIM = 12
N = 60


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_orthogonal(d, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


class TestFit:
    def test_recovers_known_rotation(self, rng):
        """Y = X @ R_true → fitted R must recover R_true and map X back to Y."""
        X = rng.standard_normal((N, DIM))
        R_true = random_orthogonal(DIM, rng)
        Y = X @ R_true

        aligner = ProcrustesAligner().fit(X, Y)

        assert aligner._fitted
        assert aligner.R.shape == (DIM, DIM)
        np.testing.assert_allclose(aligner.R, R_true, atol=1e-4)
        np.testing.assert_allclose(aligner.transform(X), Y, atol=1e-3)
        diag = aligner.diagnostics()
        assert diag["mse"] < 1e-6
        assert diag["mean_cosine"] > 0.9999

    def test_recovers_rotation_with_translation(self, rng):
        """Centering must absorb a constant offset between spaces."""
        X = rng.standard_normal((N, DIM)) + 5.0
        R_true = random_orthogonal(DIM, rng)
        Y = X @ R_true - 3.0

        aligner = ProcrustesAligner().fit(X, Y, center=True)

        np.testing.assert_allclose(aligner.transform(X), Y, atol=1e-3)

    def test_identity_matrices_give_identity_rotation(self, rng):
        X = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner().fit(X, X)

        np.testing.assert_allclose(aligner.R, np.eye(DIM), atol=1e-4)
        assert aligner.diagnostics()["mse"] < 1e-8

    def test_r_is_orthogonal(self, rng):
        X = rng.standard_normal((N, DIM))
        Y = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner().fit(X, Y)

        np.testing.assert_allclose(aligner.R.T @ aligner.R, np.eye(DIM), atol=1e-5)

    def test_shape_mismatch_raises(self, rng):
        X = rng.standard_normal((N, DIM))
        Y = rng.standard_normal((N, DIM + 1))
        with pytest.raises(ValueError, match="Shape mismatch"):
            ProcrustesAligner().fit(X, Y)

    def test_scale_option_keeps_orthogonality(self, rng):
        X = rng.standard_normal((N, DIM)) * 100.0
        Y = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner().fit(X, Y, scale=True)

        np.testing.assert_allclose(aligner.R.T @ aligner.R, np.eye(DIM), atol=1e-5)

    def test_fit_returns_self(self, rng):
        X = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner(d=DIM)
        assert aligner.fit(X, X) is aligner
        assert aligner.d == DIM


class TestTransform:
    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            ProcrustesAligner().transform(np.zeros(DIM))

    def test_single_vector_shape_and_dtype(self, rng):
        X = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner().fit(X, X)

        out = aligner.transform(X[0])
        assert out.shape == (DIM,)
        assert out.dtype == np.float32

    def test_batch_matches_single(self, rng):
        X = rng.standard_normal((N, DIM))
        Y = X @ random_orthogonal(DIM, rng)
        aligner = ProcrustesAligner().fit(X, Y)

        batch = aligner.transform(X[:5])
        singles = np.stack([aligner.transform(x) for x in X[:5]])
        assert batch.shape == (5, DIM)
        np.testing.assert_allclose(batch, singles, atol=1e-6)

    def test_isometry_without_centering(self, rng):
        """With center=False, transform is a pure rotation → preserves norms."""
        X = rng.standard_normal((N, DIM))
        Y = X @ random_orthogonal(DIM, rng)
        aligner = ProcrustesAligner().fit(X, Y, center=False)

        x = rng.standard_normal(DIM)
        out = aligner.transform(x)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(x), rel=1e-4)


class TestDiagnostics:
    def test_empty_before_fit(self):
        assert ProcrustesAligner().diagnostics() == {}

    def test_contents(self, rng):
        X = rng.standard_normal((N, DIM))
        aligner = ProcrustesAligner().fit(X, X)
        diag = aligner.diagnostics()

        assert diag["n_samples"] == N
        assert diag["dim"] == DIM
        assert len(diag["singular_values"]) == DIM
        assert set(diag) == {"n_samples", "dim", "mse", "mean_cosine", "singular_values"}


class TestPersistence:
    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Not fitted"):
            ProcrustesAligner().save(str(tmp_path / "a.npz"))

    def test_roundtrip(self, tmp_path, rng):
        X = rng.standard_normal((N, DIM))
        Y = X @ random_orthogonal(DIM, rng) + 2.0
        aligner = ProcrustesAligner().fit(X, Y)

        path = str(tmp_path / "aligner.npz")
        aligner.save(path)
        loaded = ProcrustesAligner.load(path)

        assert loaded._fitted
        assert loaded.d == DIM
        np.testing.assert_allclose(loaded.R, aligner.R, atol=1e-6)
        np.testing.assert_allclose(loaded.src_mean, aligner.src_mean, atol=1e-6)
        np.testing.assert_allclose(loaded.tgt_mean, aligner.tgt_mean, atol=1e-6)
        np.testing.assert_allclose(loaded.transform(X), aligner.transform(X), atol=1e-5)
        assert loaded.diagnostics()["n_samples"] == N


class _FakeTokenizer:
    def encode(self, text):
        return [hash(text) % 1000]


class _FakeSifEmbedder:
    def __init__(self, table):
        self.tokenizer = _FakeTokenizer()
        self._table = table

    def embed(self, token_ids):
        return self._table[token_ids[0]]


class TestConvenienceFunction:
    def test_happy_path(self, rng):
        texts = [f"document number {i}" for i in range(10)]
        table = rng.standard_normal((1000, DIM))
        sif = _FakeSifEmbedder(table)
        R_true = random_orthogonal(DIM, rng)
        teacher = lambda batch: np.stack([table[hash(t) % 1000] @ R_true for t in batch])  # noqa: E731

        aligner = procrustes_align_sif_to_teacher(sif, texts, teacher, batch_size=4)

        assert aligner._fitted
        sif_embs = np.stack([table[hash(t) % 1000] for t in texts])
        np.testing.assert_allclose(aligner.transform(sif_embs), sif_embs @ R_true, atol=1e-3)

    def test_dimension_mismatch_raises(self, rng):
        texts = ["a", "b", "c"]
        sif = _FakeSifEmbedder(rng.standard_normal((1000, DIM)))
        teacher = lambda batch: rng.standard_normal((len(batch), DIM + 2))  # noqa: E731

        with pytest.raises(ValueError, match="Dimension mismatch"):
            procrustes_align_sif_to_teacher(sif, texts, teacher)
