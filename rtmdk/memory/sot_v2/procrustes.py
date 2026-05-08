"""Procrustes alignment: distill transformer knowledge into SIF space.

Orthogonal Procrustes Problem (Schönemann, 1966):
    Given source embeddings X (n×d) and target embeddings Y (n×d),
    find orthogonal matrix R (d×d) minimizing ||X·R − Y||_F.

Closed-form solution:
    R = U·Vᵀ  where  U·S·Vᵀ = SVD(Xᵀ·Y)

Properties:
- Preserves vector norms (orthogonal → isometry)
- No PyTorch needed at inference time (pure NumPy dot product)
- One-shot: fit once on corpus, apply forever
- Theoretical guarantee: optimal linear isometric alignment

References:
- Schönemann (1966). A generalized solution of the orthogonal Procrustes problem.
- Smith et al. (2017). Offline bilingual word vectors without parallel data.
- Artetxe et al. (2018). A robust self-learning method for fully unsupervised
  cross-lingual mappings of word embeddings.
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


class ProcrustesAligner:
    """Aligns source embeddings to target embeddings via orthogonal Procrustes."""

    def __init__(self, d: Optional[int] = None):
        self.d = d
        self.R: Optional[NDArray] = None          # alignment matrix (d×d)
        self.src_mean: Optional[NDArray] = None   # source centering vector
        self.tgt_mean: Optional[NDArray] = None   # target centering vector
        self._fitted = False

    def fit(
        self,
        src_embs: NDArray,
        tgt_embs: NDArray,
        center: bool = True,
        scale: bool = False,
    ) -> "ProcrustesAligner":
        """Fit orthogonal alignment matrix R.

        Args:
            src_embs: Source embeddings, shape (n, d)
            tgt_embs: Target embeddings, shape (n, d)
            center: Whether to mean-center both spaces before alignment.
                Recommended True — removes location bias.
            scale: Whether to normalize Frobenius norm of source before alignment.
                Usually not needed for embeddings already on same scale.
        """
        if src_embs.shape != tgt_embs.shape:
            raise ValueError(
                f"Shape mismatch: src {src_embs.shape} vs tgt {tgt_embs.shape}"
            )
        n, d = src_embs.shape
        self.d = d

        X = src_embs.astype(np.float64)
        Y = tgt_embs.astype(np.float64)

        # Centering
        if center:
            self.src_mean = X.mean(axis=0)
            self.tgt_mean = Y.mean(axis=0)
            X = X - self.src_mean
            Y = Y - self.tgt_mean
        else:
            self.src_mean = np.zeros(d, dtype=np.float64)
            self.tgt_mean = np.zeros(d, dtype=np.float64)

        # Optional scaling (Frobenius normalization)
        if scale:
            x_norm = np.linalg.norm(X, "fro")
            if x_norm > 1e-8:
                X = X / x_norm

        # Core: Orthogonal Procrustes via SVD
        # Minimize ||X·R − Y||_F  →  R = U·Vᵀ  where  USVᵀ = SVD(Xᵀ·Y)
        M = X.T @ Y  # (d, d)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        self.R = (U @ Vt).astype(np.float32)

        # Diagnostics
        aligned = self._transform_raw(X.astype(np.float32))
        mse = float(np.mean((aligned - Y.astype(np.float32)) ** 2))
        cosine_sim = float(
            np.mean(
                np.sum(aligned * Y.astype(np.float32), axis=1)
                / (np.linalg.norm(aligned, axis=1) * np.linalg.norm(Y.astype(np.float32), axis=1) + 1e-8)
            )
        )
        self._diagnostics = {
            "n_samples": n,
            "dim": d,
            "mse": mse,
            "mean_cosine": cosine_sim,
            "singular_values": S.tolist(),
        }
        self._fitted = True
        return self

    def _transform_raw(self, X: NDArray) -> NDArray:
        """Apply alignment without centering (internal)."""
        assert self.R is not None
        return X @ self.R

    def transform(self, src_emb: NDArray) -> NDArray:
        """Apply alignment to source embedding(s).

        Args:
            src_emb: Single vector (d,) or batch (n, d).
        Returns:
            Aligned embedding(s), same shape.
        """
        if not self._fitted:
            raise RuntimeError("ProcrustesAligner not fitted. Call fit() first.")
        assert self.R is not None and self.src_mean is not None and self.tgt_mean is not None

        arr = np.asarray(src_emb, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[np.newaxis, :]

        centered = arr - self.src_mean.astype(np.float32)
        aligned = centered @ self.R
        result = aligned + self.tgt_mean.astype(np.float32)

        if single:
            return result[0]
        return result

    def diagnostics(self) -> dict:
        if not self._fitted:
            return {}
        return dict(self._diagnostics)

    def save(self, path: str):
        """Save alignment parameters to NPZ."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        np.savez(
            path,
            R=self.R,
            src_mean=self.src_mean,
            tgt_mean=self.tgt_mean,
            diagnostics=self._diagnostics,
        )

    @classmethod
    def load(cls, path: str) -> "ProcrustesAligner":
        """Load alignment parameters from NPZ."""
        data = np.load(path, allow_pickle=True)
        inst = cls()
        inst.R = data["R"]
        inst.src_mean = data["src_mean"]
        inst.tgt_mean = data["tgt_mean"]
        inst._diagnostics = data["diagnostics"].item()
        inst.d = inst.R.shape[0]
        inst._fitted = True
        return inst


def procrustes_align_sif_to_teacher(
    sif_embedder,
    corpus_texts: List[str],
    teacher,
    batch_size: int = 64,
    center: bool = True,
) -> ProcrustesAligner:
    """Convenience: fit Procrustes alignment from a SIF embedder to a teacher model.

    Args:
        sif_embedder: Instance with .embed(token_ids) and .tokenizer.encode(text).
        corpus_texts: Training corpus.
        teacher: Callable(texts) -> np.ndarray of teacher embeddings.
        batch_size: Batch size for teacher inference.
        center: Mean-center before alignment.

    Returns:
        Fitted ProcrustesAligner.
    """
    # Encode corpus with SIF
    sif_embs = []
    for text in corpus_texts:
        tok = sif_embedder.tokenizer.encode(text)
        sif_embs.append(sif_embedder.embed(tok))
    sif_embs = np.stack(sif_embs, axis=0)

    # Encode corpus with teacher (batched)
    teacher_embs = []
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i : i + batch_size]
        embs = teacher(batch)
        if not isinstance(embs, np.ndarray):
            embs = np.asarray(embs)
        teacher_embs.append(embs)
    teacher_embs = np.concatenate(teacher_embs, axis=0)

    # Ensure same dimensionality
    if sif_embs.shape[1] != teacher_embs.shape[1]:
        raise ValueError(
            f"Dimension mismatch: SIF {sif_embs.shape[1]} vs teacher {teacher_embs.shape[1]}"
        )

    aligner = ProcrustesAligner(d=sif_embs.shape[1])
    aligner.fit(sif_embs, teacher_embs, center=center)
    return aligner
