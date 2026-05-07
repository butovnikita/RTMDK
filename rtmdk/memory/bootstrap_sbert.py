"""
Standalone SBERT bootstrap utility for SOT cold-start.

Usage:
    from rtmdk.memory.bootstrap_sbert import run_bootstrap
    run_bootstrap(
        texts=[...],
        output_path="sot_bootstrap.npz",
        model_name="all-MiniLM-L6-v2",
    )

Then in config:
    sot_bootstrap_projection = "sot_bootstrap.npz"
"""
import json
import logging
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_bootstrap(
    texts: List[str],
    output_path: str = "sot_bootstrap.npz",
    model_name: str = "all-MiniLM-L6-v2",
    n_epochs: int = 30,
    lr: float = 0.05,
    teacher_embed_fn: Optional[Callable[[str], np.ndarray]] = None,
):
    """Generate a bootstrap projection matrix from SBERT (or any teacher).

    The resulting .npz file contains:
        - projection: (token_dim, latent_dim) projection matrix
        - token_embeddings: dict mapping token_id -> embedding vector
        - token_frequencies: dict mapping token_id -> frequency count
    """
    if teacher_embed_fn is None:
        from sentence_transformers import SentenceTransformer
        teacher = SentenceTransformer(model_name)
        def teacher_embed_fn(t): return teacher.encode(
            t, show_progress_bar=False)

    logger.info(f"SBERT bootstrap: {len(texts)} texts -> {output_path}")
    teacher_embs = []
    valid_texts = []
    for text in texts:
        try:
            emb = teacher_embed_fn(text)
            if emb is not None and np.isfinite(emb).all():
                teacher_embs.append(emb)
                valid_texts.append(text)
        except Exception:
            continue
    if not teacher_embs:
        logger.warning("No valid teacher embeddings")
        return
    teacher_matrix = np.stack(teacher_embs).astype(np.float32)
    teacher_matrix.shape[1]
    logger.info(f"Teacher embeddings shape: {teacher_matrix.shape}")

    # Simple byte tokenizer (same logic as SOT)
    def tokenize(text: str) -> List[int]:
        return [b for b in text.encode("utf-8")]

    token_freq: dict = {}
    token_to_texts: dict = {}
    for idx, text in enumerate(valid_texts):
        tokens = tokenize(text)
        for t in tokens:
            token_freq[t] = token_freq.get(t, 0) + 1
            token_to_texts.setdefault(t, []).append(idx)

    # Build average teacher embedding per token
    token_embeddings = {}
    for t, indices in token_to_texts.items():
        vecs = teacher_matrix[indices]
        token_embeddings[t] = vecs.mean(axis=0)

    # Learn linear projection from token space to teacher space via ridge regression
    # X: (N_samples, vocab) one-hot-ish aggregated counts
    # y: (N_samples, teacher_dim)
    vocab = sorted(token_embeddings.keys())
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_samples = len(valid_texts)
    n_vocab = len(vocab)

    X: np.ndarray = np.zeros((n_samples, n_vocab), dtype=np.float32)
    for i, text in enumerate(valid_texts):
        tokens = tokenize(text)
        for t in tokens:
            j = token_to_idx[t]
            X[i, j] += 1.0
    y = teacher_matrix

    # Ridge regression: W = (X^T X + alpha I)^{-1} X^T y
    alpha = 1.0
    XtX = X.T @ X
    XtX[np.arange(n_vocab), np.arange(n_vocab)] += alpha
    XtY = X.T @ y
    try:
        W = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    # W shape: (n_vocab, teacher_dim)
    # We also store token embeddings aligned to this vocab
    emb_matrix = np.stack([token_embeddings[t]
                          for t in vocab]).astype(np.float32)

    np.savez(
        output_path,
        projection=W,
        vocab=np.array(vocab, dtype=np.int32),
        token_embeddings=emb_matrix,
        token_frequencies=np.array([token_freq[t] for t in vocab], dtype=np.float32),
    )
    logger.info(
        f"Saved bootstrap to {output_path}: vocab={n_vocab}, proj={W.shape}")


def load_bootstrap(path: str, tokenizer):
    """Load a bootstrap file into an existing SOTokenizer."""
    data = np.load(path, allow_pickle=False)
    vocab = data["vocab"].tolist()
    proj = data["projection"]
    embs = data["token_embeddings"]
    freqs = data["token_frequencies"]

    # Map onto tokenizer's existing structures
    for i, t in enumerate(vocab):
        tokenizer.token_embeddings[t] = embs[i]
        tokenizer.token_frequency[t] = float(freqs[i])

    # If dimensions match, overwrite projection
    if proj.shape == tokenizer.projection.shape:
        tokenizer.projection = proj.astype(np.float32)
        logger.info(f"Loaded bootstrap projection {proj.shape} from {path}")
    else:
        logger.warning(
            f"Bootstrap projection shape {proj.shape} mismatches "
            f"tokenizer {tokenizer.projection.shape}; skipping projection load"
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python bootstrap_sbert.py <input_json> <output.npz>")
        print("  input_json: [{\"text\": \"...\"}, ...]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records if "text" in r]
    run_bootstrap(texts, output_path=sys.argv[2])
