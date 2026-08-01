"""SOT v2.0 Integration — Self-contained embedder for RTMDK.

Usage:
    embedder = SOTv2Embedder(latent_dim=384)
    embedder.train(corpus_texts)          # one-time training on corpus
    emb = embedder("query text")          # returns np.ndarray (latent_dim,)
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np

from .sif_embedder import SIFEmbedder

logger = logging.getLogger(__name__)


def _word_tokenize(text: str) -> List[str]:
    """Unicode-aware word tokenization."""
    text = text.lower()
    tokens = []
    current = []
    for ch in text:
        is_cjk = (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u309f"
            or "\u30a0" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        )
        if is_cjk:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for text in texts:
        for word in _word_tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class SOTv2Embedder:
    """Fully self-contained sentence embedder using SIF over PMI word vectors.

    No external model dependencies after training.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        window_size: int = 5,
        a: float = 1e-3,
        remove_pc: bool = True,
    ):
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.a = a
        self.remove_pc = remove_pc

        self._trained = False
        self._vocab: Dict[str, int] = {}
        self._embedder: Optional[SIFEmbedder] = None

    def train(
        self,
        corpus_texts: List[str],
        tokenized_queries: Optional[List[List[int]]] = None,
        tokenized_positives: Optional[List[List[int]]] = None,
        contrastive_epochs: int = 30,
    ) -> "SOTv2Embedder":
        """Train on a corpus of raw texts.

        Args:
            corpus_texts: List of strings (e.g. all document texts + queries).
            tokenized_queries: Optional query token IDs for contrastive fine-tuning.
            tokenized_positives: Optional positive context token IDs for contrastive fine-tuning.
            contrastive_epochs: Number of contrastive fine-tuning epochs.
        """
        if not corpus_texts:
            raise ValueError("corpus_texts must not be empty")
        logger.info("SOTv2Embedder: training on %d texts", len(corpus_texts))

        self._vocab = _build_vocab(corpus_texts)
        logger.info("SOTv2Embedder: vocab size = %d", len(self._vocab))

        tokenized = [
            [self._vocab[w] for w in _word_tokenize(text)]
            for text in corpus_texts
        ]

        self._id2word = {v: k for k, v in self._vocab.items()}

        self._embedder = SIFEmbedder(
            latent_dim=self.latent_dim,
            window_size=self.window_size,
            min_count=1,
            a=self.a,
            remove_pc=self.remove_pc,
        )
        self._embedder.fit(tokenized, vocab_size=len(self._vocab), id2word=self._id2word)

        # Contrastive fine-tuning if query/positive pairs provided
        if tokenized_queries is not None and tokenized_positives is not None:
            logger.info("SOTv2Embedder: running contrastive fine-tuning...")
            self._embedder.contrastive_fine_tune(
                tokenized_queries,
                tokenized_positives,
                n_epochs=contrastive_epochs,
            )

        self._trained = True
        logger.info("SOTv2Embedder: training complete")
        return self

    def align_to_teacher(
        self,
        corpus_texts: List[str],
        teacher: Callable[[List[str]], np.ndarray],
        batch_size: int = 64,
        center: bool = True,
    ) -> "SOTv2Embedder":
        """Align SIF embeddings to a teacher model via orthogonal Procrustes.

        This distills the teacher's semantic knowledge into the lightweight
        SIF space.  At inference time no PyTorch / transformers are needed.

        Args:
            corpus_texts: Representative corpus for alignment.
            teacher: Callable(texts) -> embeddings array (n, latent_dim).
            batch_size: Batch size for teacher inference.
            center: Mean-center both spaces before alignment (recommended).
        """
        if not self._trained or self._embedder is None:
            raise RuntimeError("Train the embedder before alignment.")
        logger.info("SOTv2Embedder: aligning to teacher on %d texts...", len(corpus_texts))

        # Encode corpus with SIF
        sif_embs = self.embed_batch(corpus_texts)

        # Encode corpus with teacher (batched) and normalize
        teacher_embs = []
        for i in range(0, len(corpus_texts), batch_size):
            batch = corpus_texts[i : i + batch_size]
            embs = teacher(batch)
            if not isinstance(embs, np.ndarray):
                embs = np.asarray(embs)
            # Normalize to unit sphere for consistent Procrustes alignment
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = embs / norms
            teacher_embs.append(embs)
        teacher_embs = np.concatenate(teacher_embs, axis=0)

        self._embedder.align_to_teacher(sif_embs, teacher_embs, center=center)
        logger.info("SOTv2Embedder: alignment complete")
        return self

    def save_aligner(self, path: str):
        """Save the Procrustes alignment matrix to an NPZ file."""
        if self._embedder is None or self._embedder._aligner is None:
            raise RuntimeError("No aligner to save.")
        self._embedder._aligner.save(path)

    def load_aligner(self, path: str):
        """Load a previously saved Procrustes alignment matrix."""
        from .procrustes import ProcrustesAligner
        if self._embedder is None:
            raise RuntimeError("Train the embedder before loading aligner.")
        self._embedder._aligner = ProcrustesAligner.load(path)
        logger.info("SOTv2Embedder: aligner loaded from %s", path)

    def __call__(self, text: str, expand_query: bool = False) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text.
            expand_query: If True, expand query with PMI-based synonyms
                (useful for short queries, no effect on long documents).
        """
        if not self._trained:
            raise RuntimeError("SOTv2Embedder has not been trained yet. Call .train() first.")
        tokens = [self._vocab[w] for w in _word_tokenize(text) if w in self._vocab]
        if expand_query and self._embedder is not None:
            expanded = self._embedder.expand_query(tokens, n_terms=3, min_pmi=0.5)
            for t, weight in expanded[:3]:
                # Add expansion token multiple times proportional to weight
                n_copies = max(1, int(weight))
                tokens.extend([t] * n_copies)
        return self._embedder.embed(tokens)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.vstack([self(t) for t in texts])

    def expand_query_terms(self, text: str, n_terms: int = 3) -> List[Tuple[str, float]]:
        """Return human-readable query expansion terms with PMI scores."""
        if not self._trained or self._embedder is None:
            return []
        tokens = [self._vocab[w] for w in _word_tokenize(text) if w in self._vocab]
        expanded = self._embedder.expand_query(tokens, n_terms=n_terms)
        # Reverse vocab mapping
        rev = {v: k for k, v in self._vocab.items()}
        return [(rev.get(t, str(t)), score) for t, score in expanded]

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "window_size": self.window_size,
            "a": self.a,
            "remove_pc": self.remove_pc,
            "vocab": self._vocab,
            "embedder": self._embedder.get_state() if self._embedder else None,
        }

    def load_state(self, state: dict) -> "SOTv2Embedder":
        self.latent_dim = state["latent_dim"]
        self.window_size = state["window_size"]
        self.a = state["a"]
        self.remove_pc = state["remove_pc"]
        self._vocab = state["vocab"]
        if state["embedder"] is not None:
            self._embedder = SIFEmbedder().load_state(state["embedder"])
        self._trained = True
        return self

    def save(self, path: str):
        """Save embedder state to a JSON file."""
        import json
        state = self.get_state()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("SOTv2Embedder: saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SOTv2Embedder":
        """Load embedder state from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        inst = cls(
            latent_dim=state["latent_dim"],
            window_size=state["window_size"],
            a=state["a"],
            remove_pc=state["remove_pc"],
        )
        inst.load_state(state)
        logger.info("SOTv2Embedder: loaded from %s", path)
        return inst

    def save_npz(self, path: str):
        """Save to compact NPZ binary format (~10x smaller than JSON)."""
        # Vocab: two parallel arrays
        vocab_words = sorted(self._vocab.keys())
        vocab_ids = np.array([self._vocab[w] for w in vocab_words], dtype=np.int32)

        embedder_path = path.replace(".npz", "_embedder.npz")
        self._embedder.save_npz(embedder_path)

        np.savez_compressed(
            path,
            latent_dim=self.latent_dim,
            window_size=self.window_size,
            a=self.a,
            remove_pc=self.remove_pc,
            vocab_words=vocab_words,
            vocab_ids=vocab_ids,
            embedder_file=embedder_path,
        )
        logger.info("SOTv2Embedder: saved NPZ to %s", path)

    @classmethod
    def load_npz(cls, path: str) -> "SOTv2Embedder":
        """Load from compact NPZ binary format."""
        data = np.load(path, allow_pickle=True)
        inst = cls(
            latent_dim=int(data["latent_dim"]),
            window_size=int(data["window_size"]),
            a=float(data["a"]),
            remove_pc=bool(data["remove_pc"]),
        )
        vocab_words = list(data["vocab_words"])
        vocab_ids = data["vocab_ids"]
        inst._vocab = {str(w): int(vid) for w, vid in zip(vocab_words, vocab_ids)}
        embedder_file = str(data["embedder_file"])
        inst._embedder = SIFEmbedder.load_npz(embedder_file)
        inst._trained = True
        logger.info("SOTv2Embedder: loaded NPZ from %s", path)
        return inst
