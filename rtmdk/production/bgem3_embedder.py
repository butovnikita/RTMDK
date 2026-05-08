"""BGE-M3 embedder wrapper for RTMDK.

Provides dense + sparse + colbert representations from a single model.
Lazy-loads the model on first use to avoid startup overhead.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class BGEM3Embedding:
    """Container for BGE-M3 multi-representation output."""

    def __init__(self, dense: np.ndarray, sparse: Optional[Dict[int, float]] = None,
                 colbert: Optional[np.ndarray] = None):
        self.dense = dense
        self.sparse = sparse or {}
        self.colbert = colbert


class BGEM3Embedder:
    """Lazy-loading BGE-M3 embedder."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True,
                 batch_size: int = 12, max_length: int = 8192):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            logger.info(f"Loaded BGE-M3 model: {self.model_name}")
        except Exception as exc:
            logger.error(f"Failed to load BGE-M3: {exc}")
            raise

    def encode(self, texts: List[str]) -> List[BGEM3Embedding]:
        """Encode a batch of texts into dense + sparse + colbert."""
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        try:
            output = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,  # Skip colbert for speed unless needed
            )
            dense_vecs = output["dense_vecs"]
            sparse_vecs = output.get("lexical_weights", [])
            results = []
            for i, d in enumerate(dense_vecs):
                sparse = sparse_vecs[i] if i < len(sparse_vecs) else {}
                results.append(BGEM3Embedding(dense=d.astype(np.float32),
                                               sparse=sparse))
            return results
        except Exception as exc:
            logger.error(f"BGE-M3 encode failed: {exc}")
            raise

    def encode_query(self, text: str) -> BGEM3Embedding:
        """Encode a single query."""
        results = self.encode([text])
        return results[0]

    def __call__(self, text: str) -> np.ndarray:
        """Return dense vector only (compatible with Callable[[str], np.ndarray])."""
        emb = self.encode_query(text)
        return emb.dense
