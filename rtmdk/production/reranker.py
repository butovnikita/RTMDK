"""Cross-encoder reranker for RTMDK retrieval pipeline.

Integrates BGE-reranker-v2-m3 (or any sentence-transformers CrossEncoder)
as a Stage-2 reranker over resonance/retrieval candidates.
"""

import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lightweight cross-encoder reranker for top-K candidates."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self._device)
            logger.info(f"Loaded reranker: {self.model_name}")
        except Exception as exc:
            logger.warning(f"Failed to load cross-encoder {self.model_name}: {exc}")

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Any]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Any]]:
        """Rerank results using cross-encoder scores.

        Args:
            query: Original query text.
            results: List of (node_id, score, node) from resonance retrieval.
            top_k: Return top_k after reranking. If None, keep original length.

        Returns:
            Re-ranked list of (node_id, score, node).
        """
        if not results:
            return results

        self._load()
        if self._model is None:
            return results

        # Prepare query-passage pairs
        pairs = []
        for nid, _, node in results:
            text = ""
            if isinstance(node.content, dict):
                text = node.content.get("text", node.content.get("content", ""))
            else:
                text = str(node.content)
            pairs.append([query, text])

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            # scores is np.ndarray of floats
            scored = []
            for i, (nid, _, node) in enumerate(results):
                scored.append((nid, float(scores[i]), node))
            scored.sort(key=lambda x: x[1], reverse=True)
            if top_k is not None:
                scored = scored[:top_k]
            return scored
        except Exception as exc:
            logger.warning(f"Reranking failed: {exc}")
            return results
