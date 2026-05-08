"""RAG Quality improvements: sentence reranking, query decomposition, feedback loop."""
from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple regex-based splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


class SentenceReranker:
    """Re-rank documents by max sentence-level similarity.

    Instead of document-level cosine, split documents into sentences
    and score by the best-matching sentence.  This is more precise
    for long documents where only one paragraph is relevant.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, any]],
        top_k: int = 5,
    ) -> List[Tuple[str, float, any]]:
        """Re-rank results by sentence-level similarity."""
        if not results:
            return results

        q_emb = self.embedder(query)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        scored = []
        for nid, orig_score, node in results:
            text = node.content.get("text", "")
            if not text:
                scored.append((nid, orig_score, node))
                continue

            sentences = _split_sentences(text)
            if len(sentences) <= 1:
                scored.append((nid, orig_score, node))
                continue

            try:
                sent_embs = []
                for sent in sentences:
                    se = self.embedder(sent)
                    se = se / (np.linalg.norm(se) + 1e-8)
                    sent_embs.append(se)
                sent_matrix = np.stack(sent_embs)
                sims = sent_matrix @ q_emb
                max_sim = float(np.max(sims))
                # Blend original score with sentence max-sim
                blended = 0.6 * orig_score + 0.4 * max_sim
                scored.append((nid, blended, node))
            except Exception:
                scored.append((nid, orig_score, node))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class QueryDecomposer:
    """Decompose complex queries into sub-queries for multi-hop retrieval.

    Example:
        "What causes earthquakes and how do we predict them?"
        → ["What causes earthquakes?", "how do we predict earthquakes?"]
    """

    SPLIT_PATTERN = re.compile(
        r'\b(and|plus|also|additionally|furthermore|moreover)\b',
        re.IGNORECASE,
    )

    def decompose(self, query: str) -> List[str]:
        """Split query into sub-queries."""
        query = query.strip()
        if not query:
            return []

        parts = self.SPLIT_PATTERN.split(query)
        if len(parts) == 1:
            return [query]

        sub_queries = []
        current = parts[0].strip()
        for i in range(1, len(parts), 2):
            conjunction = parts[i].strip().lower() if i < len(parts) else ""
            rest = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if conjunction in ("and", "plus", "also", "additionally"):
                if current:
                    sub_queries.append(current)
                current = rest
            else:
                current = current + " " + conjunction + " " + rest
        if current:
            sub_queries.append(current)

        # Filter out empty or too-short sub-queries
        sub_queries = [q for q in sub_queries if len(q) > 8]
        return sub_queries if sub_queries else [query]


class FeedbackLoop:
    """Explicit feedback loop for embedding refinement.

    Users provide (query, node_id, relevant) feedback.
    Positive feedback nudges query→node embeddings closer.
    """

    def __init__(self, embedder, lr: float = 0.001):
        self.embedder = embedder
        self.lr = lr
        self.feedback_count = 0

    def add_feedback(
        self,
        query: str,
        node_text: str,
        relevant: bool,
    ) -> bool:
        """Apply feedback to word embeddings.

        Args:
            query: The query text.
            node_text: The text of the retrieved node.
            relevant: True if relevant, False if irrelevant.

        Returns:
            True if feedback was applied.
        """
        # Only works with SOTv2Embedder which exposes vocab and _embedder
        if not hasattr(self.embedder, '_vocab') or not hasattr(self.embedder, '_embedder'):
            logger.debug("FeedbackLoop: embedder does not support online updates")
            return False

        vocab = self.embedder._vocab
        sif = self.embedder._embedder
        if not hasattr(sif, 'word_embeddings'):
            return False

        q_tokens = [vocab[w] for w in self.embedder._word_tokenize(query) if w in vocab]
        n_tokens = [vocab[w] for w in self.embedder._word_tokenize(node_text) if w in vocab]
        if not q_tokens or not n_tokens:
            return False

        q_emb = sif._embed_sentence_raw(q_tokens)
        n_emb = sif._embed_sentence_raw(n_tokens)
        if q_emb is None or n_emb is None:
            return False

        # Gradient: push towards each other if relevant, away if not
        direction = 1.0 if relevant else -1.0
        delta = (n_emb - q_emb) * direction * self.lr

        # Update query word embeddings
        for t in q_tokens:
            if t in sif.word_embeddings:
                sif.word_embeddings[t] += delta / len(q_tokens)
                sif.word_embeddings[t] /= np.linalg.norm(sif.word_embeddings[t]) + 1e-8

        # Update node word embeddings
        for t in n_tokens:
            if t in sif.word_embeddings:
                sif.word_embeddings[t] -= delta / len(n_tokens)
                sif.word_embeddings[t] /= np.linalg.norm(sif.word_embeddings[t]) + 1e-8

        self.feedback_count += 1
        logger.info("FeedbackLoop: applied %s feedback (#%d)", "positive" if relevant else "negative", self.feedback_count)
        return True
