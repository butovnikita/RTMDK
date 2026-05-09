"""RAG Quality improvements: sentence reranking, query decomposition, feedback loop."""
from __future__ import annotations
import json
import os
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


class SentenceReranker:
    """Re-rank documents by max sentence-level similarity.

    Supports batched sentence embedding for better throughput.
    """

    def __init__(self, embedder, batch_size: int = 8):
        self.embedder = embedder
        self.batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts in batches if embedder supports it."""
        if hasattr(self.embedder, "embed_batch"):
            return self.embedder.embed_batch(texts)
        return [self.embedder(t) for t in texts]

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
                # Batch embed sentences
                all_texts = sentences
                sent_embs = []
                for i in range(0, len(all_texts), self.batch_size):
                    batch = all_texts[i:i + self.batch_size]
                    batch_embs = self._embed_batch(batch)
                    for se in batch_embs:
                        se = se / (np.linalg.norm(se) + 1e-8)
                        sent_embs.append(se)
                sent_matrix = np.stack(sent_embs)
                sims = sent_matrix @ q_emb
                max_sim = float(np.max(sims))
                blended = 0.6 * orig_score + 0.4 * max_sim
                scored.append((nid, blended, node))
            except Exception:
                scored.append((nid, orig_score, node))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class QueryDecomposer:
    """Decompose complex queries into sub-queries for multi-hop retrieval.

    Supports heuristic AND-splitting and optional LLM-based decomposition.
    """

    SPLIT_PATTERN = re.compile(
        r'\b(and|plus|also|additionally|furthermore|moreover)\b',
        re.IGNORECASE,
    )

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def decompose(self, query: str) -> List[str]:
        """Split query into sub-queries."""
        query = query.strip()
        if not query:
            return []

        # Try LLM-based decomposition if available
        if self.llm_client is not None:
            try:
                return self._decompose_llm(query)
            except Exception:
                logger.debug("LLM decomposition failed, falling back to heuristic")

        # Heuristic fallback
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

        sub_queries = [q for q in sub_queries if len(q) > 8]
        return sub_queries if sub_queries else [query]

    def _decompose_llm(self, query: str) -> List[str]:
        """Use LLM to decompose query into sub-queries."""
        prompt = (
            "Decompose the following user query into 1-3 simpler sub-queries "
            "that can be answered independently. Return ONLY a JSON array of strings.\n\n"
            f"Query: {query}\n\n"
            'Example: ["What is X?", "How does Y work?"]'
        )
        response = self.llm_client.complete(prompt, max_tokens=200)
        text = response.strip()
        # Try to extract JSON array
        try:
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, list) and len(parsed) > 0:
                    return [str(q).strip() for q in parsed if len(str(q).strip()) > 5]
        except Exception:
            pass
        return [query]


class FeedbackLoop:
    """Explicit feedback loop for embedding refinement.

    Users provide (query, node_id, relevant) feedback.
    Positive feedback nudges query→node embeddings closer.
    Supports persistence to disk.
    """

    def __init__(self, embedder, lr: float = 0.001, persist_path: Optional[str] = None):
        self.embedder = embedder
        self.lr = lr
        self.feedback_count = 0
        self.persist_path = persist_path
        self._buffer: List[Dict] = []

    def add_feedback(
        self,
        query: str,
        node_text: str,
        relevant: bool,
    ) -> bool:
        """Apply feedback to word embeddings."""
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

        direction = 1.0 if relevant else -1.0
        delta = (n_emb - q_emb) * direction * self.lr

        for t in q_tokens:
            if t in sif.word_embeddings:
                sif.word_embeddings[t] += delta / len(q_tokens)
                sif.word_embeddings[t] /= np.linalg.norm(sif.word_embeddings[t]) + 1e-8

        for t in n_tokens:
            if t in sif.word_embeddings:
                sif.word_embeddings[t] -= delta / len(n_tokens)
                sif.word_embeddings[t] /= np.linalg.norm(sif.word_embeddings[t]) + 1e-8

        self.feedback_count += 1
        self._buffer.append({
            "query": query,
            "node_text": node_text,
            "relevant": relevant,
            "timestamp": time.time(),
        })
        if self.persist_path and len(self._buffer) >= 10:
            self._flush()

        logger.info("FeedbackLoop: applied %s feedback (#%d)", "positive" if relevant else "negative", self.feedback_count)
        return True

    def _flush(self) -> None:
        """Write buffered feedback to disk."""
        if not self.persist_path:
            return
        try:
            existing = []
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.extend(self._buffer)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            self._buffer.clear()
        except Exception:
            logger.warning("FeedbackLoop: flush failed", exc_info=True)

    def load(self) -> None:
        """Load persisted feedback."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.feedback_count = len(data)
            logger.info("FeedbackLoop: loaded %d feedback records", self.feedback_count)
        except Exception:
            logger.warning("FeedbackLoop: load failed", exc_info=True)
