"""Self-Organizing Tokenizer + Embedding Field for RTMDK.

Replaces static external embeddings with a dynamic field that:
1. Builds vocabulary from bytes up to subtokens via field co-occurrence.
2. Learns embeddings via online contrastive Hebbian rules.
3. Synchronizes with SSM dynamics for smooth latent trajectories.

Architecture: token_dim != latent_dim supported via learnable projection.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from rtmdk.engines.ssm_dynamics import SSMDynamics

logger = logging.getLogger(__name__)


class CooccurrenceStore:
    """Bounded co-occurrence dictionary with periodic pruning.

    Keeps up to `max_size` entries. When the threshold is exceeded,
    drops the lowest-weight entries to free space.
    """

    def __init__(self, max_size: int = 100_000, prune_factor: float = 1.2):
        self.max_size = max_size
        self.prune_threshold = int(max_size * prune_factor)
        self._data: Dict[Tuple[int, int], float] = {}
        self._total_inserts = 0
        self._total_prunes = 0

    def __getitem__(self, key: Tuple[int, int]) -> float:
        return self._data.get(key, 0.0)

    def get(self, key: Tuple[int, int], default: float = 0.0) -> float:
        return self._data.get(key, default)

    def pop(self, key: Tuple[int, int], default=None):
        return self._data.pop(key, default)

    def __setitem__(self, key: Tuple[int, int], value: float):
        self._data[key] = value
        self._total_inserts += 1

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        return iter(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def prune_if_needed(self):
        """Remove lowest-weight entries if over threshold."""
        if len(self._data) <= self.prune_threshold:
            return
        # Sort by weight descending, keep top max_size
        sorted_items = sorted(
            self._data.items(),
            key=lambda kv: kv[1],
            reverse=True)
        kept = sorted_items[: self.max_size]
        dropped = len(sorted_items) - len(kept)
        self._data = dict(kept)
        self._total_prunes += dropped
        logger.info(
            f"CooccurrenceStore pruned: dropped {dropped} entries, kept {len(self._data)}")

    def get_stats(self) -> Dict[str, int]:
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "total_inserts": self._total_inserts,
            "total_prunes": self._total_prunes,
        }


class SOTokenizer:
    """Self-organizing tokenizer: bytes -> subtokens, driven by field co-occurrence.

    Supports token_dim != latent_dim via a learnable projection matrix.
    This allows high-capacity token representations (e.g., 256d) while
    keeping the field space lightweight (e.g., 64d).
    """

    def __init__(
        self,
        latent_dim: int,
        token_dim: Optional[int] = None,
        max_vocab: int = 4096,
        initial_byte_vocab: int = 256,
        seed: int = 42,
        subword_seed: bool = False,
        attention_pooling: bool = False,
        skipgram_window: int = 1,
        tokenization_mode: str = "byte",
        max_cooccurrence: int = 100_000,
        adaptive_lr: bool = True,
    ):
        self.latent_dim = latent_dim
        self.token_dim = token_dim or latent_dim
        self.adaptive_lr = adaptive_lr
        self.max_vocab = max_vocab
        self.initial_byte_vocab = initial_byte_vocab
        self.next_token_id = initial_byte_vocab
        self._rng = np.random.default_rng(seed)
        self.subword_seed = subword_seed
        self.attention_pooling = attention_pooling
        self.skipgram_window = max(1, skipgram_window)
        self.tokenization_mode = tokenization_mode
        self.max_cooccurrence = max_cooccurrence

        self.token_embeddings: Dict[int, np.ndarray] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.cooccurrence = CooccurrenceStore(max_size=max_cooccurrence)
        self.token_frequency: Dict[int, float] = defaultdict(float)
        self.token_idf: Dict[int, float] = {}

        # Word-level vocab
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._unk_token_id: Optional[int] = None
        self._pruned_words: Set[str] = set()

        # Learnable projection: token_dim -> latent_dim
        if self.token_dim == self.latent_dim:
            self.projection = np.eye(self.token_dim, dtype=np.float32)
        else:
            self.projection = self._rng.standard_normal(
                (self.token_dim, self.latent_dim)).astype(
                np.float32) * 0.1
            # Orthogonal initialization for better conditioning
            if self.token_dim >= self.latent_dim:
                q, _ = np.linalg.qr(self.projection)
                self.projection = q[:, : self.latent_dim].astype(np.float32)

        if self.tokenization_mode == "byte":
            self._init_byte_embeddings()
            if subword_seed:
                self._seed_subword_tokens()
        else:
            # Word mode: reserve IDs 0..255 for special / fallback
            for i in range(self.initial_byte_vocab):
                emb = self._rng.standard_normal(
                    self.token_dim).astype(
                    np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                self.token_embeddings[i] = emb

    def _init_byte_embeddings(self):
        for i in range(self.initial_byte_vocab):
            emb = self._rng.standard_normal(self.token_dim).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self.token_embeddings[i] = emb

    def _seed_subword_tokens(
            self,
            top_n_bigrams: int = 500,
            top_n_trigrams: int = 200):
        """Pre-seed vocabulary with common English byte patterns.
        These are learned from a small static corpus (no runtime dependency)."""
        # Common byte bigrams in English text (from frequency analysis)
        common_bigrams = [
            (101, 32),
            (116, 104),
            (105, 110),
            (115, 32),
            (32, 116),  # 'e ', 'th', 'in', 's ', ' t'
            (32, 97),
            (32, 105),
            (111, 110),
            (101, 114),
            (114, 101),  # ' a', ' i', 'on', 'er', 're'
            (97, 116),
            (101, 115),
            (32, 111),
            (114, 97),
            (110, 100),  # 'at', 'es', ' o', 'ra', 'nd'
            (116, 111),
            (101, 110),
            (97, 110),
            (116, 105),
            (32, 115),  # 'to', 'en', 'an', 'ti', ' s'
            (104, 97),
            (108, 121),
            (111, 117),
            (116, 101),
            (97, 114),  # 'ha', 'ly', 'ou', 'te', 'ar'
            (105, 115),
            (115, 116),
            (110, 103),
            (104, 101),
            (100, 101),  # 'is', 'st', 'ng', 'he', 'de'
            (116, 97),
            (32, 119),
            (105, 116),
            (101, 100),
            (101, 97),  # 'ta', ' w', 'it', 'ed', 'ea'
            (111, 114),
            (111, 111),
            (101, 101),
            (117, 114),
            (115, 101),  # 'or', 'oo', 'ee', 'ur', 'se'
            (32, 99),
            (32, 102),
            (32, 109),
            (32, 112),
            (32, 98),  # ' c', ' f', ' m', ' p', ' b'
            (32, 104),
            (32, 100),
            (32, 114),
            (32, 108),
            (32, 110),  # ' h', ' d', ' r', ' l', ' n'
            (105, 99),
            (97, 108),
            (97, 99),
            (111, 109),
            (97, 115),  # 'ic', 'al', 'ac', 'om', 'as'
            (105, 111),
            (117, 115),
            (115, 105),
            (114, 116),
            (108, 105),  # 'io', 'us', 'si', 'rt', 'li'
        ]
        common_trigrams = [
            (116, 104, 101),
            (105, 110, 103),
            (116, 105, 111),
            (110, 116, 104),  # 'the', 'ing', 'tio', 'nth'
            (97, 116, 105),
            (116, 104, 97),
            (115, 116, 104),
            (104, 97, 116),  # 'ati', 'tha', 'sth', 'hat'
            (101, 114, 101),
            (111, 117, 116),
            (102, 111, 114),
            (104, 101, 114),  # 'ere', 'out', 'for', 'her'
            (97, 116, 101),
            (119, 105, 116),
            (104, 32, 32),
            (105, 115, 32),
            (101, 115, 116),  # 'ate', 'wit', 'h  ', 'is ', 'est'
            (97, 110, 100),
            (115, 116, 97),
            (111, 110, 32),
            (101, 110, 116),  # 'and', 'sta', 'on ', 'ent'
            (104, 97, 116),
            (114, 101, 32),
            (97, 116, 32),
            (101, 114, 115),  # 'hat', 're ', 'at ', 'ers'
            (97, 114, 101),
            (116, 104, 32),
            (105, 110, 32),
            (101, 114, 97),  # 'are', 'th ', 'in ', 'era'
            (111, 110, 97),
            (115, 116, 32),
            (109, 101, 110),
            (97, 110, 32),  # 'ona', 'st ', 'men', 'an '
        ]
        for pair in common_bigrams[:top_n_bigrams]:
            if len(self.token_embeddings) >= self.max_vocab:
                break
            if all(p < self.initial_byte_vocab for p in pair):
                self._create_merge(pair)
        for triple in common_trigrams[:top_n_trigrams]:
            if len(self.token_embeddings) >= self.max_vocab:
                break
            # Merge first two, then merge result with third
            if all(t < self.initial_byte_vocab for t in triple[:2]):
                if triple[:2] not in self.merges:
                    self._create_merge(triple[:2])
                first = self.merges[triple[:2]]
                if (first,
                        triple[2]) not in self.merges and first in self.token_embeddings:
                    self._create_merge((first, triple[2]))

    def _create_merge(self, pair: Tuple[int, int]):
        """Create a merge rule without recording cooccurrence (for seeding)."""
        a, b = pair
        if a not in self.token_embeddings or b not in self.token_embeddings:
            return
        new_id = self.next_token_id
        self.next_token_id += 1
        new_emb = 0.5 * (self.token_embeddings[a] + self.token_embeddings[b])
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb /= norm
        self.token_embeddings[new_id] = new_emb.astype(np.float32)
        self.merges[pair] = new_id

    def bootstrap_from_teacher(
        self,
        texts: List[str],
        teacher_embed_fn,
        n_epochs: int = 30,
        lr: float = 0.05,
        fit_projection_only: bool = True,
    ):
        """Bootstrap SOT embeddings from a teacher model (e.g. SBERT).

        Two modes:
        - fit_projection_only=True (default): keeps token embeddings fixed,
          learns a projection matrix W such that mean(token_emb) @ W ≈ teacher_emb.
          This is more stable and preserves byte-level structure.
        - fit_projection_only=False: iteratively updates token embeddings toward
          teacher targets, then fine-tunes projection. May overfit.

        Args:
            texts: List of texts to use for bootstrap.
            teacher_embed_fn: Callable(text) -> np.ndarray (teacher embedding).
            n_epochs: Number of optimization epochs (only used when fit_projection_only=False).
            lr: Learning rate for token embedding updates (only used when fit_projection_only=False).
            fit_projection_only: If True, only fit the projection matrix via Ridge regression.
        """
        if not texts:
            return
        logger.info(
            f"SOT bootstrap: computing teacher embeddings for {len(texts)} texts...")
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
            logger.warning("SOT bootstrap: no valid teacher embeddings")
            return
        teacher_matrix = np.stack(teacher_embs).astype(np.float32)
        teacher_dim = teacher_matrix.shape[1]

        # Reduce teacher to latent_dim via PCA if needed
        if teacher_dim > self.latent_dim:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=self.latent_dim)
            teacher_reduced = pca.fit_transform(
                teacher_matrix).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(teacher_reduced, axis=1, keepdims=True)
            teacher_reduced /= np.maximum(norms, 1e-8)
        else:
            teacher_reduced = teacher_matrix.copy()
            # Pad with zeros if teacher_dim < latent_dim
            if teacher_dim < self.latent_dim:
                pad = np.zeros(
                    (len(teacher_reduced),
                     self.latent_dim - teacher_dim),
                    dtype=np.float32)
                teacher_reduced = np.concatenate(
                    [teacher_reduced, pad], axis=1)
            norms = np.linalg.norm(teacher_reduced, axis=1, keepdims=True)
            teacher_reduced /= np.maximum(norms, 1e-8)

        logger.info(
            f"SOT bootstrap: teacher shape {teacher_matrix.shape}, reduced to {teacher_reduced.shape}")

        if not fit_projection_only:
            # Build token → text indices mapping
            token_to_indices: Dict[int, List[int]] = defaultdict(list)
            for idx, text in enumerate(valid_texts):
                tokens = self.encode(text)
                for t in set(tokens):
                    token_to_indices[t].append(idx)

            # Iteratively update token embeddings toward teacher targets
            for epoch in range(n_epochs):
                total_delta = 0.0
                for token_id, indices in token_to_indices.items():
                    if token_id not in self.token_embeddings:
                        continue
                    target = np.mean(
                        teacher_reduced[indices],
                        axis=0).astype(
                        np.float32)
                    current = self.token_embeddings[token_id] @ self.projection
                    error = target - current
                    delta_token = lr * (error @ self.projection.T)
                    delta_norm = np.linalg.norm(delta_token)
                    if delta_norm > 1.0:
                        delta_token /= delta_norm
                    self.token_embeddings[token_id] += delta_token
                    norm = np.linalg.norm(self.token_embeddings[token_id])
                    if norm > 0:
                        self.token_embeddings[token_id] /= norm
                    total_delta += float(np.linalg.norm(delta_token))

                if epoch % 10 == 0:
                    logger.info(
                        f"SOT bootstrap epoch {epoch}: avg delta={total_delta / len(token_to_indices):.4f}")

        # Fit projection matrix via Ridge regression
        logger.info("SOT bootstrap: fitting projection matrix...")
        X = []
        Y = []
        for idx, text in enumerate(valid_texts):
            tokens = self.encode(text)
            if not tokens:
                continue
            pooled = np.mean([self.token_embeddings[t]
                             for t in tokens if t in self.token_embeddings], axis=0)
            X.append(pooled)
            Y.append(teacher_reduced[idx])
        if len(X) >= self.latent_dim:
            X = np.stack(X).astype(np.float32)
            Y = np.stack(Y).astype(np.float32)
            # Ridge: W = (X^T X + λI)^{-1} X^T Y
            lam = 1.0  # stronger regularization for stability
            XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
            try:
                W = np.linalg.solve(XtX, X.T @ Y).astype(np.float32)
                if W.shape == self.projection.shape:
                    self.projection = W
                    logger.info(
                        "SOT bootstrap: projection matrix updated via Ridge regression")
                else:
                    logger.warning(
                        f"SOT bootstrap: projection shape mismatch {W.shape} vs {self.projection.shape}")
            except np.linalg.LinAlgError:
                logger.warning(
                    "SOT bootstrap: Ridge regression failed, keeping original projection")

        logger.info("SOT bootstrap complete")

    def warm_start_from_corpus(self, corpus_texts: List[str]):
        """Pre-train token embeddings on a corpus using PMI-based initialization.

        This gives semantically meaningful starting points for byte embeddings,
        dramatically improving cold-start recall.
        """
        if not corpus_texts:
            return
        # Build byte co-occurrence statistics
        cooc: Dict[Tuple[int, int], float] = defaultdict(float)
        totals: Dict[int, float] = defaultdict(float)
        for text in corpus_texts:
            bytes_list = list(text.encode("utf-8"))
            for b in bytes_list:
                totals[b] += 1.0
            for i in range(len(bytes_list) - 1):
                a, b = bytes_list[i], bytes_list[i + 1]
                cooc[(a, b)] += 1.0
                cooc[(b, a)] += 1.0  # symmetric

        n_total = sum(totals.values())
        if n_total == 0:
            return

        # Compute PMI and use it to nudge embeddings
        for (a, b), count in cooc.items():
            if a >= self.initial_byte_vocab or b >= self.initial_byte_vocab:
                continue
            p_a = totals[a] / n_total
            p_b = totals[b] / n_total
            p_ab = count / n_total
            if p_a > 0 and p_b > 0 and p_ab > 0:
                pmi = math.log(p_ab / (p_a * p_b) + 1e-10)
                if pmi > 0:
                    # Pull embeddings of co-occurring bytes closer
                    if a in self.token_embeddings and b in self.token_embeddings:
                        direction = self.token_embeddings[b] - \
                            self.token_embeddings[a]
                        self.token_embeddings[a] += 0.05 * pmi * direction
                        self.token_embeddings[b] += 0.05 * pmi * (-direction)

        # Renormalize
        for i in range(self.initial_byte_vocab):
            if i in self.token_embeddings:
                norm = np.linalg.norm(self.token_embeddings[i])
                if norm > 0:
                    self.token_embeddings[i] /= norm

        # Pre-compute IDF weights for attention pooling
        for b, count in totals.items():
            if b < self.initial_byte_vocab:
                idf = math.log(len(corpus_texts) / (count + 1.0) + 1.0)
                self.token_idf[b] = idf
        logger.info(
            f"SOT warm-start: processed {len(corpus_texts)} texts, "
            f"{len(cooc)} co-occurrence pairs, IDF computed for {len(self.token_idf)} bytes")

    def _word_tokenize(self, text: str) -> List[str]:
        """Unicode-aware word tokenization.

        Supports Latin, Cyrillic, Arabic, Devanagari, and other scripts.
        CJK (Chinese, Japanese, Korean) characters are tokenized individually
        since they are not whitespace-delimited.
        """
        import unicodedata

        text = text.lower()
        tokens = []
        current: List[str] = []

        for ch in text:
            # CJK characters are individual semantic units regardless of
            # category
            is_cjk = (
                "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
                or "\u3040" <= ch <= "\u309f"  # Hiragana
                or "\u30a0" <= ch <= "\u30ff"  # Katakana
                or "\uac00" <= ch <= "\ud7af"  # Hangul Syllables
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

    def _get_word_id(self, word: str) -> int:
        """Map a word to its token ID, expanding vocab if under limit."""
        if word in self.word_to_id:
            return self.word_to_id[word]
        if word in self._pruned_words:
            return self._unk_token_id or 0
        if self.next_token_id < self.max_vocab:
            wid = self.next_token_id
            self.next_token_id += 1
            self.word_to_id[word] = wid
            self.id_to_word[wid] = word
            # Initialize embedding
            emb = self._rng.standard_normal(self.token_dim).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self.token_embeddings[wid] = emb
            return wid
        # Fallback to unk
        if self._unk_token_id is None:
            self._unk_token_id = self.next_token_id
            self.next_token_id += 1
            emb = self._rng.standard_normal(self.token_dim).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self.token_embeddings[self._unk_token_id] = emb
        return self._unk_token_id

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []
        if self.tokenization_mode == "word":
            words = self._word_tokenize(text)
            return [self._get_word_id(w) for w in words]
        # Byte mode (default)
        raw_bytes = list(text.encode("utf-8"))
        if not self.merges:
            return raw_bytes
        tokens = raw_bytes[:]
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i],
                                            tokens[i + 1]) in self.merges:
                    new_tokens.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 2
                    changed = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to a string."""
        if not tokens:
            return ""
        if self.tokenization_mode == "word":
            words = []
            for tid in tokens:
                if tid in self.id_to_word:
                    words.append(self.id_to_word[tid])
                elif tid == self._unk_token_id:
                    words.append("<unk>")
                elif tid < self.initial_byte_vocab:
                    words.append(f"<byte_{tid}>")
                else:
                    words.append(f"<id_{tid}>")
            return " ".join(words)
        # Byte mode
        byte_seq = bytearray()
        stack = list(tokens)
        while stack:
            tid = stack.pop(0)
            if tid < self.initial_byte_vocab:
                byte_seq.append(tid)
            else:
                found = False
                for (a, b), merged_id in self.merges.items():
                    if merged_id == tid:
                        stack.insert(0, b)
                        stack.insert(0, a)
                        found = True
                        break
                if not found:
                    logger.warning(
                        f"SOT decode: token {tid} has no merge rule")
                    byte_seq.append(tid % self.initial_byte_vocab)
        return byte_seq.decode("utf-8", errors="replace")

    def embed(self, tokens: List[int]) -> np.ndarray:
        """Pool token embeddings and project to latent_dim.
        Supports attention-weighted pooling if attention_pooling=True."""
        if not tokens:
            return np.zeros(self.latent_dim, dtype=np.float32)
        vecs = []
        weights = []
        for i, t in enumerate(tokens):
            if t not in self.token_embeddings:
                continue
            vecs.append(self.token_embeddings[t])
            if self.attention_pooling:
                # IDF weight: rare tokens are more informative
                idf = self.token_idf.get(t, 1.0)
                # Position weight: first and last tokens often more important
                pos_weight = 1.0 + 0.5 * \
                    (1.0 if i == 0 or i == len(tokens) - 1 else 0.0)
                weights.append(idf * pos_weight)
            else:
                weights.append(1.0)

        if not vecs:
            return np.zeros(self.latent_dim, dtype=np.float32)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum() + 1e-8
        pooled = np.average(vecs, axis=0, weights=weights).astype(np.float32)
        latent = pooled @ self.projection
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent /= norm
        return latent.astype(np.float32)

    def record_cooccurrence(self, tokens: List[int], weight: float = 1.0):
        """Record token co-occurrences for merge proposals.
        Uses skip-gram window if skipgram_window > 1."""
        if len(tokens) < 2:
            return
        window = self.skipgram_window
        for i in range(len(tokens)):
            self.token_frequency[tokens[i]] += weight
            for j in range(i + 1, min(i + window + 1, len(tokens))):
                a, b = tokens[i], tokens[j]
                dist = j - i
                self.cooccurrence[(a, b)] += weight / dist
                if a != b:
                    self.cooccurrence[(b, a)] += weight / dist

        self.cooccurrence.prune_if_needed()

        # Update IDF cache if attention pooling enabled
        if self.attention_pooling and self.token_frequency:
            n_docs = sum(1 for _ in self.token_frequency.values())
            for tid, freq in self.token_frequency.items():
                self.token_idf[tid] = math.log(
                    max(n_docs, 1) / (freq + 1.0) + 1.0)

    def propose_merges(self, n: int) -> List[Tuple[int, int]]:
        """Return top-N merge candidates by co-occurrence score."""
        if not self.cooccurrence:
            return []
        sorted_pairs = sorted(
            self.cooccurrence.items(),
            key=lambda kv: kv[1],
            reverse=True)
        return [pair for pair, _ in sorted_pairs[:n]]

    def merge(self, pair: Tuple[int, int]):
        """Execute a merge: create new token embedding as weighted average."""
        if len(self.token_embeddings) >= self.max_vocab:
            raise RuntimeError(
                f"Max vocab size {self.max_vocab} reached, cannot merge")
        a, b = pair
        if a not in self.token_embeddings or b not in self.token_embeddings:
            raise ValueError(f"Invalid pair {pair}: missing embeddings")
        weight = self.cooccurrence.get(pair, 1.0)
        new_id = self.next_token_id
        self.next_token_id += 1
        wa = weight
        wb = weight
        new_emb = (
            wa * self.token_embeddings[a] + wb * self.token_embeddings[b]) / (wa + wb)
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb /= norm
        self.token_embeddings[new_id] = new_emb.astype(np.float32)
        self.merges[pair] = new_id
        logger.debug(
            f"SOT merge: {pair} -> {new_id} (vocab={len(self.token_embeddings)})")

    def update_projection(
        self,
        positive_pairs: List[Tuple[int, int]],
        negative_pairs: List[Tuple[int, int]] = None,
        lr: float = 0.001,
    ):
        """Update projection matrix via contrastive Hebbian rule.

        Pulls projected positive pairs closer, pushes negatives apart.
        """
        if self.token_dim == self.latent_dim:
            return  # Identity projection, nothing to learn
        negative_pairs = negative_pairs or []
        delta = np.zeros_like(self.projection)
        for i, j in positive_pairs:
            if i not in self.token_embeddings or j not in self.token_embeddings:
                continue
            t_i = self.token_embeddings[i]  # (token_dim,)
            t_j = self.token_embeddings[j]  # (token_dim,)
            p_i = t_i @ self.projection  # (latent_dim,)
            p_j = t_j @ self.projection  # (latent_dim,)
            error = p_j - p_i
            # Gradient: d(loss)/dW = t_i^T * error
            delta += lr * np.outer(t_i, error)
        for i, k in negative_pairs:
            if i not in self.token_embeddings or k not in self.token_embeddings:
                continue
            t_i = self.token_embeddings[i]
            t_k = self.token_embeddings[k]
            p_i = t_i @ self.projection
            p_k = t_k @ self.projection
            error = p_k - p_i
            delta -= lr * 0.1 * np.outer(t_i, error)
        self.projection += delta
        # Normalize columns to prevent explosion
        norms = np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.projection /= np.maximum(norms, 1e-8)

    def contrastive_step(
        self,
        query_text: str,
        positive_text: str,
        negative_texts,
        lr: float = 0.01,
        adaptive_lr: Optional[bool] = None,
    ):
        """Online contrastive learning step for SOT token embeddings.

        Pulls query tokens toward the mean positive embedding and pushes
        them away from the mean negative embeddings. Also updates positive
        tokens toward the mean query embedding for symmetry.

        Args:
            adaptive_lr: If True, scales lr by sqrt(token_dim / latent_dim)
                so that the effective update strength stays constant across
                different token dimensionalities. Defaults to self.adaptive_lr.
        """

        if adaptive_lr is None:
            adaptive_lr = self.adaptive_lr

        if isinstance(negative_texts, str):
            negative_texts = [negative_texts]

        effective_lr = lr
        if adaptive_lr and self.token_dim > self.latent_dim:
            effective_lr = lr * np.sqrt(self.token_dim / self.latent_dim)

        q_tokens = self.encode(query_text)
        p_tokens = self.encode(positive_text)
        if not q_tokens or not p_tokens:
            return

        q_ids = [t for t in q_tokens if t in self.token_embeddings]
        p_ids = [t for t in p_tokens if t in self.token_embeddings]
        if not q_ids or not p_ids:
            return

        pos_mean = np.mean([self.token_embeddings[t] for t in p_ids], axis=0)

        neg_means = []
        for neg_text in negative_texts:
            n_tokens = self.encode(neg_text)
            n_ids = [t for t in n_tokens if t in self.token_embeddings]
            if n_ids:
                neg_means.append(
                    np.mean([self.token_embeddings[t] for t in n_ids], axis=0))

        # Update query tokens
        for tid in q_ids:
            emb = self.token_embeddings[tid]
            delta = effective_lr * (pos_mean - emb)
            for neg_mean in neg_means:
                delta -= effective_lr * 0.1 * (neg_mean - emb)
            emb = emb + delta
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.token_embeddings[tid] = emb.astype(np.float32)

        # Update positive tokens toward query mean
        q_mean = np.mean([self.token_embeddings[t] for t in q_ids], axis=0)
        for tid in p_ids:
            emb = self.token_embeddings[tid]
            delta = effective_lr * 0.5 * (q_mean - emb)
            emb = emb + delta
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.token_embeddings[tid] = emb.astype(np.float32)

    def prune_vocab(self, min_freq: float = 2.0):
        """Remove rare tokens from vocabulary (word mode only).

        Tokens with frequency < min_freq are removed and mapped to unk.
        This reduces memory footprint and speeds up retrieval.
        """
        if self.tokenization_mode != "word":
            logger.warning("prune_vocab only meaningful in word mode")
            return
        to_remove = [tid for tid, freq in self.token_frequency.items(
        ) if freq < min_freq and tid >= self.initial_byte_vocab]
        if not to_remove:
            return
        # Ensure unk token exists
        if self._unk_token_id is None:
            self._unk_token_id = self.next_token_id
            self.next_token_id += 1
            emb = self._rng.standard_normal(self.token_dim).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self.token_embeddings[self._unk_token_id] = emb
        for tid in to_remove:
            # Remove from embeddings
            self.token_embeddings.pop(tid, None)
            # Remove from word maps
            word = self.id_to_word.pop(tid, None)
            if word:
                self.word_to_id.pop(word, None)
                self._pruned_words.add(word)
            # Remove from frequency and idf
            self.token_frequency.pop(tid, None)
            self.token_idf.pop(tid, None)
        # Clean cooccurrence pairs involving removed tokens
        keys_to_remove = [k for k in self.cooccurrence.keys(
        ) if k[0] in to_remove or k[1] in to_remove]
        for k in keys_to_remove:
            self.cooccurrence.pop(k)
        logger.info(
            f"prune_vocab: removed {len(to_remove)} rare tokens, vocab={len(self.token_embeddings)}")

    def get_state(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "token_dim": self.token_dim,
            "max_vocab": self.max_vocab,
            "initial_byte_vocab": self.initial_byte_vocab,
            "next_token_id": self.next_token_id,
            "token_embeddings": {k: v.tolist() for k, v in self.token_embeddings.items()},
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "cooccurrence": {f"{a},{b}": v for (a, b), v in self.cooccurrence.items()},
            "projection": self.projection.tolist(),
            "tokenization_mode": self.tokenization_mode,
            "max_cooccurrence": self.max_cooccurrence,
            "skipgram_window": self.skipgram_window,
            "attention_pooling": self.attention_pooling,
            "subword_seed": self.subword_seed,
            "word_to_id": self.word_to_id,
            "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
            "unk_token_id": self._unk_token_id,
            "pruned_words": list(self._pruned_words),
            "token_frequency": dict(self.token_frequency),
            "token_idf": dict(self.token_idf),
        }

    def load_state(self, state: dict):
        self.latent_dim = state.get("latent_dim", self.latent_dim)
        self.token_dim = state.get("token_dim", self.token_dim)
        self.max_vocab = state.get("max_vocab", self.max_vocab)
        self.initial_byte_vocab = state.get(
            "initial_byte_vocab", self.initial_byte_vocab)
        self.next_token_id = state.get(
            "next_token_id", self.initial_byte_vocab)
        self.tokenization_mode = state.get(
            "tokenization_mode", self.tokenization_mode)
        self.max_cooccurrence = state.get(
            "max_cooccurrence", self.max_cooccurrence)
        self.skipgram_window = state.get(
            "skipgram_window", self.skipgram_window)
        self.attention_pooling = state.get(
            "attention_pooling", self.attention_pooling)
        self.subword_seed = state.get("subword_seed", self.subword_seed)
        self.token_embeddings = {
            int(k): np.array(
                v, dtype=np.float32) for k, v in state.get(
                "token_embeddings", {}).items()}
        self.merges = {}
        for k, v in state.get("merges", {}).items():
            a_str, b_str = k.split(",")
            self.merges[(int(a_str), int(b_str))] = int(v)
        self.cooccurrence = CooccurrenceStore(max_size=self.max_cooccurrence)
        for k, v in state.get("cooccurrence", {}).items():
            a_str, b_str = k.split(",")
            self.cooccurrence[(int(a_str), int(b_str))] = float(v)
        if "projection" in state:
            self.projection = np.array(state["projection"], dtype=np.float32)
            assert self.projection.shape == (
                self.token_dim,
                self.latent_dim,
            ), f"Projection shape mismatch: {self.projection.shape} vs ({self.token_dim}, {self.latent_dim})"
        self.word_to_id = state.get("word_to_id", {})
        self.id_to_word = {
            int(k): v for k,
            v in state.get(
                "id_to_word",
                {}).items()}
        self._unk_token_id = state.get("unk_token_id", None)
        self._pruned_words = set(state.get("pruned_words", []))
        self.token_frequency = defaultdict(
            float, state.get("token_frequency", {}))
        self.token_idf = state.get("token_idf", {})


class ContrastiveHebbian:
    """Online contrastive Hebbian learning for embeddings."""

    def __init__(
            self,
            lr: float = 0.01,
            neg_ratio: float = 0.2,
            temperature: float = 0.1):
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.temperature = temperature

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def update(
        self,
        embeddings: Dict[int, np.ndarray],
        positives: List[int],
        negatives: List[int],
    ):
        """Update token embeddings in-place via contrastive Hebbian rule.

        Positives pull each other closer; negatives are pushed away from positives.
        Negatives do NOT get pulled toward positives.
        """
        if len(positives) < 2 and not negatives:
            return
        for i in positives:
            if i not in embeddings:
                continue
            delta = np.zeros_like(embeddings[i])
            for j in positives:
                if i == j or j not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[j])
                delta += self.lr * sim * (embeddings[j] - embeddings[i])
            for k in negatives:
                if k not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[k])
                delta -= self.lr * 0.1 * sim * (embeddings[k] - embeddings[i])
            embeddings[i] += delta
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        for k in negatives:
            if k not in embeddings:
                continue
            delta = np.zeros_like(embeddings[k])
            for i in positives:
                if i not in embeddings:
                    continue
                sim = self._cosine_sim(embeddings[k], embeddings[i])
                delta -= self.lr * 0.1 * sim * (embeddings[i] - embeddings[k])
            embeddings[k] += delta
            norm = np.linalg.norm(embeddings[k])
            if norm > 0:
                embeddings[k] /= norm

    def update_with_hard_negatives(
        self,
        embeddings: Dict[int, np.ndarray],
        positives: List[int],
        all_candidates: List[int],
        n_negatives: int = 5,
    ):
        """Contrastive update using hard negatives (closest non-positive embeddings).

        This provides stronger gradient signal than random negatives.
        """
        if len(positives) < 2:
            return
        positive_set = set(positives)
        candidates = [
            c for c in all_candidates if c not in positive_set and c in embeddings]

        if not candidates:
            return

        # For each positive, find hardest negatives (highest similarity among
        # non-positives)
        hard_negatives = set()
        for i in positives:
            if i not in embeddings:
                continue
            sims = []
            for c in candidates:
                sim = self._cosine_sim(embeddings[i], embeddings[c])
                sims.append((sim, c))
            sims.sort(reverse=True)
            for _, c in sims[:n_negatives]:
                hard_negatives.add(c)

        self.update(embeddings, positives, list(hard_negatives))

    def field_update(
        self,
        node_embeddings: np.ndarray,
        positives: List[int],
        negatives: List[int],
    ):
        """Update node latent positions in-place (node_embeddings is (N, latent_dim)).

        Positives pull each other closer; negatives are pushed away from positives.
        """
        if node_embeddings.ndim != 2:
            return
        n_nodes = node_embeddings.shape[0]
        if n_nodes == 0:
            return
        valid_pos = [p for p in positives if 0 <= p < n_nodes]
        valid_neg = [n for n in negatives if 0 <= n < n_nodes]
        if len(valid_pos) < 2 and not valid_neg:
            return
        for i in valid_pos:
            delta = np.zeros_like(node_embeddings[i])
            for j in valid_pos:
                if i == j:
                    continue
                sim = self._cosine_sim(node_embeddings[i], node_embeddings[j])
                delta += self.lr * sim * \
                    (node_embeddings[j] - node_embeddings[i])
            for k in valid_neg:
                sim = self._cosine_sim(node_embeddings[i], node_embeddings[k])
                delta -= self.lr * 0.1 * sim * \
                    (node_embeddings[k] - node_embeddings[i])
            node_embeddings[i] += delta
            norm = np.linalg.norm(node_embeddings[i])
            if norm > 0:
                node_embeddings[i] /= norm
        for k in valid_neg:
            delta = np.zeros_like(node_embeddings[k])
            for i in valid_pos:
                sim = self._cosine_sim(node_embeddings[k], node_embeddings[i])
                delta -= self.lr * 0.1 * sim * \
                    (node_embeddings[i] - node_embeddings[k])
            node_embeddings[k] += delta
            norm = np.linalg.norm(node_embeddings[k])
            if norm > 0:
                node_embeddings[k] /= norm


class EmbeddingFieldSSM:
    """Bridges SSMDynamics with the embedding field for smooth trajectories."""

    def __init__(
            self,
            latent_dim: int,
            tokenizer: SOTokenizer,
            diagonal: bool = True):
        self.latent_dim = latent_dim
        self.tokenizer = tokenizer
        self.ssm = SSMDynamics(
            state_dim=latent_dim,
            input_dim=latent_dim,
            output_dim=latent_dim,
            n_nodes=1,
            dt=0.1,
            learnable=False,
            diagonal=diagonal,
        )

    def step(
            self,
            token_ids: List[int],
            field_state: np.ndarray) -> np.ndarray:
        """Compute momentum from SSM given current token sequence and field state."""
        if field_state.ndim != 1 or field_state.shape[0] != self.latent_dim:
            field_state = np.zeros(self.latent_dim, dtype=np.float32)
        u = self.tokenizer.embed(token_ids)
        if u.ndim != 1:
            u = u.reshape(-1)
        h = field_state.reshape(1, -1)
        u = u.reshape(1, -1)
        _, y = self.ssm.step(h, u)
        if y.ndim != 1:
            y = y.reshape(-1)
        return y.astype(np.float32)

    def sync_embeddings(self, token_ids: List[int], momentum: np.ndarray):
        """Add SSM momentum to token embeddings via projection."""
        if momentum.ndim != 1 or momentum.shape[0] != self.latent_dim:
            return
        for tid in token_ids:
            if tid not in self.tokenizer.token_embeddings:
                continue
            # Apply momentum in latent space, backpropagate through projection
            # delta_token = momentum @ projection.T
            delta_token = momentum @ self.tokenizer.projection.T
            self.tokenizer.token_embeddings[tid] += delta_token * 0.01
            norm = np.linalg.norm(self.tokenizer.token_embeddings[tid])
            if norm > 0:
                self.tokenizer.token_embeddings[tid] /= norm
