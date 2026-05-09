"""ProjectionManager — encapsulates projection + SOT tokenizer logic."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from rtmdk.support.projection import IncPCAProjection, IdentityProjection

logger = logging.getLogger(__name__)


class ProjectionManager:
    """Manages embedding → latent projection and optional SOT tokenizer.

    Responsibilities:
    - Identity / Incremental-PCA / random projection
    - Hyperbolic clamp into Poincaré ball
    - Self-Organizing Tokenizer (SOT) lifecycle
    - SOT contrastive learning & retrieval feedback
    """

    def __init__(
        self,
        cfg: Any,
        projection_matrix: Optional[NDArray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg
        self._rng = rng or np.random.default_rng(cfg.seed)

        # --- Projection learner / raw matrix ---
        if cfg.projection_mode == "identity":
            self.projection_learner = IdentityProjection(
                cfg.embedding_dim, cfg.latent_dim)
            self._raw_projection = None
        elif cfg.learn_projection:
            self.projection_learner = IncPCAProjection(
                cfg.embedding_dim,
                cfg.pca_n_components or cfg.latent_dim,
                cfg.projection_lr,
                cfg.projection_update_freq,
                cfg.l2_regularization,
            )
            if projection_matrix is not None:
                self.projection_learner.set_matrix(projection_matrix)
            self._raw_projection = None
        else:
            self.projection_learner = None
            if projection_matrix is not None:
                self._raw_projection = projection_matrix.astype(np.float32)
            else:
                self._raw_projection = (
                    self._rng.standard_normal(
                        (cfg.embedding_dim, cfg.latent_dim)
                    ).astype(np.float32)
                    * 0.1
                )

        # --- SOT ---
        self.sot_tokenizer: Optional[Any] = None
        self.sot_hebbian: Optional[Any] = None
        self._sot_field_ema: Optional[NDArray] = None
        if cfg.sot_enabled:
            from rtmdk.memory.self_organizing_field import (
                SOTokenizer, ContrastiveHebbian
            )
            token_dim = cfg.sot_token_dim or cfg.latent_dim
            self.sot_tokenizer = SOTokenizer(
                latent_dim=cfg.latent_dim,
                token_dim=token_dim,
                max_vocab=cfg.sot_max_vocab,
                seed=cfg.seed,
                subword_seed=cfg.sot_subword_seed,
                attention_pooling=cfg.sot_attention_pooling,
                skipgram_window=cfg.sot_skipgram_window,
                tokenization_mode=cfg.sot_tokenization_mode,
                max_cooccurrence=cfg.sot_max_cooccurrence,
                adaptive_lr=cfg.sot_adaptive_lr,
            )
            self._maybe_warm_start_sot()
            self._maybe_bootstrap_sot()
            self.sot_hebbian = ContrastiveHebbian(lr=cfg.sot_contrastive_lr)
            self._sot_field_ema = np.zeros(cfg.latent_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(self, embedding: NDArray) -> NDArray:
        """Project a single embedding into latent space."""
        if len(embedding) == self.cfg.latent_dim:
            latent = embedding.astype(np.float32)
        elif self.projection_learner:
            latent = self.projection_learner.project(embedding)
        else:
            emb = embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            latent = (emb @ self._raw_projection).astype(np.float32)
        if self.cfg.hyperbolic:
            latent = self._hyperbolic_clamp(latent)
        return latent

    def project_batch(self, embeddings: NDArray) -> NDArray:
        """Vectorized projection for batch inserts."""
        n, d = embeddings.shape
        if d == self.cfg.latent_dim:
            latents = embeddings.astype(np.float32)
        elif self.projection_learner:
            # Sequential fallback for stateful projection learner
            latents = np.array(
                [self.projection_learner.project(e) for e in embeddings],
                dtype=np.float32,
            )
        else:
            embs = embeddings.astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.maximum(norms, 1e-8)
            latents = embs @ self._raw_projection
        if self.cfg.hyperbolic:
            latents = self._hyperbolic_clamp_batch(latents)
        return latents

    def update_projection(self, embedding: NDArray) -> NDArray:
        """Online update (IncPCA) + project."""
        if self.projection_learner is None:
            return self.project(embedding)
        latent = self.projection_learner.update(embedding)
        if self.cfg.hyperbolic:
            latent = self._hyperbolic_clamp(latent)
        return latent

    def fit_projection(self, corpus_embeddings: NDArray) -> None:
        """Batch-fit projection learner."""
        if self.projection_learner is None:
            return
        if hasattr(self.projection_learner, "fit"):
            self.projection_learner.fit(corpus_embeddings)
            logger.info(
                f"Projection fitted on {corpus_embeddings.shape[0]} samples")
        else:
            logger.warning(
                "projection_learner does not support fit() — skipping corpus fit")

    # ------------------------------------------------------------------
    # SOT helpers
    # ------------------------------------------------------------------
    def _maybe_warm_start_sot(self) -> None:
        cfg = self.cfg
        if not cfg.sot_warm_start_corpus:
            return
        try:
            import json
            with open(cfg.sot_warm_start_corpus, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts: List[str] = []
            if isinstance(data, dict) and "records" in data:
                texts = [
                    r.get("context", "")
                    + " "
                    + r.get("answer", "")
                    + " "
                    + r.get("query", "")
                    for r in data["records"]
                ]
            elif isinstance(data, list):
                texts = [str(item) for item in data]
            self.sot_tokenizer.warm_start_from_corpus(texts)
        except Exception as e:
            logger.warning(f"SOT warm-start failed: {e}")

    def _maybe_bootstrap_sot(self) -> None:
        cfg = self.cfg
        # External bootstrap projection
        if cfg.sot_bootstrap_projection:
            try:
                from rtmdk.memory.bootstrap_sbert import load_bootstrap
                load_bootstrap(cfg.sot_bootstrap_projection, self.sot_tokenizer)
            except Exception as e:
                logger.warning(f"SOT bootstrap projection load failed: {e}")
        # FastText auto-bootstrap
        if cfg.sot_bootstrap_fasttext_model and cfg.sot_bootstrap_corpus:
            try:
                import json
                with open(cfg.sot_bootstrap_corpus, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts: List[str] = []
                if isinstance(data, dict) and "records" in data:
                    texts = [
                        r.get("context", "") + " " + r.get("answer", "")
                        for r in data["records"]
                    ]
                elif isinstance(data, list):
                    texts = [str(item) for item in data]
                from rtmdk.memory.bootstrap_fasttext import run_bootstrap
                run_bootstrap(
                    self.sot_tokenizer,
                    texts=texts,
                    model_path=cfg.sot_bootstrap_fasttext_model,
                )
            except Exception as e:
                logger.warning(f"SOT FastText auto-bootstrap failed: {e}")
        # SBERT auto-bootstrap from corpus
        elif cfg.sot_bootstrap_corpus and not cfg.sot_bootstrap_projection:
            try:
                import json
                with open(cfg.sot_bootstrap_corpus, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts: List[str] = []
                if isinstance(data, dict) and "records" in data:
                    texts = [
                        r.get("context", "") + " " + r.get("answer", "")
                        for r in data["records"]
                    ]
                elif isinstance(data, list):
                    texts = [str(item) for item in data]
                from sentence_transformers import SentenceTransformer
                teacher = SentenceTransformer(cfg.sot_bootstrap_model)
                self.sot_tokenizer.bootstrap_from_teacher(
                    texts,
                    lambda t: teacher.encode(t, show_progress_bar=False),
                    fit_projection_only=False,
                    n_epochs=10,
                    lr=0.05,
                )
            except Exception as e:
                logger.warning(f"SOT auto-bootstrap failed: {e}")

    # ------------------------------------------------------------------
    # SOT public API
    # ------------------------------------------------------------------
    def sot_bootstrap(
        self,
        texts: List[str],
        teacher_model: str = "all-MiniLM-L6-v2",
        fit_projection_only: bool = True,
        n_epochs: int = 30,
    ) -> None:
        if not self.sot_tokenizer:
            raise RuntimeError("SOT not enabled in config")
        try:
            from sentence_transformers import SentenceTransformer
            teacher = SentenceTransformer(teacher_model)
            logger.info(f"SOT bootstrap: loading teacher model {teacher_model}")
            self.sot_tokenizer.bootstrap_from_teacher(
                texts,
                lambda t: teacher.encode(t, show_progress_bar=False),
                fit_projection_only=fit_projection_only,
                n_epochs=n_epochs,
            )
        except ImportError:
            logger.error("sentence-transformers not installed, cannot bootstrap SOT")
            raise
        except Exception as e:
            logger.error(f"SOT bootstrap failed: {e}")
            raise

    def sot_contrastive_step(
        self,
        query_text: str,
        positive_text: str,
        negative_texts: Optional[List[str]] = None,
        lr: float = 0.01,
    ) -> None:
        if not self.sot_tokenizer:
            raise RuntimeError("SOT not enabled in config")
        if negative_texts is None:
            negative_texts = []
        self.sot_tokenizer.contrastive_step(
            query_text,
            positive_text,
            negative_texts,
            lr=lr,
            adaptive_lr=self.cfg.sot_adaptive_lr,
        )

    def sot_retrieval_feedback(
        self,
        query_latent: np.ndarray,
        results: List[Tuple[str, float, Any]],
        negatives_per_query: int = 5,
    ) -> None:
        """Update SOT embeddings based on retrieval results."""
        if not self.sot_tokenizer or not self.sot_hebbian:
            return
        if not results:
            return
        top_nid, top_score, top_node = results[0]
        bottom_nid, bottom_score, bottom_node = (
            results[-1] if len(results) > 1 else (None, 0.0, None)
        )
        if top_score < 0.1:
            return
        top_text = top_node.content.get("text", "")
        bottom_text = bottom_node.content.get("text", "") if bottom_node else ""
        top_tokens = self.sot_tokenizer.encode(top_text)
        bottom_tokens = self.sot_tokenizer.encode(bottom_text) if bottom_text else []
        if top_tokens and len(top_tokens) > 1:
            self.sot_hebbian.update(
                self.sot_tokenizer.token_embeddings,
                top_tokens,
                [t for t in bottom_tokens if t not in top_tokens][:negatives_per_query],
            )
        # Update projection matrix if token_dim != latent_dim
        if self.sot_tokenizer.token_dim != self.sot_tokenizer.latent_dim and top_tokens:
            top_emb = self.sot_tokenizer.embed(top_tokens)
            error = query_latent - top_emb
            error_norm = np.linalg.norm(error)
            if error_norm > 1.0:
                error = error / error_norm
            for t in top_tokens:
                if t in self.sot_tokenizer.token_embeddings:
                    token_vec = self.sot_tokenizer.token_embeddings[t]
                    delta = 0.001 * np.outer(token_vec, error)
                    self.sot_tokenizer.projection += delta
            if np.isnan(self.sot_tokenizer.projection).any() or np.isinf(
                self.sot_tokenizer.projection
            ).any():
                logger.warning("SOT projection NaN/Inf detected, skipping update")
                return
            norms = np.linalg.norm(
                self.sot_tokenizer.projection, axis=0, keepdims=True
            )
            self.sot_tokenizer.projection /= np.maximum(norms, 1e-8)

    def sot_encode(self, text: str) -> List[int]:
        if self.sot_tokenizer:
            return self.sot_tokenizer.encode(text)
        return []

    def sot_embed(self, tokens: List[int]) -> Optional[np.ndarray]:
        if self.sot_tokenizer:
            return self.sot_tokenizer.embed(tokens)
        return None

    def sot_record_cooccurrence(self, tokens: List[int]) -> None:
        if self.sot_tokenizer:
            self.sot_tokenizer.record_cooccurrence(tokens)

    def sot_propose_merges(self, k: int = 5) -> List[Tuple[Any, Any]]:
        if self.sot_tokenizer:
            return self.sot_tokenizer.propose_merges(k)
        return []

    def sot_merge(self, pair: Tuple[Any, Any]) -> None:
        if self.sot_tokenizer:
            self.sot_tokenizer.merge(pair)

    def sot_contrastive_hebbian_field_update(
        self,
        positions: np.ndarray,
        pos_indices: List[int],
        neg_indices: List[int],
    ) -> None:
        if self.sot_hebbian:
            self.sot_hebbian.field_update(positions, pos_indices, neg_indices)

    def sot_contrastive_hebbian_token_update(
        self,
        tokens: List[int],
        vocab_ids: List[Any],
        negatives_per_query: int = 5,
        hard_negatives: bool = False,
    ) -> None:
        if not self.sot_hebbian or not self.sot_tokenizer:
            return
        n_neg = min(negatives_per_query, len(vocab_ids) - len(tokens))
        if hard_negatives and n_neg > 0:
            self.sot_hebbian.update_with_hard_negatives(
                self.sot_tokenizer.token_embeddings,
                tokens,
                vocab_ids,
                n_negatives=n_neg,
            )
        else:
            negatives = []
            if n_neg > 0:
                available = [v for v in vocab_ids if v not in tokens]
                if available:
                    negatives = self._rng.choice(
                        available, size=min(n_neg, len(available)), replace=False
                    ).tolist()
            self.sot_hebbian.update(
                self.sot_tokenizer.token_embeddings, tokens, negatives
            )

    def sot_query_latent(self, text: str) -> Optional[np.ndarray]:
        """Return query latent from SOT tokenizer, or None if disabled."""
        if self.sot_tokenizer and self.cfg.sot_use_for_query:
            tokens = self.sot_tokenizer.encode(text)
            return self.sot_tokenizer.embed(tokens)
        return None

    # ------------------------------------------------------------------
    # Hyperbolic helpers
    # ------------------------------------------------------------------
    def _hyperbolic_clamp(self, latent: NDArray) -> NDArray:
        norm = np.linalg.norm(latent)
        if norm >= self.cfg.ball_radius:
            return latent * (self.cfg.ball_radius - 1e-6) / max(norm, 1e-8)
        return latent

    def _hyperbolic_clamp_batch(self, latents: NDArray) -> NDArray:
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        mask = norms.flatten() >= self.cfg.ball_radius
        if np.any(mask):
            latents = latents.copy()
            latents[mask] = (
                latents[mask]
                * (self.cfg.ball_radius - 1e-6)
                / np.maximum(norms[mask], 1e-8)
            )
        return latents

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self.projection_learner:
            state["projection_state"] = self.projection_learner.get_state()
        elif self._raw_projection is not None:
            state["projection"] = self._raw_projection.tolist()
        if self.sot_tokenizer:
            state["sot_tokenizer"] = self.sot_tokenizer.get_state()
        if self.sot_hebbian:
            state["sot_hebbian"] = {"lr": self.sot_hebbian.lr}
        if self._sot_field_ema is not None:
            state["sot_field_ema"] = self._sot_field_ema.tolist()
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if self.projection_learner is not None and "projection_state" in state:
            self.projection_learner.load_state(state["projection_state"])
        elif "projection" in state and self._raw_projection is not None:
            self._raw_projection = np.array(state["projection"], dtype=np.float32)
        if self.sot_tokenizer and "sot_tokenizer" in state:
            self.sot_tokenizer.load_state(state["sot_tokenizer"])
        if self._sot_field_ema is not None and "sot_field_ema" in state:
            self._sot_field_ema = np.array(state["sot_field_ema"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def has_sot(self) -> bool:
        return self.sot_tokenizer is not None

    @property
    def has_sot_hebbian(self) -> bool:
        return self.sot_hebbian is not None

    @property
    def sot_field_ema(self) -> Optional[NDArray]:
        return self._sot_field_ema

    def sot_vocab_ids(self) -> List[Any]:
        if self.sot_tokenizer:
            return list(self.sot_tokenizer.token_embeddings.keys())
        return []

    def sot_cooccurrence_score(self, pair: Tuple[Any, Any]) -> float:
        if self.sot_tokenizer:
            return self.sot_tokenizer.cooccurrence.get(pair, 0.0)
        return 0.0
