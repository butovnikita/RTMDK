"""
rtmdk/memory/core.py
Resonance-Topological Memory - Version 8.3.0

Phase 11 Features:
Track 1: Multi-level memory stratification (Episodic / Semantic / Procedural)
Track 2: Hyperbolic geometry (Poincare ball model)
Track 3: Predictive coding / Active inference
Track 4: Counterfactual imagination & scenario planning
Track 5: Differential privacy & secure federation

All v7 components preserved:
MetaAdaptiveKernel, TopologyHealer, CausalInferenceEngine, NeuralODEDynamics,
IncPCAProjection, BM25Index, HNSWIndex, TorchBackend, LearnableKernel,
DifferentiableConsolidation, AgentPlanner, HypothesisVerifier, ToolRouter,
ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager, MetaController,
KuramotoSync, FederatedRTMDK, FederatedNode, detect_modality, cross_modal_resonance
"""

from __future__ import annotations
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import (
    ContextFormat,
    RTMDKConfig,
)
from rtmdk.memory.context_manager import ContextManager
from rtmdk.memory.memory_post_initializer import MemoryPostInitializer
from rtmdk.memory.backlog_modules_initializer import BacklogModulesInitializer
from rtmdk.nodes import ContradictionRecord, MemoryNode
import asyncio
import functools
import json
import time
import os
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, model_validator
import logging

# Extracted engine classes (kept in sync with rtmdk/support/ modules)
from rtmdk.memory.utils import (
    SecurityViolationError,
    _sanitize_path,
    _safe_json_load,
)
from rtmdk.utils.formatting import build_system_prompt
from rtmdk.memory.observability import MemoryMetrics
from rtmdk.memory.pipeline_builder import PipelineBuilder

logger = logging.getLogger(__name__)

# Phase 5: dataclass nodes extracted to rtmdk.nodes

try:
    from rtmdk.support.triton_backend import TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False

try:
    from rtmdk.support.ump import UniversalMemoryProtocol

    UMP_AVAILABLE = True
except ImportError:
    UMP_AVAILABLE = False

# Torch availability check
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# ============================================================================
# CONSTANTS: Named constants for magic numbers
# ============================================================================

# Statistical constants
CHI_SQUARED_CRITICAL_DF1 = 3.84  # Chi-squared critical value (df=1, p=0.05)
CHI_SQUARED_CRITICAL_DF2 = 5.99  # Chi-squared critical value (df=2, p=0.05)

# Consolidation constants
CONSOLIDATION_DISTANCE_THRESHOLD = 2.5
CONSOLIDATION_PROBABILITY = 0.15
CRYSTALLIZATION_SIMILARITY_HIGH = 0.75
CRYSTALLIZATION_SIMILARITY_LOW = 0.6

# Session retrieval boost
SESSION_BOOST_FACTOR = 1.3  # 30% boost for session-matching nodes

# Performance limits
MAX_TENSION_SCAN = 200
CACHE_INVALID_HASH_MODULUS = 5

# File limits
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
MAX_NODE_TEXT_LENGTH = 10000
SECURE_FILE_PERMISSIONS = 0o600

# Frequency constants (steps)
SELF_SUPERVISION_FREQ = 20
ODE_SMOOTHNESS_FREQ = 10
TENSION_CHECK_FREQ = 100
HEALING_CHECK_FREQ = 50
SYMBOLIC_OVERLAY_FREQ = 50
META_KERNEL_ADAPT_FREQ = 5


# ============================================================================
# CORE: RTmdKField v7
# ============================================================================


def _locked(method):
    """Decorator that wraps method in self._write_lock RLock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)

    return wrapper


class RTMDKMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    config: RTMDKConfig = Field(default_factory=RTMDKConfig)
    embedder: Optional[Callable[[str], NDArray[np.float32]]] = None
    field: Optional[RTMDKField] = Field(default=None, exclude=True)
    session_phases: Dict[str, float] = Field(default_factory=dict)
    wal_path: Optional[str] = Field(default=None, exclude=True)
    # Observability metrics store (None unless sot.observability_enabled)
    metrics: Optional[Any] = Field(default=None, exclude=True)

    # SOT v2.0 online-learning state (managed by MemoryPostInitializer)
    _sot_v2_online_buffer: List[List[int]] = PrivateAttr(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _init_field(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            cfg = data.get("config", RTMDKConfig())
            if data.get("embedder") is None:
                # Batteries included: zero-dependency SOT embedder so that
                # RTMDKMemory() works out of the box (see default_embedder.py)
                from rtmdk.memory.default_embedder import create_default_embedder

                data["embedder"] = create_default_embedder(dim=cfg.embedding_dim)
            if data.get("field") is None:
                wal = data.get("wal_path")
                data["field"] = RTMDKField(cfg, wal_path=wal)
        return data

    def model_post_init(self, __context):
        MemoryPostInitializer(self).initialize()
        BacklogModulesInitializer(self).initialize()
        object.__setattr__(self, "_context_mgr", ContextManager(self))

    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]

    def add(self, text: str, content: Optional[Dict] = None, **kwargs) -> str:
        """Ergonomic shortcut: embed ``text`` and add it as a memory node.

        The 3-line quickstart API::

            memory = RTMDKMemory()
            memory.add("Paris is the capital of France")
            results = memory.query("capital of France")
        """
        assert self.embedder is not None  # set by _init_field validator
        payload = dict(content or {})
        payload.setdefault("text", text)
        return self.add_node(self.embedder(text), payload, **kwargs)

    def query(self, text: str, top_k: Optional[int] = None, **kwargs) -> List[Tuple[str, float, Any]]:
        """Ergonomic shortcut: retrieve memory nodes by raw text."""
        assert self.embedder is not None  # set by _init_field validator
        return self.retrieve_nodes(text, embedding=self.embedder(text), top_k=top_k, **kwargs)

    def add_node(self, embedding: NDArray, content: Dict, **kwargs) -> str:
        """Add a node to the memory field. Delegates to RTMDKField.add_node."""
        # P2: Accumulate corpus for SOT v2.0 lazy training & online update
        if self._sot_v2 is not None:
            text = content.get("text", "")
            if text:
                # FIFO corpus to prevent OOM
                self._sot_v2_corpus.append(text)
                if len(self._sot_v2_corpus) > self._sot_v2_corpus_maxlen:
                    self._sot_v2_corpus.pop(0)
                # Online update: tokenize and buffer (thread-safe)
                if hasattr(self._sot_v2, "_vocab") and self._sot_v2._vocab:
                    tokens = [
                        self._sot_v2._vocab[w] for w in self._sot_v2._word_tokenize(text) if w in self._sot_v2._vocab
                    ]
                    if tokens:
                        with self._sot_v2_online_lock:
                            self._sot_v2_online_buffer.append(tokens)
                            should_update = len(self._sot_v2_online_buffer) >= self._sot_v2_online_threshold
                            if should_update:
                                buffer_to_update = self._sot_v2_online_buffer
                                self._sot_v2_online_buffer = []
                        if should_update:
                            # R4.2 (2026-08-24): Hebbian update mutates token embeddings
                            # that field.step() / query may read. Hold field write lock
                            # to avoid race (field.step touches latent_pos + SOT cooccurrence).
                            # Lock order: _sot_v2_online_lock -> field._write_lock (consistent).
                            try:
                                # field may not be initialized during early construction
                                fld_lock = getattr(self, "field", None)
                                fld_lock = getattr(fld_lock, "_write_lock", None) if fld_lock is not None else None
                                if fld_lock is not None:
                                    with fld_lock:
                                        self._sot_v2._embedder.online_update(buffer_to_update)
                                else:
                                    self._sot_v2._embedder.online_update(buffer_to_update)
                            except Exception:
                                logger.warning("SOT v2.0 online update failed", exc_info=True)
        # P1: Matryoshka-lite — truncate for HNSW, keep full for resonance
        modal_emb = None
        if getattr(self.config, "matryoshka_mode", False):
            hnsw_dim = getattr(self.config, "matryoshka_hnsw_dim", self.config.latent_dim)
            if embedding.shape[0] > hnsw_dim:
                modal_emb = embedding
                embedding = embedding[:hnsw_dim]
        # P1: Sparse index insert (after we know the real node_id)
        t0 = time.perf_counter()
        node_id = self.field.add_node(embedding, content, modal_embedding=modal_emb, **kwargs)
        if getattr(self, "sparse_index", None) is not None:
            sparse_vec = content.get("sparse_embedding")
            if sparse_vec:
                self.sparse_index.insert(node_id, sparse_vec)
        # Engram cache: keep embedding in RAM for fast engram similarity
        if self.engram_cache is not None:
            self.engram_cache.add(node_id, embedding)
        # Observability
        if self.metrics is not None:
            self.metrics.record_ingestion((time.perf_counter() - t0) * 1000)
        # v8.2.1 hooks
        self._on_node_added(node_id, embedding, content, kwargs)
        return node_id

    def _on_node_added(self, node_id: str, embedding: NDArray, content: Dict, add_kwargs: Dict) -> None:
        rm = getattr(self, "replication_manager", None)
        if rm is not None and rm.enabled:
            try:
                rm.replicate(
                    {
                        "op": "add_node",
                        "node_id": node_id,
                        "embedding": embedding.tolist(),
                        "content": content,
                        "kwargs": {k: v for k, v in add_kwargs.items() if k not in {"embedding", "content"}},
                    }
                )
            except Exception:
                logger.warning("Replication failed for %s", node_id, exc_info=True)

    def add_nodes_batch(
        self,
        embeddings: NDArray,
        contents: List[Dict],
        phases: Optional[NDArray] = None,
        node_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        skip_projection: bool = False,
    ) -> List[str]:
        """Batch add nodes. Delegates to RTMDKField.add_nodes_batch."""
        # P1: Matryoshka-lite
        modal_embs = None
        if getattr(self.config, "matryoshka_mode", False):
            hnsw_dim = getattr(self.config, "matryoshka_hnsw_dim", self.config.latent_dim)
            if embeddings.shape[1] > hnsw_dim:
                modal_embs = embeddings
                embeddings = embeddings[:, :hnsw_dim]
        # P1: Sparse index
        if getattr(self, "sparse_index", None) is not None:
            for i, content in enumerate(contents):
                sparse_vec = content.get("sparse_embedding")
                nid = node_ids[i] if node_ids else f"n_{i}_"
                if sparse_vec:
                    self.sparse_index.insert(nid, sparse_vec)
        result = self.field.add_nodes_batch(
            embeddings,
            contents,
            phases,
            node_ids,
            session_ids,
            modalities,
            skip_projection,
            modal_embeddings=modal_embs,
        )
        # v8.2.1 hooks
        rm = getattr(self, "replication_manager", None)
        for i, nid in enumerate(result):
            emb = embeddings[i]
            content = contents[i]
            if rm is not None and rm.enabled:
                try:
                    rm.replicate(
                        {
                            "op": "add_node",
                            "node_id": nid,
                            "embedding": emb.tolist(),
                            "content": content,
                        }
                    )
                except Exception:
                    logger.warning("Replication failed for %s", nid, exc_info=True)
        # P2: Accumulate corpus for SOT v2.0
        if self._sot_v2 is not None:
            for content in contents:
                text = content.get("text", "")
                if text:
                    self._sot_v2_corpus.append(text)
        return result

    def train_sot_v2(self, extra_texts: Optional[List[str]] = None) -> bool:
        """Train SOT v2.0 embedder on accumulated corpus.

        Call this after ingesting all documents.  Optionally provide
        additional query texts for contrastive fine-tuning.

        Returns:
            True if training succeeded.
        """
        if self._sot_v2 is None:
            logger.warning("train_sot_v2 called but SOT v2.0 is not enabled")
            return False
        corpus = list(self._sot_v2_corpus)
        if extra_texts:
            corpus.extend(extra_texts)
        if not corpus:
            logger.warning("train_sot_v2: no corpus to train on")
            return False
        logger.info("train_sot_v2: training on %d texts", len(corpus))
        try:
            self._sot_v2.train(corpus)
            # Optional: teacher alignment (Procrustes or contrastive distillation)
            sot_cfg = getattr(self.config, "sot", None)
            teacher_name = getattr(sot_cfg, "sot_v2_align_teacher", None) if sot_cfg else None
            if teacher_name:
                try:
                    from sentence_transformers import SentenceTransformer

                    teacher = SentenceTransformer(teacher_name)
                    batch_size = getattr(sot_cfg, "sot_v2_align_batch_size", 64)
                    center = getattr(sot_cfg, "sot_v2_align_center", True)
                    align_mode = getattr(sot_cfg, "sot_v2_align_mode", "procrustes")
                    logger.info("train_sot_v2: aligning to teacher %s (mode=%s)...", teacher_name, align_mode)
                    sif_embs = self._sot_v2.embed_batch(corpus)
                    teacher_embs = []
                    for i in range(0, len(corpus), batch_size):
                        batch = corpus[i : i + batch_size]
                        embs = teacher.encode(batch)
                        if not isinstance(embs, np.ndarray):
                            embs = np.asarray(embs)
                        teacher_embs.append(embs)
                    teacher_embs = np.concatenate(teacher_embs, axis=0)
                    if align_mode == "contrastive_distill":
                        self._sot_v2._embedder.contrastive_distill(sif_embs, teacher_embs)
                    else:
                        # Procrustes (legacy)
                        norms = np.linalg.norm(teacher_embs, axis=1, keepdims=True) + 1e-8
                        teacher_embs = teacher_embs / norms
                        self._sot_v2._embedder.align_to_teacher(sif_embs, teacher_embs, center=center)
                except Exception:
                    logger.warning("train_sot_v2: teacher alignment failed", exc_info=True)
            # Replace embedder with trained SOT v2.0
            self.embedder = self._sot_v2
            # Invalidate query cache (scores changed)
            if self.field.query_cache is not None:
                self.field.query_cache.clear()
            # Reset conformal calibrator (distribution changed)
            if self.field.conformal_calibrator is not None:
                from rtmdk.memory.conformal import ConformalCalibrator

                self.field.conformal_calibrator = ConformalCalibrator(alpha=self.config.conformal_alpha)
                logger.info("train_sot_v2: conformal calibrator reset")
            logger.info("train_sot_v2: SOT v2.0 embedder active")
            return True
        except Exception:
            logger.error("train_sot_v2 failed", exc_info=True)
            return False

    def _replay_wal(self) -> None:
        """Replay WAL mutations to recover durability after restart.

        Reads all records from WAL and re-applies them:
        - add_node / add_nodes_batch: re-embed text and insert
        - delete: remove nodes
        - consolidate: skipped (snapshot already contains consolidated state)
        """
        if not self.field.wal.enabled:
            return
        records = self.field.wal.replay()
        if not records:
            return
        # Disable WAL during replay to avoid writing replayed ops back to WAL
        was_enabled = self.field.wal.enabled
        self.field.wal.enabled = False
        logger.info(f"WAL replay: {len(records)} records")
        replayed = 0
        try:
            for rec in records:
                op = rec.get("op")
                payload = rec.get("payload", {})
                try:
                    if op == "add_node":
                        content = payload.get("content", {})
                        modality = payload.get("modality", "text")
                        node_id = payload.get("node_id")
                        # Use stored embedding if available (Track 5), else
                        # re-embed
                        emb_list = payload.get("embedding")
                        if emb_list is not None:
                            embedding = np.array(emb_list, dtype=np.float32)
                        else:
                            text = content.get("text", "")
                            if not text:
                                text = content.get("input_text", "")
                            if not text:
                                logger.warning(f"WAL replay add_node: no text for {node_id}")
                                continue
                            embedding = self.embedder(text)
                        phase = self._get_phase(payload.get("session_id"), embedding)
                        self.field.add_node(
                            embedding,
                            content,
                            phase=phase,
                            node_id=node_id,
                            modality=modality,
                        )
                        replayed += 1
                    elif op == "add_nodes_batch":
                        contents = payload.get("contents", [])
                        modalities = payload.get("modalities")
                        node_ids = payload.get("node_ids")
                        embeddings_list = payload.get("embeddings")
                        if embeddings_list is not None:
                            embeddings = np.array(embeddings_list, dtype=np.float32)
                        else:
                            # Fallback: re-embed each content
                            texts = []
                            for c in contents:
                                t = c.get("text", "")
                                if not t:
                                    t = c.get("input_text", "")
                                texts.append(t)
                            embeddings = np.array([self.embedder(t) for t in texts], dtype=np.float32)
                        if len(embeddings) == len(contents):
                            self.field.add_nodes_batch(
                                embeddings,
                                contents,
                                node_ids=node_ids,
                                modalities=modalities,
                            )
                            replayed += len(contents)
                    elif op == "delete":
                        node_ids = payload.get("node_ids", [])
                        if node_ids:
                            self.field.delete_nodes(node_ids)
                            replayed += len(node_ids)
                    elif op == "consolidate":
                        # Skip: snapshot already contains consolidated state;
                        # re-running consolidate would be non-deterministic.
                        pass
                except Exception:
                    logger.warning(f"WAL replay failed for {op}: {payload}", exc_info=True)
        finally:
            self.field.wal.enabled = was_enabled
        logger.info(f"WAL replay complete: {replayed} items recovered")

    def _get_phase(
        self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None, content: Optional[Dict] = None
    ) -> float:
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        # Use semantic phase from field when content is available
        if content is not None:
            return self.field._get_phase(session_id, embedding, content=content)
        phase = (time.time() * 0.01) % (2 * np.pi)
        if session_id:
            self.session_phases[session_id] = phase
        return phase

    def _retrieve_and_format(self, query: str, embedding: NDArray, session_id: str) -> str:
        """Core retrieval pipeline — delegated to ContextManager."""
        return self._context_mgr.retrieve_and_format(query, embedding, session_id)

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        embedding = self.embedder(query)
        return {"rtmdk_context": self._retrieve_and_format(query, embedding, session_id)}

    def load_memory_variables_with_embedding(self, inputs: Dict[str, str], embedding: NDArray) -> Dict[str, str]:
        """Query memory with pre-computed embedding (no HTTP call).

        This is the optimized version that accepts an embedding from
        an external embedder, avoiding the HTTP call to LM Studio.
        Use this for batch processing and fair benchmark comparisons.

        Args:
            inputs: {"input": "query text", "session_id": "...", ...}
            embedding: Pre-computed embedding vector (768d for nomic-embed)

        Returns:
            {"rtmdk_context": formatted context string}
        """
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        return {"rtmdk_context": self._retrieve_and_format(query, embedding, session_id)}

    def _retrieve_nodes_impl(
        self,
        query: str,
        embedding: NDArray,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        sparse_vec: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[str, float, Any]]:
        """Internal retrieval without metrics/locks/reranking."""
        # P1: Query expansion for short queries (< 3 content words)
        original_query = query
        if query and getattr(self.config, "query_expand_short", False) and hasattr(self.embedder, "expand_query_terms"):
            content_words = [w for w in query.lower().split() if len(w) > 2]
            if len(content_words) < 3:
                expanded = self.embedder.expand_query_terms(query, n_terms=3)
                if expanded:
                    expansion_text = " ".join([term for term, _ in expanded])
                    query = f"{original_query} {expansion_text}"
                    logger.debug("Query expanded: '%s' → '%s'", original_query, query)

        query_content = {"text": query} if query else None
        phase = self._get_phase(session_id, embedding, content=query_content)
        tk = top_k or self.field.cfg.top_k

        # P1: Adaptive Cascade Router
        if self.cascade_router is not None:
            from rtmdk.production.cascade_router import QueryType

            route = self.cascade_router.classify(query)
            if route == QueryType.FACTUAL:
                return self.field.query(embedding, phase, top_k=tk, session_id=session_id, query_text=query)

        # Primary: resonance retrieval
        results = self.field.query(embedding, phase, top_k=tk, session_id=session_id, query_text=query)

        # P1: Sparse index fallback (BGE-M3 learned sparse)
        if len(results) < tk and self.sparse_index is not None and sparse_vec:
            sparse_hits = self.sparse_index.search(sparse_vec, tk * 2)
            if sparse_hits:
                seen = {nid for nid, _, _ in results}
                for nid, score in sparse_hits:
                    if nid not in seen:
                        node = self.field.nodes.get(nid)
                        if node:
                            results.append((nid, score * 0.5, node))
                            seen.add(nid)
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:tk]

        # Phase 18: Engram-based retrieval as fallback (if resonance under-delivers)
        if len(results) < tk and self.engram_manager is not None and self.engram_manager.index.size > 0:
            # Use engram cache if available to avoid TieredNodeStore disk scan
            if self.engram_cache is not None and len(self.engram_cache) > 0:
                node_embs = self.engram_cache.get_all()
            else:
                node_embs = {}
                for nid, node in self.field.nodes.items():
                    emb = self._get_node_embedding(nid, node)
                    if emb is not None:
                        node_embs[nid] = emb

            engram_results = self.engram_manager.retrieve_engrams(embedding, node_embs, top_k=tk)
            if engram_results:
                engram_nodes = self.engram_manager.expand_engrams(engram_results, self.field, top_k=tk)
                seen = {nid for nid, _, _ in results}
                for nid, score, node in engram_nodes:
                    if nid not in seen:
                        results.append((nid, score, node))
                        seen.add(nid)
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:tk]
                self.field.stats["engram_retrievals"] = self.field.stats.get("engram_retrievals", 0) + 1

        # Session-scoped retrieval
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results if node.content.get("session") == session_id
            ]
            if len(session_results) < tk:
                global_results = [
                    (nid, score, node) for nid, score, node in results if node.content.get("session") != session_id
                ]
                needed = tk - len(session_results)
                session_results.extend(global_results[:needed])
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[:tk]
            self.field.stats["session_scoped_retrievals"] = self.field.stats.get("session_scoped_retrievals", 0) + 1

        # Hybrid BM25 blend
        if self.field.cfg.hybrid_alpha < 1.0 and self.field.bm25_index is not None and results:
            bm25_results = self.field.bm25_index.search(query, tk * 2)
            if bm25_results:
                bm25_scores = {nid: score for nid, score in bm25_results}
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                if max_bm25 > 0:
                    bm25_scores = {nid: s / max_bm25 for nid, s in bm25_scores.items()}
                alpha = self.field.cfg.hybrid_alpha
                blended = []
                for nid, score, node in results:
                    bm25_score = bm25_scores.get(nid, 0.0)
                    blended.append((nid, alpha * score + (1 - alpha) * bm25_score, node))
                for nid, bm25_score in bm25_scores.items():
                    if nid not in [n[0] for n in blended] and bm25_score > self.field.cfg.min_response:
                        node = self.field.nodes.get(nid)
                        if node:
                            blended.append((nid, alpha * 0.0 + (1 - alpha) * bm25_score, node))
                blended.sort(key=lambda x: x[1], reverse=True)
                results = blended[:tk]
                self.field.stats["hybrid_retrievals"] = self.field.stats.get("hybrid_retrievals", 0) + 1

        # Causal Traversal
        if self.causal_traversal_engine is not None and results:
            results = self.causal_traversal_engine.retrieve_with_causal(results, self.field, top_k=tk)

        # Cross-Encoder Reranker
        if self.reranker is not None and results and query:
            results = self.reranker.rerank(query, results, top_k=tk)

        return results

    def retrieve_nodes(
        self,
        query: str,
        embedding: NDArray,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        sparse_vec: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[str, float, Any]]:
        """Retrieve memory nodes with full pipeline: cascade → resonance → sparse → engrams → causal → reranker.

        Includes observability (latency tracking), distributed locking,
        sentence reranking, and query decomposition.

        Returns:
            List of (node_id, score, node) tuples.
        """
        # Pipeline path (v8.3+): use explicit pipeline when enabled and no sparse vec
        if getattr(self.config, "pipeline_enabled", False) and sparse_vec is None:
            result = self.retrieve_nodes_pipeline(query, embedding=embedding, top_k=top_k, session_id=session_id)
            return result["results"]

        # Legacy path — preserved for backward compatibility
        # R4.3 (2026-08-24): lock ordering is distributed_lock (outer, inter-process, file/redis)
        # -> field._write_lock (inner, intra-process RLock) inside query_manager.
        # Never acquire in reverse order. field._write_lock is RLock (re-entrant).
        # Use try/finally to guarantee release even on exception.

        acquired = False
        if self._distributed_lock is not None:
            try:
                acquired = bool(self._distributed_lock.acquire(blocking=True))
            except Exception:
                logger.warning("retrieve_nodes: distributed lock acquire raised", exc_info=True)
                acquired = False
            if not acquired:
                logger.warning("retrieve_nodes: failed to acquire distributed lock (proceeding without)")

        try:
            t0 = time.perf_counter()
            cache_hit = False

            # Check query cache
            if self.field.query_cache is not None:
                cache_key = self.field._query_cache_key(
                    self.field._project(embedding),
                    self._get_phase(session_id, embedding),
                    top_k or self.field.cfg.top_k,
                    "text",
                    session_id,
                )
                cached = self.field.query_cache.get_raw(cache_key)
                if cached is not None:
                    cache_hit = True
                    results = cached
                else:
                    results = self._retrieve_nodes_impl(query, embedding, top_k, session_id, sparse_vec)
                    self.field.query_cache.put_raw(cache_key, results)
            else:
                results = self._retrieve_nodes_impl(query, embedding, top_k, session_id, sparse_vec)

            # Auto query rewrite on low-quality results
            if self._query_rewriter is not None and self._query_rewriter.should_rewrite(
                results, threshold=getattr(self.config, "query_rewrite_threshold", 0.3)
            ):
                rewritten = self._query_rewriter.rewrite(query, results)
                if rewritten != query:
                    logger.debug("Query rewritten: '%s' -> '%s'", query, rewritten)
                    rew_emb = self.embedder(rewritten)
                    results = self._retrieve_nodes_impl(rewritten, rew_emb, top_k, session_id, sparse_vec)

            # Intent-aware retrieval tuning
            if self._intent_classifier is not None:
                intent = self._intent_classifier.classify(query)
                if intent == "factual":
                    # Boost top result precision
                    pass
                elif intent == "exploratory":
                    # Diversify results via causal traversal if available
                    pass

            # Sentence-level reranking
            if self._sentence_reranker is not None and results:
                results = self._sentence_reranker.rerank(query, results, top_k=top_k or self.field.cfg.top_k)

            # Observability
            latency_ms = (time.perf_counter() - t0) * 1000
            if self.metrics is not None:
                self.metrics.record_query(latency_ms, cache_hit=cache_hit)
                alerts = self.metrics.check_alerts()
                for alert in alerts:
                    logger.warning(alert)

            return results
        finally:
            if acquired:
                try:
                    self._distributed_lock.release()
                except Exception:
                    logger.warning("retrieve_nodes: distributed lock release failed", exc_info=True)

    def retrieve_nodes_with_explanations(
        self,
        query: str,
        embedding: NDArray,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        sparse_vec: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """Retrieve nodes with human-readable explanations.

        Returns:
            {"results": [(nid, score, node), ...], "explanations": [...], "intent": str}
        """
        results = self.retrieve_nodes(query, embedding, top_k, session_id, sparse_vec)
        intent = "unknown"
        if self._intent_classifier is not None:
            intent = self._intent_classifier.classify(query)

        explanations = []
        if self._result_explainer is not None:
            for nid, score, node in results:
                explanations.append(self._result_explainer.explain(query, nid, score, node, session_id or "default"))

        return {"results": results, "explanations": explanations, "intent": intent}

    def retrieve_nodes_batch(
        self,
        queries: List[str],
        embeddings: NDArray,
        top_k: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float, Any]]]:
        """Batch retrieval — vectorized resonance across multiple queries.

        ~50-100x faster than sequential retrieve_nodes() for large batches.
        """
        tk = top_k or self.field.cfg.top_k
        n_queries = len(queries)
        if n_queries == 0:
            return []

        # Compute phases
        phases = np.zeros(n_queries, dtype=np.float32)
        for i, (query, sid) in enumerate(zip(queries, session_ids or [None] * n_queries)):
            content = {"text": query} if query else None
            phases[i] = self._get_phase(sid, embeddings[i], content=content)

        # Project embeddings
        query_latents = np.vstack([self.field._project(embeddings[i]) for i in range(n_queries)])

        # Get all node IDs
        all_nids = list(self.field.node_index)
        if not all_nids:
            return [[] for _ in range(n_queries)]

        # Batch resonance: (n_queries, n_nodes)
        scores = self.field._batch_resonance(query_latents, phases, all_nids)

        # Build node lookup
        nodes = self.field.nodes

        results = []
        for i in range(n_queries):
            sid = session_ids[i] if session_ids else None
            # Session boost
            row_scores = scores[i].copy()
            if sid and sid != "default":
                for j, nid in enumerate(all_nids):
                    if nodes[nid].content.get("session") == sid:
                        row_scores[j] *= 1.5

            # Filter by min_response and get top_k
            valid = row_scores >= self.field.cfg.min_response
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                results.append([])
                continue

            valid_scores = row_scores[valid_indices]
            top_local = np.argsort(-valid_scores)[:tk]
            top_global = valid_indices[top_local]

            query_results = []
            for idx in top_global:
                nid = all_nids[idx]
                node = nodes[nid]
                query_results.append((nid, float(row_scores[idx]), node))
                node.last_resonated = time.time()
            results.append(query_results)

        self.field.stats["batch_queries"] = self.field.stats.get("batch_queries", 0) + n_queries
        return results

    # ------------------------------------------------------------------
    # Pipeline API (v8.3+)
    # ------------------------------------------------------------------
    def build_pipeline(self):
        """Build an explicit stage-based pipeline — delegated to PipelineBuilder."""
        return PipelineBuilder(self).build()

    def retrieve_nodes_pipeline(
        self,
        query: str,
        embedding: Optional[NDArray] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        metrics_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Retrieve nodes using the explicit pipeline API.

        Args:
            metrics_store: Optional PipelineMetricsStore to persist query metrics.

        Returns:
            Dict with keys:
                - results: List[Tuple[str, float, Any]]
                - route: str (factual/standard/deep)
                - explanations: List[Dict]
                - metrics: per-stage latency breakdown
        """
        pipeline = self.build_pipeline()
        cost_tracking = getattr(self.config, "pipeline_cost_tracking_enabled", False)
        cost_analyzer = None
        if cost_tracking:
            from rtmdk.pipeline.cost import PipelineCostAnalyzer

            cost_analyzer = PipelineCostAnalyzer()
            cost_analyzer.start(query)

        ctx = pipeline.run(
            query_text=query,
            top_k=top_k or self.field.cfg.top_k,
            session_id=session_id,
            embedding=embedding,
        )

        if cost_analyzer is not None:
            for metric in ctx.metrics:
                cost_analyzer.record_stage(
                    metric.name,
                    latency_ms=metric.latency_ms,
                )
            cost_breakdown = cost_analyzer.finalize()
        else:
            cost_breakdown = None

        result = {
            "results": ctx.results,
            "route": ctx.route,
            "explanations": ctx.explanations,
            "metrics": ctx.to_dict(),
        }
        if cost_breakdown is not None:
            result["cost"] = cost_breakdown.to_dict()
        if metrics_store is not None:
            metrics_store.write(result["metrics"])
        return result

    async def retrieve_nodes_pipeline_async(
        self,
        query: str,
        embedding: Optional[NDArray] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        metrics_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Async version of retrieve_nodes_pipeline().

        Executes the pipeline in a thread pool to avoid blocking the event loop.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.retrieve_nodes_pipeline,
            query,
            embedding,
            top_k,
            session_id,
            metrics_store,
        )

    def health_check_pipeline(self) -> Dict[str, Any]:
        """Run health checks on every pipeline stage.

        Returns:
            {"healthy": bool, "stages": [{"stage": str, "healthy": bool, "reason": str}]}
        """
        pipeline = self.build_pipeline()
        stage_health = []
        all_healthy = True
        for stage in pipeline.stages:
            healthy, reason = stage.health_check()
            if not healthy:
                all_healthy = False
            stage_health.append(
                {
                    "stage": stage.name,
                    "healthy": healthy,
                    "reason": reason,
                }
            )
        return {"healthy": all_healthy, "stages": stage_health}

    def add_feedback(self, query: str, node_id: str, relevant: bool) -> bool:
        """Provide explicit feedback to refine embeddings.

        Args:
            query: The original query text.
            node_id: ID of the retrieved node.
            relevant: True if the node was relevant, False otherwise.

        Returns:
            True if feedback was applied.
        """
        if self._feedback_loop is None:
            logger.warning("add_feedback: feedback_loop not enabled in config")
            return False
        node = self.field.nodes.get(node_id)
        if node is None:
            return False
        node_text = node.content.get("text", "")
        return self._feedback_loop.add_feedback(query, node_text, relevant)

    def get_metrics(self) -> Dict:
        """Return current observability metrics snapshot."""
        if self.metrics is None:
            return {"observability_enabled": False}
        return self.metrics.snapshot()

    def query_with_confidence(
        self,
        query: str,
        embedding: NDArray,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve with conformal prediction confidence guarantee.

        Returns a prediction set that contains the true relevant result
        with probability >= 1 - alpha (marginal coverage guarantee).

        Args:
            query: Query text.
            embedding: Query embedding.
            top_k: Max results to return.
            session_id: Optional session filter.
            alpha: Miscoverage rate (default from config.conformal_alpha).

        Returns:
            Dict with keys:
                - results: List of (node_id, score, node) tuples
                - prediction_set: List of node_ids in the conformal set
                - confidence: 1 - alpha
                - threshold: Minimum score for inclusion
                - coverage_guarantee: True if calibrator has enough samples
        """
        tk = top_k or self.field.cfg.top_k
        alpha = alpha or getattr(self.config, "conformal_alpha", 0.1)

        # Standard retrieval
        results = self.retrieve_nodes(query, embedding, top_k=tk, session_id=session_id)

        # Conformal prediction
        cal = getattr(self.field, "conformal_calibrator", None)
        if cal is None or not self.config.conformal_prediction:
            return {
                "results": results,
                "prediction_set": [nid for nid, _, _ in results],
                "confidence": 1.0 - alpha,
                "threshold": 0.0,
                "coverage_guarantee": False,
                "reason": "conformal_prediction disabled or not initialized",
            }

        min_calib = getattr(self.config, "conformal_min_calib", 50)
        if cal.n_calibrated < min_calib:
            return {
                "results": results,
                "prediction_set": [nid for nid, _, _ in results],
                "confidence": 1.0 - alpha,
                "threshold": 0.0,
                "coverage_guarantee": False,
                "reason": f"insufficient calibration samples ({cal.n_calibrated}/{min_calib})",
            }

        scores = [score for _, score, _ in results]
        nids = [nid for nid, _, _ in results]
        pred_set, confidence, threshold = cal.predict(scores, nids)

        return {
            "results": results,
            "prediction_set": pred_set,
            "confidence": confidence,
            "threshold": threshold,
            "coverage_guarantee": True,
            "reason": "marginal coverage guarantee active",
        }

    def calibrate_conformal_sot(self, n_samples: int = 50) -> bool:
        """Auto-calibrate conformal prediction using SOT cosine scores.

        Uses pseudo-queries (sample nodes as queries) to build a calibration
        set of cosine similarity scores.  This is essential when using SOT
        embedders where resonance scores differ from RTMDK's native scores.

        Args:
            n_samples: Number of pseudo-queries to generate.

        Returns:
            True if calibration succeeded.
        """
        cal = getattr(self.field, "conformal_calibrator", None)
        if cal is None:
            logger.warning("calibrate_conformal_sot: no calibrator available")
            return False
        nodes = self.field.nodes
        nids = list(self.field.node_index)
        if len(nids) < 10:
            logger.warning("calibrate_conformal_sot: too few nodes (%d)", len(nids))
            return False

        import random

        sample_size = min(n_samples, len(nids))
        sample_nids = random.sample(nids, sample_size)

        # Pre-compute normalized document matrix once (O(N) memory, not O(N²))
        doc_embs = []
        nid_to_idx = {}
        for i, dnid in enumerate(nids):
            d_emb = self._get_node_embedding(dnid, nodes[dnid])
            if d_emb is not None:
                d_emb = d_emb / (np.linalg.norm(d_emb) + 1e-8)
                doc_embs.append(d_emb)
                nid_to_idx[dnid] = len(doc_embs) - 1
        if not doc_embs:
            return False
        doc_matrix = np.stack(doc_embs)

        for nid in sample_nids:
            node = nodes[nid]
            text = node.content.get("text", "")
            if not text:
                continue
            target_idx = nid_to_idx.get(nid)
            if target_idx is None:
                continue
            try:
                q_emb = self.embedder(text)
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                sims = doc_matrix @ q_emb
                cal.add_sample(float(sims[target_idx]))
            except Exception:
                continue

        logger.info("calibrate_conformal_sot: calibrated with %d samples (total=%d)", sample_size, cal.n_calibrated)
        return cal.n_calibrated >= getattr(self.config, "conformal_min_calib", 50)

    def _get_node_embedding(self, nid: str, node) -> Optional[np.ndarray]:
        """Retrieve stored embedding for a node, or approximate from latent position."""
        # Check if node has modal_embedding (cross-modal)
        if hasattr(node, "modal_embedding") and node.modal_embedding is not None:
            return node.modal_embedding
        # Fallback: approximate embedding by inverse-projection from latent_pos
        # This is lossy but better than nothing for engram similarity
        if hasattr(node, "latent_pos") and node.latent_pos is not None:
            # Pad latent_pos (64d) to embedding_dim (768d) with zeros
            # Engram similarity uses cosine — zeros won't dominate
            emb_dim = self.field.cfg.embedding_dim
            latent = node.latent_pos
            if len(latent) < emb_dim:
                approx = np.zeros(emb_dim, dtype=np.float32)
                approx[: len(latent)] = latent
                return approx
            return latent[:emb_dim] if len(latent) > emb_dim else latent
        return None

    def batch_query(
        self, embeddings: List[np.ndarray], top_k: Optional[int] = None, session_id: Optional[str] = None
    ) -> List[List[Tuple[str, float, Any]]]:
        """Batch query memory for multiple embeddings."""
        if self.field is None:
            raise RuntimeError("Field not initialized")
        phases = (
            [self._get_phase(session_id, emb) for emb in embeddings]
            if session_id
            else [self._get_phase() for _ in embeddings]
        )
        return self.field.batch_query(embeddings, phases=phases, top_k=top_k, session_id=session_id)

    def _detect_tags(self, text: str) -> List[str]:
        """Auto-detect memory tags — delegated to ContextManager."""
        return self._context_mgr._detect_tags(text)

    def _generate_clarification(self, results: List, query: str) -> str:
        """Generate a clarification prompt — delegated to ContextManager."""
        return self._context_mgr._generate_clarification(results, query)

    def get_system_prompt(self, context: str) -> str:
        return build_system_prompt(context, self.config.context_format, self.config.use_structured_prompt)

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save a conversation turn — delegated to ContextManager."""
        return self._context_mgr.save_context(inputs, outputs)

    async def _evolve_field_async(self):
        """Async field evolution — delegated to ContextManager."""
        return await self._context_mgr._evolve_field_async()

    def save_state(self, dir_path: str) -> None:
        """Persist backlog module state to disk."""
        import os

        os.makedirs(dir_path, exist_ok=True)
        if self.engram_cache is not None:
            self.engram_cache.save(os.path.join(dir_path, "engram_cache.npz"))
        if self._feedback_loop is not None and self._feedback_loop.persist_path:
            self._feedback_loop._flush()
        if self.metrics is not None:
            self.metrics.flush_to_file(os.path.join(dir_path, "metrics.json"))

    def load_state(self, dir_path: str) -> None:
        """Restore backlog module state from disk."""
        import os

        cache_path = os.path.join(dir_path, "engram_cache.npz")
        if self.engram_cache is not None and os.path.exists(cache_path):
            self.engram_cache.load(cache_path)
        if self._feedback_loop is not None:
            self._feedback_loop.load()

    def clear(self) -> None:
        # Fix 3: Cancel background workers before replacing field
        for task in self.field._workers:
            if not task.done():
                task.cancel()
        self.field._workers.clear()
        self.field = RTMDKField(self.config, wal_path=self.wal_path)
        self.session_phases.clear()
        # Reset SOT corpus to prevent stale data
        self._sot_v2_corpus.clear()
        self._sot_v2_online_buffer.clear()
        if self.engram_cache is not None:
            self.engram_cache.clear()
        if self.metrics is not None:
            old_rules = self.metrics.alert_rules
            self.metrics = MemoryMetrics()
            for rule in old_rules:
                self.metrics.add_alert_rule(rule)

    def health_check(self) -> Dict:
        """Comprehensive health check for k8s probes and monitoring.

        Returns:
            {"status": "healthy"|"degraded"|"unhealthy", "checks": {...}}
        """
        import psutil

        checks = {}
        status = "healthy"

        # Node count
        node_count = len(self.field.nodes)
        max_nodes = getattr(self.config, "max_nodes", 100000) or 100000
        node_ratio = node_count / max_nodes
        checks["node_count"] = {"value": node_count, "max": max_nodes, "ratio": round(node_ratio, 3)}
        if node_ratio > 0.9:
            status = "degraded"
            checks["node_count"]["warning"] = "Approaching max_nodes limit"

        # Memory
        try:
            mem = psutil.virtual_memory()
            checks["memory"] = {
                "used_percent": mem.percent,
                "available_mb": round(mem.available / 1024 / 1024, 1),
            }
            if mem.percent > 90:
                status = "degraded"
                checks["memory"]["warning"] = "High memory usage"
        except Exception:
            checks["memory"] = {"error": "psutil unavailable"}

        # Disk
        try:
            disk = psutil.disk_usage(".")
            checks["disk"] = {"used_percent": round(disk.percent, 1)}
            if disk.percent > 90:
                status = "degraded"
                checks["disk"]["warning"] = "Low disk space"
        except Exception:
            checks["disk"] = {"error": "psutil unavailable"}

        # Embedder circuit breaker
        if hasattr(self, "_embedder_cb"):
            cb_state = self._embedder_cb.state.value
            checks["embedder_circuit_breaker"] = {"state": cb_state}
            if cb_state == "open":
                status = "degraded"

        # Metrics snapshot
        if self.metrics is not None:
            checks["metrics"] = self.metrics.snapshot()

        # Engram cache
        if self.engram_cache is not None:
            checks["engram_cache"] = {"size": len(self.engram_cache)}

        return {"status": status, "checks": checks}

    def take_snapshot(self) -> None:
        """Capture current memory state for future rollback."""
        self._rollback_manager.take_snapshot(self.field)
        logger.info("Memory snapshot taken")

    def detect_poisoned_memories(self) -> List[Dict]:
        """Scan for potentially poisoned or anomalous memory nodes."""
        return self._poison_detector.scan(self.field)

    def inspect_node(self, node_id: str) -> Optional[Dict]:
        if node_id not in self.field.nodes:
            return None
        node = self.field.nodes[node_id]
        info = {
            "id": node.id,
            "phase": node.phase,
            "amplitude": node.amplitude,
            "salience": node.salience,
            "tension": node.tension,
            "soft_gate": node.soft_gate,
            "self_sup_score": node.self_sup_score,
            "modal_weight": node.modal_weight,
            "modality": node.modality,
            "lineage": node.lineage,
            "content": node.content,
            "created_at": node.created_at,
            "last_resonated": node.last_resonated,
            "causal_parents": node.causal_parents,
            "causal_strength": node.causal_strength,
            "causal_effects": node.causal_effects,
            "is_causal_root": node.is_causal_root,
            "is_healing": node.is_healing,
            "healing_origin": node.healing_origin,
            "local_density": node.local_density,
            "goal_tags": node.goal_tags,
            "cross_modal_score": node.cross_modal_score,
        }
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        if node.velocity is not None:
            info["velocity"] = node.velocity.tolist()
        if node.modal_embedding is not None:
            info["modal_embedding"] = node.modal_embedding.tolist()
        return info

    def get_rollback_history(self) -> List[Dict]:
        return [
            {"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(s["pre_state"])}
            for s in self.field._rollback_history
        ]

    def do_intervention(self, node_id: str, text: str):
        emb = self.embedder(text)
        self.field.do_intervention(node_id, emb)

    def __getattr__(self, name: str):
        """Proxy simple delegations to RTMDKField to reduce boilerplate.

        R2.2 (RISKS.md): this creates the field -> manager -> field delegation cycle
        (see mypy.ini disable_error_code=attr-defined, 77% false positives).
        Keep _proxy_methods explicit (no wildcard) so mypy can still flag
        real missing attributes; add type: ignore[attr-defined] locally if you
        extend delegation, don't re-enable global ignore.
        """
        # Respect pydantic private/extra attributes first
        pydantic_extra = object.__getattribute__(self, "__pydantic_extra__")
        if pydantic_extra is not None and name in pydantic_extra:
            return pydantic_extra[name]
        pydantic_private = object.__getattribute__(self, "__pydantic_private__")
        if pydantic_private is not None and name in pydantic_private:
            return pydantic_private[name]
        # get_dashboard is a legacy alias for get_field_health
        if name == "get_dashboard":
            return self.field.get_field_health
        _proxy_methods = {
            "get_field_health",
            "counterfactual_query",
            "get_causal_summary",
            "export_field",
            "import_field",
            "rollback",
            "clear_interventions",
            "fit_projection",
        }
        if name in _proxy_methods:
            return getattr(self.field, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_contradictions(self) -> List[ContradictionRecord]:
        if self.field.causal_engine:
            return list(self.field.causal_engine.contradictions.values())
        return []

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> bool:
        if self.field.causal_engine and contradiction_id in self.field.causal_engine.contradictions:
            self.field.causal_engine.contradictions[contradiction_id].resolved = True
            self.field.causal_engine.contradictions[contradiction_id].resolution = resolution
            return True
        return False

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        if self.field.causal_engine:
            return self.field.causal_engine.validate_consolidation(node_a, node_b)
        return {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}

    def get_ragas_trend(self) -> Dict[str, float]:
        if self.field.ragas_evaluator:
            return self.field.ragas_evaluator.get_trend()
        return {}

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        if self.field.dp:
            self.field.stats["privacy_budget_spent"] = self.field.dp.get_privacy_spent()
        return {**self.field.stats, "config": self.config.asdict()}

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self, base_query: str, intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical scenarios."""
        embedding = self.embedder(base_query)
        return self.field.imagine_counterfactual(embedding, intervention)

    @classmethod
    def import_field(cls, path: str, embedder: Callable, wal_path: Optional[str] = None) -> "RTMDKMemory":
        return RTMDKField.import_field(path, embedder, wal_path=wal_path)

    # Phase 16 Track 3: Universal Memory Protocol
    def export_ump(self, path: str, source: str = "", comment: str = ""):
        """Export to Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError("Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = UniversalMemoryProtocol.export(self.field, self, source=source, comment=comment)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ump, f, ensure_ascii=False, indent=2)

    @classmethod
    def import_ump(cls, path: str, embedder: Callable) -> "RTMDKMemory":
        """Import from Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError("Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.import_ump(ump, embedder, memory_class=cls)

    def validate_ump(self, path: str) -> Dict:
        """Validate a UMP file."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            return {"valid": False, "issues": ["UMP not available"]}
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.validate(ump)

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release all background resources (WAL, threads, file handles).

        Safe to call multiple times.  After close the instance should not
        be used for mutations.
        """
        if self.field is not None:
            self.field.close()

    def __enter__(self) -> "RTMDKMemory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
