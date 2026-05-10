"""MemoryPostInitializer — extracts the ~160-line model_post_init from core.py."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.core import RTMDKMemory

logger = logging.getLogger(__name__)


class MemoryPostInitializer:
    """Encapsulates RTMDKMemory.model_post_init logic."""

    def __init__(self, memory: "RTMDKMemory") -> None:
        self._mem = memory

    def initialize(self) -> None:
        """Complete RTMDKMemory initialization after pydantic validation.

        Creates the field if missing, replays WAL, wires async workers,
        engrams, causal traversal, reranker, and all backlog modules.
        """
        mem = self._mem
        if mem.field is None:
            from rtmdk.memory.field import RTMDKField
            object.__setattr__(
                mem, "field", RTMDKField(
                    mem.config, wal_path=mem.wal_path))
        # Track 5: Replay WAL mutations for durability
        mem._replay_wal()
        # Fix 4: Auto-start async workers if async_pipeline is enabled
        if mem.config.async_pipeline and not mem.field._workers_started:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(mem.field._start_workers())
            except RuntimeError:
                pass

        self._init_engrams()
        self._init_causal_traversal()
        self._init_reranker()
        self._init_contextual_retrieval()
        self._init_bgem3()
        self._init_sot_v2()
        self._init_cascade_router()
        self._init_embedder_circuit_breaker()
        self._validate_config()

    def _init_engrams(self) -> None:
        mem = self._mem
        if mem.config.enable_engrams:
            try:
                from rtmdk.engrams import EngramManager
                object.__setattr__(mem, "engram_manager", EngramManager(
                    min_nodes=mem.config.engram_min_nodes,
                    max_nodes=mem.config.engram_max_nodes,
                    creation_threshold=mem.config.engram_creation_threshold,
                    decay_rate=mem.config.engram_decay_rate,
                    pattern_completion=mem.config.engram_pattern_completion,
                    overlap_threshold=mem.config.engram_overlap_threshold,
                ))
            except Exception:
                logger.warning(
                    "Engram manager initialization failed, disabling",
                    exc_info=True)
                object.__setattr__(mem, "engram_manager", None)
        else:
            object.__setattr__(mem, "engram_manager", None)

    def _init_causal_traversal(self) -> None:
        mem = self._mem
        mem.causal_traversal_engine = None
        if mem.config.causal_traversal:
            try:
                from rtmdk.engines.causal_traversal import CausalTraversalEngine
                mem.causal_traversal_engine = CausalTraversalEngine(
                    max_hops=mem.config.causal_max_hops,
                    decay_per_hop=0.5,
                )
            except Exception:
                logger.warning(
                    "CausalTraversalEngine initialization failed, disabling",
                    exc_info=True)

    def _init_reranker(self) -> None:
        mem = self._mem
        mem.reranker = None
        if getattr(mem.config, "reranker_enabled", False):
            try:
                from rtmdk.production.reranker import CrossEncoderReranker
                mem.reranker = CrossEncoderReranker(
                    model_name=getattr(
                        mem.config, "reranker_model",
                        "BAAI/bge-reranker-v2-m3"),
                )
            except Exception:
                logger.warning(
                    "CrossEncoderReranker initialization failed, disabling",
                    exc_info=True)

    def _init_contextual_retrieval(self) -> None:
        mem = self._mem
        mem.header_generator = None
        if getattr(mem.config, "contextual_retrieval", False):
            try:
                from rtmdk.production.contextual_retrieval import (
                    ContextualHeaderGenerator, ContextualEmbedderWrapper)
                sot = getattr(
                    mem.field, "sot_tokenizer", None) if hasattr(
                        mem, "field") else None
                mem.header_generator = ContextualHeaderGenerator(
                    backend=getattr(
                        mem.config, "contextual_backend", "heuristic"),
                    sot_tokenizer=sot,
                )
                mem.embedder = ContextualEmbedderWrapper(
                    mem.embedder, mem.header_generator)
                logger.info(
                    "Contextual retrieval enabled (%s)",
                    mem.header_generator.backend)
            except Exception:
                logger.warning(
                    "Contextual retrieval init failed, disabling",
                    exc_info=True)

    def _init_bgem3(self) -> None:
        mem = self._mem
        mem.bgem3_embedder = None
        mem.sparse_index = None
        if getattr(mem.config, "bgem3_enabled", False):
            try:
                from rtmdk.production.bgem3_embedder import BGEM3Embedder
                from rtmdk.production.sparse_index import SparseIndex
                mem.bgem3_embedder = BGEM3Embedder(
                    model_name=getattr(
                        mem.config, "bgem3_model_name", "BAAI/bge-m3"),
                )
                mem.sparse_index = SparseIndex()
                logger.info("BGE-M3 hybrid retrieval enabled")
            except Exception:
                logger.warning("BGE-M3 init failed, disabling", exc_info=True)

    def _init_sot_v2(self) -> None:
        mem = self._mem
        mem._sot_v2 = None
        mem._sot_v2_corpus: List[str] = []
        mem._sot_v2_corpus_maxlen: int = getattr(
            mem.config, "sot_max_corpus", 10000)
        mem._sot_v2_online_buffer: List[List[int]] = []
        mem._sot_v2_online_threshold: int = getattr(
            mem.config, "sot_online_update_threshold", 10)
        mem._sot_v2_online_lock = threading.Lock()
        sot_cfg = getattr(mem.config, "sot", None)
        if sot_cfg and getattr(sot_cfg, "sot_v2_enabled", False):
            try:
                from rtmdk.memory.sot_v2.integration import SOTv2Embedder
                mem._sot_v2 = SOTv2Embedder(
                    latent_dim=getattr(mem.config, "latent_dim", 384),
                    a=getattr(sot_cfg, "sot_v2_a", 0.01),
                    window_size=getattr(sot_cfg, "sot_v2_window", 5),
                    remove_pc=getattr(sot_cfg, "sot_v2_remove_pc", True),
                )
                logger.info("SOT v2.0 embedder initialised (lazy training)")
                aligner_path = getattr(sot_cfg, "sot_v2_aligner_path", None)
                if aligner_path:
                    try:
                        mem._sot_v2.load_aligner(aligner_path)
                    except Exception:
                        logger.warning(
                            "SOT v2.0 aligner load failed from %s",
                            aligner_path, exc_info=True)
            except Exception:
                logger.warning("SOT v2.0 init failed, disabling", exc_info=True)

    def _init_cascade_router(self) -> None:
        mem = self._mem
        mem.cascade_router = None
        if getattr(mem.config, "cascade_enabled", False):
            try:
                from rtmdk.production.cascade_router import AdaptiveCascadeRouter
                mem.cascade_router = AdaptiveCascadeRouter(
                    causal_threshold=getattr(
                        mem.config, "cascade_causal_threshold", 0.3),
                    factual_threshold=getattr(
                        mem.config, "cascade_factual_threshold", 0.3),
                )
                logger.info("Adaptive cascade router enabled")
            except Exception:
                logger.warning(
                    "Cascade router init failed, disabling", exc_info=True)

    def _init_embedder_circuit_breaker(self) -> None:
        mem = self._mem
        if getattr(mem.config, "embedder_circuit_breaker_enabled", True):
            from rtmdk.support.circuit_breaker import CircuitBreaker
            original_embedder = mem.embedder
            dim = getattr(mem.config, "embedding_dim", 384)
            cb = CircuitBreaker(
                "embedder",
                failure_threshold=getattr(
                    mem.config, "embedder_cb_threshold", 3),
                recovery_timeout=getattr(
                    mem.config, "embedder_cb_recovery", 30.0),
                default=np.zeros(dim, dtype=np.float32),
            )

            def _safe_embed(text):
                return cb.call(original_embedder, text)
            mem.embedder = _safe_embed
            mem._embedder_cb = cb
            logger.info("Embedder circuit breaker enabled")

    def _validate_config(self) -> None:
        for warning in self._mem.config.validate():
            logger.warning("Config validation: %s", warning)
