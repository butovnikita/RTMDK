"""BacklogModulesInitializer — extracts _init_backlog_modules from core.py."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rtmdk.memory.engram_cache import EngramEmbeddingCache
from rtmdk.memory.distributed_lock import DistributedLock
from rtmdk.memory.observability import MemoryMetrics, AlertRule
from rtmdk.memory.rag_quality import SentenceReranker, QueryDecomposer, FeedbackLoop

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rtmdk.memory.core import RTMDKMemory


class BacklogModulesInitializer:
    """Encapsulates backlog module initialization for RTMDKMemory."""

    def __init__(self, memory: "RTMDKMemory") -> None:
        self._mem = memory

    def initialize(self) -> None:
        """Initialize production-grade backlog modules.

        Wires replication, engram cache, distributed lock, observability,
        RAG quality, explainability, safety, and security subsystems.
        """
        self._init_replication()
        self._init_engram_cache()
        self._init_distributed_lock()
        self._init_observability()
        self._init_rag_quality()
        self._init_explainability()
        self._init_safety()

    def _init_replication(self) -> None:
        mem = self._mem
        peers = mem.config.replication_peers
        if peers:
            try:
                from rtmdk.production.replication import ReplicationManager
                rm = ReplicationManager(
                    peers=peers,
                    node_id=mem.config.replication_node_id,
                    wal_path=mem.config.replication_wal_path,
                )
                object.__setattr__(mem, "replication_manager", rm)
                logger.info("ReplicationManager enabled with peers: %s", peers)
            except Exception:
                logger.warning(
                    "ReplicationManager init failed, disabling", exc_info=True)
                object.__setattr__(mem, "replication_manager", None)
        else:
            object.__setattr__(mem, "replication_manager", None)

    def _init_engram_cache(self) -> None:
        mem = self._mem
        sot_cfg = getattr(mem.config, "sot", None)
        if sot_cfg and getattr(sot_cfg, "engram_cache_enabled", True):
            mem.engram_cache = EngramEmbeddingCache(
                max_hot=getattr(sot_cfg, "engram_cache_max_hot", 10_000),
                max_warm=getattr(sot_cfg, "engram_cache_max_warm", 90_000))
        else:
            mem.engram_cache = None

    def _init_distributed_lock(self) -> None:
        mem = self._mem
        sot_cfg = getattr(mem.config, "sot", None)
        lock_path = getattr(
            sot_cfg, "distributed_lock_path", None) if sot_cfg else None
        lock_backend = getattr(sot_cfg, "distributed_lock_backend", "file")
        redis_url = getattr(sot_cfg, "distributed_lock_redis_url", None)
        if lock_path:
            mem._distributed_lock = DistributedLock(
                lock_path, backend=lock_backend, redis_url=redis_url)
        else:
            mem._distributed_lock = None

    def _init_observability(self) -> None:
        mem = self._mem
        sot_cfg = getattr(mem.config, "sot", None)
        if sot_cfg and getattr(sot_cfg, "observability_enabled", False):
            mem.metrics = MemoryMetrics()
            mem.metrics.add_alert_rule(
                AlertRule("high_latency", "query_p99", threshold=100.0))
            mem.metrics.add_alert_rule(
                AlertRule("low_cache", "cache_hit_ratio",
                         threshold=0.3, comparison="lt"))
            webhook_url = getattr(sot_cfg, "alert_webhook_url", None)
            slack_url = getattr(sot_cfg, "alert_slack_url", None)
            pagerduty_key = getattr(sot_cfg, "alert_pagerduty_key", None)
            if webhook_url:
                from rtmdk.memory.observability import WebhookAlertHandler
                mem.metrics.add_alert_handler(
                    WebhookAlertHandler(webhook_url))
            if slack_url:
                from rtmdk.memory.observability import SlackAlertHandler
                mem.metrics.add_alert_handler(SlackAlertHandler(slack_url))
            if pagerduty_key:
                from rtmdk.memory.observability import PagerDutyAlertHandler
                mem.metrics.add_alert_handler(
                    PagerDutyAlertHandler(pagerduty_key))
        else:
            mem.metrics = None

    def _init_rag_quality(self) -> None:
        mem = self._mem
        sot_cfg = getattr(mem.config, "sot", None)
        mem._sentence_reranker = None
        mem._query_decomposer = None
        mem._feedback_loop = None
        if sot_cfg and getattr(sot_cfg, "sentence_reranker_enabled", False):
            mem._sentence_reranker = SentenceReranker(mem.embedder)
        if sot_cfg and getattr(sot_cfg, "query_decomposition_enabled", False):
            llm_client = getattr(mem, "_llm_client", None)
            mem._query_decomposer = QueryDecomposer(llm_client=llm_client)
        if sot_cfg and getattr(sot_cfg, "feedback_loop_enabled", False):
            fb_path = getattr(sot_cfg, "feedback_loop_persist_path", None)
            mem._feedback_loop = FeedbackLoop(
                mem.embedder, persist_path=fb_path)
            mem._feedback_loop.load()

    def _init_explainability(self) -> None:
        mem = self._mem
        sot_cfg = getattr(mem.config, "sot", None)
        mem._result_explainer = None
        mem._query_rewriter = None
        mem._intent_classifier = None
        if sot_cfg and getattr(sot_cfg, "result_explainability_enabled", False):
            from rtmdk.memory.explainability import ResultExplainer
            mem._result_explainer = ResultExplainer()
        if sot_cfg and getattr(sot_cfg, "query_rewrite_enabled", False):
            from rtmdk.memory.explainability import QueryRewriter
            llm_client = getattr(mem, "_llm_client", None)
            mem._query_rewriter = QueryRewriter(
                embedder=mem.embedder, llm_client=llm_client)
        if sot_cfg and getattr(sot_cfg, "query_intent_classification_enabled", False):
            from rtmdk.memory.explainability import QueryIntentClassifier
            llm_client = getattr(mem, "_llm_client", None)
            mem._intent_classifier = QueryIntentClassifier(
                llm_client=llm_client)

    def _init_safety(self) -> None:
        from rtmdk.memory.safety import RollbackManager, PoisonedMemoryDetector
        self._mem._rollback_manager = RollbackManager()
        self._mem._poison_detector = PoisonedMemoryDetector()
