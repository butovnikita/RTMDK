"""Unit tests for BacklogModulesInitializer."""

from unittest.mock import patch

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.backlog_modules_initializer import BacklogModulesInitializer


class _MockMemory:
    def __init__(self, **cfg_overrides):
        self.config = RTMDKConfig(**cfg_overrides)
        self.embedder = lambda x: [0.1] * self.config.latent_dim
        self.field = None


def _make_bmi(**cfg_overrides):
    mem = _MockMemory(**cfg_overrides)
    return BacklogModulesInitializer(mem), mem


class TestBacklogModulesInitializer:
    def test_initialize_default_state(self):
        bmi, mem = _make_bmi()
        bmi.initialize()
        # engram cache is enabled by default in RTMDKConfig.sot
        assert mem.engram_cache is not None
        assert mem.replication_manager is None
        assert mem._distributed_lock is None
        assert mem.metrics is None
        assert mem._sentence_reranker is None
        assert mem._query_decomposer is None
        assert mem._feedback_loop is None
        assert mem._result_explainer is None
        assert mem._query_rewriter is None
        assert mem._intent_classifier is None
        assert mem._rollback_manager is not None
        assert mem._poison_detector is not None

    def test_replication_enabled_with_peers(self):
        with patch("rtmdk.production.replication.ReplicationManager") as MockRepl:
            bmi, mem = _make_bmi(
                replication_peers=["http://peer1:8080"],
                replication_node_id="node-1",
            )
            bmi._init_replication()
            MockRepl.assert_called_once()
            assert mem.replication_manager is MockRepl.return_value

    def test_engram_cache_default_enabled(self):
        bmi, mem = _make_bmi()
        bmi._init_engram_cache()
        assert mem.engram_cache is not None
        assert mem.engram_cache._max_hot == 10_000

    def test_engram_cache_disabled(self):
        bmi, mem = _make_bmi()
        mem.config.sot.engram_cache_enabled = False
        bmi._init_engram_cache()
        assert mem.engram_cache is None

    def test_distributed_lock_with_path(self):
        with patch("rtmdk.memory.backlog_modules_initializer.DistributedLock") as MockLock:
            bmi, mem = _make_bmi()
            mem.config.sot.distributed_lock_path = "/tmp/lock"
            mem.config.sot.distributed_lock_backend = "file"
            mem.config.sot.distributed_lock_redis_url = None
            bmi._init_distributed_lock()
            MockLock.assert_called_once_with("/tmp/lock", backend="file", redis_url=None)
            assert mem._distributed_lock is MockLock.return_value

    def test_observability_with_alert_handlers(self):
        with (
            patch("rtmdk.memory.observability.WebhookAlertHandler") as MockWebhook,
            patch("rtmdk.memory.observability.SlackAlertHandler") as MockSlack,
        ):
            bmi, mem = _make_bmi()
            mem.config.sot.observability_enabled = True
            mem.config.sot.alert_webhook_url = "http://hook"
            mem.config.sot.alert_slack_url = "http://slack"
            mem.config.sot.alert_pagerduty_key = None
            bmi._init_observability()
            assert mem.metrics is not None
            MockWebhook.assert_called_once_with("http://hook")
            MockSlack.assert_called_once_with("http://slack")

    def test_rag_quality_sentence_reranker(self):
        with patch("rtmdk.memory.backlog_modules_initializer.SentenceReranker") as MockReranker:
            bmi, mem = _make_bmi()
            mem.config.sot.sentence_reranker_enabled = True
            mem.config.sot.query_decomposition_enabled = False
            mem.config.sot.feedback_loop_enabled = False
            bmi._init_rag_quality()
            MockReranker.assert_called_once_with(mem.embedder)
            assert mem._sentence_reranker is MockReranker.return_value
            assert mem._query_decomposer is None
            assert mem._feedback_loop is None

    def test_explainability_result_explainer(self):
        with patch("rtmdk.memory.explainability.ResultExplainer") as MockExplainer:
            bmi, mem = _make_bmi()
            mem.config.sot.result_explainability_enabled = True
            mem.config.sot.query_rewrite_enabled = False
            mem.config.sot.query_intent_classification_enabled = False
            bmi._init_explainability()
            MockExplainer.assert_called_once()
            assert mem._result_explainer is MockExplainer.return_value
