"""rtmdk/production/__init__.py — Production modules for RTMDK."""

# Direct imports for commonly used classes
from .query_cache import QueryCache
from .bm25_fallback import BM25FallbackRetriever
from .backup_restore import BackupManager
from .export import MemoryExporter
from .analytics import MemoryAnalytics
from .analytics_engine import AnalyticsStore, EventType
from .health_monitor import HealthMonitor
from .events import EventSystem
from .tagging import TaggingSystem
from .rate_limiter import RateLimiter
from .smart_pruning import SmartPruner
from .session_persistence import SessionPersistence
from .import_pipeline import ImportPipeline
from .feedback_loop import FeedbackLoop
from .context_optimizer import ContextOptimizer
from .memory_refresh import MemoryRefresh
from .embedding_cache import EmbeddingCache

# Advanced retrieval classes (lazy to avoid heavy imports)
_advanced_retrieval = None
def __getattr__(name):
    global _advanced_retrieval
    _lazy_map = {
        "RTMDKRetriever": "langchain_adapter",
        "RTMDKChatMessageHistory": "langchain_adapter",
        "RTMDKVectorStore": "langchain_adapter",
        "RTMDKDocument": "langchain_adapter",
        "HybridRetriever": "advanced_retrieval",
        "ConfidenceAwareFallback": "advanced_retrieval",
        "QueryExpander": "advanced_retrieval",
        "AdaptiveDepthRetriever": "advanced_retrieval",
        "TemporalDecayLearner": "advanced_retrieval",
        "CausalAugmentedRetriever": "advanced_retrieval",
        "MetaRetrievalController": "advanced_retrieval",
        "AdvancedRTMDKRetriever": "advanced_retrieval",
        "OfflineDreamer": "offline_dreamer",
        "MultiTenantRouter": "multi_tenant",
        "ABTesting": "ab_testing",
        "CircuitBreaker": "circuit_breaker",
        "OnboardingWizard": "onboarding",
        "ConversationReplay": "replay",
        "MemoryDiff": "memory_diff",
        "LLMEvaluator": "llm_eval",
    }
    if name in _lazy_map:
        module = __import__(f"rtmdk.production.{_lazy_map[name]}", fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QueryCache",
    "BM25FallbackRetriever",
    "BackupManager",
    "MemoryExporter",
    "MemoryAnalytics",
    "AnalyticsStore", "EventType",
    "HealthMonitor",
    "EventSystem",
    "TaggingSystem",
    "RateLimiter",
    "SmartPruner",
    "SessionPersistence",
    "ImportPipeline",
    "FeedbackLoop",
    "ContextOptimizer",
    "MemoryRefresh",
    "EmbeddingCache",
    # Lazy
    "HybridRetriever", "ConfidenceAwareFallback", "QueryExpander",
    "AdaptiveDepthRetriever", "TemporalDecayLearner", "CausalAugmentedRetriever",
    "MetaRetrievalController", "AdvancedRTMDKRetriever",
    "OfflineDreamer", "MultiTenantRouter", "ABTesting",
    "CircuitBreaker", "OnboardingWizard", "ConversationReplay",
    "MemoryDiff", "LLMEvaluator",
    # LangChain integration (lazy to avoid hard dependency)
    "RTMDKRetriever", "RTMDKChatMessageHistory", "RTMDKVectorStore", "RTMDKDocument",
]
