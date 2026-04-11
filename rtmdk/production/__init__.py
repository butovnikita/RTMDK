"""rtmdk/production/__init__.py — Production modules for RTMDK.

Lazy imports to avoid circular dependency issues.
"""

__all__ = []

def __getattr__(name):
    """Lazy import on attribute access."""
    import importlib
    _lazy_map = {
        "QueryCache": ".query_cache",
        "BM25FallbackRetriever": ".bm25_fallback",
        "HybridRetriever": ".advanced_retrieval",
        "ConfidenceAwareFallback": ".advanced_retrieval",
        "QueryExpander": ".advanced_retrieval",
        "AdaptiveDepthRetriever": ".advanced_retrieval",
        "TemporalDecayLearner": ".advanced_retrieval",
        "CausalAugmentedRetriever": ".advanced_retrieval",
        "MetaRetrievalController": ".advanced_retrieval",
        "AdvancedRTMDKRetriever": ".advanced_retrieval",
    }
    if name in _lazy_map:
        module = importlib.import_module(_lazy_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
