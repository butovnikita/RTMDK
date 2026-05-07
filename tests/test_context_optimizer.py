"""Tests for rtmdk.production.context_optimizer."""

from rtmdk.production.context_optimizer import ContextOptimizer, estimate_tokens


class TestContextOptimizer:
    def test_estimate_tokens_english(self):
        assert estimate_tokens("hello world") == 2  # 11 chars // 4

    def test_estimate_tokens_cyrillic(self):
        assert estimate_tokens("привет") == 1  # 6 chars / 6

    def test_optimize_empty(self):
        opt = ContextOptimizer()
        assert opt.optimize("") == ""
        assert opt.optimize("No relevant memory.") == "No relevant memory."

    def test_optimize_fallback_lines(self):
        opt = ContextOptimizer(max_tokens=100)
        raw = "Line one\nLine two\nLine three"
        result = opt.optimize(raw)
        assert "Line one" in result

    def test_optimize_structured(self):
        opt = ContextOptimizer(max_tokens=50, min_tokens=5)
        raw = "[ATTN:0.9][SAL:0.8][TIER:semantic] Important fact\n[ATTN:0.5] Less important"
        result = opt.optimize(raw)
        assert "Important fact" in result
        assert "0.90" in result or "0.9" in result

    def test_deduplicate(self):
        opt = ContextOptimizer(deduplicate=True, max_tokens=200)
        raw = "Unique fact one\nUnique fact one\nAnother fact"
        result = opt.optimize(raw)
        # Deduplication should reduce repeated lines
        assert result.count("Unique fact one") == 1

    def test_min_tokens_enforcement(self):
        opt = ContextOptimizer(min_tokens=500, max_tokens=1000)
        raw = "Short text"
        result = opt.optimize(raw)
        assert "Short text" in result

    def test_get_stats(self):
        opt = ContextOptimizer(model="gpt-4o")
        stats = opt.get_stats()
        assert stats["model"] == "gpt-4o"
        assert stats["model_context_limit"] == 128000
