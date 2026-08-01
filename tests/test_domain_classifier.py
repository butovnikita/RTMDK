"""Tests for rtmdk.utils.domain_classifier."""

from rtmdk.utils.domain_classifier import detect_domain, detect_domain_batch, get_domain_stats


class TestDomainClassifier:
    def test_detect_domain_it(self):
        domain, subdomain, topic = detect_domain("How to create a SQL index?")
        assert domain == "IT"
        assert subdomain == "Databases"
        assert topic == "SQL"

    def test_detect_domain_general(self):
        domain, subdomain, topic = detect_domain("asdfghjkl qwerty")
        assert domain == "general"

    def test_detect_domain_batch(self):
        results = detect_domain_batch(["contract law", "stock market trading"])
        assert results[0][0] == "Law"
        assert results[1][0] == "Finance"

    def test_get_domain_stats(self):
        from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
        import numpy as np

        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=lambda t: np.random.randn(768).astype(np.float32))
        mem.save_context({"input": "contract law", "session_id": "s1"}, {"output": ""})
        mem.save_context({"input": "buy stocks", "session_id": "s1"}, {"output": ""})
        stats = get_domain_stats(mem.field)
        assert stats["total"] == 2
        assert "domains" in stats
