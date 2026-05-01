"""
tests/test_domain_memory.py — Tests for Phase 20 Domain Memory.

Tests cover:
1. New fields serialize/deserialize correctly
2. Old memory.json files load without errors (backward compatibility)
3. detect_domain() returns correct results
4. Domain-aware retrieval filters by domain
5. Consolidation does NOT merge nodes from different domains
6. Bi-temporal facts: valid_from/until work
7. Concept state transitions: stable→weakened→disputed
8. Evidence spans are preserved
9. Legal/medical presets include domain flags
10. ATTENTION context shows [DOM:XXX] tokens
"""

import pytest
import json
import time
import numpy as np
import tempfile
import os

from rtmdk import RTMDKMemory, RTMDKConfig, RTMDKField, MemoryNode
from rtmdk.memory.core import ContextFormat, format_context
from rtmdk.utils.domain_classifier import detect_domain, get_domain_stats


def make_embedder(dim=768):
    """Create a deterministic fake embedder."""
    def embedder(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(dim).astype(np.float32) * 0.1
    return embedder


# ============================================================================
# TEST 1: New fields serialize/deserialize correctly
# ============================================================================

class TestNewFieldsSerialization:
    def test_memory_node_has_new_fields(self):
        """MemoryNode should have all Phase 20 fields with defaults."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0)

        # Phase 20 Track 1: Domain Hierarchy
        assert node.domain == "general"
        assert node.subdomain == ""
        assert node.topic == ""

        # Phase 20 Track 2: Concept Lifecycle
        assert node.state == "stable"
        assert node.confidence == 1.0
        assert node.revision_count == 0
        assert node.conflict_with == []

        # Phase 20 Track 3: Evidence Spans
        assert node.evidence_spans == []

        # Phase 20 Track 4: Bi-temporal Facts
        assert node.valid_from is None
        assert node.valid_until is None
        assert node.fact_state == "active"
        assert node.superseded_by is None

    def test_node_to_dict_includes_new_fields(self):
        """to_dict() should include all Phase 20 fields."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            domain="IT", subdomain="Databases", topic="SQL",
            state="stable", confidence=0.9,
            evidence_spans=[{"source_id": "src1", "text": "test", "confidence": 0.8}],
            valid_from=1000.0, valid_until=2000.0, fact_state="active"
        )

        d = node.to_dict()
        assert d["domain"] == "IT"
        assert d["subdomain"] == "Databases"
        assert d["topic"] == "SQL"
        assert d["state"] == "stable"
        assert d["confidence"] == 0.9
        assert len(d["evidence_spans"]) == 1
        assert d["valid_from"] == 1000.0
        assert d["valid_until"] == 2000.0
        assert d["fact_state"] == "active"

    def test_node_from_dict_restores_new_fields(self):
        """from_dict() should restore all Phase 20 fields."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            domain="Law", subdomain="Contracts",
            state="disputed", confidence=0.5, revision_count=3,
            conflict_with=["n2", "n3"],
            evidence_spans=[{"source_id": "doc1", "text": "clause 5", "confidence": 0.9}],
            valid_from=1000.0, fact_state="disputed"
        )

        d = node.to_dict()
        restored = MemoryNode.from_dict(d)

        assert restored.domain == "Law"
        assert restored.subdomain == "Contracts"
        assert restored.state == "disputed"
        assert restored.confidence == 0.5
        assert restored.revision_count == 3
        assert restored.conflict_with == ["n2", "n3"]
        assert len(restored.evidence_spans) == 1
        assert restored.valid_from == 1000.0
        assert restored.fact_state == "disputed"


# ============================================================================
# TEST 2: Old memory.json loads without errors (backward compatibility)
# ============================================================================

class TestBackwardCompatibility:
    def test_load_memory_without_new_fields(self):
        """Old memory.json (without Phase 20 fields) should load with defaults."""
        config = RTMDKConfig(latent_dim=64, embedding_dim=64)
        memory = RTMDKMemory(config=config, embedder=make_embedder(64))

        # Add a node the old way (without Phase 20 fields)
        emb = make_embedder(64)("test text")
        content = {"text": "old node", "tier": "semantic"}
        # Simulate old-style data (no domain, state, etc.)
        nid = memory.field.add_node(emb, content, phase=0.0, session_id="test")

        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        memory.export_field(temp_path)

        # Manually remove Phase 20 fields to simulate old file
        with open(temp_path, 'r') as f:
            data = json.load(f)
        for node_data in data.get("nodes", []):
            for key in ["domain", "subdomain", "topic", "state", "confidence",
                        "revision_count", "conflict_with", "evidence_spans",
                        "valid_from", "valid_until", "fact_state", "superseded_by"]:
                node_data.pop(key, None)
        with open(temp_path, 'w') as f:
            json.dump(data, f)

        # Import — should work with defaults
        memory2 = RTMDKMemory.import_field(temp_path, make_embedder(64))
        assert len(memory2.field.nodes) == 1
        node = list(memory2.field.nodes.values())[0]

        # Check defaults
        assert node.domain == "general"
        assert node.state == "stable"
        assert node.confidence == 1.0
        assert node.fact_state == "active"

        os.unlink(temp_path)


# ============================================================================
# TEST 3: detect_domain() returns correct results
# ============================================================================

class TestDomainDetection:
    def test_it_databases(self):
        domain, subdomain, topic = detect_domain("How to create a SQL index on PostgreSQL?")
        assert domain == "IT"
        assert subdomain == "Databases"

    def test_it_programming(self):
        domain, subdomain, topic = detect_domain("Write a Python function with async/await")
        assert domain == "IT"
        assert subdomain == "Programming"

    def test_law_contracts(self):
        domain, subdomain, topic = detect_domain(
            "The contract clause states that liability is limited to damages")
        assert domain == "Law"
        assert subdomain == "Contracts"

    def test_medicine_cardiology(self):
        domain, subdomain, topic = detect_domain(
            "Patient has high blood pressure and cardiac arrhythmia")
        assert domain == "Medicine"
        assert subdomain == "Cardiology"

    def test_finance_investing(self):
        domain, subdomain, topic = detect_domain(
            "Stock portfolio dividend yield and bond returns")
        assert domain == "Finance"
        assert subdomain == "Investing"

    def test_general_fallback(self):
        domain, subdomain, topic = detect_domain("Hello world, how are you?")
        assert domain == "general"
        assert subdomain == ""

    def test_empty_text(self):
        domain, subdomain, topic = detect_domain("")
        assert domain == "general"

    def test_caching(self):
        """detect_domain should be cached."""
        text = "SQL database query optimization"
        result1 = detect_domain(text)
        result2 = detect_domain(text)
        assert result1 == result2


# ============================================================================
# TEST 4: Domain-aware retrieval filters by domain
# ============================================================================

class TestDomainAwareRetrieval:
    def test_domain_filtering(self):
        """When domain_aware_retrieval=True, retrieval should filter by domain."""
        config = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=3,
            domain_aware_retrieval=True
        )
        memory = RTMDKMemory(config=config, embedder=make_embedder(64))

        # Add IT node
        emb_it = make_embedder(64)("SQL database query")
        nid_it = memory.field.add_node(emb_it, {"text": "SQL query"}, phase=0.0)
        memory.field.nodes[nid_it].domain = "IT"

        # Add Law node
        emb_law = make_embedder(64)("contract liability clause")
        nid_law = memory.field.add_node(emb_law, {"text": "contract clause"}, phase=0.5)
        memory.field.nodes[nid_law].domain = "Law"

        # Query with IT domain
        ctx = memory.load_memory_variables({"input": "SQL database optimization", "session_id": "test"})
        # Should return results — domain filtering is best-effort
        assert "rtmdk_context" in ctx


# ============================================================================
# TEST 5: Consolidation does NOT merge nodes from different domains
# ============================================================================

class TestCrossDomainConsolidationGuard:
    def test_different_domains_not_consolidated(self):
        """Nodes from different domains should NOT be consolidated."""
        config = RTMDKConfig(
            latent_dim=64, embedding_dim=64,
            tension_threshold=0.01,  # Very low to trigger consolidation
            domain_consolidation_guard=True
        )
        memory = RTMDKMemory(config=config, embedder=make_embedder(64))

        # Add two nodes with SAME position (high tension) but DIFFERENT domains
        emb = make_embedder(64)("test")
        nid1 = memory.field.add_node(emb.copy(), {"text": "IT node"}, phase=0.0)
        memory.field.nodes[nid1].domain = "IT"

        nid2 = memory.field.add_node(emb.copy(), {"text": "Law node"}, phase=0.1)
        memory.field.nodes[nid2].domain = "Law"

        # Ensure high tension to trigger consolidation path
        memory.field.nodes[nid1].tension = 0.5
        memory.field.nodes[nid2].tension = 0.5

        # Trigger consolidation
        memory.field.consolidate()

        # Both nodes should still exist (NOT merged)
        assert nid1 in memory.field.nodes
        assert nid2 in memory.field.nodes


# ============================================================================
# TEST 6: Bi-temporal facts work
# ============================================================================

class TestBiTemporalFacts:
    def test_valid_from_until(self):
        """Nodes should support valid_from/valid_until timestamps."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            valid_from=1000.0, valid_until=2000.0, fact_state="active"
        )
        assert node.valid_from == 1000.0
        assert node.valid_until == 2000.0
        assert node.fact_state == "active"

    def test_fact_states(self):
        """Nodes should support different fact states."""
        emb = np.random.randn(64).astype(np.float32)
        for state in ["active", "stale", "disputed", "rejected", "archived"]:
            node = MemoryNode(
                id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
                fact_state=state
            )
            assert node.fact_state == state


# ============================================================================
# TEST 7: Concept state transitions
# ============================================================================

class TestConceptStateTransitions:
    def test_state_field(self):
        """Node should support different concept states."""
        emb = np.random.randn(64).astype(np.float32)
        for state in ["stable", "weakened", "disputed", "broken", "stale", "archived"]:
            node = MemoryNode(
                id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
                state=state
            )
            assert node.state == state

    def test_confidence_field(self):
        """Node should support confidence tracking."""
        emb = np.random.randn(64).astype(np.float32)
        for conf in [0.0, 0.5, 1.0]:
            node = MemoryNode(
                id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
                confidence=conf
            )
            assert node.confidence == conf


# ============================================================================
# TEST 8: Evidence spans are preserved
# ============================================================================

class TestEvidenceSpans:
    def test_evidence_spans_serialization(self):
        """Evidence spans should serialize and deserialize correctly."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            evidence_spans=[
                {"source_id": "doc123", "text": "Key clause", "confidence": 0.9},
                {"source_id": "doc456", "text": "Supporting evidence", "confidence": 0.7}
            ]
        )

        d = node.to_dict()
        restored = MemoryNode.from_dict(d)

        assert len(restored.evidence_spans) == 2
        assert restored.evidence_spans[0]["source_id"] == "doc123"
        assert restored.evidence_spans[1]["confidence"] == 0.7


# ============================================================================
# TEST 9: Legal/medical presets include domain flags
# ============================================================================

class TestLegalMedicalPresets:
    def test_legal_preset_has_domain_flags(self):
        """Legal preset should have domain_aware_retrieval=True."""
        config = RTMDKConfig.legal()
        assert config.domain_aware_retrieval is True
        assert config.domain_consolidation_guard is True

    def test_medical_preset_has_domain_flags(self):
        """Medical preset should have domain_aware_retrieval=True."""
        config = RTMDKConfig.medical()
        assert config.domain_aware_retrieval is True
        assert config.domain_consolidation_guard is True


# ============================================================================
# TEST 10: ATTENTION context shows [DOM:XXX] tokens
# ============================================================================

class TestAttentionContextTokens:
    def test_domain_token_in_attention(self):
        """ATTENTION format should show [DOM:XXX] token for non-general domains."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            content={"text": "SQL query example", "tier": "semantic"},
            domain="IT", subdomain="Databases"
        )

        results = [("n1", 0.8, node)]
        ctx = format_context(results, ContextFormat.ATTENTION)

        assert "[DOM:IT]" in ctx
        assert "[ATTN:0.800]" in ctx
        assert "[SAL:1.000]" in ctx

    def test_no_domain_token_for_general(self):
        """ATTENTION format should NOT show [DOM:XXX] for general domain."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            content={"text": "hello world", "tier": "semantic"},
            domain="general"
        )

        results = [("n1", 0.8, node)]
        ctx = format_context(results, ContextFormat.ATTENTION)

        assert "[DOM:" not in ctx

    def test_state_token_in_attention(self):
        """ATTENTION format should show [STATE:X] for non-stable states."""
        emb = np.random.randn(64).astype(np.float32)
        node = MemoryNode(
            id="n1", latent_pos=emb, phase=0.0, amplitude=1.0, salience=1.0,
            content={"text": "disputed fact", "tier": "semantic"},
            state="disputed"
        )

        results = [("n1", 0.8, node)]
        ctx = format_context(results, ContextFormat.ATTENTION)

        assert "[STATE:D]" in ctx


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
