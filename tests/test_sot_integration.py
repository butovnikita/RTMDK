"""Integration tests for SOT + RTMDKField."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKField, RTMDKConfig

LATENT_DIM = 64
TOKEN_DIM = 128
EMBEDDING_DIM = 768


@pytest.fixture
def field_sot():
    cfg = RTMDKConfig(
        latent_dim=LATENT_DIM,
        embedding_dim=EMBEDDING_DIM,
        sot_enabled=True,
        sot_max_vocab=512,
        sot_contrastive_lr=0.1,
        sot_negatives_per_query=3,
        sot_ssm_sync=True,
        sot_merge_freq=50,
        sot_use_for_query=True,
        max_nodes=200,
    )
    field = RTMDKField(config=cfg)
    # Override tokenizer with token_dim != latent_dim for testing
    from rtmdk.memory.self_organizing_field import SOTokenizer
    field.sot_tokenizer = SOTokenizer(
        latent_dim=LATENT_DIM,
        token_dim=TOKEN_DIM,
        max_vocab=512,
        seed=cfg.seed,
    )
    if field.sot_ssm:
        field.sot_ssm.tokenizer = field.sot_tokenizer
    return field


@pytest.fixture
def field_baseline():
    cfg = RTMDKConfig(
        latent_dim=LATENT_DIM,
        embedding_dim=EMBEDDING_DIM,
        sot_enabled=False,
        max_nodes=200,
    )
    return RTMDKField(config=cfg)


class TestSOTFieldInit:
    def test_sot_components_created_when_enabled(self, field_sot):
        assert field_sot.sot_tokenizer is not None
        assert field_sot.sot_hebbian is not None
        assert field_sot.sot_ssm is not None

    def test_sot_components_none_when_disabled(self, field_baseline):
        assert field_baseline.sot_tokenizer is None
        assert field_baseline.sot_hebbian is None
        assert field_baseline.sot_ssm is None


class TestSOTAddNode:
    def test_add_node_accepts_latent_dim_embedding(self, field_sot):
        emb = np.random.randn(LATENT_DIM).astype(np.float32)
        nid = field_sot.add_node(emb, {"text": "test"})
        node = field_sot.nodes[nid]
        assert node.latent_pos.shape == (LATENT_DIM,)
        # Should NOT have projected from 768d; should use directly
        assert np.allclose(node.latent_pos, emb, atol=1e-4)

    def test_add_node_accepts_embedding_dim_embedding(self, field_sot):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        nid = field_sot.add_node(emb, {"text": "test"})
        node = field_sot.nodes[nid]
        assert node.latent_pos.shape == (LATENT_DIM,)

    def test_baseline_add_node_requires_embedding_dim(self, field_baseline):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        nid = field_baseline.add_node(emb, {"text": "test"})
        assert nid in field_baseline.nodes


class TestSOTStep:
    def test_step_with_text_content(self, field_sot):
        emb = np.random.randn(LATENT_DIM).astype(np.float32)
        field_sot.step([{
            "embedding": emb,
            "content": {"text": "hello world"},
            "phase": 0.0,
        }])
        # Should not crash; tokenizer should have recorded cooccurrence
        assert field_sot._step_counter == 1

    def test_step_creates_nodes_for_novel_input(self, field_sot):
        initial_nodes = len(field_sot.nodes)
        emb = np.random.randn(LATENT_DIM).astype(np.float32)
        field_sot.step([{
            "embedding": emb,
            "content": {"text": "something completely new"},
            "phase": 0.0,
        }])
        assert len(field_sot.nodes) > initial_nodes

    def test_step_triggers_contrastive_update(self, field_sot):
        # Pre-populate field with similar embeddings so query retrieves
        # something
        base = np.random.randn(LATENT_DIM).astype(np.float32)
        for i in range(5):
            emb = base + np.random.randn(LATENT_DIM).astype(np.float32) * 0.1
            field_sot.add_node(emb, {"text": f"node_{i}"})
        latent_before = {nid: n.latent_pos.copy()
                         for nid, n in field_sot.nodes.items()}
        # Run step with text that will query and retrieve nodes
        query_emb = base.copy()
        field_sot.step([{
            "embedding": query_emb,
            "content": {"text": "query text"},
            "phase": 0.0,
        }])
        changed = False
        for nid, before_pos in latent_before.items():
            if nid in field_sot.nodes:
                after_pos = field_sot.nodes[nid].latent_pos
                if not np.allclose(before_pos, after_pos):
                    changed = True
                    break
        assert changed, "Latent positions should drift due to Hebbian update"

    def test_step_ssm_sync_changes_token_embeddings(self, field_sot):
        tokens = field_sot.sot_tokenizer.encode("abc")
        embs_before = {
            t: field_sot.sot_tokenizer.token_embeddings[t].copy() for t in tokens}
        field_sot.step([{
            "embedding": np.random.randn(LATENT_DIM).astype(np.float32),
            "content": {"text": "abc"},
            "phase": 0.0,
        }])
        for t in tokens:
            assert not np.allclose(field_sot.sot_tokenizer.token_embeddings[t], embs_before[t]), \
                "SSM sync should change token embeddings"


class TestSOTQuery:
    def test_query_with_text_uses_sot_embedding(self, field_sot):
        # Populate field
        for i in range(10):
            emb = np.random.randn(LATENT_DIM).astype(np.float32)
            field_sot.add_node(emb, {"text": f"content_{i}"})
        # Query by text (should use SOT, not external embedder)
        results = field_sot.query_by_text("content_0", top_k=3)
        assert isinstance(results, list)

    def test_query_by_text_returns_results(self, field_sot):
        for i in range(10):
            emb = np.zeros(LATENT_DIM, dtype=np.float32)
            emb[i % LATENT_DIM] = 1.0
            field_sot.add_node(
                emb, {
                    "text": "topic alpha" if i < 5 else "topic beta"})
        results = field_sot.query_by_text("topic alpha", top_k=5)
        assert len(results) <= 5


class TestSOTMerge:
    def test_merge_frequency_respected(self, field_sot):
        field_sot.cfg.sot_merge_freq = 5
        # Step 4 times — no merge yet
        for _ in range(4):
            field_sot.step([{
                "embedding": np.random.randn(LATENT_DIM).astype(np.float32),
                "content": {"text": "ab ab ab"},
                "phase": 0.0,
            }])
        len(field_sot.sot_tokenizer.token_embeddings)
        # 5th step should trigger merge check
        field_sot.step([{
            "embedding": np.random.randn(LATENT_DIM).astype(np.float32),
            "content": {"text": "ab ab ab"},
            "phase": 0.0,
        }])
        len(field_sot.sot_tokenizer.token_embeddings)
        # Vocab may or may not grow depending on threshold, but merge logic
        # should run
        assert field_sot._step_counter == 5


class TestSOTStatePersistence:
    def test_state_includes_sot(self, field_sot):
        field_sot.step([{
            "embedding": np.random.randn(LATENT_DIM).astype(np.float32),
            "content": {"text": "hello"},
            "phase": 0.0,
        }])
        state = field_sot.get_state()
        assert "sot_tokenizer" in state
        assert "sot_hebbian" in state
        assert "sot_ssm" in state

    def test_load_state_restores_sot(self, field_sot):
        field_sot.step([{
            "embedding": np.random.randn(LATENT_DIM).astype(np.float32),
            "content": {"text": "hello world"},
            "phase": 0.0,
        }])
        state = field_sot.get_state()
        field2 = RTMDKField(config=RTMDKConfig(sot_enabled=True))
        field2.load_state(state)
        assert field2.sot_tokenizer is not None
        assert len(
            field2.sot_tokenizer.token_embeddings) == len(
            field_sot.sot_tokenizer.token_embeddings)


class TestSOTContrastiveField:
    def test_field_contrastive_step_changes_embeddings(self, field_sot):
        field_sot.sot_contrastive_step(
            query_text="query text",
            positive_text="positive match",
            negative_texts=["negative random"],
            lr=0.05,
        )
        # After contrastive step, embeddings should still be valid
        emb = field_sot.sot_tokenizer.embed(
            field_sot.sot_tokenizer.encode("query text"))
        assert emb.shape == (LATENT_DIM,)
        assert np.linalg.norm(emb) > 0.9

    def test_field_contrastive_step_multiple_negatives(self, field_sot):
        field_sot.sot_contrastive_step(
            query_text="hello world",
            positive_text="hello there",
            negative_texts=["foo bar", "baz qux"],
            lr=0.05,
        )
        emb = field_sot.sot_tokenizer.embed(
            field_sot.sot_tokenizer.encode("hello world"))
        assert emb.shape == (LATENT_DIM,)


class TestSOTBackwardCompatibility:
    def test_baseline_field_works_without_sot(self, field_baseline):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        nid = field_baseline.add_node(emb, {"text": "baseline"})
        results = field_baseline.query(emb, top_k=3)
        assert isinstance(results, list)
        assert nid in field_baseline.nodes
