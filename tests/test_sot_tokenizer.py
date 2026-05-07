"""Tests for Self-Organizing Tokenizer (SOTokenizer)."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.memory.self_organizing_field import SOTokenizer


LATENT_DIM = 64
TOKEN_DIM = 128
MAX_VOCAB = 4096


class TestSOTokenizerInit:
    def test_default_init(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, max_vocab=MAX_VOCAB)
        assert tok.latent_dim == LATENT_DIM
        assert tok.token_dim == LATENT_DIM  # default
        assert tok.max_vocab == MAX_VOCAB
        assert len(tok.token_embeddings) == 256
        assert tok.next_token_id == 256

    def test_token_dim_different(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=MAX_VOCAB)
        assert tok.latent_dim == LATENT_DIM
        assert tok.token_dim == TOKEN_DIM
        assert tok.projection.shape == (TOKEN_DIM, LATENT_DIM)
        assert len(tok.token_embeddings) == 256
        for emb in tok.token_embeddings.values():
            assert emb.shape == (TOKEN_DIM,)

    def test_embeddings_have_correct_shape(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        for tid, emb in tok.token_embeddings.items():
            assert emb.shape == (
                TOKEN_DIM,), f"Token {tid} has wrong shape {emb.shape}"
            assert emb.dtype == np.float32

    def test_byte_tokens_cover_0_to_255(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        assert set(tok.token_embeddings.keys()) == set(range(256))


class TestSOTokenizerEncodeDecode:
    def test_encode_ascii(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        text = "hello"
        tokens = tok.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == len(text.encode("utf-8"))

    def test_encode_decode_roundtrip_ascii(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        text = "Hello world 123"
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        assert decoded == text

    def test_encode_decode_roundtrip_unicode(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        text = "Привет мир 🌍"
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        assert decoded == text

    def test_encode_empty_string(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        assert tok.encode("") == []

    def test_after_merge_uses_merged_tokens(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        a_byte = ord("a")
        b_byte = ord("b")
        for _ in range(10):
            tok.record_cooccurrence([a_byte, b_byte])
        merges = tok.propose_merges(1)
        assert len(merges) > 0
        pair = merges[0]
        tok.merge(pair)
        tokens_before = len(tok.encode("ab"))
        assert tokens_before <= 2


class TestSOTokenizerEmbed:
    def test_embed_returns_float32_vector(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        tokens = tok.encode("abc")
        emb = tok.embed(tokens)
        assert emb.shape == (LATENT_DIM,)
        assert emb.dtype == np.float32

    def test_embed_empty_tokens(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        emb = tok.embed([])
        assert emb.shape == (LATENT_DIM,)
        assert np.allclose(emb, 0.0)

    def test_embed_mean_property(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        tokens = [0, 1, 2]
        emb = tok.embed(tokens)
        manual = np.mean([tok.token_embeddings[t]
                         for t in tokens], axis=0) @ tok.projection
        norm = np.linalg.norm(manual)
        if norm > 0:
            manual = manual / norm
        assert np.allclose(emb, manual, atol=1e-4)

    def test_embed_different_token_dim(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        tokens = [0, 1]
        emb = tok.embed(tokens)
        assert emb.shape == (LATENT_DIM,)
        assert emb.dtype == np.float32
        # Should be normalized
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5 or np.allclose(emb, 0.0)


class TestSOTokenizerCooccurrence:
    def test_record_cooccurrence_increments_counts(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        tok.record_cooccurrence([1, 2, 3])
        tok.record_cooccurrence([1, 2])
        assert tok.cooccurrence[(1, 2)] >= 2

    def test_propose_merges_returns_sorted_pairs(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        tok.record_cooccurrence([10, 20])
        tok.record_cooccurrence([10, 20])
        tok.record_cooccurrence([30, 40])
        merges = tok.propose_merges(2)
        assert len(merges) <= 2
        assert isinstance(merges, list)
        if merges:
            assert isinstance(merges[0], tuple)
            assert len(merges[0]) == 2

    def test_merge_creates_new_token(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok.record_cooccurrence([5, 6])
        initial_len = len(tok.token_embeddings)
        tok.merge((5, 6))
        assert len(tok.token_embeddings) == initial_len + 1
        assert tok.next_token_id == initial_len + 1

    def test_merge_embedding_is_weighted_average(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok.record_cooccurrence([5, 6], weight=3.0)
        tok.record_cooccurrence([5, 7], weight=1.0)
        tok.merge((5, 6))
        new_id = 256
        raw = (3.0 * tok.token_embeddings[5] + 3.0 *
               tok.token_embeddings[6]) / (3.0 + 3.0)
        expected = raw / np.linalg.norm(raw)
        assert np.allclose(tok.token_embeddings[new_id], expected, atol=1e-5)

    def test_merge_respects_max_vocab(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=257)
        tok.merge((0, 1))
        assert len(tok.token_embeddings) == 257
        with pytest.raises(RuntimeError):
            tok.merge((2, 3))

    def test_merge_adds_to_merge_table(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok.merge((1, 2))
        assert tok.merges[(1, 2)] == 256


class TestSOTokenizerState:
    def test_get_state_roundtrip(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok.record_cooccurrence([1, 2])
        tok.merge((1, 2))
        state = tok.get_state()
        tok2 = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok2.load_state(state)
        assert tok2.merges == tok.merges
        assert set(
            tok2.token_embeddings.keys()) == set(
            tok.token_embeddings.keys())
        for k in tok.token_embeddings:
            assert np.allclose(
                tok2.token_embeddings[k],
                tok.token_embeddings[k])
        assert np.allclose(tok2.projection, tok.projection)
        assert tok2.next_token_id == tok.next_token_id

    def test_load_state_restores_encode_behavior(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok.record_cooccurrence([65, 66])  # 'A', 'B'
        tok.merge((65, 66))
        state = tok.get_state()
        tok2 = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        tok2.load_state(state)
        text = "AB"
        assert tok2.encode(text) == tok.encode(text)

    def test_projection_update(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        # Positive pair should move projections closer
        p_before_0 = tok.token_embeddings[0] @ tok.projection
        p_before_1 = tok.token_embeddings[1] @ tok.projection
        dist_before = np.linalg.norm(p_before_0 - p_before_1)
        tok.update_projection(positive_pairs=[(0, 1)], lr=0.1)
        p_after_0 = tok.token_embeddings[0] @ tok.projection
        p_after_1 = tok.token_embeddings[1] @ tok.projection
        dist_after = np.linalg.norm(p_after_0 - p_after_1)
        assert dist_after < dist_before, "Projection update should pull positives closer"


class TestSOTokenizerGreedyEncoding:
    def test_prefers_longer_merged_tokens(self):
        tok = SOTokenizer(
            latent_dim=LATENT_DIM,
            token_dim=TOKEN_DIM,
            max_vocab=300)
        a, b = ord("a"), ord("b")
        tok.record_cooccurrence([a, b])
        tok.merge((a, b))
        ab_id = tok.merges[(a, b)]
        c = ord("c")
        tok.record_cooccurrence([ab_id, c])
        tok.merge((ab_id, c))
        tokens = tok.encode("abc")
        assert len(tokens) <= 2


class TestSOTokenizerMultilingual:
    def test_cyrillic_tokenization(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        tokens = tok.encode("Привет мир")
        decoded = tok.decode(tokens)
        assert decoded == "привет мир"

    def test_cjk_character_tokenization(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        tokens = tok.encode("你好世界")
        decoded = tok.decode(tokens)
        assert decoded == "你 好 世 界"

    def test_mixed_scripts(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        tokens = tok.encode("Hello привет 你好")
        decoded = tok.decode(tokens)
        assert decoded == "hello привет 你 好"

    def test_arabic_tokenization(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        tokens = tok.encode("مرحبا بالعالم")
        decoded = tok.decode(tokens)
        assert decoded == "مرحبا بالعالم"

    def test_numbers_preserved(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        tokens = tok.encode("Test 123 numbers")
        decoded = tok.decode(tokens)
        assert decoded == "test 123 numbers"


class TestSOTokenizerContrastive:
    def test_contrastive_step_pulls_query_to_positive(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, tokenization_mode="word")
        # Bootstrap with initial texts so tokens exist
        tok.encode("query text here")
        tok.encode("positive match")
        tok.encode("negative random")

        q_emb_before = tok.embed(tok.encode("query text here"))
        p_emb_before = tok.embed(tok.encode("positive match"))
        n_emb_before = tok.embed(tok.encode("negative random"))

        # Run contrastive step
        for _ in range(20):
            tok.contrastive_step(
                query_text="query text here",
                positive_text="positive match",
                negative_texts=["negative random"],
                lr=0.05,
            )

        q_emb_after = tok.embed(tok.encode("query text here"))
        p_emb_after = tok.embed(tok.encode("positive match"))
        n_emb_after = tok.embed(tok.encode("negative random"))

        sim_qp_before = float(np.dot(q_emb_before, p_emb_before))
        sim_qp_after = float(np.dot(q_emb_after, p_emb_after))
        sim_qn_before = float(np.dot(q_emb_before, n_emb_before))
        sim_qn_after = float(np.dot(q_emb_after, n_emb_after))

        assert sim_qp_after > sim_qp_before, (
            f"Query-positive similarity should increase: "
            f"{sim_qp_after:.3f} vs {sim_qp_before:.3f}")
        assert sim_qn_after < sim_qn_before, (
            f"Query-negative similarity should decrease: "
            f"{sim_qn_after:.3f} vs {sim_qn_before:.3f}")
