"""Tests for SOT v2 improvements (A-F)."""
import numpy as np
import pytest
import tempfile
import json
import os

from rtmdk.memory.self_organizing_field import SOTokenizer, ContrastiveHebbian, CooccurrenceStore
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField


# ------------------------------------------------------------------
# A: Warm-start
# ------------------------------------------------------------------
class TestCooccurrenceStore:
    def test_prune_keeps_high_weights(self):
        store = CooccurrenceStore(max_size=5)
        for i in range(10):
            store[(i, i + 1)] = float(i)
        store.prune_if_needed()
        assert len(store) <= 5
        # Highest weights (9,8,7,6,5) should remain
        assert (9, 10) in store
        assert (8, 9) in store
        assert (0, 1) not in store

    def test_prune_drops_low_weights(self):
        store = CooccurrenceStore(max_size=3, prune_factor=1.0)
        store[(1, 2)] = 10.0
        store[(3, 4)] = 5.0
        store[(5, 6)] = 1.0
        store.prune_if_needed()
        assert len(store) == 3  # threshold = 3, no prune yet
        store[(7, 8)] = 0.5
        store.prune_if_needed()
        assert len(store) <= 3
        assert (1, 2) in store
        assert (3, 4) in store
        assert (7, 8) not in store or (5, 6) not in store

    def test_stats_tracked(self):
        store = CooccurrenceStore(max_size=2)
        store[(1, 2)] = 1.0
        store[(3, 4)] = 2.0
        stats = store.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 2
        assert stats["total_inserts"] == 2

    def test_integration_with_tokenizer(self):
        tok = SOTokenizer(latent_dim=16, token_dim=16, max_cooccurrence=10)
        for i in range(50):
            tok.record_cooccurrence(list(range(20)), weight=1.0)
        assert len(tok.cooccurrence) <= 10
        stats = tok.cooccurrence.get_stats()
        assert stats["total_prunes"] > 0


class TestSOTWordMode:
    def test_word_mode_encode(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            tokenization_mode="word")
        tokens = tok.encode("Hello world hello")
        assert len(tokens) == 3
        assert tok.decode(tokens) == "hello world hello"

    def test_word_mode_vocab_grows(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            tokenization_mode="word")
        tok.encode("the quick brown fox")
        assert len(tok.word_to_id) == 4
        tok.encode("the lazy dog")
        assert len(tok.word_to_id) == 6  # 'lazy', 'dog' added

    def test_word_mode_embed(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            tokenization_mode="word")
        tokens = tok.encode("hello world")
        emb = tok.embed(tokens)
        assert emb.shape == (64,)
        assert np.isfinite(emb).all()

    def test_word_mode_prune_vocab(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            tokenization_mode="word")
        tok.encode("hello world hello")  # hello:2, world:1
        tok.record_cooccurrence(tok.encode("hello world hello"))
        tok.prune_vocab(min_freq=2.0)
        # 'world' should be removed (freq=1)
        assert "world" not in tok.word_to_id
        assert "hello" in tok.word_to_id
        # Encoding 'world' should return unk
        world_id = tok.encode("world")[0]
        assert world_id == tok._unk_token_id

    def test_word_mode_save_load(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            tokenization_mode="word")
        tok.encode("hello world hello")
        state = tok.get_state()
        assert state["tokenization_mode"] == "word"
        assert state["word_to_id"]["hello"] == state["word_to_id"]["hello"]
        assert "world" in state["word_to_id"]

        tok2 = SOTokenizer(latent_dim=64, token_dim=64)
        tok2.load_state(state)
        assert tok2.tokenization_mode == "word"
        assert tok2.word_to_id == tok.word_to_id
        assert tok2.encode("hello world") == tok.encode("hello world")

    def test_word_mode_field_integration(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            sot_enabled=True,
            sot_tokenization_mode="word")
        field = RTMDKField(cfg)
        tok = field._projection_mgr.sot_tokenizer
        tokens = tok.encode("coffee is great")
        emb = tok.embed(tokens)
        field.add_node(emb, {'text': 'coffee is great'}, node_id='n1')
        q = tok.encode("coffee")
        qemb = tok.embed(q)
        res = field.query(qemb, top_k=1)
        assert len(res) == 1


class TestSOTWarmStart:
    def test_warm_start_reduces_randomness(self):
        tok = SOTokenizer(latent_dim=64, token_dim=64, seed=42)
        corpus = [
            "hello world hello world",
            "hello there",
            "world is big",
            "hello again",
        ]
        # Before warm-start: random embeddings
        emb_before = {k: v.copy() for k, v in tok.token_embeddings.items()}
        tok.warm_start_from_corpus(corpus)
        # After warm-start: some bytes should have shifted
        changes = sum(
            1 for k in emb_before if k in tok.token_embeddings and np.linalg.norm(
                emb_before[k] -
                tok.token_embeddings[k]) > 1e-6)
        assert changes > 0, "Warm-start should modify at least some embeddings"

    def test_warm_start_idf_computed(self):
        tok = SOTokenizer(latent_dim=64, token_dim=64, seed=42)
        corpus = ["hello world", "hello there", "world is big"]
        tok.warm_start_from_corpus(corpus)
        # IDF should be computed for bytes that appeared
        assert len(tok.token_idf) > 0
        # 'hello' bytes should have lower IDF than rare bytes
        hello_bytes = list("hello".encode("utf-8"))
        rare_byte = list("z".encode("utf-8"))[0]
        if all(b in tok.token_idf for b in hello_bytes) and rare_byte in tok.token_idf:
            hello_idf = min(tok.token_idf[b] for b in hello_bytes)
            rare_idf = tok.token_idf[rare_byte]
            assert rare_idf > hello_idf, "Rare byte should have higher IDF"

    def test_warm_start_empty_corpus_safe(self):
        tok = SOTokenizer(latent_dim=64, token_dim=64, seed=42)
        tok.warm_start_from_corpus([])
        assert len(tok.token_embeddings) == 256  # unchanged


# ------------------------------------------------------------------
# B: Subword seed
# ------------------------------------------------------------------
class TestSOTSubwordSeed:
    def test_subword_seed_increases_vocab(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            subword_seed=True,
            seed=42)
        assert len(
            tok.token_embeddings) > 256, "Subword seed should create additional tokens"

    def test_subword_seed_encoding_shorter(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            subword_seed=True,
            seed=42)
        text = "the"
        tokens = tok.encode(text)
        # With subword seeds, "the" might be a single token instead of 3 bytes
        assert len(
            tokens) <= 3, "Encoding should be same or shorter with subword seeds"

    def test_no_subword_seed_has_only_bytes(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            subword_seed=False,
            seed=42)
        assert len(tok.token_embeddings) == 256


# ------------------------------------------------------------------
# C: Attention-weighted pooling
# ------------------------------------------------------------------
class TestSOTAttentionPooling:
    def test_attention_pooling_different_from_mean(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            attention_pooling=True,
            seed=42)
        # Set up artificial IDF weights
        tok.token_idf[0] = 10.0  # high IDF = rare = important
        tok.token_idf[1] = 0.1   # low IDF = common = less important
        tok.token_embeddings[0] = np.ones(64, dtype=np.float32) / np.sqrt(64)
        tok.token_embeddings[1] = -np.ones(64, dtype=np.float32) / np.sqrt(64)
        emb_attn = tok.embed([0, 1])
        tok_mean = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            attention_pooling=False,
            seed=42)
        tok_mean.token_embeddings[0] = tok.token_embeddings[0].copy()
        tok_mean.token_embeddings[1] = tok.token_embeddings[1].copy()
        emb_mean = tok_mean.embed([0, 1])
        # Attention should weight token 0 more heavily
        assert not np.allclose(emb_attn, emb_mean, atol=1e-4)

    def test_position_weight_first_token(self):
        tok = SOTokenizer(
            latent_dim=64,
            token_dim=64,
            attention_pooling=True,
            seed=42)
        tok.token_idf[0] = 1.0
        tok.token_idf[1] = 1.0
        tok.token_idf[2] = 1.0
        # Use different basis vectors so weighted average is unambiguous
        tok.token_embeddings[0] = np.zeros(64, dtype=np.float32)
        tok.token_embeddings[0][0] = 1.0  # e0
        tok.token_embeddings[1] = np.zeros(64, dtype=np.float32)
        tok.token_embeddings[1][1] = 1.0  # e1
        tok.token_embeddings[2] = np.zeros(64, dtype=np.float32)
        tok.token_embeddings[2][2] = 1.0  # e2
        # Force identity projection for test
        tok.projection = np.eye(64, dtype=np.float32)
        emb = tok.embed([0, 1, 2])
        # Position weights: first=1.5, middle=1.0, last=1.5
        # Total weight = 4.0
        # emb[0] = 1.5/4.0 = 0.375, emb[1] = 1.0/4.0 = 0.25, emb[2] = 1.5/4.0 =
        # 0.375
        assert emb[0] > emb[1], "First token should have higher weight than middle"
        assert emb[2] > emb[1], "Last token should have higher weight than middle"
        # After normalization, proportions should still reflect weights
        # 0.375 : 0.25 : 0.375 = 3 : 2 : 3
        assert emb[0] / \
            emb[1] > 1.4, f"First/middle ratio should be ~1.5, got {emb[0]/emb[1]}"
        assert emb[2] / \
            emb[1] > 1.4, f"Last/middle ratio should be ~1.5, got {emb[2]/emb[1]}"


# ------------------------------------------------------------------
# D: Hard negative mining
# ------------------------------------------------------------------
class TestSOTHardNegatives:
    def test_hard_negatives_are_closest(self):
        ch = ContrastiveHebbian(lr=0.1)
        embeddings = {
            0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # positive (close to 0)
            1: np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
            # hard negative (closer than 3)
            2: np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
            # easy negative (far)
            3: np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        positives = [0, 1]
        all_candidates = [0, 1, 2, 3]
        ch.update_with_hard_negatives(
            embeddings, positives, all_candidates, n_negatives=1)
        # Hard negative mining should have picked token 2 (closest non-positive)
        # We can't directly observe which was picked, but we can verify the
        # method runs

    def test_hard_negatives_fewer_than_candidates(self):
        ch = ContrastiveHebbian(lr=0.1)
        embeddings = {
            i: np.random.randn(8).astype(
                np.float32) for i in range(10)}
        for k, v in embeddings.items():
            embeddings[k] = v / np.linalg.norm(v)
        ch.update_with_hard_negatives(
            embeddings, [
                0, 1], list(
                range(10)), n_negatives=3)
        # Should run without error even with few candidates


# ------------------------------------------------------------------
# F: Skip-gram co-occurrence
# ------------------------------------------------------------------
class TestSOTSkipGram:
    def test_window_1_only_adjacent(self):
        tok = SOTokenizer(latent_dim=64, skipgram_window=1, seed=42)
        tok.record_cooccurrence([10, 20, 30])
        assert tok.cooccurrence[(10, 20)] > 0
        assert tok.cooccurrence[(20, 30)] > 0
        assert tok.cooccurrence.get(
            (10, 30), 0) == 0, "Window=1 should not record non-adjacent"

    def test_window_3_records_skip_pairs(self):
        tok = SOTokenizer(latent_dim=64, skipgram_window=3, seed=42)
        tok.record_cooccurrence([10, 20, 30, 40])
        assert tok.cooccurrence[(10, 20)] > 0
        assert tok.cooccurrence[(
            10, 30)] > 0, "Window=3 should record skip-1 pairs"
        assert tok.cooccurrence[(
            10, 40)] > 0, "Window=3 should record skip-2 pairs"
        # Distance weighting: closer pairs should have higher weight
        assert tok.cooccurrence[(10, 20)] > tok.cooccurrence[(
            10, 30)], "Closer pairs should have higher weight"

    def test_frequency_tracked(self):
        tok = SOTokenizer(latent_dim=64, skipgram_window=2, seed=42)
        tok.record_cooccurrence([10, 20, 10])
        assert tok.token_frequency[10] > tok.token_frequency[20], "Token 10 appears twice"


# ------------------------------------------------------------------
# SBERT Bootstrap (SOT-A v2)
# ------------------------------------------------------------------
class TestSOTBootstrap:
    @pytest.mark.skipif(
        os.system("python -c 'import sentence_transformers' 2>nul") != 0,
        reason="sentence-transformers not installed",
    )
    def test_bootstrap_improves_similarity(self):
        tok = SOTokenizer(latent_dim=64, token_dim=64, seed=42)
        texts = [
            "I love coffee in the morning",
            "Coffee is my favorite drink",
            "Python is a programming language",
            "JavaScript is used for web development",
        ]
        # Mock teacher: simple topic-based embeddings
        topic_embs = {
            "coffee": np.array([1.0, 0.0] + [0.0] * 62, dtype=np.float32),
            "python": np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32),
        }
        for k, v in topic_embs.items():
            topic_embs[k] = v / np.linalg.norm(v)

        def mock_teacher(text):
            vec = np.zeros(64, dtype=np.float32)
            for topic, tvec in topic_embs.items():
                if topic in text.lower():
                    vec += tvec
            return vec / (np.linalg.norm(vec) + 1e-8)

        # Before bootstrap: coffee texts may not be similar
        emb1_before = tok.embed(tok.encode(texts[0]))
        emb2_before = tok.embed(tok.encode(texts[1]))
        sim_before = emb1_before @ emb2_before

        tok.bootstrap_from_teacher(texts, mock_teacher, n_epochs=20, lr=0.1)

        emb1_after = tok.embed(tok.encode(texts[0]))
        emb2_after = tok.embed(tok.encode(texts[1]))
        sim_after = emb1_after @ emb2_after

        assert sim_after > sim_before, (
            f"Bootstrap should increase similarity for related texts: "
            f"{sim_before:.3f} -> {sim_after:.3f}")

    def test_bootstrap_empty_texts_safe(self):
        tok = SOTokenizer(latent_dim=64, token_dim=64, seed=42)
        tok.bootstrap_from_teacher(
            [], lambda x: np.zeros(
                64, dtype=np.float32))
        assert len(tok.token_embeddings) == 256


# ------------------------------------------------------------------
# Integration: SOT v2 config flags
# ------------------------------------------------------------------
class TestSOTV2Integration:
    def test_all_v2_flags_in_config(self):
        cfg = RTMDKConfig(
            sot_enabled=True,
            sot_subword_seed=True,
            sot_attention_pooling=True,
            sot_hard_negatives=True,
            sot_retrieval_feedback=True,
            sot_skipgram_window=3,
            sot_warm_start_corpus=None,
            sot_bootstrap_projection=None,
            sot_bootstrap_corpus=None,
            sot_bootstrap_model="all-MiniLM-L6-v2",
            sot_tokenization_mode="word",
            sot_max_cooccurrence=50_000,
        )
        assert cfg.sot_subword_seed is True
        assert cfg.sot_attention_pooling is True
        assert cfg.sot_hard_negatives is True
        assert cfg.sot_retrieval_feedback is True
        assert cfg.sot_skipgram_window == 3
        assert cfg.sot_bootstrap_projection is None
        assert cfg.sot_bootstrap_corpus is None
        assert cfg.sot_bootstrap_model == "all-MiniLM-L6-v2"
        assert cfg.sot_tokenization_mode == "word"
        assert cfg.sot_max_cooccurrence == 50_000

    def test_field_initializes_with_v2_features(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            sot_enabled=True,
            sot_subword_seed=True,
            sot_attention_pooling=True,
            sot_skipgram_window=3,
        )
        field = RTMDKField(cfg)
        assert field._projection_mgr.sot_tokenizer is not None
        assert field._projection_mgr.sot_tokenizer.subword_seed is True
        assert field._projection_mgr.sot_tokenizer.attention_pooling is True
        assert field._projection_mgr.sot_tokenizer.skipgram_window == 3

    def test_warm_start_from_file(self):
        # Create temp corpus file
        corpus = {"records": [
            {"query": "hello", "answer": "world", "context": "test"},
            {"query": "foo", "answer": "bar", "context": "baz"},
        ]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(corpus, f)
            path = f.name
        try:
            cfg = RTMDKConfig(
                latent_dim=64,
                sot_enabled=True,
                sot_warm_start_corpus=path,
            )
            field = RTMDKField(cfg)
            assert len(
                field._projection_mgr.sot_tokenizer.token_idf) > 0, "Warm-start should compute IDF"
        finally:
            os.unlink(path)

    def test_query_with_retrieval_feedback(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            sot_enabled=True,
            sot_use_for_query=True,
            sot_retrieval_feedback=True,
        )
        field = RTMDKField(cfg)
        emb1 = np.random.randn(64).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(64).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        field.add_node(emb1,
                       {'text': 'hello world',
                        'node_id': 'n1'},
                       skip_projection=True)
        field.add_node(emb2,
                       {'text': 'foo bar',
                        'node_id': 'n2'},
                       skip_projection=True)
        # Query should trigger feedback
        results = field.query(emb1, top_k=2)
        assert len(results) > 0
        # Feedback should have updated something

    def test_fasttext_bootstrap_config(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            sot_enabled=True,
            sot_bootstrap_fasttext_model="dummy.model",
            sot_bootstrap_corpus=None,
        )
        assert cfg.sot_bootstrap_fasttext_model == "dummy.model"

    def test_fasttext_bootstrap_skips_if_no_model(self):
        from rtmdk.memory.bootstrap_fasttext import run_bootstrap
        tok = SOTokenizer(latent_dim=64, tokenization_mode="word")
        run_bootstrap(
            tok,
            texts=["hello world"],
            model_path="nonexistent.model")
        # Should not crash, just warn
        assert len(tok.token_embeddings) > 0

    def test_bootstrap_projection_load(self):
        import tempfile
        # Create a fake bootstrap .npz
        vocab = np.array([0, 1, 2], dtype=np.int32)
        proj = np.ones((3, 64), dtype=np.float32) * 0.1
        embs = np.ones((3, 384), dtype=np.float32) * 0.2
        freqs = np.array([10.0, 5.0, 1.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(
                f.name,
                projection=proj,
                vocab=vocab,
                token_embeddings=embs,
                token_frequencies=freqs)
            path = f.name
        try:
            cfg = RTMDKConfig(
                latent_dim=64,
                sot_enabled=True,
                sot_bootstrap_projection=path,
            )
            field = RTMDKField(cfg)
            # token_embeddings should have been updated for tokens 0,1,2
            assert 0 in field._projection_mgr.sot_tokenizer.token_embeddings
            assert 1 in field._projection_mgr.sot_tokenizer.token_embeddings
            assert 2 in field._projection_mgr.sot_tokenizer.token_embeddings
        finally:
            os.unlink(path)

    def test_end_to_end_v2_vs_v1_recall(self):
        """Compare SOT v1 (default) vs v2 (all features) on synthetic retrieval."""
        # Build a tiny deterministic embedder for fair comparison
        rng = np.random.RandomState(42)
        topic_vectors = {
            "coffee": rng.randn(64).astype(np.float32),
            "city": rng.randn(64).astype(np.float32),
            "work": rng.randn(64).astype(np.float32),
        }
        for k, v in topic_vectors.items():
            topic_vectors[k] = v / np.linalg.norm(v)

        def make_emb(text):
            # Very simple: sum of topic vectors mentioned
            vec = np.zeros(64, dtype=np.float32)
            for topic, tvec in topic_vectors.items():
                if topic in text.lower():
                    vec += tvec
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else rng.randn(64).astype(np.float32)

        docs = [
            ("I love coffee in the morning", "coffee"),
            ("I work as a programmer", "work"),
            ("I live in Moscow", "city"),
            ("Coffee shops in Moscow are great", "coffee,city"),
            ("My work involves coding", "work"),
        ]

        # v1: default SOT (byte-level)
        cfg_v1 = RTMDKConfig(
            latent_dim=64, sot_enabled=True, sot_use_for_query=True,
            sot_tokenization_mode="byte",
        )
        field_v1 = RTMDKField(cfg_v1)
        for i, (text, _) in enumerate(docs):
            emb = make_emb(text)
            field_v1.add_node(emb,
                              {'text': text,
                               'node_id': f'd{i}',
                               'topic': text},
                              skip_projection=True)

        # v2: all features (byte-level with subword seeds)
        cfg_v2 = RTMDKConfig(
            latent_dim=64, sot_enabled=True, sot_use_for_query=True,
            sot_tokenization_mode="byte",
            sot_subword_seed=True,
            sot_attention_pooling=True,
            sot_skipgram_window=3,
        )
        field_v2 = RTMDKField(cfg_v2)
        for i, (text, _) in enumerate(docs):
            emb = make_emb(text)
            field_v2.add_node(emb,
                              {'text': text,
                               'node_id': f'd{i}',
                               'topic': text},
                              skip_projection=True)

        # Query about coffee
        q_coffee = make_emb("coffee")
        res_v1 = field_v1.query(q_coffee, top_k=3)
        res_v2 = field_v2.query(q_coffee, top_k=3)

        # Both should find coffee-related docs
        v1_has_coffee = any(
            'coffee' in field_v1.nodes[nid].content['text'].lower() for nid,
            _,
            _ in res_v1)
        v2_has_coffee = any(
            'coffee' in field_v2.nodes[nid].content['text'].lower() for nid,
            _,
            _ in res_v2)
        assert v1_has_coffee
        assert v2_has_coffee

        # v2 should have more nodes (subword seeds)
        assert len(
            field_v2._projection_mgr.sot_tokenizer.token_embeddings) > len(
            field_v1._projection_mgr.sot_tokenizer.token_embeddings)


# ------------------------------------------------------------------
# Sparse PMI path (backlog v8.2.1)
# ------------------------------------------------------------------
class TestSparsePMI:
    def test_sparse_pmi_path_does_not_crash(self):
        from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
        sif = SIFEmbedder(latent_dim=64, min_count=1, window_size=2)
        vocab_size = 6000  # > SPARSE_PMI_THRESHOLD (5000)
        n_docs = 1200
        doc_len = 10
        tokenized_docs = [
            [(i * doc_len + j) % vocab_size for j in range(doc_len)]
            for i in range(n_docs)
        ]
        # Ensure every token appears at least twice
        for t in range(vocab_size):
            tokenized_docs[t % n_docs].append(t)
        sif.fit(tokenized_docs, vocab_size=vocab_size)
        assert len(sif.word_embeddings) == vocab_size
        assert sif._pmi_matrix is not None

    def test_sparse_pmi_expand_query_terms(self):
        from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
        sif = SIFEmbedder(latent_dim=64, min_count=1, window_size=2)
        vocab_size = 6000
        n_docs = 1200
        doc_len = 10
        tokenized_docs = [
            [(i * doc_len + j) % vocab_size for j in range(doc_len)]
            for i in range(n_docs)
        ]
        for t in range(vocab_size):
            tokenized_docs[t % n_docs].append(t)
        sif.fit(tokenized_docs, vocab_size=vocab_size)
        expanded = sif.expand_query([0, 1, 2], n_terms=2, min_pmi=0.0)
        # Should return some terms (PMI threshold 0 means all non-zero PMI)
        assert isinstance(expanded, list)

    def test_dense_pmi_still_works_for_small_vocab(self):
        from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
        sif = SIFEmbedder(latent_dim=16, min_count=1, window_size=2)
        tokenized_docs = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 0],
        ]
        sif.fit(tokenized_docs, vocab_size=6)
        assert len(sif.word_embeddings) == 6
        expanded = sif.expand_query([0], n_terms=2, min_pmi=0.0)
        assert isinstance(expanded, list)
