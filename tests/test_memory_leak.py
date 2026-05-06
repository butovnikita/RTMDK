"""Memory pressure and leak tests for RTMDK."""
import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField
from rtmdk.memory.self_organizing_field import SOTokenizer


class TestCooccurrenceBounded:
    def test_cooccurrence_does_not_exceed_max(self):
        tok = SOTokenizer(latent_dim=16, max_cooccurrence=50)
        # Record many cooccurrences
        for i in range(100):
            tokens = list(range(100))
            tok.record_cooccurrence(tokens, weight=1.0)
        assert len(tok.cooccurrence) <= 50
        stats = tok.cooccurrence.get_stats()
        assert stats["total_prunes"] > 0

    def test_cooccurrence_prune_keeps_high_weights(self):
        tok = SOTokenizer(latent_dim=16, max_cooccurrence=5)
        # Create specific high-weight pairs
        for _ in range(100):
            tok.record_cooccurrence([1, 2], weight=10.0)
            tok.record_cooccurrence([3, 4], weight=1.0)
        # After prune, high-weight pair should remain
        assert (1, 2) in tok.cooccurrence
        assert tok.cooccurrence[(1, 2)] > 50.0


class TestVocabBounded:
    def test_word_vocab_respects_max_vocab(self):
        tok = SOTokenizer(latent_dim=16, max_vocab=20, tokenization_mode="word")
        # Encode many unique words
        text = " ".join([f"word{i}" for i in range(100)])
        tokens = tok.encode(text)
        # Vocab should not exceed max_vocab (including byte fallback)
        assert len(tok.token_embeddings) <= tok.max_vocab + tok.initial_byte_vocab
        # All tokens should be valid IDs
        assert all(t < tok.next_token_id for t in tokens)

    def test_byte_vocab_never_exceeds_256_plus_merges(self):
        tok = SOTokenizer(latent_dim=16, max_vocab=4096, tokenization_mode="byte")
        for i in range(500):
            tok.record_cooccurrence(list(range(256)), weight=1.0)
            if i % 50 == 0:
                tok.propose_merges(1)
        assert len(tok.token_embeddings) <= tok.max_vocab


class TestMemoryPressure:
    @pytest.mark.slow
    def test_field_under_load_cooccurrence_bounded(self):
        cfg = RTMDKConfig(
            latent_dim=32,
            sot_enabled=True,
            sot_tokenization_mode="word",
            sot_max_cooccurrence=100,
        )
        field = RTMDKField(cfg)
        tok = field.sot_tokenizer

        import time
        # Simulate 500 documents
        for i in range(500):
            text = f"document number {i} with some content about topic{i % 10}"
            tokens = tok.encode(text)
            emb = tok.embed(tokens)
            field.add_node(emb, {"text": text}, node_id=f"n{i}")
            time.sleep(0.011)

        assert len(tok.cooccurrence) <= 100

    def test_field_token_frequency_tracked(self):
        cfg = RTMDKConfig(latent_dim=16, sot_enabled=True, sot_tokenization_mode="word")
        field = RTMDKField(cfg)
        tok = field.sot_tokenizer

        for i in range(100):
            text = f"hello world test {i}"
            tokens = tok.encode(text)
            tok.record_cooccurrence(tokens)

        assert len(tok.token_frequency) > 0
        assert tok.token_frequency[tok.word_to_id["hello"]] > 0
