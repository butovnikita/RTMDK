"""Tests for ContrastiveHebbian and EmbeddingFieldSSM."""
from __future__ import annotations

import numpy as np

from rtmdk.memory.self_organizing_field import ContrastiveHebbian, EmbeddingFieldSSM, SOTokenizer

LATENT_DIM = 64
TOKEN_DIM = 128
LR = 0.1


class TestContrastiveHebbianInit:
    def test_default_init(self):
        ch = ContrastiveHebbian()
        assert ch.lr == 0.01
        assert ch.neg_ratio == 0.2
        assert ch.temperature == 0.1

    def test_custom_init(self):
        ch = ContrastiveHebbian(lr=0.05, neg_ratio=0.3, temperature=0.2)
        assert ch.lr == 0.05
        assert ch.neg_ratio == 0.3
        assert ch.temperature == 0.2


class TestContrastiveHebbianTokenUpdate:
    def test_positives_move_closer(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {
            0: np.array([0.9, 0.4, 0.0], dtype=np.float32),
            1: np.array([0.4, 0.9, 0.0], dtype=np.float32),
            2: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        # Normalize
        for k in embeddings:
            embeddings[k] /= np.linalg.norm(embeddings[k])
        original_dist = np.linalg.norm(embeddings[0] - embeddings[1])
        ch.update(embeddings, positives=[0, 1], negatives=[2])
        new_dist = np.linalg.norm(embeddings[0] - embeddings[1])
        assert new_dist < original_dist, "Positives should be pulled closer"

    def test_negatives_move_apart(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {
            0: np.array([0.9, 0.4, 0.0], dtype=np.float32),
            1: np.array([0.4, 0.9, 0.0], dtype=np.float32),
            # slight overlap with 0
            2: np.array([0.5, 0.0, 0.8], dtype=np.float32),
        }
        for k in embeddings:
            embeddings[k] /= np.linalg.norm(embeddings[k])
        original_dist = np.linalg.norm(embeddings[0] - embeddings[2])
        ch.update(embeddings, positives=[0, 1], negatives=[2])
        new_dist = np.linalg.norm(embeddings[0] - embeddings[2])
        assert new_dist > original_dist, "Negatives should be pushed apart"

    def test_embeddings_remain_normalized(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {
            0: np.array([0.9, 0.4, 0.0], dtype=np.float32),
            1: np.array([0.4, 0.9, 0.0], dtype=np.float32),
            2: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        for k in embeddings:
            embeddings[k] /= np.linalg.norm(embeddings[k])
        ch.update(embeddings, positives=[0, 1], negatives=[2])
        for emb in embeddings.values():
            assert np.abs(np.linalg.norm(emb) -
                          1.0) < 1e-4, "Embeddings should stay unit norm"

    def test_empty_positives_no_crash(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {0: np.array([1.0, 0.0], dtype=np.float32)}
        ch.update(embeddings, positives=[], negatives=[])
        assert np.allclose(embeddings[0], np.array(
            [1.0, 0.0], dtype=np.float32))

    def test_single_positive_no_change(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {0: np.array([1.0, 0.0], dtype=np.float32)}
        original = embeddings[0].copy()
        ch.update(embeddings, positives=[0], negatives=[])
        assert np.allclose(embeddings[0], original)


class TestContrastiveHebbianFieldUpdate:
    def test_field_positives_move_closer(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array([
            [0.9, 0.4, 0.0],
            [0.4, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        for i in range(len(node_embeddings)):
            node_embeddings[i] /= np.linalg.norm(node_embeddings[i])
        original_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[1])
        ch.field_update(node_embeddings, positives=[0, 1], negatives=[2])
        new_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[1])
        assert new_dist < original_dist

    def test_field_negatives_move_apart(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array([
            [0.9, 0.4, 0.0],
            [0.4, 0.9, 0.0],
            [0.5, 0.0, 0.8],  # slight overlap with 0
        ], dtype=np.float32)
        for i in range(len(node_embeddings)):
            node_embeddings[i] /= np.linalg.norm(node_embeddings[i])
        original_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[2])
        ch.field_update(node_embeddings, positives=[0, 1], negatives=[2])
        new_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[2])
        assert new_dist > original_dist

    def test_field_update_returns_none(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        result = ch.field_update(node_embeddings, positives=[0], negatives=[])
        assert result is None

    def test_field_update_in_place(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        original_ptr = node_embeddings.ctypes.data
        ch.field_update(node_embeddings, positives=[0, 1], negatives=[])
        assert node_embeddings.ctypes.data == original_ptr, "Should modify in-place"


class TestEmbeddingFieldSSMInit:
    def test_init_with_tokenizer(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        assert ssm.latent_dim == LATENT_DIM
        assert ssm.tokenizer is tok
        assert ssm.ssm is not None
        assert ssm.ssm.diagonal is True

    def test_diagonal_ssm_complexity(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        stats = ssm.ssm.get_stats()
        assert "O(N*64)" in stats["complexity"] or "diagonal" in str(
            stats).lower() or stats["diagonal"] is True


class TestEmbeddingFieldSSMStep:
    def test_step_returns_momentum_vector(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        tokens = tok.encode("hi")
        field_state = np.zeros(LATENT_DIM, dtype=np.float32)
        momentum = ssm.step(tokens, field_state)
        assert momentum.shape == (LATENT_DIM,)
        assert momentum.dtype == np.float32

    def test_step_empty_tokens(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        momentum = ssm.step([], np.zeros(LATENT_DIM, dtype=np.float32))
        assert momentum.shape == (LATENT_DIM,)

    def test_step_different_field_states_give_different_outputs(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        tokens = tok.encode("abc")
        m1 = ssm.step(tokens, np.zeros(LATENT_DIM, dtype=np.float32))
        m2 = ssm.step(tokens, np.ones(LATENT_DIM, dtype=np.float32) * 0.5)
        assert not np.allclose(
            m1, m2), "Different field states should produce different momentum"


class TestEmbeddingFieldSSMSync:
    def test_sync_embeddings_updates_token_embeddings(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        tokens = tok.encode("ab")
        original_embs = [tok.token_embeddings[t].copy() for t in tokens]
        momentum = np.ones(LATENT_DIM, dtype=np.float32) * 0.1
        ssm.sync_embeddings(tokens, momentum)
        for t, orig in zip(tokens, original_embs):
            assert not np.allclose(
                tok.token_embeddings[t], orig), "Token embeddings should change after sync"

    def test_sync_embeddings_preserves_shape(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        tokens = tok.encode("test")
        momentum = np.ones(LATENT_DIM, dtype=np.float32) * 0.01
        ssm.sync_embeddings(tokens, momentum)
        for t in tokens:
            assert tok.token_embeddings[t].shape == (TOKEN_DIM,)
            assert tok.token_embeddings[t].dtype == np.float32

    def test_sync_empty_tokens_no_crash(self):
        tok = SOTokenizer(latent_dim=LATENT_DIM, token_dim=TOKEN_DIM)
        ssm = EmbeddingFieldSSM(
            latent_dim=LATENT_DIM,
            tokenizer=tok,
            diagonal=True)
        ssm.sync_embeddings([], np.zeros(LATENT_DIM, dtype=np.float32))
