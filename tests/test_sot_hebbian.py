"""Tests for ContrastiveHebbian."""

from __future__ import annotations

import numpy as np

from rtmdk.memory.self_organizing_field import ContrastiveHebbian

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
            assert np.abs(np.linalg.norm(emb) - 1.0) < 1e-4, "Embeddings should stay unit norm"

    def test_empty_positives_no_crash(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {0: np.array([1.0, 0.0], dtype=np.float32)}
        ch.update(embeddings, positives=[], negatives=[])
        assert np.allclose(embeddings[0], np.array([1.0, 0.0], dtype=np.float32))

    def test_single_positive_no_change(self):
        ch = ContrastiveHebbian(lr=LR)
        embeddings = {0: np.array([1.0, 0.0], dtype=np.float32)}
        original = embeddings[0].copy()
        ch.update(embeddings, positives=[0], negatives=[])
        assert np.allclose(embeddings[0], original)


class TestContrastiveHebbianFieldUpdate:
    def test_field_positives_move_closer(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array(
            [
                [0.9, 0.4, 0.0],
                [0.4, 0.9, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for i in range(len(node_embeddings)):
            node_embeddings[i] /= np.linalg.norm(node_embeddings[i])
        original_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[1])
        ch.field_update(node_embeddings, positives=[0, 1], negatives=[2])
        new_dist = np.linalg.norm(node_embeddings[0] - node_embeddings[1])
        assert new_dist < original_dist

    def test_field_negatives_move_apart(self):
        ch = ContrastiveHebbian(lr=LR)
        node_embeddings = np.array(
            [
                [0.9, 0.4, 0.0],
                [0.4, 0.9, 0.0],
                [0.5, 0.0, 0.8],  # slight overlap with 0
            ],
            dtype=np.float32,
        )
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
        node_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        original_ptr = node_embeddings.ctypes.data
        ch.field_update(node_embeddings, positives=[0, 1], negatives=[])
        assert node_embeddings.ctypes.data == original_ptr, "Should modify in-place"
