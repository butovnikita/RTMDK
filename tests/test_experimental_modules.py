"""Tests for rtmdk/experimental — TPR, AdversarialArena, ActiveInferenceLoop."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

import rtmdk.experimental  # noqa: F401 — covers package __init__ re-exports
from rtmdk.experimental.active_inference import ActiveInferenceLoop
from rtmdk.experimental.adversarial_arena import AdversarialArena
from rtmdk.experimental.tpr import TensorProductRepresentation


class TestTensorProductRepresentation:
    def test_bind_unbind_roundtrip(self):
        tpr = TensorProductRepresentation(role_dim=16, filler_dim=8)
        rng = np.random.default_rng(0)
        filler = rng.standard_normal(8).astype(np.float32)

        bound = tpr.bind("subject", filler)
        assert bound.shape == (16 * 8,)

        recovered = tpr.unbind(bound, "subject")
        # Role vector is unit-norm → unbind recovers the filler exactly
        np.testing.assert_allclose(recovered, filler, atol=1e-5)

    def test_role_vectors_cached_and_normalized(self):
        tpr = TensorProductRepresentation(role_dim=16, filler_dim=8)
        v1 = tpr._get_role_vector("subject")
        v2 = tpr._get_role_vector("subject")

        assert v1 is v2
        assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-6)

    def test_different_roles_different_vectors(self):
        tpr = TensorProductRepresentation(role_dim=16, filler_dim=8)
        assert not np.allclose(tpr._get_role_vector("a"), tpr._get_role_vector("b"))

    def test_cross_role_unbind_gives_near_zero(self):
        """Unbinding with an unrelated role should not recover the filler."""
        tpr = TensorProductRepresentation(role_dim=64, filler_dim=8)
        rng = np.random.default_rng(1)
        filler = rng.standard_normal(8).astype(np.float32)

        bound = tpr.bind("subject", filler)
        noise = tpr.unbind(bound, "object")

        assert np.linalg.norm(noise) < np.linalg.norm(filler)


class TestAdversarialArena:
    def _make_memory(self, context="some retrieved context"):
        return SimpleNamespace(load_memory_variables=lambda inputs: {"rtmdk_context": context})

    def test_generate_adversarial_query_variants(self):
        random.seed(0)
        arena = AdversarialArena(memory=None)
        variants = {arena.generate_adversarial_query("What is RTMDK?") for _ in range(30)}

        assert "what is rtmdk?" in variants  # lowercase
        assert "What is RTMDK can you tell me?" in variants  # politeness injection
        assert "Is it true that what is rtmdk?" in variants  # framing

    def test_robustness_all_consistent(self):
        arena = AdversarialArena(self._make_memory())
        random.seed(1)

        report = arena.test_robustness(["q1?", "q2?", "q3?"], top_k=3)

        assert report["robustness_rate"] == 1.0
        assert arena.attack_stats == {"total": 3, "successful": 3, "failed": 0}
        assert len(report["details"]) == 3
        assert all(d["consistent"] for d in report["details"])

    def test_robustness_detects_inconsistency(self):
        class FlakyMemory:
            def __init__(self):
                self.calls = 0

            def load_memory_variables(self, inputs):
                self.calls += 1
                return {"rtmdk_context": f"context-variant-{self.calls}"}

        arena = AdversarialArena(FlakyMemory())
        random.seed(2)

        report = arena.test_robustness(["q1?", "q2?"])

        assert report["robustness_rate"] == 0.0
        assert arena.attack_stats["failed"] == 2
        for detail in report["details"]:
            assert set(detail) == {"query", "adv_query", "consistent"}
            assert detail["consistent"] is False


class TestActiveInferenceLoop:
    def _field(self, saliences):
        return SimpleNamespace(
            nodes={nid: SimpleNamespace(salience=s, content={"text": f"note {nid}"}) for nid, s in saliences.items()}
        )

    def test_compute_uncertainty_is_one_minus_salience(self):
        loop = ActiveInferenceLoop()
        uncertainties = loop.compute_uncertainty(self._field({"a": 0.9, "b": 0.2}))

        assert uncertainties == {"a": pytest.approx(0.1), "b": pytest.approx(0.8)}

    def test_intervention_targets_most_uncertain_node(self):
        loop = ActiveInferenceLoop(uncertainty_threshold=0.3)
        intervention = loop.generate_intervention(self._field({"a": 0.9, "b": 0.1}))

        assert intervention["type"] == "active_query"
        assert intervention["target_node"] == "b"
        assert "note b" in intervention["query"]
        assert intervention["expected_outcome"] == "reduce_uncertainty"
        assert loop._interventions_generated == 1

    def test_no_intervention_when_confident(self):
        loop = ActiveInferenceLoop(uncertainty_threshold=0.3)
        assert loop.generate_intervention(self._field({"a": 0.95})) is None
        assert loop._interventions_generated == 0

    def test_no_intervention_on_empty_field(self):
        loop = ActiveInferenceLoop()
        assert loop.generate_intervention(SimpleNamespace(nodes={})) is None

    def test_curiosity_drive(self):
        loop = ActiveInferenceLoop(curiosity_weight=0.5)
        assert loop.get_curiosity_drive() == 0.0

        loop.record_prediction_error(0.4)
        loop.record_prediction_error(0.6)
        assert loop.get_curiosity_drive() == pytest.approx(0.25)  # mean(0.4, 0.6) * 0.5
