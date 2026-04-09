"""
test_rtmdk_v6.py
Тесты для Фазы 7-9: Neural ODE, Agent Orchestration, Production Stack.
"""

import pytest
import json
import numpy as np
import math
from typing import Dict

from rtmdk_memory_v6 import (
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    NeuralODEDynamics, AgentPlanner, HypothesisVerifier, ToolRouter,
    ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager,
    AgentPlan, ToolCall, Hypothesis, EvalResult,
    format_context, build_system_prompt, ContextFormat,
)


@pytest.fixture
def dummy_embedder():
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return _embed


@pytest.fixture
def config_v6():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


# ============================================================================
# ФАЗА 7: NEURAL ODE/SDE
# ============================================================================

class TestNeuralODEDynamics:
    def test_creation(self):
        ode = NeuralODEDynamics(latent_dim=64, noise_level=0.01, time_horizon=1.0, n_steps=20)
        assert ode.latent_dim == 64
        assert ode.noise_level == 0.01

    def test_evolve(self):
        ode = NeuralODEDynamics(latent_dim=8, noise_level=0.0, time_horizon=0.5, n_steps=5)
        initial = np.random.randn(3, 8).astype(np.float32).flatten()
        trajectory = ode.evolve(initial)
        assert trajectory.shape[0] == 5
        assert trajectory.shape[1] == 24

    def test_evolve_with_noise(self):
        ode = NeuralODEDynamics(latent_dim=8, noise_level=0.1, time_horizon=0.5, n_steps=5)
        initial = np.zeros((3, 8), dtype=np.float32).flatten()
        trajectory = ode.evolve_with_noise(initial)
        assert len(trajectory) > 1

    def test_compute_topology_gradient(self):
        ode = NeuralODEDynamics(latent_dim=8)
        nodes = {}
        for i in range(5):
            pos = np.random.randn(8).astype(np.float32)
            nodes[f"n{i}"] = MemoryNode(id=f"n{i}", latent_pos=pos, phase=0.0, amplitude=0.7, salience=0.6)
        grad = ode.compute_topology_gradient(nodes)
        assert grad is not None
        assert len(grad) == 40  # 5 nodes * 8 dims

    def test_response_smoothness(self):
        ode = NeuralODEDynamics(latent_dim=8)
        for v in [0.5, 0.51, 0.49, 0.5, 0.5]:
            ode.record_response(v)
        smoothness = ode.compute_response_smoothness()
        assert smoothness > 0.9  # Low variance → high smoothness

    def test_response_smoothness_high_variance(self):
        ode = NeuralODEDynamics(latent_dim=8)
        for v in [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]:
            ode.record_response(v)
        smoothness = ode.compute_response_smoothness()
        assert smoothness < 0.7  # High variance → lower smoothness

    def test_get_load_state(self):
        ode = NeuralODEDynamics(latent_dim=8, noise_level=0.05)
        state = ode.get_state()
        ode2 = NeuralODEDynamics(latent_dim=8)
        ode2.load_state(state)
        assert ode2.noise_level == 0.05

    def test_chunked_evolve(self):
        ode = NeuralODEDynamics(latent_dim=8, chunk_size=4, time_horizon=0.3, n_steps=3)
        initial = np.random.randn(10, 8).astype(np.float32).flatten()  # 10 nodes > chunk_size
        trajectory = ode.evolve(initial)
        assert trajectory.shape[0] == 3
        assert trajectory.shape[1] == 80


class TestFieldODE:
    def test_field_with_continuous_dynamics(self, config_v6):
        config_v6.continuous_dynamics = True
        config_v6.ode_n_steps = 5
        config_v6.ode_time_horizon = 0.1
        field = RTMDKField(config_v6)
        assert field.ode_dynamics is not None

    def test_evolve_continuous(self, config_v6):
        config_v6.continuous_dynamics = True
        config_v6.ode_n_steps = 5
        config_v6.ode_time_horizon = 0.1
        field = RTMDKField(config_v6)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0, node_id="n1")
        field.add_node(emb + 0.01, {"text": "test2"}, phase=0.1, node_id="n2")
        trajectory = field.evolve_continuous()
        assert len(trajectory) > 0
        assert field.stats["ode_steps"] >= 1

    def test_evolve_with_sde(self, config_v6):
        config_v6.continuous_dynamics = True
        config_v6.sde_noise_level = 0.05
        config_v6.ode_n_steps = 5
        config_v6.ode_time_horizon = 0.1
        field = RTMDKField(config_v6)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, node_id="n1")
        trajectory = field.evolve_continuous(use_sde=True)
        assert len(trajectory) > 1

    def test_response_smoothness_tracked(self, config_v6):
        config_v6.continuous_dynamics = True
        config_v6.ode_n_steps = 3
        config_v6.ode_time_horizon = 0.1
        field = RTMDKField(config_v6)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, node_id="n1")
        field.add_node(emb + 0.01, {"text": "test2"}, node_id="n2")
        field.query(emb, phase=0.0)
        field.consolidate()
        assert "response_smoothness" in field.stats


# ============================================================================
# ФАЗА 8: АГЕНТНАЯ ОРКЕСТРАЦИЯ
# ============================================================================

class TestAgentPlanner:
    def test_creation(self):
        p = AgentPlanner(max_depth=3, max_tool_calls=5, tool_timeout=15.0)
        assert p.max_depth == 3

    def test_create_plan(self):
        p = AgentPlanner()
        plan = p.create_plan("Answer question about coffee", ["retrieve", "verify", "synthesize"], {})
        assert isinstance(plan, AgentPlan)
        assert plan.goal == "Answer question about coffee"
        assert len(plan.subtasks) > 0
        assert len(plan.tools_needed) > 0

    def test_can_call_tool(self):
        p = AgentPlanner(max_tool_calls=2)
        assert p.can_call_tool("retrieve")
        p.record_tool_call("retrieve")
        assert p.can_call_tool("verify")
        p.record_tool_call("verify")
        assert not p.can_call_tool("synthesize")

    def test_reset(self):
        p = AgentPlanner(max_tool_calls=1)
        p.record_tool_call("retrieve")
        assert not p.can_call_tool("verify")
        p.reset()
        assert p.can_call_tool("verify")


class TestHypothesisVerifier:
    def test_creation(self):
        v = HypothesisVerifier(confidence_threshold=0.7)
        assert v.confidence_threshold == 0.7

    def test_verify_no_causal_engine(self):
        v = HypothesisVerifier()
        h = v.verify("Coffee causes alertness", None, ["n1", "n2"])
        assert h.confidence == 0.5
        assert not h.verified

    def test_verify_with_causal_engine(self):
        from rtmdk_memory_v6 import CausalInferenceEngine
        v = HypothesisVerifier(confidence_threshold=0.5)
        engine = CausalInferenceEngine()
        engine.causal_effects[("cause", "effect")] = type('CausalEdge', (), {
            'strength': 0.8, 'confidence': 0.9})()
        h = v.verify("cause → effect", engine, ["cause", "effect"])
        assert h.confidence > 0.5


class TestToolRouter:
    def test_creation(self):
        r = ToolRouter(timeout=15.0)
        assert r.timeout == 15.0

    def test_register_and_execute(self):
        r = ToolRouter()
        def dummy_tool(x: int) -> int:
            return x * 2
        r.register_tool("double", dummy_tool)
        result = r.execute("double", {"x": 5})
        assert result.success
        assert result.result == 10

    def test_execute_unknown_tool(self):
        r = ToolRouter()
        result = r.execute("unknown", {})
        assert not result.success
        assert "not registered" in result.error

    def test_misuse_rate(self):
        r = ToolRouter()
        r.register_tool("good", lambda: True)
        r.register_tool("bad", lambda: 1/0)
        r.execute("good", {})
        r.execute("bad", {})
        rate = r.get_misuse_rate()
        assert rate == 0.5


class TestFieldAgent:
    def test_field_with_agent(self, config_v6):
        config_v6.agent_orchestration = True
        field = RTMDKField(config_v6)
        assert field.agent_planner is not None
        assert field.hypothesis_verifier is not None
        assert field.tool_router is not None

    def test_create_plan(self, config_v6):
        config_v6.agent_orchestration = True
        field = RTMDKField(config_v6)
        plan = field.create_plan("Find info about coffee", ["retrieve", "verify"])
        assert isinstance(plan, AgentPlan)
        assert field.stats["plans_created"] >= 1

    def test_verify_hypothesis(self, config_v6):
        config_v6.agent_orchestration = True
        config_v6.causal_topological = True
        field = RTMDKField(config_v6)
        h = field.verify_hypothesis("test hypothesis", ["n1"])
        assert isinstance(h, Hypothesis)
        assert field.stats["hypotheses_verified"] >= 1

    def test_execute_tool(self, config_v6):
        config_v6.agent_orchestration = True
        field = RTMDKField(config_v6)
        field.register_tool("test_tool", lambda: "ok")
        result = field.execute_tool("test_tool", {})
        assert isinstance(result, ToolCall)
        assert field.stats["tool_calls"] >= 1


# ============================================================================
# ФАЗА 9: ПРОДАКШЕН-СТЕК
# ============================================================================

class TestShadowModeEvaluator:
    def test_creation(self):
        e = ShadowModeEvaluator(fallback_threshold=0.3)
        assert e.fallback_threshold == 0.3

    def test_compare(self):
        e = ShadowModeEvaluator(fallback_threshold=0.3)
        result = e.compare(0.8, 0.7)
        assert result["shadow_value"] == 0.8
        assert result["production_value"] == 0.7
        assert result["shadow_better"] is True
        assert not result["fallback_triggered"]

    def test_compare_fallback(self):
        e = ShadowModeEvaluator(fallback_threshold=0.1)
        result = e.compare(0.9, 0.5)
        assert result["fallback_triggered"] is True

    def test_correlation(self):
        e = ShadowModeEvaluator()
        for i in range(10):
            e.compare(0.5 + i * 0.01, 0.5 + i * 0.01 + 0.02)
        corr = e.get_correlation()
        assert corr > 0.9  # High correlation

    def test_fallback_rate(self):
        e = ShadowModeEvaluator(fallback_threshold=0.05)
        e.compare(0.5, 0.9)  # Triggers fallback
        e.compare(0.5, 0.51)  # No fallback
        assert e.get_fallback_rate() == 0.5


class TestRAGASPlusEvaluator:
    def test_creation(self):
        e = RAGASPlusEvaluator()
        assert e is not None

    def test_evaluate_basic(self):
        e = RAGASPlusEvaluator()
        result = e.evaluate(
            question="What is coffee?",
            answer="Coffee is a drink made from beans",
            contexts=["Coffee is a popular drink", "Coffee beans are roasted"],
            ground_truth="Coffee is a drink"
        )
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.context_precision <= 1.0
        assert 0.0 <= result.faithfulness <= 1.0

    def test_evaluate_empty(self):
        e = RAGASPlusEvaluator()
        result = e.evaluate(question="", answer="", contexts=[])
        assert result.overall_score < 0.3  # Low score for empty input

    def test_causal_consistency(self):
        e = RAGASPlusEvaluator()
        causal_edges = [("coffee", "alertness", 0.8)]
        result = e.evaluate(
            question="Does coffee help?",
            answer="Coffee causes alertness",
            contexts=["Coffee helps with alertness"],
            causal_edges=causal_edges
        )
        assert result.causal_consistency > 0.5

    def test_get_trend(self):
        e = RAGASPlusEvaluator()
        for i in range(15):
            e.evaluate(f"Q{i}", f"A{i} about topic", [f"Context {i}"])
        trend = e.get_trend()
        assert "recent_overall" in trend
        assert "trend" in trend


class TestAutoRollbackManager:
    def test_creation(self):
        m = AutoRollbackManager(threshold=0.15)
        assert m.threshold == 0.15

    def test_no_rollback_without_baseline(self):
        m = AutoRollbackManager()
        assert not m.record_score(0.5)

    def test_rollback_on_degradation(self):
        m = AutoRollbackManager(threshold=0.1)
        m.set_baseline(0.9)
        for _ in range(15):
            m.record_score(0.5)  # Significant degradation
        assert m.get_rollback_rate() > 0

    def test_no_rollback_when_stable(self):
        m = AutoRollbackManager(threshold=0.2)
        m.set_baseline(0.8)
        for _ in range(15):
            m.record_score(0.75)  # Small degradation, below threshold
        assert m.get_rollback_rate() == 0

    def test_get_state(self):
        m = AutoRollbackManager()
        m.set_baseline(0.8)
        state = m.get_state()
        assert state["baseline_score"] == 0.8


class TestFieldProduction:
    def test_field_with_production(self, config_v6):
        config_v6.production_mode = True
        config_v6.ragas_enabled = True
        config_v6.auto_rollback = True
        config_v6.shadow_mode = True
        field = RTMDKField(config_v6)
        assert field.ragas_evaluator is not None
        assert field.rollback_manager is not None
        assert field.shadow_evaluator is not None

    def test_evaluate_response(self, config_v6):
        config_v6.production_mode = True
        config_v6.ragas_enabled = True
        field = RTMDKField(config_v6)
        result = field.evaluate_response(
            question="What is coffee?",
            answer="Coffee is a drink",
            contexts=["Coffee is popular"]
        )
        assert isinstance(result, EvalResult)
        assert field.stats["evaluations"] >= 1

    def test_compare_shadow(self, config_v6):
        config_v6.production_mode = True
        config_v6.shadow_mode = True
        field = RTMDKField(config_v6)
        result = field.compare_shadow(0.8, 0.7)
        assert "shadow_value" in result
        assert field.stats["shadow_comparisons"] >= 1


# ============================================================================
# EXPORT/IMPORT v6
# ============================================================================

class TestExportImportV6:
    def test_export_import_ode(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            continuous_dynamics=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v6_ode.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.ode_dynamics is not None

    def test_export_import_full_v6(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            causal_topological=True, meta_adaptive=True,
            self_healing=True, differentiable=True,
            continuous_dynamics=True, production_mode=True,
            ragas_enabled=True, auto_rollback=True,
            shadow_mode=True, agent_orchestration=True,
            eval_mode="production", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(3):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        path = str(tmp_path / "v6_full.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.ode_dynamics is not None
        assert imported.field.causal_engine is not None
        assert imported.field.ragas_evaluator is not None
        assert imported.field.agent_planner is not None
        assert len(imported.field.nodes) == 3


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibilityV6:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx

    def test_v5_import_works(self, dummy_embedder, tmp_path):
        from rtmdk_memory_v5 import RTMDKConfig as V5Config, RTMDKMemory as V5Memory
        v5_config = V5Config(embedding_dim=768, latent_dim=64, enable_async=False)
        v5_memory = V5Memory(config=v5_config, embedder=dummy_embedder)
        v5_memory.save_context({"input": "v5 test", "session_id": "s1"}, {"output": "v5 out"})
        path = str(tmp_path / "v5_compat.json")
        v5_memory.export_field(path)
        v6_memory = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(v6_memory.field.nodes) == 1
