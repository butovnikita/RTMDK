"""
rtmdk_memory_v6.py
Резонансно-топологическая память — Версия 6.0

Фаза 7: Непрерывная динамика (Neural ODE/SDE)
  - NeuralODEDynamics: dX/dt = F(X, u(t)) + σ·dW
  - AdjointConsolidation: дифференцируемая консолидация
  - SmoothEvolution: плавная эволюция без дискретных разрывов

Фаза 8: Агентная оркестрация (Memory as Working Memory)
  - AgentPlanner: планирование целей и маршрутизация
  - HypothesisVerifier: проверка гипотез через do-calculus
  - ToolRouter: маршрутизация инструментов с верификацией

Фаза 9: Продакшен-стек & Автономная оценка
  - ShadowMode: теневой режим с фоллбэком
  - RAGASPlusEvaluator: RAGAS++ метрики
  - AutoRollback: автоматический откат при деградации
"""

from __future__ import annotations
import asyncio
import json
import math
import re
import time
import os
import hashlib
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Set, FrozenSet
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.integrate import odeint, solve_ivp
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ v6
# ============================================================================

class ConsolidationMode(Enum):
    DIALECTICAL = "dialectical"
    MERGE = "merge"
    PRUNE = "prune"

class Backend(Enum):
    NUMPY = "numpy"
    TORCH = "torch"

class ContextFormat(Enum):
    PLAIN = "plain"
    JSON = "json"
    YAML = "yaml"

class FieldHealth(Enum):
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    HEALING = "healing"

class EvalMode(Enum):
    PRODUCTION = "production"
    SHADOW = "shadow"
    EVALUATION = "evaluation"

@dataclass
class RTMDKConfig:
    embedding_dim: int = 768
    latent_dim: int = 64
    resonance_kernel: str = "gaussian_phase"
    phase_coupling: float = 0.3
    bandwidth: float = 1.0
    attraction_lr: float = 0.02
    phase_sync_lr: float = 0.01
    decay_rate: float = 0.998
    min_amplitude: float = 0.05
    tension_threshold: float = 0.25
    consolidation_mode: ConsolidationMode = ConsolidationMode.DIALECTICAL
    max_nodes: Optional[int] = 5000
    top_k: int = 5
    min_response: float = 0.1
    enable_async: bool = True
    log_level: str = "INFO"

    # ФАЗА 1
    context_format: ContextFormat = ContextFormat.PLAIN
    use_structured_prompt: bool = True
    adaptive_threshold: bool = False
    adaptive_window: int = 30
    learn_projection: bool = False
    projection_lr: float = 0.001
    projection_update_freq: int = 50
    pca_n_components: Optional[int] = None
    bm25_fallback: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ФАЗА 2
    soft_gates: bool = False
    gate_temperature: float = 0.15
    self_supervision: bool = False
    self_sup_threshold: float = 0.3
    self_sup_verify_after_consolidate: bool = False
    backend: Backend = Backend.NUMPY
    gpu_batch_size: int = 512
    l2_regularization: float = 0.0001
    false_merge_threshold: float = 0.4
    field_stability_window: int = 20
    enable_rollback: bool = False
    max_rollback_history: int = 50

    # ФАЗА 3
    multimodal: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])
    modality_phase_shifts: Dict[str, float] = field(default_factory=dict)
    use_hnsw: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    tda_monitoring: bool = False
    tda_check_freq: int = 50

    # ТРЕК 1: Дифференцируемое поле
    differentiable: bool = False
    learnable_bandwidth: bool = False
    learnable_phase_coupling: bool = False
    learnable_decay: bool = False
    gradient_clip: float = 1.0
    consolidation_loss_weight: float = 0.1

    # ФАЗА 5
    meta_adaptive: bool = False
    meta_adaptation_lr: float = 0.005
    kurtosis_target_min: float = 1.5
    kurtosis_target_max: float = 4.0
    self_healing: bool = False
    healing_check_freq: int = 25
    dead_zone_threshold: float = 0.15
    hyperconvergence_threshold: float = 0.05
    fragmentation_threshold: float = 0.6
    healing_strength: float = 0.1
    max_healing_nodes_per_step: int = 5

    # ФАЗА 6
    causal_topological: bool = False
    causal_discovery_min_samples: int = 20
    causal_p_threshold: float = 0.05
    do_calculus_validation: bool = True
    counterfactual_enabled: bool = False
    counterfactual_max_depth: int = 3
    contradiction_detection: bool = True
    contradiction_threshold: float = 0.3
    causal_adjustment_sets: bool = True

    # ФАЗА 7: Neural ODE/SDE
    continuous_dynamics: bool = False
    ode_solver: str = "RK45"
    ode_atol: float = 1e-6
    ode_rtol: float = 1e-5
    ode_time_horizon: float = 1.0
    ode_n_steps: int = 20
    ode_chunk_size: int = 256
    sde_noise_level: float = 0.01
    adjoint_enabled: bool = False
    response_smoothness_target: float = 0.92

    # ФАЗА 8: Агентная оркестрация
    agent_orchestration: bool = False
    max_plan_depth: int = 3
    max_tool_calls: int = 5
    tool_timeout: float = 15.0
    hypothesis_verification: bool = True
    verification_confidence_threshold: float = 0.7
    goal_directed_routing: bool = False

    # ФАЗА 9: Продакшен
    production_mode: bool = False
    eval_mode: EvalMode = EvalMode.PRODUCTION
    shadow_mode: bool = False
    shadow_fallback_threshold: float = 0.3
    auto_rollback: bool = False
    auto_rollback_threshold: float = 0.15
    eval_frequency: int = 100
    ragas_enabled: bool = False
    drift_detection: bool = False
    drift_window: int = 100
    drift_threshold: float = 0.05
    metrics_retention: int = 10000

    def __post_init__(self):
        logger.setLevel(getattr(logging, self.log_level.upper()))
        if not self.modality_phase_shifts:
            self.modality_phase_shifts = {
                "text": 0.0, "audio": np.pi / 3,
                "image": np.pi / 2, "video": np.pi,
            }
        if self.pca_n_components is None:
            self.pca_n_components = self.latent_dim


# ============================================================================
# ТИПЫ ДАННЫХ v6
# ============================================================================

@dataclass
class CausalEdge:
    source: str
    target: str
    strength: float
    confidence: float
    adjustment_set: List[str] = field(default_factory=list)
    evidence_count: int = 0
    is_contradicted: bool = False
    contradiction_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> CausalEdge:
        return cls(**data)


@dataclass
class ContradictionRecord:
    id: str
    effect_node: str
    causes: List[Tuple[str, float]]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""
    contradiction_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "effect_node": self.effect_node, "causes": self.causes,
            "timestamp": self.timestamp, "resolved": self.resolved,
            "resolution": self.resolution, "contradiction_reason": self.contradiction_reason,
        }


@dataclass
class CounterfactualResult:
    query: str
    intervention: Dict[str, Any]
    predicted_outcomes: List[Tuple[str, float]]
    confidence: float
    reasoning_path: List[str]
    assumptions: List[str]

    def to_dict(self) -> Dict:
        return {
            "query": self.query, "intervention": self.intervention,
            "predicted_outcomes": [{"node": n, "probability": p} for n, p in self.predicted_outcomes],
            "confidence": self.confidence, "reasoning_path": self.reasoning_path,
            "assumptions": self.assumptions,
        }


@dataclass
class AgentPlan:
    """План агента: цели → подзадачи → инструменты."""
    goal: str
    subtasks: List[Dict[str, Any]]
    tools_needed: List[str]
    estimated_steps: int
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = False
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Hypothesis:
    statement: str
    confidence: float
    evidence_nodes: List[str]
    causal_path: List[str]
    verified: bool = False
    verification_score: float = 0.0


@dataclass
class EvalResult:
    """Результат RAGAS++ оценки."""
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    causal_consistency: float = 0.0
    temporal_coherence: float = 0.0
    overall_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MemoryNode:
    id: str
    latent_pos: NDArray[np.float32]
    phase: float
    amplitude: float
    salience: float
    tension: float = 0.0
    soft_gate: float = 1.0
    content: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_resonated: float = 0.0
    lineage: List[str] = field(default_factory=list)
    modality: str = "text"
    self_sup_score: float = 1.0
    modal_weight: float = 1.0
    pre_consolidation_pos: Optional[NDArray[np.float32]] = None
    causal_parents: List[str] = field(default_factory=list)
    causal_strength: Dict[str, float] = field(default_factory=dict)
    gradient_cache: Optional[NDArray[np.float32]] = None
    is_healing: bool = False
    healing_origin: Optional[str] = None
    local_density: float = 0.0
    causal_effects: Dict[str, float] = field(default_factory=dict)
    do_interventions: Dict[str, NDArray] = field(default_factory=dict)
    is_causal_root: bool = False
    causal_context: Dict[str, Any] = field(default_factory=dict)
    # Фаза 7: ODE state
    velocity: Optional[NDArray[np.float32]] = None
    acceleration: Optional[NDArray[np.float32]] = None
    # Фаза 8: agent state
    goal_tags: List[str] = field(default_factory=list)
    tool_usage_count: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["latent_pos"] = self.latent_pos.tolist()
        if self.pre_consolidation_pos is not None:
            d["pre_consolidation_pos"] = self.pre_consolidation_pos.tolist()
        if self.gradient_cache is not None:
            d["gradient_cache"] = self.gradient_cache.tolist()
        if self.velocity is not None:
            d["velocity"] = self.velocity.tolist()
        if self.acceleration is not None:
            d["acceleration"] = self.acceleration.tolist()
        for k, v in self.do_interventions.items():
            if isinstance(v, np.ndarray):
                d["do_interventions"][k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryNode:
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        if data.get("pre_consolidation_pos"):
            data["pre_consolidation_pos"] = np.array(data["pre_consolidation_pos"], dtype=np.float32)
        if data.get("gradient_cache"):
            data["gradient_cache"] = np.array(data["gradient_cache"], dtype=np.float32)
        if data.get("velocity"):
            data["velocity"] = np.array(data["velocity"], dtype=np.float32)
        if data.get("acceleration"):
            data["acceleration"] = np.array(data["acceleration"], dtype=np.float32)
        for k, v in data.get("do_interventions", {}).items():
            if isinstance(v, list):
                data["do_interventions"][k] = np.array(v, dtype=np.float32)
        return cls(**data)


# ============================================================================
# ФАЗА 7: NEURAL ODE/SDE DYNAMICS
# ============================================================================

class NeuralODEDynamics:
    """
    Непрерывная динамика поля: dX/dt = F(X, u(t)) + σ·dW

    F(X, u) = -α·X + W·σ(X) + β·(u - X) + γ·∇_topology(X)

    где:
    -α·X: естественное затухание
    W·σ(X): нелинейная самоорганизация
    β·(u - X): притяжение к входу
    γ·∇_topology(X): топологическая регуляризация
    """

    def __init__(self, latent_dim: int, noise_level: float = 0.01,
                 time_horizon: float = 1.0, n_steps: int = 20,
                 chunk_size: int = 256, solver: str = "RK45",
                 atol: float = 1e-6, rtol: float = 1e-5):
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        # Обучаемые параметры динамики
        self.alpha = 0.1  # затухание
        self.beta = 0.05  # притяжение к входу
        self.gamma = 0.02  # топологическая регуляризация
        self.W = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01

        # История для smoothness tracking
        self._response_history: deque = deque(maxlen=100)
        self._state_history: List[NDArray] = []

    def _sigma(self, x: NDArray) -> NDArray:
        """Nonlinearity: tanh for bounded dynamics."""
        return np.tanh(x)

    def _dynamics(self, t: float, state: NDArray, input_signal: Optional[NDArray] = None,
                  topology_gradient: Optional[NDArray] = None) -> NDArray:
        """F(X, u(t)): continuous field dynamics."""
        n_nodes = len(state) // self.latent_dim
        if n_nodes == 0:
            return state

        X = state.reshape(n_nodes, self.latent_dim)

        # Damping
        damping = -self.alpha * X

        # Nonlinear self-organization
        nonlinear = self.W @ self._sigma(X.T)
        nonlinear = nonlinear.T

        # Attraction to input
        if input_signal is not None:
            u = input_signal.reshape(n_nodes, self.latent_dim)
            attraction = self.beta * (u - X)
        else:
            attraction = 0.0

        # Topology regularization
        if topology_gradient is not None:
            topo = self.gamma * topology_gradient.reshape(n_nodes, self.latent_dim)
        else:
            topo = 0.0

        dX = damping + nonlinear + attraction + topo
        return dX.flatten()

    def evolve(self, initial_state: NDArray, input_signal: Optional[NDArray] = None,
               topology_gradient: Optional[NDArray] = None,
               t_span: Optional[NDArray] = None) -> NDArray:
        """Эволюция поля через ODE solver."""
        if t_span is None:
            t_span = np.linspace(0, self.time_horizon, self.n_steps)

        # Chunking for large N
        n_nodes = len(initial_state) // self.latent_dim
        if n_nodes > self.chunk_size:
            return self._evolve_chunked(initial_state, input_signal, topology_gradient, t_span)

        def ode_func(t, state):
            return self._dynamics(t, state, input_signal, topology_gradient)

        solution = solve_ivp(
            ode_func, [t_span[0], t_span[-1]], initial_state.flatten(),
            t_eval=t_span, method=self.solver, atol=self.atol, rtol=self.rtol
        )

        if solution.success:
            trajectory = solution.y.T
        else:
            # Fallback to simpler solver
            trajectory = odeint(ode_func, initial_state.flatten(), t_span,
                                atol=self.atol * 10, rtol=self.rtol * 10)

        self._state_history.append(trajectory[-1].copy())
        return trajectory

    def _evolve_chunked(self, initial_state: NDArray, input_signal: Optional[NDArray],
                        topology_gradient: Optional[NDArray], t_span: NDArray) -> NDArray:
        """Батчинг для больших N."""
        n_nodes = len(initial_state) // self.latent_dim
        chunks = []
        for i in range(0, n_nodes, self.chunk_size):
            end = min(i + self.chunk_size, n_nodes)
            chunk_state = initial_state[i * self.latent_dim:end * self.latent_dim]
            chunk_input = input_signal[i * self.latent_dim:end * self.latent_dim] if input_signal is not None else None
            chunk_topo = topology_gradient[i * self.latent_dim:end * self.latent_dim] if topology_gradient is not None else None

            def ode_func(t, state):
                return self._dynamics(t, state, chunk_input, chunk_topo)

            sol = solve_ivp(ode_func, [t_span[0], t_span[-1]], chunk_state.flatten(),
                            t_eval=t_span, method=self.solver, atol=self.atol, rtol=self.rtol)
            if sol.success:
                chunks.append(sol.y.T)
            else:
                chunks.append(odeint(ode_func, chunk_state.flatten(), t_span))

        return np.concatenate(chunks, axis=1)

    def evolve_with_noise(self, initial_state: NDArray, input_signal: Optional[NDArray] = None,
                          topology_gradient: Optional[NDArray] = None,
                          dt: float = 0.05) -> NDArray:
        """SDE: dX = F(X,u)dt + σ·dW (Euler-Maruyama with chunking)."""
        n_steps = int(self.time_horizon / dt)
        state = initial_state.flatten().copy()
        trajectory = [state.copy()]

        n_nodes = len(state) // self.latent_dim
        for _ in range(n_steps):
            deterministic = self._dynamics(0, state, input_signal, topology_gradient) * dt
            noise = self.noise_level * np.random.randn(len(state)) * np.sqrt(dt)
            state = state + deterministic + noise
            trajectory.append(state.copy())

        self._state_history.append(trajectory[-1].copy())
        return np.array(trajectory)

    def compute_topology_gradient(self, nodes: Dict[str, MemoryNode]) -> Optional[NDArray]:
        """Вычислить градиент топологической регуляризации."""
        if len(nodes) < 2:
            return None

        node_ids = list(nodes.keys())
        positions = np.array([nodes[nid].latent_pos for nid in node_ids])
        n = len(positions)

        # Gradient: push apart close nodes, pull together distant ones
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)

        gradient = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < 2.0:
                    direction = (positions[i] - positions[j]) / (dists[i, j] + 1e-8)
                    gradient[i] += direction * 0.01
                    gradient[j] -= direction * 0.01

        return gradient.flatten()

    def compute_response_smoothness(self) -> float:
        """response_smoothness = 1 - std(responses_over_time)."""
        if len(self._response_history) < 2:
            return 1.0
        responses = np.array(self._response_history)
        std = np.std(responses)
        return max(0.0, 1.0 - std)

    def record_response(self, response: float):
        self._response_history.append(response)

    def get_state(self) -> Dict:
        return {
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "W": self.W.tolist(), "noise_level": self.noise_level,
            "smoothness": self.compute_response_smoothness(),
        }

    def load_state(self, state: Dict):
        self.alpha = state.get("alpha", self.alpha)
        self.beta = state.get("beta", self.beta)
        self.gamma = state.get("gamma", self.gamma)
        if "W" in state:
            self.W = np.array(state["W"], dtype=np.float32)
        self.noise_level = state.get("noise_level", self.noise_level)


# ============================================================================
# ФАЗА 8: АГЕНТНАЯ ОРКЕСТРАЦИЯ
# ============================================================================

class AgentPlanner:
    """Планирование целей и маршрутизация инструментов."""

    def __init__(self, max_depth: int = 3, max_tool_calls: int = 5,
                 tool_timeout: float = 15.0):
        self.max_depth = max_depth
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout
        self._visited_tools: Set[str] = set()
        self._call_count = 0

    def create_plan(self, goal: str, available_tools: List[str],
                    context: Dict[str, Any]) -> AgentPlan:
        """Создать план достижения цели."""
        subtasks = self._decompose_goal(goal, context)
        tools_needed = self._select_tools(goal, subtasks, available_tools)

        return AgentPlan(
            goal=goal,
            subtasks=subtasks,
            tools_needed=tools_needed,
            estimated_steps=len(subtasks),
            confidence=self._estimate_confidence(goal, subtasks, tools_needed),
            reasoning=f"Decomposed '{goal}' into {len(subtasks)} subtasks"
        )

    def _decompose_goal(self, goal: str, context: Dict) -> List[Dict[str, Any]]:
        """Декомпозировать цель на подзадачи."""
        # Simple heuristic decomposition
        subtasks = []

        # 1. Retrieve relevant memories
        subtasks.append({
            "type": "retrieve",
            "description": f"Find memories related to: {goal}",
            "priority": 1,
        })

        # 2. Verify hypotheses
        if context.get("hypothesis_verification", False):
            subtasks.append({
                "type": "verify",
                "description": "Verify causal hypotheses",
                "priority": 2,
            })

        # 3. Synthesize answer
        subtasks.append({
            "type": "synthesize",
            "description": f"Synthesize response for: {goal}",
            "priority": 3,
        })

        return subtasks[:self.max_depth]

    def _select_tools(self, goal: str, subtasks: List[Dict],
                      available_tools: List[str]) -> List[str]:
        """Выбрать инструменты для подзадач."""
        selected = []
        for task in subtasks:
            task_type = task.get("type", "")
            for tool in available_tools:
                if task_type in tool.lower() and tool not in selected:
                    selected.append(tool)
        return selected[:self.max_tool_calls]

    def _estimate_confidence(self, goal: str, subtasks: List[Dict],
                             tools: List[str]) -> float:
        """Оценить уверенность в плане."""
        base = 0.5
        base += min(0.2, len(subtasks) * 0.05)
        base += min(0.2, len(tools) * 0.05)
        base += 0.1 if len(subtasks) <= self.max_depth else -0.1
        return min(1.0, max(0.0, base))

    def reset(self):
        self._visited_tools.clear()
        self._call_count = 0

    def can_call_tool(self, tool_name: str) -> bool:
        """Проверить, можно ли вызвать инструмент."""
        if tool_name in self._visited_tools and tool_name != "retrieve":
            return False
        return self._call_count < self.max_tool_calls

    def record_tool_call(self, tool_name: str):
        self._visited_tools.add(tool_name)
        self._call_count += 1


class HypothesisVerifier:
    """Проверка гипотез через do-calculus и каузальную структуру."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def verify(self, hypothesis: str, causal_engine: Any,
               active_nodes: List[str]) -> Hypothesis:
        """Верифицировать гипотезу через каузальный анализ."""
        evidence_nodes = []
        causal_path = []
        confidence = 0.5

        if causal_engine and hasattr(causal_engine, 'causal_effects'):
            # Check if hypothesis matches known causal effects
            for (cause, effect), edge in causal_engine.causal_effects.items():
                if cause in active_nodes or effect in active_nodes:
                    evidence_nodes.append(cause)
                    evidence_nodes.append(effect)
                    causal_path.append(f"{cause} → {effect} (P={edge.strength:.2f})")
                    confidence = max(confidence, edge.strength * edge.confidence)

        verified = confidence >= self.confidence_threshold

        return Hypothesis(
            statement=hypothesis,
            confidence=confidence,
            evidence_nodes=list(set(evidence_nodes)),
            causal_path=causal_path,
            verified=verified,
            verification_score=confidence,
        )


class ToolRouter:
    """Маршрутизация инструментов с верификацией."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._tool_registry: Dict[str, Callable] = {}
        self._call_history: deque = deque(maxlen=100)

    def register_tool(self, name: str, func: Callable):
        self._tool_registry[name] = func

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """Выполнить инструмент с таймаутом и логированием."""
        t0 = time.time()
        call = ToolCall(tool_name=tool_name, arguments=arguments)

        if tool_name not in self._tool_registry:
            call.error = f"Tool '{tool_name}' not registered"
            call.latency_ms = (time.time() - t0) * 1000
            return call

        try:
            func = self._tool_registry[tool_name]
            result = func(**arguments)
            call.result = result
            call.success = True
        except Exception as e:
            call.error = str(e)

        call.latency_ms = (time.time() - t0) * 1000
        self._call_history.append(call)
        return call

    def get_misuse_rate(self) -> float:
        """Доля неудачных вызовов инструментов."""
        if not self._call_history:
            return 0.0
        failures = sum(1 for c in self._call_history if not c.success)
        return failures / len(self._call_history)


# ============================================================================
# ФАЗА 9: ПРОДАКШЕН-СТЕК
# ============================================================================

class ShadowModeEvaluator:
    """Теневой режим: параллельная оценка без влияния на продакшен."""

    def __init__(self, fallback_threshold: float = 0.3):
        self.fallback_threshold = fallback_threshold
        self._shadow_results: List[Dict] = []
        self._production_results: List[Dict] = []
        self._fallback_count = 0
        self._total_comparisons = 0

    def compare(self, shadow_output: Any, production_output: Any,
                metric_name: str = "response_quality") -> Dict[str, Any]:
        """Сравнить теневой и продакшен результаты."""
        self._shadow_results.append({"value": shadow_output, "metric": metric_name})
        self._production_results.append({"value": production_output, "metric": metric_name})
        self._total_comparisons += 1

        diff = abs(float(shadow_output) - float(production_output))
        is_better = shadow_output > production_output

        if diff > self.fallback_threshold:
            self._fallback_count += 1

        return {
            "shadow_value": shadow_output,
            "production_value": production_output,
            "difference": diff,
            "shadow_better": is_better,
            "fallback_triggered": diff > self.fallback_threshold,
        }

    def get_correlation(self) -> float:
        """Корреляция между shadow и production."""
        if len(self._shadow_results) < 3:
            return 0.0
        shadow_vals = [r["value"] for r in self._shadow_results]
        prod_vals = [r["value"] for r in self._production_results]
        if np.std(shadow_vals) < 1e-8 or np.std(prod_vals) < 1e-8:
            return 1.0
        corr = np.corrcoef(shadow_vals, prod_vals)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def get_fallback_rate(self) -> float:
        return self._fallback_count / max(self._total_comparisons, 1)


class RAGASPlusEvaluator:
    """RAGAS++ метрики: context_precision, recall, faithfulness, causal_consistency."""

    def __init__(self):
        self._eval_history: List[EvalResult] = []

    def evaluate(self, question: str, answer: str, contexts: List[str],
                 ground_truth: Optional[str] = None,
                 causal_edges: Optional[List[Tuple[str, str, float]]] = None) -> EvalResult:
        """Полная RAGAS++ оценка."""
        result = EvalResult()

        # Context precision
        result.context_precision = self._compute_context_precision(question, contexts)

        # Context recall
        if ground_truth:
            result.context_recall = self._compute_context_recall(ground_truth, contexts)
        else:
            result.context_recall = result.context_precision * 0.8

        # Answer relevance
        result.answer_relevance = self._compute_answer_relevance(question, answer)

        # Faithfulness
        result.faithfulness = self._compute_faithfulness(answer, contexts)

        # Causal consistency
        if causal_edges:
            result.causal_consistency = self._compute_causal_consistency(answer, causal_edges)
        else:
            result.causal_consistency = 0.5

        # Temporal coherence
        result.temporal_coherence = self._compute_temporal_coherence(contexts)

        # Overall
        weights = [0.2, 0.15, 0.2, 0.2, 0.15, 0.1]
        scores = [result.context_precision, result.context_recall,
                  result.answer_relevance, result.faithfulness,
                  result.causal_consistency, result.temporal_coherence]
        result.overall_score = sum(w * s for w, s in zip(weights, scores))

        self._eval_history.append(result)
        return result

    def _compute_context_precision(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        # Heuristic: keyword overlap
        q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        if not q_tokens:
            return 0.0
        precision_scores = []
        for ctx in contexts:
            c_tokens = set(re.findall(r'\b\w+\b', ctx.lower()))
            if c_tokens:
                precision_scores.append(len(q_tokens & c_tokens) / len(q_tokens))
        return float(np.mean(precision_scores)) if precision_scores else 0.0

    def _compute_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        gt_tokens = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        if not gt_tokens:
            return 0.0
        all_ctx_tokens = set()
        for ctx in contexts:
            all_ctx_tokens.update(re.findall(r'\b\w+\b', ctx.lower()))
        if not all_ctx_tokens:
            return 0.0
        return len(gt_tokens & all_ctx_tokens) / len(gt_tokens)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens)

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        if not a_tokens:
            return 0.0
        all_ctx = " ".join(contexts).lower()
        ctx_tokens = set(re.findall(r'\b\w+\b', all_ctx))
        if not ctx_tokens:
            return 0.5
        return len(a_tokens & ctx_tokens) / len(a_tokens)

    def _compute_causal_consistency(self, answer: str,
                                     causal_edges: List[Tuple[str, str, float]]) -> float:
        if not causal_edges:
            return 0.5
        answer_lower = answer.lower()
        consistent = 0
        for cause, effect, strength in causal_edges:
            if cause.lower() in answer_lower and effect.lower() in answer_lower:
                consistent += strength
        return consistent / len(causal_edges) if causal_edges else 0.5

    def _compute_temporal_coherence(self, contexts: List[str]) -> float:
        if len(contexts) < 2:
            return 1.0
        # Check if contexts have temporal markers
        temporal_markers = ["then", "after", "before", "next", "later", "previously",
                           "затем", "после", "до", "далее", "потом", "ранее"]
        coherent = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            if any(m in ctx_lower for m in temporal_markers):
                coherent += 1
        return coherent / len(contexts)

    def get_trend(self) -> Dict[str, float]:
        if len(self._eval_history) < 5:
            return {}
        recent = self._eval_history[-10:]
        older = self._eval_history[-20:-10] if len(self._eval_history) >= 20 else self._eval_history[:5]
        return {
            "recent_overall": np.mean([e.overall_score for e in recent]),
            "older_overall": np.mean([e.overall_score for e in older]),
            "trend": "improving" if np.mean([e.overall_score for e in recent]) > np.mean([e.overall_score for e in older]) else "degrading",
        }


class AutoRollbackManager:
    """Автоматический откат при деградации."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self._baseline_score: Optional[float] = None
        self._recent_scores: deque = deque(maxlen=50)
        self._rollback_count = 0
        self._last_rollback_time: float = 0
        self._cooldown_period: float = 300.0  # 5 minutes

    def set_baseline(self, score: float):
        self._baseline_score = score

    def record_score(self, score: float) -> bool:
        """Записать оценку и проверить необходимость отката."""
        self._recent_scores.append(score)

        if self._baseline_score is None or len(self._recent_scores) < 10:
            return False

        # Cooldown
        if time.time() - self._last_rollback_time < self._cooldown_period:
            return False

        recent_mean = np.mean(self._recent_scores)
        degradation = self._baseline_score - recent_mean

        if degradation > self.threshold:
            self._rollback_count += 1
            self._last_rollback_time = time.time()
            return True

        return False

    def get_rollback_rate(self) -> float:
        return self._rollback_count / max(len(self._recent_scores), 1)

    def get_state(self) -> Dict:
        return {
            "baseline_score": self._baseline_score,
            "recent_mean": float(np.mean(self._recent_scores)) if self._recent_scores else 0,
            "rollback_count": self._rollback_count,
            "rollback_rate": self.get_rollback_rate(),
        }


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КОМПОНЕНТЫ (из v5)
# ============================================================================

class MetaAdaptiveKernel:
    def __init__(self, base_bandwidth: float = 1.0, base_phase_coupling: float = 0.3,
                 adaptation_lr: float = 0.005, kurtosis_target_min: float = 1.5,
                 kurtosis_target_max: float = 4.0):
        self.base_bandwidth = base_bandwidth
        self.base_phase_coupling = base_phase_coupling
        self.adaptation_lr = adaptation_lr
        self.kurtosis_target_min = kurtosis_target_min
        self.kurtosis_target_max = kurtosis_target_max
        self.effective_bandwidth = base_bandwidth
        self.effective_phase_coupling = base_phase_coupling
        self._response_history: deque = deque(maxlen=100)
        self._semantic_density: deque = deque(maxlen=50)
        self._uncertainty: deque = deque(maxlen=20)
        self._kurtosis_history: deque = deque(maxlen=50)

    def record_response(self, response: float):
        self._response_history.append(response)

    def record_semantic_density(self, density: float):
        self._semantic_density.append(density)

    def record_uncertainty(self, entropy: float):
        self._uncertainty.append(entropy)

    def compute_resonance_kurtosis(self) -> float:
        if len(self._response_history) < 4:
            return 3.0
        responses = np.array(self._response_history)
        if np.std(responses) < 1e-8:
            return 3.0
        return float(scipy_stats.kurtosis(responses) + 3.0)

    def adapt(self):
        kurtosis = self.compute_resonance_kurtosis()
        self._kurtosis_history.append(kurtosis)
        if kurtosis < self.kurtosis_target_min:
            self.effective_bandwidth *= (1.0 - self.adaptation_lr)
        elif kurtosis > self.kurtosis_target_max:
            self.effective_bandwidth *= (1.0 + self.adaptation_lr)
        if self._semantic_density:
            density = np.mean(self._semantic_density)
            if density > 0.7:
                self.effective_phase_coupling = min(0.9, self.effective_phase_coupling + self.adaptation_lr * 0.5)
            elif density < 0.2:
                self.effective_phase_coupling = max(0.05, self.effective_phase_coupling - self.adaptation_lr * 0.5)
        if self._uncertainty:
            uncertainty = np.mean(self._uncertainty)
            if uncertainty > 1.5:
                self.effective_bandwidth *= (1.0 + self.adaptation_lr)
        self.effective_bandwidth = max(0.1, min(10.0, self.effective_bandwidth))
        self.effective_phase_coupling = max(0.0, min(1.0, self.effective_phase_coupling))

    def get_bandwidth(self) -> float:
        return self.effective_bandwidth

    def get_phase_coupling(self) -> float:
        return self.effective_phase_coupling

    def get_state(self) -> Dict:
        return {
            "base_bandwidth": self.base_bandwidth, "base_phase_coupling": self.base_phase_coupling,
            "effective_bandwidth": self.effective_bandwidth, "effective_phase_coupling": self.effective_phase_coupling,
            "kurtosis": self.compute_resonance_kurtosis(),
            "avg_density": float(np.mean(self._semantic_density)) if self._semantic_density else 0,
            "avg_uncertainty": float(np.mean(self._uncertainty)) if self._uncertainty else 0,
        }

    def load_state(self, state: Dict):
        self.base_bandwidth = state.get("base_bandwidth", self.base_bandwidth)
        self.base_phase_coupling = state.get("base_phase_coupling", self.base_phase_coupling)
        self.effective_bandwidth = state.get("effective_bandwidth", self.base_bandwidth)
        self.effective_phase_coupling = state.get("effective_phase_coupling", self.base_phase_coupling)


class TopologyHealer:
    def __init__(self, dead_zone_threshold: float = 0.15, hyperconvergence_threshold: float = 0.05,
                 fragmentation_threshold: float = 0.6, healing_strength: float = 0.1,
                 max_healing_nodes: int = 5):
        self.dead_zone_threshold = dead_zone_threshold
        self.hyperconvergence_threshold = hyperconvergence_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.healing_strength = healing_strength
        self.max_healing_nodes = max_healing_nodes
        self._health_history: deque = deque(maxlen=100)

    def detect_dead_zones(self, nodes: Dict[str, MemoryNode]) -> List[str]:
        if len(nodes) < 3:
            return []
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        threshold = np.median(min_dists) * (1.0 + self.dead_zone_threshold * 5)
        return [nid for i, nid in enumerate(nodes) if min_dists[i] > threshold]

    def detect_hyperconvergence(self, nodes: Dict[str, MemoryNode]) -> bool:
        if len(nodes) < 3:
            return False
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        return np.mean(dists[dists < np.inf]) < self.hyperconvergence_threshold

    def detect_fragmentation(self, nodes: Dict[str, MemoryNode], radius: float = 2.0) -> float:
        if len(nodes) < 2:
            return 0.0
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        isolated = np.sum(np.all(dists > radius, axis=1))
        return float(isolated / len(nodes))

    def compute_field_health(self, nodes: Dict[str, MemoryNode]) -> Tuple[FieldHealth, Dict]:
        diagnostics = {}
        dead = self.detect_dead_zones(nodes)
        diagnostics["dead_zones"] = len(dead)
        diagnostics["dead_zone_nodes"] = dead
        hyperconv = self.detect_hyperconvergence(nodes)
        diagnostics["hyperconvergence"] = hyperconv
        frag = self.detect_fragmentation(nodes)
        diagnostics["fragmentation"] = frag
        if len(nodes) >= 3:
            positions = np.array([n.latent_pos for n in nodes.values()])
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            valid = dists[dists < np.inf]
            diagnostics["avg_pairwise_dist"] = float(np.mean(valid))
            diagnostics["std_pairwise_dist"] = float(np.std(valid))
            diagnostics["density_cv"] = float(np.std(valid) / max(np.mean(valid), 1e-8))
        else:
            diagnostics["avg_pairwise_dist"] = 0.0
            diagnostics["density_cv"] = 0.0
        if hyperconv or frag > 0.8:
            health = FieldHealth.CRITICAL
        elif len(dead) > len(nodes) * 0.3 or frag > self.fragmentation_threshold:
            health = FieldHealth.DEGRADED
        else:
            health = FieldHealth.STABLE
        self._health_history.append(health.value)
        diagnostics["health"] = health.value
        diagnostics["stable_fraction"] = (
            sum(1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1))
        return health, diagnostics

    def heal_dead_zones(self, nodes: Dict[str, MemoryNode], dead_ids: List[str]) -> List[Dict]:
        healed = []
        alive_ids = [nid for nid in nodes if nid not in dead_ids]
        if not alive_ids or not dead_ids:
            return healed
        alive_positions = np.array([nodes[nid].latent_pos for nid in alive_ids])
        for dead_id in dead_ids[:self.max_healing_nodes]:
            dead_node = nodes[dead_id]
            dists = np.linalg.norm(alive_positions - dead_node.latent_pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_id = alive_ids[nearest_idx]
            old_pos = dead_node.latent_pos.copy()
            dead_node.latent_pos = ((1.0 - self.healing_strength) * old_pos + self.healing_strength * nodes[nearest_id].latent_pos).astype(np.float32)
            dead_node.is_healing = True
            dead_node.healing_origin = nearest_id
            dead_node.salience = max(dead_node.salience, 0.1)
            dead_node.amplitude = max(dead_node.amplitude, 0.1)
            healed.append({"node_id": dead_id, "from": old_pos.tolist(), "to": dead_node.latent_pos.tolist(), "type": "dead_zone"})
        return healed

    def heal_hyperconvergence(self, nodes: Dict[str, MemoryNode]) -> List[Dict]:
        healed = []
        if len(nodes) < 3:
            return healed
        positions = np.array([n.latent_pos for n in nodes.values()])
        centroid = np.mean(positions, axis=0)
        for nid in list(nodes.keys())[:self.max_healing_nodes]:
            node = nodes[nid]
            direction = node.latent_pos - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.random.randn(len(centroid)).astype(np.float32)
                norm = 1.0
            direction = direction / norm
            old_pos = node.latent_pos.copy()
            node.latent_pos = (old_pos + self.healing_strength * direction).astype(np.float32)
            node.is_healing = True
            node.healing_origin = "hyperconvergence"
            healed.append({"node_id": nid, "from": old_pos.tolist(), "to": node.latent_pos.tolist(), "type": "hyperconvergence"})
        return healed

    def heal_fragmentation(self, nodes: Dict[str, MemoryNode], isolated_ids: List[str]) -> List[Dict]:
        healed = []
        non_isolated = [nid for nid in nodes if nid not in isolated_ids]
        if not non_isolated or not isolated_ids:
            return healed
        non_iso_positions = np.array([nodes[nid].latent_pos for nid in non_isolated])
        centroid = np.mean(non_iso_positions, axis=0)
        for iso_id in isolated_ids[:self.max_healing_nodes]:
            node = nodes[iso_id]
            old_pos = node.latent_pos.copy()
            node.latent_pos = ((1.0 - self.healing_strength) * old_pos + self.healing_strength * centroid).astype(np.float32)
            node.is_healing = True
            node.healing_origin = "fragmentation"
            healed.append({"node_id": iso_id, "from": old_pos.tolist(), "to": node.latent_pos.tolist(), "type": "fragmentation"})
        return healed

    def get_state(self) -> Dict:
        return {"health_history": list(self._health_history),
                "stable_fraction": sum(1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1)}

    def load_state(self, state: Dict):
        self._health_history = deque(state.get("health_history", []), maxlen=100)


class CausalInferenceEngine:
    def __init__(self, min_samples: int = 20, p_threshold: float = 0.05,
                 adjustment_sets_enabled: bool = True):
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.adjustment_sets_enabled = adjustment_sets_enabled
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.ancestors: Dict[str, Set[str]] = defaultdict(set)
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._conditional_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._node_counts: Dict[str, int] = defaultdict(int)
        self._total_observations = 0
        self.causal_effects: Dict[Tuple[str, str], CausalEdge] = {}
        self.contradictions: Dict[str, ContradictionRecord] = {}
        self._contradiction_counter = 0
        self._counterfactual_cache: Dict[str, CounterfactualResult] = {}

    def record_cooccurrence(self, a: str, b: str):
        self._cooccurrence[(a, b)] += 1
        self._cooccurrence[(b, a)] += 1
        self._node_counts[a] += 1
        self._node_counts[b] += 1
        self._total_observations += 1

    def record_observation(self, active_nodes: List[str], context: Optional[Dict] = None):
        self._total_observations += 1
        for node in active_nodes:
            self._node_counts[node] += 1
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i+1:]:
                self._cooccurrence[(a, b)] += 1
                self._cooccurrence[(b, a)] += 1
                if context:
                    for ctx_key, ctx_val in context.items():
                        self._conditional_counts[(a, b, f"{ctx_key}={ctx_val}")] += 1

    def discover_causal_structure(self) -> Dict[str, Set[str]]:
        nodes = list(self._node_counts.keys())
        if len(nodes) < 3 or self._total_observations < self.min_samples:
            return dict(self.parents)
        skeleton: Dict[str, Set[str]] = defaultdict(set)
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if self._test_independence(a, b, set()):
                    continue
                skeleton[a].add(b)
                skeleton[b].add(a)
        new_parents: Dict[str, Set[str]] = defaultdict(set)
        new_children: Dict[str, Set[str]] = defaultdict(set)
        for z in nodes:
            neighbors = list(skeleton.get(z, set()))
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if y not in skeleton.get(x, set()):
                        new_parents[z].add(x)
                        new_parents[z].add(y)
                        new_children[x].add(z)
                        new_children[y].add(z)
        self.parents = new_parents
        self.children = new_children
        self._compute_ancestors()
        return dict(self.parents)

    def _test_independence(self, a: str, b: str, cond_set: Set[str]) -> bool:
        n_ab = self._cooccurrence.get((a, b), 0)
        n_a = self._node_counts.get(a, 0)
        n_b = self._node_counts.get(b, 0)
        n = max(self._total_observations, 1)
        if n_a < 3 or n_b < 3 or n_ab < 2:
            return True
        if not cond_set:
            expected = (n_a / n) * (n_b / n) * n
            if expected < 5:
                return True
            chi2 = (n_ab - expected) ** 2 / expected
            return chi2 < 3.84
        return True

    def _compute_ancestors(self):
        for node in self.parents:
            self.ancestors[node] = self._get_ancestors(node, set())

    def _get_ancestors(self, node: str, visited: Set[str]) -> Set[str]:
        if node in visited:
            return set()
        visited.add(node)
        ancestors = set()
        for parent in self.parents.get(node, set()):
            ancestors.add(parent)
            ancestors.update(self._get_ancestors(parent, visited))
        return ancestors

    def _get_descendants(self, node: str, visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)
        descendants = set()
        for child in self.children.get(node, set()):
            descendants.add(child)
            descendants.update(self._get_descendants(child, visited))
        return descendants

    def compute_do_probability(self, effect: str, intervention: str,
                               evidence: Optional[Dict[str, Any]] = None) -> float:
        edge = self.causal_effects.get((intervention, effect))
        if edge:
            return edge.strength
        return self._naive_causal_estimate(intervention, effect)

    def _naive_causal_estimate(self, cause: str, effect: str) -> float:
        n_cause = self._node_counts.get(cause, 0)
        n_both = self._cooccurrence.get((cause, effect), 0)
        if n_cause < 3:
            return 0.5
        return min(1.0, n_both / n_cause)

    def _find_adjustment_set(self, cause: str, effect: str) -> Set[str]:
        if not self.adjustment_sets_enabled:
            return set()
        parents_of_cause = self.parents.get(cause, set())
        descendants = self._get_descendants(cause)
        return parents_of_cause - descendants

    def _validate_do_calculus(self, effect: str, intervention: str) -> bool:
        z_set = self._find_adjustment_set(intervention, effect)
        has_frontdoor = self._has_frontdoor_path(intervention, effect)
        descendants = self._get_descendants(intervention)
        return bool(z_set) or has_frontdoor or effect in descendants

    def _has_frontdoor_path(self, cause: str, effect: str) -> bool:
        for mediator in self.children.get(cause, set()):
            if effect in self.children.get(mediator, set()):
                return True
        return False

    def detect_contradictions(self, threshold: float = 0.3) -> List[ContradictionRecord]:
        new_contradictions = []
        effect_causes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (cause, effect), edge in self.causal_effects.items():
            if edge.strength > 0.1:
                effect_causes[effect].append((cause, edge.strength))
        for effect_node, causes in effect_causes.items():
            if len(causes) < 2:
                continue
            for i, (cause_a, strength_a) in enumerate(causes):
                for cause_b, strength_b in causes[i+1:]:
                    cooc = self._cooccurrence.get((cause_a, cause_b), 0)
                    n_a = self._node_counts.get(cause_a, 0)
                    n_b = self._node_counts.get(cause_b, 0)
                    if n_a > 0 and n_b > 0:
                        expected = (n_a / self._total_observations) * (n_b / self._total_observations) * self._total_observations
                        if expected > 0 and cooc / expected < (1.0 - threshold):
                            self._contradiction_counter += 1
                            record = ContradictionRecord(
                                id=f"contr_{self._contradiction_counter}",
                                effect_node=effect_node,
                                causes=[(cause_a, strength_a), (cause_b, strength_b)],
                                contradiction_reason=f"Causes {cause_a} and {cause_b} are negatively correlated"
                            )
                            self.contradictions[record.id] = record
                            new_contradictions.append(record)
                            if (cause_a, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_a, effect_node)].is_contradicted = True
                            if (cause_b, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_b, effect_node)].is_contradicted = True
        return new_contradictions

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None,
                             max_depth: int = 3) -> CounterfactualResult:
        query_str = f"do({intervention})|{query_nodes}"
        if query_str in self._counterfactual_cache:
            return self._counterfactual_cache[query_str]
        outcomes = []
        reasoning_path = []
        for target in query_nodes[:max_depth]:
            if target in intervention:
                outcomes.append((target, 1.0))
                reasoning_path.append(f"{target} is directly set")
                continue
            best_prob = 0.0
            best_path = ""
            for int_var, int_val in intervention.items():
                prob = self.compute_do_probability(target, int_var)
                if prob > best_prob:
                    best_prob = prob
                    best_path = f"do({int_var}) → {target} (P={prob:.3f})"
            if best_path:
                outcomes.append((target, best_prob))
                reasoning_path.append(best_path)
            else:
                outcomes.append((target, 0.5))
                reasoning_path.append(f"No causal path to {target}")
        confidence = np.mean([p for _, p in outcomes]) if outcomes else 0.5
        result = CounterfactualResult(
            query=query_str, intervention=intervention, predicted_outcomes=outcomes,
            confidence=float(confidence), reasoning_path=reasoning_path, assumptions=[])
        self._counterfactual_cache[query_str] = result
        return result

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        result = {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}
        common_targets = set(self.children.get(node_a, set())) & set(self.children.get(node_b, set()))
        for target in common_targets:
            edge_a = self.causal_effects.get((node_a, target))
            edge_b = self.causal_effects.get((node_b, target))
            if edge_a and edge_b:
                diff = abs(edge_a.strength - edge_b.strength)
                if diff > 0.4:
                    result["safe"] = False
                    result["causal_conflicts"].append({"target": target, "effect_a": edge_a.strength, "effect_b": edge_b.strength, "difference": diff})
                    result["reasons"].append(f"Opposing effects on {target}")
        if node_b in self.children.get(node_a, set()) or node_a in self.children.get(node_b, set()):
            result["safe"] = False
            result["reasons"].append(f"Causal relationship exists")
            result["recommendation"] = "preserve_separate"
        if node_a in self.ancestors.get(node_b, set()) or node_b in self.ancestors.get(node_a, set()):
            result["safe"] = False
            result["reasons"].append("Merging would create causal cycle")
            result["recommendation"] = "preserve_separate"
        for cid, record in self.contradictions.items():
            if record.resolved:
                continue
            causes = [c for c, _ in record.causes]
            if node_a in causes and node_b in causes:
                result["safe"] = False
                result["reasons"].append(f"Unresolved contradiction: {cid}")
                result["recommendation"] = "resolve_contradiction_first"
        return result

    def do_intervention(self, node_id: str, new_pos: NDArray):
        pass  # Placeholder

    def clear_interventions(self):
        pass

    def get_state(self) -> Dict:
        return {
            "parents": {k: list(v) for k, v in self.parents.items()},
            "children": {k: list(v) for k, v in self.children.items()},
            "causal_effects": {f"{k[0]}->{k[1]}": v.to_dict() for k, v in self.causal_effects.items()},
            "contradictions": {k: v.to_dict() for k, v in self.contradictions.items()},
            "node_counts": dict(self._node_counts),
            "total_observations": self._total_observations,
        }

    def load_state(self, state: Dict):
        self.parents = defaultdict(set, {k: set(v) for k, v in state.get("parents", {}).items()})
        self.children = defaultdict(set, {k: set(v) for k, v in state.get("children", {}).items()})
        self._node_counts = defaultdict(int, state.get("node_counts", {}))
        self._total_observations = state.get("total_observations", 0)
        for key, edge_data in state.get("causal_effects", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                self.causal_effects[(parts[0], parts[1])] = CausalEdge.from_dict(edge_data)
        for cid, record_data in state.get("contradictions", {}).items():
            self.contradictions[cid] = ContradictionRecord(
                id=record_data["id"], effect_node=record_data["effect_node"],
                causes=record_data["causes"], timestamp=record_data["timestamp"],
                resolved=record_data["resolved"], resolution=record_data["resolution"],
                contradiction_reason=record_data.get("contradiction_reason", ""))
        self._compute_ancestors()


class IncPCAProjection:
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.001, update_freq: int = 50, l2_reg: float = 0.0001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.update_freq = update_freq
        self.l2_reg = l2_reg
        self.projection = np.random.randn(input_dim, latent_dim).astype(np.float32) * 0.1
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.buffer: List[NDArray] = []
        self.n_samples = 0
        self._try_sklearn()

    def _try_sklearn(self):
        try:
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(n_components=self.latent_dim, batch_size=min(64, self.update_freq))
            self.use_sklearn = True
            self._ipca_fitted = False
        except ImportError:
            self.use_sklearn = False

    def update(self, embedding: NDArray) -> NDArray:
        self.n_samples += 1
        self.buffer.append(embedding.copy())
        if len(self.buffer) >= self.update_freq:
            batch = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            if self.use_sklearn:
                self.ipca.partial_fit(batch)
                self._ipca_fitted = True
                self.projection = self.ipca.components_.T.astype(np.float32)
                self.mean = self.ipca.mean_.astype(np.float32)
            else:
                alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                self.mean += alpha * (batch.mean(axis=0) - self.mean)
                for emb in batch:
                    centered = emb - self.mean
                    latent = centered @ self.projection
                    reconstructed = latent @ self.projection.T
                    error = centered - reconstructed
                    self.projection += alpha * (np.outer(centered, latent) - np.outer(error, latent))
                    self.projection -= alpha * self.l2_reg * self.projection
                    norm = np.linalg.norm(self.projection, axis=0, keepdims=True)
                    self.projection /= np.maximum(norm, 1e-8)
        return self.project(embedding)

    def project(self, embedding: NDArray) -> NDArray:
        if self.use_sklearn and self._ipca_fitted:
            return self.ipca.transform(embedding.reshape(1, -1))[0].astype(np.float32)
        return ((embedding - self.mean) @ self.projection).astype(np.float32)

    def get_state(self) -> Dict:
        return {"projection": self.projection.tolist(), "mean": self.mean.tolist(), "n_samples": self.n_samples, "use_sklearn": self.use_sklearn}

    def load_state(self, state: Dict):
        self.projection = np.array(state["projection"], dtype=np.float32)
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.n_samples = state.get("n_samples", 0)
        if self.use_sklearn and state.get("use_sklearn"):
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(n_components=self.latent_dim)
            self.ipca.mean_ = self.mean
            self.ipca.components_ = self.projection.T
            self.ipca.n_samples_seen_ = self.n_samples
            self._ipca_fitted = True


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_length = np.mean(list(self.doc_lengths.values())) if self.doc_lengths else 0.0

    def remove_document(self, doc_id: str):
        if doc_id in self.documents:
            text = self.documents.pop(doc_id)
            for token in set(self._tokenize(text)):
                self.doc_freq[token] = max(0, self.doc_freq.get(token, 1) - 1)
                if self.doc_freq[token] == 0:
                    del self.doc_freq[token]
            self.doc_lengths.pop(doc_id, None)
            if self.doc_lengths:
                self.avg_doc_length = np.mean(list(self.doc_lengths.values()))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        n = len(self.documents)
        scores = {doc_id: 0.0 for doc_id in self.documents}
        for token in self._tokenize(query):
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, text in self.documents.items():
                tf = text.lower().count(token)
                doc_len = self.doc_lengths.get(doc_id, 1)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1)))
        return [(d, s) for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k] if s > 0]


class AdaptiveThreshold:
    def __init__(self, window_size: int = 30, base_threshold: float = 0.25, sensitivity: float = 0.5):
        self.window: deque = deque(maxlen=window_size)
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.current_threshold = base_threshold

    def record_tension(self, tension: float):
        self.window.append(tension)
        if len(self.window) >= 5:
            self.current_threshold = max(0.01, np.mean(self.window) + self.sensitivity * np.std(self.window))

    def get_threshold(self) -> float:
        return self.current_threshold


class TDAMonitor:
    def __init__(self):
        self.history: List[Dict] = []

    def compute_persistence(self, nodes: Dict[str, MemoryNode]) -> Dict:
        if len(nodes) < 3:
            return {"H0": 0, "H1": 0, "avg_persistence": 0.0}
        positions = np.array([n.latent_pos for n in nodes.values()])
        n = len(positions)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        valid = dists[dists < np.inf]
        if len(valid) < 2:
            return {"H0": n, "H1": 0, "avg_persistence": 0.0}
        threshold = np.median(valid)
        connected = [[i] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < threshold:
                    ci = cj = -1
                    for c_idx, c in enumerate(connected):
                        if i in c: ci = c_idx
                        if j in c: cj = c_idx
                    if ci != cj and ci >= 0 and cj >= 0:
                        connected[ci].extend(connected[cj])
                        connected.pop(cj)
        h0 = len(connected)
        h1 = max(0, len(valid) - n + h0)
        result = {"H0": h0, "H1": h1, "avg_persistence": 0.0}
        self.history.append(result)
        return result

    def get_trend(self) -> str:
        if len(self.history) < 2:
            return "stable"
        recent = [h["H1"] for h in self.history[-5:]]
        if len(recent) >= 3 and recent[-1] > recent[0] * 1.5:
            return "growing_contradictions"
        return "stable"


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 200):
        self.m = m
        self.ef_construction = ef_construction
        self.graph: Dict[str, List[str]] = {}
        self.positions: Dict[str, NDArray] = {}

    def insert(self, node_id: str, pos: NDArray):
        self.positions[node_id] = pos
        self.graph[node_id] = []
        if len(self.positions) <= 1:
            return
        candidates = [c for c in list(self.positions.keys()) if c != node_id][:self.ef_construction]
        if candidates:
            cand_pos = np.array([self.positions[c] for c in candidates])
            dists = np.linalg.norm(cand_pos - pos, axis=1)
            nearest = [candidates[i] for i in np.argsort(dists)[:self.m]]
            self.graph[node_id] = nearest
            for nb in nearest:
                if nb in self.graph:
                    self.graph[nb].append(node_id)
                    if len(self.graph[nb]) > self.m * 2:
                        self.graph[nb] = self.graph[nb][-self.m:]

    def remove(self, node_id: str):
        self.graph.pop(node_id, None)
        self.positions.pop(node_id, None)
        for nid in self.graph:
            self.graph[nid] = [n for n in self.graph[nid] if n != node_id]

    def search(self, query_pos: NDArray, top_k: int = 10) -> List[str]:
        if not self.positions:
            return []
        start = list(self.positions.keys())[0]
        candidates = {start}
        visited = set()
        for _ in range(min(self.ef_construction, len(self.positions))):
            best = min((c for c in candidates - visited), key=lambda c: np.linalg.norm(self.positions[c] - query_pos), default=None)
            if best is None:
                break
            visited.add(best)
            candidates.update(self.graph.get(best, []))
        return sorted(candidates, key=lambda nid: np.linalg.norm(self.positions[nid] - query_pos))[:top_k]


class TorchBackend:
    def __init__(self):
        self.torch = None
        self.device = None
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self.torch is not None

    def batch_resonance(self, ql, qp, np_, nph, na, ns, bw, pc):
        if not self.available:
            return self._numpy(ql, qp, np_, nph, na, ns, bw, pc)
        tq = self.torch.from_numpy(ql).to(self.device)
        dists = self.torch.cdist(tq, self.torch.from_numpy(np_).to(self.device))
        spatial = self.torch.exp(-dists / bw)
        pd = qp.unsqueeze(1) - self.torch.from_numpy(nph).to(self.device).unsqueeze(0)
        pa = 0.5 + 0.5 * self.torch.cos(pd)
        r = spatial * ((1 - pc) + pc * pa)
        return (r * self.torch.from_numpy(na).to(self.device).unsqueeze(0) * self.torch.from_numpy(ns).to(self.device).unsqueeze(0)).cpu().numpy()

    @staticmethod
    def _numpy(ql, qp, np_, nph, na, ns, bw, pc):
        dists = cdist(ql, np_)
        spatial = np.exp(-dists / bw)
        pd = qp[:, np.newaxis] - nph[np.newaxis, :]
        pa = 0.5 + 0.5 * np.cos(pd)
        return spatial * ((1 - pc) + pc * pa) * na[np.newaxis, :] * ns[np.newaxis, :]


class LearnableKernel:
    def __init__(self, bandwidth: float = 1.0, phase_coupling: float = 0.3,
                 decay_rate: float = 0.998, gradient_clip: float = 1.0):
        self.bandwidth = bandwidth
        self.phase_coupling = phase_coupling
        self.decay_rate = decay_rate
        self.gradient_clip = gradient_clip
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0
        self._adam_state = {
            "bandwidth": {"m": 0.0, "v": 0.0, "t": 0},
            "phase_coupling": {"m": 0.0, "v": 0.0, "t": 0},
        }

    def resonance_response(self, dist: float, phase_diff: float, amplitude: float, salience: float) -> float:
        spatial = math.exp(-dist / self.bandwidth)
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        return spatial * ((1 - self.phase_coupling) + self.phase_coupling * phase_align) * amplitude * salience

    def compute_gradients(self, dist: float, phase_diff: float, amplitude: float, salience: float, loss_gradient: float = 1.0):
        spatial = math.exp(-dist / self.bandwidth)
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        self._grad_bandwidth += loss_gradient * spatial * (dist / self.bandwidth ** 2) * ((1 - self.phase_coupling) + self.phase_coupling * phase_align) * amplitude * salience
        self._grad_phase_coupling += loss_gradient * spatial * (phase_align - 1.0) * amplitude * salience

    def step(self):
        for param_name, grad in [("bandwidth", self._grad_bandwidth), ("phase_coupling", self._grad_phase_coupling)]:
            if abs(grad) < 1e-12:
                continue
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
            s = self._adam_state[param_name]
            s["t"] += 1
            s["m"] = 0.9 * s["m"] + 0.1 * grad
            s["v"] = 0.999 * s["v"] + 0.001 * grad ** 2
            m_hat = s["m"] / (1 - 0.9 ** s["t"])
            v_hat = s["v"] / (1 - 0.999 ** s["t"])
            lr = 0.001
            update = lr * m_hat / (math.sqrt(v_hat) + 1e-8)
            if param_name == "bandwidth":
                self.bandwidth = max(0.1, self.bandwidth - update)
            elif param_name == "phase_coupling":
                self.phase_coupling = float(np.clip(self.phase_coupling - update, 0.0, 1.0))
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0

    def get_state(self) -> Dict:
        return {"bandwidth": self.bandwidth, "phase_coupling": self.phase_coupling, "decay_rate": self.decay_rate,
                "adam_state": {k: dict(v) for k, v in self._adam_state.items()}}

    def load_state(self, state: Dict):
        self.bandwidth = state["bandwidth"]
        self.phase_coupling = state["phase_coupling"]
        self.decay_rate = state.get("decay_rate", self.decay_rate)
        if "adam_state" in state:
            self._adam_state = state["adam_state"]


class DifferentiableConsolidation:
    def __init__(self, loss_weight: float = 0.1):
        self.loss_weight = loss_weight
        self.consolidation_loss = 0.0

    def compute_synthesis(self, node1: MemoryNode, node2: MemoryNode, gate: float) -> Dict:
        w1, w2 = gate, 1.0 - gate
        new_latent = w1 * node1.latent_pos + w2 * node2.latent_pos
        new_phase = np.arctan2(w1*np.sin(node1.phase)+w2*np.sin(node2.phase),
                               w1*np.cos(node1.phase)+w2*np.cos(node2.phase)) % (2*np.pi)
        new_amp = min(1.0, w1*node1.amplitude + w2*node2.amplitude)
        new_sal = w1*node1.salience + w2*node2.salience
        pos_loss = np.sum((new_latent - node1.latent_pos)**2) + np.sum((new_latent - node2.latent_pos)**2)
        phase_loss = min(abs(new_phase-node1.phase), 2*np.pi-abs(new_phase-node1.phase)) + \
                     min(abs(new_phase-node2.phase), 2*np.pi-abs(new_phase-node2.phase))
        self.consolidation_loss = self.loss_weight * (pos_loss + phase_loss * 0.1)
        return {"latent_pos": new_latent, "phase": new_phase, "amplitude": new_amp,
                "salience": new_sal, "loss": self.consolidation_loss}


# ============================================================================
# ФОРМАТИРОВАНИЕ КОНТЕКСТА
# ============================================================================

SYSTEM_PROMPT_TEMPLATES = {
    ContextFormat.PLAIN: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories from previous conversations. "
        "Use them to provide accurate, context-aware answers. "
        "Higher resonance (R) means more relevant memory.\n\n"
        "Relevant memories:\n{context}"
    ),
    ContextFormat.JSON: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories in JSON format. Each entry has:\n"
        "- resonance: how well it matches the current query (higher = more relevant)\n"
        "- salience: overall importance in the memory field\n"
        "- text: the actual memory content\n"
        "- lineage: history of how this memory was formed through consolidation\n"
        "Use these memories to provide accurate, context-aware answers.\n\n"
        "Relevant memories:\n{context}"
    ),
    ContextFormat.YAML: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories in YAML format with resonance and salience scores. "
        "Higher scores indicate more relevant/important memories. Use them for context-aware answers.\n\n"
        "Relevant memories:\n{context}"
    ),
}


def format_context(results: List[Tuple[str, float, MemoryNode]], fmt: ContextFormat) -> str:
    if fmt == ContextFormat.JSON:
        items = []
        for nid, resp, node in results:
            item = {"resonance": round(resp, 4), "salience": round(node.salience, 4),
                    "text": node.content.get("text", ""), "lineage": node.lineage,
                    "modality": node.modality, "self_sup_score": round(node.self_sup_score, 4)}
            meta = {k: v for k, v in node.content.items() if k != "text"}
            if meta:
                item["metadata"] = meta
            items.append(item)
        return json.dumps(items, ensure_ascii=False, indent=2) if items else "[]"
    elif fmt == ContextFormat.YAML:
        lines = []
        for nid, resp, node in results:
            lines.extend([f"- resonance: {resp:.4f}", f"  salience: {node.salience:.4f}",
                          f"  text: \"{node.content.get('text', '')}\"",
                          f"  lineage: {node.lineage}", f"  modality: {node.modality}"])
        return "\n".join(lines) if lines else "No relevant memory."
    else:
        parts = [f"[R:{r:.2f}|S:{n.salience:.2f}] {n.content.get('text', '')}" for _, r, n in results]
        return "\n".join(parts) if parts else "No relevant memory."


def build_system_prompt(context: str, fmt: ContextFormat, use_structured: bool) -> str:
    if not use_structured or not context or context in ("No relevant memory.", "[]"):
        return "You are a helpful assistant with long-term memory."
    return SYSTEM_PROMPT_TEMPLATES.get(fmt, SYSTEM_PROMPT_TEMPLATES[ContextFormat.PLAIN]).format(context=context)


# ============================================================================
# ЯДРО: RTMDKField v6
# ============================================================================

class RTMDKField:
    def __init__(self, config: RTMDKConfig, projection_matrix: Optional[NDArray] = None):
        self.cfg = config
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_index: List[str] = []

        # Projection
        if config.learn_projection:
            self.projection_learner = IncPCAProjection(
                config.embedding_dim, config.pca_n_components or config.latent_dim,
                config.projection_lr, config.projection_update_freq, config.l2_regularization)
            if projection_matrix is not None:
                self.projection_learner.set_matrix(projection_matrix)
        else:
            self.projection_learner = None
            self._raw_projection = (projection_matrix.astype(np.float32) if projection_matrix is not None
                                    else np.random.randn(config.embedding_dim, config.latent_dim).astype(np.float32) * 0.1)

        self.adaptive_threshold = AdaptiveThreshold(config.adaptive_window, config.tension_threshold) if config.adaptive_threshold else None
        self.bm25_index = BM25Index(config.bm25_k1, config.bm25_b) if config.bm25_fallback else None
        self.tda_monitor = TDAMonitor() if config.tda_monitoring else None
        self.gpu_backend = TorchBackend() if config.backend == Backend.TORCH else None
        if self.gpu_backend and not self.gpu_backend.available:
            self.gpu_backend = None
        self.hnsw_index = HNSWIndex(config.hnsw_m, config.hnsw_ef_construction) if config.use_hnsw else None

        self.learnable_kernel: Optional[LearnableKernel] = None
        self.diff_consolidation: Optional[DifferentiableConsolidation] = None
        if config.differentiable:
            self.learnable_kernel = LearnableKernel(config.bandwidth, config.phase_coupling, config.decay_rate, config.gradient_clip)
            self.diff_consolidation = DifferentiableConsolidation(config.consolidation_loss_weight)

        self.monitor: Optional[Any] = None  # Legacy

        self.meta_kernel: Optional[MetaAdaptiveKernel] = None
        if config.meta_adaptive:
            self.meta_kernel = MetaAdaptiveKernel(config.bandwidth, config.phase_coupling, config.meta_adaptation_lr,
                                                  config.kurtosis_target_min, config.kurtosis_target_max)

        self.healer: Optional[TopologyHealer] = None
        if config.self_healing:
            self.healer = TopologyHealer(config.dead_zone_threshold, config.hyperconvergence_threshold,
                                        config.fragmentation_threshold, config.healing_strength, config.max_healing_nodes_per_step)

        # Фаза 6: Causal
        self.causal_engine: Optional[CausalInferenceEngine] = None
        if config.causal_topological:
            self.causal_engine = CausalInferenceEngine(
                min_samples=config.causal_discovery_min_samples,
                p_threshold=config.causal_p_threshold,
                adjustment_sets_enabled=config.causal_adjustment_sets)

        # Фаза 7: Neural ODE
        self.ode_dynamics: Optional[NeuralODEDynamics] = None
        if config.continuous_dynamics:
            self.ode_dynamics = NeuralODEDynamics(
                config.latent_dim, config.sde_noise_level, config.ode_time_horizon,
                config.ode_n_steps, config.ode_chunk_size, config.ode_solver,
                config.ode_atol, config.ode_rtol)

        # Фаза 8: Agent
        self.agent_planner: Optional[AgentPlanner] = None
        self.hypothesis_verifier: Optional[HypothesisVerifier] = None
        self.tool_router: Optional[ToolRouter] = None
        if config.agent_orchestration:
            self.agent_planner = AgentPlanner(config.max_plan_depth, config.max_tool_calls, config.tool_timeout)
            self.hypothesis_verifier = HypothesisVerifier(config.verification_confidence_threshold)
            self.tool_router = ToolRouter(config.tool_timeout)

        # Фаза 9: Production
        self.shadow_evaluator: Optional[ShadowModeEvaluator] = None
        self.ragas_evaluator: Optional[RAGASPlusEvaluator] = None
        self.rollback_manager: Optional[AutoRollbackManager] = None
        if config.production_mode:
            if config.shadow_mode:
                self.shadow_evaluator = ShadowModeEvaluator(config.shadow_fallback_threshold)
            if config.ragas_enabled:
                self.ragas_evaluator = RAGASPlusEvaluator()
            if config.auto_rollback:
                self.rollback_manager = AutoRollbackManager(config.auto_rollback_threshold)

        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0,
            "projection_updates": 0, "self_sup_checks": 0, "tda_checks": 0,
            "bm25_fallbacks": 0, "adaptive_threshold_value": config.tension_threshold,
            "false_merges": 0, "field_stability": 1.0,
            "causal_edges": 0, "contradictions": 0, "counterfactual_queries": 0,
            "consolidation_validations": 0, "blocked_consolidations": 0,
            "meta_kurtosis": 3.0, "meta_bandwidth": config.bandwidth,
            "meta_phase_coupling": config.phase_coupling,
            "field_health": "stable", "healing_events": 0, "healing_history": [],
            # Фаза 7
            "ode_steps": 0, "response_smoothness": 1.0,
            # Фаза 8
            "plans_created": 0, "hypotheses_verified": 0, "tool_calls": 0, "tool_misuse_rate": 0.0,
            # Фаза 9
            "evaluations": 0, "shadow_comparisons": 0, "rollbacks": 0,
            "ragas_overall": 0.0,
        }
        self._step_counter = 0
        self._rollback_history: List[Dict] = []
        self._stability_buffer: deque = deque(maxlen=config.field_stability_window)
        self._active_node_history: deque = deque(maxlen=50)

    def _project(self, embedding: NDArray) -> NDArray:
        if self.projection_learner:
            return self.projection_learner.project(embedding)
        return ((embedding - 0) @ self._raw_projection).astype(np.float32) if embedding.ndim == 1 \
            else (embedding @ self._raw_projection).astype(np.float32)

    def _get_phase(self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None,
                   modality: str = "text") -> float:
        base = (time.time() * 0.01) % (2 * np.pi)
        if self.cfg.multimodal and modality in self.cfg.modality_phase_shifts:
            base += self.cfg.modality_phase_shifts[modality]
        return base % (2 * np.pi)

    def _resonance_response(self, query_latent: NDArray, query_phase: float, node: MemoryNode) -> float:
        dist = np.linalg.norm(query_latent - node.latent_pos)
        phase_diff = node.phase - query_phase
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        pc = self.meta_kernel.get_phase_coupling() if self.meta_kernel else self.cfg.phase_coupling

        if self.learnable_kernel:
            resp = self.learnable_kernel.resonance_response(dist, phase_diff, node.amplitude, node.salience)
        else:
            if self.cfg.resonance_kernel == "gaussian":
                spatial = math.exp(-dist ** 2 / (2 * bw ** 2))
            elif self.cfg.resonance_kernel == "cosine":
                nq = np.linalg.norm(query_latent)
                nn = np.linalg.norm(node.latent_pos)
                spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (nq * nn + 1e-8) if nq > 1e-8 and nn > 1e-8 else 0.5
            else:
                spatial = math.exp(-dist / bw)
            phase_align = 0.5 + 0.5 * math.cos(phase_diff)
            resp = spatial * ((1 - pc) + pc * phase_align) * node.amplitude * node.salience

        gate = node.soft_gate if self.cfg.soft_gates else 1.0
        if self.causal_engine and node.causal_parents:
            causal_boost = sum(node.causal_strength.get(p, 0) for p in node.causal_parents)
            resp *= (1.0 + 0.1 * causal_boost)

        return resp * gate * node.modal_weight

    def query(self, embedding: NDArray, phase: float = 0.0, top_k: Optional[int] = None) -> List[Tuple[str, float, MemoryNode]]:
        t0 = time.time()
        top_k = top_k or self.cfg.top_k
        query_latent = self._project(embedding)

        if self.cfg.use_hnsw and self.hnsw_index and len(self.hnsw_index.positions) > top_k * 2:
            candidate_ids = self.hnsw_index.search(query_latent, top_k * 3)
            search_nodes = [(nid, self.nodes[nid]) for nid in candidate_ids if nid in self.nodes]
        else:
            search_nodes = [(nid, self.nodes[nid]) for nid in self.node_index]

        results = []
        for nid, node in search_nodes:
            resp = self._resonance_response(query_latent, phase, node)
            if resp >= self.cfg.min_response:
                results.append((nid, resp, node))
                node.last_resonated = time.time()

        results.sort(key=lambda x: x[1], reverse=True)
        self.stats["total_queries"] += 1

        if len(results) == 0 and self.cfg.bm25_fallback and self.bm25_index:
            text = " ".join(self.nodes[nid].content.get("text", "") for nid in self.node_index[:100])
            if text:
                for doc_id, score in self.bm25_index.search(text, top_k):
                    if doc_id in self.nodes:
                        results.append((doc_id, score * 0.1, self.nodes[doc_id]))
                self.stats["bm25_fallbacks"] += 1

        if results:
            self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]
            if self.ode_dynamics:
                self.ode_dynamics.record_response(results[0][1])

        if self.meta_kernel:
            self.meta_kernel.record_response(results[0][1] if results else 0.0)
            if len(results) >= 2:
                positions = np.array([n.latent_pos for _, _, n in results])
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                valid = dists[dists < np.inf]
                density = 1.0 / (1.0 + np.mean(valid)) if len(valid) > 0 else 0.0
                self.meta_kernel.record_semantic_density(float(density))
            if len(results) >= 2:
                responses = np.array([r for _, r, _ in results])
                normalized = responses / (np.sum(responses) + 1e-8)
                entropy = -np.sum(normalized * np.log(normalized + 1e-8))
                self.meta_kernel.record_uncertainty(float(entropy))

        if self.causal_engine and len(results) >= 2:
            self.causal_engine.record_cooccurrence(results[0][0], results[1][0])
            active = [nid for nid, resp, _ in results if resp > self.cfg.min_response * 0.5]
            if active:
                self.causal_engine.record_observation(active)
                self._active_node_history.append(active)

        return results[:top_k]

    def add_node(self, embedding: NDArray, content: Dict, phase: Optional[float] = None,
                 node_id: Optional[str] = None, session_id: Optional[str] = None, modality: str = "text") -> str:
        nid = node_id or f"n_{len(self.nodes)}_{int(time.time() * 1000)}"
        if self.projection_learner:
            latent = self.projection_learner.update(embedding)
            self.stats["projection_updates"] += 1
        else:
            latent = self._project(embedding)
        if phase is None:
            phase = self._get_phase(session_id, embedding, modality)

        node = MemoryNode(id=nid, latent_pos=latent, phase=phase,
                          amplitude=0.7, salience=0.6, content=content,
                          lineage=[], modality=modality)
        self.nodes[nid] = node
        self.node_index.append(nid)
        self.stats["total_adds"] += 1

        if self.cfg.use_hnsw and self.hnsw_index:
            self.hnsw_index.insert(nid, latent)
        if self.cfg.bm25_fallback and self.bm25_index:
            text = content.get("text", "")
            if text:
                self.bm25_index.add_document(nid, text)

        return nid

    def _compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        node = self.nodes[node_id]
        neighbors = [self.nodes[oid] for oid in self.node_index
                     if oid != node_id and np.linalg.norm(node.latent_pos - self.nodes[oid].latent_pos) < neighborhood_radius]
        if len(neighbors) < 2:
            return 0.0
        phases = np.array([n.phase for n in neighbors])
        saliences = np.array([n.salience for n in neighbors])
        return 0.6 * (np.std(np.cos(phases)) + np.std(np.sin(phases))) + 0.4 * np.std(saliences)

    def _soft_gate(self, tension: float) -> float:
        if not self.cfg.soft_gates:
            return 1.0
        eff = self.adaptive_threshold.get_threshold() if self.adaptive_threshold else self.cfg.tension_threshold
        return float(1 / (1 + math.exp(-(tension - eff) / self.cfg.gate_temperature)))

    def get_effective_threshold(self) -> float:
        return self.adaptive_threshold.get_threshold() if self.adaptive_threshold else self.cfg.tension_threshold

    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        mode = mode or self.cfg.consolidation_mode
        updated = []
        eff_threshold = self.get_effective_threshold()

        pre_state = {}
        if self.cfg.enable_rollback or self.cfg.self_sup_verify_after_consolidate:
            for nid in self.node_index:
                n = self.nodes[nid]
                pre_state[nid] = {"latent_pos": n.latent_pos.copy(), "phase": n.phase,
                                  "amplitude": n.amplitude, "salience": n.salience}

        for nid in self.node_index:
            tension = self._compute_tension(nid)
            self.nodes[nid].tension = tension
            self.nodes[nid].soft_gate = self._soft_gate(tension)
            if self.adaptive_threshold:
                self.adaptive_threshold.record_tension(tension)
                self.stats["adaptive_threshold_value"] = self.adaptive_threshold.get_threshold()

        high_tension = [nid for nid in self.node_index if self.nodes[nid].tension > eff_threshold]
        processed = set()

        for nid in high_tension:
            if nid in processed or nid not in self.nodes:
                continue
            node = self.nodes[nid]
            candidates = []
            for oid in self.node_index:
                if oid == nid or oid in processed or oid not in self.nodes:
                    continue
                other = self.nodes[oid]
                dist = np.linalg.norm(node.latent_pos - other.latent_pos)
                pd = min(abs(node.phase - other.phase), 2 * np.pi - abs(node.phase - other.phase))
                if dist < 2.5 and pd > 1.0:
                    candidates.append((oid, dist, pd))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[1])
            pid = candidates[0][0]
            partner = self.nodes[pid]

            if self.cfg.do_calculus_validation and self.causal_engine:
                validation = self.causal_engine.validate_consolidation(nid, pid)
                self.stats["consolidation_validations"] += 1
                if not validation["safe"]:
                    self.stats["blocked_consolidations"] += 1
                    processed.add(nid)
                    processed.add(pid)
                    continue

            gate = self._soft_gate(max(node.tension, partner.tension))

            if self.cfg.enable_rollback:
                node.pre_consolidation_pos = node.latent_pos.copy()

            if self.diff_consolidation and mode == ConsolidationMode.DIALECTICAL:
                synth = self.diff_consolidation.compute_synthesis(node, partner, gate)
                node.latent_pos = synth["latent_pos"]
                node.phase = synth["phase"]
                node.amplitude = synth["amplitude"]
                node.salience = synth["salience"]
            elif mode == ConsolidationMode.DIALECTICAL:
                node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                node.phase = np.arctan2(0.5*(np.sin(node.phase)+np.sin(partner.phase)),
                                        0.5*(np.cos(node.phase)+np.cos(partner.phase))) % (2*np.pi)
                node.amplitude = min(1.0, 0.8*(node.amplitude+partner.amplitude))
                node.salience = 0.7*(node.salience+partner.salience)

            node.tension = 0.0
            node.soft_gate = 1.0
            node.lineage = [f"{node.id}+{pid}"] + node.lineage + partner.lineage
            node.content["synthesis_note"] = f"Consolidated with {pid} at t={time.time():.0f}"

            if self.causal_engine:
                for parent, strength in partner.causal_strength.items():
                    if parent not in node.causal_strength:
                        node.causal_strength[parent] = strength
                    else:
                        node.causal_strength[parent] = max(node.causal_strength[parent], strength)

            if self.cfg.use_hnsw and self.hnsw_index:
                self.hnsw_index.remove(pid)
                self.hnsw_index.insert(nid, node.latent_pos)
            if self.cfg.bm25_fallback and self.bm25_index:
                self.bm25_index.remove_document(pid)
            del self.nodes[pid]
            self.node_index.remove(pid)
            processed.add(pid)
            updated.append(nid)
            self.stats["consolidations"] += 1
            processed.add(nid)

        if updated:
            self._verify_consistency(updated, pre_state)

        self._prune_dead_nodes()
        self.stats["active_nodes"] = len(self.nodes)

        if pre_state and updated:
            scores = []
            for nid in updated:
                if nid in self.nodes and nid in pre_state:
                    o, n = pre_state[nid]["latent_pos"], self.nodes[nid].latent_pos
                    scores.append(max(0, np.dot(o, n) / (np.linalg.norm(o)*np.linalg.norm(n)+1e-8)))
            if scores:
                self._stability_buffer.append(np.mean(scores))
                self.stats["field_stability"] = float(np.mean(self._stability_buffer))

        if self.cfg.enable_rollback and pre_state:
            self._rollback_history.append({"timestamp": time.time(), "pre_state": pre_state, "updated": updated})
            if len(self._rollback_history) > self.cfg.max_rollback_history:
                self._rollback_history.pop(0)

        if self.causal_engine and self._step_counter % max(self.cfg.causal_discovery_freq, 1) == 0:
            self.causal_engine.discover_causal_structure()
            for (cause, effect), edge in self.causal_engine.causal_effects.items():
                if effect in self.nodes:
                    self.nodes[effect].causal_parents.append(cause)
                    self.nodes[effect].causal_strength[cause] = edge.strength
                if cause in self.nodes:
                    self.nodes[cause].causal_effects[effect] = edge.strength
            self.stats["causal_edges"] = len(self.causal_engine.causal_effects)
            if self.cfg.contradiction_detection:
                self.causal_engine.detect_contradictions(self.cfg.contradiction_threshold)
                self.stats["contradictions"] = len(self.causal_engine.contradictions)

        if self.learnable_kernel:
            self.learnable_kernel.step()

        if self.meta_kernel:
            self.meta_kernel.adapt()
            self.stats["meta_kurtosis"] = self.meta_kernel.compute_resonance_kurtosis()
            self.stats["meta_bandwidth"] = self.meta_kernel.get_bandwidth()
            self.stats["meta_phase_coupling"] = self.meta_kernel.get_phase_coupling()

        # Фаза 7: smoothness tracking
        if self.ode_dynamics:
            self.stats["response_smoothness"] = self.ode_dynamics.compute_response_smoothness()

        return updated

    def _verify_consistency(self, updated_nodes: List[str], pre_state: Optional[Dict] = None):
        for nid in updated_nodes:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.lineage:
                results = self.query(np.zeros(self.cfg.embedding_dim, dtype=np.float32), phase=node.phase, top_k=1)
                if results and results[0][0] == nid:
                    node.self_sup_score = max(0.5, results[0][1])
                else:
                    node.self_sup_score *= 0.9

    def _self_supervise(self):
        if not self.cfg.self_supervision:
            return
        self.stats["self_sup_checks"] += 1
        for nid in list(self.node_index):
            if nid not in self.nodes or not self.nodes[nid].lineage:
                continue
            node = self.nodes[nid]
            results = self.query(np.zeros(self.cfg.embedding_dim, dtype=np.float32), phase=node.phase, top_k=1)
            if results and results[0][0] == nid:
                node.self_sup_score = max(0.5, results[0][1])
            else:
                node.self_sup_score *= 0.9

    def _check_tda(self):
        if not self.cfg.tda_monitoring or not self.tda_monitor:
            return
        self.stats["tda_checks"] += 1
        r = self.tda_monitor.compute_persistence(self.nodes)
        self.stats["tda_H0"] = r["H0"]
        self.stats["tda_H1"] = r["H1"]
        if self.tda_monitor.get_trend() == "growing_contradictions":
            self.consolidate()

    def _prune_dead_nodes(self):
        to_remove = [nid for nid in self.node_index
                     if self.nodes[nid].amplitude < self.cfg.min_amplitude
                     or self.nodes[nid].salience < self.cfg.min_amplitude * 0.5]
        for nid in to_remove:
            if self.cfg.use_hnsw and self.hnsw_index:
                self.hnsw_index.remove(nid)
            if self.cfg.bm25_fallback and self.bm25_index:
                self.bm25_index.remove_document(nid)
            del self.nodes[nid]
            self.node_index.remove(nid)

    # --- Фаза 7: Continuous evolution ---
    def evolve_continuous(self, inputs: Optional[List[Dict]] = None, use_sde: bool = False) -> NDArray:
        if not self.ode_dynamics or not self.nodes:
            return np.array([])

        initial_state = np.array([n.latent_pos for n in self.nodes.values()]).flatten()
        input_signal = None
        if inputs:
            input_signal = np.array([self._project(inp["embedding"]) for inp in inputs]).flatten()

        topo_grad = self.ode_dynamics.compute_topology_gradient(self.nodes)

        if use_sde:
            trajectory = self.ode_dynamics.evolve_with_noise(initial_state, input_signal, topo_grad)
        else:
            trajectory = self.ode_dynamics.evolve(initial_state, input_signal, topo_grad)

        self.stats["ode_steps"] += 1

        final_state = trajectory[-1].reshape(len(self.nodes), self.cfg.latent_dim)
        for i, nid in enumerate(self.node_index):
            if i < len(final_state):
                old_pos = self.nodes[nid].latent_pos.copy()
                self.nodes[nid].latent_pos = final_state[i].astype(np.float32)
                self.nodes[nid].velocity = (self.nodes[nid].latent_pos - old_pos).astype(np.float32)

        return trajectory

    # --- Фаза 8: Agent orchestration ---
    def create_plan(self, goal: str, available_tools: List[str], context: Optional[Dict] = None) -> AgentPlan:
        """Создать план агента."""
        if not self.agent_planner:
            return AgentPlan(goal=goal, subtasks=[], tools_needed=[],
                           estimated_steps=0, confidence=0.0, reasoning="Agent orchestration not enabled")
        self.stats["plans_created"] += 1
        ctx = context or {}
        ctx["hypothesis_verification"] = self.cfg.hypothesis_verification
        return self.agent_planner.create_plan(goal, available_tools, ctx)

    def verify_hypothesis(self, hypothesis: str, active_nodes: Optional[List[str]] = None) -> Hypothesis:
        """Верифицировать гипотезу."""
        if not self.hypothesis_verifier or not self.causal_engine:
            return Hypothesis(statement=hypothesis, confidence=0.5, evidence_nodes=[],
                            causal_path=[], verified=False, verification_score=0.5)
        self.stats["hypotheses_verified"] += 1
        nodes = active_nodes or self.node_index
        return self.hypothesis_verifier.verify(hypothesis, self.causal_engine, nodes)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """Выполнить инструмент."""
        if not self.tool_router:
            return ToolCall(tool_name=tool_name, arguments=arguments, error="Tool router not enabled")
        if self.agent_planner and not self.agent_planner.can_call_tool(tool_name):
            return ToolCall(tool_name=tool_name, arguments=arguments, error="Tool call limit reached")
        self.stats["tool_calls"] += 1
        if self.agent_planner:
            self.agent_planner.record_tool_call(tool_name)
        result = self.tool_router.execute(tool_name, arguments)
        if result.success:
            self.stats["tool_misuse_rate"] = self.tool_router.get_misuse_rate()
        return result

    def register_tool(self, name: str, func: Callable):
        """Зарегистрировать инструмент."""
        if self.tool_router:
            self.tool_router.register_tool(name, func)

    # --- Фаза 9: Production ---
    def evaluate_response(self, question: str, answer: str, contexts: List[str],
                          ground_truth: Optional[str] = None) -> EvalResult:
        """RAGAS++ оценка ответа."""
        if not self.ragas_evaluator:
            return EvalResult()
        self.stats["evaluations"] += 1
        causal_edges = None
        if self.causal_engine:
            causal_edges = [(k[0], k[1], v.strength) for k, v in self.causal_engine.causal_effects.items()]
        result = self.ragas_evaluator.evaluate(question, answer, contexts, ground_truth, causal_edges)
        self.stats["ragas_overall"] = result.overall_score

        # Auto-rollback check
        if self.rollback_manager:
            self.rollback_manager.record_score(result.overall_score)
            if self.rollback_manager.record_score(result.overall_score):
                self.stats["rollbacks"] += 1

        return result

    def compare_shadow(self, shadow_score: float, production_score: float) -> Dict[str, Any]:
        """Сравнить shadow и production."""
        if not self.shadow_evaluator:
            return {}
        self.stats["shadow_comparisons"] += 1
        return self.shadow_evaluator.compare(shadow_score, production_score)

    def step(self, inputs: Optional[List[Dict]] = None):
        self._step_counter += 1

        if self.cfg.continuous_dynamics and self.ode_dynamics:
            self.evolve_continuous(inputs, use_sde=self.cfg.sde_noise_level > 0)
            return

        if inputs:
            for inp in inputs:
                emb = inp["embedding"]
                phase = inp.get("phase", 0.0)
                content = inp.get("content", {})
                session_id = inp.get("session_id")
                modality = inp.get("modality", "text")

                results = self.query(emb, phase, top_k=1)
                if results and results[0][1] > 0.3:
                    nid, _, node = results[0]
                    target = self._project(emb)
                    node.latent_pos += self.cfg.attraction_lr * (target - node.latent_pos)
                    pd = (phase - node.phase + np.pi) % (2*np.pi) - np.pi
                    node.phase += self.cfg.phase_sync_lr * pd
                    node.amplitude = min(1.0, node.amplitude + 0.05)
                    node.salience = min(1.0, node.salience + 0.03)
                else:
                    self.add_node(emb, content, phase, session_id=session_id, modality=modality)

        if len(self.nodes) > 10 and np.random.random() < 0.15:
            self.consolidate()

        if self.cfg.self_healing and self._step_counter % self.cfg.healing_check_freq == 0:
            self._self_heal()

        for node in self.nodes.values():
            dk = self.learnable_kernel.decay_rate if self.learnable_kernel else self.cfg.decay_rate
            node.amplitude *= dk
            node.salience *= dk
            node.amplitude = np.clip(node.amplitude, self.cfg.min_amplitude, 1.0)
            node.salience = np.clip(node.salience, self.cfg.min_amplitude * 0.5, 1.0)

        if self.cfg.max_nodes and len(self.nodes) > self.cfg.max_nodes:
            sorted_nodes = sorted(self.node_index, key=lambda nid: self.nodes[nid].salience * self.nodes[nid].amplitude)
            for nid in sorted_nodes[:len(self.nodes) - self.cfg.max_nodes]:
                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(nid)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(nid)
                del self.nodes[nid]
                self.node_index.remove(nid)

        if self.cfg.self_supervision and self._step_counter % 20 == 0:
            self._self_supervise()
        if self.cfg.tda_monitoring and self._step_counter % self.cfg.tda_check_freq == 0:
            self._check_tda()

    def _self_heal(self) -> List[Dict]:
        if not self.healer or len(self.nodes) < 3:
            return []
        health, diagnostics = self.healer.compute_field_health(self.nodes)
        self.stats["field_health"] = health.value
        healed = []
        if health == FieldHealth.STABLE:
            for nid in self.node_index:
                self.nodes[nid].is_healing = False
                self.nodes[nid].healing_origin = None
            return []
        self.stats["field_health"] = FieldHealth.HEALING.value
        if diagnostics.get("dead_zones", 0) > 0:
            healed.extend(self.healer.heal_dead_zones(self.nodes, diagnostics["dead_zone_nodes"]))
        if diagnostics.get("hyperconvergence", False):
            healed.extend(self.healer.heal_hyperconvergence(self.nodes))
        if diagnostics.get("fragmentation", 0) > self.cfg.fragmentation_threshold:
            if len(self.nodes) >= 2:
                positions = np.array([n.latent_pos for n in self.nodes.values()])
                dists = cdist(positions, positions)
                np.fill_diagonal(dists, np.inf)
                isolated = [self.node_index[i] for i in range(len(self.node_index)) if np.all(dists[i] > 2.0)]
                if isolated:
                    healed.extend(self.healer.heal_fragmentation(self.nodes, isolated))
        if healed:
            self.stats["healing_events"] += len(healed)
            self.stats["healing_history"].extend(healed)
            if len(self.stats["healing_history"]) > 1000:
                self.stats["healing_history"] = self.stats["healing_history"][-500:]
        return healed

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        if not self._rollback_history or n_steps > len(self._rollback_history):
            return False
        snapshot = self._rollback_history[-n_steps]
        for nid, state in snapshot["pre_state"].items():
            if nid in self.nodes:
                self.nodes[nid].latent_pos = state["latent_pos"].copy()
                self.nodes[nid].phase = state["phase"]
                self.nodes[nid].amplitude = state["amplitude"]
                self.nodes[nid].salience = state["salience"]
                self.nodes[nid].pre_consolidation_pos = None
        self._rollback_history = self._rollback_history[:-n_steps]
        return True

    def do_intervention(self, node_id: str, new_embedding: NDArray):
        if node_id not in self.nodes:
            return
        new_pos = self._project(new_embedding)
        if self.causal_engine:
            self.causal_engine.do_intervention(node_id, new_pos)
        self.nodes[node_id].latent_pos = new_pos

    def clear_interventions(self):
        if self.causal_engine:
            self.causal_engine.clear_interventions()

    def get_monitor_dashboard(self) -> Dict:
        return {}

    def record_ab_metric(self, metric_name: str, value: float):
        pass

    def get_field_health(self) -> Dict:
        if self.healer:
            health, diagnostics = self.healer.compute_field_health(self.nodes)
            diagnostics["kurtosis"] = self.stats.get("meta_kurtosis", 3.0)
            return diagnostics
        return {"health": "unknown", "kurtosis": 3.0}

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        if not self.causal_engine:
            return CounterfactualResult(query=str(intervention), intervention=intervention,
                predicted_outcomes=[], confidence=0.0, reasoning_path=["Causal engine not enabled"], assumptions=[])
        self.stats["counterfactual_queries"] += 1
        return self.causal_engine.counterfactual_query(intervention, query_nodes, evidence, self.cfg.counterfactual_max_depth)

    def get_causal_summary(self) -> Dict:
        if not self.causal_engine:
            return {"enabled": False}
        return {
            "enabled": True,
            "causal_edges": len(self.causal_engine.causal_effects),
            "contradictions": len([c for c in self.causal_engine.contradictions.values() if not c.resolved]),
            "nodes_with_effects": len(set(k[0] for k in self.causal_engine.causal_effects)),
            "nodes_affected": len(set(k[1] for k in self.causal_engine.causal_effects)),
            "top_effects": sorted([(f"{k[0]}→{k[1]}", v.strength) for k, v in self.causal_engine.causal_effects.items()],
                                 key=lambda x: x[1], reverse=True)[:10],
        }


# ============================================================================
# RTMDKMemory v6
# ============================================================================

class RTMDKMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    config: RTMDKConfig = Field(default_factory=RTMDKConfig)
    embedder: Callable[[str], NDArray[np.float32]]
    field: Optional[RTMDKField] = Field(default=None, exclude=True)
    session_phases: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _init_field(cls, data):
        if isinstance(data, dict) and data.get("field") is None:
            cfg = data.get("config", RTMDKConfig())
            data = dict(data)
            data["field"] = RTMDKField(cfg)
        return data

    def model_post_init(self, __context):
        if self.field is None:
            object.__setattr__(self, "field", RTMDKField(self.config))

    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]

    def _get_phase(self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None) -> float:
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        phase = (time.time() * 0.01) % (2 * np.pi)
        if session_id:
            self.session_phases[session_id] = phase
        return phase

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        embedding = self.embedder(query)
        phase = self._get_phase(session_id, embedding)
        results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k)
        context = format_context(results, self.config.context_format)
        return {"rtmdk_context": context}

    def get_system_prompt(self, context: str) -> str:
        return build_system_prompt(context, self.config.context_format, self.config.use_structured_prompt)

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        text = outputs.get("output", inputs.get("input", ""))
        session_id = inputs.get("session_id", "default")
        if not text.strip():
            return
        embedding = self.embedder(text)
        phase = self._get_phase(session_id, embedding)
        content = {"text": text, "timestamp": time.time(), "session": session_id,
                   **{k: v for k, v in inputs.items() if k not in ["input", "query", "session_id"]}}
        self.field.add_node(embedding, content, phase, session_id=session_id)

        if self.config.enable_async:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._evolve_field_async())
            except RuntimeError:
                self.field.step()
        else:
            self.field.step()

    async def _evolve_field_async(self):
        await asyncio.sleep(0.01)
        self.field.step()

    def clear(self) -> None:
        self.field = RTMDKField(self.config)
        self.session_phases.clear()

    def inspect_node(self, node_id: str) -> Optional[Dict]:
        if node_id not in self.field.nodes:
            return None
        node = self.field.nodes[node_id]
        info = {"id": node.id, "phase": node.phase, "amplitude": node.amplitude,
                "salience": node.salience, "tension": node.tension, "soft_gate": node.soft_gate,
                "self_sup_score": node.self_sup_score, "modal_weight": node.modal_weight,
                "modality": node.modality, "lineage": node.lineage, "content": node.content,
                "created_at": node.created_at, "last_resonated": node.last_resonated,
                "causal_parents": node.causal_parents, "causal_strength": node.causal_strength,
                "causal_effects": node.causal_effects, "is_causal_root": node.is_causal_root,
                "is_healing": node.is_healing, "healing_origin": node.healing_origin,
                "local_density": node.local_density, "goal_tags": node.goal_tags}
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        if node.velocity is not None:
            info["velocity"] = node.velocity.tolist()
        return info

    def rollback(self, n_steps: int = 1) -> bool:
        return self.field.rollback_consolidation(n_steps)

    def get_rollback_history(self) -> List[Dict]:
        return [{"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(s["pre_state"])}
                for s in self.field._rollback_history]

    def do_intervention(self, node_id: str, text: str):
        emb = self.embedder(text)
        self.field.do_intervention(node_id, emb)

    def clear_interventions(self):
        self.field.clear_interventions()

    def get_dashboard(self) -> Dict:
        return self.field.get_monitor_dashboard()

    def record_ab_metric(self, metric_name: str, value: float):
        self.field.record_ab_metric(metric_name, value)

    def get_field_health(self) -> Dict:
        return self.field.get_field_health()

    def trigger_healing(self) -> List[Dict]:
        return self.field._self_heal()

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        return self.field.counterfactual_query(intervention, query_nodes, evidence)

    def get_causal_summary(self) -> Dict:
        return self.field.get_causal_summary()

    def get_contradictions(self) -> List[ContradictionRecord]:
        if self.field.causal_engine:
            return list(self.field.causal_engine.contradictions.values())
        return []

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> bool:
        if self.field.causal_engine and contradiction_id in self.field.causal_engine.contradictions:
            self.field.causal_engine.contradictions[contradiction_id].resolved = True
            self.field.causal_engine.contradictions[contradiction_id].resolution = resolution
            return True
        return False

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        if self.field.causal_engine:
            return self.field.causal_engine.validate_consolidation(node_a, node_b)
        return {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}

    # Фаза 7: ODE
    def evolve_continuous(self, inputs: Optional[List[Dict]] = None, use_sde: bool = False) -> NDArray:
        return self.field.evolve_continuous(inputs, use_sde)

    def get_response_smoothness(self) -> float:
        return self.field.stats.get("response_smoothness", 1.0)

    # Фаза 8: Agent
    def create_plan(self, goal: str, available_tools: List[str], context: Optional[Dict] = None) -> AgentPlan:
        return self.field.create_plan(goal, available_tools, context)

    def verify_hypothesis(self, hypothesis: str, active_nodes: Optional[List[str]] = None) -> Hypothesis:
        return self.field.verify_hypothesis(hypothesis, active_nodes)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        return self.field.execute_tool(tool_name, arguments)

    def register_tool(self, name: str, func: Callable):
        self.field.register_tool(name, func)

    # Фаза 9: Production
    def evaluate_response(self, question: str, answer: str, contexts: List[str],
                          ground_truth: Optional[str] = None) -> EvalResult:
        return self.field.evaluate_response(question, answer, contexts, ground_truth)

    def compare_shadow(self, shadow_score: float, production_score: float) -> Dict[str, Any]:
        return self.field.compare_shadow(shadow_score, production_score)

    def get_ragas_trend(self) -> Dict[str, float]:
        if self.field.ragas_evaluator:
            return self.field.ragas_evaluator.get_trend()
        return {}

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        return {**self.field.stats, "config": asdict(self.config)}

    def export_field(self, path: str):
        cd = asdict(self.config)
        cd["consolidation_mode"] = cd["consolidation_mode"].value
        cd["backend"] = cd["backend"].value
        cd["context_format"] = cd["context_format"].value
        cd["eval_mode"] = cd["eval_mode"].value if isinstance(cd.get("eval_mode"), EvalMode) else cd.get("eval_mode", "production")
        data = {"config": cd, "nodes": [n.to_dict() for n in self.field.nodes.values()], "stats": self.field.stats}
        if self.field.projection_learner:
            data["projection_state"] = self.field.projection_learner.get_state()
        else:
            data["projection"] = self.field._raw_projection.tolist()
        if self.field.learnable_kernel:
            data["learnable_kernel"] = self.field.learnable_kernel.get_state()
        if self.field.tda_monitor:
            data["tda_history"] = self.field.tda_monitor.history
        if self.field.meta_kernel:
            data["meta_kernel"] = self.field.meta_kernel.get_state()
        if self.field.healer:
            data["healer"] = self.field.healer.get_state()
        if self.field.causal_engine:
            data["causal_engine"] = self.field.causal_engine.get_state()
        if self.field.ode_dynamics:
            data["ode_dynamics"] = self.field.ode_dynamics.get_state()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def import_field(cls, path: str, embedder: Callable) -> RTMDKMemory:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cd = data["config"]
        if isinstance(cd.get("consolidation_mode"), str):
            cd["consolidation_mode"] = ConsolidationMode(cd["consolidation_mode"])
        if isinstance(cd.get("backend"), str):
            cd["backend"] = Backend(cd["backend"])
        if isinstance(cd.get("context_format"), str):
            cd["context_format"] = ContextFormat(cd["context_format"])
        if isinstance(cd.get("eval_mode"), str):
            cd["eval_mode"] = EvalMode(cd["eval_mode"])
        # Handle v5→v6 field renames and remove unknown fields
        if "causal_modeling" in cd and "causal_topological" not in cd:
            cd["causal_topological"] = cd.pop("causal_modeling")
        elif "causal_modeling" in cd:
            cd.pop("causal_modeling")
        # Filter out fields not in RTMDKConfig
        valid_fields = set(f.name for f in RTMDKConfig.__dataclass_fields__.values())
        cd = {k: v for k, v in cd.items() if k in valid_fields}
        config = RTMDKConfig(**cd)
        memory = cls(config=config, embedder=embedder)

        if config.learn_projection and "projection_state" in data:
            memory.field.projection_learner.load_state(data["projection_state"])
        elif "projection" in data:
            memory.field._raw_projection = np.array(data["projection"], dtype=np.float32)
        if config.differentiable and "learnable_kernel" in data:
            memory.field.learnable_kernel.load_state(data["learnable_kernel"])
        if config.tda_monitoring and "tda_history" in data:
            memory.field.tda_monitor.history = data["tda_history"]
        if config.meta_adaptive and "meta_kernel" in data:
            memory.field.meta_kernel.load_state(data["meta_kernel"])
        if config.self_healing and "healer" in data:
            memory.field.healer.load_state(data["healer"])
        if config.causal_topological and "causal_engine" in data:
            memory.field.causal_engine.load_state(data["causal_engine"])
        if config.continuous_dynamics and "ode_dynamics" in data:
            ode_state = data["ode_dynamics"]
            memory.field.ode_dynamics.alpha = ode_state.get("alpha", 0.1)
            memory.field.ode_dynamics.beta = ode_state.get("beta", 0.05)
            memory.field.ode_dynamics.gamma = ode_state.get("gamma", 0.02)
            if "W" in ode_state:
                memory.field.ode_dynamics.W = np.array(ode_state["W"], dtype=np.float32)
            memory.field.ode_dynamics.noise_level = ode_state.get("noise_level", 0.01)

        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory
