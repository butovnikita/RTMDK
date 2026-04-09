"""
rtmdk_memory_v3.py
Резонансно-топологическая память — Версия 3.0

Трек 1: Дифференцируемое поле (learnable kernel, gradient flow)
Трек 2: Neural ODE / SDE непрерывная динамика
Трек 3: Каузально-топологическая интеграция
Трек 4: Продакшен мониторинг + drift detection + A/B
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
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Set
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.integrate import odeint
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ v3
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

    # ТРЕК 2: Neural ODE / SDE
    continuous_dynamics: bool = False
    ode_solver: str = "dopri5"
    ode_atol: float = 1e-6
    ode_rtol: float = 1e-5
    sde_noise_level: float = 0.01
    ode_time_horizon: float = 1.0
    ode_n_steps: int = 20

    # ТРЕК 3: Каузальность
    causal_modeling: bool = False
    causal_discovery_freq: int = 100
    causal_threshold: float = 0.3
    intervention_mode: str = "do_calculus"

    # ТРЕК 4: Продакшен мониторинг
    production_mode: bool = False
    drift_detection: bool = False
    drift_window: int = 100
    drift_threshold: float = 0.05
    ab_testing: bool = False
    ab_variant: str = "control"
    auto_rollback_on_anomaly: bool = False
    anomaly_threshold: float = 3.0
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
# ТИПЫ ДАННЫХ v3
# ============================================================================

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
    # Трек 3: каузальные связи
    causal_parents: List[str] = field(default_factory=list)
    causal_strength: Dict[str, float] = field(default_factory=dict)
    # Трек 1: градиенты
    gradient_cache: Optional[NDArray[np.float32]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["latent_pos"] = self.latent_pos.tolist()
        if self.pre_consolidation_pos is not None:
            d["pre_consolidation_pos"] = self.pre_consolidation_pos.tolist()
        if self.gradient_cache is not None:
            d["gradient_cache"] = self.gradient_cache.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryNode:
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        if data.get("pre_consolidation_pos"):
            data["pre_consolidation_pos"] = np.array(data["pre_consolidation_pos"], dtype=np.float32)
        if data.get("gradient_cache"):
            data["gradient_cache"] = np.array(data["gradient_cache"], dtype=np.float32)
        return cls(**data)


# ============================================================================
# ТРЕК 1: ДИФФЕРЕНЦИРУЕМОЕ ПОЛЕ
# ============================================================================

class LearnableKernel:
    """Обучаемые параметры ядра резонанса с gradient flow."""

    def __init__(self, bandwidth: float = 1.0, phase_coupling: float = 0.3,
                 decay_rate: float = 0.998, gradient_clip: float = 1.0):
        self.bandwidth = bandwidth
        self.phase_coupling = phase_coupling
        self.decay_rate = decay_rate
        self.gradient_clip = gradient_clip
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0
        self._grad_decay = 0.0
        self._optim_params = {
            "bandwidth": {"lr": 0.001, "beta1": 0.9, "beta2": 0.999},
            "phase_coupling": {"lr": 0.001, "beta1": 0.9, "beta2": 0.999},
            "decay_rate": {"lr": 0.0001, "beta1": 0.9, "beta2": 0.999},
        }
        self._adam_state = {
            "bandwidth": {"m": 0.0, "v": 0.0, "t": 0},
            "phase_coupling": {"m": 0.0, "v": 0.0, "t": 0},
            "decay_rate": {"m": 0.0, "v": 0.0, "t": 0},
        }

    def resonance_response(self, dist: float, phase_diff: float,
                           amplitude: float, salience: float) -> float:
        """Дифференцируемый резонансный отклик."""
        spatial = math.exp(-dist / self.bandwidth)
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        response = spatial * ((1 - self.phase_coupling) + self.phase_coupling * phase_align)
        return response * amplitude * salience

    def compute_gradients(self, dist: float, phase_diff: float,
                          amplitude: float, salience: float,
                          loss_gradient: float = 1.0):
        """Вычисление градиентов по параметрам ядра."""
        spatial = math.exp(-dist / self.bandwidth)
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        base = spatial * ((1 - self.phase_coupling) + self.phase_coupling * phase_align)

        d_response_d_bw = spatial * (dist / self.bandwidth ** 2) * \
                          ((1 - self.phase_coupling) + self.phase_coupling * phase_align)
        d_response_d_pc = spatial * (phase_align - 1.0)
        d_response_d_decay = 0.0

        self._grad_bandwidth += loss_gradient * d_response_d_bw * amplitude * salience
        self._grad_phase_coupling += loss_gradient * d_response_d_pc * amplitude * salience
        self._grad_decay += loss_gradient * d_response_d_decay * amplitude * salience

    def step(self):
        """Adam optimizer step для параметров ядра."""
        for param_name, grad in [("bandwidth", self._grad_bandwidth),
                                   ("phase_coupling", self._grad_phase_coupling),
                                   ("decay_rate", self._grad_decay)]:
            if abs(grad) < 1e-12:
                continue
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
            s = self._adam_state[param_name]
            s["t"] += 1
            s["m"] = 0.9 * s["m"] + 0.1 * grad
            s["v"] = 0.999 * s["v"] + 0.001 * grad ** 2
            m_hat = s["m"] / (1 - 0.9 ** s["t"])
            v_hat = s["v"] / (1 - 0.999 ** s["t"])
            lr = self._optim_params[param_name]["lr"]
            update = lr * m_hat / (math.sqrt(v_hat) + 1e-8)

            if param_name == "bandwidth":
                self.bandwidth = max(0.1, self.bandwidth - update)
            elif param_name == "phase_coupling":
                self.phase_coupling = np.clip(self.phase_coupling - update, 0.0, 1.0)
            elif param_name == "decay_rate":
                self.decay_rate = np.clip(self.decay_rate - update, 0.9, 1.0)

        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0
        self._grad_decay = 0.0

    def get_state(self) -> Dict:
        return {
            "bandwidth": self.bandwidth,
            "phase_coupling": self.phase_coupling,
            "decay_rate": self.decay_rate,
            "adam_state": {k: dict(v) for k, v in self._adam_state.items()},
        }

    def load_state(self, state: Dict):
        self.bandwidth = state["bandwidth"]
        self.phase_coupling = state["phase_coupling"]
        self.decay_rate = state["decay_rate"]
        if "adam_state" in state:
            self._adam_state = state["adam_state"]


class DifferentiableConsolidation:
    """Дифференцируемая консолидация: мягкое взвешенное слияние."""

    def __init__(self, loss_weight: float = 0.1):
        self.loss_weight = loss_weight
        self.consolidation_loss = 0.0

    def compute_synthesis(self, node1: MemoryNode, node2: MemoryNode,
                          gate: float) -> Dict[str, Any]:
        """Мягкий синтез с сохранением градиентного потока."""
        w1 = gate
        w2 = 1.0 - gate

        new_latent = w1 * node1.latent_pos + w2 * node2.latent_pos
        new_phase = np.arctan2(
            w1 * np.sin(node1.phase) + w2 * np.sin(node2.phase),
            w1 * np.cos(node1.phase) + w2 * np.cos(node2.phase)
        ) % (2 * np.pi)
        new_amp = min(1.0, w1 * node1.amplitude + w2 * node2.amplitude)
        new_sal = w1 * node1.salience + w2 * node2.salience

        # Loss: information preservation
        pos_loss = np.sum((new_latent - node1.latent_pos) ** 2) + \
                   np.sum((new_latent - node2.latent_pos) ** 2)
        phase_loss = min(
            abs(new_phase - node1.phase),
            2 * np.pi - abs(new_phase - node1.phase)
        ) + min(
            abs(new_phase - node2.phase),
            2 * np.pi - abs(new_phase - node2.phase)
        )
        self.consolidation_loss = self.loss_weight * (pos_loss + phase_loss * 0.1)

        return {
            "latent_pos": new_latent,
            "phase": new_phase,
            "amplitude": new_amp,
            "salience": new_sal,
            "loss": self.consolidation_loss,
        }


# ============================================================================
# ТРЕК 2: NEURAL ODE / SDE
# ============================================================================

class NeuralODEField:
    """Непрерывная динамика поля: dX/dt = F(X, u(t)) + ξ(t)"""

    def __init__(self, latent_dim: int, noise_level: float = 0.01,
                 time_horizon: float = 1.0, n_steps: int = 20):
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self._weights = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01
        self._bias = np.zeros(latent_dim, dtype=np.float32)

    def _dynamics(self, state: NDArray, t: float, input_signal: Optional[NDArray] = None) -> NDArray:
        """F(X, u(t)): нейронная динамика поля."""
        n_nodes = len(state) // self.latent_dim
        state_2d = state.reshape(n_nodes, self.latent_dim)

        # Self-attention-like dynamics
        transformed = state_2d @ self._weights + self._bias
        damping = -0.1 * state_2d  # естественное затухание

        if input_signal is not None:
            input_signal_2d = input_signal.reshape(n_nodes, self.latent_dim)
            attraction = 0.05 * (input_signal_2d - state_2d)
        else:
            attraction = 0.0

        dstate = damping + 0.02 * transformed + attraction
        return dstate.flatten()

    def evolve(self, initial_state: NDArray, input_signal: Optional[NDArray] = None,
               t_span: Optional[NDArray] = None) -> NDArray:
        """Эволюция поля через ODE solver."""
        if t_span is None:
            t_span = np.linspace(0, self.time_horizon, self.n_steps)

        def ode_func(state, t):
            return self._dynamics(state, t, input_signal)

        solution = odeint(ode_func, initial_state.flatten(), t_span,
                          atol=1e-6, rtol=1e-5)
        return solution

    def evolve_with_noise(self, initial_state: NDArray,
                          input_signal: Optional[NDArray] = None,
                          dt: float = 0.05) -> NDArray:
        """SDE: dX = F(X,u)dt + σ dW (Euler-Maruyama)."""
        n_steps = int(self.time_horizon / dt)
        state = initial_state.flatten().copy()
        trajectory = [state.copy()]

        for _ in range(n_steps):
            deterministic = self._dynamics(state, 0, input_signal) * dt
            noise = self.noise_level * np.random.randn(len(state)) * np.sqrt(dt)
            state = state + deterministic + noise
            trajectory.append(state.copy())

        return np.array(trajectory)

    def get_state(self) -> Dict:
        return {
            "weights": self._weights.tolist(),
            "bias": self._bias.tolist(),
            "noise_level": self.noise_level,
        }

    def load_state(self, state: Dict):
        self._weights = np.array(state["weights"], dtype=np.float32)
        self._bias = np.array(state["bias"], dtype=np.float32)
        self.noise_level = state.get("noise_level", self.noise_level)


# ============================================================================
# ТРЕК 3: КАУЗАЛЬНО-ТОПОЛОГИЧЕСКАЯ ИНТЕГРАЦИЯ
# ============================================================================

class CausalGraph:
    """Каузальный граф поверх резонансного поля."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.interventions: Dict[str, NDArray] = {}
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._node_counts: Dict[str, int] = defaultdict(int)
        self._total_queries = 0

    def record_cooccurrence(self, node_a: str, node_b: str):
        """Записать совместную активацию узлов."""
        self._cooccurrence[(node_a, node_b)] += 1
        self._cooccurrence[(node_b, node_a)] += 1
        self._node_counts[node_a] += 1
        self._node_counts[node_b] += 1
        self._total_queries += 1

    def discover_causal_links(self) -> Dict[str, Dict[str, float]]:
        """Обнаружить каузальные связи через условную независимость."""
        new_edges: Dict[str, Dict[str, float]] = defaultdict(dict)

        for (a, b), count in self._cooccurrence.items():
            if a >= b:
                continue
            p_a = self._node_counts.get(a, 0) / max(self._total_queries, 1)
            p_b = self._node_counts.get(b, 0) / max(self._total_queries, 1)
            p_ab = count / max(self._total_queries, 1)

            if p_a > 0 and p_b > 0:
                pmi = math.log(p_ab / (p_a * p_b) + 1e-10)
                causal_strength = np.clip(pmi / 5.0, 0, 1)

                if causal_strength > self.threshold:
                    new_edges[a][b] = causal_strength
                    new_edges[b][a] = causal_strength

        self.edges = new_edges
        return new_edges

    def do_intervention(self, node_id: str, new_pos: NDArray):
        """do(X = x): каузальное вмешательство."""
        self.interventions[node_id] = new_pos.copy()

    def clear_interventions(self):
        self.interventions.clear()

    def get_causal_parents(self, node_id: str) -> List[str]:
        return list(self.edges.get(node_id, {}).keys())

    def get_causal_effect(self, cause: str, effect: str) -> float:
        """P(effect | do(cause))."""
        return self.edges.get(cause, {}).get(effect, 0.0)

    def get_state(self) -> Dict:
        return {
            "edges": {k: dict(v) for k, v in self.edges.items()},
            "threshold": self.threshold,
        }

    def load_state(self, state: Dict):
        self.edges = defaultdict(dict, {k: dict(v) for k, v in state.get("edges", {}).items()})
        self.threshold = state.get("threshold", self.threshold)


# ============================================================================
# ТРЕК 4: ПРОДАКШЕН МОНИТОРИНГ
# ============================================================================

class ProductionMonitor:
    """Мониторинг, drift detection, A/B тестирование, auto-rollback."""

    def __init__(self, drift_window: int = 100, drift_threshold: float = 0.05,
                 anomaly_threshold: float = 3.0, metrics_retention: int = 10000):
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        self.metrics_retention = metrics_retention

        self._embedding_history: deque = deque(maxlen=drift_window)
        self._response_history: deque = deque(maxlen=metrics_retention)
        self._latency_history: deque = deque(maxlen=metrics_retention)
        self._consolidation_history: deque = deque(maxlen=metrics_retention)
        self._gate_history: deque = deque(maxlen=metrics_retention)

        self._ab_results: Dict[str, Dict] = {}
        self._anomaly_log: List[Dict] = []
        self._is_drifting = False
        self._last_drift_score = 0.0

    def record_embedding(self, embedding: NDArray):
        self._embedding_history.append(embedding.copy())

    def record_response(self, response: float, latency_ms: float,
                        n_consolidations: int = 0, avg_gate: float = 1.0):
        self._response_history.append(response)
        self._latency_history.append(latency_ms)
        self._consolidation_history.append(n_consolidations)
        self._gate_history.append(avg_gate)

    def detect_drift(self) -> Dict[str, Any]:
        """KS-test на дрейф распределения эмбеддингов."""
        if len(self._embedding_history) < self.drift_window // 2:
            return {"drifting": False, "score": 0.0}

        embeddings = np.array(list(self._embedding_history))
        n = len(embeddings)
        half = n // 2
        early = embeddings[:half]
        late = embeddings[half:]

        max_score = 0.0
        dim = min(early.shape[1], 20)
        for d in range(dim):
            stat, _ = scipy_stats.ks_2samp(early[:, d], late[:, d])
            max_score = max(max_score, stat)

        self._last_drift_score = max_score
        self._is_drifting = max_score > self.drift_threshold
        return {"drifting": self._is_drifting, "score": max_score}

    def detect_anomaly(self, current_metric: float) -> bool:
        """Z-score anomaly detection."""
        if len(self._response_history) < 30:
            return False
        values = np.array(list(self._response_history))
        mean, std = np.mean(values), np.std(values)
        if std < 1e-8:
            return False
        z_score = abs(current_metric - mean) / std
        is_anomaly = z_score > self.anomaly_threshold
        if is_anomaly:
            self._anomaly_log.append({
                "timestamp": time.time(),
                "value": current_metric,
                "z_score": z_score,
                "mean": mean,
                "std": std,
            })
        return is_anomaly

    def record_ab_result(self, variant: str, metric_name: str, value: float):
        if variant not in self._ab_results:
            self._ab_results[variant] = {}
        if metric_name not in self._ab_results[variant]:
            self._ab_results[variant][metric_name] = []
        self._ab_results[variant][metric_name].append(value)

    def get_ab_comparison(self, metric_name: str) -> Dict[str, Any]:
        if len(self._ab_results) < 2:
            return {}
        result = {}
        for variant, metrics in self._ab_results.items():
            values = metrics.get(metric_name, [])
            if values:
                result[variant] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "n": len(values),
                }
        return result

    def get_dashboard(self) -> Dict[str, Any]:
        dashboard = {
            "drift": {
                "is_drifting": self._is_drifting,
                "drift_score": self._last_drift_score,
            },
            "response": {
                "mean": float(np.mean(self._response_history)) if self._response_history else 0,
                "std": float(np.std(self._response_history)) if self._response_history else 0,
            },
            "latency_ms": {
                "mean": float(np.mean(self._latency_history)) if self._latency_history else 0,
                "p95": float(np.percentile(self._latency_history, 95)) if self._latency_history else 0,
            },
            "gate_distribution": {
                "mean": float(np.mean(self._gate_history)) if self._gate_history else 1.0,
                "std": float(np.std(self._gate_history)) if self._gate_history else 0,
            },
            "anomalies": len(self._anomaly_log),
            "ab_variants": list(self._ab_results.keys()),
        }
        return dashboard

    def get_state(self) -> Dict:
        return {
            "ab_results": self._ab_results,
            "anomaly_log": self._anomaly_log[-100:],
            "is_drifting": self._is_drifting,
        }

    def load_state(self, state: Dict):
        self._ab_results = state.get("ab_results", {})
        self._anomaly_log = state.get("anomaly_log", [])
        self._is_drifting = state.get("is_drifting", False)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КОМПОНЕНТЫ (из v2)
# ============================================================================

class IncPCAProjection:
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.001,
                 update_freq: int = 50, l2_reg: float = 0.0001):
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
        return {"projection": self.projection.tolist(), "mean": self.mean.tolist(),
                "n_samples": self.n_samples, "use_sklearn": self.use_sklearn}

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
        query_tokens = self._tokenize(query)
        scores = {doc_id: 0.0 for doc_id in self.documents}
        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, text in self.documents.items():
                tf = text.lower().count(token)
                doc_len = self.doc_lengths.get(doc_id, 1)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
                scores[doc_id] += idf * num / den
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
            best = min((c for c in candidates - visited),
                       key=lambda c: np.linalg.norm(self.positions[c] - query_pos), default=None)
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
        return (r * self.torch.from_numpy(na).to(self.device).unsqueeze(0) *
                self.torch.from_numpy(ns).to(self.device).unsqueeze(0)).cpu().numpy()

    @staticmethod
    def _numpy(ql, qp, np_, nph, na, ns, bw, pc):
        dists = cdist(ql, np_)
        spatial = np.exp(-dists / bw)
        pd = qp[:, np.newaxis] - nph[np.newaxis, :]
        pa = 0.5 + 0.5 * np.cos(pd)
        r = spatial * ((1 - pc) + pc * pa)
        return r * na[np.newaxis, :] * ns[np.newaxis, :]


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
# ЯДРО: RTMDKField v3
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

        # Adaptive threshold
        self.adaptive_threshold = AdaptiveThreshold(config.adaptive_window, config.tension_threshold) if config.adaptive_threshold else None

        # BM25
        self.bm25_index = BM25Index(config.bm25_k1, config.bm25_b) if config.bm25_fallback else None

        # TDA
        self.tda_monitor = TDAMonitor() if config.tda_monitoring else None

        # GPU
        self.gpu_backend = TorchBackend() if config.backend == Backend.TORCH else None
        if self.gpu_backend and not self.gpu_backend.available:
            self.gpu_backend = None

        # HNSW
        self.hnsw_index = HNSWIndex(config.hnsw_m, config.hnsw_ef_construction) if config.use_hnsw else None

        # ТРЕК 1: Дифференцируемое поле
        self.learnable_kernel: Optional[LearnableKernel] = None
        self.diff_consolidation: Optional[DifferentiableConsolidation] = None
        if config.differentiable:
            self.learnable_kernel = LearnableKernel(
                config.bandwidth, config.phase_coupling, config.decay_rate, config.gradient_clip)
            self.diff_consolidation = DifferentiableConsolidation(config.consolidation_loss_weight)

        # ТРЕК 2: Neural ODE
        self.neural_ode: Optional[NeuralODEField] = None
        if config.continuous_dynamics:
            self.neural_ode = NeuralODEField(
                config.latent_dim, config.sde_noise_level,
                config.ode_time_horizon, config.ode_n_steps)

        # ТРЕК 3: Каузальность
        self.causal_graph: Optional[CausalGraph] = None
        if config.causal_modeling:
            self.causal_graph = CausalGraph(config.causal_threshold)

        # ТРЕК 4: Продакшен мониторинг
        self.monitor: Optional[ProductionMonitor] = None
        if config.production_mode:
            self.monitor = ProductionMonitor(
                config.drift_window, config.drift_threshold,
                config.anomaly_threshold, config.metrics_retention)

        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0,
            "projection_updates": 0, "self_sup_checks": 0, "tda_checks": 0,
            "bm25_fallbacks": 0, "adaptive_threshold_value": config.tension_threshold,
            "false_merges": 0, "field_stability": 1.0,
            "differentiable_loss": 0.0, "ode_steps": 0, "causal_links": 0,
            "drift_detected": False, "anomalies": 0,
        }
        self._step_counter = 0
        self._rollback_history: List[Dict] = []
        self._stability_buffer: deque = deque(maxlen=config.field_stability_window)

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

    def _resonance_response(self, query_latent: NDArray, query_phase: float,
                            node: MemoryNode) -> float:
        dist = np.linalg.norm(query_latent - node.latent_pos)
        phase_diff = node.phase - query_phase

        if self.learnable_kernel:
            resp = self.learnable_kernel.resonance_response(dist, phase_diff, node.amplitude, node.salience)
        else:
            if self.cfg.resonance_kernel == "gaussian":
                spatial = math.exp(-dist ** 2 / (2 * self.cfg.bandwidth ** 2))
            elif self.cfg.resonance_kernel == "cosine":
                nq = np.linalg.norm(query_latent)
                nn = np.linalg.norm(node.latent_pos)
                spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (nq * nn + 1e-8) if nq > 1e-8 and nn > 1e-8 else 0.5
            else:
                spatial = math.exp(-dist / self.cfg.bandwidth)
            phase_align = 0.5 + 0.5 * math.cos(phase_diff)
            resp = spatial * ((1 - self.cfg.phase_coupling) + self.cfg.phase_coupling * phase_align)
            resp *= node.amplitude * node.salience

        gate = node.soft_gate if self.cfg.soft_gates else 1.0

        # Трек 3: каузальная коррекция
        if self.causal_graph and node.causal_parents:
            causal_boost = sum(node.causal_strength.get(p, 0) for p in node.causal_parents)
            resp *= (1.0 + 0.1 * causal_boost)

        return resp * gate * node.modal_weight

    def query(self, embedding: NDArray, phase: float = 0.0,
              top_k: Optional[int] = None) -> List[Tuple[str, float, MemoryNode]]:
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

        # Трек 3: record cooccurrence для causal discovery
        if self.causal_graph and len(results) >= 2:
            self.causal_graph.record_cooccurrence(results[0][0], results[1][0])

        # Трек 4: monitor
        if self.monitor:
            latency_ms = (time.time() - t0) * 1000
            self.monitor.record_embedding(embedding)
            avg_gate = np.mean([n.soft_gate for _, _, n in results]) if results else 1.0
            self.monitor.record_response(results[0][1] if results else 0, latency_ms, 0, avg_gate)

        return results[:top_k]

    def add_node(self, embedding: NDArray, content: Dict,
                 phase: Optional[float] = None, node_id: Optional[str] = None,
                 session_id: Optional[str] = None, modality: str = "text") -> str:
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
            gate = self._soft_gate(max(node.tension, partner.tension))

            if self.cfg.enable_rollback:
                node.pre_consolidation_pos = node.latent_pos.copy()

            if self.diff_consolidation and mode == ConsolidationMode.DIALECTICAL:
                synth = self.diff_consolidation.compute_synthesis(node, partner, gate)
                node.latent_pos = synth["latent_pos"]
                node.phase = synth["phase"]
                node.amplitude = synth["amplitude"]
                node.salience = synth["salience"]
                self.stats["differentiable_loss"] = synth["loss"]
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

            # Трек 3: causal link transfer
            if self.causal_graph:
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

        # Трек 3: causal discovery
        if self.causal_graph and self._step_counter % self.cfg.causal_discovery_freq == 0:
            self.causal_graph.discover_causal_links()
            self.stats["causal_links"] = sum(len(v) for v in self.causal_graph.edges.values())

        # Трек 1: gradient step
        if self.learnable_kernel:
            self.learnable_kernel.step()

        # Трек 4: drift check
        if self.monitor:
            drift = self.monitor.detect_drift()
            self.stats["drift_detected"] = drift["drifting"]
            if drift["drifting"] and self.cfg.auto_rollback_on_anomaly:
                self.monitor.record_anomaly if hasattr(self.monitor, 'record_anomaly') else None
                self.stats["anomalies"] += 1

        return updated

    def _verify_consistency(self, updated_nodes: List[str], pre_state: Optional[Dict] = None):
        for nid in updated_nodes:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.lineage:
                results = self.query(np.zeros(self.cfg.embedding_dim, dtype=np.float32),
                                     phase=node.phase, top_k=1)
                if results and results[0][0] == nid:
                    node.self_sup_score = max(0.5, results[0][1])
                else:
                    node.self_sup_score *= 0.9
            if pre_state and nid in pre_state:
                old_pos = pre_state[nid]["latent_pos"]
                old_resp = self._resonance_response(old_pos, pre_state[nid]["phase"], node)
                new_resp = self._resonance_response(old_pos, pre_state[nid]["phase"], node)
                if old_resp > 0.01:
                    drop = (old_resp - new_resp) / old_resp
                    if drop > self.cfg.false_merge_threshold:
                        self.stats["false_merges"] = self.stats.get("false_merges", 0) + 1

    def _self_supervise(self):
        if not self.cfg.self_supervision:
            return
        self.stats["self_sup_checks"] += 1
        for nid in list(self.node_index):
            if nid not in self.nodes or not self.nodes[nid].lineage:
                continue
            node = self.nodes[nid]
            results = self.query(np.zeros(self.cfg.embedding_dim, dtype=np.float32),
                                 phase=node.phase, top_k=1)
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

    # --- Трек 2: Непрерывная эволюция ---
    def evolve_continuous(self, inputs: Optional[List[Dict]] = None,
                          use_sde: bool = False) -> NDArray:
        """Эволюция поля через Neural ODE / SDE."""
        if not self.neural_ode or not self.nodes:
            return np.array([])

        initial_state = np.array([n.latent_pos for n in self.nodes.values()]).flatten()
        input_signal = None
        if inputs:
            input_signal = np.array([self._project(inp["embedding"]) for inp in inputs]).flatten()

        if use_sde:
            trajectory = self.neural_ode.evolve_with_noise(initial_state, input_signal)
        else:
            trajectory = self.neural_ode.evolve(initial_state, input_signal)

        self.stats["ode_steps"] += 1

        # Обновить позиции узлов из финального состояния
        final_state = trajectory[-1].reshape(len(self.nodes), self.cfg.latent_dim)
        for i, nid in enumerate(self.node_index):
            if i < len(final_state):
                self.nodes[nid].latent_pos = final_state[i].astype(np.float32)

        return trajectory

    def step(self, inputs: Optional[List[Dict]] = None):
        self._step_counter += 1

        if self.cfg.continuous_dynamics and self.neural_ode:
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
        """Трек 3: каузальное вмешательство do(X = x)."""
        if node_id not in self.nodes:
            return
        new_pos = self._project(new_embedding)
        if self.causal_graph:
            self.causal_graph.do_intervention(node_id, new_pos)
        self.nodes[node_id].latent_pos = new_pos

    def clear_interventions(self):
        """Очистить все каузальные вмешательства."""
        if self.causal_graph:
            self.causal_graph.clear_interventions()

    def get_monitor_dashboard(self) -> Dict:
        """Трек 4: получить дашборд мониторинга."""
        if self.monitor:
            return self.monitor.get_dashboard()
        return {}

    def record_ab_metric(self, metric_name: str, value: float):
        """Трек 4: записать метрику для A/B теста."""
        if self.monitor:
            self.monitor.record_ab_result(self.cfg.ab_variant, metric_name, value)


# ============================================================================
# RTMDKMemory v3
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

    def _get_phase(self, session_id: Optional[str] = None,
                   embedding: Optional[NDArray] = None) -> float:
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
                "causal_parents": node.causal_parents, "causal_strength": node.causal_strength}
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        return info

    def rollback(self, n_steps: int = 1) -> bool:
        return self.field.rollback_consolidation(n_steps)

    def get_rollback_history(self) -> List[Dict]:
        return [{"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(s["pre_state"])}
                for s in self.field._rollback_history]

    def do_intervention(self, node_id: str, text: str):
        """Каузальное вмешательство: do(node = text)."""
        emb = self.embedder(text)
        self.field.do_intervention(node_id, emb)

    def clear_interventions(self):
        self.field.clear_interventions()

    def get_dashboard(self) -> Dict:
        return self.field.get_monitor_dashboard()

    def record_ab_metric(self, metric_name: str, value: float):
        self.field.record_ab_metric(metric_name, value)

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
        data = {"config": cd, "nodes": [n.to_dict() for n in self.field.nodes.values()],
                "stats": self.field.stats}
        if self.field.projection_learner:
            data["projection_state"] = self.field.projection_learner.get_state()
        else:
            data["projection"] = self.field._raw_projection.tolist()
        if self.field.learnable_kernel:
            data["learnable_kernel"] = self.field.learnable_kernel.get_state()
        if self.field.neural_ode:
            data["neural_ode"] = self.field.neural_ode.get_state()
        if self.field.causal_graph:
            data["causal_graph"] = self.field.causal_graph.get_state()
        if self.field.tda_monitor:
            data["tda_history"] = self.field.tda_monitor.history
        if self.field.monitor:
            data["monitor"] = self.field.monitor.get_state()
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
        config = RTMDKConfig(**cd)
        memory = cls(config=config, embedder=embedder)

        if config.learn_projection and "projection_state" in data:
            memory.field.projection_learner.load_state(data["projection_state"])
        elif "projection" in data:
            memory.field._raw_projection = np.array(data["projection"], dtype=np.float32)

        if config.differentiable and "learnable_kernel" in data:
            memory.field.learnable_kernel.load_state(data["learnable_kernel"])
        if config.continuous_dynamics and "neural_ode" in data:
            memory.field.neural_ode.load_state(data["neural_ode"])
        if config.causal_modeling and "causal_graph" in data:
            memory.field.causal_graph.load_state(data["causal_graph"])
        if config.tda_monitoring and "tda_history" in data:
            memory.field.tda_monitor.history = data["tda_history"]
        if config.production_mode and "monitor" in data:
            memory.field.monitor.load_state(data["monitor"])

        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory
