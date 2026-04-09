"""
rtmdk_memory_v2.py
Резонансно-топологическая память с диалектической консолидацией — Версия 2.0

ФАЗА 1: Structured context, Adaptive threshold, IncPCA, BM25 fallback
ФАЗА 2: Soft gates, Self-supervision, GPU resonance
ФАЗА 3: Multimodal, HNSW sharding, TDA monitoring
"""

from __future__ import annotations
import asyncio
import json
import math
import re
import time
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ
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
# ТИПЫ ДАННЫХ
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

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["latent_pos"] = self.latent_pos.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> MemoryNode:
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        return cls(**data)


# ============================================================================
# ФАЗА 1.3: IncPCA проекция
# ============================================================================

class IncPCAProjection:
    """Инкрементальный PCA через sklearn или ручную реализацию."""

    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.001, update_freq: int = 50,
                 l2_reg: float = 0.0001):
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
                components = self.ipca.components_.T  # (input_dim, latent_dim)
                self.projection = components.astype(np.float32)
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
                    # L2 regularization — защита от переобучения
                    self.projection -= alpha * self.l2_reg * self.projection
                    norm = np.linalg.norm(self.projection, axis=0, keepdims=True)
                    self.projection /= np.maximum(norm, 1e-8)

        return self.project(embedding)

    def project(self, embedding: NDArray) -> NDArray:
        if self.use_sklearn and self._ipca_fitted:
            return self.ipca.transform(embedding.reshape(1, -1))[0].astype(np.float32)
        centered = embedding - self.mean
        return (centered @ self.projection).astype(np.float32)

    def get_matrix(self) -> NDArray:
        return self.projection.copy()

    def set_matrix(self, matrix: NDArray):
        self.projection = matrix.astype(np.float32)

    def get_state(self) -> Dict:
        return {
            "projection": self.projection.tolist(),
            "mean": self.mean.tolist(),
            "n_samples": self.n_samples,
            "use_sklearn": self.use_sklearn,
        }

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


# ============================================================================
# ФАЗА 1.4: BM25 fallback
# ============================================================================

class BM25Index:
    """Упрощённый BM25 для текстового fallback-поиска."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self._tokenize_cache: Dict[str, List[str]] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_length = np.mean(list(self.doc_lengths.values())) if self.doc_lengths else 0.0

    def remove_document(self, doc_id: str):
        if doc_id in self.documents:
            text = self.documents.pop(doc_id)
            tokens = set(self._tokenize(text))
            for token in tokens:
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
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.documents}

        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, text in self.documents.items():
                tf = text.lower().count(token)
                doc_len = self.doc_lengths.get(doc_id, 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]


# ============================================================================
# ФАЗА 1.2: Adaptive threshold
# ============================================================================

class AdaptiveThreshold:
    """Скользящее окно + std для автоматической настройки порога консолидации."""

    def __init__(self, window_size: int = 30, base_threshold: float = 0.25, sensitivity: float = 0.5):
        self.window: deque = deque(maxlen=window_size)
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.current_threshold = base_threshold

    def record_tension(self, tension: float):
        self.window.append(tension)
        if len(self.window) >= 5:
            mean_t = np.mean(self.window)
            std_t = np.std(self.window)
            self.current_threshold = max(0.01, mean_t + self.sensitivity * std_t)

    def get_threshold(self) -> float:
        return self.current_threshold

    def is_high_tension(self, tension: float) -> bool:
        return tension > self.get_threshold()


# ============================================================================
# ФАЗА 3.3: TDA Monitoring
# ============================================================================

class TDAMonitor:
    """Упрощённые персистентные гомологии для мониторинга топологии поля."""

    def __init__(self):
        self.history: List[Dict] = []

    def compute_persistence(self, nodes: Dict[str, MemoryNode]) -> Dict:
        if len(nodes) < 3:
            return {"H0": 0, "H1": 0, "avg_persistence": 0.0, "n_betti_0": 0, "n_betti_1": 0}

        positions = np.array([n.latent_pos for n in nodes.values()])
        n = len(positions)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        valid_dists = dists[dists < np.inf]
        if len(valid_dists) < 2:
            return {"H0": n, "H1": 0, "avg_persistence": 0.0, "n_betti_0": n, "n_betti_1": 0}

        threshold = np.median(valid_dists)
        connected = [[i] for i in range(n)]
        birth_death = []

        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < threshold:
                    ci = cj = -1
                    for c_idx, c in enumerate(connected):
                        if i in c: ci = c_idx
                        if j in c: cj = c_idx
                    if ci != cj and ci >= 0 and cj >= 0:
                        birth_death.append((0.0, d))
                        connected[ci].extend(connected[cj])
                        connected.pop(cj)

        h0 = len(connected)
        h1 = max(0, len(valid_dists) - n + h0)
        avg_persist = float(np.mean([d - b for b, d in birth_death])) if birth_death else 0.0

        result = {"H0": h0, "H1": h1, "avg_persistence": avg_persist, "n_betti_0": h0, "n_betti_1": h1}
        self.history.append(result)
        return result

    def get_trend(self) -> str:
        if len(self.history) < 2:
            return "stable"
        recent_h1 = [h["H1"] for h in self.history[-5:]]
        if len(recent_h1) >= 3 and recent_h1[-1] > recent_h1[0] * 1.5:
            return "growing_contradictions"
        return "stable"


# ============================================================================
# ФАЗА 3.2: HNSW Index
# ============================================================================

class HNSWIndex:
    """Упрощённый HNSW для быстрого поиска ближайших соседей."""

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
        candidates = [c for c in list(self.positions.keys()) if c != node_id]
        candidates = candidates[:min(self.ef_construction, len(candidates))]
        if candidates:
            cand_pos = np.array([self.positions[c] for c in candidates])
            dists = np.linalg.norm(cand_pos - pos, axis=1)
            nearest = [candidates[i] for i in np.argsort(dists)[:self.m]]
            self.graph[node_id] = nearest
            for neighbor in nearest:
                if neighbor in self.graph:
                    self.graph[neighbor].append(node_id)
                    if len(self.graph[neighbor]) > self.m * 2:
                        self.graph[neighbor] = self.graph[neighbor][-self.m:]

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
            best = None
            best_dist = float("inf")
            for c in candidates - visited:
                d = np.linalg.norm(self.positions[c] - query_pos)
                if d < best_dist:
                    best_dist = d
                    best = c
            if best is None:
                break
            visited.add(best)
            for neighbor in self.graph.get(best, []):
                candidates.add(neighbor)
        ranked = sorted(candidates, key=lambda nid: np.linalg.norm(self.positions[nid] - query_pos))
        return ranked[:top_k]


# ============================================================================
# ФАЗА 2.3: GPU Backend
# ============================================================================

class TorchBackend:
    """GPU-ускоренный бэкенд для резонансных вычислений."""

    def __init__(self):
        self.torch = None
        self.device = None
        self._try_init()

    def _try_init(self):
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self.torch is not None

    def batch_resonance(self, query_latents: NDArray, query_phases: NDArray,
                        node_positions: NDArray, node_phases: NDArray,
                        node_amplitudes: NDArray, node_saliences: NDArray,
                        bandwidth: float, phase_coupling: float) -> NDArray:
        if not self.available:
            return self._numpy_batch(query_latents, query_phases, node_positions,
                                     node_phases, node_amplitudes, node_saliences,
                                     bandwidth, phase_coupling)
        tq = self.torch.from_numpy(query_latents).to(self.device)
        qp = self.torch.from_numpy(query_phases).to(self.device)
        np_ = self.torch.from_numpy(node_positions).to(self.device)
        nph = self.torch.from_numpy(node_phases).to(self.device)
        na = self.torch.from_numpy(node_amplitudes).to(self.device)
        ns = self.torch.from_numpy(node_saliences).to(self.device)

        dists = self.torch.cdist(tq, np_)
        spatial = self.torch.exp(-dists / bandwidth)
        phase_diff = qp.unsqueeze(1) - nph.unsqueeze(0)
        phase_align = 0.5 + 0.5 * self.torch.cos(phase_diff)
        response = spatial * ((1 - phase_coupling) + phase_coupling * phase_align)
        response = response * na.unsqueeze(0) * ns.unsqueeze(0)
        return response.cpu().numpy()

    @staticmethod
    def _numpy_batch(query_latents, query_phases, node_positions, node_phases,
                     node_amplitudes, node_saliences, bandwidth, phase_coupling):
        dists = cdist(query_latents, node_positions)
        spatial = np.exp(-dists / bandwidth)
        phase_diff = query_phases[:, np.newaxis] - node_phases[np.newaxis, :]
        phase_align = 0.5 + 0.5 * np.cos(phase_diff)
        response = spatial * ((1 - phase_coupling) + phase_coupling * phase_align)
        return response * node_amplitudes[np.newaxis, :] * node_saliences[np.newaxis, :]


# ============================================================================
# ФАЗА 1.1: Structured context formatters
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


def format_context_plain(results: List[Tuple[str, float, MemoryNode]]) -> str:
    parts = []
    for nid, resp, node in results:
        text = node.content.get("text", "")
        parts.append(f"[R:{resp:.2f}|S:{node.salience:.2f}] {text}")
    return "\n".join(parts) if parts else "No relevant memory."


def format_context_json(results: List[Tuple[str, float, MemoryNode]]) -> str:
    items = []
    for nid, resp, node in results:
        item = {
            "resonance": round(resp, 4),
            "salience": round(node.salience, 4),
            "text": node.content.get("text", ""),
            "lineage": node.lineage,
            "modality": node.modality,
            "self_sup_score": round(node.self_sup_score, 4),
        }
        meta = {k: v for k, v in node.content.items() if k != "text"}
        if meta:
            item["metadata"] = meta
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2) if items else "[]"


def format_context_yaml(results: List[Tuple[str, float, MemoryNode]]) -> str:
    lines = []
    for nid, resp, node in results:
        lines.append(f"- resonance: {resp:.4f}")
        lines.append(f"  salience: {node.salience:.4f}")
        lines.append(f"  text: \"{node.content.get('text', '')}\"")
        lines.append(f"  lineage: {node.lineage}")
        lines.append(f"  modality: {node.modality}")
    return "\n".join(lines) if lines else "No relevant memory."


def format_context(results: List[Tuple[str, float, MemoryNode]], fmt: ContextFormat) -> str:
    if fmt == ContextFormat.JSON:
        return format_context_json(results)
    elif fmt == ContextFormat.YAML:
        return format_context_yaml(results)
    return format_context_plain(results)


def build_system_prompt(context: str, fmt: ContextFormat, use_structured: bool) -> str:
    if not use_structured or not context or context in ("No relevant memory.", "[]"):
        return "You are a helpful assistant with long-term memory."
    template = SYSTEM_PROMPT_TEMPLATES.get(fmt, SYSTEM_PROMPT_TEMPLATES[ContextFormat.PLAIN])
    return template.format(context=context)


# ============================================================================
# ЯДРО: RTMDKField v2
# ============================================================================

class RTMDKField:
    """Дискретизированная аппроксимация непрерывного семантического многообразия v2."""

    def __init__(self, config: RTMDKConfig, projection_matrix: Optional[NDArray] = None):
        self.cfg = config
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_index: List[str] = []

        # ФАЗА 1.3: IncPCA проекция
        if config.learn_projection:
            self.projection_learner = IncPCAProjection(
                config.embedding_dim, config.pca_n_components or config.latent_dim,
                config.projection_lr, config.projection_update_freq,
                config.l2_regularization
            )
            if projection_matrix is not None:
                self.projection_learner.set_matrix(projection_matrix)
        else:
            self.projection_learner = None
            if projection_matrix is None:
                self._raw_projection = np.random.randn(config.embedding_dim, config.latent_dim).astype(np.float32) * 0.1
            else:
                self._raw_projection = projection_matrix.astype(np.float32)

        # ФАЗА 1.2: Adaptive threshold
        self.adaptive_threshold: Optional[AdaptiveThreshold] = None
        if config.adaptive_threshold:
            self.adaptive_threshold = AdaptiveThreshold(
                config.adaptive_window, config.tension_threshold
            )

        # ФАЗА 1.4: BM25 fallback
        self.bm25_index: Optional[BM25Index] = None
        if config.bm25_fallback:
            self.bm25_index = BM25Index(config.bm25_k1, config.bm25_b)

        # ФАЗА 3.3: TDA
        self.tda_monitor: Optional[TDAMonitor] = None
        if config.tda_monitoring:
            self.tda_monitor = TDAMonitor()

        # ФАЗА 2.3: GPU
        self.gpu_backend: Optional[TorchBackend] = None
        if config.backend == Backend.TORCH:
            self.gpu_backend = TorchBackend()
            if self.gpu_backend and not self.gpu_backend.available:
                logger.warning("Torch backend requested but not available, falling back to numpy")
                self.gpu_backend = None

        # ФАЗА 3.2: HNSW
        self.hnsw_index: Optional[HNSWIndex] = None
        if config.use_hnsw:
            self.hnsw_index = HNSWIndex(config.hnsw_m, config.hnsw_ef_construction)

        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0,
            "projection_updates": 0, "self_sup_checks": 0, "tda_checks": 0,
            "bm25_fallbacks": 0, "adaptive_threshold_value": config.tension_threshold,
            "false_merges": 0, "field_stability": 1.0,
        }
        self._step_counter = 0
        self._rollback_history: List[Dict] = []
        self._stability_buffer: deque = deque(maxlen=config.field_stability_window)

    # --- Projection ---
    def _project(self, embedding: NDArray) -> NDArray:
        if self.projection_learner:
            return self.projection_learner.project(embedding)
        if embedding.ndim == 1:
            return (embedding @ self._raw_projection).astype(np.float32)
        return (embedding @ self._raw_projection).astype(np.float32)

    def _update_projection(self, embedding: NDArray):
        if self.projection_learner:
            self.projection_learner.update(embedding)
            self.stats["projection_updates"] += 1

    # --- Phase routing ---
    def _get_phase(self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None,
                   modality: str = "text") -> float:
        base_phase = (time.time() * 0.01) % (2 * np.pi)
        if self.cfg.multimodal and modality in self.cfg.modality_phase_shifts:
            base_phase += self.cfg.modality_phase_shifts[modality]
        return base_phase % (2 * np.pi)

    # --- Resonance ---
    def _resonance_response(self, query_latent: NDArray, query_phase: float,
                            node: MemoryNode) -> float:
        if self.cfg.resonance_kernel == "gaussian":
            dist_sq = np.sum((query_latent - node.latent_pos) ** 2)
            spatial = math.exp(-dist_sq / (2 * self.cfg.bandwidth ** 2))
        elif self.cfg.resonance_kernel == "cosine":
            nq = np.linalg.norm(query_latent)
            nn = np.linalg.norm(node.latent_pos)
            spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (nq * nn + 1e-8) if nq > 1e-8 and nn > 1e-8 else 0.5
        else:
            dist = np.linalg.norm(query_latent - node.latent_pos)
            spatial = math.exp(-dist / self.cfg.bandwidth)

        phase_align = 0.5 + 0.5 * math.cos(node.phase - query_phase)
        response = spatial * ((1 - self.cfg.phase_coupling) + self.cfg.phase_coupling * phase_align)
        gate = node.soft_gate if self.cfg.soft_gates else 1.0
        return response * node.amplitude * node.salience * gate * node.modal_weight

    def _batch_resonance(self, query_latents: NDArray, query_phases: NDArray) -> NDArray:
        """GPU/CPU batch resonance computation with chunk_size for large N."""
        if not self.nodes:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        node_positions = np.array([n.latent_pos for n in self.nodes.values()])
        node_phases = np.array([n.phase for n in self.nodes.values()])
        node_amplitudes = np.array([n.amplitude for n in self.nodes.values()])
        node_saliences = np.array([n.salience for n in self.nodes.values()])
        n_nodes = len(node_positions)

        if self.gpu_backend and self.gpu_backend.available:
            chunk_size = self.cfg.gpu_batch_size
            if n_nodes > chunk_size:
                results = []
                for i in range(0, n_nodes, chunk_size):
                    chunk_pos = node_positions[i:i+chunk_size]
                    chunk_ph = node_phases[i:i+chunk_size]
                    chunk_amp = node_amplitudes[i:i+chunk_size]
                    chunk_sal = node_saliences[i:i+chunk_size]
                    r = self.gpu_backend.batch_resonance(
                        query_latents, query_phases, chunk_pos, chunk_ph,
                        chunk_amp, chunk_sal, self.cfg.bandwidth, self.cfg.phase_coupling
                    )
                    results.append(r)
                return np.concatenate(results, axis=1)
            return self.gpu_backend.batch_resonance(
                query_latents, query_phases, node_positions, node_phases,
                node_amplitudes, node_saliences, self.cfg.bandwidth, self.cfg.phase_coupling
            )

        dists = cdist(query_latents, node_positions)
        if self.cfg.resonance_kernel == "gaussian":
            spatial = np.exp(-np.sum((query_latents[:, np.newaxis, :] - node_positions[np.newaxis, :, :]) ** 2, axis=2)
                             / (2 * self.cfg.bandwidth ** 2))
        else:
            spatial = np.exp(-dists / self.cfg.bandwidth)

        phase_diff = query_phases[:, np.newaxis] - node_phases[np.newaxis, :]
        phase_align = 0.5 + 0.5 * np.cos(phase_diff)
        response = spatial * ((1 - self.cfg.phase_coupling) + self.cfg.phase_coupling * phase_align)
        return response * node_amplitudes[np.newaxis, :] * node_saliences[np.newaxis, :]

    # --- Query ---
    def query(self, embedding: NDArray, phase: float = 0.0,
              top_k: Optional[int] = None) -> List[Tuple[str, float, MemoryNode]]:
        top_k = top_k or self.cfg.top_k
        query_latent = self._project(embedding)

        # HNSW candidate selection
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

        # BM25 fallback
        if len(results) == 0 and self.cfg.bm25_fallback and self.bm25_index:
            text = ""
            if self.node_index:
                text = " ".join(self.nodes[nid].content.get("text", "") for nid in self.node_index[:100])
            if text:
                bm25_results = self.bm25_index.search(text, top_k)
                for doc_id, score in bm25_results:
                    if doc_id in self.nodes:
                        node = self.nodes[doc_id]
                        resp = score * 0.1
                        results.append((doc_id, resp, node))
                self.stats["bm25_fallbacks"] += 1

        if results:
            self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]

        return results[:top_k]

    # --- Add node ---
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

        node = MemoryNode(
            id=nid, latent_pos=latent, phase=phase,
            amplitude=0.7, salience=0.6, content=content,
            lineage=[], modality=modality,
        )
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

    # --- Tension ---
    def _compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        node = self.nodes[node_id]
        neighbors = []
        for other_id in self.node_index:
            if other_id == node_id:
                continue
            other = self.nodes[other_id]
            dist = np.linalg.norm(node.latent_pos - other.latent_pos)
            if dist < neighborhood_radius:
                neighbors.append(other)
        if len(neighbors) < 2:
            return 0.0
        phases = np.array([n.phase for n in neighbors])
        saliences = np.array([n.salience for n in neighbors])
        phase_var = np.std(np.cos(phases)) + np.std(np.sin(phases))
        salience_var = np.std(saliences)
        return 0.6 * phase_var + 0.4 * salience_var

    # --- ФАЗА 2.1: Soft gates ---
    def _soft_gate(self, tension: float) -> float:
        if not self.cfg.soft_gates:
            return 1.0
        x = (tension - self.get_effective_threshold()) / self.cfg.gate_temperature
        return float(1 / (1 + math.exp(-x)))

    def get_effective_threshold(self) -> float:
        if self.adaptive_threshold:
            return self.adaptive_threshold.get_threshold()
        return self.cfg.tension_threshold

    # --- Consolidation ---
    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        mode = mode or self.cfg.consolidation_mode
        updated = []
        eff_threshold = self.get_effective_threshold()

        # Save pre-consolidation state for rollback and stability tracking
        pre_state = {}
        if self.cfg.enable_rollback or self.cfg.self_sup_verify_after_consolidate:
            for nid in self.node_index:
                node = self.nodes[nid]
                pre_state[nid] = {
                    "latent_pos": node.latent_pos.copy(),
                    "phase": node.phase,
                    "amplitude": node.amplitude,
                    "salience": node.salience,
                }

        for nid in self.node_index:
            tension = self._compute_tension(nid)
            self.nodes[nid].tension = tension
            self.nodes[nid].soft_gate = self._soft_gate(tension)
            if self.adaptive_threshold:
                self.adaptive_threshold.record_tension(tension)
                self.stats["adaptive_threshold_value"] = self.adaptive_threshold.get_threshold()

        high_tension = [nid for nid in self.node_index
                        if self.nodes[nid].tension > eff_threshold]

        processed = set()
        for nid in high_tension:
            if nid in processed or nid not in self.nodes:
                continue
            node = self.nodes[nid]
            candidates = []
            for other_id in self.node_index:
                if other_id == nid or other_id in processed or other_id not in self.nodes:
                    continue
                other = self.nodes[other_id]
                dist = np.linalg.norm(node.latent_pos - other.latent_pos)
                phase_diff = min(abs(node.phase - other.phase), 2 * np.pi - abs(node.phase - other.phase))
                if dist < 2.5 and phase_diff > 1.0:
                    candidates.append((other_id, dist, phase_diff))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[1])
            partner_id = candidates[0][0]
            partner = self.nodes[partner_id]
            gate = self._soft_gate(max(node.tension, partner.tension))

            # Save pre-consolidation positions for false_merge detection
            if self.cfg.enable_rollback:
                node.pre_consolidation_pos = node.latent_pos.copy()
                partner.pre_consolidation_pos = partner.latent_pos.copy()

            if mode == ConsolidationMode.DIALECTICAL:
                new_latent = 0.5 * (node.latent_pos + partner.latent_pos)
                new_phase = np.arctan2(
                    0.5 * (np.sin(node.phase) + np.sin(partner.phase)),
                    0.5 * (np.cos(node.phase) + np.cos(partner.phase))
                ) % (2 * np.pi)

                if self.cfg.soft_gates:
                    new_amp = gate * min(1.0, 0.8 * (node.amplitude + partner.amplitude)) + (1 - gate) * node.amplitude
                    new_sal = gate * 0.7 * (node.salience + partner.salience) + (1 - gate) * node.salience
                else:
                    new_amp = min(1.0, 0.8 * (node.amplitude + partner.amplitude))
                    new_sal = 0.7 * (node.salience + partner.salience)

                new_lineage = [f"{node.id}+{partner.id}"] + node.lineage + partner.lineage
                node.latent_pos = new_latent
                node.phase = new_phase
                node.amplitude = new_amp
                node.salience = new_sal
                node.tension = 0.0
                node.soft_gate = 1.0
                node.lineage = new_lineage
                node.content["synthesis_note"] = f"Consolidated with {partner_id} at t={time.time():.0f}"

                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(partner_id)
                    self.hnsw_index.insert(nid, new_latent)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(partner_id)

                del self.nodes[partner_id]
                self.node_index.remove(partner_id)
                processed.add(partner_id)
                updated.append(nid)

            elif mode == ConsolidationMode.MERGE:
                node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                node.phase = (node.phase + partner.phase) / 2
                node.amplitude = min(1.0, 0.9 * (node.amplitude + partner.amplitude))
                node.salience = 0.8 * (node.salience + partner.salience)
                node.tension = 0.0
                node.soft_gate = 1.0
                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(partner_id)
                    self.hnsw_index.insert(nid, node.latent_pos)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(partner_id)
                del self.nodes[partner_id]
                self.node_index.remove(partner_id)
                processed.add(partner_id)
                updated.append(nid)

            elif mode == ConsolidationMode.PRUNE:
                if node.salience * node.amplitude >= partner.salience * partner.amplitude:
                    if self.cfg.use_hnsw and self.hnsw_index:
                        self.hnsw_index.remove(partner_id)
                    if self.cfg.bm25_fallback and self.bm25_index:
                        self.bm25_index.remove_document(partner_id)
                    del self.nodes[partner_id]
                    self.node_index.remove(partner_id)
                    processed.add(partner_id)
                else:
                    if self.cfg.use_hnsw and self.hnsw_index:
                        self.hnsw_index.remove(nid)
                    if self.cfg.bm25_fallback and self.bm25_index:
                        self.bm25_index.remove_document(nid)
                    del self.nodes[nid]
                    self.node_index.remove(nid)
                    processed.add(nid)
                updated.append(nid if nid in self.nodes else partner_id)

            self.stats["consolidations"] += 1
            processed.add(nid)

        # ФАЗА 2.2: Self-supervision after consolidation + false_merge detection
        if updated:
            self._verify_consistency(updated, pre_state)

        self._prune_dead_nodes()
        self.stats["active_nodes"] = len(self.nodes)

        # Compute field_stability
        if pre_state and updated:
            stability_scores = []
            for nid in updated:
                if nid in self.nodes and nid in pre_state:
                    old_pos = pre_state[nid]["latent_pos"]
                    new_pos = self.nodes[nid].latent_pos
                    cos_sim = np.dot(old_pos, new_pos) / (
                        np.linalg.norm(old_pos) * np.linalg.norm(new_pos) + 1e-8
                    )
                    stability_scores.append(max(0, cos_sim))
            if stability_scores:
                self._stability_buffer.append(np.mean(stability_scores))
                self.stats["field_stability"] = float(np.mean(self._stability_buffer))

        # Save rollback history
        if self.cfg.enable_rollback and pre_state:
            self._rollback_history.append({
                "timestamp": time.time(),
                "pre_state": pre_state,
                "updated": updated,
            })
            if len(self._rollback_history) > self.cfg.max_rollback_history:
                self._rollback_history.pop(0)

        return updated

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        """Откат последних консолидаций."""
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

    # --- ФАЗА 2.2: Self-supervision ---
    def _verify_consistency(self, updated_nodes: List[str],
                            pre_state: Optional[Dict] = None):
        """После консолидации проверяем, что новые узлы сохраняют резонанс."""
        for nid in updated_nodes:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.lineage:
                results = self.query(
                    np.zeros(self.cfg.embedding_dim, dtype=np.float32),
                    phase=node.phase, top_k=1
                )
                if results and results[0][0] == nid:
                    node.self_sup_score = max(0.5, results[0][1])
                else:
                    node.self_sup_score *= 0.9

            # False merge detection: resonance drop > 40%
            if pre_state and nid in pre_state:
                old_pos = pre_state[nid]["latent_pos"]
                old_query_phase = pre_state[nid]["phase"]
                old_resp = self._resonance_response(old_query_phase, old_query_phase, node)
                new_resp = self._resonance_response(old_pos, old_query_phase, node)
                if old_resp > 0.01:
                    drop = (old_resp - new_resp) / old_resp
                    if drop > self.cfg.false_merge_threshold:
                        self.stats["false_merges"] = self.stats.get("false_merges", 0) + 1

    def _self_supervise(self):
        if not self.cfg.self_supervision:
            return
        self.stats["self_sup_checks"] += 1
        for nid in list(self.node_index):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.lineage and len(node.lineage) > 0:
                results = self.query(
                    np.zeros(self.cfg.embedding_dim, dtype=np.float32),
                    phase=node.phase, top_k=1
                )
                if results and results[0][0] == nid:
                    node.self_sup_score = max(0.5, results[0][1])
                else:
                    node.self_sup_score *= 0.9

    # --- TDA ---
    def _check_tda(self):
        if not self.cfg.tda_monitoring or not self.tda_monitor:
            return
        self.stats["tda_checks"] += 1
        tda_result = self.tda_monitor.compute_persistence(self.nodes)
        self.stats["tda_H0"] = tda_result["H0"]
        self.stats["tda_H1"] = tda_result["H1"]
        self.stats["tda_avg_persistence"] = tda_result["avg_persistence"]
        if self.tda_monitor.get_trend() == "growing_contradictions":
            logger.info("TDA: Growing contradictions detected, triggering consolidation")
            self.consolidate()

    # --- Prune ---
    def _prune_dead_nodes(self):
        to_remove = [
            nid for nid in self.node_index
            if self.nodes[nid].amplitude < self.cfg.min_amplitude
            or self.nodes[nid].salience < self.cfg.min_amplitude * 0.5
        ]
        for nid in to_remove:
            if self.cfg.use_hnsw and self.hnsw_index:
                self.hnsw_index.remove(nid)
            if self.cfg.bm25_fallback and self.bm25_index:
                self.bm25_index.remove_document(nid)
            del self.nodes[nid]
            self.node_index.remove(nid)

    # --- Step ---
    def step(self, inputs: Optional[List[Dict]] = None):
        self._step_counter += 1
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
                    target_latent = self._project(emb)
                    node.latent_pos += self.cfg.attraction_lr * (target_latent - node.latent_pos)
                    phase_diff = (phase - node.phase + np.pi) % (2 * np.pi) - np.pi
                    node.phase += self.cfg.phase_sync_lr * phase_diff
                    node.amplitude = min(1.0, node.amplitude + 0.05)
                    node.salience = min(1.0, node.salience + 0.03)
                else:
                    self.add_node(emb, content, phase, session_id=session_id, modality=modality)

        if len(self.nodes) > 10 and np.random.random() < 0.15:
            self.consolidate()

        for node in self.nodes.values():
            node.amplitude *= self.cfg.decay_rate
            node.salience *= self.cfg.decay_rate
            node.amplitude = np.clip(node.amplitude, self.cfg.min_amplitude, 1.0)
            node.salience = np.clip(node.salience, self.cfg.min_amplitude * 0.5, 1.0)

        if self.cfg.max_nodes and len(self.nodes) > self.cfg.max_nodes:
            sorted_nodes = sorted(
                self.node_index,
                key=lambda nid: self.nodes[nid].salience * self.nodes[nid].amplitude
            )
            to_remove = sorted_nodes[:len(self.nodes) - self.cfg.max_nodes]
            for nid in to_remove:
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


# ============================================================================
# RTMDKMemory v2
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
        """ФАЗА 1.1: Системный промпт с инструкцией по использованию памяти."""
        return build_system_prompt(context, self.config.context_format, self.config.use_structured_prompt)

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        text = outputs.get("output", inputs.get("input", ""))
        session_id = inputs.get("session_id", "default")
        if not text.strip():
            return

        embedding = self.embedder(text)
        phase = self._get_phase(session_id, embedding)
        content = {
            "text": text, "timestamp": time.time(), "session": session_id,
            **{k: v for k, v in inputs.items() if k not in ["input", "query", "session_id"]}
        }
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
        """Инспектировать узел: вернуть полную информацию включая историю."""
        if node_id not in self.field.nodes:
            return None
        node = self.field.nodes[node_id]
        info = {
            "id": node.id,
            "phase": node.phase,
            "amplitude": node.amplitude,
            "salience": node.salience,
            "tension": node.tension,
            "soft_gate": node.soft_gate,
            "self_sup_score": node.self_sup_score,
            "modal_weight": node.modal_weight,
            "modality": node.modality,
            "lineage": node.lineage,
            "content": node.content,
            "created_at": node.created_at,
            "last_resonated": node.last_resonated,
        }
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        return info

    def rollback(self, n_steps: int = 1) -> bool:
        """Откат последних консолидаций."""
        return self.field.rollback_consolidation(n_steps)

    def get_rollback_history(self) -> List[Dict]:
        """История консолидаций для ручного отката."""
        return [
            {"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(s["pre_state"])}
            for s in self.field._rollback_history
        ]

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        return {**self.field.stats, "config": asdict(self.config)}

    def export_field(self, path: str):
        config_dict = asdict(self.config)
        config_dict["consolidation_mode"] = config_dict["consolidation_mode"].value
        config_dict["backend"] = config_dict["backend"].value
        config_dict["context_format"] = config_dict["context_format"].value
        data = {
            "config": config_dict,
            "nodes": [node.to_dict() for node in self.field.nodes.values()],
            "stats": self.field.stats,
        }
        if self.field.projection_learner:
            data["projection_state"] = self.field.projection_learner.get_state()
        else:
            data["projection"] = self.field._raw_projection.tolist()
        if self.field.tda_monitor:
            data["tda_history"] = self.field.tda_monitor.history
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def import_field(cls, path: str, embedder: Callable) -> RTMDKMemory:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_data = data["config"]
        if isinstance(config_data.get("consolidation_mode"), str):
            config_data["consolidation_mode"] = ConsolidationMode(config_data["consolidation_mode"])
        if isinstance(config_data.get("backend"), str):
            config_data["backend"] = Backend(config_data["backend"])
        if isinstance(config_data.get("context_format"), str):
            config_data["context_format"] = ContextFormat(config_data["context_format"])
        config = RTMDKConfig(**config_data)
        memory = cls(config=config, embedder=embedder)

        if config.learn_projection and "projection_state" in data:
            memory.field.projection_learner.load_state(data["projection_state"])
        elif "projection" in data:
            memory.field._raw_projection = np.array(data["projection"], dtype=np.float32)

        if config.tda_monitoring:
            memory.field.tda_monitor = TDAMonitor()
            if "tda_history" in data:
                memory.field.tda_monitor.history = data["tda_history"]

        for node_data in data["nodes"]:
            node = MemoryNode.from_dict(node_data)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory
