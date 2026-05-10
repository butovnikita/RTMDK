"""
rtmdk/memory/core.py
Resonance-Topological Memory - Version 8.1+

Phase 11 Features:
Track 1: Multi-level memory stratification (Episodic / Semantic / Procedural)
Track 2: Hyperbolic geometry (Poincare ball model)
Track 3: Predictive coding / Active inference
Track 4: Counterfactual imagination & scenario planning
Track 5: Differential privacy & secure federation

All v7 components preserved:
MetaAdaptiveKernel, TopologyHealer, CausalInferenceEngine,
IncPCAProjection, BM25Index, HNSWIndex, TorchBackend, LearnableKernel,
DifferentiableConsolidation, AgentPlanner, HypothesisVerifier, ToolRouter,
ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager, MetaController,
KuramotoSync, FederatedRTMDK, FederatedNode, detect_modality, cross_modal_resonance
"""

from __future__ import annotations
from rtmdk.support.meta_controller import MetaController
from rtmdk.engines.causal import CausalInferenceEngine

from rtmdk.memory.geometry import exp_map_poincare
from rtmdk.memory.config import (
    ConsolidationMode, RTMDKConfig,
)
from rtmdk.nodes import (
    MemoryNode, CounterfactualResult,
)
import functools
import math
import random
import re
import os

import hashlib
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray


import logging

from rtmdk.memory.field_initializer import FieldInitializer
from rtmdk.memory.utils import _sanitize_path

logger = logging.getLogger(__name__)

# Stop-word lists for content-word extraction in semantic phase
_STOP_WORDS_EN = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "we", "us", "our", "ours", "ourselves", "i", "me", "my",
    "myself", "mine", "about", "against", "out", "up", "down", "off",
    "over", "s", "t", "don", "doesn", "didn", "wasn", "weren", "haven",
    "hasn", "hadn", "won", "wouldn", "shouldn", "isn", "aren", "ain",
    "let", "ll", "re", "ve", "y", "ma", "d", "o", "an", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
})

_STOP_WORDS_RU = frozenset({
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни",
    "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж",
    "вам", "сказал", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы",
    "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто",
    "человек", "чего", "раз", "тоже", "себе", "под", "жизнь", "будет",
    "ж", "тогда", "кто", "этот", "говорил", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "сказала", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между", "это", "который", "которая", "которые",
    "которых", "которому", "которой", "которым", "которыми", "котором",
    "котором", "какой", "какая", "какое", "какие", "какого", "какой",
    "какому", "каким", "каком", "такой", "такая", "такое", "такие",
    "такого", "такой", "такому", "таким", "таком", "весь", "вся",
    "все", "всего", "всему", "всем", "всеми", "всех", "всею",
})

_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_RU

# Phase 5: dataclass nodes extracted to rtmdk.nodes

# Phase 15: New modules
try:
    from rtmdk.support.version_control import VersionControl
    VC_AVAILABLE = True
except ImportError:
    VC_AVAILABLE = False

# Phase 16: New modules (cleaned: removed toy implementations)

try:
    UMP_AVAILABLE = True
except ImportError:
    UMP_AVAILABLE = False

# Phase 17: RoleShardRouter
try:
    from rtmdk.support.role_shard_router import RoleShardRouter, DEFAULT_ROLE
    ROLE_SHARD_AVAILABLE = True
except ImportError:
    ROLE_SHARD_AVAILABLE = False
    DEFAULT_ROLE = "default"  # Fallback

# Torch availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# ============================================================================
# CONSTANTS: Named constants for magic numbers
# ============================================================================

# Statistical constants
CHI_SQUARED_CRITICAL_DF1 = 3.84  # Chi-squared critical value (df=1, p=0.05)
CHI_SQUARED_CRITICAL_DF2 = 5.99  # Chi-squared critical value (df=2, p=0.05)

# Consolidation constants
CONSOLIDATION_DISTANCE_THRESHOLD = 2.5
CONSOLIDATION_PROBABILITY = 0.15
CRYSTALLIZATION_SIMILARITY_HIGH = 0.75
CRYSTALLIZATION_SIMILARITY_LOW = 0.6

# Session retrieval boost
SESSION_BOOST_FACTOR = 1.3  # 30% boost for session-matching nodes

# Performance limits
MAX_TENSION_SCAN = 200
CACHE_INVALID_HASH_MODULUS = 5

# File limits
MAX_NODE_TEXT_LENGTH = 10000
SECURE_FILE_PERMISSIONS = 0o600

# Frequency constants (steps)
SELF_SUPERVISION_FREQ = 20
ODE_SMOOTHNESS_FREQ = 10
TENSION_CHECK_FREQ = 100
HEALING_CHECK_FREQ = 50
SYMBOLIC_OVERLAY_FREQ = 50
META_KERNEL_ADAPT_FREQ = 5
MAX_NODES_PRUNE_CHECK_FREQ = 10


def _enum_value(val, default):
    """Safely extract enum value for serialization."""
    return val.value if isinstance(
        val, Enum) else (
        val if val is not None else default)


def _locked(method):
    """Decorator that wraps method in self._write_lock RLock."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)
    return wrapper


class RTMDKField:
    def __init__(
            self,
            config: RTMDKConfig,
            projection_matrix: Optional[NDArray] = None,
            wal_path: Optional[str] = None):
        FieldInitializer(self, config, projection_matrix, wal_path).initialize()

    # ------------------------------------------------------------------
    # Projection-manager aliases (backward-compatible during refactor)
    # ------------------------------------------------------------------
    @property
    def projection_learner(self):
        return self._projection_mgr.projection_learner if self._projection_mgr else None

    @property
    def _raw_projection(self):
        return self._projection_mgr._raw_projection if self._projection_mgr else None

    @property
    def sot_tokenizer(self):
        return self._projection_mgr.sot_tokenizer if self._projection_mgr else None

    @property
    def sot_hebbian(self):
        return self._projection_mgr.sot_hebbian if self._projection_mgr else None

    @property
    def _sot_field_ema(self):
        return self._projection_mgr._sot_field_ema if self._projection_mgr else None

    # ------------------------------------------------------------------
    # Cache-manager aliases (backward-compatible during refactor)
    # ------------------------------------------------------------------
    @property
    def _cached_positions(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_positions

    @_cached_positions.setter
    def _cached_positions(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_positions = v

    @property
    def _cached_phases(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_phases

    @_cached_phases.setter
    def _cached_phases(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_phases = v

    @property
    def _cached_amplitudes(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_amplitudes

    @_cached_amplitudes.setter
    def _cached_amplitudes(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_amplitudes = v

    @property
    def _cached_saliences(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_saliences

    @_cached_saliences.setter
    def _cached_saliences(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_saliences = v

    @property
    def _cached_modal_weights(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_modal_weights

    @_cached_modal_weights.setter
    def _cached_modal_weights(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_modal_weights = v

    @property
    def _cached_gates(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_gates

    @_cached_gates.setter
    def _cached_gates(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_gates = v

    @property
    def _cached_causal_boost(self) -> Optional[NDArray]:
        return self._cache_mgr._cached_causal_boost

    @_cached_causal_boost.setter
    def _cached_causal_boost(self, v: Optional[NDArray]) -> None:
        self._cache_mgr._cached_causal_boost = v

    @property
    def _cache_dirty(self) -> bool:
        return self._cache_mgr._cache_dirty

    @_cache_dirty.setter
    def _cache_dirty(self, v: bool) -> None:
        self._cache_mgr._cache_dirty = v

    @property
    def _node_id_to_cached_idx(self) -> Dict[str, int]:
        return self._cache_mgr._node_id_to_cached_idx

    @_node_id_to_cached_idx.setter
    def _node_id_to_cached_idx(self, v: Dict[str, int]) -> None:
        self._cache_mgr._node_id_to_cached_idx = v

    def _build_node_cache(self) -> None:
        """Rebuild cached arrays — delegates to NodeCacheManager."""
        self._cache_mgr.build(self)

    @staticmethod
    def _extract_text(content: Dict) -> str:
        """Extract primary text from node content, handling v1/v2 formats."""
        text = content.get("text", "")
        if text:
            return text
        return f"{content.get('input_text', '')} {content.get('output_text', '')}".strip()

    def _project(self, embedding: NDArray) -> NDArray:
        return self._projection_mgr.project(embedding)

    def _project_batch(self, embeddings: NDArray) -> NDArray:
        """Vectorized projection for batch inserts."""
        return self._projection_mgr.project_batch(embeddings)

    def _semantic_phase(
            self,
            session_id: Optional[str] = None,
            content: Optional[Dict] = None,
            modality: str = "text",
    ) -> float:
        """Compute a semantically meaningful phase from session/topic/content.

        Nodes sharing the same session or topic cluster into phase
        neighbourhoods, so phase coupling (cos Δφ) naturally boosts
        intra-cluster retrieval.  The phase is deterministic for identical
        keys, with a small spread to avoid exact collisions.

        Uses content-bearing words (skipping stop words) for robust
        cross-lingual phase extraction.
        """
        parts = []
        if session_id:
            parts.append(f"s:{session_id}")
        if content:
            topic = content.get("topic", "")
            if topic:
                parts.append(f"t:{topic}")
            text = content.get("text", "") or content.get("input_text", "")
            if text:
                # Extract alphanumeric tokens (supports Unicode)
                tokens = re.findall(r"[\w']+", text.lower())
                # Filter stop words, keep content words
                content_words = [w for w in tokens if w not in _STOP_WORDS and len(w) > 2]
                if content_words:
                    # Deduplicate while preserving order, then take top-3
                    seen = set()
                    deduped = []
                    for w in content_words:
                        if w not in seen:
                            seen.add(w)
                            deduped.append(w)
                    words = deduped[:3]
                else:
                    # Fallback to first 3 raw tokens
                    words = tokens[:3]
                if words:
                    parts.append(f"w:{'_'.join(words)}")
        parts.append(f"m:{modality}")

        seed_text = "|".join(parts)
        cached = self._semantic_phase_cache.get(seed_text)
        if cached is not None:
            return cached
        h = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
        base = (int(h, 16) % 6283) / 1000.0  # [0, 2π]
        rng = random.Random(h)
        spread = rng.uniform(-0.15, 0.15)
        result = (base + spread) % (2 * math.pi)
        self._semantic_phase_cache[seed_text] = result
        return result

    def _get_phase(
            self,
            session_id: Optional[str] = None,
            embedding: Optional[NDArray] = None,
            modality: str = "text",
            content: Optional[Dict] = None,
    ) -> float:
        phase = self._semantic_phase(session_id, content, modality)
        if self.cfg.cross_modal and modality in self.cfg.modal_phase_offsets:
            phase += self.cfg.modal_phase_offsets[modality]
        elif self.cfg.multimodal and modality in self.cfg.modality_phase_shifts:
            phase += self.cfg.modality_phase_shifts[modality]
        return phase % (2 * np.pi)

    def _ensure_adaptive_pc(self, query_latent: NDArray) -> None:
        self._query_mgr._ensure_adaptive_pc(query_latent)

    def _resonance_response(self, query_latent: NDArray, query_phase: float, node: MemoryNode, query_modality: str = "text") -> float:
        return self._query_mgr._resonance_response(query_latent, query_phase, node, query_modality)

    def _batch_resonance(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        return self._query_mgr._batch_resonance(query_latents, query_phases, node_ids)

    def _batch_resonance_nodes(self, query_latents: NDArray, query_phases: NDArray, nodes: List[Any]) -> NDArray:
        return self._query_mgr._batch_resonance_nodes(query_latents, query_phases, nodes)

    def _batch_resonance_numpy(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        return self._query_mgr._batch_resonance_numpy(query_latents, query_phases, node_ids)

    def _batch_resonance_cached(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        return self._query_mgr._batch_resonance_cached(query_latents, query_phases, node_ids)

    def _batch_resonance_torch(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        return self._query_mgr._batch_resonance_torch(query_latents, query_phases, node_ids)

    def _compute_resonance_chunk(self, positions, phases, amplitudes, saliences, modal_weights, gates, causal_boost, query_latent, query_phase, bw=None, pc=None):
        return self._query_mgr._compute_resonance_chunk(positions, phases, amplitudes, saliences, modal_weights, gates, causal_boost, query_latent, query_phase, bw, pc)

    def _query_vectorized(self, query_latent: NDArray, query_phase: float, top_k: int, modality: str, session_id: Optional[str], t0: float) -> List[Tuple[str, float, MemoryNode]]:
        return self._query_mgr._query_vectorized(query_latent, query_phase, top_k, modality, session_id, t0)

    def _query_cache_key(self, query_latent: NDArray, phase: float, top_k: int, modality: str, session_id: Optional[str]) -> str:
        return self._query_mgr._query_cache_key(query_latent, phase, top_k, modality, session_id)

    def _apply_adaptive_top_k(self, results: List[Tuple[str, float, MemoryNode]]) -> List[Tuple[str, float, MemoryNode]]:
        return self._query_mgr._apply_adaptive_top_k(results)

    def query_batch(self, embeddings: NDArray, phase: float = 0.0, top_k: Optional[int] = None, modality: str = "text", session_id: Optional[str] = None, query_texts: Optional[List[str]] = None) -> List[List[Tuple[str, float, MemoryNode]]]:
        return self._query_mgr.query_batch(embeddings, phase, top_k, modality, session_id, query_texts)

    def query(self, embedding: NDArray, phase: float = 0.0, top_k: Optional[int] = None, modality: str = "text", session_id: Optional[str] = None, query_text: Optional[str] = None) -> List[Tuple[str, float, MemoryNode]]:
        return self._query_mgr.query(embedding, phase, top_k, modality, session_id, query_text)

    def batch_query(self, embeddings: List[NDArray], phases: Optional[List[float]] = None, top_k: Optional[int] = None, modality: str = "text", session_id: Optional[str] = None) -> List[List[Tuple[str, float, MemoryNode]]]:
        return self._query_mgr.batch_query(embeddings, phases, top_k, modality, session_id)

    def fit_projection(self, corpus_embeddings: NDArray) -> None:
        """Batch-fit projection learner on a corpus of embeddings."""
        self._projection_mgr.fit_projection(corpus_embeddings)

    def sot_bootstrap(
            self,
            texts: List[str],
            teacher_model: str = 'all-MiniLM-L6-v2',
            fit_projection_only: bool = True,
            n_epochs: int = 30):
        self._projection_mgr.sot_bootstrap(
            texts, teacher_model=teacher_model,
            fit_projection_only=fit_projection_only, n_epochs=n_epochs)

    def sot_contrastive_step(
        self,
        query_text: str,
        positive_text: str,
        negative_texts=None,
        lr: float = 0.01,
    ):
        self._projection_mgr.sot_contrastive_step(
            query_text, positive_text, negative_texts, lr=lr)

    def _sot_retrieval_feedback(
            self, query_latent: np.ndarray, results: List[Tuple[str, float, Any]]):
        self._projection_mgr.sot_retrieval_feedback(
            query_latent, results,
            negatives_per_query=self.cfg.sot_negatives_per_query)

    def query_by_text(self, text: str, top_k: Optional[int] = None, session_id: Optional[str] = None) -> List[Tuple[str, float, Any]]:
        return self._query_mgr.query_by_text(text, top_k, session_id)

    @property
    def causal_engine(self) -> Optional["CausalInferenceEngine"]:
        if self._causal_engine_initialized and self._causal_engine is None:
            self._causal_engine = CausalInferenceEngine(
                min_samples=self.cfg.causal_discovery_min_samples,
                p_threshold=self.cfg.causal_p_threshold,
                adjustment_sets_enabled=self.cfg.causal_adjustment_sets)
            self._resonance_engine.causal_engine = self._causal_engine
        return self._causal_engine

    @causal_engine.setter
    def causal_engine(self, value: Optional["CausalInferenceEngine"]):
        self._causal_engine = value
        self._causal_engine_initialized = value is not None

    # B2: Lazy property for meta-controller
    @property
    def meta_controller(self) -> Optional["MetaController"]:
        if self._meta_controller_initialized and self._meta_controller is None:
            self._meta_controller = MetaController(
                n_trials=self.cfg.meta_n_trials,
                optimize_params=self.cfg.meta_optimize_params,
                optimization_freq=self.cfg.meta_optimization_freq,
            )
        return self._meta_controller

    @meta_controller.setter
    def meta_controller(self, value: Optional["MetaController"]):
        self._meta_controller = value
        self._meta_controller_initialized = value is not None

    @_locked
    def add_node(self, embedding: NDArray, content: Dict, phase: Optional[float] = None, node_id: Optional[str] = None, session_id: Optional[str] = None, modality: str = "text", skip_projection: bool = False, modal_embedding: Optional[NDArray] = None) -> str:
        return self._node_mgr.add_node(embedding, content, phase, node_id, session_id, modality, skip_projection, modal_embedding)
    def add_nodes_batch(self, embeddings: NDArray, contents: List[Dict], phases: Optional[NDArray] = None, node_ids: Optional[List[str]] = None, session_ids: Optional[List[str]] = None, modalities: Optional[List[str]] = None, skip_projection: bool = False, modal_embeddings: Optional[NDArray] = None) -> List[str]:
        return self._node_mgr.add_nodes_batch(embeddings, contents, phases, node_ids, session_ids, modalities, skip_projection, modal_embeddings)

    def delete_nodes(self, node_ids: List[str]) -> None:
        self._node_mgr.delete_nodes(node_ids)

    def queue_add_nodes(self, embeddings: NDArray, contents: List[Dict], modalities: Optional[List[str]] = None) -> None:
        self._node_mgr.queue_add_nodes(embeddings, contents, modalities)
    def _apply_conformal_filter(self, results: List[Tuple[str, float, MemoryNode]]) -> List[Tuple[str, float, MemoryNode]]:
        return self._query_mgr._apply_conformal_filter(results)

    def _invalidate_tension_cache(self, node_id: Optional[str] = None) -> None:
        self._topology_mgr.invalidate_tension_cache(node_id)

    def _sweep_tension_cache(self) -> None:
        self._topology_mgr.sweep_tension_cache()

    def _compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        return self._topology_mgr.compute_tension(node_id, neighborhood_radius)

    def _soft_gate(self, tension: float) -> float:
        return self._topology_mgr.soft_gate(tension)

    def get_effective_threshold(self) -> float:
        return self.adaptive_threshold.get_threshold(
        ) if self.adaptive_threshold else self.cfg.tension_threshold

    @_locked
    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        """Run one consolidation pass over the field.

        Delegated to ConsolidationManager for maintainability.
        """
        return self._consolidation_mgr.consolidate(mode)

    def _self_supervise(self) -> None:
        self._cognitive_mgr.self_supervise()

    def _check_tda(self) -> None:
        self._cognitive_mgr.check_tda()

    def _encode_field_state(self) -> NDArray:
        return self._cognitive_mgr.encode_field_state()

    # ========================================================================
    # Operational methods (delegated to OperationalManager)
    # ========================================================================
    def calibrate(self, query_embedding: NDArray, node_id: str,
                  is_relevant: bool) -> None:
        self._operational_mgr.calibrate(query_embedding, node_id, is_relevant)

    def imagine_counterfactual(self, base_query: NDArray,
                               intervention: Dict[str, float]) -> List[Dict]:
        return self._operational_mgr.imagine_counterfactual(base_query, intervention)

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        return self._operational_mgr.rollback_consolidation(n_steps)

    def do_intervention(self, node_id: str, new_embedding: NDArray):
        self._operational_mgr.do_intervention(node_id, new_embedding)

    def clear_interventions(self):
        self._operational_mgr.clear_interventions()

    def get_field_health(self) -> Dict:
        return self._operational_mgr.get_field_health()

    def counterfactual_query(self, intervention: Dict[str, Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        return self._operational_mgr.counterfactual_query(
            intervention, query_nodes, evidence)

    def get_causal_summary(self) -> Dict:
        return self._operational_mgr.get_causal_summary()

    def _merge_latents(self, node, partner):
        self._merge_mgr.merge_latents(node, partner)

    def _train_learned_consolidator(self):
        self._merge_mgr.train_learned_consolidator()

    def _prune_dead_nodes(self) -> None:
        self._topology_mgr.prune_dead_nodes()

    def _check_field_integrity(self) -> Dict[str, Any]:
        return self._topology_mgr.check_field_integrity()
    def step(self, inputs: Optional[List[Dict]] = None):
        self._step_counter += 1

        # Throttle: Skip non-critical heavy tasks if backpressure is high
        backpressure_ok = self._backpressure_events < 3 and not self._heavy_modules_degraded

        if inputs:
            for inp in inputs:
                emb = inp["embedding"]
                phase = inp.get("phase", 0.0)
                content = inp.get("content", {})
                session_id = inp.get("session_id")
                modality = inp.get("modality", "text")
                text = content.get("text", "")

                # Phase 21: SOT tokenization and optional query-by-text
                sot_tokens = self._projection_mgr.sot_encode(text)
                if sot_tokens:
                    self._projection_mgr.sot_record_cooccurrence(sot_tokens)

                # Validate embedding dimension — allow both embedding_dim and
                # latent_dim
                emb_dim = len(emb)
                if emb_dim not in (
                        self.cfg.embedding_dim,
                        self.cfg.latent_dim):
                    logger.warning(
                        f"Embedding dimension mismatch in step(): "
                        f"expected {self.cfg.embedding_dim} or "
                        f"{self.cfg.latent_dim}, got {emb_dim}. Skipping.")
                    continue

                results = self.query(
                    emb,
                    phase,
                    top_k=max(
                        1,
                        self.cfg.sot_negatives_per_query +
                        1),
                    modality=modality)
                if results and results[0][1] > 0.3:
                    nid, _, node = results[0]
                    target = emb if emb_dim == self.cfg.latent_dim else self._project(
                        emb)
                    if self.cfg.hyperbolic:
                        # Riemannian SGD: gradient is scaled by conformal
                        # factor 1/λ²
                        grad_e = target - node.latent_pos
                        norm_sq = np.sum(node.latent_pos ** 2)
                        conformal = (1.0 - norm_sq /
                                     (self.cfg.ball_radius ** 2)) ** 2 / 4.0
                        grad_r = conformal * grad_e
                        node.latent_pos = exp_map_poincare(
                            -self.cfg.attraction_lr * grad_r,
                            node.latent_pos,
                            self.cfg.ball_radius,
                        )
                    else:
                        node.latent_pos += self.cfg.attraction_lr * \
                            (target - node.latent_pos)
                    pd = (phase - node.phase + np.pi) % (2 * np.pi) - np.pi
                    node.phase = (
                        node.phase + self.cfg.phase_sync_lr * pd) % (2 * np.pi)
                    node.amplitude = min(1.0, node.amplitude + 0.05)
                    node.salience = min(1.0, node.salience + 0.03)
                else:
                    self.add_node(
                        emb,
                        content,
                        phase,
                        session_id=session_id,
                        modality=modality)

                # Phase 21: Contrastive Hebbian update on field nodes
                if self._projection_mgr.has_sot_hebbian and results and len(self.node_index) > 1:
                    snap_id_to_idx = {
                        nid: idx for idx, nid in enumerate(
                            self.node_index)}
                    pos_indices = []
                    for nid, _, _ in results:
                        idx = snap_id_to_idx.get(nid)
                        if idx is not None:
                            pos_indices.append(idx)
                    n_neg = min(
                        self.cfg.sot_negatives_per_query, len(
                            self.node_index) - len(pos_indices))
                    neg_indices = []
                    if n_neg > 0:
                        all_idx = set(range(len(self.node_index)))
                        available = list(all_idx - set(pos_indices))
                        if available:
                            neg_indices = self._rng.choice(available, size=min(
                                n_neg, len(available)), replace=False).tolist()
                    if pos_indices:
                        positions = np.array([self.nodes[self.node_index[i]].latent_pos for i in range(
                            len(self.node_index))], dtype=np.float32)
                        self._projection_mgr.sot_contrastive_hebbian_field_update(
                            positions, pos_indices, neg_indices)
                        # Write back
                        for i in range(len(self.node_index)):
                            self.nodes[self.node_index[i]
                                       ].latent_pos = positions[i]

                # Phase 21: Contrastive Hebbian update on token embeddings
                if sot_tokens and len(sot_tokens) > 1:
                    vocab_ids = self._projection_mgr.sot_vocab_ids()
                    n_neg = min(
                        self.cfg.sot_negatives_per_query,
                        len(vocab_ids) - len(sot_tokens))
                    self._projection_mgr.sot_contrastive_hebbian_token_update(
                        sot_tokens, vocab_ids,
                        negatives_per_query=self.cfg.sot_negatives_per_query,
                        hard_negatives=self.cfg.sot_hard_negatives)

                # Phase 21: Periodic merge
                if self._projection_mgr.has_sot and self._step_counter % self.cfg.sot_merge_freq == 0 and self._step_counter > 0:
                    candidates = self._projection_mgr.sot_propose_merges(5)
                    for pair in candidates:
                        score = self._projection_mgr.sot_cooccurrence_score(pair)
                        if score >= self.cfg.sot_merge_threshold and score >= self.cfg.sot_min_cooccurrence:
                            try:
                                self._projection_mgr.sot_merge(pair)
                            except RuntimeError:
                                break  # Max vocab reached

        self._run_periodic_tasks(backpressure_ok)

    def _run_periodic_tasks(self, backpressure_ok: bool) -> None:
        """Execute all periodic maintenance tasks (consolidation, decay, pruning, etc.).

        Delegated to StepScheduler for maintainability.
        """
        self._scheduler.run(backpressure_ok)

    def _self_heal(self) -> List[Dict]:
        return self._operational_mgr.self_heal()

    # ======================================================================
    # PHASE 12 TRACK 1: SPARSE RESONANT ROUTING (MoE-memory)
    # ========================================================================

    def _get_node_shard(self, node_id: str) -> int:
        return self._routing_mgr.get_node_shard(node_id)

    def _route_query(self, query_latent: NDArray, top_shards: int = 3) -> List[int]:
        return self._routing_mgr.route_query(query_latent, top_shards)

    def _update_shard_centers(self) -> None:
        self._routing_mgr.update_shard_centers()

    def _update_shard_centers_bm25(self) -> None:
        self._routing_mgr.update_shard_centers_bm25()

    # ========================================================================
    # PHASE 12 TRACK 2: COGNITIVE CONTEXT COMPRESSION
    # ========================================================================

    def _cognitive_compress(self, results: List[Tuple[str, float, MemoryNode]]) -> str:
        return self._cognitive_mgr.cognitive_compress(results)

    # ========================================================================
    # PHASE 12 TRACK 3: CRYSTALLIZATION (episodic → semantic/procedural)
    # ========================================================================

    def _crystallize_recurring(self, window: int = 100, similarity_thresh: float = 0.75) -> None:
        self._crystallization_mgr.crystallize_recurring(window, similarity_thresh)

    # ========================================================================
    # PHASE 12 TRACK 4: ASYNC MULTI-THREADED EVOLUTION PIPELINE
    # ========================================================================

    async def _start_workers(self):
        await self._async_pipeline_mgr.start_workers()

    def _track_queue_depth(self):
        self._async_pipeline_mgr._track_queue_depth()

    # ========================================================================
    # PHASE 13 TRACK 4: LOW-RANK COMPRESSION
    # ========================================================================

    def _compress_field(self):
        """Compress node latent positions via incremental SVD."""
        if not self.low_rank_compressor or len(self.nodes) < 10:
            return
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        compressed, reconstructed = self.low_rank_compressor.compress(
            positions)
        ratio = self.low_rank_compressor.get_compression_ratio(positions.shape)
        self.stats["compression_ratio"] = ratio
        self.stats["compression_updates"] = self.low_rank_compressor._update_count
        # Update node positions with reconstructed (lossy but preserves
        # resonance)
        for i, nid in enumerate(self.node_index):
            if i < len(reconstructed) and nid in self.nodes:
                self.nodes[nid].latent_pos = reconstructed[i].astype(
                    np.float32)

    # ========================================================================
    # PHASE 13 TRACK 1: GOAL MANAGEMENT
    @_locked
    def export_field(self, path: str, fmt: Optional[str] = None):
        """Export field state to file.

        Args:
            path: Output file path
            fmt: "msgpack" (default if available) or "json" (fallback)
        """
        if fmt is None:
            try:
                fmt = "msgpack"
            except ImportError:
                fmt = "json"
        path = _sanitize_path(path)
        # Safety check: prevent overwriting non-empty file with empty memory
        n_nodes = len(self.nodes)
        if n_nodes == 0 and os.path.exists(path):
            try:
                existing_size = os.path.getsize(path)
                if existing_size > 1000:  # File has content (>1KB)
                    logger.warning(
                        f"export_field blocked: refusing to overwrite "
                        f"{path} ({existing_size / 1024:.0f}KB) with empty "
                        f"memory (0 nodes). This prevents accidental data loss.")
                    return  # Silently skip export to protect existing data
            except OSError:
                pass  # If we can't check, proceed with export

        logger.info(f"export_field: exporting {n_nodes} nodes to {path}")
        from rtmdk.memory.serialization import FieldSerializer
        FieldSerializer.field_to_file(self, path, fmt)
        self._dirty = False
        self.wal.truncate()

    def close(self) -> None:
        """Release background resources (async builder, WAL)."""
        if self._async_index_builder is not None:
            self._async_index_builder.close()
            self._async_index_builder = None
        self.wal.close()

    def get_state(self) -> Dict[str, Any]:
        """Get lightweight state dict for SOT persistence."""
        state: Dict[str, Any] = {
            "step_counter": self._step_counter,
        }
        state.update(self._projection_mgr.get_state())
        return state

    def load_state(self, state: Dict[str, Any]):
        """Load lightweight state dict for SOT persistence."""
        self._step_counter = state.get("step_counter", self._step_counter)
        self._projection_mgr.load_state(state)

    @classmethod
    def import_field(cls, path: str, embedder: Callable,
                     wal_path: Optional[str] = None):
        path = _sanitize_path(path)
        from rtmdk.memory.serialization import FieldSerializer
        return FieldSerializer.field_from_file(
            path, embedder, wal_path=wal_path)

    def export_to_dict(self) -> Dict:
        """Export field state to a dict (for UMP and other protocols)."""
        from rtmdk.memory.serialization import FieldSerializer
        return FieldSerializer.field_to_dict(self)

    @classmethod
    def import_from_dict(cls, data: Dict, embedder: Callable):
        """Import field state from a dict (for UMP and other protocols)."""
        from rtmdk.memory.serialization import FieldSerializer
        return FieldSerializer.field_from_dict(data, embedder)

# ============================================================================
# RTMDKMemory v7
# ============================================================================
