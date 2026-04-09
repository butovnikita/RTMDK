"""
rtmdk_memory_v5.py
Резонансно-топологическая память — Версия 5.0

Фаза 6: Каузально-топологическая память
  - CausalInferenceEngine: P(Y|do(X)) вместо косинусного сходства
  - DoCalculusValidator: проверка консолидации через do-calculus
  - CounterfactualQueryEngine: ответы на "Что если?"
  - ContradictionMarker: do(A)→B vs do(C)→B маркировка
"""

from __future__ import annotations
import asyncio
import json
import math
import re
import time
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Set, FrozenSet
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
# КОНФИГУРАЦИЯ v5
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

    # ТРЕК 3: Каузальность (legacy)
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

    # ФАЗА 5
    meta_adaptive: bool = False
    meta_adaptation_lr: float = 0.005
    semantic_density_window: int = 50
    session_style_window: int = 30
    uncertainty_window: int = 20
    kurtosis_target_min: float = 1.5
    kurtosis_target_max: float = 4.0
    self_healing: bool = False
    healing_check_freq: int = 25
    dead_zone_threshold: float = 0.15
    hyperconvergence_threshold: float = 0.05
    fragmentation_threshold: float = 0.6
    healing_strength: float = 0.1
    max_healing_nodes_per_step: int = 5

    # ФАЗА 6: Каузально-топологическая память
    causal_topological: bool = False
    causal_discovery_min_samples: int = 20
    causal_p_threshold: float = 0.05
    do_calculus_validation: bool = True
    counterfactual_enabled: bool = False
    counterfactual_max_depth: int = 3
    contradiction_detection: bool = True
    contradiction_threshold: float = 0.3
    causal_adjustment_sets: bool = True

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
# ТИПЫ ДАННЫХ v5
# ============================================================================

@dataclass
class CausalEdge:
    """Каузальная связь между узлами."""
    source: str
    target: str
    strength: float  # P(Y|do(X))
    confidence: float  # уверенность в оценке
    adjustment_set: List[str] = field(default_factory=list)  # набор корректировки
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
    """Запись противоречия: do(A)→B vs do(C)→B."""
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
    """Результат контрфактуального запроса."""
    query: str
    intervention: Dict[str, Any]  # do(X=x)
    predicted_outcomes: List[Tuple[str, float]]  # (node_id, P(Y|do(X)))
    confidence: float
    reasoning_path: List[str]
    assumptions: List[str]

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "intervention": self.intervention,
            "predicted_outcomes": [{"node": n, "probability": p} for n, p in self.predicted_outcomes],
            "confidence": self.confidence,
            "reasoning_path": self.reasoning_path,
            "assumptions": self.assumptions,
        }


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
    # Фаза 6: каузальные атрибуты
    causal_effects: Dict[str, float] = field(default_factory=dict)  # P(Y|do(self))
    do_interventions: Dict[str, NDArray] = field(default_factory=dict)
    is_causal_root: bool = False
    causal_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["latent_pos"] = self.latent_pos.tolist()
        if self.pre_consolidation_pos is not None:
            d["pre_consolidation_pos"] = self.pre_consolidation_pos.tolist()
        if self.gradient_cache is not None:
            d["gradient_cache"] = self.gradient_cache.tolist()
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
        for k, v in data.get("do_interventions", {}).items():
            if isinstance(v, list):
                data["do_interventions"][k] = np.array(v, dtype=np.float32)
        return cls(**data)


# ============================================================================
# ФАЗА 6: CAUSAL INFERENCE ENGINE
# ============================================================================

class CausalInferenceEngine:
    """
    Вычисление P(Y|do(X)) из наблюдательных данных.
    Реализует:
    - Backdoor criterion для идентификации каузальных эффектов
    - Frontdoor criterion когда backdoor недоступен
    - Do-calculus rules для трансформации выражений
    - Counterfactual reasoning через abduction-action-prediction
    """

    def __init__(self, min_samples: int = 20, p_threshold: float = 0.05,
                 adjustment_sets_enabled: bool = True):
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.adjustment_sets_enabled = adjustment_sets_enabled

        # Граф: node -> set of parents
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.ancestors: Dict[str, Set[str]] = defaultdict(set)

        # Наблюдательные данные
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._conditional_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._node_counts: Dict[str, int] = defaultdict(int)
        self._total_observations = 0

        # Каузальные эффекты: (cause, effect) -> P(effect|do(cause))
        self.causal_effects: Dict[Tuple[str, str], CausalEdge] = {}

        # Contradictions
        self.contradictions: Dict[str, ContradictionRecord] = {}
        self._contradiction_counter = 0

        # Counterfactual cache
        self._counterfactual_cache: Dict[str, CounterfactualResult] = {}

    # --- Data recording ---
    def record_cooccurrence(self, a: str, b: str):
        """Record direct cooccurrence between two nodes."""
        self._cooccurrence[(a, b)] += 1
        self._cooccurrence[(b, a)] += 1
        self._node_counts[a] += 1
        self._node_counts[b] += 1
        self._total_observations += 1

    def record_observation(self, active_nodes: List[str], context: Optional[Dict] = None):
        """Записать наблюдение: какие узлы активны одновременно (без double-counting)."""
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

    # --- Causal discovery ---
    def discover_causal_structure(self) -> Dict[str, Set[str]]:
        """
        Обнаружить каузальную структуру через PC-algorithm упрощённую версию:
        1. Построить полный граф сопутствующих узлов
        2. Удалить рёбра при условной независимости
        3. Ориентировать рёбра через v-structures
        """
        nodes = list(self._node_counts.keys())
        if len(nodes) < 3 or self._total_observations < self.min_samples:
            return dict(self.parents)

        # Step 1: Build skeleton (correlation-based)
        skeleton: Dict[str, Set[str]] = defaultdict(set)
        sep_sets: Dict[Tuple[str, str], FrozenSet[str]] = {}

        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                # Test marginal independence
                if self._test_independence(a, b, set()):
                    continue  # Independent, no edge
                skeleton[a].add(b)
                skeleton[b].add(a)

                # Test conditional independence with each other node as separator
                for z in nodes:
                    if z == a or z == b:
                        continue
                    if self._test_independence(a, b, {z}):
                        sep_sets[(a, b)] = frozenset({z})
                        sep_sets[(b, a)] = frozenset({z})
                        break

        # Step 2: Orient v-structures (X -> Z <- Y where X and Y not connected)
        new_parents: Dict[str, Set[str]] = defaultdict(set)
        new_children: Dict[str, Set[str]] = defaultdict(set)

        for z in nodes:
            neighbors = list(skeleton.get(z, set()))
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    # X and Y not connected → potential v-structure
                    if y not in skeleton.get(x, set()):
                        # Check if Z is NOT in sep_set(X, Y)
                        sep = sep_sets.get((x, y), frozenset())
                        if z not in sep:
                            # X -> Z <- Y
                            new_parents[z].add(x)
                            new_parents[z].add(y)
                            new_children[x].add(z)
                            new_children[y].add(z)

        # Step 3: Propagate orientations (Meek rules simplified)
        changed = True
        max_iterations = 10
        while changed and max_iterations > 0:
            changed = False
            max_iterations -= 1
            for a in nodes:
                for b in skeleton.get(a, set()):
                    if b not in new_parents.get(a, set()) and a not in new_parents.get(b, set()):
                        # Undirected edge a - b
                        # If there exists c such that c -> a and c not connected to b
                        for c in new_children.get(a, set()):
                            if c not in skeleton.get(b, set()) and c != b:
                                # c -> a, so orient a -> b
                                new_parents[b].add(a)
                                new_children[a].add(b)
                                changed = False  # Continue iterating

        self.parents = new_parents
        self.children = new_children
        self._compute_ancestors()
        return dict(self.parents)

    def _test_independence(self, a: str, b: str, cond_set: Set[str]) -> bool:
        """Test if a ⊥⊥ b | cond_set using chi-squared test."""
        n_ab = self._cooccurrence.get((a, b), 0)
        n_a = self._node_counts.get(a, 0)
        n_b = self._node_counts.get(b, 0)
        n = max(self._total_observations, 1)

        if n_a < 3 or n_b < 3 or n_ab < 2:
            return True  # Not enough data → assume independent

        if not cond_set:
            # Marginal independence: test if P(a,b) ≈ P(a)P(b)
            expected = (n_a / n) * (n_b / n) * n
            if expected < 5:
                return True
            chi2 = (n_ab - expected) ** 2 / expected
            # Chi-squared with 1 df, p > 0.05 → independent
            return chi2 < 3.84  # chi2 critical value for p=0.05, df=1
        else:
            # Conditional independence: simplified test
            # Check if association persists within each level of cond
            for cond_var in cond_set:
                # Simplified: check if correlation drops when conditioning
                cond_count = sum(1 for k in self._conditional_counts
                                if k[0] == a and k[1] == b and cond_var in k[2])
                if cond_count > 0:
                    return True  # Conditioning changes the relationship
            return False

    def _compute_ancestors(self):
        """Compute ancestor sets for all nodes."""
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

    # --- Do-calculus ---
    def compute_do_probability(self, effect: str, intervention: str,
                               evidence: Optional[Dict[str, Any]] = None) -> float:
        """
        Вычислить P(effect | do(intervention)) используя do-calculus.

        Rules:
        1. P(y|do(x),z,w) = P(y|do(x),w) if Y ⊥⊥ Z | X,W in G_do(x)
        2. P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y ⊥⊥ Z | X,W in G_do(x,z)
        3. P(y|do(x),z) = P(y|do(x)) if Y ⊥⊥ Z | X,W in G_do(x)_do(z)
        """
        cache_key = f"do({intervention})->{effect}"
        if evidence:
            cache_key += f"|{sorted(evidence.items())}"

        if cache_key in self._counterfactual_cache:
            cf = self._counterfactual_cache[cache_key]
            return cf.confidence

        # Check if we have direct causal effect
        edge = self.causal_effects.get((intervention, effect))
        if edge:
            prob = edge.strength
        else:
            # Compute from observational data using backdoor adjustment
            prob = self._backdoor_adjustment(effect, intervention)

        # Validate with do-calculus
        if self._validate_do_calculus(effect, intervention):
            confidence = min(1.0, edge.confidence if edge else 0.5)
        else:
            confidence = 0.1  # Low confidence if do-calculus fails

        # Store causal edge if not exists
        if (intervention, effect) not in self.causal_effects:
            adjustment = self._find_adjustment_set(intervention, effect)
            self.causal_effects[(intervention, effect)] = CausalEdge(
                source=intervention, target=effect, strength=prob,
                confidence=confidence, adjustment_set=list(adjustment),
                evidence_count=self._cooccurrence.get((intervention, effect), 0)
            )

        return prob

    def _backdoor_adjustment(self, effect: str, cause: str) -> float:
        """
        P(effect|do(cause)) = Σ_z P(effect|cause,z) P(z)
        where Z is a valid backdoor adjustment set.
        """
        z_set = self._find_adjustment_set(cause, effect)

        if not z_set:
            # No valid adjustment set → use naive estimation
            return self._naive_causal_estimate(cause, effect)

        # Backdoor adjustment: Σ_z P(effect|cause,z) P(z)
        total_prob = 0.0
        total_z_weight = 0.0

        for z in z_set:
            n_z = self._node_counts.get(z, 0)
            p_z = n_z / max(self._total_observations, 1)

            n_cause_z = self._cooccurrence.get((cause, z), 0)
            n_effect_cause_z = self._conditional_counts.get((effect, cause, z), 0)

            if n_cause_z > 0:
                p_effect_given_cause_z = n_effect_cause_z / n_cause_z
            else:
                p_effect_given_cause_z = 0.5  # Prior

            total_prob += p_effect_given_cause_z * p_z
            total_z_weight += p_z

        if total_z_weight > 0:
            return total_prob / total_z_weight
        return self._naive_causal_estimate(cause, effect)

    def _naive_causal_estimate(self, cause: str, effect: str) -> float:
        """P(effect|do(cause)) ≈ P(effect|cause) when no confounders detected."""
        n_cause = self._node_counts.get(cause, 0)
        n_both = self._cooccurrence.get((cause, effect), 0)
        if n_cause < 3:
            return 0.5  # Prior
        return min(1.0, n_both / n_cause)

    def _find_adjustment_set(self, cause: str, effect: str) -> Set[str]:
        """
        Find a valid backdoor adjustment set Z such that:
        1. Z blocks all backdoor paths from cause to effect
        2. Z doesn't contain descendants of cause
        """
        if not self.adjustment_sets_enabled:
            return set()

        # Backdoor paths: paths from cause to effect that start with an arrow INTO cause
        # Adjustment set: parents of cause (simplest valid set)
        parents_of_cause = self.parents.get(cause, set())

        # Remove descendants of cause from adjustment set
        descendants_of_cause = self._get_descendants(cause)
        valid_z = parents_of_cause - descendants_of_cause

        # Also consider common causes of cause and effect
        common_causes = set()
        for node in self._node_counts:
            if node == cause or node == effect:
                continue
            if effect in self.children.get(node, set()) and node in parents_of_cause:
                common_causes.add(node)

        return valid_z | common_causes

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

    def _validate_do_calculus(self, effect: str, intervention: str) -> bool:
        """
        Validate that P(effect|do(intervention)) is identifiable.
        Checks:
        1. No unblocked backdoor paths
        2. No selection bias
        3. Do-calculus rules can reduce to observational distribution
        """
        # Check 1: Can we find a valid adjustment set?
        z_set = self._find_adjustment_set(intervention, effect)

        # Check 2: Is there a frontdoor path?
        has_frontdoor = self._has_frontdoor_path(intervention, effect)

        # Check 3: Is the effect a descendant of the intervention?
        descendants = self._get_descendants(intervention)
        is_descendant = effect in descendants

        # Identifiable if: valid adjustment set exists OR frontdoor path exists OR direct descendant
        return bool(z_set) or has_frontdoor or is_descendant

    def _has_frontdoor_path(self, cause: str, effect: str) -> bool:
        """Check if there's a frontdoor path: cause -> M -> effect where M mediates."""
        for mediator in self.children.get(cause, set()):
            if effect in self.children.get(mediator, set()):
                # Check no unblocked backdoor from cause to mediator
                # and all backdoor paths from mediator to effect blocked by cause
                return True
        return False

    # --- Contradiction detection ---
    def detect_contradictions(self, threshold: float = 0.3) -> List[ContradictionRecord]:
        """
        Найти противоречия: do(A)→B и do(C)→B с противоположными эффектами.
        """
        new_contradictions = []

        # Group by effect
        effect_causes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (cause, effect), edge in self.causal_effects.items():
            if edge.strength > 0.1:  # Only consider meaningful effects
                effect_causes[effect].append((cause, edge.strength))

        for effect_node, causes in effect_causes.items():
            if len(causes) < 2:
                continue

            # Check for conflicting causes
            for i, (cause_a, strength_a) in enumerate(causes):
                for cause_b, strength_b in causes[i+1:]:
                    # Contradiction: both causes lead to same effect but are themselves
                    # negatively correlated or mutually exclusive
                    cooc = self._cooccurrence.get((cause_a, cause_b), 0)
                    n_a = self._node_counts.get(cause_a, 0)
                    n_b = self._node_counts.get(cause_b, 0)

                    if n_a > 0 and n_b > 0:
                        expected = (n_a / self._total_observations) * (n_b / self._total_observations) * self._total_observations
                        if expected > 0 and cooc / expected < (1.0 - threshold):
                            # Causes are negatively correlated → contradiction
                            self._contradiction_counter += 1
                            record = ContradictionRecord(
                                id=f"contr_{self._contradiction_counter}",
                                effect_node=effect_node,
                                causes=[(cause_a, strength_a), (cause_b, strength_b)],
                                contradiction_reason=f"Causes {cause_a} and {cause_b} are negatively correlated (cooc={cooc}, expected={expected:.1f})"
                            )
                            self.contradictions[record.id] = record
                            new_contradictions.append(record)

                            # Mark edges as contradicted
                            if (cause_a, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_a, effect_node)].is_contradicted = True
                                self.causal_effects[(cause_a, effect_node)].contradiction_reason = record.id
                            if (cause_b, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_b, effect_node)].is_contradicted = True
                                self.causal_effects[(cause_b, effect_node)].contradiction_reason = record.id

        return new_contradictions

    # --- Counterfactual reasoning ---
    def counterfactual_query(self, intervention: Dict[str, Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None,
                             max_depth: int = 3) -> CounterfactualResult:
        """
        Ответить на "Что если?": P(Y|do(X=x), evidence)

        Алгоритм abduction-action-prediction:
        1. Abduction: infer exogenous variables from evidence
        2. Action: apply do(X=x) to structural equations
        3. Prediction: compute outcomes in modified model
        """
        query_str = f"do({intervention})|{query_nodes}"
        if query_str in self._counterfactual_cache:
            return self._counterfactual_cache[query_str]

        outcomes = []
        reasoning_path = []
        assumptions = []

        for target in query_nodes[:max_depth]:
            # Step 1: Check if target is in intervention
            if target in intervention:
                outcomes.append((target, 1.0))
                reasoning_path.append(f"{target} is directly set to {intervention[target]}")
                continue

            # Step 2: Compute P(target | do(intervention))
            best_prob = 0.0
            best_path = ""

            for int_var, int_val in intervention.items():
                prob = self.compute_do_probability(target, int_var)
                if prob > best_prob:
                    best_prob = prob
                    best_path = f"do({int_var}) → {target} (P={prob:.3f})"

                # Multi-step causal chains
                for intermediate in self.children.get(int_var, set()):
                    if intermediate == target:
                        continue
                    p1 = self.compute_do_probability(intermediate, int_var)
                    p2 = self.compute_do_probability(target, intermediate)
                    chain_prob = p1 * p2
                    if chain_prob > best_prob:
                        best_prob = chain_prob
                        best_path = f"do({int_var}) → {intermediate} → {target} (P={chain_prob:.3f})"

            if best_path:
                outcomes.append((target, best_prob))
                reasoning_path.append(best_path)
            else:
                outcomes.append((target, 0.5))
                reasoning_path.append(f"No causal path found from intervention to {target}")
                assumptions.append(f"Assuming prior P({target}) = 0.5")

        confidence = np.mean([p for _, p in outcomes]) if outcomes else 0.5

        result = CounterfactualResult(
            query=query_str,
            intervention=intervention,
            predicted_outcomes=outcomes,
            confidence=float(confidence),
            reasoning_path=reasoning_path,
            assumptions=assumptions,
        )

        self._counterfactual_cache[query_str] = result
        return result

    # --- Consolidation validation ---
    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        """
        Проверить, можно ли безопасно объединить узлы A и B.
        Консолидация НЕ безопасна если:
        1. A и B имеют противоположные каузальные эффекты на общий узел C
        2. A и B находятся в отношении cause-effect (объединение теряет направление)
        3. Объединение создаёт каузальный цикл
        """
        result = {
            "safe": True,
            "reasons": [],
            "causal_conflicts": [],
            "recommendation": "proceed",
        }

        # Check 1: Opposing causal effects
        common_targets = set(self.children.get(node_a, set())) & set(self.children.get(node_b, set()))
        for target in common_targets:
            edge_a = self.causal_effects.get((node_a, target))
            edge_b = self.causal_effects.get((node_b, target))
            if edge_a and edge_b:
                diff = abs(edge_a.strength - edge_b.strength)
                if diff > 0.4:
                    result["safe"] = False
                    result["causal_conflicts"].append({
                        "target": target,
                        "effect_a": edge_a.strength,
                        "effect_b": edge_b.strength,
                        "difference": diff,
                    })
                    result["reasons"].append(f"Opposing effects on {target}: {edge_a.strength:.2f} vs {edge_b.strength:.2f}")

        # Check 2: Cause-effect relationship
        if node_b in self.children.get(node_a, set()) or node_a in self.children.get(node_b, set()):
            result["safe"] = False
            result["reasons"].append(f"Causal relationship exists between {node_a} and {node_b}")
            result["recommendation"] = "preserve_separate"

        # Check 3: Would merging create a cycle?
        if node_a in self.ancestors.get(node_b, set()) or node_b in self.ancestors.get(node_a, set()):
            result["safe"] = False
            result["reasons"].append("Merging would create causal cycle")
            result["recommendation"] = "preserve_separate"

        # Check 4: Contradiction records
        for cid, record in self.contradictions.items():
            if record.resolved:
                continue
            causes = [c for c, _ in record.causes]
            if node_a in causes and node_b in causes:
                result["safe"] = False
                result["reasons"].append(f"Both nodes involved in unresolved contradiction: {cid}")
                result["recommendation"] = "resolve_contradiction_first"

        return result

    # --- State management ---
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
                id=record_data["id"],
                effect_node=record_data["effect_node"],
                causes=record_data["causes"],
                timestamp=record_data["timestamp"],
                resolved=record_data["resolved"],
                resolution=record_data["resolution"],
            )

        self._compute_ancestors()


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КОМПОНЕНТЫ (из v4)
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
        self._session_style: deque = deque(maxlen=30)
        self._uncertainty: deque = deque(maxlen=20)
        self._kurtosis_history: deque = deque(maxlen=50)

    def record_response(self, response: float):
        self._response_history.append(response)

    def record_semantic_density(self, density: float):
        self._semantic_density.append(density)

    def record_session_style(self, style_vector: NDArray):
        self._session_style.append(np.mean(style_vector))

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
            nearest_pos = nodes[nearest_id].latent_pos
            old_pos = dead_node.latent_pos.copy()
            dead_node.latent_pos = ((1.0 - self.healing_strength) * old_pos + self.healing_strength * nearest_pos).astype(np.float32)
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


class NeuralODEField:
    def __init__(self, latent_dim: int, noise_level: float = 0.01, time_horizon: float = 1.0, n_steps: int = 20):
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self._weights = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01
        self._bias = np.zeros(latent_dim, dtype=np.float32)

    def _dynamics(self, state: NDArray, t: float, input_signal: Optional[NDArray] = None) -> NDArray:
        n_nodes = len(state) // self.latent_dim
        state_2d = state.reshape(n_nodes, self.latent_dim)
        transformed = state_2d @ self._weights + self._bias
        damping = -0.1 * state_2d
        attraction = 0.0
        if input_signal is not None:
            attraction = 0.05 * (input_signal.reshape(n_nodes, self.latent_dim) - state_2d)
        return (damping + 0.02 * transformed + attraction).flatten()

    def evolve(self, initial_state: NDArray, input_signal: Optional[NDArray] = None, t_span: Optional[NDArray] = None) -> NDArray:
        if t_span is None:
            t_span = np.linspace(0, self.time_horizon, self.n_steps)
        return odeint(lambda s, t: self._dynamics(s, t, input_signal), initial_state.flatten(), t_span, atol=1e-6, rtol=1e-5)

    def evolve_with_noise(self, initial_state: NDArray, input_signal: Optional[NDArray] = None, dt: float = 0.05) -> NDArray:
        n_steps = int(self.time_horizon / dt)
        state = initial_state.flatten().copy()
        trajectory = [state.copy()]
        for _ in range(n_steps):
            state += self._dynamics(state, 0, input_signal) * dt + self.noise_level * np.random.randn(len(state)) * np.sqrt(dt)
            trajectory.append(state.copy())
        return np.array(trajectory)

    def get_state(self) -> Dict:
        return {"weights": self._weights.tolist(), "bias": self._bias.tolist(), "noise_level": self.noise_level}

    def load_state(self, state: Dict):
        self._weights = np.array(state["weights"], dtype=np.float32)
        self._bias = np.array(state["bias"], dtype=np.float32)
        self.noise_level = state.get("noise_level", self.noise_level)


class ProductionMonitor:
    def __init__(self, drift_window: int = 100, drift_threshold: float = 0.05, anomaly_threshold: float = 3.0, metrics_retention: int = 10000):
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        self._embedding_history: deque = deque(maxlen=drift_window)
        self._response_history: deque = deque(maxlen=metrics_retention)
        self._latency_history: deque = deque(maxlen=metrics_retention)
        self._gate_history: deque = deque(maxlen=metrics_retention)
        self._ab_results: Dict[str, Dict] = {}
        self._anomaly_log: List[Dict] = []
        self._is_drifting = False
        self._last_drift_score = 0.0

    def record_embedding(self, embedding: NDArray):
        self._embedding_history.append(embedding.copy())

    def record_response(self, response: float, latency_ms: float, n_consolidations: int = 0, avg_gate: float = 1.0):
        self._response_history.append(response)
        self._latency_history.append(latency_ms)
        self._gate_history.append(avg_gate)

    def detect_drift(self) -> Dict[str, Any]:
        if len(self._embedding_history) < self.drift_window // 2:
            return {"drifting": False, "score": 0.0}
        embeddings = np.array(list(self._embedding_history))
        half = len(embeddings) // 2
        max_score = 0.0
        for d in range(min(embeddings.shape[1], 20)):
            stat, _ = scipy_stats.ks_2samp(embeddings[:half, d], embeddings[half:, d])
            max_score = max(max_score, stat)
        self._last_drift_score = max_score
        self._is_drifting = max_score > self.drift_threshold
        return {"drifting": self._is_drifting, "score": max_score}

    def detect_anomaly(self, current_metric: float) -> bool:
        if len(self._response_history) < 30:
            return False
        values = np.array(list(self._response_history))
        mean, std = np.mean(values), np.std(values)
        if std < 1e-8:
            return False
        z = abs(current_metric - mean) / std
        is_anomaly = z > self.anomaly_threshold
        if is_anomaly:
            self._anomaly_log.append({"timestamp": time.time(), "value": current_metric, "z_score": z})
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
        return {v: {"mean": np.mean(vals), "std": np.std(vals), "n": len(vals)}
                for v, m in self._ab_results.items() for metric, vals in [(metric_name, m.get(metric_name, []))] if vals}

    def get_dashboard(self) -> Dict[str, Any]:
        return {
            "drift": {"is_drifting": self._is_drifting, "drift_score": self._last_drift_score},
            "response": {"mean": float(np.mean(self._response_history)) if self._response_history else 0,
                         "std": float(np.std(self._response_history)) if self._response_history else 0},
            "latency_ms": {"mean": float(np.mean(self._latency_history)) if self._latency_history else 0,
                           "p95": float(np.percentile(self._latency_history, 95)) if self._latency_history else 0},
            "gate_distribution": {"mean": float(np.mean(self._gate_history)) if self._gate_history else 1.0,
                                  "std": float(np.std(self._gate_history)) if self._gate_history else 0},
            "anomalies": len(self._anomaly_log), "ab_variants": list(self._ab_results.keys()),
        }

    def get_state(self) -> Dict:
        return {"ab_results": self._ab_results, "anomaly_log": self._anomaly_log[-100:], "is_drifting": self._is_drifting}

    def load_state(self, state: Dict):
        self._ab_results = state.get("ab_results", {})
        self._anomaly_log = state.get("anomaly_log", [])
        self._is_drifting = state.get("is_drifting", False)


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
# ЯДРО: RTMDKField v5
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

        self.neural_ode: Optional[NeuralODEField] = None
        if config.continuous_dynamics:
            self.neural_ode = NeuralODEField(config.latent_dim, config.sde_noise_level, config.ode_time_horizon, config.ode_n_steps)

        self.monitor: Optional[ProductionMonitor] = None
        if config.production_mode:
            self.monitor = ProductionMonitor(config.drift_window, config.drift_threshold, config.anomaly_threshold, config.metrics_retention)

        self.meta_kernel: Optional[MetaAdaptiveKernel] = None
        if config.meta_adaptive:
            self.meta_kernel = MetaAdaptiveKernel(config.bandwidth, config.phase_coupling, config.meta_adaptation_lr,
                                                  config.kurtosis_target_min, config.kurtosis_target_max)

        self.healer: Optional[TopologyHealer] = None
        if config.self_healing:
            self.healer = TopologyHealer(config.dead_zone_threshold, config.hyperconvergence_threshold,
                                        config.fragmentation_threshold, config.healing_strength, config.max_healing_nodes_per_step)

        # ФАЗА 6: Каузально-топологическая память
        self.causal_engine: Optional[CausalInferenceEngine] = None
        if config.causal_topological:
            self.causal_engine = CausalInferenceEngine(
                min_samples=config.causal_discovery_min_samples,
                p_threshold=config.causal_p_threshold,
                adjustment_sets_enabled=config.causal_adjustment_sets)

        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0,
            "projection_updates": 0, "self_sup_checks": 0, "tda_checks": 0,
            "bm25_fallbacks": 0, "adaptive_threshold_value": config.tension_threshold,
            "false_merges": 0, "field_stability": 1.0,
            "differentiable_loss": 0.0, "ode_steps": 0,
            "drift_detected": False, "anomalies": 0,
            "meta_kurtosis": 3.0, "meta_bandwidth": config.bandwidth,
            "meta_phase_coupling": config.phase_coupling,
            "field_health": "stable", "healing_events": 0, "healing_history": [],
            # Фаза 6
            "causal_edges": 0, "contradictions": 0, "counterfactual_queries": 0,
            "consolidation_validations": 0, "blocked_consolidations": 0,
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

        # Фаза 6: causal boost
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
            # Record observation for causal discovery
            active = [nid for nid, resp, _ in results if resp > self.cfg.min_response * 0.5]
            if active:
                self.causal_engine.record_observation(active)
                self._active_node_history.append(active)

        if self.monitor:
            latency_ms = (time.time() - t0) * 1000
            self.monitor.record_embedding(embedding)
            avg_gate = np.mean([n.soft_gate for _, _, n in results]) if results else 1.0
            self.monitor.record_response(results[0][1] if results else 0, latency_ms, 0, avg_gate)

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

            # Фаза 6: Do-calculus validation перед консолидацией
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

            # Фаза 6: Transfer causal knowledge
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

        # Фаза 6: Causal discovery
        if self.causal_engine and self._step_counter % max(self.cfg.causal_discovery_freq, 1) == 0:
            self.causal_engine.discover_causal_structure()
            # Update node causal attributes
            for (cause, effect), edge in self.causal_engine.causal_effects.items():
                if effect in self.nodes:
                    self.nodes[effect].causal_parents.append(cause)
                    self.nodes[effect].causal_strength[cause] = edge.strength
                if cause in self.nodes:
                    self.nodes[cause].causal_effects[effect] = edge.strength

            self.stats["causal_edges"] = len(self.causal_engine.causal_effects)

            # Detect contradictions
            if self.cfg.contradiction_detection:
                contradictions = self.causal_engine.detect_contradictions(self.cfg.contradiction_threshold)
                self.stats["contradictions"] = len(self.causal_engine.contradictions)

        if self.learnable_kernel:
            self.learnable_kernel.step()

        if self.meta_kernel:
            self.meta_kernel.adapt()
            self.stats["meta_kurtosis"] = self.meta_kernel.compute_resonance_kurtosis()
            self.stats["meta_bandwidth"] = self.meta_kernel.get_bandwidth()
            self.stats["meta_phase_coupling"] = self.meta_kernel.get_phase_coupling()

        if self.monitor:
            drift = self.monitor.detect_drift()
            self.stats["drift_detected"] = drift["drifting"]
            if drift["drifting"] and self.cfg.auto_rollback_on_anomaly:
                self.stats["anomalies"] += 1

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

    def evolve_continuous(self, inputs: Optional[List[Dict]] = None, use_sde: bool = False) -> NDArray:
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
        final_state = trajectory[-1].reshape(len(self.nodes), self.cfg.latent_dim)
        for i, nid in enumerate(self.node_index):
            if i < len(final_state):
                self.nodes[nid].latent_pos = final_state[i].astype(np.float32)
        return trajectory

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

    # --- Фаза 6: Counterfactual query ---
    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        """Ответить на "Что если?" через каузальный движок."""
        if not self.causal_engine:
            return CounterfactualResult(
                query=str(intervention), intervention=intervention,
                predicted_outcomes=[], confidence=0.0,
                reasoning_path=["Causal engine not enabled"],
                assumptions=["Enable causal_topological=True in config"])

        self.stats["counterfactual_queries"] += 1
        return self.causal_engine.counterfactual_query(
            intervention, query_nodes, evidence, self.cfg.counterfactual_max_depth)

    def get_causal_summary(self) -> Dict:
        """Получить сводку каузальной структуры поля."""
        if not self.causal_engine:
            return {"enabled": False}
        return {
            "enabled": True,
            "causal_edges": len(self.causal_engine.causal_effects),
            "contradictions": len([c for c in self.causal_engine.contradictions.values() if not c.resolved]),
            "nodes_with_effects": len(set(k[0] for k in self.causal_engine.causal_effects)),
            "nodes_affected": len(set(k[1] for k in self.causal_engine.causal_effects)),
            "top_effects": sorted(
                [(f"{k[0]}→{k[1]}", v.strength) for k, v in self.causal_engine.causal_effects.items()],
                key=lambda x: x[1], reverse=True)[:10],
        }

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
        if self.monitor:
            return self.monitor.get_dashboard()
        return {}

    def record_ab_metric(self, metric_name: str, value: float):
        if self.monitor:
            self.monitor.record_ab_result(self.cfg.ab_variant, metric_name, value)

    def get_field_health(self) -> Dict:
        if self.healer:
            health, diagnostics = self.healer.compute_field_health(self.nodes)
            diagnostics["kurtosis"] = self.stats.get("meta_kurtosis", 3.0)
            return diagnostics
        return {"health": "unknown", "kurtosis": 3.0}


# ============================================================================
# RTMDKMemory v5
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
                "local_density": node.local_density}
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
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

    # Фаза 6: Causal API
    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        """Что если? do(intervention) → outcomes."""
        return self.field.counterfactual_query(intervention, query_nodes, evidence)

    def get_causal_summary(self) -> Dict:
        """Сводка каузальной структуры."""
        return self.field.get_causal_summary()

    def get_contradictions(self) -> List[ContradictionRecord]:
        """Получить все каузальные противоречия."""
        if self.field.causal_engine:
            return list(self.field.causal_engine.contradictions.values())
        return []

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> bool:
        """Разрешить противоречие."""
        if self.field.causal_engine and contradiction_id in self.field.causal_engine.contradictions:
            self.field.causal_engine.contradictions[contradiction_id].resolved = True
            self.field.causal_engine.contradictions[contradiction_id].resolution = resolution
            return True
        return False

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        """Проверить безопасность консолидации через do-calculus."""
        if self.field.causal_engine:
            return self.field.causal_engine.validate_consolidation(node_a, node_b)
        return {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}

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
        data = {"config": cd, "nodes": [n.to_dict() for n in self.field.nodes.values()], "stats": self.field.stats}
        if self.field.projection_learner:
            data["projection_state"] = self.field.projection_learner.get_state()
        else:
            data["projection"] = self.field._raw_projection.tolist()
        if self.field.learnable_kernel:
            data["learnable_kernel"] = self.field.learnable_kernel.get_state()
        if self.field.neural_ode:
            data["neural_ode"] = self.field.neural_ode.get_state()
        if self.field.tda_monitor:
            data["tda_history"] = self.field.tda_monitor.history
        if self.field.monitor:
            data["monitor"] = self.field.monitor.get_state()
        if self.field.meta_kernel:
            data["meta_kernel"] = self.field.meta_kernel.get_state()
        if self.field.healer:
            data["healer"] = self.field.healer.get_state()
        if self.field.causal_engine:
            data["causal_engine"] = self.field.causal_engine.get_state()
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
        if config.tda_monitoring and "tda_history" in data:
            memory.field.tda_monitor.history = data["tda_history"]
        if config.production_mode and "monitor" in data:
            memory.field.monitor.load_state(data["monitor"])
        if config.meta_adaptive and "meta_kernel" in data:
            memory.field.meta_kernel.load_state(data["meta_kernel"])
        if config.self_healing and "healer" in data:
            memory.field.healer.load_state(data["healer"])
        if config.causal_topological and "causal_engine" in data:
            memory.field.causal_engine.load_state(data["causal_engine"])

        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory
