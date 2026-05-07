"""
rtmdk/nodes.py
Data classes for RTMDK memory nodes and related structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
import time
import numpy as np
from numpy.typing import NDArray


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
    causal_effects: Dict[str, Any] = field(default_factory=dict)
    do_interventions: Dict[str, NDArray] = field(default_factory=dict)
    is_causal_root: bool = False
    causal_context: Dict[str, Any] = field(default_factory=dict)
    velocity: Optional[NDArray[np.float32]] = None
    acceleration: Optional[NDArray[np.float32]] = None
    goal_tags: List[str] = field(default_factory=list)
    tool_usage_count: int = 0
    modal_embedding: Optional[NDArray[np.float32]] = None
    cross_modal_score: float = 0.0
    latent_scale: float = 1.0
    latent_zero_point: float = 0.0
    tier: str = "semantic"
    role: str = "default"  # Phase 17: Role shard assignment
    goal_relevance: float = 0.0
    rl_reward: float = 0.0

    # Phase 20: Domain Memory & Concept Lifecycle
    domain: str = "general"
    subdomain: str = ""
    topic: str = ""
    state: str = "stable"  # stable | weakened | disputed | deprecated
    confidence: float = 1.0
    revision_count: int = 0
    conflict_with: List[str] = field(default_factory=list)
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    evidence_spans: List[Dict] = field(default_factory=list)
    fact_state: str = "active"  # active | expired | superseded
    superseded_by: Optional[str] = None
    covariance: Optional[NDArray[np.float32]
                         ] = None  # P2.2: Kalman uncertainty

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
        if self.modal_embedding is not None:
            d["modal_embedding"] = self.modal_embedding.tolist()
        if self.covariance is not None:
            d["covariance"] = self.covariance.tolist()
        for k, v in self.do_interventions.items():
            if isinstance(v, np.ndarray):
                d["do_interventions"][k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryNode":
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        if data.get("pre_consolidation_pos"):
            data["pre_consolidation_pos"] = np.array(
                data["pre_consolidation_pos"], dtype=np.float32)
        if data.get("gradient_cache"):
            data["gradient_cache"] = np.array(
                data["gradient_cache"], dtype=np.float32)
        if data.get("velocity"):
            data["velocity"] = np.array(data["velocity"], dtype=np.float32)
        if data.get("acceleration"):
            data["acceleration"] = np.array(
                data["acceleration"], dtype=np.float32)
        if data.get("modal_embedding"):
            data["modal_embedding"] = np.array(
                data["modal_embedding"], dtype=np.float32)
        if data.get("covariance"):
            data["covariance"] = np.array(data["covariance"], dtype=np.float32)
        for k, v in data.get("do_interventions", {}).items():
            if isinstance(v, list):
                data["do_interventions"][k] = np.array(v, dtype=np.float32)
        return cls(**data)


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
    def from_dict(cls, data: Dict) -> "CausalEdge":
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
            "id": self.id,
            "effect_node": self.effect_node,
            "causes": self.causes,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "contradiction_reason": self.contradiction_reason,
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
        return {"query": self.query,
                "intervention": self.intervention,
                "predicted_outcomes": [{"node": n,
                                        "probability": p} for n,
                                       p in self.predicted_outcomes],
                "confidence": self.confidence,
                "reasoning_path": self.reasoning_path,
                "assumptions": self.assumptions,
                }


@dataclass
class AgentPlan:
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
class FederatedNode:
    node_id: str
    phase: float
    natural_freq: float = 1.0
    amplitude: float = 1.0
    last_sync_time: float = field(default_factory=time.time)
    params: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    sync_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "FederatedNode":
        return cls(**data)


@dataclass
class GoalNode:
    id: str
    description: str
    subgoals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    completion: float = 0.0
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: str = "active"
    related_nodes: List[str] = field(default_factory=list)
    intent_signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "GoalNode":
        return cls(**data)
