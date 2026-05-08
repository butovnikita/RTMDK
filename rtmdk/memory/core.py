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
MetaAdaptiveKernel, TopologyHealer, CausalInferenceEngine, NeuralODEDynamics,
IncPCAProjection, BM25Index, HNSWIndex, TorchBackend, LearnableKernel,
DifferentiableConsolidation, AgentPlanner, HypothesisVerifier, ToolRouter,
ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager, MetaController,
KuramotoSync, FederatedRTMDK, FederatedNode, detect_modality, cross_modal_resonance
"""

from __future__ import annotations
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import (
    ContextFormat, RTMDKConfig,
)
from rtmdk.nodes import (
    MemoryNode, ContradictionRecord, CounterfactualResult, AgentPlan,
    ToolCall, Hypothesis, EvalResult,
)
import asyncio
import functools
import json
import re
import time
import os
import copy
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

# Extracted engine classes (kept in sync with rtmdk/support/ modules)
try:
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False
from rtmdk.memory.utils import SecurityViolationError, detect_modality

logger = logging.getLogger(__name__)

# Phase 5: dataclass nodes extracted to rtmdk.nodes

# Phase 15: New modules
try:
    VC_AVAILABLE = True
except ImportError:
    VC_AVAILABLE = False

try:
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False

try:
    from rtmdk.support.triton_backend import GPUBackend, TritonBackend, TRITON_AVAILABLE
except ImportError:
    GPUBackend = None  # type: ignore
    TritonBackend = None  # type: ignore
    TRITON_AVAILABLE = False

# Phase 16: New modules
try:
    SYMBOLIC_AVAILABLE = True
except ImportError:
    SYMBOLIC_AVAILABLE = False

try:
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

try:
    from rtmdk.support.ump import UniversalMemoryProtocol
    UMP_AVAILABLE = True
except ImportError:
    UMP_AVAILABLE = False

# Phase 17: RoleShardRouter
try:
    from rtmdk.support.role_shard_router import DEFAULT_ROLE
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
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
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


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def _sanitize_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal attacks.

    Rejects paths containing '..' (path traversal).
    Returns normalized path.
    """
    import os
    # Reject parent directory references BEFORE normalization
    # (normpath collapses 'a/../b' to 'b', which would hide the attack)
    if ".." in path.replace("\\", "/").split("/"):
        raise SecurityViolationError(f"Path traversal detected: {path}")
    # Normalize to catch unicode tricks and mixed separators
    normalized = os.path.normpath(path)
    return normalized


def _safe_json_load(path: str) -> Dict:
    """Load JSON with size limit to prevent memory exhaustion."""
    file_size = os.path.getsize(path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB)")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if len(raw.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            "File exceeds maximum allowed size after encoding check")
    return json.loads(raw)


# ============================================================================
# CONFIGURATION v7 — extracted to rtmdk/memory/config.py (P0 refactor)
# ============================================================================
# Extracted engine classes (P0 refactor � imported from canonical modules)


# ============================================================================
# PHASE 11 TRACK 1: MEMORY STRATIFICATION
# ============================================================================

def _enum_value(val, default):
    """Safely extract enum value for serialization."""
    return val.value if isinstance(
        val, Enum) else (
        val if val is not None else default)


def detect_tier(text: str, context: Optional[Dict] = None) -> str:
    """Auto-detect memory tier from content."""
    context = context or {}
    text_lower = text.lower()
    # Procedural: how-to, tool usage
    if context.get("tool_used"):
        return "procedural"
    if any(
        p in text_lower for p in [
            "how to",
            "how do",
            "how can",
            "steps to",
            "tutorial",
            "guide"]):
        return "procedural"
    # Episodic: dates, temporal markers
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text):
        return "episodic"
    if any(
        p in text_lower for p in [
            "yesterday",
            "last week",
            "last month",
            "ago",
            "вчера",
            "на прошлой",
            "неделю назад"]):
        return "episodic"
    return "semantic"


# ============================================================================
# PHASE 11 TRACK 4: COUNTERFACTUAL IMAGINATION
# ============================================================================

# ============================================================================
# PHASE 13 TRACK 1: TELEOLOGICAL LAYER (Goal/Intent Tracking)
# ============================================================================
# ============================================================================
# PHASE 13 TRACK 2: COGNITIVE ATTENTION BIAS
# ============================================================================

def apply_attention_bias(results: List[Tuple[str, float, MemoryNode]],
                         temperature: float = 1.0) -> List[Tuple[str, float, MemoryNode]]:
    """
    Transform raw resonance scores into attention-biased scores.
    Incorporates causal_strength, tension, salience as structural signals.
    """
    if not results:
        return results

    # Extract raw scores
    raw_scores = np.array([r for _, r, _ in results])
    if len(raw_scores) < 2:
        return results

    # Compute attention weights
    weights = []
    for nid, resp, node in results:
        # Base resonance
        score = resp
        # Causal boost
        causal_boost = sum(
            node.causal_strength.values()) if hasattr(
            node, 'causal_strength') else 0
        score *= (1.0 + 0.2 * min(1.0, causal_boost))
        # Tension penalty (high tension = less reliable)
        score *= max(0.5, 1.0 - node.tension)
        # Goal relevance boost (Phase 13 Track 1)
        goal_rel = getattr(node, 'goal_relevance', 0.0)
        score *= (1.0 + 0.3 * goal_rel)
        weights.append(score)

    weights = np.array(weights)
    # Softmax with temperature
    if temperature > 0:
        exp_weights = np.exp(weights / temperature)
        normalized = exp_weights / (exp_weights.sum() + 1e-8)
    else:
        normalized = weights / (weights.sum() + 1e-8)

    # Re-rank by attention-biased scores
    biased_results = []
    for i, (nid, resp, node) in enumerate(results):
        biased_results.append((nid, float(normalized[i]), node))

    biased_results.sort(key=lambda x: x[1], reverse=True)
    return biased_results


def format_cognitive_context(results: List[Tuple[str, float, MemoryNode]],
                             bias_applied: bool = False) -> str:
    """Format memory results with structural attention signals for LLM.

    Handles both structured nodes (v2: input_text, output_text, emotion, tags)
    and legacy nodes (v1: text).
    """
    if not results:
        return "### COGNITIVE_CONTEXT\nNo relevant structures."

    lines = ["### COGNITIVE_CONTEXT"]
    for nid, score, node in results:
        content = node.content

        # Check for structured node (v2)
        if content.get("version") == "2.0":
            input_text = content.get("input_text", "")
            output_text = content.get("output_text", "")
            emotion = content.get("emotion", "neutral")
            tags = content.get("tags", [])
            session = content.get("session", "")

            # Format structured context
            text_parts = []
            if input_text:
                text_parts.append(f"User: {input_text[:80]}")
            if output_text:
                text_parts.append(f"AI: {output_text[:80]}")
            text = " | ".join(text_parts) if text_parts else content.get(
                "text", "unknown")[:80]

            tier = content.get("tier", getattr(node, 'tier', 'semantic'))
            tokens = f"[SCORE:{score:.3f}]"
            tokens += f"[TIER:{tier[0].upper()}]"
            if emotion != "neutral":
                tokens += f"[EMO:{emotion[:4]}]"
            if tags:
                tokens += f"[TAGS:{','.join(tags[:3])}]"
            if session:
                tokens += f"[SESS:{session[:10]}]"
        else:
            # Legacy node (v1)
            text = content.get("text", "unknown")[:80]
            tier = content.get("tier", getattr(node, 'tier', 'semantic'))
            tokens = f"[SCORE:{score:.3f}]"
            tokens += f"[TIER:{tier[0].upper()}]"

        causal = len(
            node.causal_strength) if hasattr(
            node, 'causal_strength') else 0
        tension = node.tension
        lineage = len(node.lineage) if node.lineage else 0

        if causal > 0:
            tokens += f"[CAUSAL:{causal}]"
        if tension > 0.3:
            tokens += f"[TENSION:{tension:.2f}]"
        if lineage > 0:
            tokens += f"[LINEAGE:{lineage}]"

        lines.append(f"{tokens} {text}")

    return "\n".join(lines)


# ============================================================================
# PHASE 13 TRACK 3: CLOSED-LOOP RL FROM LLM FEEDBACK
# ============================================================================

# ============================================================================
# PHASE 13 TRACK 4: EVENT-DRIVEN + LOW-RANK COMPRESSION
# ============================================================================

# ============================================================================
# PHASE 14 TRACK 1: INTROSPECTIVE META-MEMORY
# ============================================================================

# ============================================================================
# PHASE 14 TRACK 2: FORMAL SECURITY
# ============================================================================

# ============================================================================
# PHASE 14 TRACK 5: SWARM MEMORY
# ============================================================================

# ============================================================================
# SUPPORTING COMPONENTS
# ============================================================================


# ============================================================================
# CONTEXT FORMATTING
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
    ContextFormat.ATTENTION: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories with attention-weighted tokens. "
        "Each memory starts with tokens like [ATTN:x.xxxx][SAL:x.xxxx][TIER:X].\n"
        "- ATTN: attention weight — how relevant this memory is to the current query (higher = more relevant)\n"
        "- SAL: salience — overall importance in the memory field\n"
        "- TIER: memory tier (E=episodic, S=semantic, P=procedural)\n"
        "- CAUSAL: number of causal connections (if present)\n"
        "- GOAL: goal relevance score (if present)\n"
        "Use the ATTN weights to focus your attention on the most relevant memories.\n\n"
        "Relevant memories:\n{context}"
    ),
}


def format_context(
        results: List[Tuple[str, float, MemoryNode]], fmt: ContextFormat) -> str:
    if fmt == ContextFormat.JSON:
        items = []
        for nid, resp, node in results:
            content = node.content

            # Check for structured node (v2)
            if content.get("version") == "2.0":
                item = {
                    "resonance": round(resp, 4),
                    "salience": round(node.salience, 4),
                    "input_text": content.get("input_text", ""),
                    "output_text": content.get("output_text", ""),
                    "role": content.get("role", ""),
                    "session": content.get("session", ""),
                    "emotion": content.get("emotion", ""),
                    "tags": content.get("tags", []),
                    "tier": content.get("tier", ""),
                    "timestamp": content.get("timestamp", 0),
                    "lineage": node.lineage,
                    "modality": node.modality,
                }
            else:
                # Legacy node (v1)
                item = {
                    "resonance": round(resp, 4),
                    "salience": round(node.salience, 4),
                    "text": content.get("text", ""),
                    "lineage": node.lineage,
                    "modality": node.modality,
                    "self_sup_score": round(node.self_sup_score, 4),
                    "cross_modal_score": round(node.cross_modal_score, 4),
                }
                meta = {k: v for k, v in content.items() if k != "text"}
                if meta:
                    item["metadata"] = meta
            items.append(item)
        return json.dumps(
            items,
            ensure_ascii=False,
            indent=2) if items else "[]"

    elif fmt == ContextFormat.YAML:
        lines = []
        for nid, resp, node in results:
            content = node.content
            if content.get("version") == "2.0":
                lines.extend([
                    f"- resonance: {resp:.4f}",
                    f"  salience: {node.salience:.4f}",
                    "  input: \"{content.get('input_text', '')}\"",
                    "  output: \"{content.get('output_text', '')}\"",
                    f"  role: {content.get('role', '')}",
                    f"  emotion: {content.get('emotion', '')}",
                    f"  tier: {content.get('tier', '')}",
                ])
            else:
                lines.extend([
                    f"- resonance: {resp:.4f}",
                    f"  salience: {node.salience:.4f}",
                    "  text: \"{content.get('text', '')}\"",
                    f"  lineage: {node.lineage}",
                    f"  modality: {node.modality}",
                    f"  cross_modal_score: {node.cross_modal_score:.4f}",
                ])
        return "\n".join(lines) if lines else "No relevant memory."

    elif fmt == ContextFormat.ATTENTION:
        lines = ["### ATTENTION_CONTEXT"]
        for nid, resp, node in results:
            content = node.content
            causal = len(
                node.causal_strength) if hasattr(
                node, 'causal_strength') else 0
            goal_rel = getattr(node, 'goal_relevance', 0.0)
            tokens = (
                f"[ATTN:{resp:.3f}][SAL:{node.salience:.3f}]"
                f"[TIER:{content.get('tier', getattr(node, 'tier', 'semantic'))[0].upper()}]")
            # Phase 20: Domain & State tokens
            domain = getattr(node, 'domain', 'general')
            if domain and domain != 'general':
                tokens += f"[DOM:{domain.upper()[:3]}]"
            state = getattr(node, 'state', '')
            if state and state != 'stable':
                tokens += f"[STATE:{state[0].upper()}]"
            if causal > 0:
                tokens += f"[CAUSAL:{causal}]"
            if goal_rel > 0.3:
                tokens += f"[GOAL:{goal_rel:.2f}]"

            # Extract text from structured or legacy node
            if content.get("version") == "2.0":
                input_t = content.get("input_text", "")[:60]
                output_t = content.get("output_text", "")[:60]
                if input_t and output_t:
                    text = f"U:{input_t} | AI:{output_t}"
                elif input_t:
                    text = f"U:{input_t}"
                elif output_t:
                    text = f"AI:{output_t}"
                else:
                    text = content.get("text", "unknown")[:100]
                # Add emotion/tag if present
                emotion = content.get("emotion", "")
                tags = content.get("tags", [])
                if emotion != "neutral":
                    text += f" [{emotion}]"
                if tags:
                    text += f" #{','.join(tags[:2])}"
            else:
                text = node.content.get("text", "unknown")[:100]

            lines.append(f"{tokens} {text}")
        return "\n".join(lines) if len(lines) > 1 else "No relevant memory."
    else:
        parts = []
        for _, r, n in results:
            content = n.content
            if content.get("version") == "2.0":
                input_t = content.get("input_text", "")[:50]
                output_t = content.get("output_text", "")[:50]
                text = f"U:{input_t} | AI:{output_t}" if input_t and output_t else (
                    input_t or output_t or "unknown")
            else:
                text = n.content.get('text', '')
            parts.append(
                f"[R:{r:.2f}|S:{n.salience:.2f}|CM:{n.cross_modal_score:.2f}] {text}")
        return "\n".join(parts) if parts else "No relevant memory."


def build_system_prompt(
        context: str,
        fmt: ContextFormat,
        use_structured: bool) -> str:
    if not use_structured or not context or context in (
            "No relevant memory.", "[]"):
        return "You are a helpful assistant with long-term memory."
    return SYSTEM_PROMPT_TEMPLATES.get(
        fmt, SYSTEM_PROMPT_TEMPLATES[ContextFormat.PLAIN]).format(context=context)


# ============================================================================
# CORE: RTmdKField v7
# ============================================================================

def _locked(method):
    """Decorator that wraps method in self._write_lock RLock."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)
    return wrapper


def _copy_node(node):
    """Shallow copy of a MemoryNode with copied mutable fields."""
    n = copy.copy(node)
    n.latent_pos = node.latent_pos.copy()
    n.lineage = list(node.lineage)
    n.content = dict(node.content)
    n.causal_strength = dict(node.causal_strength)
    n.causal_parents = list(node.causal_parents)
    n.conflict_with = list(node.conflict_with)
    if node.pre_consolidation_pos is not None:
        n.pre_consolidation_pos = node.pre_consolidation_pos.copy()
    if node.gradient_cache is not None:
        n.gradient_cache = node.gradient_cache.copy()
    if node.velocity is not None:
        n.velocity = node.velocity.copy()
    if node.acceleration is not None:
        n.acceleration = node.acceleration.copy()
    if node.modal_embedding is not None:
        n.modal_embedding = node.modal_embedding.copy()
    if node.covariance is not None:
        n.covariance = node.covariance.copy()
    n.do_interventions = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in node.do_interventions.items()}
    return n


class RTMDKMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    config: RTMDKConfig = Field(default_factory=RTMDKConfig)
    embedder: Callable[[str], NDArray[np.float32]]
    field: Optional[RTMDKField] = Field(default=None, exclude=True)
    session_phases: Dict[str, float] = Field(default_factory=dict)
    wal_path: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _init_field(cls, data):
        if isinstance(data, dict) and data.get("field") is None:
            cfg = data.get("config", RTMDKConfig())
            wal = data.get("wal_path")
            data = dict(data)
            data["field"] = RTMDKField(cfg, wal_path=wal)
        return data

    def model_post_init(self, __context):
        if self.field is None:
            object.__setattr__(
                self, "field", RTMDKField(
                    self.config, wal_path=self.wal_path))
        # Track 5: Replay WAL mutations for durability
        self._replay_wal()
        # Fix 4: Auto-start async workers if async_pipeline is enabled
        if self.config.async_pipeline and not self.field._workers_started:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.field._start_workers())
            except RuntimeError:
                pass
        # Phase 18: Initialize Engram Manager
        if self.config.enable_engrams:
            try:
                from rtmdk.engrams import EngramManager
                object.__setattr__(self, "engram_manager", EngramManager(
                    min_nodes=self.config.engram_min_nodes,
                    max_nodes=self.config.engram_max_nodes,
                    creation_threshold=self.config.engram_creation_threshold,
                    decay_rate=self.config.engram_decay_rate,
                    pattern_completion=self.config.engram_pattern_completion,
                    overlap_threshold=self.config.engram_overlap_threshold,
                ))
            except Exception:
                logger.warning(
                    "Engram manager initialization failed, disabling",
                    exc_info=True)
                object.__setattr__(self, "engram_manager", None)
        else:
            object.__setattr__(self, "engram_manager", None)

        # v8.2.1 production distributed features
        self._init_vector_storage()
        self._init_replication_manager()

    def _init_vector_storage(self) -> None:
        dsn = self.config.vector_storage_dsn
        if dsn:
            try:
                from rtmdk.production.vector_storage import VectorStorage
                vs = VectorStorage.create(dsn, dim=self.config.latent_dim)
                object.__setattr__(self, "vector_storage", vs)
                logger.info("VectorStorage backend: %s (dsn=%s)", type(vs).__name__, dsn)
            except Exception:
                logger.warning("VectorStorage init failed, disabling", exc_info=True)
                object.__setattr__(self, "vector_storage", None)
        else:
            object.__setattr__(self, "vector_storage", None)

    def _init_replication_manager(self) -> None:
        peers = self.config.replication_peers
        if peers:
            try:
                from rtmdk.production.replication import ReplicationManager
                rm = ReplicationManager(
                    peers=peers,
                    node_id=self.config.replication_node_id,
                    wal_path=self.config.replication_wal_path,
                )
                object.__setattr__(self, "replication_manager", rm)
                logger.info("ReplicationManager enabled with peers: %s", peers)
            except Exception:
                logger.warning("ReplicationManager init failed, disabling", exc_info=True)
                object.__setattr__(self, "replication_manager", None)
        else:
            object.__setattr__(self, "replication_manager", None)

    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]

    def add_node(self, embedding: NDArray, content: Dict, **kwargs) -> str:
        """Add a node to the memory field. Delegates to RTMDKField.add_node."""
        node_id = self.field.add_node(embedding, content, **kwargs)
        # v8.2.1 hooks
        self._on_node_added(node_id, embedding, content, kwargs)
        return node_id

    def _on_node_added(
        self, node_id: str, embedding: NDArray, content: Dict, add_kwargs: Dict
    ) -> None:
        vs = getattr(self, "vector_storage", None)
        if vs is not None:
            try:
                vs.insert(node_id, embedding, {"content": content})
            except Exception:
                logger.warning("VectorStorage insert failed for %s", node_id, exc_info=True)
        rm = getattr(self, "replication_manager", None)
        if rm is not None and rm.enabled:
            try:
                rm.replicate({
                    "op": "add_node",
                    "node_id": node_id,
                    "embedding": embedding.tolist(),
                    "content": content,
                    "kwargs": {k: v for k, v in add_kwargs.items() if k not in {"embedding", "content"}},
                })
            except Exception:
                logger.warning("Replication failed for %s", node_id, exc_info=True)

    def add_nodes_batch(
        self,
        embeddings: NDArray,
        contents: List[Dict],
        phases: Optional[NDArray] = None,
        node_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        skip_projection: bool = False,
    ) -> List[str]:
        """Batch add nodes. Delegates to RTMDKField.add_nodes_batch."""
        result = self.field.add_nodes_batch(
            embeddings,
            contents,
            phases,
            node_ids,
            session_ids,
            modalities,
            skip_projection)
        # v8.2.1 hooks
        vs = getattr(self, "vector_storage", None)
        rm = getattr(self, "replication_manager", None)
        for i, nid in enumerate(result):
            emb = embeddings[i]
            content = contents[i]
            if vs is not None:
                try:
                    vs.insert(nid, emb, {"content": content})
                except Exception:
                    logger.warning("VectorStorage insert failed for %s", nid, exc_info=True)
            if rm is not None and rm.enabled:
                try:
                    rm.replicate({
                        "op": "add_node",
                        "node_id": nid,
                        "embedding": emb.tolist(),
                        "content": content,
                    })
                except Exception:
                    logger.warning("Replication failed for %s", nid, exc_info=True)
        return result

    def _replay_wal(self) -> None:
        """Replay WAL mutations to recover durability after restart.

        Reads all records from WAL and re-applies them:
        - add_node / add_nodes_batch: re-embed text and insert
        - delete: remove nodes
        - consolidate: skipped (snapshot already contains consolidated state)
        """
        if not self.field.wal.enabled:
            return
        records = self.field.wal.replay()
        if not records:
            return
        # Disable WAL during replay to avoid writing replayed ops back to WAL
        was_enabled = self.field.wal.enabled
        self.field.wal.enabled = False
        logger.info(f"WAL replay: {len(records)} records")
        replayed = 0
        try:
            for rec in records:
                op = rec.get("op")
                payload = rec.get("payload", {})
                try:
                    if op == "add_node":
                        content = payload.get("content", {})
                        modality = payload.get("modality", "text")
                        node_id = payload.get("node_id")
                        # Use stored embedding if available (Track 5), else
                        # re-embed
                        emb_list = payload.get("embedding")
                        if emb_list is not None:
                            embedding = np.array(emb_list, dtype=np.float32)
                        else:
                            text = content.get("text", "")
                            if not text:
                                text = content.get("input_text", "")
                            if not text:
                                logger.warning(
                                    f"WAL replay add_node: no text for {node_id}")
                                continue
                            embedding = self.embedder(text)
                        phase = self._get_phase(
                            payload.get("session_id"), embedding)
                        self.field.add_node(
                            embedding, content, phase=phase,
                            node_id=node_id, modality=modality,
                        )
                        replayed += 1
                    elif op == "add_nodes_batch":
                        contents = payload.get("contents", [])
                        modalities = payload.get("modalities")
                        node_ids = payload.get("node_ids")
                        embeddings_list = payload.get("embeddings")
                        if embeddings_list is not None:
                            embeddings = np.array(
                                embeddings_list, dtype=np.float32)
                        else:
                            # Fallback: re-embed each content
                            texts = []
                            for c in contents:
                                t = c.get("text", "")
                                if not t:
                                    t = c.get("input_text", "")
                                texts.append(t)
                            embeddings = np.array(
                                [self.embedder(t) for t in texts], dtype=np.float32)
                        if len(embeddings) == len(contents):
                            self.field.add_nodes_batch(
                                embeddings, contents,
                                node_ids=node_ids, modalities=modalities,
                            )
                            replayed += len(contents)
                    elif op == "delete":
                        node_ids = payload.get("node_ids", [])
                        if node_ids:
                            self.field.delete_nodes(node_ids)
                            replayed += len(node_ids)
                    elif op == "consolidate":
                        # Skip: snapshot already contains consolidated state;
                        # re-running consolidate would be non-deterministic.
                        pass
                except Exception:
                    logger.warning(
                        f"WAL replay failed for {op}: {payload}",
                        exc_info=True)
        finally:
            self.field.wal.enabled = was_enabled
        logger.info(f"WAL replay complete: {replayed} items recovered")

    def _get_phase(
            self,
            session_id: Optional[str] = None,
            embedding: Optional[NDArray] = None) -> float:
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        phase = (time.time() * 0.01) % (2 * np.pi)
        if session_id:
            self.session_phases[session_id] = phase
        return phase

    def _retrieve_and_format(
            self,
            query: str,
            embedding: NDArray,
            session_id: str) -> str:
        """Core retrieval pipeline shared by load_memory_variables and with_embedding."""
        phase = self._get_phase(session_id, embedding)

        # Phase 18: Engram-based retrieval (if enabled)
        if self.engram_manager is not None and self.engram_manager.index.size > 0:
            node_embs = {}
            for nid, node in self.field.nodes.items():
                emb = self._get_node_embedding(nid, node)
                if emb is not None:
                    node_embs[nid] = emb

            engram_results = self.engram_manager.retrieve_engrams(
                embedding, node_embs, top_k=self.field.cfg.top_k
            )

            if engram_results:
                results = self.engram_manager.expand_engrams(
                    engram_results, self.field, top_k=self.field.cfg.top_k
                )
                self.field.stats["engram_retrievals"] += 1
            else:
                results = self.field.query(
                    embedding,
                    phase,
                    top_k=self.field.cfg.top_k,
                    session_id=session_id)
        else:
            results = self.field.query(
                embedding,
                phase,
                top_k=self.field.cfg.top_k,
                session_id=session_id)

        # Session-scoped retrieval: filter results by session_id, with global
        # fallback
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results
                if node.content.get("session") == session_id
            ]
            if len(session_results) < self.field.cfg.top_k:
                global_results = [
                    (nid, score, node) for nid, score, node in results
                    if node.content.get("session") != session_id
                ]
                needed = self.field.cfg.top_k - len(session_results)
                session_results.extend(global_results[:needed])
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5  # 50% boost for session match
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[:self.field.cfg.top_k]
            self.field.stats["session_scoped_retrievals"] = self.field.stats.get(
                "session_scoped_retrievals", 0) + 1

        # Phase 1: Hybrid retrieval — blend RTMDK resonance with BM25 text
        # scores
        if self.field.cfg.hybrid_alpha < 1.0 and self.field.bm25_index is not None and results:
            bm25_results = self.field.bm25_index.search(
                query, self.field.cfg.top_k * 2)
            if bm25_results:
                bm25_scores = {nid: score for nid, score in bm25_results}
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                if max_bm25 > 0:
                    bm25_scores = {
                        nid: s / max_bm25 for nid,
                        s in bm25_scores.items()}

                alpha = self.field.cfg.hybrid_alpha
                blended = []
                for nid, score, node in results:
                    bm25_score = bm25_scores.get(nid, 0.0)
                    blended_score = alpha * score + (1 - alpha) * bm25_score
                    blended.append((nid, blended_score, node))

                for nid, bm25_score in bm25_scores.items():
                    if nid not in [
                            n[0] for n in blended] and bm25_score > self.field.cfg.min_response:
                        node = self.field.nodes.get(nid)
                        if node:
                            blended_score = alpha * 0.0 + \
                                (1 - alpha) * bm25_score
                            blended.append((nid, blended_score, node))

                blended.sort(key=lambda x: x[1], reverse=True)
                results = blended[:self.field.cfg.top_k]
                self.field.stats["hybrid_retrievals"] = self.field.stats.get(
                    "hybrid_retrievals", 0) + 1

        # Phase 15 Track 2: Proactive Clarification
        if self.config.proactive_clarification and results:
            max_score = results[0][1] if results else 0.0
            threshold = self.field.cfg.min_response * \
                self.config.clarification_threshold_ratio
            if 0 < max_score < threshold:
                clarification = self._generate_clarification(results, query)
                self.field.stats["clarifications_generated"] += 1
                return clarification

        # Context formatting
        if self.config.attention_tokens and results:
            context = format_context(results, ContextFormat.ATTENTION)
        elif self.config.attention_bias and results:
            context = format_cognitive_context(results, bias_applied=True)
            self.field.stats["attention_bias_applied"] += 1
        elif self.config.cognitive_compression and results:
            context = self.field._cognitive_compress(results)
            raw_context = format_context(results, self.config.context_format)
            tokens_saved = max(0, len(raw_context) - len(context))
            self.field.stats["context_tokens_saved"] += tokens_saved
            self.field.stats["cognitive_compressions"] += 1
        else:
            context = format_context(results, self.config.context_format)

        # Phase 16 Track 1: SymbolicOverlay
        if self.config.symbolic_overlay and self.field.symbolic_overlay and results:
            facts = []
            for nid, score, node in results[:3]:
                text = node.content.get("text", "")
                concepts = self.field.symbolic_overlay._extract_concepts(text)
                facts.extend(concepts)
            if facts:
                symbolic_ctx = self.field.symbolic_overlay.get_symbolic_context(
                    facts, max_depth=2)
                if symbolic_ctx:
                    context += "\n\n" + symbolic_ctx
                    self.field.stats["n_symbolic_inferences"] += 1
                    n_conflicts = sum(
                        1 for r in self.field.symbolic_overlay.rules.values() if r.is_contextual_exception)
                    self.field.stats["n_symbolic_conflicts"] = n_conflicts

        return context

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        embedding = self.embedder(query)
        return {
            "rtmdk_context": self._retrieve_and_format(
                query, embedding, session_id)}

    def load_memory_variables_with_embedding(
        self, inputs: Dict[str, str], embedding: NDArray
    ) -> Dict[str, str]:
        """Query memory with pre-computed embedding (no HTTP call).

        This is the optimized version that accepts an embedding from
        an external embedder, avoiding the HTTP call to LM Studio.
        Use this for batch processing and fair benchmark comparisons.

        Args:
            inputs: {"input": "query text", "session_id": "...", ...}
            embedding: Pre-computed embedding vector (768d for nomic-embed)

        Returns:
            {"rtmdk_context": formatted context string}
        """
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        return {
            "rtmdk_context": self._retrieve_and_format(
                query, embedding, session_id)}

    def _get_node_embedding(self, nid: str, node) -> Optional[np.ndarray]:
        """Retrieve stored embedding for a node, or approximate from latent position."""
        # Check if node has modal_embedding (cross-modal)
        if hasattr(
                node,
                'modal_embedding') and node.modal_embedding is not None:
            return node.modal_embedding
        # Fallback: approximate embedding by inverse-projection from latent_pos
        # This is lossy but better than nothing for engram similarity
        if hasattr(node, 'latent_pos') and node.latent_pos is not None:
            # Pad latent_pos (64d) to embedding_dim (768d) with zeros
            # Engram similarity uses cosine — zeros won't dominate
            emb_dim = self.field.cfg.embedding_dim
            latent = node.latent_pos
            if len(latent) < emb_dim:
                approx = np.zeros(emb_dim, dtype=np.float32)
                approx[:len(latent)] = latent
                return approx
            return latent[:emb_dim] if len(latent) > emb_dim else latent
        return None

    def _detect_tags(self, text: str) -> List[str]:
        """Auto-detect memory tags from text content."""
        tags = []
        lower = text.lower()

        # Greeting/name tags
        if any(
            w in lower for w in [
                "hello",
                "hi ",
                "hey",
                "привет",
                "здравствуй",
                "hi,",
                "hey,"]):
            tags.append("greeting")
        if any(
            w in lower for w in [
                "my name is",
                "i'm ",
                "i am ",
                "меня зовут",
                "мое имя"]):
            tags.append("name")

        # Topic tags
        if any(
            w in lower for w in [
                "code",
                "program",
                "python",
                "java",
                "javascript",
                "функци",
                "код",
                "програм"]):
            tags.append("coding")
        if any(
            w in lower for w in [
                "coffee",
                "tea",
                "food",
                "drink",
                "кофе",
                "чай",
                "еда"]):
            tags.append("food_drink")
        if any(
            w in lower for w in [
                "love",
                "like",
                "prefer",
                "enjoy",
                "люб",
                "нрав",
                "предпочита"]):
            tags.append("preference")
        if any(
            w in lower for w in [
                "work",
                "job",
                "career",
                "работ",
                "карьер",
                "професс"]):
            tags.append("work")
        if any(
            w in lower for w in [
                "live",
                "city",
                "country",
                "home",
                "жив",
                "город",
                "стран",
                "дом"]):
            tags.append("location")
        if any(
            w in lower for w in [
                "family",
                "friend",
                "dog",
                "cat",
                "pet",
                "семь",
                "друг",
                "собак",
                "кот",
                "питом"]):
            tags.append("relationships")

        return tags[:5]  # Limit to 5 tags

    def _generate_clarification(self, results: List, query: str) -> str:
        """Generate a clarification prompt from weak-resonance nodes."""
        lines = [
            "[CLARIFICATION] Не нашёл точных воспоминаний по запросу: \"{query[:80]}\""]
        lines.append("Полусовпадения (низкий резонанс):")
        for nid, score, node in results[:3]:
            text = node.content.get("text", "")[:60]
            lines.append(f"  [R:{score:.2f}] {text}")
        lines.append(
            "Уточните запрос или предоставьте дополнительный контекст.")
        return "\n".join(lines)

    def get_system_prompt(self, context: str) -> str:
        return build_system_prompt(
            context,
            self.config.context_format,
            self.config.use_structured_prompt)

    def save_context(
            self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save a conversation turn to memory with structured node format.

        Args:
            inputs: {"input": "user text", "session_id": "...", ...}
            outputs: {"output": "assistant text", ...}

        Node structure:
            input_text: User's message
            output_text: Assistant's response (empty if only input)
            role: "user" or "assistant"
            session: Session/character ID
            timestamp: Unix timestamp
            emotion: Detected emotion (neutral by default)
            tags: Auto-detected memory tags
            tier: episodic/semantic/procedural
            context: Additional metadata
        """
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")

        # If output is empty, still save the input
        if not output_text.strip():
            if not input_text.strip():
                return
            text_for_embedding = input_text
        else:
            text_for_embedding = output_text if len(
                output_text) > len(input_text) else input_text

        session_id = inputs.get("session_id", "default")
        timestamp = time.time()

        # Detect emotion from text
        emotion = "neutral"
        if input_text:
            lower_input = input_text.lower()
            if any(
                w in lower_input for w in [
                    "happy",
                    "love",
                    "great",
                    "wonderful",
                    "amazing",
                    "рад",
                    "люб",
                    "отличн",
                    "прекрасн"]):
                emotion = "positive"
            elif any(w in lower_input for w in [
                    "sad", "hate", "bad", "terrible", "angry",
                    "грустн", "ненавиж", "плох", "зл"]):
                emotion = "negative"
            elif any(w in lower_input for w in [
                    "?", "what", "why", "how", "when",
                    "где", "что", "как", "когда", "почему"]):
                emotion = "negative"
                emotion = "questioning"

        # Auto-detect tags from text
        all_text = f"{input_text} {output_text}"
        tags = self._detect_tags(all_text)

        # Build structured node content
        content = {
            "input_text": input_text,
            "output_text": output_text,
            "role": "assistant" if output_text.strip() else "user",
            "session": session_id,
            "timestamp": timestamp,
            "emotion": emotion,
            "tags": tags,
            "tier": "episodic",  # Will be refined by tier detection
            "context": {
                k: v for k, v in inputs.items()
                if k not in ["input", "query", "session_id", "embedding"]
            },
            "version": "2.0",  # Structured node version
        }

        embedding = self.embedder(text_for_embedding)
        phase = self._get_phase(session_id, embedding)
        modality = detect_modality(
            text_for_embedding) if self.config.cross_modal else "text"

        # Detect memory tier
        tier = detect_tier(text_for_embedding, inputs)
        content["tier"] = tier

        try:
            nid = self.field.add_node(
                embedding,
                content,
                phase,
                session_id=session_id,
                modality=modality)
        except SecurityViolationError:
            return

        # Set tier on the newly added node
        if nid in self.field.nodes:
            self.field.nodes[nid].tier = tier

        # Phase 18: Create/update engrams from co-activated nodes
        if self.engram_manager is not None:
            related_nodes = []
            for existing_nid, existing_node in self.field.nodes.items():
                existing_emb = self._get_node_embedding(
                    existing_nid, existing_node)
                if existing_emb is not None:
                    sim = float(np.dot(embedding, existing_emb) /
                                ((np.linalg.norm(embedding) +
                                  1e-8) *
                                 (np.linalg.norm(existing_emb) +
                                  1e-8)))
                    if sim > 0.5:
                        related_nodes.append((existing_nid, sim))
            related_nodes.append((nid, 1.0))

            if len(related_nodes) >= self.config.engram_min_nodes:
                node_embs = {}
                for nid, _ in related_nodes:
                    emb = self._get_node_embedding(
                        nid, self.field.nodes.get(nid))
                    if emb is not None:
                        node_embs[nid] = emb

                self.engram_manager.create_engram_from_nodes(
                    activated_nodes=related_nodes[:self.config.engram_max_nodes],
                    node_embeddings=node_embs,
                    semantic_core=text_for_embedding[:100],
                    context_tags=set(tags + [tier, session_id]),
                    tier=tier,
                )

        if self.config.enable_async:
            # Fix 4: Lazy async worker startup
            if self.config.async_pipeline and not self.field._workers_started:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.field._start_workers())
                    self.field._workers_started = True
                    # Enqueue for async processing
                    loop.create_task(self.field.evolve_q.put({"inputs": None}))
                except RuntimeError:
                    self.field.step()
            else:
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
        # Fix 3: Cancel background workers before replacing field
        for task in self._workers:
            if not task.done():
                task.cancel()
        self._workers.clear()
        self.field = RTMDKField(self.config)
        self.session_phases.clear()

    def inspect_node(self, node_id: str) -> Optional[Dict]:
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
            "causal_parents": node.causal_parents,
            "causal_strength": node.causal_strength,
            "causal_effects": node.causal_effects,
            "is_causal_root": node.is_causal_root,
            "is_healing": node.is_healing,
            "healing_origin": node.healing_origin,
            "local_density": node.local_density,
            "goal_tags": node.goal_tags,
            "cross_modal_score": node.cross_modal_score}
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        if node.velocity is not None:
            info["velocity"] = node.velocity.tolist()
        if node.modal_embedding is not None:
            info["modal_embedding"] = node.modal_embedding.tolist()
        return info

    def rollback(self, n_steps: int = 1) -> bool:
        return self.field.rollback_consolidation(n_steps)

    def get_rollback_history(self) -> List[Dict]:
        return [{"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(
            s["pre_state"])} for s in self.field._rollback_history]

    def do_intervention(self, node_id: str, text: str):
        emb = self.embedder(text)
        self.field.do_intervention(node_id, emb)

    def clear_interventions(self):
        self.field.clear_interventions()

    def get_dashboard(self) -> Dict:
        return self.field.get_field_health()

    def get_field_health(self) -> Dict:
        return self.field.get_field_health()

    def trigger_healing(self) -> List[Dict]:
        return self.field._self_heal()

    def counterfactual_query(self,
                             intervention: Dict[str,
                                                Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str,
                                                     Any]] = None) -> CounterfactualResult:
        return self.field.counterfactual_query(
            intervention, query_nodes, evidence)

    def get_causal_summary(self) -> Dict:
        return self.field.get_causal_summary()

    def get_contradictions(self) -> List[ContradictionRecord]:
        if self.field.causal_engine:
            return list(self.field.causal_engine.contradictions.values())
        return []

    def resolve_contradiction(
            self,
            contradiction_id: str,
            resolution: str) -> bool:
        if self.field.causal_engine and contradiction_id in self.field.causal_engine.contradictions:
            self.field.causal_engine.contradictions[contradiction_id].resolved = True
            self.field.causal_engine.contradictions[contradiction_id].resolution = resolution
            return True
        return False

    def validate_consolidation(
            self, node_a: str, node_b: str) -> Dict[str, Any]:
        if self.field.causal_engine:
            return self.field.causal_engine.validate_consolidation(
                node_a, node_b)
        return {
            "safe": True,
            "reasons": [],
            "causal_conflicts": [],
            "recommendation": "proceed"}

    # Phase 7: ODE
    def evolve_continuous(self,
                          inputs: Optional[List[Dict]] = None,
                          use_sde: bool = False) -> NDArray:
        return self.field.evolve_continuous(inputs, use_sde)

    def get_response_smoothness(self) -> float:
        return self.field.stats.get("response_smoothness", 1.0)

    # Phase 8: Agent
    def create_plan(
            self,
            goal: str,
            available_tools: List[str],
            context: Optional[Dict] = None) -> AgentPlan:
        return self.field.create_plan(goal, available_tools, context)

    def verify_hypothesis(self, hypothesis: str,
                          active_nodes: Optional[List[str]] = None) -> Hypothesis:
        return self.field.verify_hypothesis(hypothesis, active_nodes)

    def execute_tool(self,
                     tool_name: str,
                     arguments: Dict[str,
                                     Any]) -> ToolCall:
        return self.field.execute_tool(tool_name, arguments)

    def register_tool(self, name: str, func: Callable):
        self.field.register_tool(name, func)

    # Phase 9: Production
    def evaluate_response(
            self,
            question: str,
            answer: str,
            contexts: List[str],
            ground_truth: Optional[str] = None) -> EvalResult:
        return self.field.evaluate_response(
            question, answer, contexts, ground_truth)

    def compare_shadow(self, shadow_score: float,
                       production_score: float) -> Dict[str, Any]:
        return self.field.compare_shadow(shadow_score, production_score)

    def get_ragas_trend(self) -> Dict[str, float]:
        if self.field.ragas_evaluator:
            return self.field.ragas_evaluator.get_trend()
        return {}

    # Track 10: New methods
    def get_cross_modal_stats(self) -> Dict:
        return self.field.get_cross_modal_stats()

    def get_meta_controller_state(self) -> Dict:
        return self.field.get_meta_controller_state()

    def get_federated_status(self) -> Dict:
        return self.field.get_federated_status()

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        if self.field.dp:
            self.field.stats["privacy_budget_spent"] = self.field.dp.get_privacy_spent(
            )
        return {**self.field.stats, "config": self.config.asdict()}

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self,
                               base_query: str,
                               intervention: Dict[str,
                                                  float]) -> List[Dict]:
        """Generate hypothetical scenarios."""
        embedding = self.embedder(base_query)
        return self.field.imagine_counterfactual(embedding, intervention)

    def export_field(self, path: str, fmt: Optional[str] = None):
        self.field.export_field(path, fmt=fmt)

    @classmethod
    def import_field(cls, path: str, embedder: Callable,
                     wal_path: Optional[str] = None) -> "RTMDKMemory":
        return RTMDKField.import_field(path, embedder, wal_path=wal_path)

    # Phase 16 Track 3: Universal Memory Protocol
    def export_ump(self, path: str, source: str = "", comment: str = ""):
        """Export to Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError(
                "Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = UniversalMemoryProtocol.export(
            self.field, self, source=source, comment=comment)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ump, f, ensure_ascii=False, indent=2)

    @classmethod
    def import_ump(cls, path: str, embedder: Callable) -> "RTMDKMemory":
        """Import from Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError(
                "Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.import_ump(
            ump, embedder, memory_class=cls)

    def validate_ump(self, path: str) -> Dict:
        """Validate a UMP file."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            return {"valid": False, "issues": ["UMP not available"]}
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.validate(ump)
