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
import threading
import os
import copy
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

# Extracted engine classes (kept in sync with rtmdk/support/ modules)
from rtmdk.memory.utils import SecurityViolationError, detect_modality, apply_attention_bias, _enum_value
from rtmdk.memory.engram_cache import EngramEmbeddingCache
from rtmdk.memory.distributed_lock import DistributedLock
from rtmdk.memory.observability import MemoryMetrics, AlertRule
from rtmdk.memory.rag_quality import SentenceReranker, QueryDecomposer, FeedbackLoop
from rtmdk.pipeline import (
    PipelineExecutor,
    EmbedStage,
    RouteStage,
    RetrieveStage,
    RerankStage,
    CalibrateStage,
    ExplainStage,
)

logger = logging.getLogger(__name__)

# Phase 5: dataclass nodes extracted to rtmdk.nodes

try:
    from rtmdk.support.triton_backend import GPUBackend, TritonBackend, TRITON_AVAILABLE
except ImportError:
    GPUBackend = None  # type: ignore
    TritonBackend = None  # type: ignore
    TRITON_AVAILABLE = False

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

        # Phase 18b: Causal Traversal Engine
        self.causal_traversal_engine = None
        if self.config.causal_traversal:
            try:
                from rtmdk.engines.causal_traversal import CausalTraversalEngine
                self.causal_traversal_engine = CausalTraversalEngine(
                    max_hops=self.config.causal_max_hops,
                    decay_per_hop=0.5,
                )
            except Exception:
                logger.warning(
                    "CausalTraversalEngine initialization failed, disabling",
                    exc_info=True)

        # Phase 18c: Cross-Encoder Reranker
        self.reranker = None
        if getattr(self.config, "reranker_enabled", False):
            try:
                from rtmdk.production.reranker import CrossEncoderReranker
                self.reranker = CrossEncoderReranker(
                    model_name=getattr(self.config, "reranker_model", "BAAI/bge-reranker-v2-m3"),
                )
            except Exception:
                logger.warning(
                    "CrossEncoderReranker initialization failed, disabling",
                    exc_info=True)

        # P1: Contextual Retrieval
        self.header_generator = None
        if getattr(self.config, "contextual_retrieval", False):
            try:
                from rtmdk.production.contextual_retrieval import ContextualHeaderGenerator, ContextualEmbedderWrapper
                sot = getattr(self.field, "sot_tokenizer", None) if hasattr(self, "field") else None
                self.header_generator = ContextualHeaderGenerator(
                    backend=getattr(self.config, "contextual_backend", "heuristic"),
                    sot_tokenizer=sot,
                )
                self.embedder = ContextualEmbedderWrapper(self.embedder, self.header_generator)
                logger.info("Contextual retrieval enabled (%s)", self.header_generator.backend)
            except Exception:
                logger.warning("Contextual retrieval init failed, disabling", exc_info=True)

        # P1: BGE-M3 Hybrid
        self.bgem3_embedder = None
        self.sparse_index = None
        if getattr(self.config, "bgem3_enabled", False):
            try:
                from rtmdk.production.bgem3_embedder import BGEM3Embedder
                from rtmdk.production.sparse_index import SparseIndex
                self.bgem3_embedder = BGEM3Embedder(
                    model_name=getattr(self.config, "bgem3_model_name", "BAAI/bge-m3"),
                )
                self.sparse_index = SparseIndex()
                logger.info("BGE-M3 hybrid retrieval enabled")
            except Exception:
                logger.warning("BGE-M3 init failed, disabling", exc_info=True)

        # P2: SOT v2.0 Self-Supervised Embedder
        self._sot_v2 = None
        self._sot_v2_corpus: List[str] = []
        self._sot_v2_corpus_maxlen: int = getattr(
            self.config, "sot_max_corpus", 10000)
        self._sot_v2_online_buffer: List[List[int]] = []  # Tokenized docs for online update
        self._sot_v2_online_threshold: int = getattr(
            self.config, "sot_online_update_threshold", 10)
        self._sot_v2_online_lock = threading.Lock()
        sot_cfg = getattr(self.config, "sot", None)
        if sot_cfg and getattr(sot_cfg, "sot_v2_enabled", False):
            try:
                from rtmdk.memory.sot_v2.integration import SOTv2Embedder
                self._sot_v2 = SOTv2Embedder(
                    latent_dim=getattr(self.config, "latent_dim", 384),
                    a=getattr(sot_cfg, "sot_v2_a", 0.01),
                    window_size=getattr(sot_cfg, "sot_v2_window", 5),
                    remove_pc=getattr(sot_cfg, "sot_v2_remove_pc", True),
                )
                logger.info("SOT v2.0 embedder initialised (lazy training)")
                # Load pre-fitted aligner if path provided
                aligner_path = getattr(sot_cfg, "sot_v2_aligner_path", None)
                if aligner_path:
                    try:
                        self._sot_v2.load_aligner(aligner_path)
                    except Exception:
                        logger.warning("SOT v2.0 aligner load failed from %s", aligner_path, exc_info=True)
            except Exception:
                logger.warning("SOT v2.0 init failed, disabling", exc_info=True)

        # P1: Adaptive Cascade Router
        self.cascade_router = None
        if getattr(self.config, "cascade_enabled", False):
            try:
                from rtmdk.production.cascade_router import AdaptiveCascadeRouter
                self.cascade_router = AdaptiveCascadeRouter(
                    causal_threshold=getattr(self.config, "cascade_causal_threshold", 0.3),
                    factual_threshold=getattr(self.config, "cascade_factual_threshold", 0.3),
                )
                logger.info("Adaptive cascade router enabled")
            except Exception:
                logger.warning("Cascade router init failed, disabling", exc_info=True)

        # Circuit breaker for embedder (fallback to zero vector on repeated failures)
        if getattr(self.config, "embedder_circuit_breaker_enabled", True):
            from rtmdk.support.circuit_breaker import CircuitBreaker
            original_embedder = self.embedder
            dim = getattr(self.config, "embedding_dim", 384)
            cb = CircuitBreaker(
                "embedder",
                failure_threshold=getattr(self.config, "embedder_cb_threshold", 3),
                recovery_timeout=getattr(self.config, "embedder_cb_recovery", 30.0),
                default=np.zeros(dim, dtype=np.float32),
            )
            def _safe_embed(text):
                return cb.call(original_embedder, text)
            self.embedder = _safe_embed
            self._embedder_cb = cb
            logger.info("Embedder circuit breaker enabled")

        # Config validation warnings
        for warning in self.config.validate():
            logger.warning("Config validation: %s", warning)

        # v8.2.1 production distributed features
        self._init_backlog_modules()

    def _init_backlog_modules(self) -> None:
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

        # Engram embedding cache (avoids TieredNodeStore disk scans)
        sot_cfg = getattr(self.config, "sot", None)
        if sot_cfg and getattr(sot_cfg, "engram_cache_enabled", True):
            self.engram_cache = EngramEmbeddingCache(
                max_hot=getattr(sot_cfg, "engram_cache_max_hot", 10_000),
                max_warm=getattr(sot_cfg, "engram_cache_max_warm", 90_000))
        else:
            self.engram_cache = None

        # Distributed lock
        lock_path = getattr(sot_cfg, "distributed_lock_path", None) if sot_cfg else None
        lock_backend = getattr(sot_cfg, "distributed_lock_backend", "file")
        redis_url = getattr(sot_cfg, "distributed_lock_redis_url", None)
        if lock_path:
            self._distributed_lock = DistributedLock(
                lock_path, backend=lock_backend, redis_url=redis_url)
        else:
            self._distributed_lock = None

        # Observability
        if sot_cfg and getattr(sot_cfg, "observability_enabled", False):
            self.metrics = MemoryMetrics()
            self.metrics.add_alert_rule(AlertRule("high_latency", "query_p99", threshold=100.0))
            self.metrics.add_alert_rule(AlertRule("low_cache", "cache_hit_ratio", threshold=0.3, comparison="lt"))
            # Alert handlers
            webhook_url = getattr(sot_cfg, "alert_webhook_url", None)
            slack_url = getattr(sot_cfg, "alert_slack_url", None)
            pagerduty_key = getattr(sot_cfg, "alert_pagerduty_key", None)
            if webhook_url:
                from rtmdk.memory.observability import WebhookAlertHandler
                self.metrics.add_alert_handler(WebhookAlertHandler(webhook_url))
            if slack_url:
                from rtmdk.memory.observability import SlackAlertHandler
                self.metrics.add_alert_handler(SlackAlertHandler(slack_url))
            if pagerduty_key:
                from rtmdk.memory.observability import PagerDutyAlertHandler
                self.metrics.add_alert_handler(PagerDutyAlertHandler(pagerduty_key))
        else:
            self.metrics = None

        # RAG Quality
        self._sentence_reranker = None
        self._query_decomposer = None
        self._feedback_loop = None
        if sot_cfg and getattr(sot_cfg, "sentence_reranker_enabled", False):
            self._sentence_reranker = SentenceReranker(self.embedder)
        if sot_cfg and getattr(sot_cfg, "query_decomposition_enabled", False):
            # Optional LLM client for advanced decomposition
            llm_client = getattr(self, "_llm_client", None)
            self._query_decomposer = QueryDecomposer(llm_client=llm_client)
        if sot_cfg and getattr(sot_cfg, "feedback_loop_enabled", False):
            fb_path = getattr(sot_cfg, "feedback_loop_persist_path", None)
            self._feedback_loop = FeedbackLoop(self.embedder, persist_path=fb_path)
            self._feedback_loop.load()

        # Explainability & Query Enhancement
        self._result_explainer = None
        self._query_rewriter = None
        self._intent_classifier = None
        if sot_cfg and getattr(sot_cfg, "result_explainability_enabled", False):
            from rtmdk.memory.explainability import ResultExplainer
            self._result_explainer = ResultExplainer()
        if sot_cfg and getattr(sot_cfg, "query_rewrite_enabled", False):
            from rtmdk.memory.explainability import QueryRewriter
            llm_client = getattr(self, "_llm_client", None)
            self._query_rewriter = QueryRewriter(embedder=self.embedder, llm_client=llm_client)
        if sot_cfg and getattr(sot_cfg, "query_intent_classification_enabled", False):
            from rtmdk.memory.explainability import QueryIntentClassifier
            llm_client = getattr(self, "_llm_client", None)
            self._intent_classifier = QueryIntentClassifier(llm_client=llm_client)

        # Safety & Rollback
        from rtmdk.memory.safety import RollbackManager, PoisonedMemoryDetector
        self._rollback_manager = RollbackManager()
        self._poison_detector = PoisonedMemoryDetector()

    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]

    def add_node(self, embedding: NDArray, content: Dict, **kwargs) -> str:
        """Add a node to the memory field. Delegates to RTMDKField.add_node."""
        # P2: Accumulate corpus for SOT v2.0 lazy training & online update
        if self._sot_v2 is not None:
            text = content.get("text", "")
            if text:
                # FIFO corpus to prevent OOM
                self._sot_v2_corpus.append(text)
                if len(self._sot_v2_corpus) > self._sot_v2_corpus_maxlen:
                    self._sot_v2_corpus.pop(0)
                # Online update: tokenize and buffer (thread-safe)
                if hasattr(self._sot_v2, '_vocab') and self._sot_v2._vocab:
                    tokens = [self._sot_v2._vocab[w] for w in self._sot_v2._word_tokenize(text)
                              if w in self._sot_v2._vocab]
                    if tokens:
                        with self._sot_v2_online_lock:
                            self._sot_v2_online_buffer.append(tokens)
                            should_update = len(self._sot_v2_online_buffer) >= self._sot_v2_online_threshold
                            if should_update:
                                buffer_to_update = self._sot_v2_online_buffer
                                self._sot_v2_online_buffer = []
                        if should_update:
                            try:
                                self._sot_v2._embedder.online_update(buffer_to_update)
                            except Exception:
                                logger.warning("SOT v2.0 online update failed", exc_info=True)
        # P1: Matryoshka-lite — truncate for HNSW, keep full for resonance
        modal_emb = None
        if getattr(self.config, "matryoshka_mode", False):
            hnsw_dim = getattr(self.config, "matryoshka_hnsw_dim", self.config.latent_dim)
            if embedding.shape[0] > hnsw_dim:
                modal_emb = embedding
                embedding = embedding[:hnsw_dim]
        # P1: Sparse index insert (after we know the real node_id)
        t0 = time.perf_counter()
        node_id = self.field.add_node(embedding, content, modal_embedding=modal_emb, **kwargs)
        if getattr(self, "sparse_index", None) is not None:
            sparse_vec = content.get("sparse_embedding")
            if sparse_vec:
                self.sparse_index.insert(node_id, sparse_vec)
        # Engram cache: keep embedding in RAM for fast engram similarity
        if self.engram_cache is not None:
            self.engram_cache.add(node_id, embedding)
        # Observability
        if self.metrics is not None:
            self.metrics.record_ingestion((time.perf_counter() - t0) * 1000)
        # v8.2.1 hooks
        self._on_node_added(node_id, embedding, content, kwargs)
        return node_id

    def _on_node_added(
        self, node_id: str, embedding: NDArray, content: Dict, add_kwargs: Dict
    ) -> None:
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
        # P1: Matryoshka-lite
        modal_embs = None
        if getattr(self.config, "matryoshka_mode", False):
            hnsw_dim = getattr(self.config, "matryoshka_hnsw_dim", self.config.latent_dim)
            if embeddings.shape[1] > hnsw_dim:
                modal_embs = embeddings
                embeddings = embeddings[:, :hnsw_dim]
        # P1: Sparse index
        if getattr(self, "sparse_index", None) is not None:
            for i, content in enumerate(contents):
                sparse_vec = content.get("sparse_embedding")
                nid = node_ids[i] if node_ids else f"n_{i}_"
                if sparse_vec:
                    self.sparse_index.insert(nid, sparse_vec)
        result = self.field.add_nodes_batch(
            embeddings,
            contents,
            phases,
            node_ids,
            session_ids,
            modalities,
            skip_projection,
            modal_embeddings=modal_embs)
        # v8.2.1 hooks
        rm = getattr(self, "replication_manager", None)
        for i, nid in enumerate(result):
            emb = embeddings[i]
            content = contents[i]
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
        # P2: Accumulate corpus for SOT v2.0
        if self._sot_v2 is not None:
            for content in contents:
                text = content.get("text", "")
                if text:
                    self._sot_v2_corpus.append(text)
        return result

    def train_sot_v2(self, extra_texts: Optional[List[str]] = None) -> bool:
        """Train SOT v2.0 embedder on accumulated corpus.

        Call this after ingesting all documents.  Optionally provide
        additional query texts for contrastive fine-tuning.

        Returns:
            True if training succeeded.
        """
        if self._sot_v2 is None:
            logger.warning("train_sot_v2 called but SOT v2.0 is not enabled")
            return False
        corpus = list(self._sot_v2_corpus)
        if extra_texts:
            corpus.extend(extra_texts)
        if not corpus:
            logger.warning("train_sot_v2: no corpus to train on")
            return False
        logger.info("train_sot_v2: training on %d texts", len(corpus))
        try:
            self._sot_v2.train(corpus)
            # Optional: teacher alignment (Procrustes or contrastive distillation)
            sot_cfg = getattr(self.config, "sot", None)
            teacher_name = getattr(sot_cfg, "sot_v2_align_teacher", None) if sot_cfg else None
            if teacher_name:
                try:
                    from sentence_transformers import SentenceTransformer
                    teacher = SentenceTransformer(teacher_name)
                    batch_size = getattr(sot_cfg, "sot_v2_align_batch_size", 64)
                    center = getattr(sot_cfg, "sot_v2_align_center", True)
                    align_mode = getattr(sot_cfg, "sot_v2_align_mode", "procrustes")
                    logger.info("train_sot_v2: aligning to teacher %s (mode=%s)...", teacher_name, align_mode)
                    sif_embs = self._sot_v2.embed_batch(corpus)
                    teacher_embs = []
                    for i in range(0, len(corpus), batch_size):
                        batch = corpus[i:i + batch_size]
                        embs = teacher.encode(batch)
                        if not isinstance(embs, np.ndarray):
                            embs = np.asarray(embs)
                        teacher_embs.append(embs)
                    teacher_embs = np.concatenate(teacher_embs, axis=0)
                    if align_mode == "contrastive_distill":
                        self._sot_v2._embedder.contrastive_distill(sif_embs, teacher_embs)
                    else:
                        # Procrustes (legacy)
                        norms = np.linalg.norm(teacher_embs, axis=1, keepdims=True) + 1e-8
                        teacher_embs = teacher_embs / norms
                        self._sot_v2._embedder.align_to_teacher(sif_embs, teacher_embs, center=center)
                except Exception:
                    logger.warning("train_sot_v2: teacher alignment failed", exc_info=True)
            # Replace embedder with trained SOT v2.0
            self.embedder = self._sot_v2
            # Invalidate query cache (scores changed)
            if self.field.query_cache is not None:
                self.field.query_cache.clear()
            # Reset conformal calibrator (distribution changed)
            if self.field.conformal_calibrator is not None:
                from rtmdk.memory.conformal import ConformalCalibrator
                self.field.conformal_calibrator = ConformalCalibrator(
                    alpha=self.config.conformal_alpha)
                logger.info("train_sot_v2: conformal calibrator reset")
            logger.info("train_sot_v2: SOT v2.0 embedder active")
            return True
        except Exception:
            logger.error("train_sot_v2 failed", exc_info=True)
            return False

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
            embedding: Optional[NDArray] = None,
            content: Optional[Dict] = None) -> float:
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        # Use semantic phase from field when content is available
        if content is not None:
            return self.field._get_phase(session_id, embedding, content=content)
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
        # Query decomposition for multi-hop retrieval
        if self._query_decomposer is not None:
            sub_queries = self._query_decomposer.decompose(query)
        else:
            sub_queries = [query]

        all_results = []
        for sub_q in sub_queries:
            sub_emb = self.embedder(sub_q)
            phase = self._get_phase(session_id, sub_emb)

            # Phase 18: Engram-based retrieval (if enabled)
            if self.engram_manager is not None and self.engram_manager.index.size > 0:
                if self.engram_cache is not None and len(self.engram_cache) > 0:
                    node_embs = self.engram_cache.get_all()
                else:
                    node_embs = {}
                    for nid, node in self.field.nodes.items():
                        emb = self._get_node_embedding(nid, node)
                        if emb is not None:
                            node_embs[nid] = emb

                engram_results = self.engram_manager.retrieve_engrams(
                    sub_emb, node_embs, top_k=self.field.cfg.top_k
                )

                if engram_results:
                    results = self.engram_manager.expand_engrams(
                        engram_results, self.field, top_k=self.field.cfg.top_k
                    )
                    self.field.stats["engram_retrievals"] += 1
                else:
                    results = self.field.query(
                        sub_emb,
                        phase,
                        top_k=self.field.cfg.top_k,
                        session_id=session_id)
            else:
                results = self.field.query(
                    sub_emb,
                    phase,
                    top_k=self.field.cfg.top_k,
                    session_id=session_id)
            all_results.extend(results)

        # Deduplicate and re-rank combined results
        seen = set()
        results = []
        for nid, score, node in sorted(all_results, key=lambda x: x[1], reverse=True):
            if nid not in seen:
                results.append((nid, score, node))
                seen.add(nid)

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

    def _retrieve_nodes_impl(
            self,
            query: str,
            embedding: NDArray,
            top_k: Optional[int] = None,
            session_id: Optional[str] = None,
            sparse_vec: Optional[Dict[int, float]] = None) -> List[Tuple[str, float, Any]]:
        """Internal retrieval without metrics/locks/reranking."""
        # P1: Query expansion for short queries (< 3 content words)
        original_query = query
        if (query and
                getattr(self.config, "query_expand_short", False) and
                hasattr(self.embedder, "expand_query_terms")):
            content_words = [w for w in query.lower().split() if len(w) > 2]
            if len(content_words) < 3:
                expanded = self.embedder.expand_query_terms(query, n_terms=3)
                if expanded:
                    expansion_text = " ".join([term for term, _ in expanded])
                    query = f"{original_query} {expansion_text}"
                    logger.debug("Query expanded: '%s' → '%s'", original_query, query)

        query_content = {"text": query} if query else None
        phase = self._get_phase(session_id, embedding, content=query_content)
        tk = top_k or self.field.cfg.top_k

        # P1: Adaptive Cascade Router
        if self.cascade_router is not None:
            from rtmdk.production.cascade_router import QueryType
            route = self.cascade_router.classify(query)
            if route == QueryType.FACTUAL:
                return self.field.query(embedding, phase, top_k=tk, session_id=session_id, query_text=query)

        # Primary: resonance retrieval
        results = self.field.query(
            embedding, phase, top_k=tk, session_id=session_id, query_text=query)

        # P1: Sparse index fallback (BGE-M3 learned sparse)
        if len(results) < tk and self.sparse_index is not None and sparse_vec:
            sparse_hits = self.sparse_index.search(sparse_vec, tk * 2)
            if sparse_hits:
                seen = {nid for nid, _, _ in results}
                for nid, score in sparse_hits:
                    if nid not in seen:
                        node = self.field.nodes.get(nid)
                        if node:
                            results.append((nid, score * 0.5, node))
                            seen.add(nid)
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:tk]

        # Phase 18: Engram-based retrieval as fallback (if resonance under-delivers)
        if len(results) < tk and self.engram_manager is not None and self.engram_manager.index.size > 0:
            # Use engram cache if available to avoid TieredNodeStore disk scan
            if self.engram_cache is not None and len(self.engram_cache) > 0:
                node_embs = self.engram_cache.get_all()
            else:
                node_embs = {}
                for nid, node in self.field.nodes.items():
                    emb = self._get_node_embedding(nid, node)
                    if emb is not None:
                        node_embs[nid] = emb

            engram_results = self.engram_manager.retrieve_engrams(
                embedding, node_embs, top_k=tk)
            if engram_results:
                engram_nodes = self.engram_manager.expand_engrams(
                    engram_results, self.field, top_k=tk)
                seen = {nid for nid, _, _ in results}
                for nid, score, node in engram_nodes:
                    if nid not in seen:
                        results.append((nid, score, node))
                        seen.add(nid)
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:tk]
                self.field.stats["engram_retrievals"] = self.field.stats.get(
                    "engram_retrievals", 0) + 1

        # Session-scoped retrieval
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results
                if node.content.get("session") == session_id
            ]
            if len(session_results) < tk:
                global_results = [
                    (nid, score, node) for nid, score, node in results
                    if node.content.get("session") != session_id
                ]
                needed = tk - len(session_results)
                session_results.extend(global_results[:needed])
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[:tk]
            self.field.stats["session_scoped_retrievals"] = self.field.stats.get(
                "session_scoped_retrievals", 0) + 1

        # Hybrid BM25 blend
        if self.field.cfg.hybrid_alpha < 1.0 and self.field.bm25_index is not None and results:
            bm25_results = self.field.bm25_index.search(query, tk * 2)
            if bm25_results:
                bm25_scores = {nid: score for nid, score in bm25_results}
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                if max_bm25 > 0:
                    bm25_scores = {nid: s / max_bm25 for nid, s in bm25_scores.items()}
                alpha = self.field.cfg.hybrid_alpha
                blended = []
                for nid, score, node in results:
                    bm25_score = bm25_scores.get(nid, 0.0)
                    blended.append((nid, alpha * score + (1 - alpha) * bm25_score, node))
                for nid, bm25_score in bm25_scores.items():
                    if nid not in [n[0] for n in blended] and bm25_score > self.field.cfg.min_response:
                        node = self.field.nodes.get(nid)
                        if node:
                            blended.append((nid, alpha * 0.0 + (1 - alpha) * bm25_score, node))
                blended.sort(key=lambda x: x[1], reverse=True)
                results = blended[:tk]
                self.field.stats["hybrid_retrievals"] = self.field.stats.get(
                    "hybrid_retrievals", 0) + 1

        # Causal Traversal
        if self.causal_traversal_engine is not None and results:
            results = self.causal_traversal_engine.retrieve_with_causal(
                results, self.field, top_k=tk)

        # Cross-Encoder Reranker
        if self.reranker is not None and results and query:
            results = self.reranker.rerank(query, results, top_k=tk)

        return results

    def retrieve_nodes(
            self,
            query: str,
            embedding: NDArray,
            top_k: Optional[int] = None,
            session_id: Optional[str] = None,
            sparse_vec: Optional[Dict[int, float]] = None) -> List[Tuple[str, float, Any]]:
        """Retrieve memory nodes with full pipeline: cascade → resonance → sparse → engrams → causal → reranker.

        Includes observability (latency tracking), distributed locking,
        sentence reranking, and query decomposition.

        Returns:
            List of (node_id, score, node) tuples.
        """
        # Distributed lock for multi-process safety
        if self._distributed_lock is not None:
            if not self._distributed_lock.acquire(blocking=True):
                logger.warning("retrieve_nodes: failed to acquire distributed lock")

        t0 = time.perf_counter()
        cache_hit = False

        # Check query cache
        if self.field.query_cache is not None:
            cache_key = self.field._query_cache_key(
                self.field._project(embedding),
                self._get_phase(session_id, embedding),
                top_k or self.field.cfg.top_k,
                "text",
                session_id,
            )
            cached = self.field.query_cache.get_raw(cache_key)
            if cached is not None:
                cache_hit = True
                results = cached
            else:
                results = self._retrieve_nodes_impl(query, embedding, top_k, session_id, sparse_vec)
                self.field.query_cache.set_raw(cache_key, results)
        else:
            results = self._retrieve_nodes_impl(query, embedding, top_k, session_id, sparse_vec)

        # Auto query rewrite on low-quality results
        if self._query_rewriter is not None and self._query_rewriter.should_rewrite(
            results, threshold=getattr(self.config, "query_rewrite_threshold", 0.3)
        ):
            rewritten = self._query_rewriter.rewrite(query, results)
            if rewritten != query:
                logger.debug("Query rewritten: '%s' -> '%s'", query, rewritten)
                rew_emb = self.embedder(rewritten)
                results = self._retrieve_nodes_impl(
                    rewritten, rew_emb, top_k, session_id, sparse_vec)

        # Intent-aware retrieval tuning
        if self._intent_classifier is not None:
            intent = self._intent_classifier.classify(query)
            if intent == "factual":
                # Boost top result precision
                pass
            elif intent == "exploratory":
                # Diversify results via causal traversal if available
                pass

        # Sentence-level reranking
        if self._sentence_reranker is not None and results:
            results = self._sentence_reranker.rerank(query, results, top_k=top_k or self.field.cfg.top_k)

        # Observability
        latency_ms = (time.perf_counter() - t0) * 1000
        if self.metrics is not None:
            self.metrics.record_query(latency_ms, cache_hit=cache_hit)
            alerts = self.metrics.check_alerts()
            for alert in alerts:
                logger.warning(alert)

        # Release distributed lock
        if self._distributed_lock is not None:
            self._distributed_lock.release()

        return results

    def retrieve_nodes_with_explanations(
            self,
            query: str,
            embedding: NDArray,
            top_k: Optional[int] = None,
            session_id: Optional[str] = None,
            sparse_vec: Optional[Dict[int, float]] = None) -> Dict:
        """Retrieve nodes with human-readable explanations.

        Returns:
            {"results": [(nid, score, node), ...], "explanations": [...], "intent": str}
        """
        results = self.retrieve_nodes(query, embedding, top_k, session_id, sparse_vec)
        intent = "unknown"
        if self._intent_classifier is not None:
            intent = self._intent_classifier.classify(query)

        explanations = []
        if self._result_explainer is not None:
            for nid, score, node in results:
                explanations.append(
                    self._result_explainer.explain(query, nid, score, node, session_id or "default")
                )

        return {"results": results, "explanations": explanations, "intent": intent}

    def retrieve_nodes_batch(
            self,
            queries: List[str],
            embeddings: NDArray,
            top_k: Optional[int] = None,
            session_ids: Optional[List[str]] = None) -> List[List[Tuple[str, float, Any]]]:
        """Batch retrieval — vectorized resonance across multiple queries.

        ~50-100x faster than sequential retrieve_nodes() for large batches.
        """
        tk = top_k or self.field.cfg.top_k
        n_queries = len(queries)
        if n_queries == 0:
            return []

        # Compute phases
        phases = np.zeros(n_queries, dtype=np.float32)
        for i, (query, sid) in enumerate(zip(queries, session_ids or [None] * n_queries)):
            content = {"text": query} if query else None
            phases[i] = self._get_phase(sid, embeddings[i], content=content)

        # Project embeddings
        query_latents = np.vstack([self.field._project(embeddings[i]) for i in range(n_queries)])

        # Get all node IDs
        all_nids = list(self.field.node_index)
        if not all_nids:
            return [[] for _ in range(n_queries)]

        # Batch resonance: (n_queries, n_nodes)
        scores = self.field._batch_resonance(query_latents, phases, all_nids)

        # Build node lookup
        nodes = self.field.nodes

        results = []
        for i in range(n_queries):
            sid = session_ids[i] if session_ids else None
            # Session boost
            row_scores = scores[i].copy()
            if sid and sid != "default":
                for j, nid in enumerate(all_nids):
                    if nodes[nid].content.get("session") == sid:
                        row_scores[j] *= 1.5

            # Filter by min_response and get top_k
            valid = row_scores >= self.field.cfg.min_response
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                results.append([])
                continue

            valid_scores = row_scores[valid_indices]
            top_local = np.argsort(-valid_scores)[:tk]
            top_global = valid_indices[top_local]

            query_results = []
            for idx in top_global:
                nid = all_nids[idx]
                node = nodes[nid]
                query_results.append((nid, float(row_scores[idx]), node))
                node.last_resonated = time.time()
            results.append(query_results)

        self.field.stats["batch_queries"] = self.field.stats.get("batch_queries", 0) + n_queries
        return results

    # ------------------------------------------------------------------
    # Pipeline API (v8.3+)
    # ------------------------------------------------------------------
    def build_pipeline(self) -> PipelineExecutor:
        """Build an explicit stage-based pipeline for retrieval.

        Each stage is independently observable and swappable.
        Stages are constructed from the features enabled in config.
        Circuit breakers are attached for automatic SLO enforcement.
        """
        from rtmdk.production.cascade_router import AdaptiveCascadeRouter
        from rtmdk.pipeline.health import PipelineHealthMonitor

        stages = []
        monitor = PipelineHealthMonitor()

        # Default SLO thresholds (ms) — can be overridden via config in future
        monitor.set_threshold("embed", 5000.0)
        monitor.set_threshold("route", 100.0)
        monitor.set_threshold("retrieve", 500.0)
        monitor.set_threshold("rerank", 1000.0)
        monitor.set_threshold("calibrate", 200.0)
        monitor.set_threshold("explain", 100.0)

        # Stage 1: Embed (optional — caller may provide embedding directly)
        embed_stage = EmbedStage(self.embedder)
        embed_stage.circuit_breaker = monitor.get_breaker("embed")
        stages.append(embed_stage)

        # Stage 2: Route
        router = None
        if getattr(self.config, "cascade_enabled", False):
            router = AdaptiveCascadeRouter()
        route_stage = RouteStage(router)
        route_stage.circuit_breaker = monitor.get_breaker("route")
        stages.append(route_stage)

        # Stage 3: Retrieve
        retrieve_stage = RetrieveStage(self.field)
        retrieve_stage.circuit_breaker = monitor.get_breaker("retrieve")
        stages.append(retrieve_stage)

        # Stage 4: Rerank
        rerank_stage = RerankStage(self._sentence_reranker)
        rerank_stage.circuit_breaker = monitor.get_breaker("rerank")
        stages.append(rerank_stage)

        # Stage 5: Calibrate
        calibrator = getattr(self.field, "conformal_calibrator", None)
        calibrate_stage = CalibrateStage(calibrator)
        calibrate_stage.circuit_breaker = monitor.get_breaker("calibrate")
        stages.append(calibrate_stage)

        # Stage 6: Explain
        explain_stage = ExplainStage(self._result_explainer)
        explain_stage.circuit_breaker = monitor.get_breaker("explain")
        stages.append(explain_stage)

        return PipelineExecutor(stages)

    def retrieve_nodes_pipeline(
        self,
        query: str,
        embedding: Optional[NDArray] = None,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve nodes using the explicit pipeline API.

        Returns:
            Dict with keys:
                - results: List[Tuple[str, float, Any]]
                - route: str (factual/standard/deep)
                - explanations: List[Dict]
                - metrics: per-stage latency breakdown
        """
        pipeline = self.build_pipeline()
        ctx = pipeline.run(
            query_text=query,
            top_k=top_k or self.field.cfg.top_k,
            session_id=session_id,
            embedding=embedding,
        )
        return {
            "results": ctx.results,
            "route": ctx.route,
            "explanations": ctx.explanations,
            "metrics": ctx.to_dict(),
        }

    def health_check_pipeline(self) -> Dict[str, Any]:
        """Run health checks on every pipeline stage.

        Returns:
            {"healthy": bool, "stages": [{"stage": str, "healthy": bool, "reason": str}]}
        """
        pipeline = self.build_pipeline()
        stage_health = []
        all_healthy = True
        for stage in pipeline.stages:
            healthy, reason = stage.health_check()
            if not healthy:
                all_healthy = False
            stage_health.append({
                "stage": stage.name,
                "healthy": healthy,
                "reason": reason,
            })
        return {"healthy": all_healthy, "stages": stage_health}

    def add_feedback(self, query: str, node_id: str, relevant: bool) -> bool:
        """Provide explicit feedback to refine embeddings.

        Args:
            query: The original query text.
            node_id: ID of the retrieved node.
            relevant: True if the node was relevant, False otherwise.

        Returns:
            True if feedback was applied.
        """
        if self._feedback_loop is None:
            logger.warning("add_feedback: feedback_loop not enabled in config")
            return False
        node = self.field.nodes.get(node_id)
        if node is None:
            return False
        node_text = node.content.get("text", "")
        return self._feedback_loop.add_feedback(query, node_text, relevant)

    def get_metrics(self) -> Dict:
        """Return current observability metrics snapshot."""
        if self.metrics is None:
            return {"observability_enabled": False}
        return self.metrics.snapshot()

    def query_with_confidence(
        self,
        query: str,
        embedding: NDArray,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve with conformal prediction confidence guarantee.

        Returns a prediction set that contains the true relevant result
        with probability >= 1 - alpha (marginal coverage guarantee).

        Args:
            query: Query text.
            embedding: Query embedding.
            top_k: Max results to return.
            session_id: Optional session filter.
            alpha: Miscoverage rate (default from config.conformal_alpha).

        Returns:
            Dict with keys:
                - results: List of (node_id, score, node) tuples
                - prediction_set: List of node_ids in the conformal set
                - confidence: 1 - alpha
                - threshold: Minimum score for inclusion
                - coverage_guarantee: True if calibrator has enough samples
        """
        tk = top_k or self.field.cfg.top_k
        alpha = alpha or getattr(
            self.config, "conformal_alpha", 0.1)

        # Standard retrieval
        results = self.retrieve_nodes(
            query, embedding, top_k=tk, session_id=session_id)

        # Conformal prediction
        cal = getattr(self.field, "conformal_calibrator", None)
        if cal is None or not self.config.conformal_prediction:
            return {
                "results": results,
                "prediction_set": [nid for nid, _, _ in results],
                "confidence": 1.0 - alpha,
                "threshold": 0.0,
                "coverage_guarantee": False,
                "reason": "conformal_prediction disabled or not initialized",
            }

        min_calib = getattr(self.config, "conformal_min_calib", 50)
        if cal.n_calibrated < min_calib:
            return {
                "results": results,
                "prediction_set": [nid for nid, _, _ in results],
                "confidence": 1.0 - alpha,
                "threshold": 0.0,
                "coverage_guarantee": False,
                "reason": f"insufficient calibration samples ({cal.n_calibrated}/{min_calib})",
            }

        scores = [score for _, score, _ in results]
        nids = [nid for nid, _, _ in results]
        pred_set, confidence, threshold = cal.predict(scores, nids)

        return {
            "results": results,
            "prediction_set": pred_set,
            "confidence": confidence,
            "threshold": threshold,
            "coverage_guarantee": True,
            "reason": "marginal coverage guarantee active",
        }

    def calibrate_conformal_sot(self, n_samples: int = 50) -> bool:
        """Auto-calibrate conformal prediction using SOT cosine scores.

        Uses pseudo-queries (sample nodes as queries) to build a calibration
        set of cosine similarity scores.  This is essential when using SOT
        embedders where resonance scores differ from RTMDK's native scores.

        Args:
            n_samples: Number of pseudo-queries to generate.

        Returns:
            True if calibration succeeded.
        """
        cal = getattr(self.field, "conformal_calibrator", None)
        if cal is None:
            logger.warning("calibrate_conformal_sot: no calibrator available")
            return False
        nodes = self.field.nodes
        nids = list(self.field.node_index)
        if len(nids) < 10:
            logger.warning("calibrate_conformal_sot: too few nodes (%d)", len(nids))
            return False

        import random
        sample_size = min(n_samples, len(nids))
        sample_nids = random.sample(nids, sample_size)

        # Pre-compute normalized document matrix once (O(N) memory, not O(N²))
        doc_embs = []
        nid_to_idx = {}
        for i, dnid in enumerate(nids):
            d_emb = self._get_node_embedding(dnid, nodes[dnid])
            if d_emb is not None:
                d_emb = d_emb / (np.linalg.norm(d_emb) + 1e-8)
                doc_embs.append(d_emb)
                nid_to_idx[dnid] = len(doc_embs) - 1
        if not doc_embs:
            return False
        doc_matrix = np.stack(doc_embs)

        for nid in sample_nids:
            node = nodes[nid]
            text = node.content.get("text", "")
            if not text:
                continue
            target_idx = nid_to_idx.get(nid)
            if target_idx is None:
                continue
            try:
                q_emb = self.embedder(text)
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                sims = doc_matrix @ q_emb
                cal.add_sample(float(sims[target_idx]))
            except Exception:
                continue

        logger.info(
            "calibrate_conformal_sot: calibrated with %d samples (total=%d)",
            sample_size, cal.n_calibrated)
        return cal.n_calibrated >= getattr(self.config, "conformal_min_calib", 50)

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

    def batch_query(
            self,
            embeddings: List[np.ndarray],
            top_k: Optional[int] = None,
            session_id: Optional[str] = None) -> List[List[Tuple[str, float, Any]]]:
        """Batch query memory for multiple embeddings."""
        if self.field is None:
            raise RuntimeError("Field not initialized")
        phases = [
            self._get_phase(
                session_id,
                emb) for emb in embeddings] if session_id else [
            self._get_phase() for _ in embeddings]
        return self.field.batch_query(
            embeddings,
            phases=phases,
            top_k=top_k,
            session_id=session_id)

    def fit_projection(self, corpus_embeddings: np.ndarray) -> None:
        """Fit projection learner on corpus embeddings."""
        if self.field is not None:
            self.field.fit_projection(corpus_embeddings)

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
            f"[CLARIFICATION] Не нашёл точных воспоминаний по запросу: \"{query[:80]}\""]
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
        # Use retrieval instead of O(N) scan for scalability
        if self.engram_manager is not None:
            # Fast path: retrieve top-k similar nodes via HNSW/resonance
            try:
                retrieved = self.retrieve_nodes(
                    text_for_embedding, embedding,
                    top_k=self.config.engram_max_nodes * 2,
                    session_id=session_id)
                related_nodes = []
                for rnid, rscore, _ in retrieved:
                    if rscore >= self.config.min_response:
                        related_nodes.append((rnid, float(rscore)))
            except Exception:
                related_nodes = []
            related_nodes.append((nid, 1.0))

            if len(related_nodes) >= self.config.engram_min_nodes:
                node_embs = {}
                for rnid, _ in related_nodes:
                    emb = self._get_node_embedding(
                        rnid, self.field.nodes.get(rnid))
                    if emb is not None:
                        node_embs[rnid] = emb

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

    def save_state(self, dir_path: str) -> None:
        """Persist backlog module state to disk."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        if self.engram_cache is not None:
            self.engram_cache.save(os.path.join(dir_path, "engram_cache.npz"))
        if self._feedback_loop is not None and self._feedback_loop.persist_path:
            self._feedback_loop._flush()
        if self.metrics is not None:
            self.metrics.flush_to_file(os.path.join(dir_path, "metrics.json"))

    def load_state(self, dir_path: str) -> None:
        """Restore backlog module state from disk."""
        import os
        cache_path = os.path.join(dir_path, "engram_cache.npz")
        if self.engram_cache is not None and os.path.exists(cache_path):
            self.engram_cache.load(cache_path)
        if self._feedback_loop is not None:
            self._feedback_loop.load()

    def clear(self) -> None:
        # Fix 3: Cancel background workers before replacing field
        for task in self.field._workers:
            if not task.done():
                task.cancel()
        self.field._workers.clear()
        self.field = RTMDKField(self.config, wal_path=self.wal_path)
        self.session_phases.clear()
        # Reset SOT corpus to prevent stale data
        self._sot_v2_corpus.clear()
        self._sot_v2_online_buffer.clear()
        if self.engram_cache is not None:
            self.engram_cache.clear()
        if self.metrics is not None:
            old_rules = self.metrics.alert_rules
            self.metrics = MemoryMetrics()
            for rule in old_rules:
                self.metrics.add_alert_rule(rule)

    def health_check(self) -> Dict:
        """Comprehensive health check for k8s probes and monitoring.

        Returns:
            {"status": "healthy"|"degraded"|"unhealthy", "checks": {...}}
        """
        import psutil
        checks = {}
        status = "healthy"

        # Node count
        node_count = len(self.field.nodes)
        max_nodes = getattr(self.config, "max_nodes", 100000) or 100000
        node_ratio = node_count / max_nodes
        checks["node_count"] = {"value": node_count, "max": max_nodes, "ratio": round(node_ratio, 3)}
        if node_ratio > 0.9:
            status = "degraded"
            checks["node_count"]["warning"] = "Approaching max_nodes limit"

        # Memory
        try:
            mem = psutil.virtual_memory()
            checks["memory"] = {
                "used_percent": mem.percent,
                "available_mb": round(mem.available / 1024 / 1024, 1),
            }
            if mem.percent > 90:
                status = "degraded"
                checks["memory"]["warning"] = "High memory usage"
        except Exception:
            checks["memory"] = {"error": "psutil unavailable"}

        # Disk
        try:
            disk = psutil.disk_usage(".")
            checks["disk"] = {"used_percent": round(disk.percent, 1)}
            if disk.percent > 90:
                status = "degraded"
                checks["disk"]["warning"] = "Low disk space"
        except Exception:
            checks["disk"] = {"error": "psutil unavailable"}

        # Embedder circuit breaker
        if hasattr(self, "_embedder_cb"):
            cb_state = self._embedder_cb.state.value
            checks["embedder_circuit_breaker"] = {"state": cb_state}
            if cb_state == "open":
                status = "degraded"

        # Metrics snapshot
        if self.metrics is not None:
            checks["metrics"] = self.metrics.snapshot()

        # Engram cache
        if self.engram_cache is not None:
            checks["engram_cache"] = {"size": len(self.engram_cache)}

        return {"status": status, "checks": checks}

    def take_snapshot(self) -> None:
        """Capture current memory state for future rollback."""
        self._rollback_manager.take_snapshot(self.field)
        logger.info("Memory snapshot taken")

    def rollback(self, timestamp: Optional[float] = None) -> bool:
        """Rollback memory to a previous snapshot."""
        success = self._rollback_manager.rollback(self.field, timestamp)
        if success:
            logger.info("Memory rolled back to snapshot")
        else:
            logger.warning("Rollback failed: no suitable snapshot")
        return success

    def detect_poisoned_memories(self) -> List[Dict]:
        """Scan for potentially poisoned or anomalous memory nodes."""
        return self._poison_detector.scan(self.field)

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

    def __getattr__(self, name: str):
        """Proxy simple delegations to RTMDKField to reduce boilerplate."""
        # Respect pydantic private/extra attributes first
        pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
        if pydantic_extra is not None and name in pydantic_extra:
            return pydantic_extra[name]
        # get_dashboard is a legacy alias for get_field_health
        if name == "get_dashboard":
            return self.field.get_field_health
        _proxy_methods = {
            "get_field_health", "trigger_healing",
            "counterfactual_query", "get_causal_summary",
            "evolve_continuous", "get_response_smoothness",
            "create_plan", "verify_hypothesis", "execute_tool",
            "register_tool", "evaluate_response", "compare_shadow",
            "get_cross_modal_stats", "get_meta_controller_state",
            "get_federated_status", "export_field", "import_field",
        }
        if name in _proxy_methods:
            return getattr(self.field, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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

    def get_ragas_trend(self) -> Dict[str, float]:
        if self.field.ragas_evaluator:
            return self.field.ragas_evaluator.get_trend()
        return {}

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        if self.field.dp:
            self.field.stats["privacy_budget_spent"] = self.field.dp.get_privacy_spent()
        return {**self.field.stats, "config": self.config.asdict()}

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self,
                               base_query: str,
                               intervention: Dict[str,
                                                  float]) -> List[Dict]:
        """Generate hypothetical scenarios."""
        embedding = self.embedder(base_query)
        return self.field.imagine_counterfactual(embedding, intervention)

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
