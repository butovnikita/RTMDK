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
from rtmdk.support.circuit_breaker import CircuitBreaker
from rtmdk.support.meta_controller import MetaController
from rtmdk.support.learnable import LearnableKernel, DifferentiableConsolidation
from rtmdk.support.torch_backend import TorchBackend
from rtmdk.support.projection import IncPCAProjection
from rtmdk.support.production import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
from rtmdk.memory.projection_manager import ProjectionManager
from rtmdk.memory.consolidation_manager import ConsolidationManager
from rtmdk.memory.scheduler import StepScheduler
from rtmdk.support.agents import AgentPlanner, HypothesisVerifier, ToolRouter
from rtmdk.support.healer import TopologyHealer
from rtmdk.support.meta_adaptive import MetaAdaptiveKernel
from rtmdk.engines.causal import CausalInferenceEngine
from rtmdk.engines.causal_extraction import extract_causal_edges_from_content
from rtmdk.engines.privacy import DifferentialPrivacy
from rtmdk.engines.predictive import PredictiveCodingModel
from rtmdk.memory.geometry import (
    poincare_dist, exp_map_poincare, log_map_poincare, poincare_midpoint,
)
from rtmdk.memory.quantization import QuantizationHelper
from rtmdk.memory.config import (
    ConsolidationMode, Backend, ContextFormat, FieldHealth, EvalMode,
    RTMDKConfig,
)
from rtmdk.nodes import (
    MemoryNode, CounterfactualResult,
)
import asyncio
import functools
import json
import math
import random
import threading
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from collections import deque
from typing import List, Dict, Optional, Tuple, Callable, Any, Set
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import cKDTree
import logging

# Extracted engine classes (kept in sync with rtmdk/support/ modules)
from rtmdk.support.kuramoto import FederatedRTMDK

try:
    from rtmdk.support.hnsw_lib import HNSWLibIndex
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

from rtmdk.memory.conformal import ConformalCalibrator
from rtmdk.memory.spectral import spectral_cluster_nodes
from rtmdk.memory.kalman import KalmanFilter
from rtmdk.engines.counterfactual import ScenarioPlanner
from rtmdk.support.goal_tracker import GoalTracker
from rtmdk.support.rl_feedback import RLFeedbackLoop
from rtmdk.support.event_driven import LowRankCompressor, EventDrivenScheduler
from rtmdk.support.meta_memory import MetaMemoryEvaluator
from rtmdk.support.security import SecurityValidator
from rtmdk.support.threshold import AdaptiveThreshold
from rtmdk.support.tda import TDAMonitor
from rtmdk.memory.resonance import ResonanceEngine
from rtmdk.memory.utils import SecurityViolationError

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


class RTMDKField:
    def __init__(
            self,
            config: RTMDKConfig,
            projection_matrix: Optional[NDArray] = None,
            wal_path: Optional[str] = None):
        self.cfg = config
        self._quant = QuantizationHelper(config.quantization)
        self._rng = np.random.default_rng(config.seed)
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_index: List[str] = []

        # ResonanceEngine — pure math extracted from field (created early so
        # subsystems can register themselves)
        self._resonance_engine = ResonanceEngine(
            cfg=config,
            meta_kernel=None,
            learnable_kernel=None,
            causal_engine=None,
            gpu_backend=None,
            quant=self._quant,
        )

        # Normalize projection_mode and latent_dim for identity
        if config.projection_mode == "identity":
            if config.latent_dim != config.embedding_dim:
                logger.warning(
                    f"projection_mode='identity' but latent_dim ({config.latent_dim}) != "
                    f"embedding_dim ({config.embedding_dim}). Aligning latent_dim to embedding_dim.")
                config.latent_dim = config.embedding_dim
                config.pca_n_components = config.latent_dim
        elif config.projection_mode == "pca":
            config.learn_projection = True

        # Track 2: Tiered Storage (Hot / Warm / Cold)
        self._tiered_store: Optional[Any] = None
        if config.tiered_storage_v2_enabled:
            from rtmdk.storage.tiered import TieredNodeStore
            from rtmdk.storage.tiered_adapter import TieredNodeStoreAdapter
            hot_limit = max(1, int(config.max_nodes *
                                   config.tiered_hot_pct)) if config.max_nodes else 100
            warm_limit = max(1, int(config.max_nodes *
                                    config.tiered_warm_pct)) if config.max_nodes else 1000
            cold_dir = config.tiered_storage_path or "./rtmdk_cold_storage_v2"
            inner = TieredNodeStore(
                max_hot=hot_limit, max_warm=warm_limit,
                cold_dir=cold_dir, latent_dim=config.latent_dim)
            self._tiered_store = TieredNodeStoreAdapter(inner)
            self.nodes = self._tiered_store  # type: ignore[assignment]
        elif config.tiered_storage_enabled:
            from rtmdk.memory.tiered_storage import TieredNodeStore
            hot_limit = max(1, int(config.max_nodes *
                                   config.tiered_hot_pct)) if config.max_nodes else 100
            warm_limit = max(1, int(config.max_nodes *
                                    config.tiered_warm_pct)) if config.max_nodes else 1000
            cold_dir = config.tiered_storage_path or "./rtmdk_cold_storage"
            self._tiered_store = TieredNodeStore(
                hot_limit, warm_limit, cold_dir, config.latent_dim)
            self.nodes = self._tiered_store  # type: ignore[assignment]

        # WAL for durability
        from rtmdk.memory.wal import WAL
        self.wal = WAL(
            wal_path,
            enabled=wal_path is not None,
            fsync_interval_ms=config.wal_fsync_interval_ms,
            batch_size=config.wal_batch_size,
        )

        # Phase 3d: Dirty flag for auto-save — only save if state changed
        self._dirty = False

        # P0: Cached numpy arrays for vectorized query — avoids O(N) Python
        # loop on every query
        from rtmdk.memory.cache_manager import NodeCacheManager
        self._cache_mgr = NodeCacheManager()

        # Track 3: Query cache
        self.query_cache: Optional[Any] = None
        if config.query_cache_size > 0:
            from rtmdk.production.query_cache import QueryCache
            self.query_cache = QueryCache(
                max_size=config.query_cache_size,
                ttl_seconds=config.query_cache_ttl)

        # P1.1: Conformal prediction calibrator
        self.conformal_calibrator: Optional[ConformalCalibrator] = None
        if config.conformal_prediction:
            self.conformal_calibrator = ConformalCalibrator(
                alpha=config.conformal_alpha)

        # P1.2: Learned consolidation MLP
        self.learned_consolidator = None
        if getattr(config, "learned_consolidation", False):
            from rtmdk.memory.learned_consolidation import LearnedConsolidator
            self.learned_consolidator = LearnedConsolidator(
                latent_dim=config.latent_dim)

        # P1.3: Adaptive bandwidth optimiser
        self.adaptive_bw = None
        if getattr(config, "adaptive_bandwidth", False):
            from rtmdk.support.adaptive_bandwidth import AdaptiveBandwidthOptimizer
            self.adaptive_bw = AdaptiveBandwidthOptimizer(
                latent_dim=config.latent_dim)

        # P1.4: Adaptive phase coupling
        self._adaptive_pc_value: Optional[float] = None
        self._adaptive_pc_estimated: bool = False
        if getattr(config, "adaptive_phase_coupling", False):
            from rtmdk.memory.adaptive_pc import estimate_optimal_pc
            self._estimate_optimal_pc_fn = estimate_optimal_pc
        else:
            self._estimate_optimal_pc_fn = None

        # P2.2: Kalman filter for position uncertainty
        self.kalman_filter: Optional[KalmanFilter] = None
        if config.enable_kalman_filter:
            self.kalman_filter = KalmanFilter(
                latent_dim=config.latent_dim,
                process_noise=config.kalman_process_noise,
                measurement_noise=config.kalman_measurement_noise,
                init_variance=config.kalman_init_variance,
                diagonal_approx=config.kalman_diagonal_approx,
                hyperbolic=config.hyperbolic,
                ball_radius=config.ball_radius,
            )

        self._projection_mgr = ProjectionManager(
            config, projection_matrix=projection_matrix, rng=self._rng)

        self.adaptive_threshold = AdaptiveThreshold(
            config.adaptive_window,
            config.tension_threshold) if config.adaptive_threshold else None
        self.tda_monitor = TDAMonitor() if config.tda_monitoring else None
        self.gpu_backend = TorchBackend() if config.backend == Backend.TORCH else None
        if self.gpu_backend and not self.gpu_backend.available:
            self.gpu_backend = None
        self._resonance_engine.gpu_backend = self.gpu_backend

        from rtmdk.memory.index_manager import IndexManager
        self._index_mgr = IndexManager(config, config.latent_dim, self._rng, self._quant)
        # Backward-compatible aliases
        self.bm25_index = self._index_mgr.bm25_index
        self.hnsw_index = self._index_mgr.hnsw_index
        self.shard_centers = self._index_mgr.shard_centers
        self._async_index_builder = self._index_mgr._async_builder

        # Pre-select batch resonance backend to avoid branching in hot path
        if self.gpu_backend and self.gpu_backend.available:
            self._batch_resonance_fn = self._batch_resonance_torch
        else:
            self._batch_resonance_fn = self._batch_resonance_numpy

        self.learnable_kernel: Optional[LearnableKernel] = None
        self.diff_consolidation: Optional[DifferentiableConsolidation] = None
        if config.differentiable:
            self.learnable_kernel = LearnableKernel(
                config.bandwidth,
                config.phase_coupling,
                config.decay_rate,
                config.gradient_clip)
            self.diff_consolidation = DifferentiableConsolidation(
                config.consolidation_loss_weight)
            self._resonance_engine.learnable_kernel = self.learnable_kernel

        self.monitor: Optional[Any] = None

        self.meta_kernel: Optional[MetaAdaptiveKernel] = None
        if config.meta_adaptive:
            self.meta_kernel = MetaAdaptiveKernel(
                config.bandwidth,
                config.phase_coupling,
                config.meta_adaptation_lr,
                config.kurtosis_target_min,
                config.kurtosis_target_max)
            self._resonance_engine.meta_kernel = self.meta_kernel

        self.healer: Optional[TopologyHealer] = None
        if config.self_healing:
            self.healer = TopologyHealer(
                config.dead_zone_threshold,
                config.hyperconvergence_threshold,
                config.fragmentation_threshold,
                config.healing_strength,
                config.max_healing_nodes_per_step)

        # B2: Lazy module initialization — store flags but don't instantiate
        # yet
        self._causal_engine: Optional[CausalInferenceEngine] = None
        self._causal_engine_initialized = config.causal_topological

        # Track 10.2: Meta-controller — B2 lazy init
        self._meta_controller: Optional[MetaController] = None
        self._meta_controller_initialized = config.meta_controller

        self.agent_planner: Optional[AgentPlanner] = None
        self.hypothesis_verifier: Optional[HypothesisVerifier] = None
        self.tool_router: Optional[ToolRouter] = None
        if config.agent_orchestration:
            self.agent_planner = AgentPlanner(
                config.max_plan_depth,
                config.max_tool_calls,
                config.tool_timeout)
            self.hypothesis_verifier = HypothesisVerifier(
                config.verification_confidence_threshold)
            self.tool_router = ToolRouter(config.tool_timeout)

        self.shadow_evaluator: Optional[ShadowModeEvaluator] = None
        self.ragas_evaluator: Optional[RAGASPlusEvaluator] = None
        self.rollback_manager: Optional[AutoRollbackManager] = None
        if config.production_mode:
            if config.shadow_mode:
                self.shadow_evaluator = ShadowModeEvaluator(
                    config.shadow_fallback_threshold)
            if config.ragas_enabled:
                self.ragas_evaluator = RAGASPlusEvaluator()
            if config.auto_rollback:
                self.rollback_manager = AutoRollbackManager(
                    config.auto_rollback_threshold)

        # Track 10.3: Federated
        self.federated: Optional[FederatedRTMDK] = None
        if config.federated:
            self.federated = FederatedRTMDK(
                node_id=config.node_id,
                sync_lr=config.federated_sync_lr,
                sync_freq=config.federated_sync_freq,
                min_resonance=config.federated_min_resonance,
            )

        # Phase 11 Track 3: Predictive coding
        self.predictor: Optional[PredictiveCodingModel] = None
        if config.predictive_coding:
            self.predictor = PredictiveCodingModel(
                config.latent_dim, lr=config.pc_lr)
        self._state_history: deque = deque(maxlen=100)

        # Phase 11 Track 4: Counterfactual imagination
        self.scenario_planner: Optional[ScenarioPlanner] = None
        if config.counterfactual_imagination:
            self.scenario_planner = ScenarioPlanner(
                self, max_scenarios=config.max_scenarios)

        # Phase 11 Track 5: Differential privacy
        self.dp: Optional[DifferentialPrivacy] = None
        if config.differential_privacy:
            self.dp = DifferentialPrivacy(
                config.dp_epsilon, config.dp_delta, config.dp_max_norm)

        # Phase 12 Track 1: Sparse resonant routing (MoE-memory)
        self.shard_router: Optional[NDArray] = None
        self._node_shard_map: Dict[str, int] = {}
        if config.sparse_routing:
            self.shard_router = np.zeros(config.num_shards, dtype=np.float32)

        # Phase 12 Track 3: Crystallization
        self._crystallization_counter = 0
        self._crystallized_nodes: Set[str] = set()

        # Fix 3: Lifecycle & Throttling Controls
        self._workers: List[asyncio.Task] = []
        self._write_lock = threading.RLock()
        self._consolidation_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rtmdk_consolidate")
        self._consolidation_future = None
        self._backpressure_events = 0
        self._heavy_modules_degraded = False  # Track if we've entered degraded mode
        self._last_successful_step = time.time()  # For recovery tracking

        # P1: Per-subsystem circuit breakers (replaces _safe_run catch-all)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "ODEEvolve": CircuitBreaker("ODEEvolve", default=0),
            "Consolidate": CircuitBreaker("Consolidate", default=[]),
            "SelfHeal": CircuitBreaker("SelfHeal"),
            "PredictorFreeEnergy": CircuitBreaker("PredictorFreeEnergy", default=0.0),
            "PredictorUpdate": CircuitBreaker("PredictorUpdate"),
            "SelfSupervise": CircuitBreaker("SelfSupervise"),
            "TDA": CircuitBreaker("TDA"),
            "MetaKernelAdapt": CircuitBreaker("MetaKernelAdapt"),
            "MetaControllerOptimize": CircuitBreaker("MetaControllerOptimize", default={}),
            "MetaControllerApply": CircuitBreaker("MetaControllerApply"),
            "FederatedSync": CircuitBreaker("FederatedSync"),
            "ODESmoothness": CircuitBreaker("ODESmoothness", default=1.0),
            "ShardUpdate": CircuitBreaker("ShardUpdate"),
        }

        # B1: Tension caching
        # node_id -> (tension, step)
        self._tension_cache: Dict[str, Tuple[float, float]] = {}
        self._tension_cache_max_age = 25  # steps — covers multiple consolidation cycles
        self._tension_cache_hits = 0
        self._tension_cache_misses = 0

        # Phase 12 Track 4: Async pipeline queues
        self.query_q: Optional[asyncio.Queue] = None
        self.save_q: Optional[asyncio.Queue] = None
        self.evolve_q: Optional[asyncio.Queue] = None
        self._workers_started = False
        if config.async_pipeline:
            self.query_q = asyncio.Queue(maxsize=config.query_queue_size)
            self.save_q = asyncio.Queue(maxsize=config.save_queue_size)
            self.evolve_q = asyncio.Queue(maxsize=config.evolve_queue_size)

        # Phase 13 Track 1: Teleological layer
        self.goal_tracker: Optional[GoalTracker] = None
        if config.goal_tracking:
            self.goal_tracker = GoalTracker(
                config.max_goals,
                config.goal_decay,
                config.goal_completion_threshold)

        # Phase 13 Track 3: RL feedback loop
        self.rl_feedback_loop: Optional[RLFeedbackLoop] = None
        if config.rl_feedback:
            self.rl_feedback_loop = RLFeedbackLoop(
                config.rl_learning_rate, config.rl_reward_window
            )

        # Phase 13 Track 4: Event-driven + Low-Rank
        self.event_scheduler: Optional[EventDrivenScheduler] = None
        self.low_rank_compressor: Optional[LowRankCompressor] = None
        if config.event_driven:
            self.event_scheduler = EventDrivenScheduler()
        if config.low_rank_compression:
            self.low_rank_compressor = LowRankCompressor(
                config.compression_rank)

        # Phase 18: Engram Manager (Fix 4: ensure attribute always exists even
        # when disabled)
        self.engram_manager: Optional[Any] = None
        if config.enable_engrams:
            try:
                from rtmdk.engrams import EngramManager
                self.engram_manager = EngramManager(
                    min_nodes=config.engram_min_nodes,
                    max_nodes=config.engram_max_nodes,
                    creation_threshold=config.engram_creation_threshold,
                    decay_rate=config.engram_decay_rate,
                    pattern_completion=config.engram_pattern_completion,
                    overlap_threshold=config.engram_overlap_threshold,
                )
            except Exception:
                logger.warning(
                    "Engram manager initialization failed in RTMDKField, disabling",
                    exc_info=True)
                self.engram_manager = None

        # Phase 14 Track 1: Meta-Memory
        self.meta_memory_eval: Optional[MetaMemoryEvaluator] = None
        if config.meta_memory:
            self.meta_memory_eval = MetaMemoryEvaluator(
                config.recall_accuracy_threshold, config.memory_age_factor,
                config.self_reflection_freq
            )

        # Phase 14 Track 2: Security
        self.security: Optional[SecurityValidator] = None
        if config.security_enabled:
            self.security = SecurityValidator(
                config.max_node_text_length, config.tension_spike_threshold,
                config.prompt_injection_patterns
            )

        # Phase 15 Track 1: Version Control (Memory Git)
        self.version_control: Optional["VersionControl"] = None
        if config.version_control and VC_AVAILABLE:
            self.version_control = VersionControl(
                max_versions=config.max_versions)
        elif config.version_control and not VC_AVAILABLE:
            logger.error(
                "version_control enabled but rtmdk.support.version_control not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append(
                "version_control unavailable")

        # Phase 17: RoleShardRouter
        self.role_router: Optional["RoleShardRouter"] = None
        if config.role_sharding and ROLE_SHARD_AVAILABLE:
            self.role_router = RoleShardRouter(
                shards=config.role_shards,
                cross_shard_threshold=config.cross_shard_threshold,
                auto_role_detection=config.auto_role_detection,
            )
        elif config.role_sharding and not ROLE_SHARD_AVAILABLE:
            logger.error(
                "role_sharding enabled but rtmdk.support.role_shard_router not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append(
                "role_shard_router unavailable")

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
            "ode_steps": 0, "response_smoothness": 1.0,
            "plans_created": 0, "hypotheses_verified": 0, "tool_calls": 0, "tool_misuse_rate": 0.0,
            "evaluations": 0, "shadow_comparisons": 0, "rollbacks": 0,
            "ragas_overall": 0.0,
            "cross_modal_queries": 0, "cross_modal_recall": 0.0,
            "meta_optimizations": 0, "meta_best_params": {},
            "federated_syncs": 0, "federated_order_parameter": 0.0,
            # Phase 11
            "tier_distribution": {}, "tier_coherence": 0.0,
            "hyperbolic_enabled": config.hyperbolic, "avg_hyperbolic_dist": 0.0,
            "free_energy": 0.0, "prediction_error": 0.0, "surprise_level": 0.0,
            "scenarios_generated": 0, "avg_scenario_confidence": 0.0,
            "privacy_budget_spent": 0.0, "noise_std": 0.0, "updates_clipped": 0,
            # Phase 12
            "shard_hits": 0, "shard_misses": 0, "avg_shard_query_time_ms": 0.0,
            "context_tokens_saved": 0, "cognitive_compressions": 0,
            "crystallizations": 0, "crystallized_clusters": 0,
            "async_queue_depth": 0, "async_backpressure_events": 0,
            # Phase 13
            "active_goals": 0, "completed_goals": 0,
            "avg_rl_reward": 0.5, "reward_trend": 0.0,
            "attention_bias_applied": 0,
            "compression_ratio": 1.0, "compression_updates": 0,
            "events_processed": 0, "event_queue_depth": 0,
            # Phase 14
            "recall_accuracy": 1.0, "meta_reflections": 0,
            "security_violations": 0, "tension_spikes_blocked": 0,
            # Phase 15
            "current_version": 0, "n_versions": 0,
            "clarifications_generated": 0,
            # Phase 17
            "n_shards": 0, "shard_distribution": {},
            "cross_shard_exchanges": 0, "role_router_enabled": False,
            # Phase 18: Engrams
            "engram_retrievals": 0, "engrams_created": 0, "engrams_merged": 0,
            "field_integrity_issues": 0,
            "backpressure_degraded_mode": 0, "last_backpressure_recovery": 0.0,
            # Fix 10: Track startup warnings for missing optional dependencies
            "startup_warnings": [],
            # B1: Tension cache stats
            "tension_cache_hits": 0, "tension_cache_misses": 0,
            "tension_cache_hit_rate": 0.0,
            # P1.1: Conformal prediction stats
            "conformal_threshold": 0.0,
            "conformal_confidence": 0.0,
            "conformal_prediction_set_size": 0,
        }
        self._step_counter = 0
        # Rate limiting: track add_node timestamps (max 100 nodes/sec)
        self._add_node_timestamps: deque = deque(maxlen=1000)
        self._rollback_history: deque = deque(
            maxlen=config.max_rollback_history)
        self._stability_buffer: deque = deque(
            maxlen=config.field_stability_window)
        self._active_node_history: deque = deque(maxlen=50)
        self._semantic_phase_cache: Dict[str, float] = {}

        self._consolidation_mgr = ConsolidationManager(self)
        self._scheduler = StepScheduler(self)

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

    def _ensure_cache(self) -> None:
        """Lazy cache rebuild if dirty."""
        self._cache_mgr.ensure_built(self)

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

    @property
    def _effective_bandwidth(self) -> float:
        """Return adaptive bandwidth if available, else config bandwidth."""
        if self.adaptive_bw is not None and self.adaptive_bw._best_bw is not None:
            return self.adaptive_bw._best_bw
        return self.cfg.bandwidth

    @property
    def _effective_pc(self) -> float:
        """Return adaptive phase coupling if estimated, else config value."""
        if self._adaptive_pc_value is not None:
            return self._adaptive_pc_value
        if self.meta_kernel is not None:
            return self.meta_kernel.get_phase_coupling()
        return self.cfg.phase_coupling

    def _ensure_adaptive_pc(self, query_latent: NDArray) -> None:
        """Run once on first query to auto-tune phase coupling."""
        if self._adaptive_pc_estimated or self._estimate_optimal_pc_fn is None:
            return
        if len(self.nodes) < 10:
            return
        try:
            # Build normalized node position matrix
            nids = list(self.node_index)
            positions = np.array([self.nodes[nid].latent_pos for nid in nids])
            norms = np.linalg.norm(positions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            doc_embs = positions / norms
            q_emb = query_latent / max(np.linalg.norm(query_latent), 1e-8)
            # Quick self-test: sample some nodes as pseudo-queries
            sample_size = min(50, len(nids))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(nids), size=sample_size, replace=False)
            sample_queries = doc_embs[idx]
            sample_targets = idx
            # Estimate embedding quality: cosine recall@1 on pseudo-queries
            sims = sample_queries @ doc_embs.T
            np.fill_diagonal(sims, -1.0)  # exclude self
            ranks = np.argmax(sims, axis=1)
            hits = np.sum(ranks == sample_targets)
            recall1 = hits / len(sample_targets)
            # If embeddings are strong, phase coupling adds no value → disable
            threshold = getattr(self.cfg, "adaptive_pc_disable_threshold", 0.93)
            if recall1 >= threshold:
                self._adaptive_pc_value = 0.0
                logger.info(
                    "Adaptive PC: embeddings strong (recall@1=%.2f ≥ %.2f) → pc=0.0",
                    recall1, threshold)
            else:
                pc = self._estimate_optimal_pc_fn(
                    doc_embs, sample_queries, sample_targets, sample_size=sample_size)
                self._adaptive_pc_value = float(pc)
                logger.info("Adaptive phase coupling estimated: pc=%.2f", self._adaptive_pc_value)
        except Exception as exc:
            logger.warning("Adaptive PC estimation failed: %s", exc)
        finally:
            self._adaptive_pc_estimated = True

    def _resonance_response(
            self,
            query_latent: NDArray,
            query_phase: float,
            node: MemoryNode,
            query_modality: str = "text") -> float:
        # Fix 1: Torch backend auto-switch for batch resonance
        # (Single-node response always uses numpy for simplicity;
        #  batch queries use TorchBackend.batch_resonance via query())
        resp = self._resonance_engine.single_response(
            query_latent, query_phase, node, query_modality)
        # Backward-compat: update hyperbolic distance stat
        if self.cfg.hyperbolic:
            dist = poincare_dist(
                query_latent, node.latent_pos, self.cfg.ball_radius)
            self.stats["avg_hyperbolic_dist"] = 0.99 * \
                self.stats["avg_hyperbolic_dist"] + 0.01 * dist
        return resp

    def _batch_resonance(self, query_latents: NDArray, query_phases: NDArray,
                         node_ids: List[str]) -> NDArray:
        """Batch resonance computation. Pre-selected backend avoids hot-path branching."""
        return self._batch_resonance_fn(query_latents, query_phases, node_ids)

    def _batch_resonance_nodes(
            self,
            query_latents: NDArray,
            query_phases: NDArray,
            nodes: List[Any]) -> NDArray:
        """Batch resonance over a pre-materialized list of MemoryNode objects.

        Avoids self.nodes[nid] lookups — critical for tiered storage where
        __getitem__ triggers promotion/demotion.
        """
        if not nodes:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        node_positions = np.array([n.latent_pos for n in nodes])
        node_phases = np.array([n.phase for n in nodes])
        node_amplitudes = np.array([n.amplitude for n in nodes])
        node_saliences = np.array([n.salience for n in nodes])
        return self._resonance_engine.batch_response_numpy(
            query_latents, query_phases,
            node_positions, node_phases, node_amplitudes, node_saliences)

    def _batch_resonance_numpy(
            self,
            query_latents: NDArray,
            query_phases: NDArray,
            node_ids: List[str]) -> NDArray:
        """Pure numpy batch resonance — no branching, no torch overhead."""
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        node_positions = np.array(
            [self.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([self.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array(
            [self.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array(
            [self.nodes[nid].salience for nid in node_ids])

        return self._resonance_engine.batch_response_numpy(
            query_latents, query_phases,
            node_positions, node_phases, node_amplitudes, node_saliences)

    def _batch_resonance_cached(
            self,
            query_latents: NDArray,
            query_phases: NDArray,
            node_ids: List[str]) -> NDArray:
        """Batch resonance using pre-cached numpy arrays — O(1) lookup per node.

        Avoids expensive self.nodes[nid] Python dict lookups by mapping
        node_ids to cached array indices.
        """
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        # Build cache if missing mapping (should not happen in steady state)
        if getattr(self, '_node_id_to_cached_idx', None) is None or self._cache_dirty:
            self._build_node_cache()
        mapping = self._node_id_to_cached_idx
        indices = np.array([mapping[nid] for nid in node_ids if nid in mapping],
                           dtype=np.int32)
        if len(indices) == 0:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        return self._resonance_engine.batch_response_numpy(
            query_latents, query_phases,
            self._cached_positions[indices],
            self._cached_phases[indices],
            self._cached_amplitudes[indices],
            self._cached_saliences[indices])

    def _batch_resonance_torch(
            self,
            query_latents: NDArray,
            query_phases: NDArray,
            node_ids: List[str]) -> NDArray:
        """Torch batch resonance — GPU accelerated."""
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        node_positions = np.array(
            [self.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([self.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array(
            [self.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array(
            [self.nodes[nid].salience for nid in node_ids])

        return self._resonance_engine.batch_response_torch(
            query_latents, query_phases,
            node_positions, node_phases,
            node_amplitudes, node_saliences)

    def _compute_resonance_chunk(
            self,
            positions,
            phases,
            amplitudes,
            saliences,
            modal_weights,
            gates,
            causal_boost,
            query_latent,
            query_phase,
            bw=None,
            pc=None):
        """Compute resonance response for a chunk of nodes.

        Args:
            bw: Optional per-node bandwidth vector (same length as positions).
                If None, uses global bandwidth from config or meta_kernel.
            pc: Optional phase coupling. If None, uses _effective_pc.
        """
        return self._resonance_engine.chunk_response(
            positions, phases, amplitudes, saliences,
            modal_weights, gates, causal_boost,
            query_latent, query_phase, bw,
            use_gates=self.cfg.soft_gates,
            use_causal=self.causal_engine is not None,
            pc=pc)

    def _query_vectorized(self, query_latent: NDArray, query_phase: float,
                          top_k: int, modality: str, session_id: Optional[str],
                          t0: float) -> List[Tuple[str, float, MemoryNode]]:
        """Vectorized query using cached numpy arrays — O(N) but vectorized.

        Mathematical model:
        - dist_i = ||q - n_i|| for all i → vectorized norm
        - spatial_i = exp(-dist_i² / 2bw²) → vectorized exp
        - phase_align_i = 0.5 + 0.5*cos(phase_i - query_phase) → vectorized cos
        - resp_i = spatial_i × ((1-pc) + pc × phase_align_i) × amp_i × sal_i
        - Session boost: resp_i × 1.5 if session matches
        - Filter: resp_i >= min_response
        - Sort and return top_k

        Complexity: O(N×d) with SIMD vectorization (~200x faster than Python loop)
        Cached arrays avoid O(N) Python loop on every query.
        """
        cfg = self.cfg
        min_response = cfg.min_response
        gpu_batch_size = cfg.gpu_batch_size
        attention_bias = getattr(cfg, "attention_bias", False)
        bias_temperature = getattr(cfg, "bias_temperature", 1.0)
        bw = cfg.bandwidth
        pc = float(self._resonance_engine._effective_pc)
        use_causal = self.causal_engine is not None

        self._ensure_adaptive_pc(query_latent)

        n_nodes = len(self.node_index)
        if n_nodes == 0:
            return []

        # Build cache if dirty (single pass through nodes)
        if self._cache_dirty or self._cached_positions is None:
            self._build_node_cache()

        # P1: Session pre-filtering — build mask once, apply to all arrays
        session_mask = None
        if session_id and session_id != "default":
            session_mask = np.array([
                self.nodes[nid].content.get("session") == session_id
                for nid in self.node_index
            ], dtype=bool)
            # If very few session nodes, use them directly
            n_session = session_mask.sum()
            if 0 < n_session < n_nodes * 0.3:
                # Session has < 30% of nodes — filter arrays
                positions = self._cached_positions[session_mask]
                phases = self._cached_phases[session_mask]
                amplitudes = self._cached_amplitudes[session_mask]
                saliences = self._cached_saliences[session_mask]
                modal_weights = self._cached_modal_weights[session_mask]
                gates = self._cached_gates[session_mask]
                causal_boost = self._cached_causal_boost[session_mask]
                session_indices = np.where(session_mask)[0]
            else:
                # Session has many nodes — use full arrays with boost
                positions = self._cached_positions
                phases = self._cached_phases
                amplitudes = self._cached_amplitudes
                saliences = self._cached_saliences
                modal_weights = self._cached_modal_weights
                gates = self._cached_gates
                causal_boost = self._cached_causal_boost
                session_indices = None
        else:
            positions = self._cached_positions
            phases = self._cached_phases
            amplitudes = self._cached_amplitudes
            saliences = self._cached_saliences
            modal_weights = self._cached_modal_weights
            gates = self._cached_gates
            causal_boost = self._cached_causal_boost
            session_indices = None

        # Phase 3c: Chunked batch computation — prevents OOM and improves cache
        # locality
        batch_size = gpu_batch_size
        n = len(positions)

        if n <= batch_size:
            # Single chunk — old fast path
            resp = self._compute_resonance_chunk(
                positions, phases, amplitudes, saliences,
                modal_weights, gates, causal_boost,
                query_latent, query_phase,
                bw=bw, pc=pc,
            )
            if session_id and session_id != "default" and session_mask is not None and session_indices is None:
                resp = resp * (1.0 + 0.5 * session_mask.astype(np.float32))
            above_threshold = resp >= min_response
            indices = np.where(above_threshold)[0]
            if len(indices) == 0:
                self.stats["total_queries"] += 1
                return []
            if session_indices is not None:
                indices = session_indices[indices]
            scores = resp[indices] if session_indices is None else resp[np.where(above_threshold)[
                0]]
            n_results = min(len(indices), top_k * 2)
            if len(indices) > top_k * 3:
                if n_results < len(scores):
                    partition_idx = np.argpartition(
                        scores, -n_results)[-n_results:]
                    top_local = partition_idx[np.argsort(
                        scores[partition_idx])[::-1][:top_k]]
                else:
                    top_local = np.argsort(scores)[::-1][:top_k]
                top_indices = indices[top_local]
                top_scores = scores[top_local]
            else:
                sorted_order = np.argsort(scores)[::-1][:top_k]
                top_indices = indices[sorted_order]
                top_scores = scores[sorted_order]
        else:
            # Multi-chunk path — accumulate local top_k then global sort
            candidates: List[Tuple[int, float]] = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                resp = self._compute_resonance_chunk(
                    positions[start:end], phases[start:end], amplitudes[start:end],
                    saliences[start:end], modal_weights[start:end], gates[start:end],
                    causal_boost[start:end], query_latent, query_phase,
                    bw=bw, pc=pc,
                )
                if session_id and session_id != "default" and session_mask is not None and session_indices is None:
                    resp = resp * \
                        (1.0 + 0.5 * session_mask[start:end].astype(np.float32))
                above = resp >= min_response
                local_idx = np.where(above)[0]
                if len(local_idx) == 0:
                    continue
                scores = resp[local_idx]
                local_idx += start
                chunk_n = min(len(local_idx), top_k * 2)
                if len(local_idx) > top_k * 3:
                    if chunk_n < len(scores):
                        part_idx = np.argpartition(scores, -chunk_n)[-chunk_n:]
                        top_local = part_idx[np.argsort(
                            scores[part_idx])[::-1][:top_k]]
                    else:
                        top_local = np.argsort(scores)[::-1][:top_k]
                else:
                    top_local = np.argsort(scores)[::-1][:top_k]
                for li in top_local:
                    candidates.append((int(local_idx[li]), float(scores[li])))
            if not candidates:
                self.stats["total_queries"] += 1
                return []
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_k]
            if session_indices is not None:
                top_indices = np.array([session_indices[idx]
                                       for idx, _ in top_candidates], dtype=np.int64)
            else:
                top_indices = np.array(
                    [idx for idx, _ in top_candidates], dtype=np.int64)
            top_scores = np.array(
                [score for _, score in top_candidates], dtype=np.float32)

        # Build result list
        results = []
        for i in range(len(top_indices)):
            idx = top_indices[i]
            nid = self.node_index[idx]
            node = self.nodes[nid]
            node.last_resonated = time.time()
            results.append((nid, float(top_scores[i]), node))

        # Update stats
        self.stats["total_queries"] += 1
        if results:
            self.stats["avg_response"] = 0.9 * \
                self.stats["avg_response"] + 0.1 * results[0][1]
            if self.goal_tracker:
                for nid, resp_val, node in results:
                    node.goal_relevance = self.goal_tracker.get_goal_relevance(
                        nid)
            if attention_bias:
                from rtmdk.memory.utils import apply_attention_bias
                results = apply_attention_bias(
                    results, bias_temperature)
                self.stats["attention_bias_applied"] += 1

        # Track timing
        elapsed_ms = (time.time() - t0) * 1000
        if self.cfg.sparse_routing:
            self.stats["avg_shard_query_time_ms"] = (
                0.95 * self.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms)

        return results

    def _query_cache_key(
            self,
            query_latent: NDArray,
            phase: float,
            top_k: int,
            modality: str,
            session_id: Optional[str]) -> str:
        """Hash embedding + query params for cache key."""
        import hashlib
        # Round to fp16 precision to tolerate tiny float noise
        vec = query_latent.astype(np.float16).tobytes()
        raw = vec + \
            f"|{phase:.4f}|{top_k}|{modality}|{session_id or ''}".encode()
        return hashlib.md5(raw).hexdigest()

    def _apply_adaptive_top_k(self,
                              results: List[Tuple[str,
                                                  float,
                                                  MemoryNode]]) -> List[Tuple[str,
                                                                              float,
                                                                              MemoryNode]]:
        """Reduce top_k when top result is highly confident."""
        if not results:
            return results
        top_score = results[0][1]
        if top_score >= 0.95:
            return results[:1]
        elif top_score >= 0.80:
            return results[:3]
        else:
            return results[:5]

    def query_batch(
        self,
        embeddings: NDArray,
        phase: float = 0.0,
        top_k: Optional[int] = None,
        modality: str = "text",
        session_id: Optional[str] = None,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float, MemoryNode]]]:
        """Batch query — vectorized resonance for multiple queries at once.

        Returns a list of result lists, one per query.
        Uses _batch_resonance for efficient SIMD computation.
        """
        t0 = time.time()
        top_k = top_k or self.cfg.top_k
        n_queries = len(embeddings)

        # Project all embeddings at once
        query_latents = np.array([self._project(e) for e in embeddings])
        for ql in query_latents:
            self._ensure_adaptive_pc(ql)

        # Build cache if dirty
        if self._cache_dirty or self._cached_positions is None:
            self._build_node_cache()

        n_nodes = len(self.node_index)
        if n_nodes == 0:
            return [[] for _ in range(n_queries)]

        # Batch resonance over ALL nodes for ALL queries
        query_phases = np.full(n_queries, phase, dtype=np.float32)
        all_scores = self._batch_resonance(
            query_latents, query_phases, self.node_index
        )  # shape: (n_queries, n_nodes)

        # Apply session boost and threshold filter per query
        results_per_query: List[List[Tuple[str, float, MemoryNode]]] = []
        for qi in range(n_queries):
            scores = all_scores[qi]
            if session_id and session_id != "default":
                session_boosts = np.array([
                    1.3 if self.nodes[nid].content.get("session") == session_id else 1.0
                    for nid in self.node_index
                ], dtype=np.float32)
                scores = scores * session_boosts

            above = scores >= self.cfg.min_response
            indices = np.where(above)[0]
            if len(indices) == 0:
                results_per_query.append([])
                continue

            filtered_scores = scores[indices]
            n_results = min(len(indices), top_k * 2)
            if len(indices) > top_k * 3:
                partition_idx = np.argpartition(filtered_scores, -n_results)[-n_results:]
                top_local = partition_idx[np.argsort(filtered_scores[partition_idx])[::-1][:top_k]]
            else:
                top_local = np.argsort(filtered_scores)[::-1][:top_k]

            top_indices = indices[top_local]
            top_scores = filtered_scores[top_local]

            query_results = []
            for idx, score in zip(top_indices, top_scores):
                nid = self.node_index[idx]
                node = self.nodes[nid]
                node.last_resonated = time.time()
                query_results.append((nid, float(score), node))
            results_per_query.append(query_results)

        self.stats["total_queries"] += n_queries
        for results in results_per_query:
            if results:
                self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]

        if self.cfg.sparse_routing:
            elapsed_ms = (time.time() - t0) * 1000
            self.stats["avg_shard_query_time_ms"] = (
                0.95 * self.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms)

        return results_per_query

    def query(self,
              embedding: NDArray,
              phase: float = 0.0,
              top_k: Optional[int] = None,
              modality: str = "text",
              session_id: Optional[str] = None,
              query_text: Optional[str] = None) -> List[Tuple[str,
                                                              float,
                                                              MemoryNode]]:
        t0 = time.time()
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        query_latent = self._project(embedding)
        self._ensure_adaptive_pc(query_latent)

        # Track 3: Query cache check
        if self.query_cache is not None:
            cache_key = self._query_cache_key(
                query_latent, phase, top_k, modality, session_id)
            cached = self.query_cache.get_raw(cache_key)
            if cached is not None:
                self.stats.setdefault("query_cache_hits", 0)
                self.stats["query_cache_hits"] += 1
                return cached
            self.stats.setdefault("query_cache_misses", 0)
            self.stats["query_cache_misses"] += 1

        # P0: BM25 first-stage pre-filtering — use BM25 to get top-K candidates,
        # then score only those with resonance. 10-100x faster than full scan.
        if (cfg.bm25_first_stage_k > 0 and
                query_text and
                self.bm25_index is not None and
                len(self.nodes) > cfg.bm25_first_stage_k):
            candidate_ids = [
                nid for nid, _ in self._index_mgr.bm25_search(query_text, cfg.bm25_first_stage_k)
                if nid in self.nodes
            ]
            if candidate_ids:
                scores = self._batch_resonance(
                    query_latent[np.newaxis, :],
                    np.array([phase], dtype=np.float32),
                    candidate_ids,
                )[0]
                results = []
                for idx, nid in enumerate(candidate_ids):
                    node = self.nodes[nid]
                    resp = float(
                        scores[idx]) * (1.3 if session_id and node.content.get("session") == session_id else 1.0)
                    if resp >= cfg.min_response:
                        results.append((nid, resp, node))
                        node.last_resonated = time.time()
                results.sort(key=lambda x: x[1], reverse=True)
                self.stats["bm25_first_stage_hits"] = self.stats.get("bm25_first_stage_hits", 0) + 1
            else:
                results = []
        # Fix 1: HNSW auto-intercept for large N (>50 nodes).
        # For small datasets, full vectorized scan is more accurate and still
        # fast (SIMD).
        elif cfg.use_hnsw:
            candidate_ids: List[str] = []
            n_pos = self._index_mgr.hnsw_count()
            if n_pos > getattr(self.cfg, "hnsw_min_nodes", 50):
                hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
                candidate_ids = self._index_mgr.hnsw_search(query_latent, hnsw_k)
                candidate_ids = [nid for nid in candidate_ids if nid in self.nodes]
            # Vectorized batch resonance on HNSW candidates using cached arrays
            if candidate_ids:
                scores = self._batch_resonance_cached(
                    query_latent[np.newaxis, :],
                    np.array([phase], dtype=np.float32),
                    candidate_ids,
                )[0]
                # Vectorized session boost and threshold filter
                if session_id and session_id != "default":
                    session_boosts = np.array([
                        1.3 if self.nodes[nid].content.get("session") == session_id else 1.0
                        for nid in candidate_ids
                    ], dtype=np.float32)
                    scores = scores * session_boosts
                above = scores >= cfg.min_response
                indices = np.where(above)[0]
                if len(indices) == 0:
                    results = []
                else:
                    filtered_scores = scores[indices]
                    filtered_ids = [candidate_ids[i] for i in indices]
                    n_results = min(len(filtered_scores), top_k)
                    if len(filtered_scores) > top_k * 2:
                        partition_idx = np.argpartition(filtered_scores, -n_results)[-n_results:]
                        top_local = partition_idx[np.argsort(filtered_scores[partition_idx])[::-1]]
                    else:
                        top_local = np.argsort(filtered_scores)[::-1]
                    top_local = top_local[:top_k]
                    results = []
                    for ti in top_local:
                        nid = filtered_ids[ti]
                        node = self.nodes[nid]
                        node.last_resonated = time.time()
                        results.append((nid, float(filtered_scores[ti]), node))
            elif n_pos <= getattr(self.cfg, "hnsw_min_nodes", 50):
                # Small dataset — full vectorized scan is more accurate and still fast
                results = self._query_vectorized(
                    query_latent, phase, top_k, modality, session_id, t0)
            else:
                results = []
        elif cfg.sparse_routing and self._index_mgr.shard_centers is not None and len(self.nodes) > cfg.num_shards * 2:
            active_shards = self._route_query(
                query_latent, cfg.top_shards)
            candidate_ids = [
                nid for nid in self.node_index if self._get_node_shard(nid) in active_shards]
            search_nodes = [(nid, self.nodes[nid])
                            for nid in candidate_ids if nid in self.nodes]
            self.stats["shard_hits"] += len(candidate_ids)
        else:
            # Always use vectorized batch resonance (removes Python-loop
            # overhead)
            results = self._query_vectorized(
                query_latent, phase, top_k, modality, session_id, t0)

        # Track 2: Fallback to warm/cold tiers if tiered storage is enabled
        if (self._tiered_store is not None and
                cfg.tiered_fallback_enabled and
                len(results) < top_k):
            needed = top_k - len(results)
            # Warm candidates — peek without promotion (fast bulk read)
            warm_ids = self._tiered_store.warm_ids()
            if warm_ids:
                warm_nodes = self._tiered_store.peek_batch(warm_ids)
                if warm_nodes:
                    scores = self._batch_resonance_nodes(
                        query_latent[np.newaxis, :],
                        np.array([phase], dtype=np.float32),
                        warm_nodes,
                    )[0]
                    for idx, node in enumerate(warm_nodes):
                        resp = float(
                            scores[idx]) * (1.3 if session_id and node.content.get("session") == session_id else 1.0)
                        if resp >= cfg.min_response:
                            results.append((node.id, resp, node))
                            node.last_resonated = time.time()
                    results.sort(key=lambda x: x[1], reverse=True)
            # Cold candidates (sample to avoid loading everything)
            if len(results) < top_k:
                cold_ids = self._tiered_store.cold_ids()
                if cold_ids:
                    import random
                    sample_size = min(len(cold_ids), needed * 5)
                    sample_ids = random.sample(cold_ids, sample_size)
                    cold_nodes = self._tiered_store.peek_batch(sample_ids)
                    if cold_nodes:
                        scores = self._batch_resonance_nodes(
                            query_latent[np.newaxis, :],
                            np.array([phase], dtype=np.float32),
                            cold_nodes,
                        )[0]
                        for idx, node in enumerate(cold_nodes):
                            resp = float(scores[idx]) * (
                                1.3 if session_id and
                                node.content.get("session") == session_id else 1.0)
                            if resp >= cfg.min_response:
                                results.append((node.id, resp, node))
                                node.last_resonated = time.time()
                        results.sort(key=lambda x: x[1], reverse=True)
                        results = results[:top_k]

        # Fallback loop path (should rarely reach here)
        if 'results' not in locals():
            search_nodes = [(nid, self.nodes[nid])
                            for nid in self.node_index if nid in self.nodes]
            if cfg.sparse_routing:
                self.stats["shard_misses"] += 1

            # Fix 3: Hyperbolic pre-filtering for candidate selection
            if cfg.hyperbolic and len(search_nodes) > top_k * 5:
                query_norm = np.linalg.norm(query_latent)
                if query_norm >= cfg.ball_radius:
                    query_latent = query_latent * \
                        (cfg.ball_radius - 1e-6) / max(query_norm, 1e-8)
                prefiltered = []
                for nid, node in search_nodes:
                    # FIX: Never mutate node.latent_pos — use a local copy for
                    # projection
                    node_norm = np.linalg.norm(node.latent_pos)
                    node_pos = node.latent_pos
                    if node_norm >= cfg.ball_radius:
                        node_pos = node.latent_pos * \
                            (cfg.ball_radius - 1e-6) / max(node_norm, 1e-8)
                    hdist = poincare_dist(
                        query_latent, node_pos, cfg.ball_radius)
                    if hdist < 3.0:
                        prefiltered.append((nid, node))
                if len(prefiltered) > 0:
                    search_nodes = prefiltered

            results = []
            for nid, node in search_nodes:
                resp = self._resonance_response(
                    query_latent, phase, node, query_modality=modality)
                # Session priority bonus: boost nodes matching the queried
                # session
                if session_id and node.content.get("session") == session_id:
                    resp *= 1.3  # 30% boost for session-matching nodes
                if resp >= cfg.min_response:
                    results.append((nid, resp, node))
                    node.last_resonated = time.time()

            results.sort(key=lambda x: x[1], reverse=True)

        # P1.3: Trigger adaptive bandwidth re-optimisation periodically
        if self.adaptive_bw is not None and self.adaptive_bw.should_optimize():
            if self._cached_positions is not None and len(self.nodes) >= self.adaptive_bw.min_nodes:
                try:
                    optimal_bw = self.adaptive_bw.optimize(
                        self._cached_positions,
                        self._cached_phases,
                        self._cached_amplitudes,
                        self._cached_saliences,
                        top_k=cfg.top_k,
                    )
                    self.stats["adaptive_bw"] = optimal_bw
                except Exception:
                    logger.warning("Adaptive bandwidth optimisation failed", exc_info=True)

        self.stats["total_queries"] += 1

        # Track shard query time
        if cfg.sparse_routing:
            elapsed_ms = (time.time() - t0) * 1000
            self.stats["avg_shard_query_time_ms"] = (
                0.95 * self.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms)

        if cfg.cross_modal:
            self.stats["cross_modal_queries"] += 1
            if results:
                cm_scores = [n.cross_modal_score for _, _, n in results]
                self.stats["cross_modal_recall"] = 0.9 * \
                    self.stats["cross_modal_recall"] + 0.1 * float(np.mean(cm_scores))

        if len(results) == 0 and cfg.bm25_fallback and self.bm25_index:
            texts = []
            for nid in self.node_index[:100]:
                t = self._extract_text(self.nodes[nid].content)
                if t:
                    texts.append(t)
            fallback_query = query_text if query_text else " ".join(texts)
            if fallback_query:
                for doc_id, score in self._index_mgr.bm25_search(fallback_query, top_k):
                    if doc_id in self.nodes:
                        results.append(
                            (doc_id, score * 0.1, self.nodes[doc_id]))
                self.stats["bm25_fallbacks"] += 1

        if results:
            self.stats["avg_response"] = 0.9 * \
                self.stats["avg_response"] + 0.1 * results[0][1]

        # P2.2: Weight retrieval scores by uncertainty (lower uncertainty →
        # higher score)
        if self.kalman_filter is not None and results:
            weighted = []
            for nid, score, node in results:
                if node.covariance is not None:
                    w = self.kalman_filter.uncertainty_weight(node.covariance)
                    score = score * w
                weighted.append((nid, score, node))
            results = weighted
            # Re-sort and re-truncate after weighting so low-uncertainty nodes
            # can displace high-uncertainty ones in the top-k.
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        # Phase 13 Track 1: Goal relevance scoring
        if self.goal_tracker and results:
            for nid, resp, node in results:
                node.goal_relevance = self.goal_tracker.get_goal_relevance(nid)

        # Phase 13 Track 2: Cognitive attention bias
        if cfg.attention_bias and results:
            results = apply_attention_bias(results, cfg.bias_temperature)
            self.stats["attention_bias_applied"] += 1

        # Phase 13 Track 4: Event-driven trigger for queries
        if self.event_scheduler and results:
            self.event_scheduler.enqueue(
                "query", {"top_score": results[0][1] if results else 0})

        if self.meta_kernel:
            self.meta_kernel.record_response(results[0][1] if results else 0.0)
            if len(results) >= 2:
                positions = np.array([n.latent_pos for _, _, n in results])
                valid = pdist(positions)
                density = 1.0 / (1.0 + np.mean(valid)
                                 ) if len(valid) > 0 else 0.0
                self.meta_kernel.record_semantic_density(float(density))
            if len(results) >= 2:
                responses = np.array([r for _, r, _ in results])
                normalized = responses / (np.sum(responses) + 1e-8)
                entropy = -np.sum(normalized * np.log(normalized + 1e-8))
                self.meta_kernel.record_uncertainty(float(entropy))

        # Phase 16 Track 2: Store results for SafetyCertifier
        self._last_query_results = results

        if self.causal_engine and len(results) >= 2:
            self.causal_engine.record_cooccurrence(
                results[0][0], results[1][0])
            active = [
                nid for nid,
                resp,
                _ in results if resp > cfg.min_response *
                0.5]
            if active:
                self.causal_engine.record_observation(active)
                self._active_node_history.append(active)

        # Phase 14 Track 1: Meta-memory recall tracking
        if self.meta_memory_eval and results:
            top_score = results[0][1]
            avg_age = np.mean(
                [time.time() - n.created_at for _, _, n in results])
            self.meta_memory_eval.record_recall(
                "", top_score, node_age=avg_age)
            self.stats["recall_accuracy"] = self.meta_memory_eval.evaluate_recall_accuracy()

        # P1.1: Conformal prediction filtering
        results = self._apply_conformal_filter(results)

        # E: Retrieval-aware feedback for SOT
        if cfg.sot_retrieval_feedback and self._projection_mgr.has_sot and results:
            self._sot_retrieval_feedback(query_latent, results)

        # Track 3: Adaptive top_k based on confidence
        if cfg.adaptive_top_k:
            results = self._apply_adaptive_top_k(results)

        final = results[:top_k]

        # Track 3: Store in cache
        if self.query_cache is not None:
            cache_key = self._query_cache_key(
                query_latent, phase, top_k, modality, session_id)
            self.query_cache.put_raw(cache_key, final)

        return final

    def batch_query(
            self,
            embeddings: List[NDArray],
            phases: Optional[List[float]] = None,
            top_k: Optional[int] = None,
            modality: str = "text",
            session_id: Optional[str] = None) -> List[List[Tuple[str,
                                                                  float,
                                                                  MemoryNode]]]:
        """Batch query with vectorized resonance for HNSW candidate sets."""
        top_k = top_k or self.cfg.top_k
        n = len(embeddings)
        if phases is None:
            phases = [0.0] * n

        query_latents = np.array(
            [self._project(e) for e in embeddings], dtype=np.float32)
        phases_arr = np.array(phases, dtype=np.float32)

        # HNSW fast path: collect union candidates, single batch resonance call
        n_pos = self._index_mgr.hnsw_count()
        if self.cfg.use_hnsw and n_pos > getattr(self.cfg, "hnsw_min_nodes", 50):
            hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
            per_query_candidates: List[List[str]] = []
            all_candidate_ids: List[str] = []
            candidate_set: Set[str] = set()
            for ql in query_latents:
                cands = self._index_mgr.hnsw_search(ql, hnsw_k)
                cands = [nid for nid in cands if nid in self.nodes]
                per_query_candidates.append(cands)
                for nid in cands:
                    if nid not in candidate_set:
                        candidate_set.add(nid)
                        all_candidate_ids.append(nid)
            if all_candidate_ids:
                all_scores = self._batch_resonance(
                    query_latents, phases_arr, all_candidate_ids)
                cand_index = {nid: idx for idx, nid in enumerate(all_candidate_ids)}
                results: List[List[Tuple[str, float, MemoryNode]]] = []
                for i, cands in enumerate(per_query_candidates):
                    row = []
                    for nid in cands:
                        j = cand_index[nid]
                        score = float(all_scores[i, j])
                        node = self.nodes[nid]
                        if session_id and node.content.get("session") == session_id:
                            score *= 1.3
                        if score >= self.cfg.min_response:
                            row.append((nid, score, node))
                            node.last_resonated = time.time()
                    row.sort(key=lambda x: x[1], reverse=True)
                    results.append(row[:top_k])
                return results
            return [[] for _ in range(n)]

        # Fallback: vectorized or per-query loop
        return [
            self.query(
                e,
                p,
                top_k=top_k,
                modality=modality,
                session_id=session_id) for e,
            p in zip(
                embeddings,
                phases)]

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

    def query_by_text(self,
                      text: str,
                      top_k: Optional[int] = None,
                      session_id: Optional[str] = None) -> List[Tuple[str,
                                                                      float,
                                                                      Any]]:
        """Query field using SOT tokenizer (no external embedder required)."""
        top_k = top_k or self.cfg.top_k
        query_latent = self._projection_mgr.sot_query_latent(text)
        if query_latent is not None:
            return self.query(query_latent, phase=0.0, top_k=top_k,
                              modality="text", session_id=session_id)
        return []

    # B2: Lazy property for causal engine
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
    def add_node(
            self,
            embedding: NDArray,
            content: Dict,
            phase: Optional[float] = None,
            node_id: Optional[str] = None,
            session_id: Optional[str] = None,
            modality: str = "text",
            skip_projection: bool = False,
            modal_embedding: Optional[NDArray] = None) -> str:
        # Rate limiting: configurable via RTMDK_ADD_RATE_LIMIT env var (default
        # 100/sec)
        _rate_limit = int(os.environ.get("RTMDK_ADD_RATE_LIMIT", "100"))
        if _rate_limit > 0:
            now = time.time()
            while self._add_node_timestamps and self._add_node_timestamps[0] < now - 1.0:
                self._add_node_timestamps.popleft()
            if len(self._add_node_timestamps) >= _rate_limit:
                raise SecurityViolationError(
                    f"Rate limit exceeded: max {_rate_limit} nodes/second")
            self._add_node_timestamps.append(now)

        # v8.2.1: Input sanitization
        try:
            from rtmdk.production.sanitization import validate_embedding
            embedding = validate_embedding(embedding)
        except Exception as exc:
            raise ValueError(f"Invalid embedding: {exc}")

        # Phase 14 Track 2: Security validation
        if self.security:
            # Check ALL text fields for prompt injection, not just 'text'
            text = content.get("text", "")
            input_text = content.get("input_text", "")
            output_text = content.get("output_text", "")
            for field_text in [text, input_text, output_text]:
                if field_text:
                    validation = self.security.validate_node_content(
                        field_text)
                    if not validation["is_safe"]:
                        self.stats["security_violations"] += 1
                        logger.warning(
                            f"Security violation in add_node: {validation['violations']}")
                        # Fix 7: Raise instead of returning "" — caller must
                        # handle
                        raise SecurityViolationError(
                            f"Security violation: {validation['violations']}")

        nid = node_id or f"n_{len(self.nodes)}_{int(time.time() * 1000)}"
        if skip_projection:
            # Input is already in latent space (e.g., crystallization)
            if len(embedding) != self.cfg.latent_dim:
                raise ValueError(
                    f"skip_projection=True but embedding dim {len(embedding)} != "
                    f"latent_dim {self.cfg.latent_dim}")
            latent = embedding
        elif len(embedding) == self.cfg.latent_dim:
            # Phase 21: SOT embeddings are already latent_dim — use directly
            latent = embedding.astype(np.float32)
        else:
            latent = self._projection_mgr.update_projection(embedding)
            if self._projection_mgr.projection_learner is not None:
                self.stats["projection_updates"] += 1

        # Track 1: Quantize latent position to reduce RAM usage
        latent, latent_scale, latent_zero_point = self._quant.quantize_with_meta(latent)

        if phase is None:
            phase = self._get_phase(session_id, embedding, modality, content)

        # OPTIMIZED: Initialize amplitude/salience based on embedding quality
        # Higher norm embeddings → more informative content → higher initial
        # salience
        emb_norm = float(np.linalg.norm(embedding))
        # Typical emb_norm range: 5-30 for real embeddings, 2-10 for synthetic
        # Normalize to [0.5, 1.0] range for salience
        salience = min(1.0, max(0.3, emb_norm / 20.0))
        amplitude = min(1.0, max(0.5, emb_norm / 15.0))

        node = MemoryNode(
            id=nid,
            latent_pos=latent,
            phase=phase,
            amplitude=amplitude,
            salience=salience,
            content=content,
            lineage=[],
            modality=modality,
            latent_scale=latent_scale,
            latent_zero_point=latent_zero_point,
            modal_embedding=modal_embedding.astype(np.float32) if modal_embedding is not None else None)

        # P2.2: Initialize uncertainty covariance
        if self.kalman_filter is not None:
            node.covariance = self.kalman_filter.init_covariance()

        if self.cfg.cross_modal:
            node.modal_embedding = embedding.copy()

        # Phase 17: Role assignment
        role = DEFAULT_ROLE
        if self.role_router:
            text = content.get("text", "")
            # Check if content has explicit role tag
            explicit_role = content.get("role") or content.get("tier_role")
            role = self.role_router.add_node(nid, text, role=explicit_role)
            node.role = role  # Set role attribute on node

        self.nodes[nid] = node
        # H1: Prevent duplicate node_id in node_index
        if nid not in self.node_index:
            self.node_index.append(nid)
        self.stats["total_adds"] += 1

        # Extract causal edges from explanation text
        causal_edges = extract_causal_edges_from_content(content)
        if causal_edges:
            for effect, cause, strength in causal_edges:
                # Store as causal metadata on the node
                node.causal_strength[cause] = max(
                    node.causal_strength.get(cause, 0.0), strength)
                node.causal_parents.append(cause)
            self.stats.setdefault("causal_edges_extracted", 0)
            self.stats["causal_edges_extracted"] += len(causal_edges)

        # P0: Invalidate cached arrays (will be rebuilt on next query)
        # For single node additions, use incremental append if cache exists
        if self._cached_positions is not None:
            # Incremental append to avoid full rebuild
            try:
                self._cached_positions = np.vstack(
                    [self._cached_positions, latent.reshape(1, -1)])
                self._cached_phases = np.append(
                    self._cached_phases,
                    phase if phase is not None else self._get_phase(
                        session_id, embedding, modality, content))
                self._cached_amplitudes = np.append(
                    self._cached_amplitudes, amplitude)
                self._cached_saliences = np.append(
                    self._cached_saliences, salience)
                self._cached_modal_weights = np.append(
                    self._cached_modal_weights, 1.0)
                self._cached_gates = np.append(self._cached_gates, 1.0)
                self._cached_causal_boost = np.append(
                    self._cached_causal_boost, 1.0)
            except Exception:
                logger.warning(
                    "Incremental cache append failed, falling back to full rebuild",
                    exc_info=True)
                # Fallback: mark dirty for full rebuild
                self._cache_dirty = True
        else:
            self._cache_dirty = True

        # B1: Invalidate tension cache for neighbors (new node affects
        # topology)
        self._invalidate_tension_cache(nid)

        # Track 3: Invalidate query cache (new node may change rankings)
        if self.query_cache is not None:
            self.query_cache.clear()

        # Phase 17: Update shard distribution stats
        if self.role_router:
            self.stats["n_shards"] = len(self.role_router.shards)
            self.stats["shard_distribution"] = {
                r: len(s.node_ids) for r, s in self.role_router.shards.items()
            }
            self.stats["role_router_enabled"] = True

        if self.cfg.use_hnsw and self.hnsw_index:
            self._index_mgr.hnsw_insert(nid, latent)
        if self.cfg.bm25_fallback and self.bm25_index:
            text = self._extract_text(content)
            if text:
                self._index_mgr.bm25_add(nid, text)

        # Phase 13 Track 1: Event-driven trigger for node added
        if self.event_scheduler:
            self.event_scheduler.enqueue(
                "node_added", {
                    "node_id": nid, "modality": modality})

        # Track 5: Store embedding in WAL for durable replay
        self.wal.append_add_node(
            nid, content, modality, embedding=latent.tolist())
        self._dirty = True
        return nid

    def add_nodes_batch(
        self,
        embeddings: NDArray,
        contents: List[Dict],
        phases: Optional[NDArray] = None,
        node_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        skip_projection: bool = False,
        modal_embeddings: Optional[NDArray] = None,
    ) -> List[str]:
        """Batch add nodes with vectorized projection, cache, and index updates.

        This is significantly faster than calling add_node() in a loop because:
        - Projection is a single matrix multiply
        - Normalization / hyperbolic clamp are vectorized
        - Cache append is a single vstack instead of N incremental appends
        - HNSW receives a batch add_items call
        - Query cache is invalidated once, not N times
        - WAL receives a single append
        """
        if len(embeddings) != len(contents):
            raise ValueError(
                f"embeddings length {len(embeddings)} != contents length {len(contents)}")
        n = len(embeddings)
        if n == 0:
            return []

        # --- Vectorized projection ---
        if skip_projection:
            if embeddings.shape[1] != self.cfg.latent_dim:
                raise ValueError(
                    f"skip_projection=True but embedding dim {embeddings.shape[1]} != "
                    f"latent_dim {self.cfg.latent_dim}")
            latents = embeddings.astype(np.float32)
        elif embeddings.shape[1] == self.cfg.latent_dim:
            latents = embeddings.astype(np.float32)
        else:
            latents = self._project_batch(embeddings)

        # Vectorized normalization
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        latents = latents / np.maximum(norms, 1e-8)

        # Track 1: Quantize (loop — quantizer may not be vectorizable)
        _q_results = [self._quant.quantize_with_meta(vec) for vec in latents]
        latents = np.array([r[0] for r in _q_results])
        _latent_scales = [r[1] for r in _q_results]
        _latent_zps = [r[2] for r in _q_results]

        # --- Vectorized phase / amplitude / salience ---
        if phases is None:
            base = (time.time() * 0.01) % (2 * np.pi)
            if modalities:
                if self.cfg.cross_modal:
                    phases = np.array([
                        (base + self.cfg.modal_phase_offsets.get(m, 0.0)) % (2 * np.pi)
                        for m in modalities
                    ], dtype=np.float32)
                elif self.cfg.multimodal:
                    phases = np.array([
                        (base + self.cfg.modality_phase_shifts.get(m, 0.0)) % (2 * np.pi)
                        for m in modalities
                    ], dtype=np.float32)
                else:
                    phases = np.full(n, base, dtype=np.float32)
            else:
                phases = np.full(n, base, dtype=np.float32)
        else:
            phases = np.asarray(phases, dtype=np.float32)

        emb_norms = np.linalg.norm(embeddings, axis=1)
        saliences = np.clip(emb_norms / 20.0, 0.3, 1.0).astype(np.float32)
        amplitudes = np.clip(emb_norms / 15.0, 0.5, 1.0).astype(np.float32)

        # --- Create nodes ---
        now = time.time()
        base_idx = len(self.nodes)
        batch_nids: List[str] = []
        for i in range(n):
            nid = node_ids[i] if node_ids else f"n_{base_idx + i}_{int(now * 1000)}_{i}"
            batch_nids.append(nid)
            node = MemoryNode(
                id=nid,
                latent_pos=latents[i],
                phase=float(phases[i]),
                amplitude=float(amplitudes[i]),
                salience=float(saliences[i]),
                content=contents[i],
                lineage=[],
                modality=modalities[i] if modalities else "text",
                latent_scale=_latent_scales[i],
                latent_zero_point=_latent_zps[i],
            )
            if self.cfg.cross_modal:
                node.modal_embedding = embeddings[i].copy()
            if modal_embeddings is not None:
                node.modal_embedding = modal_embeddings[i].astype(np.float32)
            self.nodes[nid] = node
            if nid not in self.node_index:
                self.node_index.append(nid)
            self.stats["total_adds"] += 1

        # --- Vectorized cache append ---
        if self._cached_positions is not None:
            self._cached_positions = np.vstack(
                [self._cached_positions, latents])
            self._cached_phases = np.append(self._cached_phases, phases)
            self._cached_amplitudes = np.append(
                self._cached_amplitudes, amplitudes)
            self._cached_saliences = np.append(
                self._cached_saliences, saliences)
            self._cached_modal_weights = np.append(
                self._cached_modal_weights, np.ones(n, dtype=np.float32)
            )
            self._cached_gates = np.append(
                self._cached_gates, np.ones(n, dtype=np.float32)
            )
            self._cached_causal_boost = np.append(
                self._cached_causal_boost, np.ones(n, dtype=np.float32)
            )
        else:
            self._cache_dirty = True

        # --- Batch HNSW insert ---
        if self.cfg.use_hnsw and self.hnsw_index:
            self._index_mgr.hnsw_insert_batch(batch_nids, latents)

        # --- Batch BM25 insert ---
        if self.cfg.bm25_fallback and self.bm25_index:
            for i, nid in enumerate(batch_nids):
                text = self._extract_text(contents[i])
                if text:
                    self._index_mgr.bm25_add(nid, text)

        # Track 3: Invalidate query cache once
        if self.query_cache is not None:
            self.query_cache.clear()

        self.wal.append("add_nodes_batch", {
            "count": n,
            "node_ids": batch_nids,
            "contents": contents,
            "embeddings": [vec.tolist() for vec in latents],
            "modalities": modalities if modalities else ["text"] * n,
        })
        self._dirty = True
        # Pre-build node cache to eliminate first-query penalty
        if self._cached_positions is not None:
            self._build_node_cache()
        return batch_nids

    def delete_nodes(self, node_ids: List[str]) -> None:
        """Remove nodes by ID. Used by WAL replay and consolidation."""
        for nid in node_ids:
            if nid in self.nodes:
                del self.nodes[nid]
        # Rebuild node_index (remove deleted, preserve order)
        self.node_index = [nid for nid in self.node_index if nid in self.nodes]
        if self.cfg.use_hnsw and self.hnsw_index:
            for nid in node_ids:
                self._index_mgr.hnsw_remove(nid)
        # Track 5: WAL durability for explicit deletions
        self.wal.append_delete(node_ids)
        # Invalidate caches
        self._cache_dirty = True
        if self.query_cache is not None:
            self.query_cache.clear()

    def queue_add_nodes(
        self,
        embeddings: NDArray,
        contents: List[Dict],
        modalities: Optional[List[str]] = None,
    ) -> None:
        """Queue a batch for background ingestion.

        If async_pipeline is enabled, the batch is placed on save_q and
        processed by _worker_save.  Otherwise add_nodes_batch is called
        synchronously.
        """
        if not self.cfg.async_pipeline or self.save_q is None:
            self.add_nodes_batch(embeddings, contents, modalities=modalities)
            return
        payload = {
            "embeddings": embeddings,
            "contents": contents,
            "modalities": modalities,
        }
        try:
            self.save_q.put_nowait(payload)
        except asyncio.QueueFull:
            # Backpressure: fall back to synchronous path
            logger.warning(
                "save_q full — falling back to synchronous add_nodes_batch")
            self.add_nodes_batch(embeddings, contents, modalities=modalities)

    def calibrate(
            self,
            query_embedding: NDArray,
            node_id: str,
            is_relevant: bool) -> None:
        """Add a labeled query-result pair to the conformal calibration set.

        Args:
            query_embedding: the query embedding used for retrieval
            node_id: the retrieved node id
            is_relevant: whether this node was judged relevant for the query
        """
        if not self.cfg.conformal_prediction or self.conformal_calibrator is None:
            return
        if not is_relevant or node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        query_latent = self._project(query_embedding)
        score = self._resonance_response(query_latent, node.phase, node)
        self.conformal_calibrator.add_sample(score)

    def _apply_conformal_filter(self,
                                results: List[Tuple[str,
                                                    float,
                                                    MemoryNode]]) -> List[Tuple[str,
                                                                                float,
                                                                                MemoryNode]]:
        """Filter query results through conformal prediction threshold.

        Returns only results whose score lies in the conformal prediction set.
        Records threshold and confidence in stats.
        """
        if not self.cfg.conformal_prediction or self.conformal_calibrator is None:
            return results
        if self.conformal_calibrator.n_calibrated < self.cfg.conformal_min_calib:
            return results
        scores = [score for _, score, _ in results]
        nids = [nid for nid, _, _ in results]
        pred_set, confidence, threshold = self.conformal_calibrator.predict(
            scores, nids)
        self.stats["conformal_threshold"] = threshold
        self.stats["conformal_confidence"] = confidence
        self.stats["conformal_prediction_set_size"] = len(pred_set)
        pred_set_lookup = set(pred_set)
        return [(nid, score, node)
                for nid, score, node in results if nid in pred_set_lookup]

    def _invalidate_tension_cache(self, node_id: Optional[str] = None):
        """B1: Invalidate tension cache. If node_id given, invalidate that node and neighbors.
        Otherwise, invalidate entire cache. Also cleans entries for deleted nodes."""
        # H8: Clean up entries for deleted nodes on every call
        dead_keys = [k for k in self._tension_cache if k not in self.nodes]
        for k in dead_keys:
            self._tension_cache.pop(k, None)

        if node_id is not None:
            # Remove specific node and mark neighbors for refresh
            self._tension_cache.pop(node_id, None)
            # Invalidate cache for nodes near the changed one
            node = self.nodes.get(node_id)
            if node:
                for nid in list(self._tension_cache.keys()):
                    if nid == node_id:
                        continue
                    # Simple proximity check: invalidate ~20% of cache
                    if hash(nid) % 5 == 0:
                        self._tension_cache.pop(nid, None)
        else:
            # Full invalidation
            self._tension_cache.clear()

    def _sweep_tension_cache(self):
        """Remove stale tension cache entries for live nodes (Fix 3: prevent unbounded cache growth)."""
        if not self._tension_cache:
            return
        # Only sweep if cache is large (more than 2x number of nodes)
        if len(self._tension_cache) <= len(self.nodes) * 2:
            return
        current_step = self._step_counter
        keys_to_remove = [
            k for k, (tension, step) in self._tension_cache.items()
            if current_step - step > self._tension_cache_max_age * 3
            and k in self.nodes  # Only remove for live nodes
        ]
        for k in keys_to_remove:
            self._tension_cache.pop(k, None)

    def _compute_tension(
            self,
            node_id: str,
            neighborhood_radius: float = 2.0) -> float:
        # B1: Tension cache check
        if node_id in self._tension_cache:
            cached_tension, cached_step = self._tension_cache[node_id]
            if self._step_counter - cached_step < self._tension_cache_max_age:
                self._tension_cache_hits += 1
                return cached_tension

        self._tension_cache_misses += 1

        node = self.nodes[node_id]

        # Use HNSW for fast k-NN, else fallback to deterministic k-NN via cdist
        k_neighbors = 10
        neighbor_ids = []

        if self.cfg.use_hnsw and self._index_mgr.hnsw_count() > k_neighbors:
            candidate_ids = self._index_mgr.hnsw_search(
                node.latent_pos, top_k=k_neighbors + 1)
            neighbor_ids = [
                nid for nid in candidate_ids if nid != node_id and nid in self.nodes]
        else:
            # Deterministic fallback: compute distances to a limited window
            ids_to_check = self.node_index
            max_scan = 200  # Limit scan for performance
            if len(ids_to_check) > max_scan:
                # Use reservoir-style sample with deterministic seed based on
                # node_id
                rng = np.random.RandomState(
                    int(hashlib.md5(node_id.encode()).hexdigest(), 16) % 2**32)
                ids_to_check = list(
                    rng.choice(
                        ids_to_check,
                        size=max_scan,
                        replace=False))

            if len(ids_to_check) < 2:
                return 0.0

            # Compute distances and select k nearest within radius
            others = [(oid, self.nodes[oid])
                      for oid in ids_to_check if oid != node_id and oid in self.nodes]
            if not others:
                return 0.0

            other_positions = np.array([n.latent_pos for _, n in others])
            other_ids = [oid for oid, _ in others]
            dists = np.linalg.norm(other_positions - node.latent_pos, axis=1)

            # Filter by radius and select k nearest
            within_radius = dists < neighborhood_radius
            if not np.any(within_radius):
                # Fallback: take k nearest regardless of radius
                k = min(k_neighbors, len(dists))
                nearest_idx = np.argsort(dists)[:k]
                neighbor_ids = [other_ids[i] for i in nearest_idx]
            else:
                radius_dists = [(other_ids[i], dists[i])
                                for i in range(len(dists)) if within_radius[i]]
                radius_dists.sort(key=lambda x: x[1])
                neighbor_ids = [oid for oid, _ in radius_dists[:k_neighbors]]

        if len(neighbor_ids) < 2:
            tension = 0.0
        else:
            neighbors = [self.nodes[oid] for oid in neighbor_ids]
            phases = np.array([n.phase for n in neighbors])
            saliences = np.array([n.salience for n in neighbors])
            tension = 0.6 * (np.std(np.cos(phases)) +
                             np.std(np.sin(phases))) + 0.4 * np.std(saliences)

        # Phase 14 Track 2: Security - detect tension spikes
        if self.security and not self.security.validate_tension_spike(
                float(tension)):
            self.stats["tension_spikes_blocked"] += 1

        # B1: Cache the computed tension
        result = float(tension)
        self._tension_cache[node_id] = (result, self._step_counter)
        return result

    def _soft_gate(self, tension: float) -> float:
        if not self.cfg.soft_gates:
            return 1.0
        eff = self.adaptive_threshold.get_threshold(
        ) if self.adaptive_threshold else self.cfg.tension_threshold
        return float(
            1 / (1 + math.exp(-(tension - eff) / self.cfg.gate_temperature)))

    def get_effective_threshold(self) -> float:
        return self.adaptive_threshold.get_threshold(
        ) if self.adaptive_threshold else self.cfg.tension_threshold

    @_locked
    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        """Run one consolidation pass over the field.

        Delegated to ConsolidationManager for maintainability.
        """
        return self._consolidation_mgr.consolidate(mode)

    def _self_supervise(self):
        """Fix: Use local probe instead of np.zeros to prevent false decay of peripheral nodes."""
        if not self.cfg.self_supervision:
            return
        self.stats["self_sup_checks"] += 1
        for nid in list(self.node_index):
            if nid not in self.nodes or not self.nodes[nid].lineage:
                continue
            node = self.nodes[nid]
            # FIX: Probe around the node's actual position
            probe = node.latent_pos + \
                self._rng.normal(0, 0.05, node.latent_pos.shape)
            results = self.query(probe, phase=node.phase, top_k=1)
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

    # Phase 11 Track 3: Predictive coding
    def _encode_field_state(self) -> NDArray:
        """Encode field state into a flat vector for predictive coding."""
        if not self.nodes:
            return np.zeros(self.cfg.latent_dim * 4, dtype=np.float32)
        # Aggregate: mean pos, mean phase, mean amp, mean sal
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        phases = np.array([n.phase for n in self.nodes.values()])
        amps = np.array([n.amplitude for n in self.nodes.values()])
        sals = np.array([n.salience for n in self.nodes.values()])
        mean_pos = np.mean(positions, axis=0)
        mean_phase = np.mean(phases)
        mean_amp = np.mean(amps)
        mean_sal = np.mean(sals)
        # Encode into latent_dim * 4
        state = np.zeros(self.cfg.latent_dim * 4, dtype=np.float32)
        pos_dim = min(len(mean_pos), self.cfg.latent_dim)
        state[:pos_dim] = mean_pos[:pos_dim]
        state[self.cfg.latent_dim] = mean_phase
        state[self.cfg.latent_dim * 2] = mean_amp
        state[self.cfg.latent_dim * 3] = mean_sal
        return state

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self, base_query: NDArray,
                               intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical trajectories via do-interventions."""
        if not self.scenario_planner:
            return []
        return self.scenario_planner.imagine_counterfactual(
            base_query, intervention)

    def _merge_latents(self, node, partner):
        """Merge two node latent positions using learned or heuristic method."""
        if self.learned_consolidator is not None and self.learned_consolidator._trained:
            # Undo quantization for learned merge (needs float32 latent)
            latent_a = self._quant.dequantize(
                node.latent_pos, node.latent_scale, node.latent_zero_point)
            latent_b = self._quant.dequantize(
                partner.latent_pos, partner.latent_scale, partner.latent_zero_point)
            merged = self.learned_consolidator.predict(
                latent_a, latent_b,
                node.phase, partner.phase,
                node.amplitude, partner.amplitude,
                node.salience, partner.salience,
            )
            # Re-quantize
            merged_q, scale, zp = self._quant.quantize_with_meta(merged)
            node.latent_pos = merged_q
            node.latent_scale = scale
            node.latent_zero_point = zp
        else:
            # Heuristic average (preserves existing behaviour)
            node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)

    def _train_learned_consolidator(self):
        """Collect synthetic merge examples and train the consolidator MLP."""
        if self.learned_consolidator is None:
            return
        n = len(self.nodes)
        if n < 4:
            return
        # Sample random node pairs as synthetic merge examples
        rng = np.random.default_rng(42)
        node_list = list(self.nodes.values())
        for _ in range(min(50, n * 2)):
            a, b = rng.choice(node_list, size=2, replace=False)
            # Dequantize latents for training
            la = self._quant.dequantize(a.latent_pos, a.latent_scale, a.latent_zero_point)
            lb = self._quant.dequantize(b.latent_pos, b.latent_scale, b.latent_zero_point)
            # Queries = the parent latents themselves (proxy)
            self.learned_consolidator.add_example(
                la, lb,
                queries=[la, lb],
                phase_a=a.phase, phase_b=b.phase,
                amp_a=a.amplitude, amp_b=b.amplitude,
                sal_a=a.salience, sal_b=b.salience,
            )
        self.learned_consolidator.train(epochs=10, lr=0.005)

    def _prune_dead_nodes(self):
        to_remove = [nid for nid in self.node_index
                     if self.nodes[nid].amplitude < self.cfg.min_amplitude
                     or self.nodes[nid].salience < self.cfg.min_amplitude * 0.5]
        if to_remove:
            self.wal.append_delete(to_remove)
        for nid in to_remove:
            if self.cfg.use_hnsw:
                self._index_mgr.hnsw_remove(nid)
            if self.cfg.bm25_fallback:
                self._index_mgr.bm25_remove(nid)
            del self.nodes[nid]
        # B1: Invalidate cache on node pruning
        if to_remove:
            self._invalidate_tension_cache()
            self._cache_dirty = True
        # FIX: Rebuild node_index once instead of O(N) remove per node
        self.node_index = [nid for nid in self.node_index if nid in self.nodes]

    def _check_field_integrity(self) -> Dict[str, Any]:
        """Check for NaN/inf in nodes, report issues, and heal them (Fix 11)."""
        issues = []
        n_nan = 0
        n_inf = 0
        healed = []
        for nid, node in self.nodes.items():
            needs_heal = False
            if np.any(np.isnan(node.latent_pos)):
                n_nan += 1
                issues.append(f"NaN in {nid} — will heal")
                needs_heal = True
            if np.any(np.isinf(node.latent_pos)):
                n_inf += 1
                issues.append(f"Inf in {nid} — will heal")
                needs_heal = True
            if np.isnan(node.phase) or np.isinf(node.phase):
                issues.append(f"Invalid phase in {nid} — will heal")
                needs_heal = True
                node.phase = 0.0
            if np.isnan(node.amplitude) or node.amplitude < 0:
                issues.append(f"Invalid amplitude in {nid} — will heal")
                needs_heal = True
                node.amplitude = self.cfg.min_amplitude
            # Fix 11: Actually heal NaN positions by resetting to small random
            # values
            if needs_heal:
                node.latent_pos = self._rng.standard_normal(
                    self.cfg.latent_dim).astype(np.float32) * 0.01
                healed.append(nid)
                self.stats["field_integrity_issues"] = self.stats.get(
                    "field_integrity_issues", 0) + 1
        return {
            "n_issues": len(issues),
            "n_nan": n_nan,
            "n_in": n_inf,
            "healed": healed,
            "issues": issues[:20],
        }

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
            healed.extend(
                self.healer.heal_dead_zones(
                    self.nodes,
                    diagnostics["dead_zone_nodes"]))
        if diagnostics.get("hyperconvergence", False):
            healed.extend(self.healer.heal_hyperconvergence(self.nodes))
        if diagnostics.get("fragmentation",
                           0) > self.cfg.fragmentation_threshold:
            if len(self.nodes) >= 2:
                positions = np.array(
                    [n.latent_pos for n in self.nodes.values()])
                tree = cKDTree(positions)
                neighbors = tree.query_ball_point(positions, 2.0)
                isolated = [self.node_index[i] for i in range(
                    len(self.node_index)) if len(neighbors[i]) <= 1]
                if isolated:
                    healed.extend(
                        self.healer.heal_fragmentation(
                            self.nodes, isolated))
        if healed:
            self.stats["healing_events"] += len(healed)
            self.stats["healing_history"].extend(healed)
            # Fix 3: Trim on every overflow, not just when exceeding 1000 —
            # prevents unbounded growth
            if len(self.stats["healing_history"]) > 1000:
                self.stats["healing_history"] = self.stats["healing_history"][-500:]
        return healed

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        if not self._rollback_history or n_steps > len(self._rollback_history):
            return False
        snapshot = self._rollback_history[-n_steps]
        for nid, state in snapshot["pre_state"].items():
            if nid in self.nodes:
                node = self.nodes[nid]
                node.latent_pos = state["latent_pos"].copy()
                node.phase = state["phase"]
                node.amplitude = state["amplitude"]
                node.salience = state["salience"]
                # H3: Restore full state for consistent rollback
                node.tension = state.get("tension", 0.0)
                node.soft_gate = state.get("soft_gate", 1.0)
                if "content" in state:
                    node.content = dict(state["content"])
                if "lineage" in state:
                    node.lineage = list(state["lineage"])
                if "causal_strength" in state:
                    node.causal_strength = dict(state["causal_strength"])
                if "causal_parents" in state:
                    node.causal_parents = list(state["causal_parents"])
                node.pre_consolidation_pos = None
        self._rollback_history = self._rollback_history[:-n_steps]
        # H8: Clean tension cache after rollback (nodes changed)
        self._tension_cache.clear()
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

    def get_field_health(self) -> Dict:
        if self.healer:
            health, diagnostics = self.healer.compute_field_health(self.nodes)
            diagnostics["kurtosis"] = self.stats.get("meta_kurtosis", 3.0)
            return diagnostics
        return {"health": "unknown", "kurtosis": 3.0}

    def counterfactual_query(self,
                             intervention: Dict[str,
                                                Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str,
                                                     Any]] = None) -> CounterfactualResult:
        if not self.causal_engine:
            return CounterfactualResult(
                query=str(intervention),
                intervention=intervention,
                predicted_outcomes=[],
                confidence=0.0,
                reasoning_path=["Causal engine not enabled"],
                assumptions=[])
        self.stats["counterfactual_queries"] += 1
        return self.causal_engine.counterfactual_query(
            intervention, query_nodes, evidence, self.cfg.counterfactual_max_depth)

    def get_causal_summary(self) -> Dict:
        if not self.causal_engine:
            return {"enabled": False}
        return {
            "enabled": True,
            "causal_edges": len(self.causal_engine.causal_effects),
            "contradictions": len([
                c for c in self.causal_engine.contradictions.values()
                if not c.resolved]),
            "nodes_with_effects": len(set(k[0] for k in self.causal_engine.causal_effects)),
            "nodes_affected": len(set(k[1] for k in self.causal_engine.causal_effects)),
            "top_effects": sorted(
                [(f"{k[0]}->{k[1]}", v.strength)
                 for k, v in self.causal_engine.causal_effects.items()],
                key=lambda x: x[1], reverse=True)[:10],
        }

    # ========================================================================
    # PHASE 12 TRACK 1: SPARSE RESONANT ROUTING (MoE-memory)
    # ========================================================================

    def _get_node_shard(self, node_id: str) -> int:
        """Get shard assignment for a node."""
        if node_id in self._node_shard_map:
            return self._node_shard_map[node_id]
        if node_id in self.nodes:
            pos = self.nodes[node_id].latent_pos
            dists = np.linalg.norm(self._index_mgr.shard_centers - pos, axis=1)
            shard = int(np.argmin(dists))
            self._node_shard_map[node_id] = shard
            return shard
        return 0

    def _route_query(
            self,
            query_latent: NDArray,
            top_shards: int = 3) -> List[int]:
        """Route query to top_k most relevant shards (softmax-free)."""
        if self._index_mgr.shard_centers is None:
            return list(range(self.cfg.num_shards))
        dists = np.linalg.norm(self._index_mgr.shard_centers - query_latent, axis=1)
        self.shard_router = 1.0 / (1.0 + dists)
        return list(np.argsort(self.shard_router)[-top_shards:])

    def _update_shard_centers(self):
        """Update shard centers based on current node distribution."""
        if self._index_mgr.shard_centers is None or len(self.nodes) < self.cfg.num_shards:
            return
        from sklearn.cluster import KMeans
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        if len(positions) < self.cfg.num_shards:
            return
        kmeans = KMeans(
            n_clusters=self.cfg.num_shards,
            n_init=3,
            random_state=42)
        labels = kmeans.fit_predict(positions)
        self._index_mgr.shard_centers = kmeans.cluster_centers_.astype(np.float32)
        # Update node-shard map
        self._node_shard_map.clear()
        for i, nid in enumerate(self.node_index):
            self._node_shard_map[nid] = int(labels[i])

    def _update_shard_centers_bm25(self):
        """Build topic-based shards from BM25 term vectors.

        Clusters documents by their top BM25 terms instead of embeddings.
        More robust when embeddings are weak or documents are short.
        """
        if self._index_mgr.bm25_index is None or len(self.nodes) < self.cfg.num_shards:
            return
        # Build doc-term matrix from BM25
        from collections import Counter
        term_doc = {}  # term -> list of doc indices
        doc_terms = []  # list of term lists per doc
        nids = list(self.node_index)
        for i, nid in enumerate(nids):
            node = self.nodes[nid]
            text = node.content.get("text", "")
            terms = [w for w in text.lower().split() if len(w) > 2]
            doc_terms.append(terms)
            for t in set(terms):
                term_doc.setdefault(t, []).append(i)
        # Assign each doc to its most discriminative term
        doc_cluster = {}
        for i, terms in enumerate(doc_terms):
            if not terms:
                continue
            # Score terms by inverse doc frequency (simulated)
            best_term = min(terms, key=lambda t: len(term_doc.get(t, [])))
            doc_cluster[i] = best_term
        # Group by term, merge small groups
        groups = {}
        for i, term in doc_cluster.items():
            groups.setdefault(term, []).append(i)
        # Merge until we have num_shards clusters
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        clusters = []
        for term, members in sorted_groups:
            clusters.append(members)
        # Merge smallest clusters
        while len(clusters) > self.cfg.num_shards:
            clusters.sort(key=len)
            clusters[1].extend(clusters[0])
            clusters.pop(0)
        # Assign shard IDs and compute centers as mean latent positions
        self._node_shard_map.clear()
        centers = []
        for shard_id, members in enumerate(clusters):
            for idx in members:
                self._node_shard_map[nids[idx]] = shard_id
            positions = np.array([self.nodes[nids[idx]].latent_pos for idx in members])
            centers.append(positions.mean(axis=0))
        self._index_mgr.shard_centers = np.stack(centers).astype(np.float32)
        logger.info("BM25 topic shards: %d clusters from %d docs", len(clusters), len(nids))

    # ========================================================================
    # PHASE 12 TRACK 2: COGNITIVE CONTEXT COMPRESSION
    # ========================================================================

    def _cognitive_compress(
            self, results: List[Tuple[str, float, MemoryNode]]) -> str:
        """Compress raw memory results into a structured cognitive dump for LLM."""
        if not results:
            return "### COGNITIVE_CONTEXT\nNo relevant structures."

        high_res = [(nid, r, n) for nid, r, n in results if r >
                    self.cfg.high_resonance_threshold]
        contradictions = [n for _, _, n in results if n.content.get(
            "causal_flag") == "incompatible"]
        procedural = [
            n for _, _, n in results if getattr(
                n, 'tier', 'semantic') == "procedural"]

        lines = ["### COGNITIVE_CONTEXT"]
        if high_res:
            summaries = []
            for nid, r, n in high_res:
                text = n.content.get("text", "unknown")[:60]
                summaries.append(f"[{text}...](R:{r:.2f},S:{n.salience:.2f})")
            lines.append(
                f"• High resonance ({len(high_res)} nodes): " +
                " | ".join(summaries))
        if contradictions:
            texts = [n.content.get("text", "unknown")[:40]
                     for n in contradictions[:3]]
            lines.append("[WARN] Conflicting nodes: " + " | ".join(texts))
        if procedural:
            lines.append("[TOOL] Procedural patterns available (how-to)")

        # Add lineage summary for complex nodes
        lineage_nodes = [(nid, n) for nid, r, n in results if n.lineage]
        if lineage_nodes:
            lines.append(
                f"[STATS] Consolidated memories: {len(lineage_nodes)} nodes with synthesis history")

        return "\n".join(lines)

    # ========================================================================
    # PHASE 12 TRACK 3: CRYSTALLIZATION (episodic → semantic/procedural)
    # ========================================================================

    def _crystallize_recurring(
            self,
            window: int = 100,
            similarity_thresh: float = 0.75):
        """Detect recurring episodic patterns and crystallize into semantic nodes."""
        recent_ids = self.node_index[-window:]
        recent = [
            self.nodes[nid] for nid in recent_ids if nid in self.nodes and getattr(
                self.nodes[nid],
                'tier',
                'semantic') == "episodic" and nid not in self._crystallized_nodes]
        if len(recent) < 5:
            return

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return

        pos = np.array([n.latent_pos for n in recent])
        labels = DBSCAN(
            eps=0.4,
            min_samples=self.cfg.crystallization_min_cluster).fit_predict(pos)

        crystallized_count = 0
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            members = [recent[i]
                       for i, l in enumerate(labels) if l == cluster_id]
            if len(members) >= self.cfg.crystallization_min_cluster:
                new_pos = np.mean(
                    [m.latent_pos for m in members], axis=0).astype(np.float32)
                # Circular mean for phases: arctan2(mean(sin), mean(cos))
                phases = np.array([m.phase for m in members])
                new_phase = float(
                    np.arctan2(
                        np.mean(
                            np.sin(phases)),
                        np.mean(
                            np.cos(phases)))) % (2 * np.pi)
                combined_text = " ".join(
                    [m.content.get("text", "")[:30] for m in members[:3]])
                new_content = {
                    "text": f"Crystallized: {combined_text}...",
                    "tier": "semantic",
                    "crystallized_from": [m.id for m in members],
                    "crystallized_at": time.time(),
                }
                new_id = self.add_node(
                    new_pos, new_content, phase=float(
                        new_phase %
                        (2 * np.pi)), skip_projection=True)
                self.nodes[new_id].tier = "semantic"
                # Mark originals as archived
                for m in members:
                    m.content["archived"] = True
                    self._crystallized_nodes.add(m.id)
                crystallized_count += 1

        if crystallized_count > 0:
            self.stats["crystallizations"] += crystallized_count
            self.stats["crystallized_clusters"] += crystallized_count

    # ========================================================================
    # PHASE 12 TRACK 4: ASYNC MULTI-THREADED EVOLUTION PIPELINE
    # ========================================================================

    async def _start_workers(self):
        """Start background worker tasks for async pipeline with lifecycle tracking."""
        if self._workers_started:
            return
        self._workers_started = True
        # Fix 3: Track tasks for cancellation in clear()
        t_evolve = asyncio.create_task(self._worker_evolve())
        t_save = asyncio.create_task(self._worker_save())
        self._workers.extend([t_evolve, t_save])

    async def _worker_evolve(self):
        """Background worker for field evolution with throttling."""
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(self.evolve_q.get(), timeout=1.0)
                    inputs = payload.get("inputs", {})

                    # Throttling: Skip heavy meta-ops if backpressure high
                    backpressure_ok = self._backpressure_events < 3

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.step, inputs)

                    # Fix 10: Track recovery and update last successful step
                    self._last_successful_step = time.time()

                    if backpressure_ok and self.meta_controller:
                        # Safe execution for optimization
                        if self.meta_controller.should_optimize():
                            self._circuit_breakers["MetaControllerOptimize"].call(
                                self.meta_controller.optimize, self)

                    # Decay backpressure on success — also check if we can
                    # recover from degraded mode
                    if self._backpressure_events > 0:
                        self._backpressure_events = max(
                            0, self._backpressure_events - 1)
                        # Fix 10: Recover from degraded mode if backpressure
                        # has fully decayed
                        if self._backpressure_events == 0 and self._heavy_modules_degraded:
                            self._heavy_modules_degraded = False
                            self.stats["backpressure_degraded_mode"] = self.stats.get(
                                "backpressure_degraded_mode", 0) + 1
                            logger.info(
                                "Backpressure recovered — heavy modules re-enabled")
                        if self._backpressure_events == 0:
                            self.stats["last_backpressure_recovery"] = time.time()

                    self.evolve_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    self._backpressure_events += 1
                    logger.exception("Evolve worker error")
        except asyncio.CancelledError:
            logger.info("Evolve worker cancelled cleanly.")

    async def _worker_save(self):
        """Background worker for batch ingestion."""
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(self.save_q.get(), timeout=1.0)
                    embeddings = payload.get("embeddings")
                    contents = payload.get("contents")
                    modalities = payload.get("modalities")
                    if embeddings is not None and contents is not None:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            self.add_nodes_batch,
                            embeddings,
                            contents,
                            None,  # phases
                            None,  # node_ids
                            None,  # session_ids
                            modalities,
                            False,  # skip_projection
                        )
                    self._track_queue_depth()
                    self.save_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    logger.exception("Save worker error")
        except asyncio.CancelledError:
            logger.info("Save worker cancelled cleanly.")

    def _track_queue_depth(self):
        """Track async queue depths for monitoring."""
        if self.cfg.async_pipeline and self.evolve_q:
            self.stats["async_queue_depth"] = (
                self.evolve_q.qsize() +
                (self.save_q.qsize() if self.save_q else 0) +
                (self.query_q.qsize() if self.query_q else 0)
            )

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
        cd = self.config.asdict() if hasattr(self, 'config') else self.cfg.asdict()
        cd["consolidation_mode"] = _enum_value(
            cd.get("consolidation_mode"), "dialectical")
        cd["backend"] = _enum_value(cd.get("backend"), "numpy")
        cd["context_format"] = _enum_value(cd.get("context_format"), "plain")
        cd["eval_mode"] = _enum_value(cd.get("eval_mode"), "production")
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], set):
            cd["memory_tiers"] = list(cd["memory_tiers"])
        nodes_data = list(
            self.nodes.all_node_dicts()) if hasattr(
            self.nodes, "all_node_dicts") else [
            n.to_dict() for n in self.nodes.values()]
        data = {
            "_schema_version": "1.0",
            "config": cd,
            "nodes": nodes_data,
            "stats": self.stats}
        data.update(self._projection_mgr.get_state())
        if self.learnable_kernel:
            data["learnable_kernel"] = self.learnable_kernel.get_state()
        if self.meta_kernel:
            data["meta_kernel"] = self.meta_kernel.get_state()
        if self.healer:
            data["healer"] = self.healer.get_state()
        if self.causal_engine:
            data["causal_engine"] = self.causal_engine.get_state()
        if self.learned_consolidator is not None:
            data["learned_consolidator"] = self.learned_consolidator.get_state()
        if self.meta_controller:
            data["meta_controller"] = self.meta_controller.get_state()
        if self.federated:
            data["federated"] = self.federated.export_state()
        if self.meta_memory_eval:
            data["meta_memory_eval"] = self.meta_memory_eval.get_state()
        if self.security:
            data["security"] = self.security.get_state()
        if self.version_control:
            data["version_control"] = self.version_control.export_state()
        # Fix 4: Save missing subsystems
        if self.event_scheduler:
            data["event_scheduler"] = self.event_scheduler.get_state()
        if self.low_rank_compressor:
            data["low_rank_compressor"] = self.low_rank_compressor.get_state()
        if self.goal_tracker:
            data["goal_tracker"] = self.goal_tracker.get_state()
        if self.rl_feedback_loop:
            data["rl_feedback_loop"] = self.rl_feedback_loop.get_state()
        if self.predictor:
            data["predictor"] = self.predictor.get_state()
        if self.scenario_planner:
            data["scenario_planner"] = self.scenario_planner.get_state()
        if self.engram_manager:
            data["engram_manager"] = self.engram_manager.get_state()
        return data

    @classmethod
    def import_from_dict(cls, data: Dict, embedder: Callable):
        """Import field state from a dict (for UMP and other protocols)."""
        cd = data["config"]
        if isinstance(cd.get("consolidation_mode"), str):
            cd["consolidation_mode"] = ConsolidationMode(
                cd["consolidation_mode"])
        if isinstance(cd.get("backend"), str):
            cd["backend"] = Backend(cd["backend"])
        if isinstance(cd.get("context_format"), str):
            cd["context_format"] = ContextFormat(cd["context_format"])
        if isinstance(cd.get("eval_mode"), str):
            cd["eval_mode"] = EvalMode(cd["eval_mode"])
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], list):
            cd["memory_tiers"] = set(cd["memory_tiers"])
        if "causal_modeling" in cd and "causal_topological" not in cd:
            cd["causal_topological"] = cd.pop("causal_modeling")
        elif "causal_modeling" in cd:
            cd.pop("causal_modeling")
        valid_fields = set(
            f.name for f in RTMDKConfig.__dataclass_fields__.values())
        cd = {k: v for k, v in cd.items() if k in valid_fields}
        config = RTMDKConfig(**cd)
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from rtmdk.memory.core import RTMDKMemory
        memory = RTMDKMemory(config=config, embedder=embedder)

        memory.field._projection_mgr.load_state(data)
        if config.differentiable and "learnable_kernel" in data:
            memory.field.learnable_kernel.load_state(data["learnable_kernel"])
        if config.meta_adaptive and "meta_kernel" in data:
            memory.field.meta_kernel.load_state(data["meta_kernel"])
        if config.self_healing and "healer" in data:
            memory.field.healer.load_state(data["healer"])
        if config.causal_topological and "causal_engine" in data:
            memory.field.causal_engine.load_state(data["causal_engine"])
        if config.meta_controller and "meta_controller" in data:
            memory.field.meta_controller.load_state(data["meta_controller"])
        if config.federated and "federated" in data:
            memory.field.federated.import_state(data["federated"])
        if config.meta_memory and "meta_memory_eval" in data:
            memory.field.meta_memory_eval.load_state(data["meta_memory_eval"])
        if config.security_enabled and "security" in data:
            memory.field.security.load_state(data["security"])
        if config.version_control and "version_control" in data:
            memory.field.version_control.import_state(data["version_control"])
        # Fix 4: Load missing subsystems and reset historical stats
        if "event_scheduler" in data and memory.field.event_scheduler:
            memory.field.event_scheduler.load_state(data["event_scheduler"])
        if "low_rank_compressor" in data and memory.field.low_rank_compressor:
            memory.field.low_rank_compressor.load_state(
                data["low_rank_compressor"])
        if "goal_tracker" in data and memory.field.goal_tracker:
            memory.field.goal_tracker.load_state(data["goal_tracker"])
        if "rl_feedback_loop" in data and memory.field.rl_feedback_loop:
            memory.field.rl_feedback_loop.load_state(data["rl_feedback_loop"])
        if "predictor" in data and memory.field.predictor:
            memory.field.predictor.load_state(data["predictor"])
        if "scenario_planner" in data and memory.field.scenario_planner:
            memory.field.scenario_planner.load_state(data["scenario_planner"])
        if "engram_manager" in data and memory.field.engram_manager:
            memory.field.engram_manager.load_state(data["engram_manager"])

        # Reset historical metrics to avoid stale accumulated state (matches
        # import_field behavior)
        reset_keys = [
            "projection_updates", "self_sup_checks", "total_queries",
            "consolidations", "consolidation_validations", "blocked_consolidations",
            "healing_events", "healing_history", "field_stability",
            "tension_cache_hits", "tension_cache_misses", "tension_cache_hit_rate",
            "engram_retrievals", "engrams_created", "engrams_merged",
            "cross_modal_queries", "cross_modal_recall",
            "meta_optimizations", "meta_best_params",
            "federated_syncs", "federated_order_parameter",
            "crystallizations", "crystallized_clusters",
            "evaluations", "shadow_comparisons", "rollbacks",
            "ode_steps", "response_smoothness",
            "free_energy", "prediction_error", "surprise_level",
            "scenarios_generated", "avg_scenario_confidence",
            "privacy_budget_spent", "noise_std", "updates_clipped",
            "shard_hits", "shard_misses", "avg_shard_query_time_ms",
            "context_tokens_saved", "cognitive_compressions",
            "async_queue_depth", "async_backpressure_events",
            "active_goals", "completed_goals",
            "avg_rl_reward", "reward_trend",
            "attention_bias_applied", "compression_ratio", "compression_updates",
            "events_processed", "event_queue_depth",
            "recall_accuracy", "meta_reflections",
            "security_violations", "tension_spikes_blocked",
            "current_version", "n_versions",
            "clarifications_generated",
            "n_shards", "shard_distribution", "cross_shard_exchanges",
            "role_router_enabled",
            "field_integrity_issues",
            "plans_created", "hypotheses_verified", "tool_calls", "tool_misuse_rate",
            "ragas_overall",
            "tier_coherence",
        ]
        for key in reset_keys:
            if key in memory.field.stats:
                val = memory.field.stats[key]
                if isinstance(val, (int, float)):
                    memory.field.stats[key] = 0
                elif isinstance(val, dict):
                    memory.field.stats[key] = {}
                elif isinstance(val, list):
                    memory.field.stats[key] = []

        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory

# ============================================================================
# RTMDKMemory v7
# ============================================================================
