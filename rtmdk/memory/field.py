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

from rtmdk.support.production import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
from rtmdk.memory.projection_manager import ProjectionManager
from rtmdk.memory.consolidation_manager import ConsolidationManager
from rtmdk.memory.query_manager import QueryManager
from rtmdk.memory.routing_manager import RoutingManager
from rtmdk.memory.scheduler import StepScheduler
from rtmdk.memory.topology_manager import TopologyManager
from rtmdk.memory.async_pipeline_manager import AsyncPipelineManager
from rtmdk.memory.crystallization_manager import CrystallizationManager
from rtmdk.memory.node_manager import NodeManager
from rtmdk.memory.cognitive_manager import CognitiveManager
from rtmdk.support.agents import AgentPlanner, HypothesisVerifier, ToolRouter
from rtmdk.support.healer import TopologyHealer
from rtmdk.support.meta_adaptive import MetaAdaptiveKernel
from rtmdk.engines.causal import CausalInferenceEngine

from rtmdk.engines.privacy import DifferentialPrivacy
from rtmdk.engines.predictive import PredictiveCodingModel
from rtmdk.memory.geometry import exp_map_poincare
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

import hashlib
from collections import deque
from typing import List, Dict, Optional, Tuple, Callable, Any, Set
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from scipy.spatial import cKDTree
import logging

# Extracted engine classes (kept in sync with rtmdk/support/ modules)
from rtmdk.support.kuramoto import FederatedRTMDK

try:

    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

from rtmdk.memory.conformal import ConformalCalibrator

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

        self._batch_resonance_fn = None  # set by QueryManager in its __init__

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
        self._query_mgr = QueryManager(self)
        self._routing_mgr = RoutingManager(self)
        self._topology_mgr = TopologyManager(self)
        self._async_pipeline_mgr = AsyncPipelineManager(self)
        self._crystallization_mgr = CrystallizationManager(self)
        self._node_mgr = NodeManager(self)
        self._cognitive_mgr = CognitiveManager(self)
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
        return self._query_mgr._effective_bandwidth

    @property
    def _effective_pc(self) -> float:
        return self._query_mgr._effective_pc

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
