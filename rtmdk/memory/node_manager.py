"""NodeManager — node ingestion lifecycle (add, batch, delete, queue).

Extracted from RTMDKField to reduce monolithic field.py size.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import random
import re
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from rtmdk.engines.causal_extraction import extract_causal_edges_from_content
from rtmdk.nodes import MemoryNode

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_STOP_WORDS_EN = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "am",
        "it",
        "s",
        "t",
        "don",
        "didn",
        "wasn",
        "weren",
        "haven",
        "hasn",
        "hadn",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
        "isn",
        "aren",
        "ain",
        "ve",
        "ll",
        "re",
        "d",
        "m",
        "o",
        "y",
    }
)

_STOP_WORDS_RU = frozenset(
    {
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "человек",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "жизнь",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "говорил",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "один",
        "почти",
        "мой",
        "тем",
        "чтобы",
        "нее",
        "кажется",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "снова",
        "твой",
        "разве",
        "три",
        "эту",
        "моя",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
    }
)

_STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_RU

_CONTENT_HASH_CACHE: Dict[str, float] = {}

DEFAULT_ROLE = "default"


class NodeManager:
    """Handles node ingestion, batch operations, and deletion."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_text(content: Dict) -> str:
        text = content.get("text", "")
        if text:
            return text
        return f"{content.get('input_text', '')} {content.get('output_text', '')}".strip()

    def semantic_phase(
        self,
        session_id: Optional[str] = None,
        content: Optional[Dict] = None,
        modality: str = "text",
    ) -> float:
        parts = []
        if session_id:
            parts.append(f"s:{session_id}")
        if content:
            topic = content.get("topic", "")
            if topic:
                parts.append(f"t:{topic}")
            text = content.get("text", "") or content.get("input_text", "")
            if text:
                tokens = re.findall(r"[\w']+", text.lower())
                content_words = [w for w in tokens if w not in _STOP_WORDS and len(w) > 2]
                if content_words:
                    seen = set()
                    deduped = []
                    for w in content_words:
                        if w not in seen:
                            seen.add(w)
                            deduped.append(w)
                    words = deduped[:3]
                else:
                    words = tokens[:3]
                if words:
                    parts.append(f"w:{'_'.join(words)}")
        parts.append(f"m:{modality}")

        seed_text = "|".join(parts)
        cached = self.field._semantic_phase_cache.get(seed_text)
        if cached is not None:
            return cached
        h = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
        base = (int(h, 16) % 6283) / 1000.0
        rng = random.Random(h)
        spread = rng.uniform(-0.15, 0.15)
        result = (base + spread) % (2 * math.pi)
        self.field._semantic_phase_cache[seed_text] = result
        return result

    def get_phase(
        self,
        session_id: Optional[str] = None,
        embedding: Optional[NDArray] = None,
        modality: str = "text",
        content: Optional[Dict] = None,
    ) -> float:
        phase = self.semantic_phase(session_id, content, modality)
        cfg = self.field.cfg
        if cfg.cross_modal and modality in cfg.modal_phase_offsets:
            phase += cfg.modal_phase_offsets[modality]
        elif cfg.multimodal and modality in cfg.modality_phase_shifts:
            phase += cfg.modality_phase_shifts[modality]
        return phase % (2 * np.pi)

    # ------------------------------------------------------------------
    # Single add
    # ------------------------------------------------------------------
    def add_node(
        self,
        embedding: NDArray,
        content: Dict,
        phase: Optional[float] = None,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        modality: str = "text",
        skip_projection: bool = False,
        modal_embedding: Optional[NDArray] = None,
    ) -> str:
        f = self.field
        _rate_limit = getattr(f.cfg, "rate_limit_nodes_per_sec", 100)
        _env_override = os.environ.get("RTMDK_ADD_RATE_LIMIT")
        if _env_override is not None:
            _rate_limit = int(_env_override)
        if _rate_limit > 0:
            now = time.time()
            while f._add_node_timestamps and f._add_node_timestamps[0] < now - 1.0:
                f._add_node_timestamps.popleft()
            if len(f._add_node_timestamps) >= _rate_limit:
                from rtmdk.memory.utils import SecurityViolationError

                raise SecurityViolationError(f"Rate limit exceeded: max {_rate_limit} nodes/second")
            f._add_node_timestamps.append(now)

        try:
            from rtmdk.production.sanitization import validate_embedding

            embedding = validate_embedding(embedding)
        except Exception as exc:
            raise ValueError(f"Invalid embedding: {exc}")

        if f.security:
            text = content.get("text", "")
            input_text = content.get("input_text", "")
            output_text = content.get("output_text", "")
            for field_text in [text, input_text, output_text]:
                if field_text:
                    validation = f.security.validate_node_content(field_text)
                    if not validation["is_safe"]:
                        f.stats["security_violations"] += 1
                        logger.warning(f"Security violation in add_node: {validation['violations']}")
                        from rtmdk.memory.utils import SecurityViolationError

                        raise SecurityViolationError(f"Security violation: {validation['violations']}")

        nid = node_id or f"n_{len(f.nodes)}_{int(time.time() * 1000)}"
        if skip_projection:
            if len(embedding) != f.cfg.latent_dim:
                raise ValueError(
                    f"skip_projection=True but embedding dim {len(embedding)} != " f"latent_dim {f.cfg.latent_dim}"
                )
            latent = embedding
        elif len(embedding) == f.cfg.latent_dim:
            latent = embedding.astype(np.float32)
        else:
            latent = f._projection_mgr.update_projection(embedding)
            if f._projection_mgr.projection_learner is not None:
                f.stats["projection_updates"] += 1

        q_result = f._quant.quantize_with_meta(latent)
        if len(q_result) == 4:
            latent, latent_scale, latent_zero_point, latent_scale_array = q_result
        else:
            latent, latent_scale, latent_zero_point = q_result
            latent_scale_array = None

        if phase is None:
            phase = self.get_phase(session_id, embedding, modality, content)

        emb_norm = float(np.linalg.norm(embedding))
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
            latent_scale_array=latent_scale_array.astype(np.float32) if latent_scale_array is not None else None,
            modal_embedding=modal_embedding.astype(np.float32) if modal_embedding is not None else None,
        )

        if f.kalman_filter is not None:
            node.covariance = f.kalman_filter.init_covariance()

        if f.cfg.cross_modal:
            node.modal_embedding = embedding.copy()

        role = DEFAULT_ROLE
        if f.role_router:
            text = content.get("text", "")
            explicit_role = content.get("role") or content.get("tier_role")
            role = f.role_router.add_node(nid, text, role=explicit_role)
            node.role = role

        f.nodes[nid] = node
        existing_nids = set(f.node_index)
        if nid not in existing_nids:
            f.node_index.append(nid)
        f.stats["total_adds"] += 1

        causal_edges = extract_causal_edges_from_content(content)
        if causal_edges:
            for effect, cause, strength in causal_edges:
                node.causal_strength[cause] = max(node.causal_strength.get(cause, 0.0), strength)
                node.causal_parents.append(cause)
            f.stats.setdefault("causal_edges_extracted", 0)
            f.stats["causal_edges_extracted"] += len(causal_edges)

        if f._cached_positions is not None:
            try:
                f._cached_positions = np.vstack([f._cached_positions, latent.reshape(1, -1)])
                _ph = phase if phase is not None else self.get_phase(session_id, embedding, modality, content)
                f._cached_phases = np.append(f._cached_phases, _ph)
                f._cached_amplitudes = np.append(f._cached_amplitudes, amplitude)
                f._cached_saliences = np.append(f._cached_saliences, salience)
                f._cached_modal_weights = np.append(f._cached_modal_weights, 1.0)
                f._cached_gates = np.append(f._cached_gates, 1.0)
                f._cached_causal_boost = np.append(f._cached_causal_boost, 1.0)
            except Exception:
                logger.warning("Incremental cache append failed, falling back to full rebuild", exc_info=True)
                f._cache_dirty = True
        else:
            f._cache_dirty = True

        f._invalidate_tension_cache(nid)

        if f.query_cache is not None:
            f.query_cache.clear()

        if f.role_router:
            f.stats["n_shards"] = len(f.role_router.shards)
            f.stats["shard_distribution"] = {r: len(s.node_ids) for r, s in f.role_router.shards.items()}
            f.stats["role_router_enabled"] = True

        if f.cfg.use_hnsw and f.hnsw_index:
            f._index_mgr.hnsw_insert(nid, latent)
        if f.cfg.bm25_fallback and f.bm25_index:
            text = self.extract_text(content)
            if text:
                f._index_mgr.bm25_add(nid, text)

        if f.event_scheduler:
            f.event_scheduler.enqueue("node_added", {"node_id": nid, "modality": modality})

        f.wal.append_add_node(nid, content, modality, embedding=latent.tolist())
        f._dirty = True
        return nid

    # ------------------------------------------------------------------
    # Batch add
    # ------------------------------------------------------------------
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
        f = self.field
        if len(embeddings) != len(contents):
            raise ValueError(f"embeddings length {len(embeddings)} != contents length {len(contents)}")
        n = len(embeddings)
        if n == 0:
            return []

        if skip_projection:
            if embeddings.shape[1] != f.cfg.latent_dim:
                raise ValueError(
                    f"skip_projection=True but embedding dim {embeddings.shape[1]} != " f"latent_dim {f.cfg.latent_dim}"
                )
            latents = embeddings.astype(np.float32)
        elif embeddings.shape[1] == f.cfg.latent_dim:
            latents = embeddings.astype(np.float32)
        else:
            latents = f._project_batch(embeddings)

        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        latents = latents / np.maximum(norms, 1e-8)

        _q_results = [f._quant.quantize_with_meta(vec) for vec in latents]
        latents = np.array([r[0] for r in _q_results])
        _latent_scales = [r[1] for r in _q_results]
        _latent_zps = [r[2] for r in _q_results]
        _latent_scale_arrays = [r[3] if len(r) > 3 else None for r in _q_results]

        if phases is None:
            base = (time.time() * 0.01) % (2 * np.pi)
            if modalities:
                if f.cfg.cross_modal:
                    phases = np.array(
                        [(base + f.cfg.modal_phase_offsets.get(m, 0.0)) % (2 * np.pi) for m in modalities],
                        dtype=np.float32,
                    )
                elif f.cfg.multimodal:
                    phases = np.array(
                        [(base + f.cfg.modality_phase_shifts.get(m, 0.0)) % (2 * np.pi) for m in modalities],
                        dtype=np.float32,
                    )
                else:
                    phases = np.full(n, base, dtype=np.float32)
            else:
                phases = np.full(n, base, dtype=np.float32)
        else:
            phases = np.asarray(phases, dtype=np.float32)

        emb_norms = np.linalg.norm(embeddings, axis=1)
        saliences = np.clip(emb_norms / 20.0, 0.3, 1.0).astype(np.float32)
        amplitudes = np.clip(emb_norms / 15.0, 0.5, 1.0).astype(np.float32)

        now = time.time()
        base_idx = len(f.nodes)
        batch_nids: List[str] = []
        existing_nids = set(f.node_index)
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
                latent_scale_array=(
                    _latent_scale_arrays[i].astype(np.float32) if _latent_scale_arrays[i] is not None else None
                ),
            )
            if f.cfg.cross_modal:
                node.modal_embedding = embeddings[i].copy()
            if modal_embeddings is not None:
                node.modal_embedding = modal_embeddings[i].astype(np.float32)
            f.nodes[nid] = node
            if nid not in existing_nids:
                f.node_index.append(nid)
                existing_nids.add(nid)
            f.stats["total_adds"] += 1

        if f._cached_positions is not None:
            f._cached_positions = np.vstack([f._cached_positions, latents])
            f._cached_phases = np.append(f._cached_phases, phases)
            f._cached_amplitudes = np.append(f._cached_amplitudes, amplitudes)
            f._cached_saliences = np.append(f._cached_saliences, saliences)
            f._cached_modal_weights = np.append(f._cached_modal_weights, np.ones(n, dtype=np.float32))
            f._cached_gates = np.append(f._cached_gates, np.ones(n, dtype=np.float32))
            f._cached_causal_boost = np.append(f._cached_causal_boost, np.ones(n, dtype=np.float32))
        else:
            f._cache_dirty = True

        if f.cfg.use_hnsw and f.hnsw_index:
            f._index_mgr.hnsw_insert_batch(batch_nids, latents)

        if f.cfg.bm25_fallback and f.bm25_index:
            for i, nid in enumerate(batch_nids):
                text = self.extract_text(contents[i])
                if text:
                    f._index_mgr.bm25_add(nid, text)

        if f.query_cache is not None:
            f.query_cache.clear()

        f.wal.append(
            "add_nodes_batch",
            {
                "count": n,
                "node_ids": batch_nids,
                "contents": contents,
                "embeddings": [vec.tolist() for vec in latents],
                "modalities": modalities if modalities else ["text"] * n,
            },
        )
        f._dirty = True
        return batch_nids

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_nodes(self, node_ids: List[str]) -> None:
        f = self.field
        for nid in node_ids:
            if nid in f.nodes:
                del f.nodes[nid]
        f.node_index = [nid for nid in f.node_index if nid in f.nodes]
        if f.cfg.use_hnsw and f.hnsw_index:
            for nid in node_ids:
                f._index_mgr.hnsw_remove(nid)
        f.wal.append_delete(node_ids)
        f._cache_dirty = True
        if f.query_cache is not None:
            f.query_cache.clear()

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------
    def queue_add_nodes(
        self,
        embeddings: NDArray,
        contents: List[Dict],
        modalities: Optional[List[str]] = None,
    ) -> None:
        f = self.field
        if not f.cfg.async_pipeline or f.save_q is None:
            self.add_nodes_batch(embeddings, contents, modalities=modalities)
            return
        payload = {"embeddings": embeddings, "contents": contents, "modalities": modalities}
        try:
            f.save_q.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("save_q full — falling back to synchronous add_nodes_batch")
            self.add_nodes_batch(embeddings, contents, modalities=modalities)
