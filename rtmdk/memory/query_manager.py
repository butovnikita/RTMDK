"""QueryManager — retrieval engine extracted from RTMDKField.

Handles all query paths (BM25 first-stage, HNSW, sparse routing,
vectorized fallback), batch resonance computation, and post-query
filtering (conformal, adaptive top-k, attention bias).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import pdist

from rtmdk.memory.geometry import poincare_dist
from rtmdk.memory.utils import apply_attention_bias

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from rtmdk.nodes import MemoryNode
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QueryManager:
    """All query and resonance computation logic."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field
        # Pre-select batch resonance backend to avoid branching in hot path
        if field.gpu_backend and field.gpu_backend.available:
            field._batch_resonance_fn = self._batch_resonance_torch
        else:
            field._batch_resonance_fn = self._batch_resonance_numpy

    # ------------------------------------------------------------------
    # Properties — adaptive params
    # ------------------------------------------------------------------
    @property
    def _effective_bandwidth(self) -> float:
        f = self.field
        if f.adaptive_bw is not None and f.adaptive_bw._best_bw is not None:
            return f.adaptive_bw._best_bw
        return f.cfg.bandwidth

    @property
    def _effective_pc(self) -> float:
        f = self.field
        if f._adaptive_pc_value is not None:
            return f._adaptive_pc_value
        if f.meta_kernel is not None:
            return f.meta_kernel.get_phase_coupling()
        return f.cfg.phase_coupling

    def _ensure_adaptive_pc(self, query_latent: NDArray) -> None:
        """Run once on first query to auto-tune phase coupling."""
        f = self.field
        if f._adaptive_pc_estimated or f._estimate_optimal_pc_fn is None:
            return
        if len(f.nodes) < 10:
            return
        try:
            nids = list(f.node_index)
            positions = np.array([f.nodes[nid].latent_pos for nid in nids])
            norms = np.linalg.norm(positions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            doc_embs = positions / norms
            sample_size = min(50, len(nids))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(nids), size=sample_size, replace=False)
            sample_queries = doc_embs[idx]
            sample_targets = idx
            sims = sample_queries @ doc_embs.T
            np.fill_diagonal(sims, -1.0)
            ranks = np.argmax(sims, axis=1)
            hits = int(np.sum(ranks == sample_targets))
            recall1 = hits / len(sample_targets)
            threshold = getattr(f.cfg, "adaptive_pc_disable_threshold", 0.93)
            if recall1 >= threshold:
                f._adaptive_pc_value = 0.0
                logger.info("Adaptive PC: embeddings strong (recall@1=%.2f >= %.2f) -> pc=0.0", recall1, threshold)
            else:
                pc = f._estimate_optimal_pc_fn(doc_embs, sample_queries, sample_targets, sample_size=sample_size)
                f._adaptive_pc_value = float(pc)
                logger.info("Adaptive phase coupling estimated: pc=%.2f", f._adaptive_pc_value)
        except Exception as exc:
            logger.warning("Adaptive PC estimation failed: %s", exc)
        finally:
            f._adaptive_pc_estimated = True

    # ------------------------------------------------------------------
    # Cache / filter helpers
    # ------------------------------------------------------------------
    def _query_cache_key(
        self,
        query_latent: NDArray,
        phase: float,
        top_k: int,
        modality: str,
        session_id: Optional[str],
    ) -> str:
        vec = query_latent.astype(np.float16).tobytes()
        raw = vec + f"|{phase:.4f}|{top_k}|{modality}|{session_id or ''}".encode()
        return hashlib.md5(raw).hexdigest()

    def _apply_adaptive_top_k(
        self, results: List[Tuple[str, float, MemoryNode]]
    ) -> List[Tuple[str, float, MemoryNode]]:
        if not results:
            return results
        top_score = results[0][1]
        if top_score >= 0.95:
            return results[:1]
        elif top_score >= 0.80:
            return results[:3]
        else:
            return results[:5]

    def _apply_conformal_filter(
        self, results: List[Tuple[str, float, MemoryNode]]
    ) -> List[Tuple[str, float, MemoryNode]]:
        f = self.field
        if not f.cfg.conformal_prediction or f.conformal_calibrator is None:
            return results
        if f.conformal_calibrator.n_calibrated < f.cfg.conformal_min_calib:
            return results
        scores = [score for _, score, _ in results]
        nids = [nid for nid, _, _ in results]
        pred_set, confidence, threshold = f.conformal_calibrator.predict(scores, nids)
        f.stats["conformal_threshold"] = threshold
        f.stats["conformal_confidence"] = confidence
        f.stats["conformal_prediction_set_size"] = len(pred_set)
        pred_set_lookup = set(pred_set)
        return [(nid, score, node) for nid, score, node in results if nid in pred_set_lookup]

    # ------------------------------------------------------------------
    # Single-node resonance
    # ------------------------------------------------------------------
    def _resonance_response(
        self,
        query_latent: NDArray,
        query_phase: float,
        node: MemoryNode,
        query_modality: str = "text",
    ) -> float:
        f = self.field
        resp = f._resonance_engine.single_response(query_latent, query_phase, node, query_modality)
        if f.cfg.hyperbolic:
            dist = poincare_dist(query_latent, node.latent_pos, f.cfg.ball_radius)
            f.stats["avg_hyperbolic_dist"] = 0.99 * f.stats["avg_hyperbolic_dist"] + 0.01 * dist
        return resp

    # ------------------------------------------------------------------
    # Batch resonance variants
    # ------------------------------------------------------------------
    def _batch_resonance(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        return self.field._batch_resonance_fn(query_latents, query_phases, node_ids)

    def _batch_resonance_nodes(self, query_latents: NDArray, query_phases: NDArray, nodes: List[Any]) -> NDArray:
        if not nodes:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        node_positions = np.array([n.latent_pos for n in nodes])
        node_phases = np.array([n.phase for n in nodes])
        node_amplitudes = np.array([n.amplitude for n in nodes])
        node_saliences = np.array([n.salience for n in nodes])
        return self.field._resonance_engine.batch_response_numpy(
            query_latents,
            query_phases,
            node_positions,
            node_phases,
            node_amplitudes,
            node_saliences,
        )

    def _batch_resonance_numpy(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        f = self.field
        node_positions = np.array([f.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([f.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array([f.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array([f.nodes[nid].salience for nid in node_ids])
        return f._resonance_engine.batch_response_numpy(
            query_latents,
            query_phases,
            node_positions,
            node_phases,
            node_amplitudes,
            node_saliences,
        )

    def _batch_resonance_cached(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        f = self.field
        if getattr(f, "_node_id_to_cached_idx", None) is None or f._cache_dirty:
            f._build_node_cache()
        mapping = f._node_id_to_cached_idx
        indices = np.array([mapping[nid] for nid in node_ids if nid in mapping], dtype=np.int32)
        if len(indices) == 0:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        cached_positions = f._cached_positions
        if cached_positions is None:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        # Invariant: _build_node_cache() populates all cached arrays together
        cached_phases = f._cached_phases
        cached_amplitudes = f._cached_amplitudes
        cached_saliences = f._cached_saliences
        assert (
            cached_phases is not None and cached_amplitudes is not None and cached_saliences is not None
        ), "node cache arrays are built together"
        # Fast path for int8 cached positions: avoid cdist float64 cast
        cached_norms_sq = f._cached_norms_sq
        if cached_positions.dtype == np.int8 and cached_norms_sq is not None:
            # Invariant: int8 cache always has per-vector scales
            cached_scales = f._cached_scales
            assert cached_scales is not None, "int8 node cache requires scales"
            return f._resonance_engine.batch_response_numpy_int8(
                query_latents,
                query_phases,
                cached_positions[indices],
                cached_norms_sq[indices],
                cached_scales[indices],
                cached_phases[indices],
                cached_amplitudes[indices],
                cached_saliences[indices],
            )
        return f._resonance_engine.batch_response_numpy(
            query_latents,
            query_phases,
            cached_positions[indices],
            cached_phases[indices],
            cached_amplitudes[indices],
            cached_saliences[indices],
        )

    def _batch_resonance_torch(self, query_latents: NDArray, query_phases: NDArray, node_ids: List[str]) -> NDArray:
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        f = self.field
        node_positions = np.array([f.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([f.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array([f.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array([f.nodes[nid].salience for nid in node_ids])
        return f._resonance_engine.batch_response_torch(
            query_latents,
            query_phases,
            node_positions,
            node_phases,
            node_amplitudes,
            node_saliences,
        )

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
        pc=None,
    ):
        f = self.field
        return f._resonance_engine.chunk_response(
            positions,
            phases,
            amplitudes,
            saliences,
            modal_weights,
            gates,
            causal_boost,
            query_latent,
            query_phase,
            bw,
            use_gates=f.cfg.soft_gates,
            use_causal=f.causal_engine is not None,
            pc=pc,
        )

    # ------------------------------------------------------------------
    # Vectorized query engine
    # ------------------------------------------------------------------
    def _query_vectorized(
        self,
        query_latent: NDArray,
        query_phase: float,
        top_k: int,
        modality: str,
        session_id: Optional[str],
        t0: float,
    ) -> List[Tuple[str, float, MemoryNode]]:
        f = self.field
        cfg = f.cfg
        min_response = cfg.min_response
        gpu_batch_size = cfg.gpu_batch_size
        attention_bias = getattr(cfg, "attention_bias", False)
        bias_temperature = getattr(cfg, "bias_temperature", 1.0)
        bw = cfg.bandwidth
        pc = float(f._resonance_engine._effective_pc)
        self._ensure_adaptive_pc(query_latent)

        n_nodes = len(f.node_index)
        if n_nodes == 0:
            return []

        # Snapshot all cached arrays atomically under the write lock.
        # Concurrent add/delete triggers a cache rebuild whose per-attribute
        # assignments could otherwise be interleaved with these reads,
        # producing length-mismatched arrays (broadcast ValueError downstream).
        with f._write_lock:
            if f._cache_dirty or f._cached_positions is None:
                f._build_node_cache()

            # Invariant: _build_node_cache() populates all cached arrays together
            cached_positions = f._cached_positions
            cached_phases = f._cached_phases
            cached_amplitudes = f._cached_amplitudes
            cached_saliences = f._cached_saliences
            cached_modal_weights = f._cached_modal_weights
            cached_gates = f._cached_gates
            cached_causal_boost = f._cached_causal_boost
            assert (
                cached_positions is not None
                and cached_phases is not None
                and cached_amplitudes is not None
                and cached_saliences is not None
                and cached_modal_weights is not None
                and cached_gates is not None
                and cached_causal_boost is not None
            ), "node cache arrays are built together"

            session_mask = None
            if session_id and session_id != "default":
                session_mask = np.array(
                    [f.nodes[nid].content.get("session") == session_id for nid in f.node_index], dtype=bool
                )
                n_session = session_mask.sum()
                if 0 < n_session < n_nodes * 0.3:
                    positions = cached_positions[session_mask]
                    phases = cached_phases[session_mask]
                    amplitudes = cached_amplitudes[session_mask]
                    saliences = cached_saliences[session_mask]
                    modal_weights = cached_modal_weights[session_mask]
                    gates = cached_gates[session_mask]
                    causal_boost = cached_causal_boost[session_mask]
                    session_indices = np.where(session_mask)[0]
                else:
                    positions = cached_positions
                    phases = cached_phases
                    amplitudes = cached_amplitudes
                    saliences = cached_saliences
                    modal_weights = cached_modal_weights
                    gates = cached_gates
                    causal_boost = cached_causal_boost
                    session_indices = None
            else:
                positions = cached_positions
                phases = cached_phases
                amplitudes = cached_amplitudes
                saliences = cached_saliences
                modal_weights = cached_modal_weights
                gates = cached_gates
                causal_boost = cached_causal_boost
                session_indices = None

        batch_size = gpu_batch_size
        n = len(positions)

        if n <= batch_size:
            resp = self._compute_resonance_chunk(
                positions,
                phases,
                amplitudes,
                saliences,
                modal_weights,
                gates,
                causal_boost,
                query_latent,
                query_phase,
                bw=bw,
                pc=pc,
            )
            if session_id and session_id != "default" and session_mask is not None and session_indices is None:
                resp = resp * (1.0 + 0.5 * session_mask.astype(np.float32))
            above_threshold = resp >= min_response
            indices = np.where(above_threshold)[0]
            if len(indices) == 0:
                f.stats["total_queries"] += 1
                return []
            if session_indices is not None:
                indices = session_indices[indices]
            scores = resp[indices] if session_indices is None else resp[np.where(above_threshold)[0]]
            n_results = min(len(indices), top_k * 2)
            if len(indices) > top_k * 3:
                if n_results < len(scores):
                    partition_idx = np.argpartition(scores, -n_results)[-n_results:]
                    top_local = partition_idx[np.argsort(scores[partition_idx])[::-1][:top_k]]
                else:
                    top_local = np.argsort(scores)[::-1][:top_k]
                top_indices = indices[top_local]
                top_scores = scores[top_local]
            else:
                sorted_order = np.argsort(scores)[::-1][:top_k]
                top_indices = indices[sorted_order]
                top_scores = scores[sorted_order]
        else:
            candidates: List[Tuple[int, float]] = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                resp = self._compute_resonance_chunk(
                    positions[start:end],
                    phases[start:end],
                    amplitudes[start:end],
                    saliences[start:end],
                    modal_weights[start:end],
                    gates[start:end],
                    causal_boost[start:end],
                    query_latent,
                    query_phase,
                    bw=bw,
                    pc=pc,
                )
                if session_id and session_id != "default" and session_mask is not None and session_indices is None:
                    resp = resp * (1.0 + 0.5 * session_mask[start:end].astype(np.float32))
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
                        top_local = part_idx[np.argsort(scores[part_idx])[::-1][:top_k]]
                    else:
                        top_local = np.argsort(scores)[::-1][:top_k]
                else:
                    top_local = np.argsort(scores)[::-1][:top_k]
                for li in top_local:
                    candidates.append((int(local_idx[li]), float(scores[li])))
            if not candidates:
                f.stats["total_queries"] += 1
                return []
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_k]
            if session_indices is not None:
                top_indices = np.array([session_indices[idx] for idx, _ in top_candidates], dtype=np.int64)
            else:
                top_indices = np.array([idx for idx, _ in top_candidates], dtype=np.int64)
            top_scores = np.array([score for _, score in top_candidates], dtype=np.float32)

        results = []
        for i in range(len(top_indices)):
            idx = top_indices[i]
            # Defensive: node_index may shrink due to concurrent delete_nodes
            if idx >= len(f.node_index):
                continue
            nid = f.node_index[idx]
            node = f.nodes.get(nid)
            if node is None:
                continue
            node.last_resonated = time.time()
            results.append((nid, float(top_scores[i]), node))

        f.stats["total_queries"] += 1
        if results:
            f.stats["avg_response"] = 0.9 * f.stats["avg_response"] + 0.1 * results[0][1]
            if f.goal_tracker:
                for nid, resp_val, node in results:
                    node.goal_relevance = f.goal_tracker.get_goal_relevance(nid)
            if attention_bias:
                results = apply_attention_bias(results, bias_temperature)
                f.stats["attention_bias_applied"] += 1

        elapsed_ms = (time.time() - t0) * 1000
        if cfg.sparse_routing:
            f.stats["avg_shard_query_time_ms"] = 0.95 * f.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms

        return results

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------
    def query(
        self,
        embedding: NDArray,
        phase: float = 0.0,
        top_k: Optional[int] = None,
        modality: str = "text",
        session_id: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Tuple[str, float, MemoryNode]]:
        f = self.field
        t0 = time.time()
        cfg = f.cfg
        top_k = top_k or cfg.top_k
        query_latent = f._project(embedding)
        self._ensure_adaptive_pc(query_latent)

        # Query cache check
        if f.query_cache is not None:
            cache_key = self._query_cache_key(query_latent, phase, top_k, modality, session_id)
            cached = f.query_cache.get_raw(cache_key)
            if cached is not None:
                f.stats.setdefault("query_cache_hits", 0)
                f.stats["query_cache_hits"] += 1
                return cached
            f.stats.setdefault("query_cache_misses", 0)
            f.stats["query_cache_misses"] += 1

        # P0: BM25 first-stage pre-filtering
        if (
            cfg.bm25_first_stage_k > 0
            and query_text
            and f.bm25_index is not None
            and len(f.nodes) > cfg.bm25_first_stage_k
        ):
            candidate_ids = [
                nid for nid, _ in f._index_mgr.bm25_search(query_text, cfg.bm25_first_stage_k) if nid in f.nodes
            ]
            if candidate_ids:
                scores = self._batch_resonance(
                    query_latent[np.newaxis, :],
                    np.array([phase], dtype=np.float32),
                    candidate_ids,
                )[0]
                results = []
                for idx, nid in enumerate(candidate_ids):
                    node = f.nodes[nid]
                    resp = float(scores[idx]) * (
                        1.3 if session_id and node.content.get("session") == session_id else 1.0
                    )
                    if resp >= cfg.min_response:
                        results.append((nid, resp, node))
                        node.last_resonated = time.time()
                results.sort(key=lambda x: x[1], reverse=True)
                f.stats["bm25_first_stage_hits"] = f.stats.get("bm25_first_stage_hits", 0) + 1
            else:
                results = []
        # HNSW fast path
        elif cfg.use_hnsw:
            candidate_ids = []
            n_pos = f._index_mgr.hnsw_count()
            if n_pos > getattr(cfg, "hnsw_min_nodes", 50):
                hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
                candidate_ids = f._index_mgr.hnsw_search(query_latent, hnsw_k)
                candidate_ids = [nid for nid in candidate_ids if nid in f.nodes]
            if candidate_ids:
                scores = self._batch_resonance_cached(
                    query_latent[np.newaxis, :],
                    np.array([phase], dtype=np.float32),
                    candidate_ids,
                )[0]
                if session_id and session_id != "default":
                    session_boosts = np.array(
                        [1.3 if f.nodes[nid].content.get("session") == session_id else 1.0 for nid in candidate_ids],
                        dtype=np.float32,
                    )
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
                        node = f.nodes[nid]
                        node.last_resonated = time.time()
                        results.append((nid, float(filtered_scores[ti]), node))
            elif n_pos <= getattr(cfg, "hnsw_min_nodes", 50):
                results = self._query_vectorized(query_latent, phase, top_k, modality, session_id, t0)
            else:
                results = []
        elif cfg.sparse_routing and f._index_mgr.shard_centers is not None and len(f.nodes) > cfg.num_shards * 2:
            active_shards = f._route_query(query_latent, cfg.top_shards)
            candidate_ids = [nid for nid in f.node_index if f._get_node_shard(nid) in active_shards]
            search_nodes = [(nid, f.nodes[nid]) for nid in candidate_ids if nid in f.nodes]
            f.stats["shard_hits"] += len(candidate_ids)
        else:
            results = self._query_vectorized(query_latent, phase, top_k, modality, session_id, t0)

        # Track 2: Fallback to warm/cold tiers
        if f._tiered_store is not None and cfg.tiered_fallback_enabled and len(results) < top_k:
            needed = top_k - len(results)
            warm_ids = f._tiered_store.warm_ids()
            if warm_ids:
                warm_nodes = f._tiered_store.peek_batch(warm_ids)
                if warm_nodes:
                    scores = self._batch_resonance_nodes(
                        query_latent[np.newaxis, :],
                        np.array([phase], dtype=np.float32),
                        warm_nodes,
                    )[0]
                    for idx, node in enumerate(warm_nodes):
                        resp = float(scores[idx]) * (
                            1.3 if session_id and node.content.get("session") == session_id else 1.0
                        )
                        if resp >= cfg.min_response:
                            results.append((node.id, resp, node))
                            node.last_resonated = time.time()
                    results.sort(key=lambda x: x[1], reverse=True)
            if len(results) < top_k:
                cold_ids = f._tiered_store.cold_ids()
                if cold_ids:
                    sample_size = min(len(cold_ids), needed * 5)
                    sample_ids = f._rng.choice(cold_ids, size=sample_size, replace=False).tolist()
                    cold_nodes = f._tiered_store.peek_batch(sample_ids)
                    if cold_nodes:
                        scores = self._batch_resonance_nodes(
                            query_latent[np.newaxis, :],
                            np.array([phase], dtype=np.float32),
                            cold_nodes,
                        )[0]
                        for idx, node in enumerate(cold_nodes):
                            resp = float(scores[idx]) * (
                                1.3 if session_id and node.content.get("session") == session_id else 1.0
                            )
                            if resp >= cfg.min_response:
                                results.append((node.id, resp, node))
                                node.last_resonated = time.time()
                        results.sort(key=lambda x: x[1], reverse=True)
                        results = results[:top_k]

        # Fallback loop path (should rarely reach here)
        if "results" not in locals():
            search_nodes = [(nid, f.nodes[nid]) for nid in f.node_index if nid in f.nodes]
            if cfg.sparse_routing:
                f.stats["shard_misses"] += 1

            if cfg.hyperbolic and len(search_nodes) > top_k * 5:
                query_norm = np.linalg.norm(query_latent)
                if query_norm >= cfg.ball_radius:
                    query_latent = query_latent * (cfg.ball_radius - 1e-6) / max(query_norm, 1e-8)
                prefiltered = []
                for nid, node in search_nodes:
                    node_norm = np.linalg.norm(node.latent_pos)
                    node_pos = node.latent_pos
                    if node_norm >= cfg.ball_radius:
                        node_pos = node.latent_pos * (cfg.ball_radius - 1e-6) / max(node_norm, 1e-8)
                    hdist = poincare_dist(query_latent, node_pos, cfg.ball_radius)
                    if hdist < 3.0:
                        prefiltered.append((nid, node))
                if len(prefiltered) > 0:
                    search_nodes = prefiltered

            results = []
            for nid, node in search_nodes:
                resp = self._resonance_response(query_latent, phase, node, query_modality=modality)
                if session_id and node.content.get("session") == session_id:
                    resp *= 1.3
                if resp >= cfg.min_response:
                    results.append((nid, resp, node))
                    node.last_resonated = time.time()
            results.sort(key=lambda x: x[1], reverse=True)

        # P1.3: Adaptive bandwidth re-optimisation
        if f.adaptive_bw is not None and f.adaptive_bw.should_optimize():
            if f._cached_positions is not None and len(f.nodes) >= f.adaptive_bw.min_nodes:
                try:
                    optimal_bw = f.adaptive_bw.optimize(
                        f._cached_positions,
                        f._cached_phases,
                        f._cached_amplitudes,
                        f._cached_saliences,
                        top_k=cfg.top_k,
                    )
                    f.stats["adaptive_bw"] = optimal_bw
                except Exception:
                    logger.warning("Adaptive bandwidth optimisation failed", exc_info=True)

        f.stats["total_queries"] += 1

        if cfg.sparse_routing:
            elapsed_ms = (time.time() - t0) * 1000
            f.stats["avg_shard_query_time_ms"] = 0.95 * f.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms

        if cfg.cross_modal:
            f.stats["cross_modal_queries"] += 1
            if results:
                cm_scores = [n.cross_modal_score for _, _, n in results]
                f.stats["cross_modal_recall"] = 0.9 * f.stats["cross_modal_recall"] + 0.1 * float(np.mean(cm_scores))

        if len(results) == 0 and cfg.bm25_fallback and f.bm25_index:
            texts = []
            for nid in f.node_index[:100]:
                t = f._extract_text(f.nodes[nid].content)
                if t:
                    texts.append(t)
            fallback_query = query_text if query_text else " ".join(texts)
            if fallback_query:
                for doc_id, score in f._index_mgr.bm25_search(fallback_query, top_k):
                    if doc_id in f.nodes:
                        results.append((doc_id, score * 0.1, f.nodes[doc_id]))
                f.stats["bm25_fallbacks"] += 1

        if results:
            f.stats["avg_response"] = 0.9 * f.stats["avg_response"] + 0.1 * results[0][1]

        # Kalman uncertainty weighting
        if f.kalman_filter is not None and results:
            weighted = []
            for nid, score, node in results:
                if node.covariance is not None:
                    w = f.kalman_filter.uncertainty_weight(node.covariance)
                    score = score * w
                weighted.append((nid, score, node))
            results = weighted
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        # Goal relevance
        if f.goal_tracker and results:
            for nid, resp, node in results:
                node.goal_relevance = f.goal_tracker.get_goal_relevance(nid)

        # Attention bias
        if cfg.attention_bias and results:
            results = apply_attention_bias(results, cfg.bias_temperature)
            f.stats["attention_bias_applied"] += 1

        # Event-driven trigger
        if f.event_scheduler and results:
            f.event_scheduler.enqueue("query", {"top_score": results[0][1] if results else 0})

        # Meta-kernel recording
        if f.meta_kernel:
            f.meta_kernel.record_response(results[0][1] if results else 0.0)
            if len(results) >= 2:
                positions = np.array([n.latent_pos for _, _, n in results])
                valid = pdist(positions)
                density = 1.0 / (1.0 + np.mean(valid)) if len(valid) > 0 else 0.0
                f.meta_kernel.record_semantic_density(float(density))
            if len(results) >= 2:
                responses = np.array([r for _, r, _ in results])
                normalized = responses / (np.sum(responses) + 1e-8)
                entropy = float(-np.sum(normalized * np.log(normalized + 1e-8)))
                f.meta_kernel.record_uncertainty(entropy)

        f._last_query_results = results

        if f.causal_engine and len(results) >= 2:
            f.causal_engine.record_cooccurrence(results[0][0], results[1][0])
            active = [nid for nid, resp, _ in results if resp > cfg.min_response * 0.5]
            if active:
                f.causal_engine.record_observation(active)
                f._active_node_history.append(active)

        # Meta-memory recall tracking
        if f.meta_memory_eval and results:
            top_score = results[0][1]
            avg_age = np.mean([time.time() - n.created_at for _, _, n in results])
            f.meta_memory_eval.record_recall("", top_score, node_age=avg_age)
            f.stats["recall_accuracy"] = f.meta_memory_eval.evaluate_recall_accuracy()

        # Conformal prediction
        results = self._apply_conformal_filter(results)

        # SOT retrieval feedback
        if cfg.sot_retrieval_feedback and f._projection_mgr.has_sot and results:
            f._sot_retrieval_feedback(query_latent, results)

        # Adaptive top_k
        if cfg.adaptive_top_k:
            results = self._apply_adaptive_top_k(results)

        final = results[:top_k]

        # Store in cache
        if f.query_cache is not None:
            cache_key = self._query_cache_key(query_latent, phase, top_k, modality, session_id)
            f.query_cache.put_raw(cache_key, final)

        return final

    def query_batch(
        self,
        embeddings: NDArray,
        phase: float = 0.0,
        top_k: Optional[int] = None,
        modality: str = "text",
        session_id: Optional[str] = None,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float, MemoryNode]]]:
        f = self.field
        t0 = time.time()
        top_k = top_k or f.cfg.top_k
        n_queries = len(embeddings)

        query_latents = np.array([f._project(e) for e in embeddings])
        for ql in query_latents:
            self._ensure_adaptive_pc(ql)

        if f._cache_dirty or f._cached_positions is None:
            f._build_node_cache()

        n_nodes = len(f.node_index)
        if n_nodes == 0:
            return [[] for _ in range(n_queries)]

        query_phases: np.ndarray = np.full(n_queries, phase, dtype=np.float32)
        all_scores = self._batch_resonance(query_latents, query_phases, f.node_index)

        results_per_query: List[List[Tuple[str, float, MemoryNode]]] = []
        for qi in range(n_queries):
            scores = all_scores[qi]
            if session_id and session_id != "default":
                session_boosts = np.array(
                    [1.3 if f.nodes[nid].content.get("session") == session_id else 1.0 for nid in f.node_index],
                    dtype=np.float32,
                )
                scores = scores * session_boosts

            above = scores >= f.cfg.min_response
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
                nid = f.node_index[idx]
                node = f.nodes[nid]
                node.last_resonated = time.time()
                query_results.append((nid, float(score), node))
            results_per_query.append(query_results)

        f.stats["total_queries"] += n_queries
        for results in results_per_query:
            if results:
                f.stats["avg_response"] = 0.9 * f.stats["avg_response"] + 0.1 * results[0][1]

        if f.cfg.sparse_routing:
            elapsed_ms = (time.time() - t0) * 1000
            f.stats["avg_shard_query_time_ms"] = 0.95 * f.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms

        return results_per_query

    def batch_query(
        self,
        embeddings: List[NDArray],
        phases: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        modality: str = "text",
        session_id: Optional[str] = None,
    ) -> List[List[Tuple[str, float, MemoryNode]]]:
        f = self.field
        top_k = top_k or f.cfg.top_k
        n = len(embeddings)
        if phases is None:
            phases = [0.0] * n

        query_latents = np.array([f._project(e) for e in embeddings], dtype=np.float32)
        phases_arr = np.array(phases, dtype=np.float32)

        n_pos = f._index_mgr.hnsw_count()
        if f.cfg.use_hnsw and n_pos > getattr(f.cfg, "hnsw_min_nodes", 50):
            hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
            per_query_candidates: List[List[str]] = []
            all_candidate_ids: List[str] = []
            candidate_set: Set[str] = set()
            for ql in query_latents:
                cands = f._index_mgr.hnsw_search(ql, hnsw_k)
                cands = [nid for nid in cands if nid in f.nodes]
                per_query_candidates.append(cands)
                for nid in cands:
                    if nid not in candidate_set:
                        candidate_set.add(nid)
                        all_candidate_ids.append(nid)
            if all_candidate_ids:
                all_scores = self._batch_resonance(query_latents, phases_arr, all_candidate_ids)
                cand_index = {nid: idx for idx, nid in enumerate(all_candidate_ids)}
                results: List[List[Tuple[str, float, MemoryNode]]] = []
                for i, cands in enumerate(per_query_candidates):
                    row = []
                    for nid in cands:
                        j = cand_index[nid]
                        score = float(all_scores[i, j])
                        node = f.nodes[nid]
                        if session_id and node.content.get("session") == session_id:
                            score *= 1.3
                        if score >= f.cfg.min_response:
                            row.append((nid, score, node))
                            node.last_resonated = time.time()
                    row.sort(key=lambda x: x[1], reverse=True)
                    results.append(row[:top_k])
                return results
            return [[] for _ in range(n)]

        return [
            self.query(e, p, top_k=top_k, modality=modality, session_id=session_id) for e, p in zip(embeddings, phases)
        ]

    def query_by_text(
        self, text: str, top_k: Optional[int] = None, session_id: Optional[str] = None
    ) -> List[Tuple[str, float, Any]]:
        f = self.field
        top_k = top_k or f.cfg.top_k
        query_latent = f._projection_mgr.sot_query_latent(text)
        if query_latent is not None:
            return self.query(query_latent, top_k=top_k, session_id=session_id)
        return []
