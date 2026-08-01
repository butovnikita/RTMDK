"""ContextManager — save_context & retrieve-and-format logic extracted from core.py."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Dict, List

from numpy.typing import NDArray

from rtmdk.memory.config import ContextFormat
from rtmdk.memory.utils import SecurityViolationError, detect_modality
from rtmdk.utils.modality import detect_tier
from rtmdk.utils.attention import format_cognitive_context
from rtmdk.utils.formatting import format_context

if TYPE_CHECKING:
    from rtmdk.memory.core import RTMDKMemory

logger = __import__("logging").getLogger(__name__)


class ContextManager:
    """Encapsulates the save/load context pipeline for RTMDKMemory."""

    def __init__(self, memory: "RTMDKMemory") -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------
    def retrieve_and_format(self, query: str, embedding: NDArray, session_id: str) -> str:
        """Core retrieval pipeline shared by load_memory_variables and with_embedding."""
        mem = self._memory
        field = mem.field
        # Invariant: field is created in model_post_init before any retrieval
        assert field is not None, "RTMDKMemory.field must be initialized before retrieval"
        cfg = mem.config

        # Query decomposition for multi-hop retrieval
        decomposer = getattr(mem, "_query_decomposer", None)
        if decomposer is not None:
            sub_queries = decomposer.decompose(query)
        else:
            sub_queries = [query]

        all_results = []
        for sub_q in sub_queries:
            sub_emb = mem.embedder(sub_q)
            phase = mem._get_phase(session_id, sub_emb)

            # Phase 18: Engram-based retrieval (if enabled)
            if mem.engram_manager is not None and mem.engram_manager.index.size > 0:
                cache = getattr(mem, "engram_cache", None)
                if cache is not None and len(cache) > 0:
                    node_embs = cache.get_all()
                else:
                    node_embs = {}
                    for nid, node in field.nodes.items():
                        emb = mem._get_node_embedding(nid, node)
                        if emb is not None:
                            node_embs[nid] = emb

                engram_results = mem.engram_manager.retrieve_engrams(sub_emb, node_embs, top_k=field.cfg.top_k)

                if engram_results:
                    results = mem.engram_manager.expand_engrams(engram_results, field, top_k=field.cfg.top_k)
                    field.stats["engram_retrievals"] += 1
                else:
                    results = field.query(sub_emb, phase, top_k=field.cfg.top_k, session_id=session_id)
            else:
                results = field.query(sub_emb, phase, top_k=field.cfg.top_k, session_id=session_id)
            all_results.extend(results)

        # Deduplicate and re-rank combined results
        seen = set()
        results = []
        for nid, score, node in sorted(all_results, key=lambda x: x[1], reverse=True):
            if nid not in seen:
                results.append((nid, score, node))
                seen.add(nid)

        # Session-scoped retrieval: filter results by session_id, with global fallback
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results if node.content.get("session") == session_id
            ]
            if len(session_results) < field.cfg.top_k:
                global_results = [
                    (nid, score, node) for nid, score, node in results if node.content.get("session") != session_id
                ]
                needed = field.cfg.top_k - len(session_results)
                session_results.extend(global_results[:needed])
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5  # 50% boost for session match
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[: field.cfg.top_k]
            field.stats["session_scoped_retrievals"] = field.stats.get("session_scoped_retrievals", 0) + 1

        # Phase 1: Hybrid retrieval — blend RTMDK resonance with BM25 text scores
        if field.cfg.hybrid_alpha < 1.0 and field.bm25_index is not None and results:
            bm25_results = field.bm25_index.search(query, field.cfg.top_k * 2)
            if bm25_results:
                bm25_scores = {nid: score for nid, score in bm25_results}
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                if max_bm25 > 0:
                    bm25_scores = {nid: s / max_bm25 for nid, s in bm25_scores.items()}

                alpha = field.cfg.hybrid_alpha
                blended = []
                for nid, score, node in results:
                    bm25_score = bm25_scores.get(nid, 0.0)
                    blended_score = alpha * score + (1 - alpha) * bm25_score
                    blended.append((nid, blended_score, node))

                for nid, bm25_score in bm25_scores.items():
                    if nid not in [n[0] for n in blended] and bm25_score > field.cfg.min_response:
                        node = field.nodes.get(nid)
                        if node:
                            blended_score = alpha * 0.0 + (1 - alpha) * bm25_score
                            blended.append((nid, blended_score, node))

                blended.sort(key=lambda x: x[1], reverse=True)
                results = blended[: field.cfg.top_k]
                field.stats["hybrid_retrievals"] = field.stats.get("hybrid_retrievals", 0) + 1

        # Phase 15 Track 2: Proactive Clarification
        if cfg.proactive_clarification and results:
            max_score = results[0][1] if results else 0.0
            threshold = field.cfg.min_response * cfg.clarification_threshold_ratio
            if 0 < max_score < threshold:
                clarification = self._generate_clarification(results, query)
                field.stats["clarifications_generated"] += 1
                return clarification

        # Context formatting
        if cfg.attention_tokens and results:
            context = format_context(results, ContextFormat.ATTENTION)
        elif cfg.attention_bias and results:
            context = format_cognitive_context(results, bias_applied=True)
            field.stats["attention_bias_applied"] += 1
        elif cfg.cognitive_compression and results:
            context = field._cognitive_compress(results)
            raw_context = format_context(results, cfg.context_format)
            tokens_saved = max(0, len(raw_context) - len(context))
            field.stats["context_tokens_saved"] += tokens_saved
            field.stats["cognitive_compressions"] += 1
        else:
            context = format_context(results, cfg.context_format)

        # Phase 16 Track 1: SymbolicOverlay
        if cfg.symbolic_overlay and getattr(field, "symbolic_overlay", None) and results:
            facts = []
            for nid, score, node in results[:3]:
                text = node.content.get("text", "")
                concepts = field.symbolic_overlay._extract_concepts(text)
                facts.extend(concepts)
            if facts:
                symbolic_ctx = field.symbolic_overlay.get_symbolic_context(facts, max_depth=2)
                if symbolic_ctx:
                    context += "\n\n" + symbolic_ctx
                    field.stats["n_symbolic_inferences"] = field.stats.get("n_symbolic_inferences", 0) + 1
                    n_conflicts = len(
                        [
                            r
                            for r in field.symbolic_overlay.rules.values()
                            if getattr(r, "is_contextual_exception", False)
                        ]
                    )
                    field.stats["n_symbolic_conflicts"] = n_conflicts

        return context

    # ------------------------------------------------------------------
    # Save pipeline
    # ------------------------------------------------------------------
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save a conversation turn to memory with structured node format."""
        mem = self._memory
        field = mem.field
        # Invariant: field is created in model_post_init before any save
        assert field is not None, "RTMDKMemory.field must be initialized before save_context"
        cfg = mem.config

        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")

        if not output_text.strip():
            if not input_text.strip():
                return
            text_for_embedding = input_text
        else:
            text_for_embedding = output_text if len(output_text) > len(input_text) else input_text

        session_id = inputs.get("session_id", "default")
        timestamp = time.time()

        # Detect emotion from text
        emotion = "neutral"
        if input_text:
            lower_input = input_text.lower()
            if any(
                w in lower_input
                for w in ["happy", "love", "great", "wonderful", "amazing", "рад", "люб", "отличн", "прекрасн"]
            ):
                emotion = "positive"
            elif any(
                w in lower_input for w in ["sad", "hate", "bad", "terrible", "angry", "грустн", "ненавиж", "плох", "зл"]
            ):
                emotion = "negative"
            elif any(
                w in lower_input for w in ["?", "what", "why", "how", "when", "где", "что", "как", "когда", "почему"]
            ):
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
            "tier": "episodic",
            "context": {k: v for k, v in inputs.items() if k not in ["input", "query", "session_id", "embedding"]},
            "version": "2.0",
        }

        embedding = mem.embedder(text_for_embedding)
        phase = mem._get_phase(session_id, embedding)
        modality = detect_modality(text_for_embedding) if cfg.cross_modal else "text"

        # Detect memory tier
        tier = detect_tier(text_for_embedding, inputs)
        content["tier"] = tier

        try:
            nid = field.add_node(embedding, content, phase, session_id=session_id, modality=modality)
        except SecurityViolationError:
            return

        # Set tier on the newly added node
        if nid in field.nodes:
            field.nodes[nid].tier = tier

        # Phase 18: Create/update engrams from co-activated nodes
        if mem.engram_manager is not None:
            try:
                retrieved = mem.retrieve_nodes(
                    text_for_embedding, embedding, top_k=cfg.engram_max_nodes * 2, session_id=session_id
                )
                related_nodes = []
                for rnid, rscore, _ in retrieved:
                    if rscore >= cfg.min_response:
                        related_nodes.append((rnid, float(rscore)))
            except Exception:
                related_nodes = []
            related_nodes.append((nid, 1.0))

            if len(related_nodes) >= cfg.engram_min_nodes:
                node_embs = {}
                for rnid, _ in related_nodes:
                    emb = mem._get_node_embedding(rnid, field.nodes.get(rnid))
                    if emb is not None:
                        node_embs[rnid] = emb

                mem.engram_manager.create_engram_from_nodes(
                    activated_nodes=related_nodes[: cfg.engram_max_nodes],
                    node_embeddings=node_embs,
                    semantic_core=text_for_embedding[:100],
                    context_tags=set(tags + [tier, session_id]),
                    tier=tier,
                )

        if cfg.enable_async:
            if cfg.async_pipeline and not field._workers_started:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(field._start_workers())
                    field._workers_started = True
                    loop.create_task(field.evolve_q.put({"inputs": None}))
                except RuntimeError:
                    field.step()
            else:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._evolve_field_async())
                except RuntimeError:
                    field.step()
        else:
            field.step()

    async def _evolve_field_async(self) -> None:
        await asyncio.sleep(0.01)
        field = self._memory.field
        if field is not None:
            field.step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_tags(self, text: str) -> List[str]:
        """Auto-detect memory tags from text content."""
        tags = []
        lower = text.lower()

        if any(w in lower for w in ["hello", "hi ", "hey", "привет", "здравствуй", "hi,", "hey,"]):
            tags.append("greeting")
        if any(w in lower for w in ["my name is", "i'm ", "i am ", "меня зовут", "мое имя"]):
            tags.append("name")

        if any(w in lower for w in ["code", "program", "python", "java", "javascript", "функци", "код", "програм"]):
            tags.append("coding")
        if any(w in lower for w in ["coffee", "tea", "food", "drink", "кофе", "чай", "еда"]):
            tags.append("food_drink")
        if any(w in lower for w in ["love", "like", "prefer", "enjoy", "люб", "нрав", "предпочита"]):
            tags.append("preference")
        if any(w in lower for w in ["work", "job", "career", "работ", "карьер", "професс"]):
            tags.append("work")
        if any(w in lower for w in ["live", "city", "country", "home", "жив", "город", "стран", "дом"]):
            tags.append("location")
        if any(w in lower for w in ["family", "friend", "dog", "cat", "pet", "семь", "друг", "собак", "кот", "питом"]):
            tags.append("relationships")

        return tags[:5]

    def _generate_clarification(self, results: List, query: str) -> str:
        """Generate a clarification prompt from weak-resonance nodes."""
        lines = [f'[CLARIFICATION] Не нашёл точных воспоминаний по запросу: "{query[:80]}"']
        lines.append("Полусовпадения (низкий резонанс):")
        for nid, score, node in results[:3]:
            text = node.content.get("text", "")[:60]
            lines.append(f"  [R:{score:.2f}] {text}")
        lines.append("Уточните запрос или предоставьте дополнительный контекст.")
        return "\n".join(lines)
