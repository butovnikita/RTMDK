"""rtmdk/utils/formatting.py"""
from __future__ import annotations
import json
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rtmdk.nodes import MemoryNode

from rtmdk.config import ContextFormat

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


def format_context(results: List[Tuple[str, float, "MemoryNode"]], fmt: ContextFormat) -> str:
    if fmt == ContextFormat.JSON:
        items = []
        for nid, resp, node in results:
            item = {"resonance": round(resp, 4), "salience": round(node.salience, 4),
                    "text": node.content.get("text", ""), "lineage": node.lineage,
                    "modality": node.modality, "self_sup_score": round(node.self_sup_score, 4),
                    "cross_modal_score": round(node.cross_modal_score, 4)}
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
                          f"  lineage: {node.lineage}", f"  modality: {node.modality}",
                          f"  cross_modal_score: {node.cross_modal_score:.4f}"])
        return "\n".join(lines) if lines else "No relevant memory."
    elif fmt == ContextFormat.ATTENTION:
        lines = ["### ATTENTION_CONTEXT"]
        for nid, resp, node in results:
            causal = len(node.causal_strength) if hasattr(node, 'causal_strength') else 0
            goal_rel = getattr(node, 'goal_relevance', 0.0)
            tokens = (f"[ATTN:{resp:.3f}][SAL:{node.salience:.3f}]"
                      f"[TIER:{getattr(node, 'tier', 'semantic')[0].upper()}]")
            if causal > 0:
                tokens += f"[CAUSAL:{causal}]"
            if goal_rel > 0.3:
                tokens += f"[GOAL:{goal_rel:.2f}]"
            text = node.content.get("text", "unknown")[:100]
            lines.append(f"{tokens} {text}")
        return "\n".join(lines) if len(lines) > 1 else "No relevant memory."
    else:
        parts = [f"[R:{r:.2f}|S:{n.salience:.2f}|CM:{n.cross_modal_score:.2f}] {n.content.get('text', '')}" for _, r, n in results]
        return "\n".join(parts) if parts else "No relevant memory."


def build_system_prompt(context: str, fmt: ContextFormat, use_structured: bool) -> str:
    if not use_structured or not context or context in ("No relevant memory.", "[]"):
        return "You are a helpful assistant with long-term memory."
    return SYSTEM_PROMPT_TEMPLATES.get(fmt, SYSTEM_PROMPT_TEMPLATES[ContextFormat.PLAIN]).format(context=context)
