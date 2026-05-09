"""Memory Timeline and Narrative Export.

Visualizes memory as a chronological timeline and generates
human-readable stories from episodic memories.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import time


class MemoryTimeline:
    """Build a chronological timeline of memories for a session or user."""

    def __init__(self, field):
        self.field = field

    def get_timeline(
        self,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return chronological list of memory events."""
        events = []
        for nid, node in self.field.nodes.items():
            content = node.content
            ts = content.get("timestamp", 0)
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            if session_id and content.get("session") != session_id:
                continue
            if tier and content.get("tier") != tier:
                continue

            events.append({
                "node_id": nid,
                "timestamp": ts,
                "input_text": content.get("input_text", ""),
                "output_text": content.get("output_text", ""),
                "text": content.get("text", ""),
                "tier": content.get("tier", "semantic"),
                "session": content.get("session", ""),
                "emotion": content.get("emotion", ""),
                "tags": content.get("tags", []),
                "salience": getattr(node, "salience", 0.0),
            })

        events.sort(key=lambda x: x["timestamp"])
        return events

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Summarize a single session's memories."""
        events = self.get_timeline(session_id=session_id)
        if not events:
            return {"session_id": session_id, "events": [], "topics": []}

        # Extract topics from tags
        all_tags = []
        for ev in events:
            all_tags.extend(ev.get("tags", []))
        topics = list(set(all_tags))[:10]

        return {
            "session_id": session_id,
            "event_count": len(events),
            "start_time": events[0]["timestamp"],
            "end_time": events[-1]["timestamp"],
            "events": events,
            "topics": topics,
        }


class MemoryNarrator:
    """Generate narrative stories from memory timelines."""

    def __init__(self, timeline: MemoryTimeline, llm_client=None):
        self.timeline = timeline
        self.llm_client = llm_client

    def narrate_session(self, session_id: str, max_events: int = 10) -> str:
        """Generate a human-readable story from session memories."""
        events = self.timeline.get_timeline(session_id=session_id)
        if not events:
            return "No memories found for this session."

        selected = events[:max_events]
        lines = [f"Session narrative ({len(events)} total memories):\n"]

        for i, ev in enumerate(selected, 1):
            text = ev.get("input_text") or ev.get("text", "")
            if text:
                lines.append(f"{i}. {text}")

        # LLM-based narrative generation
        if self.llm_client is not None:
            try:
                prompt = (
                    "Summarize the following conversation history as a brief narrative.\n\n"
                    + "\n".join([ev.get("input_text") or ev.get("text", "") for ev in selected if ev.get("input_text") or ev.get("text")])
                    + "\n\nNarrative:"
                )
                narrative = self.llm_client.complete(prompt, max_tokens=300).strip()
                if narrative:
                    return narrative
            except Exception:
                pass

        return "\n".join(lines)

    def export_markdown(self, session_id: Optional[str] = None, path: Optional[str] = None) -> str:
        """Export timeline as Markdown document."""
        events = self.timeline.get_timeline(session_id=session_id)
        lines = ["# Memory Timeline\n"]
        if session_id:
            lines.append(f"**Session:** {session_id}\n")

        for ev in events:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["timestamp"]))
            text = ev.get("input_text") or ev.get("text", "")
            tier = ev.get("tier", "semantic")
            lines.append(f"## {ts} ({tier})")
            lines.append(f"{text}\n")

        md = "\n".join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
        return md
