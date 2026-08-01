"""Tests for rtmdk/memory/timeline.py — MemoryTimeline and MemoryNarrator."""

from types import SimpleNamespace

import pytest

from rtmdk.memory.timeline import MemoryNarrator, MemoryTimeline


def make_node(text, ts, session="s1", tier="episodic", tags=None, salience=0.5, input_text=""):
    return SimpleNamespace(
        content={
            "text": text,
            "timestamp": ts,
            "session": session,
            "tier": tier,
            "tags": tags or [],
            "input_text": input_text,
        },
        salience=salience,
    )


@pytest.fixture
def field():
    return SimpleNamespace(
        nodes={
            "n1": make_node(
                "first coffee note", 100.0, tags=["coffee", "drink"], salience=0.9, input_text="I love coffee"
            ),
            "n2": make_node("second tea note", 50.0, tags=["tea"], salience=0.3),
            "n3": make_node("other session note", 75.0, session="s2", tier="semantic", salience=0.1),
        }
    )


@pytest.fixture
def timeline(field):
    return MemoryTimeline(field)


class TestGetTimeline:
    def test_all_events_sorted_by_time(self, timeline):
        events = timeline.get_timeline()
        assert [e["node_id"] for e in events] == ["n2", "n3", "n1"]
        assert events[0]["timestamp"] == 50.0
        # Fields are projected correctly
        assert events[2]["input_text"] == "I love coffee"
        assert events[2]["salience"] == 0.9
        assert events[2]["tags"] == ["coffee", "drink"]

    def test_filter_by_session(self, timeline):
        events = timeline.get_timeline(session_id="s1")
        assert {e["node_id"] for e in events} == {"n1", "n2"}

    def test_filter_by_time_range(self, timeline):
        events = timeline.get_timeline(start_time=60.0, end_time=110.0)
        assert [e["node_id"] for e in events] == ["n3", "n1"]

    def test_filter_by_tier(self, timeline):
        events = timeline.get_timeline(tier="semantic")
        assert [e["node_id"] for e in events] == ["n3"]

    def test_empty_field(self):
        timeline = MemoryTimeline(SimpleNamespace(nodes={}))
        assert timeline.get_timeline() == []

    def test_defaults_for_missing_content_keys(self):
        node = SimpleNamespace(content={}, salience=0.0)
        timeline = MemoryTimeline(SimpleNamespace(nodes={"x": node}))
        (event,) = timeline.get_timeline()
        assert event["timestamp"] == 0
        assert event["tier"] == "semantic"
        assert event["tags"] == []


class TestSessionSummary:
    def test_summary_with_events(self, timeline):
        summary = timeline.get_session_summary("s1")
        assert summary["session_id"] == "s1"
        assert summary["event_count"] == 2
        assert summary["start_time"] == 50.0
        assert summary["end_time"] == 100.0
        assert set(summary["topics"]) == {"coffee", "drink", "tea"}
        assert len(summary["events"]) == 2

    def test_summary_empty(self, timeline):
        summary = timeline.get_session_summary("no-such-session")
        assert summary == {"session_id": "no-such-session", "events": [], "topics": []}


class _FakeLLM:
    def __init__(self, response="  A nice narrative.  "):
        self.response = response
        self.prompts = []

    def complete(self, prompt, max_tokens=300):
        self.prompts.append(prompt)
        return self.response


class _FailingLLM:
    def complete(self, prompt, max_tokens=300):
        raise RuntimeError("LLM down")


class TestNarrator:
    def test_narrate_empty_session(self, timeline):
        narrator = MemoryNarrator(timeline)
        assert narrator.narrate_session("nope") == "No memories found for this session."

    def test_narrate_uses_input_text(self, timeline):
        narrator = MemoryNarrator(timeline)
        story = narrator.narrate_session("s1")
        assert "Session narrative (2 total memories)" in story
        assert "2. I love coffee" in story  # n1 is 2nd chronologically
        assert "1. second tea note" in story  # falls back to text when input_text empty

    def test_narrate_respects_max_events(self, timeline):
        narrator = MemoryNarrator(timeline)
        story = narrator.narrate_session("s1", max_events=1)
        assert "1. I love coffee" not in story  # n1 is 2nd chronologically; only n2 selected

    def test_llm_narrative_returned_when_available(self, timeline):
        llm = _FakeLLM()
        narrator = MemoryNarrator(timeline, llm_client=llm)
        assert narrator.narrate_session("s1") == "A nice narrative."
        assert "I love coffee" in llm.prompts[0]

    def test_llm_empty_response_falls_back(self, timeline):
        narrator = MemoryNarrator(timeline, llm_client=_FakeLLM(response="   "))
        story = narrator.narrate_session("s1")
        assert "Session narrative" in story

    def test_llm_exception_falls_back(self, timeline):
        narrator = MemoryNarrator(timeline, llm_client=_FailingLLM())
        story = narrator.narrate_session("s1")
        assert "Session narrative" in story

    def test_export_markdown(self, timeline, tmp_path):
        narrator = MemoryNarrator(timeline)
        out = tmp_path / "timeline.md"

        md = narrator.export_markdown(session_id="s1", path=str(out))

        assert md.startswith("# Memory Timeline")
        assert "**Session:** s1" in md
        assert "(episodic)" in md
        assert "I love coffee" in md
        # File was written with identical content
        assert out.read_text(encoding="utf-8") == md

    def test_export_markdown_no_path(self, timeline):
        narrator = MemoryNarrator(timeline)
        md = narrator.export_markdown()
        assert md.count("##") == 3  # all events, all sessions
