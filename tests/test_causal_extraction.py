"""Tests for causal edge extraction from LLM explanations."""

from rtmdk.engines.causal_extraction import extract_causal_edges, extract_causal_edges_from_content


class TestExtractCausalEdges:
    def test_because_pattern(self):
        text = "It rains because of low pressure."
        edges = extract_causal_edges(text)
        assert len(edges) == 1
        effect, cause, conf = edges[0]
        assert "rains" in effect
        assert "low pressure" in cause
        assert 0.5 <= conf <= 1.0

    def test_causes_pattern(self):
        text = "Smoking causes cancer."
        edges = extract_causal_edges(text)
        assert len(edges) == 1
        effect, cause, conf = edges[0]
        assert "cancer" in effect
        assert "smoking" in cause

    def test_leads_to_pattern(self):
        text = "Overeating leads to obesity."
        edges = extract_causal_edges(text)
        assert len(edges) == 1
        effect, cause, conf = edges[0]
        assert "obesity" in effect
        assert "overeating" in cause

    def test_multiple_sentences(self):
        text = "Rain happens because of clouds. Smoking causes cancer."
        edges = extract_causal_edges(text)
        assert len(edges) == 2

    def test_empty_text(self):
        assert extract_causal_edges("") == []
        assert extract_causal_edges("short") == []

    def test_from_content_dict(self):
        content = {"text": "Rain happens because of clouds.", "explanation": "Smoking causes cancer."}
        edges = extract_causal_edges_from_content(content)
        assert len(edges) == 2
