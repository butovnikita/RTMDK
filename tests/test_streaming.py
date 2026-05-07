"""Tests for rtmdk.production.streaming."""

from rtmdk.production.streaming import StreamingResponse


class MockMemory:
    def load_memory_variables(self, data):
        return {"rtmdk_context": "mock context"}


def mock_llm_callback(query, context):
    return "This is a streamed response for testing"


class TestStreamingResponse:
    def test_stream_response(self):
        streamer = StreamingResponse(MockMemory(), mock_llm_callback)
        chunks = list(streamer.stream_response("hello", chunk_size=2))
        # Should yield chunks plus final stop
        assert len(chunks) >= 2
        assert chunks[-1]["finish_reason"] == "stop"
        assert all(c["finish_reason"] is None for c in chunks[:-1])

    def test_stream_empty_response(self):
        streamer = StreamingResponse(MockMemory(), lambda q, c: "")
        chunks = list(streamer.stream_response("hello"))
        assert chunks[-1]["finish_reason"] == "stop"
