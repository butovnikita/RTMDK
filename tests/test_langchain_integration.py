"""Tests for LangChain integration (Track 9)."""

from rtmdk.production.langchain_adapter import RTMDKRetriever, RTMDKChatMessageHistory, RTMDKDocument
import pytest
import numpy as np
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


# Skip entire module if langchain-core is not installed
pytest.importorskip("langchain_core", reason="langchain-core not installed")


@pytest.fixture
def memory():
    cfg = RTMDKConfig(
        latent_dim=64,
        use_hnsw=False,
        hyperbolic=False,
        quantization="none",
        enable_engrams=False,
    )
    return RTMDKMemory(config=cfg, embedder=_make_embedder(64), wal_path=None)


class TestRTMDKRetriever:
    def test_get_relevant_documents(self, memory):
        memory.save_context({"input": "coffee is delicious", "session_id": "s1"}, {"output": ""})
        memory.save_context({"input": "tea is warm", "session_id": "s1"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory, top_k=2)
        docs = retriever.get_relevant_documents("coffee")
        assert len(docs) > 0
        assert isinstance(docs[0], RTMDKDocument)
        assert "coffee" in docs[0].page_content.lower()

    def test_score_threshold(self, memory):
        memory.save_context({"input": "very specific topic xyz", "session_id": "s1"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory, top_k=5, score_threshold=0.99)
        docs = retriever.get_relevant_documents("completely unrelated query")
        assert len(docs) == 0

    def test_top_k_limit(self, memory):
        for i in range(5):
            memory.save_context({"input": f"doc {i}", "session_id": "s1"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory, top_k=2)
        docs = retriever.get_relevant_documents("doc")
        assert len(docs) <= 2

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self, memory):
        memory.save_context({"input": "async test", "session_id": "s1"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory)
        docs = await retriever.aget_relevant_documents("async")
        assert len(docs) >= 1


class TestRTMDKRetrieverLCEL:
    """LCEL compatibility: invoke, ainvoke, batch, abatch."""

    def test_invoke(self, memory):
        memory.save_context({"input": "LCEL test query", "session_id": "lcel"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory, top_k=3)
        docs = retriever.invoke("LCEL test")
        assert isinstance(docs, list)
        assert len(docs) >= 1
        # Documents returned by BaseRetriever.invoke() are langchain_core
        # Document objects
        assert hasattr(docs[0], "page_content")

    @pytest.mark.asyncio
    async def test_ainvoke(self, memory):
        memory.save_context({"input": "async LCEL test", "session_id": "lcel"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory)
        docs = await retriever.ainvoke("async LCEL")
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_batch(self, memory):
        for i in range(3):
            memory.save_context({"input": f"batch doc {i}", "session_id": "lcel"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory, top_k=2)
        results = retriever.batch(["batch doc 0", "batch doc 1", "batch doc 2"])
        assert len(results) == 3
        for docs in results:
            assert isinstance(docs, list)
            assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_abatch(self, memory):
        memory.save_context({"input": "abatch test", "session_id": "lcel"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory)
        results = await retriever.abatch(["abatch test", "nothing"])
        assert len(results) == 2
        assert isinstance(results[0], list)

    def test_pipe_composition(self, memory):
        """Test that retriever can be composed with RunnableLambda (| operator)."""
        from langchain_core.runnables import RunnableLambda

        memory.save_context({"input": "pipe test", "session_id": "lcel"}, {"output": ""})
        retriever = RTMDKRetriever(memory=memory)

        def pick_first(docs):
            return docs[0].page_content if docs else ""

        chain = retriever | RunnableLambda(pick_first)
        result = chain.invoke("pipe test")
        assert isinstance(result, str)
        assert "pipe test" in result.lower()


class TestRTMDKChatMessageHistory:
    def test_add_and_retrieve_messages(self, memory):
        history = RTMDKChatMessageHistory(memory=memory, session_id="chat1")
        history.add_user_message("hello")
        history.add_ai_message("hi there")
        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].type == "human"
        assert msgs[0].content == "hello"
        assert msgs[1].type == "ai"
        assert msgs[1].content == "hi there"

    def test_clear(self, memory):
        history = RTMDKChatMessageHistory(memory=memory, session_id="chat1")
        history.add_user_message("hello")
        history.clear()
        assert len(history.messages) == 0

    def test_persistence_in_memory(self, memory):
        history = RTMDKChatMessageHistory(memory=memory, session_id="chat2")
        history.add_user_message("persist me")
        assert len(memory.field.nodes) == 1

    def test_add_messages_bulk(self, memory):
        from langchain_core.messages import HumanMessage, AIMessage

        history = RTMDKChatMessageHistory(memory=memory, session_id="bulk")
        history.add_messages(
            [
                HumanMessage(content="msg1"),
                AIMessage(content="msg2"),
                HumanMessage(content="msg3"),
            ]
        )
        assert len(history.messages) == 3
        assert history.messages[0].content == "msg1"
        assert history.messages[1].content == "msg2"
        assert history.messages[2].content == "msg3"
