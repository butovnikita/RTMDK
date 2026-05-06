"""
rtmdk/production/langchain_adapter.py — LangChain & LlamaIndex Integration.

Makes RTMDK usable as a retriever in LangChain and LlamaIndex ecosystems.
Supports LCEL (LangChain Expression Language) via BaseRetriever / Runnable.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Graceful langchain_core import — integration works only when installed
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from pydantic import Field, ConfigDict
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseRetriever = object  # type: ignore[misc,assignment]
    CallbackManagerForRetrieverRun = Any  # type: ignore[misc,assignment]
    AsyncCallbackManagerForRetrieverRun = Any  # type: ignore[misc,assignment]
    Document = dict  # type: ignore[misc,assignment]
    BaseChatMessageHistory = object  # type: ignore[misc,assignment]
    BaseMessage = Any  # type: ignore[misc,assignment]
    HumanMessage = AIMessage = SystemMessage = None  # type: ignore[misc,assignment]
    Field = lambda *a, **kw: None  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[misc,assignment]
    LANGCHAIN_AVAILABLE = False


@dataclass
class RTMDKDocument:
    """Document wrapper for LangChain compatibility (legacy, kept for BC)."""
    page_content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class RTMDKRetriever(BaseRetriever if LANGCHAIN_AVAILABLE else object):
    """LangChain-compatible retriever wrapper for RTMDK.

    Implements BaseRetriever → RunnableSerializable[str, list[Document]],
    giving full LCEL compatibility: invoke(), ainvoke(), batch(), abatch(),
    stream(), astream(), pipe(), etc.

    Usage (LangChain LCEL):
        from rtmdk.production.langchain_adapter import RTMDKRetriever
        from langchain_core.runnables import RunnablePassthrough

        retriever = RTMDKRetriever(memory=memory)
        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
        result = chain.invoke("What do I know about coffee?")

    Usage (legacy chains):
        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    if LANGCHAIN_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: Any = Field(default=None)  # type: ignore[valid-type]
    top_k: int = Field(default=5)  # type: ignore[valid-type]
    score_threshold: float = Field(default=0.0)  # type: ignore[valid-type]
    session_id: str = Field(default="langchain")  # type: ignore[valid-type]

    # --- BaseRetriever abstract methods ---

    def _get_relevant_documents(
        self, query: str, *, run_manager: "CallbackManagerForRetrieverRun"
    ) -> List["Document"]:
        """Sync retrieval — called by invoke(), batch(), stream()."""
        ctx = self.memory.load_memory_variables({
            "input": query,
            "session_id": self.session_id,
        })
        return self._parse_context(ctx.get("rtmdk_context", ""))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: "AsyncCallbackManagerForRetrieverRun"
    ) -> List["Document"]:
        """Async retrieval — called by ainvoke(), abatch(), astream()."""
        return self._get_relevant_documents(query, run_manager=run_manager)  # type: ignore[arg-type]

    # --- Internal helpers ---

    def _parse_context(self, context: str) -> List["Document"]:
        """Parse rtmdk_context string into LangChain Document objects."""
        if not context:
            return []

        docs: List["Document"] = []
        # Try structured format: [ATTN:x.xx][SAL:y.yy][TIER:...] text
        pattern = (
            r"\[ATTN:([0-9.]+)\](?:\[SAL:([0-9.]+)\])?(?:\[TIER:(\w+)\])?\s*(.+?)(?=\[ATTN:|$)"
        )
        matches = re.findall(pattern, context, re.DOTALL)

        if matches and LANGCHAIN_AVAILABLE and Document is not dict:
            for attn, sal, tier, text in matches:
                text = text.strip()
                if text:
                    score = float(attn)
                    if score >= self.score_threshold:
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "attention": score,
                                "salience": float(sal) if sal else 0.0,
                                "tier": tier or "",
                                "source": "rtmdk",
                            },
                        ))
        elif context.strip():
            # Fallback: whole context as one document
            docs.append(Document(
                page_content=context.strip(),
                metadata={"source": "rtmdk"},
            ))

        return docs[:self.top_k]

    # --- Legacy compatibility ---

    def get_relevant_documents(self, query: str) -> List[RTMDKDocument]:
        """Legacy interface returning RTMDKDocument (kept for backwards compat)."""
        docs = self._get_relevant_documents(
            query, run_manager=None  # type: ignore[arg-type]
        )
        return [
            RTMDKDocument(
                page_content=d.page_content,
                metadata=d.metadata,
                score=d.metadata.get("attention", 1.0),
            )
            for d in docs
        ]

    async def aget_relevant_documents(self, query: str) -> List[RTMDKDocument]:
        """Async legacy interface."""
        docs = await self._aget_relevant_documents(
            query, run_manager=None  # type: ignore[arg-type]
        )
        return [
            RTMDKDocument(
                page_content=d.page_content,
                metadata=d.metadata,
                score=d.metadata.get("attention", 1.0),
            )
            for d in docs
        ]

    @property
    def _retriever_type(self) -> str:
        return "rtmdk"


class RTMDKChatMessageHistory(BaseChatMessageHistory if LANGCHAIN_AVAILABLE else object):
    """LangChain-compatible chat message history backed by RTMDK.

    Implements BaseChatMessageHistory with full message persistence into
    RTMDK memory field. Supports bulk add_messages for efficient round-trips.

    Usage (LCEL):
        from rtmdk.production.langchain_adapter import RTMDKChatMessageHistory

        history = RTMDKChatMessageHistory(memory=memory, session_id="user123")
        history.add_messages([
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ])
        msgs = history.messages
    """

    if LANGCHAIN_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: Any = Field(default=None)  # type: ignore[valid-type]
    session_id: str = Field(default="default")  # type: ignore[valid-type]

    def __init__(self, memory: Any = None, session_id: str = "default"):
        # BaseChatMessageHistory is ABC (not Pydantic), so we set attrs directly
        self.memory = memory
        self.session_id = session_id
        self._messages: List["BaseMessage"] = []

    @property
    def messages(self) -> List["BaseMessage"]:
        """Return messages as LangChain BaseMessage objects."""
        if not LANGCHAIN_AVAILABLE or BaseMessage is Any:
            return self._messages  # type: ignore[return-value]
        return list(self._messages)

    def add_messages(self, messages: List["BaseMessage"]) -> None:
        """Bulk add messages (preferred over add_message)."""
        for message in messages:
            self._add_single_message(message)
        self._messages.extend(messages)

    def _add_single_message(self, message: "BaseMessage") -> None:
        """Persist a single message into RTMDK memory."""
        role = "user"
        if LANGCHAIN_AVAILABLE and HumanMessage is not None:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "ai"
            elif isinstance(message, SystemMessage):
                role = "system"
        else:
            role = getattr(message, "type", "user")

        content = getattr(message, "content", str(message))
        self.memory.save_context(
            {"input": content, "session_id": self.session_id, "role": role},
            {"output": content}
        )

    def clear(self) -> None:
        """Clear in-memory message cache.

        Note: This does NOT delete persisted nodes from RTMDK.
        To fully erase session history, use memory.delete() on nodes.
        """
        self._messages.clear()

    # --- Legacy convenience methods ---

    def add_user_message(self, message: Any) -> None:
        """Add a user message."""
        if LANGCHAIN_AVAILABLE and HumanMessage is not None:
            msg = message if isinstance(message, HumanMessage) else HumanMessage(content=message)
        else:
            msg = message
        self.add_messages([msg])  # type: ignore[list-item]

    def add_ai_message(self, message: Any) -> None:
        """Add an AI message."""
        if LANGCHAIN_AVAILABLE and AIMessage is not None:
            msg = message if isinstance(message, AIMessage) else AIMessage(content=message)
        else:
            msg = message
        self.add_messages([msg])  # type: ignore[list-item]


class RTMDKVectorStore:
    """LlamaIndex-compatible vector store wrapper.

    Usage (LlamaIndex):
        from rtmdk.production.langchain_adapter import RTMDKVectorStore

        store = RTMDKVectorStore(memory)
        index = VectorStoreIndex.from_vector_store(store)
    """

    def __init__(self, memory: Any):
        self.memory = memory

    def add(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add texts to the vector store."""
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self.memory.save_context(
                {"input": text, "session_id": "vectorstore", **metadata},
                {"output": text}
            )

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the vector store."""
        ctx = self.memory.load_memory_variables({
            "input": query,
            "session_id": "vectorstore",
        })
        return [{"content": ctx.get("rtmdk_context", "")}]
