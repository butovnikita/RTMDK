"""
rtmdk/langchain_adapter.py — D3: LangChain Adapter

Provides RTMDKMemory class implementing LangChain's BaseChatMessageHistory interface
and as_langchain() helper for use with LangChain chains.
"""
from __future__ import annotations

from typing import List, Callable, Optional, Dict, Any

try:
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
    from langchain_core.chat_history import BaseChatMessageHistory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object  # type: ignore
    BaseMessage = object  # type: ignore
    AIMessage = None  # type: ignore
    HumanMessage = None  # type: ignore

import numpy as np
from numpy.typing import NDArray


class RTMDKMemory(BaseChatMessageHistory):
    """D3: LangChain-compatible wrapper for RTMDK memory.

    Implements BaseChatMessageHistory interface:
    - messages property: returns list of LangChain messages
    - add_message(message): adds a message to RTMDK memory
    - clear(): clears the conversation history

    Usage:
        from rtmdk.langchain_adapter import RTMDKMemory
        from rtmdk.memory.core import RTMDKConfig, RTMDKMemory as CoreMemory

        config = RTMDKConfig(...)
        core = CoreMemory(config=config, embedder=my_embedder)
        lc_memory = RTMDKMemory(core_memory=core, session_id="chat_1")

        # Use with LangChain
        from langchain.chains import ConversationChain
        chain = ConversationChain(llm=llm, memory=lc_memory)
    """

    def __init__(
        self,
        core_memory: Any,
        session_id: str = "default",
        embedder: Optional[Callable[[str], NDArray[np.float32]]] = None,
    ):
        """
        Args:
            core_memory: RTMDKMemory instance (from rtmdk.memory.core)
            session_id: Session identifier for multi-session support
            embedder: Optional embedder function (uses core_memory.embedder if not provided)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required. Install it: pip install langchain-core"
            )
        self._core = core_memory
        self._session_id = session_id
        self._embedder = embedder or getattr(core_memory, "embedder", None)
        # Track messages for LangChain interface
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Return chat history as list of LangChain messages."""
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to memory.

        Args:
            message: LangChain BaseMessage (HumanMessage, AIMessage, etc.)
        """
        self._messages.append(message)

        # Also store in RTMDK core memory
        content = message.content
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)

        if self._embedder:
            self._embedder(str(content))
        else:
            rng = np.random.default_rng(hash(str(content)) % 2**32)
            rng.standard_normal(768).astype(np.float32) * 0.1

        role = "user" if isinstance(message, HumanMessage) else "assistant"
        self._core.save_context(
            {"input": str(content), "session_id": self._session_id, "role": role},
            {"output": "" if role == "user" else str(content)}
        )

    def clear(self) -> None:
        """Clear the conversation history."""
        self._messages.clear()

    def add_user_message(self, content: str) -> None:  # type: ignore[override]
        """Convenience: add a user message."""
        if HumanMessage is not None:
            self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:  # type: ignore[override]
        """Convenience: add an AI message."""
        if AIMessage is not None:
            self.add_message(AIMessage(content=content))

    def get_context(self, query: str, top_k: int = 5) -> str:
        """Query RTMDK semantic memory for relevant context."""
        result = self._core.load_memory_variables({
            "input": query,
            "session_id": self._session_id,
        })
        return result.get("rtmdk_context", "")

    def get_stats(self) -> Dict[str, Any]:
        """Get RTMDK memory statistics."""
        return self._core.get_stats()


def as_langchain(core_memory: Any, session_id: str = "default") -> RTMDKMemory:
    """D3: Wrap an RTMDKMemory instance for use with LangChain chains.

    Args:
        core_memory: RTMDKMemory instance from rtmdk.memory.core
        session_id: Session identifier

    Returns:
        RTMDKMemory instance implementing BaseChatMessageHistory

    Example:
        from rtmdk.memory.core import RTMDKMemory, RTMDKConfig
        from rtmdk.langchain_adapter import as_langchain

        config = RTMDKConfig(causal_topological=True)
        memory = RTMDKMemory(config=config, embedder=embedder)

        lc_memory = as_langchain(memory, session_id="research_1")

        # Use with LangChain
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain

        llm = ChatOpenAI()
        chain = ConversationChain(llm=llm, memory=lc_memory)
        response = chain.invoke("What did we discuss earlier?")
    """
    return RTMDKMemory(core_memory=core_memory, session_id=session_id)
