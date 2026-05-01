"""
rtmdk/production/langchain_adapter.py — LangChain & LlamaIndex Integration.

Makes RTMDK usable as a retriever in LangChain and LlamaIndex ecosystems.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RTMDKDocument:
    """Document wrapper for LangChain compatibility."""
    page_content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class RTMDKRetriever:
    """LangChain-compatible retriever wrapper for RTMDK.
    
    Usage (LangChain):
        from rtmdk.production.langchain_adapter import RTMDKRetriever
        
        retriever = RTMDKRetriever(memory)
        docs = retriever.get_relevant_documents("What do I know about coffee?")
        
        # Use with LangChain chains:
        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """
    
    def __init__(
        self,
        memory,
        top_k: int = 5,
        score_threshold: float = 0.0,
        session_id: str = "langchain",
    ):
        self.memory = memory
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.session_id = session_id
    
    def get_relevant_documents(self, query: str) -> List[RTMDKDocument]:
        """LangChain interface: retrieve relevant documents."""
        ctx = self.memory.load_memory_variables({
            "input": query,
            "session_id": self.session_id,
        })
        
        context = ctx.get("rtmdk_context", "")
        if not context:
            return []
        
        # Parse context into documents
        docs = []
        # Try structured format: [ATTN:x.xx] text
        import re
        pattern = r'\[ATTN:([0-9.]+)\](?:\[SAL:([0-9.]+)\])?(?:\[TIER:(\w+)\])?\s*(.+?)(?=\[ATTN:|$)'
        matches = re.findall(pattern, context, re.DOTALL)
        
        if matches:
            for attn, sal, tier, text in matches:
                text = text.strip()
                if text:
                    docs.append(RTMDKDocument(
                        page_content=text,
                        metadata={"attention": float(attn), "salience": float(sal) if sal else 0, "tier": tier or ""},
                        score=float(attn),
                    ))
        else:
            # Fallback: whole context as one document
            docs.append(RTMDKDocument(
                page_content=context,
                metadata={"source": "rtmdk"},
                score=1.0,
            ))
        
        # Filter by score threshold
        docs = [d for d in docs if d.score >= self.score_threshold]
        
        return docs[:self.top_k]
    
    async def aget_relevant_documents(self, query: str) -> List[RTMDKDocument]:
        """Async version."""
        return self.get_relevant_documents(query)
    
    @property
    def _retriever_type(self) -> str:
        return "rtmdk"


class RTMDKChatMessageHistory:
    """LangChain-compatible chat message history backed by RTMDK.
    
    Usage:
        from rtmdk.production.langchain_adapter import RTMDKChatMessageHistory
        
        history = RTMDKChatMessageHistory(memory, session_id="user123")
        history.add_user_message("Hello")
        history.add_ai_message("Hi there!")
        messages = history.messages
    """
    
    def __init__(self, memory, session_id: str = "default"):
        self.memory = memory
        self.session_id = session_id
        self._messages = []
    
    def add_user_message(self, message: str):
        """Add a user message to history."""
        self.memory.save_context(
            {"input": message, "session_id": self.session_id, "role": "user"},
            {"output": message}
        )
        self._messages.append({"role": "user", "content": message})
    
    def add_ai_message(self, message: str):
        """Add an AI message to history."""
        self.memory.save_context(
            {"input": message, "session_id": self.session_id, "role": "ai"},
            {"output": message}
        )
        self._messages.append({"role": "ai", "content": message})
    
    def clear(self):
        """Clear message history."""
        self._messages.clear()
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self._messages.copy()


class RTMDKVectorStore:
    """LlamaIndex-compatible vector store wrapper.
    
    Usage (LlamaIndex):
        from rtmdk.production.langchain_adapter import RTMDKVectorStore
        
        store = RTMDKVectorStore(memory)
        index = VectorStoreIndex.from_vector_store(store)
    """
    
    def __init__(self, memory):
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
