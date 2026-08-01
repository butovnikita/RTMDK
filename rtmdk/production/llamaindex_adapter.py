"""
rtmdk/production/llamaindex_adapter.py — LlamaIndex Integration.

Provides RTMDK retriever and vector store adapters for the LlamaIndex ecosystem.

Usage:
    from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

    retriever = RTMDKLlamaIndexRetriever(memory=memory, top_k=5)
    nodes = retriever.retrieve("What do I know about coffee?")

Optional dependency:
    pip install rtmdk[llamaindex]
"""

import re
from typing import List, Dict, Any, Optional

# Graceful llama-index import — integration works only when installed
try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    BaseRetriever = object  # type: ignore[misc,assignment]
    TextNode = None  # type: ignore[misc,assignment]
    NodeWithScore = None  # type: ignore[misc,assignment]
    QueryBundle = None  # type: ignore[misc,assignment]
    LLAMAINDEX_AVAILABLE = False


class RTMDKLlamaIndexRetriever(BaseRetriever if LLAMAINDEX_AVAILABLE else object):  # type: ignore[misc]
    """LlamaIndex-compatible retriever backed by RTMDK memory.

    Usage:
        from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

        retriever = RTMDKLlamaIndexRetriever(memory=memory)
        nodes = retriever.retrieve("What do I know about coffee?")

        # With VectorStoreIndex
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_vector_store(
            RTMDKVectorStore(memory),
            embed_model=embed_model,
        )
    """

    def __init__(
        self,
        memory: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        session_id: str = "llamaindex",
    ):
        self.memory = memory
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.session_id = session_id

    def _retrieve(self, query_bundle: "QueryBundle") -> List["NodeWithScore"]:
        """Core retrieval method called by LlamaIndex."""
        if QueryBundle is not None and hasattr(query_bundle, "query_str"):
            query_str = query_bundle.query_str
        else:
            query_str = str(query_bundle)

        ctx = self.memory.load_memory_variables({"input": query_str, "session_id": self.session_id})
        context = ctx.get("rtmdk_context", "")
        return self._parse_context(context)

    def _parse_context(self, context: str) -> List["NodeWithScore"]:
        """Parse rtmdk_context into LlamaIndex NodeWithScore objects."""
        if not context:
            return []

        nodes: List["NodeWithScore"] = []
        # Structured format: [ATTN:x.xx][SAL:y.yy][TIER:...] text
        pattern = r"\[ATTN:([0-9.]+)\](?:\[SAL:([0-9.]+)\])?(?:\[TIER:(\w+)\])?\s*(.+?)(?=\[ATTN:|$)"
        matches = re.findall(pattern, context, re.DOTALL)

        if matches and LLAMAINDEX_AVAILABLE and TextNode is not None:
            for i, (attn, sal, tier, text) in enumerate(matches):
                text = text.strip()
                if not text:
                    continue
                score = float(attn)
                if score < self.score_threshold:
                    continue
                node = TextNode(
                    id_=f"rtmdk_{i}",
                    text=text,
                    metadata={
                        "attention": score,
                        "salience": float(sal) if sal else 0.0,
                        "tier": tier or "",
                        "source": "rtmdk",
                    },
                )
                nodes.append(NodeWithScore(node=node, score=score))
        elif context.strip() and LLAMAINDEX_AVAILABLE and TextNode is not None:
            node = TextNode(
                id_="rtmdk_0",
                text=context.strip(),
                metadata={"source": "rtmdk"},
            )
            nodes.append(NodeWithScore(node=node, score=1.0))

        return nodes[: self.top_k]

    def retrieve(self, query_str: str) -> List["NodeWithScore"]:
        """Convenience sync retrieval by raw string."""
        if QueryBundle is not None:
            bundle = QueryBundle(query_str=query_str)
        else:
            bundle = query_str  # type: ignore[assignment]
        return self._retrieve(bundle)


class RTMDKVectorStore:
    """Minimal LlamaIndex-compatible vector store wrapper around RTMDK.

    This is a *naive* adapter — it stores nodes in RTMDK and retrieves via
    RTMDK resonance search. It does NOT implement the full BasePydanticVectorStore
    interface; instead it provides the subset that VectorStoreIndex needs.
    """

    def __init__(self, memory: Any, session_id: str = "vectorstore"):
        self.memory = memory
        self.session_id = session_id
        self._nodes: List[Dict[str, Any]] = []

    def add(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the store."""
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self.memory.save_context(
                {"input": text, "session_id": self.session_id, **metadata},
                {"output": text},
            )
            self._nodes.append({"text": text, "metadata": metadata})

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the store and return raw dicts."""
        ctx = self.memory.load_memory_variables({"input": query, "session_id": self.session_id})
        return [{"content": ctx.get("rtmdk_context", "")}]

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Return all stored nodes as dicts."""
        return self._nodes
