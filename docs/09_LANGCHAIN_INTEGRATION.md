# LangChain & LCEL Integration

RTMDK provides first-class integration with [LangChain](https://python.langchain.com/) and its **LCEL** (LangChain Expression Language) pipeline system. Two adapters are available:

| Adapter | Purpose | Base Class |
|---------|---------|------------|
| `RTMDKRetriever` | Semantic retrieval in chains | `BaseRetriever` → `RunnableSerializable` |
| `RTMDKChatMessageHistory` | Conversation memory | `BaseChatMessageHistory` |
| `RTMDKVectorStore` | LlamaIndex compatibility | Plain wrapper |

## Installation

```bash
# Install with LangChain support
pip install rtmdk[langchain]

# Or manually
pip install langchain-core>=0.1
```

## RTMDKRetriever — LCEL-Compatible Retriever

`RTMDKRetriever` inherits from `BaseRetriever`, which itself extends `RunnableSerializable[str, list[Document]]`. This means it supports every LCEL operation out of the box:

- `invoke(query)` — sync single query
- `ainvoke(query)` — async single query
- `batch(queries)` — sync batch
- `abatch(queries)` — async batch
- `stream(query)` / `astream(query)` — streaming (returns docs as they arrive)
- `|` (pipe) — composition with other Runnables

### Basic Usage

```python
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig
from rtmdk.production.langchain_adapter import RTMDKRetriever

# Setup memory
config = RTMDKConfig(latent_dim=128)
memory = RTMDKMemory(config=config, embedder=embedder)

# Populate
memory.save_context(
    {"input": "Coffee is a brewed drink from roasted beans.", "session_id": "demo"},
    {"output": ""}
)

# Create retriever
retriever = RTMDKRetriever(memory=memory, top_k=3, score_threshold=0.5)

# LCEL invoke
docs = retriever.invoke("What is coffee?")
for doc in docs:
    print(doc.page_content, doc.metadata)
```

### LCEL Pipeline Composition

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Build a RAG chain
prompt = ChatPromptTemplate.from_template("""
Context: {context}
Question: {question}
Answer:
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
)

result = chain.invoke("Tell me about coffee")
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory` | — | `RTMDKMemory` instance |
| `top_k` | `5` | Max documents to return |
| `score_threshold` | `0.0` | Minimum attention score filter |
| `session_id` | `"langchain"` | Session filter for retrieval |

## RTMDKChatMessageHistory — Conversation Memory

Implements `BaseChatMessageHistory` with persistent storage into RTMDK nodes. Supports bulk `add_messages()` for efficient round-trips.

```python
from rtmdk.production.langchain_adapter import RTMDKChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = RTMDKChatMessageHistory(memory=memory, session_id="user_42")

# Bulk add (preferred)
history.add_messages([
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I help?"),
])

# Convenience methods
history.add_user_message("What's the weather?")
history.add_ai_message("It's sunny.")

# Retrieve
for msg in history.messages:
    print(msg.type, msg.content)

# Clear in-memory cache (does NOT delete RTMDK nodes)
history.clear()
```

## Legacy Adapter (`rtmdk.langchain_adapter`)

The top-level `rtmdk/langchain_adapter.py` provides `RTMDKMemory` (D3-era adapter) and `as_langchain()` helper. It is kept for backwards compatibility.

```python
from rtmdk.langchain_adapter import as_langchain

lc_memory = as_langchain(core_memory=memory, session_id="chat_1")
```

For new projects, prefer `RTMDKChatMessageHistory` from `rtmdk.production.langchain_adapter`.

## LlamaIndex — RTMDKVectorStore

A minimal vector-store wrapper for LlamaIndex users:

```python
from rtmdk.production.langchain_adapter import RTMDKVectorStore

store = RTMDKVectorStore(memory)
store.add(["Document 1", "Document 2"])
results = store.query("relevant query", top_k=3)
```

## Graceful Degradation

All imports are guarded. If `langchain-core` is not installed, the module still loads but classes fall back to plain objects. Install `[langchain]` extra to enable full functionality.
