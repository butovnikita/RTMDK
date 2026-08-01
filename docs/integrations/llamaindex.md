# LlamaIndex Integration

RTMDK provides retriever and vector-store adapters for the LlamaIndex
ecosystem in `rtmdk/production/llamaindex_adapter.py`. The import is graceful —
the adapter works only when `llama-index-core` is installed.

## Installation

```bash
pip install rtmdk[llamaindex]
```

## Usage

```python
from rtmdk import RTMDKMemory, RTMDKConfig
from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

memory = RTMDKMemory(config=RTMDKConfig.local(), embedder=my_embedder)

retriever = RTMDKLlamaIndexRetriever(
    memory=memory,
    top_k=5,
    score_threshold=0.0,
    session_id="llamaindex",
)
nodes = retriever.retrieve("What do I know about coffee?")
```

### As a VectorStore

```python
from llama_index.core import VectorStoreIndex
from rtmdk.production.llamaindex_adapter import RTMDKVectorStore

index = VectorStoreIndex.from_vector_store(
    RTMDKVectorStore(memory),
    embed_model=embed_model,
)
```

## Full Documentation

- Adapter source & docstrings: `rtmdk/production/llamaindex_adapter.py`
- [API Reference §9 — agents and production integrations](../01_API_REFERENCE.md)
- For LangChain, see [LangChain Integration](langchain.md)
