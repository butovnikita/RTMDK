# Python SDK

The Python facade (`rtmdk/__init__.py`) exports everything needed for
in-process use: `RTMDKMemory`, `RTMDKConfig`, `RTMDKField`, `MemoryNode`,
enums, preset helpers, and `create_rtmdk`.

## Facade API

```python
from rtmdk import RTMDKMemory, RTMDKConfig, create_rtmdk, list_presets

list_presets()
# ['local', 'production', 'research', 'enterprise',
#  'agent', 'legal', 'medical', 'streaming', 'sillytavern']

# Shortcut: preset + embedder in one call
memory = create_rtmdk(preset="local", embedder=my_embedder)
```

## Core Operations

```python
# Ingest
nid = memory.add_node("The capital of France is Paris")
memory.add_nodes_batch(embeddings, texts)        # high-throughput

# Retrieve
results = memory.retrieve_nodes("capital of France?", top_k=5)
output  = memory.retrieve_nodes_pipeline("...", top_k=5)  # with metrics/explanations

# Maintenance
health = memory.health_check()   # status + node count + breakers + metrics
memory.save_state("./state")     # persists engram cache, feedback, metrics
memory.load_state("./state")
```

## Key Classes

| Export | Purpose |
|--------|---------|
| `RTMDKMemory` | Facade — ingestion, retrieval, feedback, snapshots |
| `RTMDKConfig` | Unified config (9 presets, `RTMDK_*` env overrides, `validate()`) |
| `RTMDKField` | Low-level coordinator (managers, dynamics) |
| `MemoryNode` | Atomic memory unit (latent_pos, phase, amplitude, salience) |
| `create_rtmdk(preset, embedder)` | One-call factory |

For detailed method signatures see the docstrings in
`rtmdk/memory/core.py` and `rtmdk/memory/config.py`.

## Full Documentation

- [API Reference — package structure, core classes, utilities, engines](../01_API_REFERENCE.md)
- [Configuration](../getting-started/configuration.md)
