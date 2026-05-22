# RTMDK — Resonance-Topological Memory

> Long-term memory for LLMs based on resonance-topological organization

---

## What is RTMDK?

RTMDK (Resonance-Topological Memory) is an open-source memory system for Large Language Models that replaces traditional vector search (FAISS, Chroma) with a physically-inspired model:

- **Memory nodes** have phase, amplitude, salience
- **Resonance** between query and nodes: K_spatial × K_phase × A × S
- **Topology** organizes memories in a dynamic field

## Key Metrics

| Metric | Value |
|--------|-------|
| Recall@1 | **99.3%** (vs 18.1% cosine) |
| Latency p50 @ 1K nodes | **0.26 ms** |
| Latency p50 @ 100K nodes | **16 ms** |
| Memory @ 10K nodes | **19–30 MB** |
| Tests | 1118 passed |
| Pipeline stages | 6 (observable, configurable) |

## Quick Start

```bash
pip install rtmdk
```

```python
from rtmdk import RTMDKMemory, RTMDKConfig
import numpy as np

cfg = RTMDKConfig.local()
memory = RTMDKMemory(config=cfg, embedder=lambda t: np.random.randn(64))

memory.add_node("The sky is blue")
results = memory.retrieve_nodes("What color is the sky?", top_k=3)
```

## Features

- **Zero embedding cost** — SOT v2 tokenizer learns online, no API calls
- **Pipeline Architecture** — 6 explicit stages with circuit breakers
- **Tiered Storage** — Hot/Warm/Cold tiers for 1M+ nodes
- **int8 Quantization** — 4× RAM reduction for edge deployment
- **Production Ready** — API keys, rate limiting, multi-tenant, audit logs
- **Multiple APIs** — REST, GraphQL, WebSocket, gRPC, SSE

## License

AGPL-3.0
