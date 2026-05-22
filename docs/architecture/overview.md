# Architecture Overview

RTMDK is organized into layers:

```
┌──────────────────────────────────────────────────────┐
│  API Layer        │  Config System  │  Production   │
│  REST / GraphQL   │  8 presets      │  Cache, Auth  │
│  WebSocket / gRPC │  59 env vars    │  Monitoring   │
├──────────────────────────────────────────────────────┤
│                 RTMDKMemory (Facade)                 │
├──────────────────────────────────────────────────────┤
│                 RTMDKField (Coordinator)             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │NodeMgr  │ │QueryMgr │ │TopoMgr  │ │IndexMgr │  │
│  │CrystMgr │ │RoutMgr  │ │ProjMgr  │ │Sched   │  │
│  │MergeMgr │ │ConsolMgr│ │OperMgr  │ │CognMgr │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
├──────────────────────────────────────────────────────┤
│  ResonanceEngine │ HNSW │ BM25 │ TieredStorage      │
└──────────────────────────────────────────────────────┘
```

## Core Concepts

### MemoryNode
The atomic unit of memory:
- `latent_pos` — position in latent space
- `phase` — oscillation phase for temporal ordering
- `amplitude` — strength of the memory
- `salience` — importance weight

### Resonance
Score between query and node:
```
response = spatial_kernel(distance) * phase_alignment * amplitude * salience
```

### Pipeline (v8.3+)
1. **Embed** — query → embedding
2. **Route** — adaptive cascade routing
3. **Retrieve** — resonance / HNSW / BM25 hybrid
4. **Rerank** — sentence-level reranking
5. **Calibrate** — conformal prediction filtering
6. **Explain** — per-result explanations
