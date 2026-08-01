# Example: Pipeline Retrieval

Canonical script: [`examples/02_pipeline_retrieval.py`](https://github.com/butovnikita/RTMDK/blob/main/examples/02_pipeline_retrieval.py)

Shows the **pipeline API** (`retrieve_nodes_pipeline`) — same retrieval as the
legacy call, but with routing decisions and per-stage latency metrics.

```python
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig

def simple_embedder(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)

cfg = RTMDKConfig.production()
memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

memory.add_node("Machine learning is a subset of AI")
memory.add_node("Deep learning uses neural networks")
memory.add_node("Transformers revolutionized NLP")

result = memory.retrieve_nodes_pipeline("What is deep learning?", top_k=3)
print("Results:", result["results"])   # ranked nodes
print("Route:",   result.get("route"))  # fast | standard | deep
print("Metrics:", result.get("metrics")) # per-stage latency breakdown
```

## Run

```bash
python examples/02_pipeline_retrieval.py
```

## Next Steps

- [Pipeline Architecture](../architecture/pipeline.md) — stages, flags, HTTP endpoints
- [Basic Memory](basic-memory.md)
