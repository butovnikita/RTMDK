# Example: Basic Memory

Canonical script: [`examples/01_basic_memory.py`](https://github.com/butovnikita/RTMDK/blob/main/examples/01_basic_memory.py)

The minimal end-to-end flow: create memory with the `local` preset, add facts,
retrieve by query. Uses a deterministic hash-based embedder so it runs with
zero external dependencies.

```python
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig

def simple_embedder(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)

cfg = RTMDKConfig.local()
memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

memory.add_node("The capital of France is Paris")
memory.add_node("The capital of Germany is Berlin")
memory.add_node("The capital of Italy is Rome")

results = memory.retrieve_nodes("What is the capital of France?", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | Text: {r['text']}")
```

## Run

```bash
python examples/01_basic_memory.py
```

## Next Steps

- [Pipeline Retrieval](pipeline-retrieval.md) — observability and routing
- [Batch Ingestion](batch-ingestion.md) — high-throughput ingestion
- [Quick Start](../getting-started/quickstart.md)
