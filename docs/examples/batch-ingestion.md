# Example: Batch Ingestion

Canonical script: [`examples/03_batch_ingestion.py`](https://github.com/butovnikita/RTMDK/blob/main/examples/03_batch_ingestion.py)

Demonstrates high-throughput ingestion with `add_nodes_batch()` — pass
pre-computed embeddings and texts in one call instead of adding nodes one by
one.

```python
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig

def simple_embedder(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)

cfg = RTMDKConfig.production()
memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

texts = [f"Document number {i} with some content" for i in range(1000)]
embeddings = np.array([simple_embedder(t) for t in texts])

memory.add_nodes_batch(embeddings, texts)
print(f"Added {len(texts)} nodes. Total: {len(memory.field.nodes)}")
```

## Run

```bash
python examples/03_batch_ingestion.py
```

For server-side bulk loading, see the `POST /v1/memory/batch_ingest` endpoint
in the [REST API](../api/rest.md), and the benchmark
`scripts/bench_batch_ingestion.py`.
