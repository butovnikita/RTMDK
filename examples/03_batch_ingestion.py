"""Example 3: Batch ingestion for high throughput."""
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig


def simple_embedder(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)


def main():
    cfg = RTMDKConfig.production()
    memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

    texts = [f"Document number {i} with some content" for i in range(1000)]
    embeddings = np.array([simple_embedder(t) for t in texts])

    memory.add_nodes_batch(embeddings, texts)
    print(f"Added {len(texts)} nodes. Total: {len(memory.field.nodes)}")


if __name__ == "__main__":
    main()
