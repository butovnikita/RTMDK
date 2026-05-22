"""Example 1: Basic memory — add and retrieve nodes."""
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig


def simple_embedder(text: str) -> np.ndarray:
    """Simple hash-based embedder for demo purposes."""
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)


def main():
    cfg = RTMDKConfig.local()
    memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

    # Add some facts
    memory.add_node("The capital of France is Paris")
    memory.add_node("The capital of Germany is Berlin")
    memory.add_node("The capital of Italy is Rome")

    # Query
    results = memory.retrieve_nodes("What is the capital of France?", top_k=3)
    for r in results:
        print(f"Score: {r['score']:.3f} | Text: {r['text']}")


if __name__ == "__main__":
    main()
