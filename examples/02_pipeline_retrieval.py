"""Example 2: Pipeline retrieval with observability."""
import numpy as np
from rtmdk import RTMDKMemory, RTMDKConfig


def simple_embedder(text: str) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)


def main():
    cfg = RTMDKConfig.production()
    memory = RTMDKMemory(config=cfg, embedder=simple_embedder)

    memory.add_node("Machine learning is a subset of AI")
    memory.add_node("Deep learning uses neural networks")
    memory.add_node("Transformers revolutionized NLP")

    result = memory.retrieve_nodes_pipeline("What is deep learning?", top_k=3)
    print("Results:", result["results"])
    print("Route:", result.get("route"))
    print("Metrics:", result.get("metrics"))


if __name__ == "__main__":
    main()
