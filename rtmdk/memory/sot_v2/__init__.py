"""SOT v2.0 — Resonance-Topological Semantic Field.

Self-contained semantic embedding system for RTMDK with no external
model dependencies after initial training.

Modules:
    tokenizer: MI-based subword tokenization (information-theoretic BPE)
    embedder: Spectral self-supervised embeddings via graph Laplacian
    retriever: MaxSim late-interaction retrieval with resonance weights
"""

from .tokenizer import MI_SubwordTokenizer
from .embedder import SpectralEmbedder
from .retriever import ResonanceRetriever
from .sif_embedder import SIFEmbedder
from .hybrid_retriever import HybridSIFBM25Retriever
from .integration import SOTv2Embedder

__all__ = [
    "MI_SubwordTokenizer",
    "SpectralEmbedder",
    "ResonanceRetriever",
    "SIFEmbedder",
    "HybridSIFBM25Retriever",
    "SOTv2Embedder",
]
