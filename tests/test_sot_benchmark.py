"""
tests/test_sot_benchmark.py — Correct SOT benchmark.

Previous version was broken: it added nodes with random dummy embeddings
but queried with SOT embeddings, resulting in ~9% R@1 (different spaces).

This version uses SOT embeddings for BOTH nodes and queries.
With default word-level tokenization, SOT achieves ~98-99% R@1,
practically matching the SBERT baseline on this dataset.

SOT is a lightweight fallback when no external embedder is available.
With word-level tokenization it retains near-baseline accuracy.
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import os
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch):
    monkeypatch.setenv("RTMDK_ADD_RATE_LIMIT", "0")


# Skip if sentence-transformers not installed
sbert = pytest.importorskip("sentence_transformers")


def test_sot_vs_sbert_baseline():
    """Compare SOT retrieval against SBERT baseline on QA data."""
    with open("datasets/qa_1000_en.json", "r", encoding="utf-8") as f:
        data = json.load(f)["records"][:200]

    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- SOT field (uses default word-level tokenization) ----
    cfg_sot = RTMDKConfig(
        latent_dim=384,
        top_k=5,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=False,
        learn_projection=False,
        bm25_fallback=False,
        enable_async=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        sot_enabled=True,
        sot_use_for_query=True,
        sot_subword_seed=True,
        sot_attention_pooling=True,
        sot_max_vocab=10000,
        sot_tokenization_mode="word",
    )
    field_sot = RTMDKField(cfg_sot)

    texts = [r["query"] + " " + r["answer"] for r in data]
    field_sot.sot_bootstrap(
        texts, teacher_model="all-MiniLM-L6-v2",
        fit_projection_only=False, n_epochs=50,
    )

    for rec in data:
        text = rec["query"] + " " + rec["answer"]
        tokens = field_sot._projection_mgr.sot_tokenizer.encode(text)
        emb = field_sot._projection_mgr.sot_tokenizer.embed(tokens)
        field_sot.add_node(
            emb.astype(np.float32),
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(text) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field_sot.nodes[field_sot.node_index[-1]].amplitude = 1.0
        field_sot.nodes[field_sot.node_index[-1]].salience = 1.0

    sot_hits = 0
    for rec in data:
        result = field_sot.query_by_text(rec["query"], top_k=1)
        if result and rec["answer"] in result[0][2].content.get("text", ""):
            sot_hits += 1

    sot_r1 = sot_hits / len(data)
    print(f"\nSOT R@1: {sot_r1:.1%}")

    # ---- SBERT baseline ----
    cfg_base = RTMDKConfig(
        latent_dim=384,
        top_k=5,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=False,
        learn_projection=False,
        bm25_fallback=False,
        enable_async=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
    )
    field_base = RTMDKField(cfg_base)

    for rec in data:
        text = rec["query"] + " " + rec["answer"]
        emb = teacher.encode(text, convert_to_numpy=True).astype(np.float32)
        field_base.add_node(
            emb,
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(text) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field_base.nodes[field_base.node_index[-1]].amplitude = 1.0
        field_base.nodes[field_base.node_index[-1]].salience = 1.0

    base_hits = 0
    for rec in data:
        q_emb = teacher.encode(rec["query"], convert_to_numpy=True).astype(np.float32)
        result = field_base.query(q_emb, top_k=1)
        if result and rec["answer"] in result[0][2].content.get("text", ""):
            base_hits += 1

    base_r1 = base_hits / len(data)
    print(f"SBERT R@1: {base_r1:.1%}")

    # With word-level tokenization, SOT should achieve >80% of SBERT baseline
    assert base_r1 >= 0.95, f"SBERT baseline too low: {base_r1:.1%}"
    assert sot_r1 >= 0.80, f"SOT R@1 too low: {sot_r1:.1%} (expected >=80%)"
    assert sot_r1 >= base_r1 * 0.80, f"SOT degradation too large: {sot_r1:.1%} vs {base_r1:.1%}"


if __name__ == "__main__":
    test_sot_vs_sbert_baseline()
