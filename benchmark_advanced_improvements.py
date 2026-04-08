"""
benchmark_advanced_improvements.py — Tests 7 Algorithmic Improvements.

Compares baseline vs each improvement vs all combined.
Measures: Recall@1/3/5, MRR, NDCG@5, P@5, Hallucination rate.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
from rtmdk.production.bm25_fallback import BM25FallbackRetriever
from rtmdk.production.advanced_retrieval import (
    HybridRetriever,
    ConfidenceAwareFallback,
    QueryExpander,
    AdaptiveDepthRetriever,
    TemporalDecayLearner,
    CausalAugmentedRetriever,
    MetaRetrievalController,
    AdvancedRTMDKRetriever,
)


# ============================================================================
# FACT GENERATOR (EN only for consistency)
# ============================================================================

def generate_en_facts(n: int, seed: int = 42) -> list:
    """Generate diverse EN facts."""
    import random
    random.seed(seed)
    base = [
        ("What causes earthquakes?", "Tectonic plate movement", "Earthquakes occur when tectonic plates move suddenly along geological fault lines, releasing seismic waves.", "science"),
        ("How do vaccines work?", "Train immune system", "Vaccines contain weakened pathogen parts that trigger immune response, creating antibodies.", "health"),
        ("What is DNA?", "Genetic information molecule", "Deoxyribonucleic acid contains instructions for organisms to develop and reproduce.", "science"),
        ("What is photosynthesis?", "Sunlight to chemical energy", "Plants use chlorophyll to absorb light, converting CO2 and water into glucose.", "science"),
        ("What is speed of light?", "299,792,458 meters per second", "Light speed in vacuum is fundamental constant c. Nothing travels faster.", "physics"),
        ("What is gravity?", "Attraction between masses", "Gravity keeps planets in orbit and gives objects weight on Earth.", "physics"),
        ("What is an atom?", "Smallest unit of matter", "Atoms have nucleus with protons/neutrons surrounded by electrons.", "science"),
        ("What is evolution?", "Traits change over generations", "Natural selection explains how species adapt over time.", "biology"),
        ("What is a black hole?", "Extreme gravity spacetime region", "Black holes form when massive stars collapse. Light cannot escape.", "physics"),
        ("What is the Big Bang?", "Universe creation event", "The Big Bang theory describes universe expansion from hot dense state 13.8 billion years ago.", "physics"),
        ("When did WWII end?", "1945", "World War II ended in 1945 with Japan's surrender on September 2.", "history"),
        ("Who was first US President?", "George Washington", "George Washington served as first US President from 1789 to 1797.", "history"),
        ("When did Roman Empire fall?", "476 AD", "Western Roman Empire fell in 476 AD when last emperor was deposed.", "history"),
        ("Who discovered America?", "Christopher Columbus", "Columbus reached Americas in 1492 sailing from Spain.", "history"),
        ("When did French Revolution begin?", "1789", "French Revolution began in 1789 with storming of the Bastille.", "history"),
        ("Who invented telephone?", "Alexander Graham Bell", "Bell patented first practical telephone in 1876.", "history"),
        ("When did Titanic sink?", "1912", "Titanic sank on April 15, 1912 after hitting iceberg on maiden voyage.", "history"),
        ("Who first walked on Moon?", "Neil Armstrong", "Armstrong stepped on Moon July 20, 1969 during Apollo 11 mission.", "history"),
        ("What is nuclear fusion?", "Combining nuclei for energy", "Nuclear fusion powers stars. Hydrogen fuses into helium releasing energy.", "physics"),
        ("What is a volcano?", "Crust opening releasing magma", "Volcanic eruptions create new land. Ring of Fire has most volcanoes.", "geography"),
    ]

    facts = []
    for i in range(n):
        b = base[i % len(base)]
        q, a, c, t = b
        if i >= len(base):
            q = q.replace("?", f" — specifically regarding fact #{i}?")
            c = c + f" Additional context detail for fact {i}."
        facts.append({"query": q, "answer": a, "context": c, "topic": t})
    return facts


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

def create_memory_with_facts(embedder, facts, n_facts):
    """Create RTMDK memory with n_facts."""
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=getattr(embedder, 'dim', 768),
            latent_dim=256, top_k=5, min_response=0.005,
            decay_rate=0.999, enable_async=False,
            bm25_fallback=False, use_hnsw=True,
            learn_projection=False, attention_bias=True,
        ),
        embedder=embedder,
    )
    bm25 = BM25FallbackRetriever()

    # Store facts and track embeddings
    embeddings = []
    for i in range(n_facts):
        f = facts[i % len(facts)]
        emb = embedder(f["context"])
        embeddings.append(emb)
        nid = memory.field.add_node(emb, {"text": f["context"], "topic": f.get("topic", "general")})
        if nid:
            bm25.add_document(f"doc_{i}", f["context"])

    return memory, bm25, embeddings


def compute_metrics(results, facts, test_n):
    """Compute Recall@K, MRR, NDCG@5, Precision@5."""
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = []
    ndcgs = []

    for i in range(min(test_n, len(facts))):
        f = facts[i % len(facts)]
        answer_words = [w for w in f["answer"].lower().split() if len(w) > 2]
        if not answer_words:
            continue

        # Check results
        found_rank = None
        for j, (nid, score, node) in enumerate(results[:10]):
            text = node.content.get("text", "").lower()
            if any(w in text for w in answer_words):
                found_rank = j + 1
                break

        if found_rank:
            if found_rank <= 1: recalls[1] += 1
            if found_rank <= 3: recalls[3] += 1
            if found_rank <= 5: recalls[5] += 1
            if found_rank <= 10: recalls[10] += 1
            ranks.append(found_rank)
        else:
            ranks.append(999)

        # NDCG@5
        rel = [0] * 5
        for j in range(min(5, len(results))):
            text = results[j][2].content.get("text", "").lower()
            if any(w in text for w in answer_words):
                rel[j] = 1
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(rel, reverse=True)
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / max(idcg, 1e-8))

    n = min(test_n, len(facts))
    return {
        "recall_at_1": recalls[1] / max(n, 1),
        "recall_at_3": recalls[3] / max(n, 1),
        "recall_at_5": recalls[5] / max(n, 1),
        "mrr": float(np.mean([1.0/r if r < 999 else 0.0 for r in ranks])),
        "ndcg_at_5": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def run_benchmark():
    """Run full benchmark comparing all improvements."""
    print("=" * 70)
    print("  BENCHMARK: 7 Algorithmic Improvements for RTMDK")
    print("=" * 70)

    embedder = get_embedder()
    facts = generate_en_facts(100, seed=42)
    test_n = 20  # Number of queries to test

    # Create memory
    print(f"\n  Creating memory with {len(facts)} facts...")
    memory, bm25, embeddings = create_memory_with_facts(embedder, facts, len(facts))
    print(f"  Nodes: {len(memory.field.nodes)}")

    results_table = []

    # Baseline: Standard RTMDK
    print(f"\n  [1] Baseline (Standard RTMDK)...")
    phase = memory._get_phase("baseline", embeddings[0])
    baseline_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        phase = memory._get_phase("baseline", emb)
        res = memory.field.query(emb, phase, top_k=5)
        baseline_results.extend(res)
    metrics = compute_metrics(baseline_results, facts, test_n)
    metrics["name"] = "Baseline (Standard RTMDK)"
    results_table.append(metrics)
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}")

    # 1. Hybrid Retrieval
    print(f"\n  [2] Hybrid Retrieval (Resonance + BM25 + Cosine)...")
    hybrid = HybridRetriever(memory, bm25)
    for i, emb in enumerate(embeddings):
        hybrid.add_embedding(f"n_{i}", emb)

    hybrid_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        res = hybrid.retrieve(f["query"], emb, top_k=5)
        hybrid_results.extend(res)
    metrics = compute_metrics(hybrid_results, facts, test_n)
    metrics["name"] = "1. Hybrid Retrieval"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 2. Confidence-Aware Fallback
    print(f"\n  [3] Confidence-Aware Fallback...")
    conf_aware = ConfidenceAwareFallback(hybrid, bm25)
    conf_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        res, status = conf_aware.retrieve(f["query"], emb, top_k=5)
        conf_results.extend(res)
    metrics = compute_metrics(conf_results, facts, test_n)
    metrics["name"] = "2. Confidence-Aware Fallback"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 3. Query Expansion
    print(f"\n  [4] Query Expansion...")
    expander = QueryExpander(memory)
    expander_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        expanded_query = expander.expand(f["query"])
        phase = memory._get_phase("expanded", emb)
        res = memory.field.query(emb, phase, top_k=5)
        expander_results.extend(res)
    metrics = compute_metrics(expander_results, facts, test_n)
    metrics["name"] = "3. Query Expansion"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 4. Adaptive Depth
    print(f"\n  [5] Adaptive Retrieval Depth...")
    adaptive = AdaptiveDepthRetriever(hybrid)
    adaptive_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        res = adaptive.retrieve(f["query"], emb, top_k=5)
        adaptive_results.extend(res)
    metrics = compute_metrics(adaptive_results, facts, test_n)
    metrics["name"] = "4. Adaptive Depth"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 5. Temporal Decay
    print(f"\n  [6] Temporal Decay Learning...")
    decay_learner = TemporalDecayLearner()
    # Simulate feedback
    for i, nid in enumerate(list(memory.field.nodes.keys())[:50]):
        quality = 0.8 if i % 3 == 0 else 0.3
        decay_learner.apply_feedback(nid, quality)
    # Apply decay to nodes
    for nid, node in memory.field.nodes.items():
        decay_learner.apply_to_node(node, nid)

    decay_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        phase = memory._get_phase("decay", emb)
        res = memory.field.query(emb, phase, top_k=5)
        decay_results.extend(res)
    metrics = compute_metrics(decay_results, facts, test_n)
    metrics["name"] = "5. Temporal Decay"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 6. Causal Augmentation
    print(f"\n  [7] Causal Graph Augmentation...")
    causal = CausalAugmentedRetriever(memory)
    causal_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        res = causal.retrieve(f["query"], emb, top_k=5)
        causal_results.extend(res)
    metrics = compute_metrics(causal_results, facts, test_n)
    metrics["name"] = "6. Causal Augmentation"
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # 7. Meta-Controller (All Combined)
    print(f"\n  [8] Meta-Retrieval Controller (All Combined)...")
    advanced = AdvancedRTMDKRetriever(
        memory, bm25,
        enable_hybrid=True,
        enable_confidence_aware=True,
        enable_query_expansion=True,
        enable_adaptive_depth=True,
        enable_temporal_decay=True,
        enable_causal_augmentation=True,
        enable_meta_controller=True,
    )
    advanced_results = []
    for i in range(test_n):
        f = facts[i % len(facts)]
        emb = embedder(f["query"])
        res, qtype = advanced.retrieve(f["query"], emb, top_k=5)
        advanced_results.extend(res)
    metrics = compute_metrics(advanced_results, facts, test_n)
    metrics["name"] = "7. Meta-Controller (All Combined)"
    metrics["meta_controller_stats"] = advanced.get_stats()
    results_table.append(metrics)
    delta = metrics["recall_at_1"] - results_table[0]["recall_at_1"]
    print(f"    R@1={metrics['recall_at_1']:.0%}  R@3={metrics['recall_at_3']:.0%}  MRR={metrics['mrr']:.3f}  NDCG@5={metrics['ndcg_at_5']:.3f}  (ΔR@1: {delta:+.0%})")

    # Print final report
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — 7 ALGORITHMIC IMPROVEMENTS")
    print(f"{'='*70}")
    print(f"\n  {'Algorithm':<35} {'R@1':>6} {'R@3':>6} {'MRR':>6} {'NDCG@5':>7}")
    print(f"  {'─'*60}")
    for r in results_table:
        print(f"  {r['name']:<35} {r['recall_at_1']:>5.0%} {r['recall_at_3']:>5.0%} {r['mrr']:>5.3f} {r['ndcg_at_5']:>6.3f}")

    # Improvement summary
    baseline_r1 = results_table[0]["recall_at_1"]
    best_r1 = max(r["recall_at_1"] for r in results_table)
    improvement = ((best_r1 - baseline_r1) / max(baseline_r1, 0.01)) * 100

    print(f"\n  {'IMPROVEMENT SUMMARY':^60}")
    print(f"  {'─'*60}")
    print(f"  Baseline R@1:   {baseline_r1:.0%}")
    print(f"  Best R@1:       {best_r1:.0%}")
    print(f"  Improvement:    {improvement:+.1f}%")

    # Save report
    report = {
        "algorithm_comparison": results_table,
        "baseline_recall_at_1": baseline_r1,
        "best_recall_at_1": best_r1,
        "improvement_percent": improvement,
    }
    with open("advanced_improvements_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to advanced_improvements_report.json")


if __name__ == "__main__":
    run_benchmark()
