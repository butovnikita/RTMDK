"""
benchmark_7_improvements.py — Proper benchmark with unique facts.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
from rtmdk.production.bm25_fallback import BM25FallbackRetriever
from rtmdk.production.advanced_retrieval import (
    HybridRetriever, CausalAugmentedRetriever, QueryExpander,
    TemporalDecayLearner, AdvancedRTMDKRetriever,
)


# 20 UNIQUE facts with distinct topics
UNIQUE_FACTS = [
    ("What causes earthquakes?", "Tectonic plates", "Earthquakes occur when tectonic plates suddenly move along geological fault lines deep underground.", "science"),
    ("Who invented the telephone?", "Alexander Graham Bell", "Bell patented the first practical telephone in 1876 in Boston Massachusetts.", "history"),
    ("What is DNA made of?", "Nucleotides", "DNA is made of nucleotide building blocks containing adenine guanine cytosine and thymine bases.", "biology"),
    ("How does photosynthesis work?", "Light to chemical energy", "Plants use chlorophyll in their leaves to absorb sunlight and convert carbon dioxide plus water into glucose.", "biology"),
    ("What is the speed of light?", "299792458 meters per second", "Light travels at exactly 299792458 meters per second in vacuum which is about 186000 miles per second.", "physics"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519 and it hangs in the Louvre Museum in Paris.", "art"),
    ("What is the largest ocean?", "Pacific Ocean", "The Pacific Ocean covers more than 63 million square miles which is more than all land masses combined.", "geography"),
    ("When did World War 2 end?", "1945", "World War 2 ended in 1945 when Japan surrendered on September 2nd aboard the USS Missouri battleship.", "history"),
    ("What is gravity?", "Attraction between masses", "Gravity is the force that attracts any two objects with mass toward each other keeping planets in orbit.", "physics"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Shakespeare wrote Romeo and Juliet around 1595 making it one of his most famous tragedy plays.", "literature"),
    ("What is the capital of Australia?", "Canberra", "Canberra was purpose-built as Australia's capital city located between Sydney and Melbourne.", "geography"),
    ("How do vaccines work?", "Train immune system", "Vaccines expose your immune system to weakened pathogens so it creates antibodies to fight future infections.", "health"),
    ("What is a black hole?", "Extreme gravity region", "A black hole has gravity so strong that nothing including light can escape beyond its event horizon.", "physics"),
    ("Who discovered penicillin?", "Alexander Fleming", "Fleming accidentally discovered penicillin in 1928 when mold contaminated his bacterial culture plates.", "medicine"),
    ("What is the smallest country?", "Vatican City", "Vatican City measures only 0.44 square kilometers and has approximately 800 residents making it the smallest country.", "geography"),
    ("What causes seasons?", "Earth's axial tilt", "Earth's 23.5 degree axial tilt causes different hemispheres to receive varying sunlight throughout the year.", "science"),
    ("Who built the pyramids?", "Ancient Egyptians", "The Great Pyramid of Giza was built around 2560 BC as a tomb for Pharaoh Khufu of Egypt.", "history"),
    ("What is an atom?", "Smallest matter unit", "An atom consists of a nucleus containing protons and neutrons orbited by electrons in specific energy shells.", "chemistry"),
    ("How old is the Earth?", "4.5 billion years", "Scientists estimate Earth formed approximately 4.54 billion years ago from the solar nebula.", "science"),
    ("What is evolution?", "Species change over time", "Evolution by natural selection causes species to gradually adapt as beneficial traits become more common.", "biology"),
]


def test_simple_retrieval():
    """Test if basic retrieval works with unique facts."""
    print("=" * 60)
    print("  SIMPLE RETRIEVAL TEST")
    print("=" * 60)

    embedder = get_embedder()
    
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=256, top_k=3, min_response=0.001,
            decay_rate=0.999, enable_async=False, bm25_fallback=True,
            use_hnsw=False, learn_projection=False,
        ),
        embedder=embedder,
    )

    bm25 = BM25FallbackRetriever()

    # Store facts
    for i, (q, a, c, t) in enumerate(UNIQUE_FACTS):
        emb = embedder(c)
        memory.field.add_node(emb, {"text": c, "topic": t})
        bm25.add_document(f"doc_{i}", c)

    print(f"  Stored {len(UNIQUE_FACTS)} unique facts")
    print(f"  Nodes: {len(memory.field.nodes)}")

    # Test each query
    n_correct = 0
    for i, (q, a, c, t) in enumerate(UNIQUE_FACTS):
        # Test RTMDK retrieval
        emb = embedder(q)
        phase = memory._get_phase("test", emb)
        results = memory.field.query(emb, phase, top_k=3)
        
        # Check if correct answer is in top-3
        found = False
        for nid, score, node in results:
            text = node.content.get("text", "").lower()
            if a.lower().split()[0].lower() in text:
                found = True
                break
        
        if found:
            n_correct += 1
            print(f"  ✓ Q{i+1}: {q[:40]}... → found")
        else:
            print(f"  ✗ Q{i+1}: {q[:40]}... → NOT found (expected: {a})")
            if results:
                print(f"      Got: {results[0][2].content.get('text', '')[:60]}")

    recall = n_correct / len(UNIQUE_FACTS)
    print(f"\n  RTMDK Recall@3: {n_correct}/{len(UNIQUE_FACTS)} = {recall:.0%}")

    # Test BM25
    bm25_correct = 0
    for i, (q, a, c, t) in enumerate(UNIQUE_FACTS):
        bm25_results = bm25.search(q, top_k=3)
        for doc_id, score in bm25_results:
            doc_text = bm25._documents.get(doc_id, "").lower()
            if a.lower().split()[0].lower() in doc_text:
                bm25_correct += 1
                break
    
    bm25_recall = bm25_correct / len(UNIQUE_FACTS)
    print(f"  BM25 Recall@3:  {bm25_correct}/{len(UNIQUE_FACTS)} = {bm25_recall:.0%}")

    return recall, bm25_recall


def test_hybrid_retrieval():
    """Test hybrid retrieval improvement."""
    print(f"\n{'='*60}")
    print(f"  HYBRID RETRIEVAL TEST")
    print(f"{'='*60}")

    embedder = get_embedder()
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=256, top_k=3, min_response=0.001,
            decay_rate=0.999, enable_async=False, bm25_fallback=False,
            use_hnsw=False, learn_projection=False,
        ),
        embedder=embedder,
    )
    bm25 = BM25FallbackRetriever()

    for i, (q, a, c, t) in enumerate(UNIQUE_FACTS):
        emb = embedder(c)
        memory.field.add_node(emb, {"text": c, "topic": t})
        bm25.add_document(f"doc_{i}", c)

    # Test Hybrid
    hybrid = HybridRetriever(memory, bm25)
    for i in range(len(UNIQUE_FACTS)):
        emb = embedder(UNIQUE_FACTS[i][2])
        hybrid.add_embedding(f"n_{i}", emb)

    n_correct = 0
    for i, (q, a, c, t) in enumerate(UNIQUE_FACTS):
        emb = embedder(q)
        results = hybrid.retrieve(q, emb, top_k=3)
        
        found = False
        for nid, score, node in results:
            text = node.content.get("text", "").lower()
            if a.lower().split()[0].lower() in text:
                found = True
                break
        if found: n_correct += 1

    hybrid_recall = n_correct / len(UNIQUE_FACTS)
    print(f"  Hybrid Recall@3: {n_correct}/{len(UNIQUE_FACTS)} = {hybrid_recall:.0%}")
    return hybrid_recall


def test_query_expansion():
    """Test query expansion improvement."""
    print(f"\n{'='*60}")
    print(f"  QUERY EXPANSION TEST")
    print(f"{'='*60}")

    embedder = get_embedder()
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=256, top_k=3, min_response=0.001,
            decay_rate=0.999, enable_async=False, bm25_fallback=True,
            use_hnsw=False, learn_projection=False,
        ),
        embedder=embedder,
    )

    for q, a, c, t in UNIQUE_FACTS:
        emb = embedder(c)
        memory.field.add_node(emb, {"text": c, "topic": t})

    expander = QueryExpander(memory)
    
    # Test expansion on vague-like queries
    test_queries = [
        ("What causes earthquakes?", "Tectonic plates"),
        ("Tell me about paintings", "Leonardo"),
        ("Who created famous plays?", "Shakespeare"),
    ]
    
    for query, expected in test_queries:
        expanded = expander.expand(query)
        print(f"  Original: {query}")
        print(f"  Expanded: {expanded}")
        print(f"  Expected keyword: {expected}")
        print()


def test_temporal_decay():
    """Test temporal decay learning."""
    print(f"\n{'='*60}")
    print(f"  TEMPORAL DECAY TEST")
    print(f"{'='*60}")

    from rtmdk.production.advanced_retrieval import TemporalDecayLearner
    
    learner = TemporalDecayLearner(base_decay=0.999)
    
    # Simulate feedback on nodes
    for i in range(20):
        nid = f"n_{i}"
        quality = 0.9 if i % 3 == 0 else 0.2  # Some good, some bad
        learner.apply_feedback(nid, quality)
    
    stats = learner.stats
    print(f"  Nodes tracked: {stats['nodes_tracked']}")
    print(f"  Avg decay rate: {stats['avg_decay']:.5f}")
    print(f"  Min decay: {stats['min_decay']:.5f} (slower forgetting)")
    print(f"  Max decay: {stats['max_decay']:.5f} (faster forgetting)")


def main():
    print("=" * 60)
    print("  RTMDK — 7 Algorithmic Improvements Benchmark")
    print("=" * 60)

    # 1. Simple retrieval
    rtmdk_recall, bm25_recall = test_simple_retrieval()

    # 2. Hybrid retrieval
    hybrid_recall = test_hybrid_retrieval()

    # 3. Query expansion
    test_query_expansion()

    # 4. Temporal decay
    test_temporal_decay()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  RTMDK Recall@3:    {rtmdk_recall:.0%}")
    print(f"  BM25 Recall@3:     {bm25_recall:.0%}")
    print(f"  Hybrid Recall@3:   {hybrid_recall:.0%}")
    
    best_recall = max(rtmdk_recall, bm25_recall, hybrid_recall)
    print(f"  Best Recall@3:     {best_recall:.0%}")

    # Save report
    report = {
        "rtmdk_recall": rtmdk_recall,
        "bm25_recall": bm25_recall,
        "hybrid_recall": hybrid_recall,
        "best_recall": best_recall,
        "n_facts": len(UNIQUE_FACTS),
    }
    with open("improvements_final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to improvements_final_report.json")


if __name__ == "__main__":
    main()
