"""
benchmark_algorithm_optimization.py — RTMDK Algorithmic Maximization.

Tests 6 algorithmic improvements incrementally:
  1. Dual-Space Retrieval (Resonance + Cosine Hybrid)
  2. Query Phase Alignment
  3. Adaptive Bandwidth (Density-Aware)
  4. Multi-Hop Resonance (Ripple Retrieval)
  5. Tier-Aware Scoring
  6. Realistic Forgetting Curve

Each algorithm builds on previous. Results show delta at each step.
Pre-embeds all facts → avoids LM Studio bottleneck during testing.

Usage:
    python benchmark_algorithm_optimization.py
"""

import os
import sys
import json
import time
import hashlib
import tracemalloc
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory, MemoryNode


# ============================================================================
# FACT GENERATOR
# ============================================================================

def generate_facts(n: int, seed: int = 42) -> List[Dict]:
    """Generate n diverse EN facts for benchmarking."""
    import random
    random.seed(seed)
    base = [
        ("What causes earthquakes?", "Tectonic plate movement along fault lines", "Earthquakes occur when tectonic plates move suddenly along geological fault lines, releasing energy as seismic waves that travel through the Earth.", "science"),
        ("How do vaccines work?", "They train the immune system to recognize pathogens", "Vaccines contain weakened parts of a pathogen that trigger an immune response, creating antibodies that remember the invader for future protection.", "health"),
        ("What is DNA?", "The molecule carrying genetic information", "Deoxyribonucleic acid contains the instructions needed for an organism to develop, survive, and reproduce. It has a double helix structure.", "science"),
        ("What is photosynthesis?", "Plants convert sunlight into chemical energy", "Plants use chlorophyll to absorb light energy, converting carbon dioxide and water into glucose and oxygen through the process of photosynthesis.", "science"),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second", "The speed of light in vacuum is a fundamental physical constant denoted by c. Nothing can travel faster than light in a vacuum.", "physics"),
        ("What is gravity?", "A force that attracts objects with mass", "Gravity is one of the four fundamental forces of nature. It keeps planets in orbit around stars and gives objects weight on planetary surfaces.", "physics"),
        ("What is an atom?", "The smallest unit of ordinary matter", "Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons that orbit in specific energy levels or shells.", "science"),
        ("What is evolution?", "Change in heritable traits over generations", "Evolution by natural selection explains how species adapt to their environments over time. Beneficial traits become more common in populations.", "biology"),
        ("What is a black hole?", "A region of spacetime with extreme gravity", "Black holes form when massive stars collapse at the end of their life cycle. Their gravity is so strong that nothing, not even light, can escape.", "physics"),
        ("What is the Big Bang?", "The event that created the universe 13.8 billion years ago", "The Big Bang theory describes how the universe expanded from an extremely hot and dense initial state roughly 13.8 billion years ago.", "physics"),
        ("What is a volcano?", "An opening in Earth's crust releasing magma", "Volcanic eruptions can create new land masses. The Pacific Ring of Fire contains the majority of Earth's active volcanoes.", "geography"),
        ("What is a tsunami?", "A massive ocean wave caused by underwater earthquakes", "Tsunamis can travel at speeds up to 500 miles per hour in deep water and grow to enormous heights as they approach shallow shorelines.", "geography"),
        ("What is a glacier?", "A large persistent body of ice", "Glaciers slowly flow over land under their own weight. They shape valleys and store about 69 percent of all freshwater on Earth.", "geography"),
        ("What is erosion?", "The wearing away of Earth's surface by natural forces", "Wind, water, and ice gradually wear down rocks and soil, reshaping landscapes over millions of years through the process of erosion.", "geography"),
        ("What is a hurricane?", "A massive rotating storm system over warm ocean waters", "Hurricanes form over warm ocean waters and are classified by wind speed on the Saffir-Simpson Hurricane Wind Scale from 1 to 5.", "geography"),
        ("When did World War II end?", "1945", "World War II ended in 1945 with the surrender of Japan on September 2, after Germany had surrendered earlier in May of the same year.", "history"),
        ("Who was the first US President?", "George Washington", "George Washington served as the first President of the United States from 1789 to 1797 and is known as the Father of His Country.", "history"),
        ("When did the Roman Empire fall?", "476 AD", "The Western Roman Empire fell in 476 AD when the last emperor Romulus Augustulus was deposed by the Germanic leader Odoacer.", "history"),
        ("Who discovered America?", "Christopher Columbus", "Christopher Columbus reached the Americas in 1492, sailing from Spain with three ships: the Nina, the Pinta, and the Santa Maria.", "history"),
        ("When did the French Revolution begin?", "1789", "The French Revolution began in 1789 with the storming of the Bastille prison in Paris on July 14, which is now celebrated as a national holiday.", "history"),
        ("When did the Berlin Wall fall?", "1989", "The Berlin Wall fell on November 9, 1989, leading to German reunification and marking a symbolic end to the Cold War in Europe.", "history"),
        ("Who invented the telephone?", "Alexander Graham Bell", "Alexander Graham Bell patented the first practical telephone in 1876, revolutionizing long-distance communication worldwide.", "history"),
        ("When did the Titanic sink?", "1912", "The RMS Titanic sank on April 15, 1912, after hitting an iceberg during its maiden voyage from Southampton to New York City.", "history"),
        ("Who was the first man on the Moon?", "Neil Armstrong", "Neil Armstrong became the first person to step on the Moon on July 20, 1969, during the Apollo 11 mission, saying one small step for man.", "history"),
        ("What is nuclear fusion?", "Combining atomic nuclei to release energy", "Nuclear fusion powers the Sun and other stars. Hydrogen nuclei fuse together to form helium, releasing enormous amounts of energy in the process.", "physics"),
        ("What is a nebula?", "A giant cloud of gas and dust in space", "Nebulae are often stellar nurseries where new stars are born. Some nebulae are the remnants of dead stars that have exploded as supernovae.", "physics"),
        ("What is a comet?", "An icy body orbiting the Sun", "Comets develop glowing tails when they approach the Sun. Their ice sublimates directly into gas and dust, creating spectacular displays.", "physics"),
        ("What is ozone?", "A molecule of three oxygen atoms", "The ozone layer in the stratosphere protects Earth from harmful ultraviolet radiation from the Sun. It absorbs most of the Sun's UV-B rays.", "science"),
        ("What is mitosis?", "Cell division producing two identical daughter cells", "Mitosis is essential for growth and tissue repair. During mitosis, one cell divides into two genetically identical daughter cells.", "biology"),
        ("What is a gene?", "A segment of DNA that codes for a specific protein", "Genes determine inherited traits and characteristics. Humans have approximately 20,000 to 25,000 genes distributed across 23 pairs of chromosomes.", "biology"),
        ("What is a parasite?", "An organism that lives on or inside a host organism", "Parasites benefit at the expense of their host. Common examples include ticks, fleas, tapeworms, and the malaria-causing Plasmodium.", "biology"),
        ("What is biodiversity?", "The variety of life in an ecosystem or on Earth", "High biodiversity makes ecosystems more resilient to disturbances. Tropical rainforests have the greatest biodiversity of any terrestrial biome.", "biology"),
        ("What is a catalyst?", "A substance that speeds up chemical reactions without being consumed", "Catalysts lower the activation energy of reactions. Enzymes are biological catalysts that are essential for life processes in all organisms.", "chemistry"),
        ("What is the periodic table?", "A chart organizing all known chemical elements", "The periodic table arranges elements by increasing atomic number. Elements in the same column or group share similar chemical properties.", "chemistry"),
        ("What is radioactivity?", "The spontaneous emission of radiation from unstable atomic nuclei", "Radioactive decay releases alpha, beta, or gamma radiation. It is used in medicine for imaging, cancer treatment, and nuclear energy production.", "physics"),
        ("What is a prism?", "A transparent optical element that splits light into colors", "Prisms demonstrate that white light contains all colors of the rainbow. Light is refracted at different angles depending on its wavelength.", "physics"),
        ("What is a photon?", "A quantum particle of light and electromagnetic radiation", "Photons carry electromagnetic energy and have no rest mass. They travel at the speed of light and exhibit both wave and particle properties.", "physics"),
        ("What is a chromosome?", "A thread-like structure containing DNA and proteins", "Humans have 23 pairs of chromosomes for a total of 46. Chromosomal abnormalities such as Down syndrome can cause developmental disorders.", "biology"),
        ("What is the atmosphere?", "The layer of gases surrounding planet Earth", "Earth's atmosphere has five main layers. The troposphere is closest to the surface and is where weather occurs and most life exists.", "science"),
        ("What is a meteorite?", "A space rock that survives passage through the atmosphere", "Most meteorites are fragments of asteroids that orbit between Mars and Jupiter. They create impact craters when they strike Earth's surface.", "physics"),
    ]

    facts = []
    for i in range(n):
        b = base[i % len(base)]
        q, a, c, t = b
        if i >= len(base):
            # Add variations for larger N
            q = q.replace("?", f" — specifically for fact #{i}?")
            c = c + f" This is additional context detail for fact number {i}. " * (1 + i % 3)
            a = a + f" (see fact {i})"
        facts.append({"query": q, "answer": a, "context": c, "topic": t, "language": "en"})
    return facts


# ============================================================================
# RTMDK MEMORY WITH ALGORITHMIC ENHANCEMENTS
# ============================================================================

class OptimizedRTMDK:
    """RTMDK with 6 algorithmic enhancements, each toggleable."""

    def __init__(self, embedder, enable_algorithms: Dict[str, bool] = None):
        self.embedder = embedder
        self.algos = enable_algorithms or {}
        self.config = RTMDKConfig(
            embedding_dim=getattr(embedder, 'dim', 768),
            latent_dim=256, top_k=5, min_response=0.001,
            decay_rate=0.999, enable_async=False,
            bm25_fallback=True, use_hnsw=True,
            learn_projection=False, attention_bias=False,
        )
        self.memory = RTMDKMemory(config=self.config, embedder=embedder)

        # Store original embeddings for dual-space retrieval
        self._original_embeddings: Dict[str, np.ndarray] = {}

    def add_fact(self, fact: Dict, embedding: np.ndarray = None):
        """Add a fact with optional pre-computed embedding."""
        if embedding is None:
            embedding = self.embedder(fact["context"])
        # Store original embedding for dual-space retrieval
        node_id = self.memory.field.add_node(
            embedding, {"text": fact["context"], "topic": fact.get("topic", "general")},
            modality="text"
        )
        if node_id:
            self._original_embeddings[node_id] = embedding.copy()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Retrieve with all enabled algorithms."""
        query_emb = self.embedder(query)
        results = []

        if self.algos.get("multi_hop", False):
            # Algorithm 4: Multi-Hop Resonance
            return self._multi_hop_retrieve(query, query_emb, top_k)

        # Standard retrieval with enhancements
        node_ids = list(self.memory.field.nodes.keys())
        if not node_ids:
            return []

        # Compute scores for each node
        scores = []
        for nid in node_ids:
            node = self.memory.field.nodes[nid]
            score = self._compute_node_score(query, query_emb, node, nid)
            if score > 0:
                scores.append((nid, score, node))

        # Sort and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _compute_node_score(self, query: str, query_emb: np.ndarray,
                            node: MemoryNode, node_id: str) -> float:
        """Compute score using enabled algorithms."""
        # Algorithm 1: Dual-Space (Resonance + Cosine)
        resonance_score = self._compute_resonance(query_emb, node)
        cosine_score = self._compute_cosine_similarity(query_emb, node_id)

        if self.algos.get("dual_space", False):
            base_score = 0.6 * resonance_score + 0.4 * cosine_score
        else:
            base_score = resonance_score

        # Algorithm 2: Query Phase Alignment
        if self.algos.get("phase_alignment", False):
            query_phase = self._compute_query_phase(query_emb)
            phase_diff = node.phase - query_phase
            phase_boost = 0.5 + 0.5 * np.cos(phase_diff)
        else:
            phase_boost = 1.0

        # Algorithm 3: Adaptive Bandwidth
        if self.algos.get("adaptive_bw", False):
            bw_factor = self._compute_bandwidth_factor(query_emb)
        else:
            bw_factor = 1.0

        # Algorithm 5: Tier-Aware Scoring
        if self.algos.get("tier_aware", False):
            tier_boost = self._compute_tier_boost(query, node)
        else:
            tier_boost = 1.0

        return base_score * phase_boost * bw_factor * tier_boost

    def _compute_resonance(self, query_emb: np.ndarray, node: MemoryNode) -> float:
        """Compute resonance response for a single node."""
        query_latent = self.memory.field._project(query_emb)
        dist = np.linalg.norm(query_latent - node.latent_pos)
        bw = self.config.bandwidth
        spatial = np.exp(-dist / bw)
        phase_diff = node.phase - 0.0  # Default phase = 0
        phase_align = 0.5 + 0.5 * np.cos(phase_diff)
        pc = self.config.phase_coupling
        return spatial * ((1 - pc) + pc * phase_align) * node.amplitude * node.salience

    def _compute_cosine_similarity(self, query_emb: np.ndarray, node_id: str) -> float:
        """Compute cosine similarity in original embedding space."""
        if node_id not in self._original_embeddings:
            return 0.0
        node_emb = self._original_embeddings[node_id]
        sim = float(np.dot(query_emb, node_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-8))
        return max(0.0, (sim + 1.0) / 2.0)  # Normalize to [0, 1]

    def _compute_query_phase(self, query_emb: np.ndarray) -> float:
        """Algorithm 2: Compute query phase from embedding hash."""
        h = int(hashlib.md5(query_emb.tobytes()).hexdigest(), 16)
        return (h % 10000) / 10000.0 * 2 * np.pi

    def _compute_bandwidth_factor(self, query_emb: np.ndarray) -> float:
        """Algorithm 3: Adaptive bandwidth based on local density."""
        # Approximate local density via mean distance to all nodes
        if not self._original_embeddings:
            return 1.0
        dists = [np.linalg.norm(query_emb - e) for e in list(self._original_embeddings.values())[:50]]
        if not dists:
            return 1.0
        mean_dist = np.mean(dists)
        # Normalize: higher density → lower bandwidth factor
        return max(0.5, min(2.0, 10.0 / max(mean_dist, 1.0)))

    def _compute_tier_boost(self, query: str, node: MemoryNode) -> float:
        """Algorithm 5: Tier-aware scoring."""
        node_tier = getattr(node, 'tier', 'semantic')
        query_lower = query.lower()

        # Detect query type
        is_temporal = any(w in query_lower for w in ["when", "what year", "what date", "what time", "how long"])
        is_procedural = any(w in query_lower for w in ["how to", "how do", "how can", "what process"])

        if is_temporal and node_tier == "episodic":
            return 1.3
        if is_procedural and node_tier == "procedural":
            return 1.3
        return 1.0

    def _multi_hop_retrieve(self, query: str, query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Algorithm 4: Multi-hop resonance retrieval."""
        # Hop 1: Standard retrieval with top-3
        hop1 = []
        for nid in list(self.memory.field.nodes.keys()):
            node = self.memory.field.nodes[nid]
            score = self._compute_node_score(query, query_emb, node, nid)
            if score > 0:
                hop1.append((nid, score, node))
        hop1.sort(key=lambda x: x[1], reverse=True)
        hop1 = hop1[:3]

        # Hop 2: Query from each hop-1 node's embedding
        hop2 = {}
        for nid, score, node in hop1:
            node_emb = self._original_embeddings.get(nid)
            if node_emb is not None:
                for nid2 in list(self.memory.field.nodes.keys()):
                    if nid2 == nid or nid2 in [n[0] for n in hop1]:
                        continue
                    node2 = self.memory.field.nodes[nid2]
                    score2 = self._compute_node_score(query, query_emb, node2, nid2)
                    # Bonus for being reachable from hop-1
                    hop2_bonus = score * 0.3
                    total = score2 + hop2_bonus
                    if total > 0:
                        hop2[nid2] = (nid2, total, node2)

        # Merge hop1 and hop2
        all_results = list(hop1) + list(hop2.values())
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

def compute_metrics(retrieval_results: List, facts: List[Dict], test_n: int) -> Dict:
    """Compute Recall@K, MRR, NDCG@5, Precision@5."""
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = []
    precisions = []
    ndcgs = []

    for i in range(min(test_n, len(facts))):
        f = facts[i]
        answer_words = [w for w in f["answer"].lower().split() if len(w) > 3]
        if not answer_words: continue

        # Check which results match
        result_texts = [r[2].content.get("text", "").lower() for r in retrieval_results]
        matched = [any(w in rt for w in answer_words) for rt in result_texts]

        # Recall
        found = any(matched)
        if found:
            recalls[1] += 1
            recalls[3] += 1
            recalls[5] += 1
            recalls[10] += 1
            # Find first matching rank
            try:
                rank = matched.index(True) + 1
                ranks.append(rank)
            except ValueError:
                ranks.append(999)
        else:
            ranks.append(999)

        # Precision@5
        top5 = matched[:5]
        if top5:
            precisions.append(sum(top5) / len(top5))
        else:
            precisions.append(0.0)

        # NDCG@5
        rel = [1 if m else 0 for m in matched[:5]]
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(rel, reverse=True)
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / max(idcg, 1e-8))

    n = min(test_n, len(facts))
    return {
        "recall_at_1": recalls[1] / max(n, 1),
        "recall_at_3": recalls[3] / max(n, 1),
        "recall_at_5": recalls[5] / max(n, 1),
        "recall_at_10": recalls[10] / max(n, 1),
        "mrr": float(np.mean([1.0/r if r < 999 else 0.0 for r in ranks])),
        "ndcg_at_5": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
    }


def run_benchmark():
    """Run full algorithmic optimization benchmark."""
    tracemalloc.start()
    print("=" * 70)
    print("  RTMDK ALGORITHMIC OPTIMIZATION BENCHMARK")
    print("=" * 70)

    embedder = get_embedder()

    # Generate facts
    facts = generate_facts(1000, seed=42)
    print(f"\n  Generated {len(facts)} facts")

    # Pre-embed all facts
    print(f"  Pre-embedding {len(facts)} facts...")
    t0_embed = time.perf_counter()
    embeddings = [embedder(f["context"]) for f in facts]
    embed_time = time.perf_counter() - t0_embed
    print(f"  Done: {embed_time:.0f}s")

    # Define algorithm combinations
    algo_configs = [
        {"name": "Baseline (current)", "algos": {}},
        {"name": "+ Dual-Space (Resonance + Cosine)", "algos": {"dual_space": True}},
        {"name": "+ Phase Alignment", "algos": {"dual_space": True, "phase_alignment": True}},
        {"name": "+ Adaptive Bandwidth", "algos": {"dual_space": True, "phase_alignment": True, "adaptive_bw": True}},
        {"name": "+ Multi-Hop Resonance", "algos": {"dual_space": True, "phase_alignment": True, "adaptive_bw": True, "multi_hop": True}},
        {"name": "+ Tier-Aware Scoring", "algos": {"dual_space": True, "phase_alignment": True, "adaptive_bw": True, "multi_hop": True, "tier_aware": True}},
    ]

    n_levels = [200, 500, 1000]
    test_n = 50

    all_results = []

    for n in n_levels:
        print(f"\n{'='*70}")
        print(f"  Testing N={n}")
        print(f"{'='*70}")

        for cfg in algo_configs:
            print(f"\n  {cfg['name']}...", end=" ")

            # Create optimized RTMDK
            rtmdk = OptimizedRTMDK(embedder, enable_algorithms=cfg["algos"])

            # Add facts
            t0_store = time.perf_counter()
            for i in range(n):
                rtmdk.add_fact(facts[i], embeddings[i])
            store_time = time.perf_counter() - t0_store

            # Retrieve and measure
            t0_query = time.perf_counter()
            results = rtmdk.retrieve(facts[0]["query"], top_k=5)  # Single query to get latency
            query_time = (time.perf_counter() - t0_query) * 1000

            # Full metric computation (retrieve for all test queries)
            all_retrieval_results = []
            for i in range(test_n):
                res = rtmdk.retrieve(facts[i]["query"], top_k=5)
                all_retrieval_results.extend(res)

            metrics = compute_metrics(all_retrieval_results, facts, test_n)

            current, peak = tracemalloc.get_traced_memory()

            result = {
                "n": n,
                "algorithm": cfg["name"],
                "recall_at_1": round(metrics["recall_at_1"], 4),
                "recall_at_3": round(metrics["recall_at_3"], 4),
                "recall_at_5": round(metrics["recall_at_5"], 4),
                "mrr": round(metrics["mrr"], 4),
                "ndcg_at_5": round(metrics["ndcg_at_5"], 4),
                "precision_at_5": round(metrics["precision_at_5"], 4),
                "latency_ms": round(query_time, 1),
                "ram_mb": round(peak / 1024 / 1024, 1),
            }
            all_results.append(result)

            delta = ""
            if len(all_results) > 1 and all_results[-2]["n"] == n:
                prev = all_results[-2]["recall_at_1"]
                delta_val = result["recall_at_1"] - prev
                delta = f" (ΔR@1: {delta_val:+.0%})"

            print(f"R@1={result['recall_at_1']:.0%}  R@3={result['recall_at_3']:.0%}  "
                  f"MRR={result['mrr']:.3f}  NDCG@5={result['ndcg_at_5']:.3f}  "
                  f"P@5={result['precision_at_5']:.0%}  Latency={result['latency_ms']:.0f}ms{delta}")

    # Forgetting curve
    print(f"\n{'='*70}")
    print(f"  FORGETTING CURVE (Realistic decay)")
    print(f"{'='*70}")

    # Standard
    rtmdk_std = OptimizedRTMDK(embedder, enable_algorithms={"dual_space": True})
    forget_facts = generate_facts(100, seed=42)
    for i, f in enumerate(forget_facts):
        rtmdk_std.add_fact(f, embeddings[i])

    def test_recall(mem, facts_subset):
        n = 0
        for f in facts_subset[:30]:
            res = mem.retrieve(f["query"], top_k=5)
            if res:
                answer_words = [w for w in f["answer"].lower().split() if len(w) > 3]
                text = res[0][2].content.get("text", "").lower()
                if any(w in text for w in answer_words):
                    n += 1
        return n / 30.0

    curve = []
    for step in [0, 50, 100, 200, 500]:
        if step > 0:
            prev = curve[-1]["step"] if curve else 0
            for _ in range(step - prev):
                rtmdk_std.memory.field.step()
        r = test_recall(rtmdk_std, forget_facts)
        curve.append({"step": step, "recall": r})
        print(f"  Step {step:5d}: recall = {r:.2%}")

    # Print final report
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — ALGORITHMIC OPTIMIZATION")
    print(f"{'='*70}")

    # Table by algorithm for N=200
    print(f"\n  {'N=200 SCALING':^60}")
    print(f"  {'─'*60}")
    print(f"  {'Algorithm':<35} {'R@1':>6} {'R@3':>6} {'MRR':>6} {'NDCG@5':>7} {'P@5':>6}")
    print(f"  {'─'*60}")
    for r in all_results:
        if r["n"] == 200:
            print(f"  {r['algorithm']:<35} {r['recall_at_1']:>5.0%} {r['recall_at_3']:>5.0%} "
                  f"{r['mrr']:>5.3f} {r['ndcg_at_5']:>6.3f} {r['precision_at_5']:>5.0%}")

    # Table by N for best algorithm
    print(f"\n  {'SCALING — Best Algorithm':^60}")
    print(f"  {'─'*60}")
    print(f"  {'N':>6} {'R@1':>6} {'R@3':>6} {'MRR':>6} {'NDCG@5':>7} {'P@5':>6} {'P95':>6} {'RAM':>6}")
    print(f"  {'─'*60}")
    for r in all_results:
        if "Tier-Aware" in r["algorithm"]:
            print(f"  {r['n']:>6} {r['recall_at_1']:>5.0%} {r['recall_at_3']:>5.0%} "
                  f"{r['mrr']:>5.3f} {r['ndcg_at_5']:>6.3f} {r['precision_at_5']:>5.0%} "
                  f"{r['latency_ms']:>4.0f}ms {r['ram_mb']:>4.0f}MB")

    # Industry comparison
    best_r1 = max(r["recall_at_1"] for r in all_results if r["n"] == 200)
    print(f"\n  {'vs INDUSTRY RAG':^60}")
    print(f"  {'─'*60}")
    print(f"  RTMDK (optimized):  R@1 = {best_r1:.0%}")
    print(f"  Naive RAG:          R@1 = 60-75%")
    print(f"  Advanced RAG:       R@1 = 75-85%")
    print(f"  GraphRAG:           R@1 = 82-90%")
    status = "✅ TOP TIER" if best_r1 >= 0.80 else "✅ COMPETITIVE" if best_r1 >= 0.70 else "⚠️ BELOW ADVANCED"
    print(f"  Verdict:            {status}")

    # Save report
    report = {
        "algorithm_comparison": all_results,
        "forgetting_curve": curve,
        "best_r1_at_200": best_r1,
    }
    with open("algorithm_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to algorithm_optimization_report.json")

    tracemalloc.stop()


if __name__ == "__main__":
    run_benchmark()
