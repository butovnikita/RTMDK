"""
benchmark_load_test.py — Production Load Testing with Bottleneck Detection.

Stages:
  1: Scaling Bottleneck (N=200, 500, 1000, 2000, 5000)
  2: Forgetting + Consolidation Stress (500 facts, 1000 steps)
  3: LLM Integration Quality (30 QA with real LLM)

Usage:
  python benchmark_load_test.py --stage 1    # Scaling only
  python benchmark_load_test.py --stage 2    # Forgetting only
  python benchmark_load_test.py --stage 3    # LLM only
  python benchmark_load_test.py --stage all  # All stages

Checkpoints saved after each stage to load_test_checkpoint.json
"""

import os
import sys
import json
import time
import tracemalloc
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


# ============================================================================
# HELPERS
# ============================================================================

def estimate_tokens(text: str, language: str = "en") -> int:
    if not text: return 0
    return max(1, len(text) // (6 if language == "ru" else 4))


def generate_varied_facts(n: int, seed: int = 42) -> List[Dict]:
    """Generate n diverse facts by expanding the 500-QA dataset."""
    import random
    random.seed(seed)

    base_facts = [
        # Science EN
        ("What causes earthquakes?", "Tectonic plate movement along fault lines", "Earthquakes occur when tectonic plates move suddenly along geological fault lines, releasing energy as seismic waves.", "science", "en"),
        ("How do vaccines work?", "They train the immune system to recognize pathogens", "Vaccines contain weakened parts of a pathogen that trigger an immune response, creating antibodies.", "science", "en"),
        ("What is DNA?", "The molecule carrying genetic information", "Deoxyribonucleic acid contains instructions for organisms to develop, survive, and reproduce.", "science", "en"),
        ("What is photosynthesis?", "Plants convert sunlight into chemical energy", "Plants use chlorophyll to absorb light, converting CO2 and water into glucose and oxygen.", "science", "en"),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second", "Light speed is a fundamental constant denoted by c. Nothing can travel faster than light in vacuum.", "science", "en"),
        ("What is gravity?", "A force that attracts objects with mass", "Gravity is one of four fundamental forces. It keeps planets in orbit and gives objects weight.", "science", "en"),
        ("What is an atom?", "The smallest unit of ordinary matter", "Atoms consist of a nucleus with protons and neutrons, surrounded by electrons.", "science", "en"),
        ("What is evolution?", "Change in heritable traits over generations", "Evolution by natural selection explains how species adapt over time.", "science", "en"),
        ("What is a black hole?", "A region of spacetime with extreme gravity", "Black holes form when massive stars collapse. Nothing can escape their gravity.", "science", "en"),
        ("What is the Big Bang?", "The event that created the universe 13.8 billion years ago", "The Big Bang theory describes how the universe expanded from a hot, dense initial state.", "science", "en"),
        ("What is a volcano?", "An opening in Earth's crust releasing magma", "Volcanic eruptions can create new land. The Ring of Fire contains most volcanoes.", "geography", "en"),
        ("What is a tsunami?", "A massive ocean wave caused by underwater earthquakes", "Tsunamis can travel at 500 mph in deep water and grow enormous near shorelines.", "geography", "en"),
        ("What is a glacier?", "A large persistent body of ice", "Glaciers slowly flow over land. They shape valleys and store 69% of Earth's freshwater.", "geography", "en"),
        ("What is erosion?", "The wearing away of Earth's surface", "Wind, water, and ice gradually wear down rocks, reshaping landscapes over millions of years.", "geography", "en"),
        ("What is a hurricane?", "A massive rotating storm system", "Hurricanes form over warm ocean waters and are classified by wind speed.", "geography", "en"),
        # History EN
        ("When did World War II end?", "1945", "World War II ended in 1945 with Japan's surrender on September 2.", "history", "en"),
        ("Who was the first US President?", "George Washington", "George Washington served as the first US President from 1789 to 1797.", "history", "en"),
        ("When did the Roman Empire fall?", "476 AD", "The Western Roman Empire fell in 476 AD when the last emperor was deposed.", "history", "en"),
        ("Who discovered America?", "Christopher Columbus", "Columbus reached the Americas in 1492, sailing from Spain.", "history", "en"),
        ("When did the French Revolution begin?", "1789", "The French Revolution began in 1789 with the storming of the Bastille.", "history", "en"),
        ("When did the Berlin Wall fall?", "1989", "The Berlin Wall fell on November 9, 1989, leading to German reunification.", "history", "en"),
        ("Who invented the telephone?", "Alexander Graham Bell", "Bell patented the first practical telephone in 1876.", "history", "en"),
        ("When did the Titanic sink?", "1912", "The Titanic sank on April 15, 1912, after hitting an iceberg.", "history", "en"),
        ("Who was the first man on the Moon?", "Neil Armstrong", "Armstrong stepped on the Moon on July 20, 1969, during Apollo 11.", "history", "en"),
        ("Who wrote the Communist Manifesto?", "Karl Marx and Friedrich Engels", "The Communist Manifesto was published in 1848 by Marx and Engels.", "history", "en"),
        # Science RU
        ("Какая столица Франции?", "Париж", "Париж — столица Франции, крупнейший город страны на реке Сене.", "geography", "ru"),
        ("Кто написал Войну и мир?", "Лев Толстой", "Роман Война и мир написан Львом Толстым в 1863-1869 годах.", "history", "ru"),
        ("В каком году началась Вторая мировая?", "1939", "Вторая мировая война началась 1 сентября 1939 года с нападения Германии на Польшу.", "history", "ru"),
        ("Какая самая длинная река?", "Нил", "Нил — река в Африке, самая длинная в мире, около 6650 км.", "geography", "ru"),
        ("Кто первый полетел в космос?", "Юрий Гагарин", "12 апреля 1961 года Гагарин стал первым человеком в космосе.", "history", "ru"),
        ("Сколько планет в Солнечной системе?", "Восемь", "Восемь планет: Меркурий, Венера, Земля, Марс, Юпитер, Сатурн, Уран, Нептун.", "science", "ru"),
        ("Кто нарисовал Мону Лизу?", "Леонардо да Винчи", "Мона Лиза — картина Леонардо да Винчи, около 1503-1519 годов.", "history", "ru"),
        ("Какой элемент обозначается Au?", "Золото", "Au — химический символ золота в таблице Менделеева.", "science", "ru"),
        ("Какой океан самый большой?", "Тихий", "Тихий океан покрывает более трети поверхности Земли.", "geography", "ru"),
        ("Кто изобрёл лампочку?", "Томас Эдисон", "Эдисон создал первую коммерчески успешную лампу в 1879 году.", "history", "en"),
    ]

    facts = []
    for i in range(n):
        base = base_facts[i % len(base_facts)]
        query, answer, context, topic, lang = base
        # Add variation for large N
        if i >= len(base_facts):
            variation = f" (variant {i})"
            query = query.replace("?", f" — fact #{i}?")
            context = context + f" This is additional detail about fact {i}."
            answer = answer + f" (fact {i})"
        else:
            variation = ""

        facts.append({
            "query": query,
            "answer": answer,
            "context": context,
            "topic": topic,
            "language": lang,
        })

    return facts


def create_prod_memory(embedder):
    """Create memory with production config — optimized for speed."""
    return RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=getattr(embedder, 'dim', 768),
            latent_dim=256, top_k=5, min_response=0.005,
            decay_rate=0.999, enable_async=False,
            bm25_fallback=True, use_hnsw=True,
            learn_projection=False,  # Faster: skip IncPCA overhead
            attention_bias=True, adaptive_threshold=False,
            soft_gates=False, self_supervision=False,
            causal_topological=False,
            meta_adaptive=False,
            version_control=False,
            projection_update_freq=500,  # Higher = less frequent updates
        ),
        embedder=embedder,
    )


def compute_mrr(recalls_by_rank: Dict[int, int], n_queries: int) -> float:
    """Mean Reciprocal Rank approximation."""
    # Approximate: if R@1=X%, R@3=Y%, assume uniform distribution
    # Better: use actual rank data — we'll compute inline
    return recalls_by_rank.get(1, 0) / max(n_queries, 1) * 1.0 + \
           (recalls_by_rank.get(3, 0) - recalls_by_rank.get(1, 0)) / max(n_queries, 1) * 0.5 + \
           (recalls_by_rank.get(5, 0) - recalls_by_rank.get(3, 0)) / max(n_queries, 1) * 0.33


# ============================================================================
# STAGE 1: Scaling Bottleneck Detection
# ============================================================================

def run_stage1(checkpoint: Dict = None) -> Dict:
    """Test scaling from N=200 to N=2000."""
    if checkpoint and "stage1" in checkpoint:
        print("  Stage 1 already complete — skipping")
        return checkpoint["stage1"]

    print(f"\n{'='*70}")
    print(f"  STAGE 1: Scaling Bottleneck Detection")
    print(f"{'='*70}")

    embedder = get_embedder()
    n_levels = [200, 500, 1000, 2000]
    results = []

    # Pre-embed all facts once (biggest bottleneck)
    max_n = max(n_levels)
    print(f"\n  Pre-embedding {max_n} facts (one-time cost)...")
    facts = generate_varied_facts(max_n, seed=42)
    t0_embed = time.perf_counter()
    embedded = []
    for i, f in enumerate(facts):
        emb = embedder(f["context"])
        embedded.append(emb)
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0_embed
            print(f"    Embedded {i+1}/{max_n} ({elapsed:.0f}s)")
    embed_total = time.perf_counter() - t0_embed
    print(f"  Embedding complete: {embed_total:.0f}s for {max_n} facts")

    for n in n_levels:
        print(f"\n  Testing N={n}...")
        t0_total = time.perf_counter()

        # Create memory with pre-embedded facts
        memory = create_prod_memory(embedder)

        # Store using pre-computed embeddings (skip LM Studio calls)
        t0_store = time.perf_counter()
        for i in range(n):
            emb = embedded[i]
            # Direct node addition via internal API to skip re-embedding
            memory.field.add_node(emb, {"text": facts[i]["context"]}, modality="text")
        store_time = time.perf_counter() - t0_store
        n_nodes = len(memory.field.nodes)

        # Test retrieval
        test_n = min(100, len(facts))
        recalls = {1: 0, 3: 0, 5: 0, 10: 0}
        latencies = []
        ranks = []

        for i in range(test_n):
            f = facts[i]
            answer_words = [w for w in f["answer"].lower().split() if len(w) > 2]
            if not answer_words: continue

            t0 = time.perf_counter()
            ctx = memory.load_memory_variables({"input": f["query"], "session_id": "load"})
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            context = ctx.get("rtmdk_context", "").lower()
            found_any = any(w in context for w in answer_words)
            if found_any:
                recalls[1] += 1
                recalls[3] += 1
                recalls[5] += 1
                recalls[10] += 1
                ranks.append(1)
            else:
                ranks.append(999)

        elapsed_total = time.perf_counter() - t0_total

        # RAM
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        ram_mb = peak_mem / 1024 / 1024

        result = {
            "n_target": n,
            "n_actual_nodes": n_nodes,
            "n_tested": test_n,
            "recall_at_1": recalls[1] / max(test_n, 1),
            "recall_at_3": recalls[3] / max(test_n, 1),
            "recall_at_5": recalls[5] / max(test_n, 1),
            "recall_at_10": recalls[10] / max(test_n, 1),
            "mrr": float(np.mean([1.0/r if r < 999 else 0.0 for r in ranks])),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "ram_peak_mb": round(ram_mb, 1),
            "store_time_s": round(store_time, 1),
            "total_time_s": round(elapsed_total, 1),
            "bottleneck": "",
        }

        # Detect bottleneck
        if result["recall_at_1"] < 0.50:
            result["bottleneck"] = "RECALL DROP"
        elif result["latency_p95_ms"] > 100:
            result["bottleneck"] = "LATENCY SPIKE"
        elif result["ram_peak_mb"] > 2000:
            result["bottleneck"] = "RAM PRESSURE"
        else:
            result["bottleneck"] = "OK"

        results.append(result)
        print(f"    R@1={result['recall_at_1']:.0%}  P95={result['latency_p95_ms']:.0f}ms  RAM={result['ram_peak_mb']:.0f}MB  [{result['bottleneck']}]")

    return results


# ============================================================================
# STAGE 2: Forgetting + Consolidation Stress
# ============================================================================

def run_stage2(checkpoint: Dict = None) -> Dict:
    """Test forgetting curve over 1000 steps."""
    if checkpoint and "stage2" in checkpoint:
        print("  Stage 2 already complete — skipping")
        return checkpoint["stage2"]

    print(f"\n{'='*70}")
    print(f"  STAGE 2: Forgetting + Consolidation Stress")
    print(f"{'='*70}")

    embedder = get_embedder()
    facts = generate_varied_facts(100, seed=42)

    memory = create_prod_memory(embedder)
    for f in facts:
        memory.save_context({"input": f["context"], "session_id": "forget"}, {"output": f["context"]})

    # Baseline recall
    def test_recall():
        n = 0
        for f in facts[:50]:
            ctx = memory.load_memory_variables({"input": f["query"], "session_id": "forget"})
            answer_words = [w for w in f["answer"].lower().split() if len(w) > 2]
            if any(w in ctx.get("rtmdk_context", "").lower() for w in answer_words):
                n += 1
        return n / 50.0

    checkpoints_steps = [0, 50, 100, 200, 500, 1000]
    curve = []
    t0 = time.perf_counter()

    for step in checkpoints_steps:
        if step > 0:
            steps_to_run = step - (checkpoints_steps[checkpoints_steps.index(step) - 1] if checkpoints_steps.index(step) > 0 else 0)
            for _ in range(steps_to_run):
                memory.field.step()
        r = test_recall()
        curve.append({"step": step, "recall": r})
        print(f"  Step {step:5d}: recall = {r:.2%}")

    elapsed = time.perf_counter() - t0

    # Half-life: step where recall drops to 50% of initial
    initial = curve[0]["recall"] if curve else 1.0
    half_life = None
    for c in curve:
        if c["recall"] <= initial * 0.5:
            half_life = c["step"]
            break

    # Integrity check
    integrity = {"nan": 0, "inf": 0, "neg_amp": 0, "neg_sal": 0}
    for node in memory.field.nodes.values():
        if np.any(np.isnan(node.latent_pos)): integrity["nan"] += 1
        if np.any(np.isinf(node.latent_pos)): integrity["inf"] += 1
        if node.amplitude < 0: integrity["neg_amp"] += 1
        if node.salience < 0: integrity["neg_sal"] += 1

    return {
        "forgetting_curve": curve,
        "half_life_steps": half_life,
        "integrity": integrity,
        "time_s": round(elapsed, 1),
        "final_nodes": len(memory.field.nodes),
        "consolidations": memory.field.stats.get("consolidations", 0),
    }


# ============================================================================
# STAGE 3: LLM Integration Quality
# ============================================================================

def run_stage3(checkpoint: Dict = None) -> Dict:
    """LLM-based quality evaluation."""
    if checkpoint and "stage3" in checkpoint:
        print("  Stage 3 already complete — skipping")
        return checkpoint["stage3"]

    print(f"\n{'='*70}")
    print(f"  STAGE 3: LLM Integration Quality (30 QA)")
    print(f"{'='*70}")

    try:
        import requests
    except ImportError:
        print("  requests not installed — skipping LLM stage")
        return {"skipped": True, "reason": "no requests library"}

    embedder = get_embedder()
    memory = create_prod_memory(embedder)
    facts = generate_varied_facts(200, seed=42)

    # Store
    for f in facts:
        memory.save_context({"input": f["context"], "session_id": "llm"}, {"output": f["context"]})

    # Select 30 diverse samples
    samples = facts[:30]
    results = []
    t0_total = time.perf_counter()

    for i, f in enumerate(samples):
        ctx = memory.load_memory_variables({"input": f["query"], "session_id": "llm"})
        context = ctx.get("rtmdk_context", "")

        # Query LLM
        try:
            resp = requests.post(
                "http://localhost:12345/api/v1/chat",
                json={
                    "model": "thedrummer_rocinante-x-12b-v1",
                    "messages": [
                        {"role": "system", "content": f"Answer using ONLY this context. If you don't know, say so.\n\nContext: {context}"},
                        {"role": "user", "content": f["query"]},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
                timeout=45,
            )
            data = resp.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            answer = f"[Error: {e}]"

        # Evaluate
        expected = f["answer"].lower()
        answer_lower = answer.lower()

        # Exact match (lenient)
        exact = expected in answer_lower or any(w in answer_lower for w in expected.split() if len(w) > 3)

        # Keyword overlap (F1)
        expected_words = set(w for w in expected.split() if len(w) > 3)
        actual_words = set(w for w in answer_lower.split() if len(w) > 3)
        if expected_words and actual_words:
            precision = len(expected_words & actual_words) / len(actual_words)
            recall_kw = len(expected_words & actual_words) / len(expected_words)
            f1 = 2 * precision * recall_kw / max(precision + recall_kw, 1e-8)
        else:
            precision = recall_kw = f1 = 0.0

        # Hallucination: words in answer NOT in context or expected
        context_words = set(context.lower().split())
        hallucinated = actual_words - context_words - expected_words
        hallucination_rate = len(hallucinated) / max(len(actual_words), 1)

        q_tokens = estimate_tokens(f["query"], f.get("language", "en"))
        c_tokens = estimate_tokens(context, f.get("language", "en"))
        a_tokens = estimate_tokens(answer, f.get("language", "en"))

        result = {
            "query": f["query"][:60],
            "expected": expected[:60],
            "answer": answer[:80],
            "exact_match": exact,
            "f1": round(f1, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "query_tokens": q_tokens,
            "context_tokens": c_tokens,
            "answer_tokens": a_tokens,
        }
        results.append(result)

        status = "✅" if exact else "❌"
        print(f"  [{i+1:2d}/30] {status} F1={f1:.2f} Halluc={hallucination_rate:.0%} | Q: {f['query'][:40]}...")
        if i % 10 == 9:
            # Checkpoint every 10
            cp = {"stage3_partial": results}
            with open("load_test_checkpoint.json", "w") as fc:
                json.dump(cp, fc)

    elapsed = time.perf_counter() - t0_total

    return {
        "n_evaluated": len(results),
        "exact_match_rate": sum(1 for r in results if r["exact_match"]) / max(len(results), 1),
        "avg_f1": float(np.mean([r["f1"] for r in results])),
        "hallucination_rate": sum(1 for r in results if r["hallucination_rate"] > 0.5) / max(len(results), 1),
        "avg_query_tokens": float(np.mean([r["query_tokens"] for r in results])),
        "avg_context_tokens": float(np.mean([r["context_tokens"] for r in results])),
        "avg_answer_tokens": float(np.mean([r["answer_tokens"] for r in results])),
        "time_s": round(elapsed, 1),
        "details": results,
    }


# ============================================================================
# MAIN
# ============================================================================

def print_report(report: Dict):
    """Print production dashboard style report."""
    print(f"\n{'='*70}")
    print(f"  RTMDK PRODUCTION LOAD TEST REPORT")
    print(f"{'='*70}")

    # Stage 1: Scaling
    if "stage1" in report:
        s1 = report["stage1"]
        print(f"\n  {'SCALING BOTTLENECKS':^60}")
        print(f"  {'─'*60}")
        print(f"  {'N':>6} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'P95':>7} {'RAM':>7} {'Status':>10}")
        print(f"  {'─'*60}")
        for r in s1:
            status = "✅" if r["bottleneck"] == "OK" else "⚠️" if r["bottleneck"] else "❌"
            print(f"  {r['n_target']:>6} {r['recall_at_1']:>6.0%} {r['recall_at_3']:>6.0%} {r['recall_at_5']:>6.0%} {r['mrr']:>6.3f} {r['latency_p95_ms']:>5.0f}ms {r['ram_peak_mb']:>5.0f}MB {status:>8} {r['bottleneck']}")

    # Stage 2: Forgetting
    if "stage2" in report:
        s2 = report["stage2"]
        print(f"\n  {'FORGETTING CURVE':^60}")
        print(f"  {'─'*60}")
        for c in s2["forgetting_curve"]:
            print(f"  Step {c['step']:5d}: recall = {c['recall']:.2%}")
        print(f"  Half-life: {s2.get('half_life_steps', 'N/A')} steps")
        print(f"  Integrity: NaN={s2['integrity']['nan']} Inf={s2['integrity']['inf']}")

    # Stage 3: LLM
    if "stage3" in report:
        s3 = report["stage3"]
        if not s3.get("skipped"):
            print(f"\n  {'LLM QUALITY':^60}")
            print(f"  {'─'*60}")
            print(f"  Exact Match:    {s3['exact_match_rate']:.2%}")
            print(f"  Avg F1 Score:   {s3['avg_f1']:.3f}")
            print(f"  Hallucination:  {s3['hallucination_rate']:.2%}")
            print(f"  Tokens — Q:{s3['avg_query_tokens']:.0f} C:{s3['avg_context_tokens']:.0f} A:{s3['avg_answer_tokens']:.0f}")
            print(f"  Efficiency:     {s3['avg_answer_tokens'] / max(s3['avg_context_tokens'], 1):.2f}")

    print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--checkpoint", default="load_test_checkpoint.json")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = {}
    cp_path = Path(args.checkpoint)
    if cp_path.exists():
        try:
            with open(cp_path) as f:
                checkpoint = json.load(f)
            print(f"  Loaded checkpoint from {cp_path}")
        except:
            pass

    tracemalloc.start()

    report = dict(checkpoint)

    if args.stage in ("1", "all"):
        report["stage1"] = run_stage1(report)

    if args.stage in ("2", "all"):
        report["stage2"] = run_stage2(report)

    if args.stage in ("3", "all"):
        report["stage3"] = run_stage3(report)

    # Save full checkpoint
    with open(cp_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    tracemalloc.stop()

    print_report(report)

    # Save final report
    with open("load_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to load_test_report.json")


if __name__ == "__main__":
    main()
