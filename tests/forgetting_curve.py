"""
test_forgetting_curve.py — Forgetting Curve with Different Decay Rates.

Tests how RTMDK forgets over time at different decay_rate values.
Also measures BM25 contribution vs resonance-based recall.

Usage:
    python tests/test_forgetting_curve.py
"""

from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try real embedder first, fallback to hash
try:
    from embedder_lmstudio import get_embedder
    embedder_fn = get_embedder()
    USING_REAL_EMBEDDER = getattr(embedder_fn, 'is_real', False)
except Exception as e:
    print(f"  Real embedder unavailable: {e}")
    print("  Using hash-based fallback")
    USING_REAL_EMBEDDER = False

    def make_hash_embedder(dim=768):
        def embed(text):
            rng = np.random.default_rng(42)
            base = rng.standard_normal(dim).astype(np.float32) * 0.01
            tokens = text.lower().split()
            for tok in tokens[:20]:
                tok_rng = np.random.default_rng(hash(tok + "fc_seed") % 2**32)
                d = tok_rng.standard_normal(dim).astype(np.float32)
                d = d / (np.linalg.norm(d) + 1e-8)
                base += d * 0.5
            return base
        return embed
    embedder_fn = make_hash_embedder()


def generate_facts(n=100, seed=42):
    import random
    random.seed(seed)
    facts = []
    for i in range(n):
        topic = ["science", "history", "geography", "tech", "health"][i % 5]
        kw = f"fc_kw_{i:05d}"
        facts.append({
            "fact": f"{topic} fact number {i} with unique keyword {kw}",
            "query": f"What is the {topic} fact number {i} keyword?",
            "keyword": kw,
        })
    return facts


def create_memory(decay_rate, bm25=True, latent_dim=128):
    config = RTMDKConfig(
        embedding_dim=embedder_fn.dim if hasattr(embedder_fn, 'dim') else 768,
        latent_dim=latent_dim, top_k=5,
        min_response=0.001, decay_rate=decay_rate,
        enable_async=False, causal_topological=False,
        meta_adaptive=False, self_healing=False,
        cross_modal=False, attention_bias=False,
        adaptive_threshold=False, bm25_fallback=bm25,
        use_hnsw=False, learn_projection=False,
        soft_gates=False, self_supervision=False,
    )
    return RTMDKMemory(config=config, embedder=embedder_fn)


def store(mem, facts):
    for item in facts:
        mem.save_context({"input": item["fact"], "session_id": "fc"}, {
                         "output": item["fact"]})
        mem.save_context({"input": item["query"], "session_id": "fc"}, {
                         "output": item["fact"]})
        mem.save_context({"input": item["keyword"], "session_id": "fc"}, {
                         "output": item["fact"]})


def recall(mem, facts):
    n = 0
    for item in facts:
        ctx = mem.load_memory_variables(
            {"input": item["query"], "session_id": "fc"})
        c = ctx.get("rtmdk_context", "").lower()
        if item["keyword"] in c:
            n += 1
    return n / max(len(facts), 1)


def run_experiment():
    print(
        f"  Embedder: {'LM Studio (nomic-embed-text-v1.5)' if USING_REAL_EMBEDDER else 'hash-based fallback'}")
    decay_rates = [0.999, 0.995, 0.990, 0.980]
    checkpoints = [0, 50, 100, 200, 500]
    facts = generate_facts(100)
    test_facts = facts[:50]

    results = {}

    for dr in decay_rates:
        print(f"\n{'='*60}")
        print(f"  decay_rate = {dr}")
        print(f"{'='*60}")

        # With BM25
        print("\n  [BM25 ON]")
        mem = create_memory(dr, bm25=True)
        store(mem, facts)
        dr_results_bm25 = {}
        for cp in checkpoints:
            steps = cp - max([k for k in dr_results_bm25] +
                             [0]) if cp > 0 else 0
            for _ in range(steps):
                mem.field.step()
            r = recall(mem, test_facts)
            dr_results_bm25[cp] = r
            nodes = len(mem.field.nodes)
            print(f"    Steps {cp:5d}: recall = {r:6.2%}  nodes = {nodes}")
        results[f"decay_{dr}_bm25"] = dr_results_bm25

        # Without BM25
        print("\n  [BM25 OFF — resonance only]")
        mem = create_memory(dr, bm25=False)
        store(mem, facts)
        dr_results_no_bm25 = {}
        for cp in checkpoints:
            steps = cp - max([k for k in dr_results_no_bm25] +
                             [0]) if cp > 0 else 0
            for _ in range(steps):
                mem.field.step()
            r = recall(mem, test_facts)
            dr_results_no_bm25[cp] = r
            nodes = len(mem.field.nodes)
            print(f"    Steps {cp:5d}: recall = {r:6.2%}  nodes = {nodes}")
        results[f"decay_{dr}_no_bm25"] = dr_results_no_bm25

    # Print summary table
    print(f"\n{'='*80}")
    print("  FORGETTING CURVE SUMMARY")
    print(f"{'='*80}")
    print(f"  {'decay_rate':>10} {'BM25':>5} {'@0':>7} {'@50':>7} {'@100':>7} {'@200':>7} {'@500':>7}")
    print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for dr in decay_rates:
        for suffix, label in [("_bm25", "ON "), ("_no_bm25", "OFF")]:
            key = f"decay_{dr}{suffix}"
            d = results[key]
            vals = [f"{d.get(cp, 0):.2%}" for cp in checkpoints]
            print(
                f"  {dr:>10.3f} {label:>5} {vals[0]:>7} {vals[1]:>7} {vals[2]:>7} {vals[3]:>7} {vals[4]:>7}")
    print(f"{'='*80}")

    # Save report
    report = {
        "decay_rates": decay_rates,
        "checkpoints": checkpoints,
        "results": results}
    with open("forgetting_curve_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n  Report saved to forgetting_curve_report.json")
    return report


if __name__ == "__main__":
    run_experiment()
