"""
RTMDK Benchmark Automation.

Usage:
    python benchmark.py --dataset datasets/qa_1000_en.json --methods bm25,word,word_fasttext
    python benchmark.py --dataset datasets/qa_1000_en.json --all --output benchmark_results.json
"""
import argparse
import json
import numpy as np
import time
import sys
from pathlib import Path


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", data)


def bm25_baseline(records):
    doc_tokens = []
    for r in records:
        tokens = set((r["context"] + " " + r["answer"]).lower().split())
        doc_tokens.append(tokens)

    hits = []
    for r in records:
        q_tokens = set(r["query"].lower().split())
        scores = [len(q_tokens & dt) for dt in doc_tokens]
        top_idx = np.argsort(scores)[::-1][:5]
        top_topics = [records[i]["topic"] for i in top_idx]
        hits.append(r["topic"] in top_topics[:1])
    return sum(hits) / len(hits)


def sot_benchmark(records, cfg, field=None):
    from rtmdk.memory.core import RTMDKField
    if field is None:
        field = RTMDKField(cfg)
    tok = field.sot_tokenizer

    node_id_to_idx = {}
    for i, r in enumerate(records):
        text = r["context"] + " " + r["answer"]
        tokens = tok.encode(text)
        emb = tok.embed(tokens)
        nid = field.add_node(emb, {"text": text, "topic": r["topic"]}, node_id=f"n{i}")
        node_id_to_idx[nid] = i
        time.sleep(0.011)

    hits = []
    lats = []
    for r in records:
        tokens = tok.encode(r["query"])
        qemb = tok.embed(tokens)
        t0 = time.time()
        res = field.query(qemb, top_k=5)
        lats.append((time.time() - t0) * 1000)
        top_topics = []
        for nid, sc, node in res:
            idx = node_id_to_idx.get(nid)
            if idx is not None:
                top_topics.append(records[idx]["topic"])
        hits.append(r["topic"] in top_topics[:1])

    return {
        "recall_at_1": sum(hits) / len(hits),
        "latency_p50_ms": float(np.median(lats)),
        "vocab_size": len(tok.token_embeddings),
    }


def run_sot_byte(records):
    from rtmdk.memory.config import RTMDKConfig
    cfg = RTMDKConfig(latent_dim=64, sot_enabled=True)
    return sot_benchmark(records, cfg)


def run_sot_word(records):
    from rtmdk.memory.config import RTMDKConfig
    cfg = RTMDKConfig(latent_dim=64, sot_enabled=True, sot_tokenization_mode="word")
    return sot_benchmark(records, cfg)


def run_sot_word_fasttext(records):
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKField
    from rtmdk.memory.bootstrap_fasttext import run_bootstrap
    cfg = RTMDKConfig(latent_dim=64, sot_enabled=True, sot_tokenization_mode="word")
    field = RTMDKField(cfg)
    texts = [r["context"] + " " + r["answer"] for r in records]
    t0 = time.time()
    run_bootstrap(field.sot_tokenizer, texts=texts, model_path="fasttext_bootstrap.model")
    bt = time.time() - t0
    result = sot_benchmark(records, cfg, field=field)
    result["bootstrap_time_s"] = bt
    return result


def run_sot_word_sbert(records):
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKField
    from sentence_transformers import SentenceTransformer
    cfg = RTMDKConfig(latent_dim=64, sot_enabled=True, sot_tokenization_mode="word")
    field = RTMDKField(cfg)
    texts = [r["context"] + " " + r["answer"] for r in records]
    teacher = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.time()
    field.sot_tokenizer.bootstrap_from_teacher(
        texts,
        lambda t: teacher.encode(t, show_progress_bar=False),
        fit_projection_only=False,
        n_epochs=10,
        lr=0.05,
    )
    bt = time.time() - t0
    result = sot_benchmark(records, cfg, field=field)
    result["bootstrap_time_s"] = bt
    return result


METHODS = {
    "bm25": lambda records: {"recall_at_1": bm25_baseline(records)},
    "byte": run_sot_byte,
    "word": run_sot_word,
    "word_fasttext": run_sot_word_fasttext,
    "word_sbert": run_sot_word_sbert,
}


def main():
    parser = argparse.ArgumentParser(description="RTMDK Benchmark")
    parser.add_argument("--dataset", "-d", type=str, default="datasets/qa_1000_en.json")
    parser.add_argument("--methods", "-m", type=str, default="bm25,word,word_fasttext")
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--compare", "-c", type=str, default=None, help="Previous results JSON for comparison")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Records: {len(records)}")
    print()

    methods = list(METHODS.keys()) if args.all else args.methods.split(",")
    results = {}

    for name in methods:
        name = name.strip()
        if name not in METHODS:
            print(f"Unknown method: {name}, skipping")
            continue
        print(f"Running {name}...")
        try:
            result = METHODS[name](records)
            results[name] = result
            print(f"  Recall@1: {result['recall_at_1']:.3f}")
            if "bootstrap_time_s" in result:
                print(f"  Bootstrap: {result['bootstrap_time_s']:.2f}s")
            if "latency_p50_ms" in result:
                print(f"  Latency P50: {result['latency_p50_ms']:.1f}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    # Comparison
    if args.compare and Path(args.compare).exists():
        with open(args.compare, "r", encoding="utf-8") as f:
            prev = json.load(f)
        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"{'Method':<20} {'Current':>10} {'Previous':>10} {'Delta':>10}")
        print("-" * 70)
        for name in methods:
            if name in results and "recall_at_1" in results[name]:
                curr = results[name]["recall_at_1"]
                prev_val = prev.get(name, {}).get("recall_at_1", None)
                if prev_val is not None:
                    delta = curr - prev_val
                    print(f"{name:<20} {curr:>10.3f} {prev_val:>10.3f} {delta:>+10.3f}")
                else:
                    print(f"{name:<20} {curr:>10.3f} {'—':>10} {'—':>10}")

    # Summary table
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'R@1':>8} {'Boot(s)':>10} {'Lat(ms)':>10} {'Vocab':>8}")
    print("-" * 70)
    for name in methods:
        if name in results and "recall_at_1" in results[name]:
            r = results[name]
            boot = f"{r.get('bootstrap_time_s', 0):.2f}" if "bootstrap_time_s" in r else "—"
            lat = f"{r.get('latency_p50_ms', 0):.1f}" if "latency_p50_ms" in r else "—"
            vocab = str(r.get("vocab_size", "—"))
            print(f"{name:<20} {r['recall_at_1']:>8.3f} {boot:>10} {lat:>10} {vocab:>8}")
    print("-" * 70)

    if args.output:
        output_data = {
            "dataset": args.dataset,
            "records": len(records),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
