import os
os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

"""End-to-end benchmark: RTMDK vs FAISS vs Pure Cosine on identical embeddings.

Fair comparison: all systems receive the SAME SBERT embeddings.
The only variable is the retrieval engine (resonance field vs FAISS vs brute-force).

Datasets:
- qa_1000_en.json (English, lexical overlap)
- comprehensive_500.json (English+Russian, harder paraphrases)
"""

import json
import time
import argparse
from typing import List, Dict

import numpy as np


def load_dataset(path: str, language: str = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data)
    if language:
        records = [r for r in records if r.get("language") == language]
    return records


def evaluate_faiss(doc_embs: np.ndarray, query_embs: np.ndarray,
                   records: List[Dict], corpus_texts: List[str], top_k: int = 5):
    """Evaluate with FAISS flat index (exact inner product on normalized vectors)."""
    import faiss
    d = doc_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(doc_embs.astype(np.float32))

    t0 = time.time()
    scores, indices = index.search(query_embs.astype(np.float32), top_k)
    search_time = time.time() - t0

    hits = 0
    hits1 = 0
    ranks = []
    for i, rec in enumerate(records):
        target = rec["context"]
        target_idx = corpus_texts.index(target)
        if target_idx in indices[i]:
            hits += 1
        rank = np.where(indices[i] == target_idx)[0]
        r = rank[0] + 1 if len(rank) > 0 else None
        ranks.append(1.0 / r if r else 0.0)
        if r == 1:
            hits1 += 1

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "search_time_ms": search_time * 1000,
        "latency_per_query_ms": search_time * 1000 / len(records),
    }


def evaluate_rtmdk(memory, records: List[Dict], corpus_texts: List[str],
                   embedder, top_k: int = 5):
    """Evaluate with RTMDKMemory.retrieve_nodes."""
    hits = 0
    hits1 = 0
    ranks = []
    query_times = []

    for rec in records:
        q_text = rec["query"]
        target = rec["context"]
        q_emb = embedder(q_text)

        t0 = time.perf_counter()
        results = memory.retrieve_nodes(q_text, q_emb, top_k=top_k)
        query_times.append(time.perf_counter() - t0)

        contexts = [r[2].content.get("text", "") for r in results]
        if target in contexts:
            hits += 1
            rank = contexts.index(target) + 1
            ranks.append(1.0 / rank)
            if rank == 1:
                hits1 += 1
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "latency_p50_ms": float(np.percentile(query_times, 50) * 1000),
        "latency_p99_ms": float(np.percentile(query_times, 99) * 1000),
    }


def evaluate_pure_cosine(doc_embs: np.ndarray, records: List[Dict],
                         embedder, corpus_texts: List[str], top_k: int = 5):
    """Pure cosine similarity without any index."""
    query_times = []
    hits = 0
    hits1 = 0
    ranks = []

    for rec in records:
        q_text = rec["query"]
        target = rec["context"]

        t0 = time.perf_counter()
        q_emb = embedder(q_text)
        sims = doc_embs @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        query_times.append(time.perf_counter() - t0)

        target_idx = corpus_texts.index(target)
        if target_idx in top_idx:
            hits += 1
            rank = np.where(top_idx == target_idx)[0][0] + 1
            ranks.append(1.0 / rank)
            if rank == 1:
                hits1 += 1
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "latency_p50_ms": float(np.percentile(query_times, 50) * 1000),
        "latency_p99_ms": float(np.percentile(query_times, 99) * 1000),
    }


def benchmark_dataset(name: str, records: List[Dict], embedder, use_faiss: bool = True, sot_v2=False, sot_align=False, teacher=None):
    corpus_texts = list({r["context"] for r in records})
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | Records: {len(records)} | Contexts: {len(corpus_texts)}")
    print("=" * 60)

    # Build SOT v2 embedder if needed
    if sot_v2:
        from rtmdk.memory.sot_v2.integration import SOTv2Embedder
        all_texts = list(corpus_texts) + [r["query"] for r in records]
        sot = SOTv2Embedder(latent_dim=384, window_size=5, a=0.01, remove_pc=True)
        sot.train(all_texts)
        if sot_align and teacher is not None:
            print("  Aligning SOT v2 to teacher...")
            sot.align_to_teacher(all_texts, teacher.encode, batch_size=64, center=True)
        embedder = sot

    # Pre-compute all embeddings
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    query_embs = np.vstack([embedder(r["query"]) for r in records])
    # Normalize for cosine
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

    # --- Pure Cosine ---
    res_cos = evaluate_pure_cosine(doc_embs, records, embedder, corpus_texts)
    print(f"  Pure Cosine:     recall@5={res_cos['recall@5']:.3f}  recall@1={res_cos.get('recall@1',0):.3f}  MRR={res_cos['mrr']:.3f}  p50={res_cos['latency_p50_ms']:.2f}ms")

    # --- FAISS ---
    if use_faiss:
        res_faiss = evaluate_faiss(doc_embs, query_embs, records, corpus_texts)
        print(f"  FAISS FlatIP:    recall@5={res_faiss['recall@5']:.3f}  recall@1={res_faiss.get('recall@1',0):.3f}  MRR={res_faiss['mrr']:.3f}  total={res_faiss['search_time_ms']:.2f}ms")

    # --- RTMDK (SBERT embeddings, no SOT v2) ---
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKMemory

    cfg = RTMDKConfig(
        latent_dim=384,
        top_k=5,
        use_hnsw=False,
        sparse_routing=False,
        adaptive_phase_coupling=True,
    )
    mem = RTMDKMemory(config=cfg, embedder=embedder)

    for rec in records:
        ctx = rec["context"]
        topic = rec.get("topic", "")
        emb = embedder(ctx)
        content = {"text": ctx}
        if topic:
            content["topic"] = topic
        mem.add_node(emb, content)
    # Also add any contexts that weren't in records
    seen = {r["context"] for r in records}
    for ctx in corpus_texts:
        if ctx not in seen:
            mem.add_node(embedder(ctx), {"text": ctx})

    res_rtmdk = evaluate_rtmdk(mem, records, corpus_texts, embedder)
    label = "RTMDK+SOTv2" if sot_v2 else "RTMDK (SBERT)"
    print(f"  {label:16s} recall@5={res_rtmdk['recall@5']:.3f}  recall@1={res_rtmdk.get('recall@1',0):.3f}  MRR={res_rtmdk['mrr']:.3f}  p50={res_rtmdk['latency_p50_ms']:.2f}ms")

    return {
        "cosine": res_cos,
        "faiss": res_faiss if use_faiss else None,
        "rtmdk": res_rtmdk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "en", "ru", "hard"])
    parser.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no_faiss", action="store_true")
    parser.add_argument("--sot_v2", action="store_true", help="Use SOT v2 self-contained embedder instead of SBERT")
    parser.add_argument("--sot_align", action="store_true", help="Align SOT v2 to teacher via Procrustes")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"Loading teacher: {args.teacher}")
    teacher = SentenceTransformer(args.teacher)
    print(f"Teacher dim: {teacher.get_embedding_dimension()}")

    def make_sot_embedder(corpus_texts, query_texts, teacher_model=None):
        from rtmdk.memory.sot_v2.integration import SOTv2Embedder
        all_texts = list(corpus_texts) + list(query_texts)
        sot = SOTv2Embedder(latent_dim=384, window_size=5, a=0.01, remove_pc=True)
        sot.train(all_texts)
        if args.sot_align and teacher_model is not None:
            print("Aligning SOT v2 to teacher via Procrustes...")
            sot.align_to_teacher(all_texts, teacher_model.encode, batch_size=64, center=True)
        return sot

    if args.sot_v2:
        print("SOT v2 mode: self-contained embedder (no SBERT at inference)")
        # Embedder will be created per-dataset after we know the corpus
        embedder = None
    else:
        def embedder(text: str) -> np.ndarray:
            emb = teacher.encode(text)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            return emb

    results = {}

    if args.dataset in ("all", "en"):
        records = load_dataset("datasets/qa_1000_en.json", language="en")[:200]
        results["qa_1000_en"] = benchmark_dataset("qa_1000_en (200)", records, embedder, not args.no_faiss, args.sot_v2, args.sot_align, teacher)

    if args.dataset in ("all", "ru"):
        records = load_dataset("datasets/comprehensive_500.json", language="ru")
        if records:
            if args.sot_v2:
                embedder_ru = None
                teacher_ru = teacher
            else:
                print("\nLoading multilingual teacher for Russian...")
                teacher_ru = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                def embedder_ru(text: str) -> np.ndarray:
                    emb = teacher_ru.encode(text)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    return emb
            results["comprehensive_500_ru"] = benchmark_dataset(f"comprehensive_500_ru ({len(records)})", records, embedder_ru, not args.no_faiss, args.sot_v2, args.sot_align, teacher_ru)

    if args.dataset in ("all", "hard"):
        records = load_dataset("datasets/comprehensive_500.json", language="en")
        results["comprehensive_500"] = benchmark_dataset(f"comprehensive_500 ({len(records)})", records, embedder, not args.no_faiss, args.sot_v2, args.sot_align, teacher)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds_name, res in results.items():
        print(f"\n{ds_name}:")
        for sys_name, metrics in res.items():
            if metrics:
                print(f"  {sys_name:12s}  recall@5={metrics['recall@5']:.3f}  MRR={metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
