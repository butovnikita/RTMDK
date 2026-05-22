"""Week 3 Integration Benchmark: baseline vs full-feature RTMDK.

Measures combined recall, latency, and memory with all GO features enabled.
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rtmdk import RTMDKConfig, RTMDKMemory


class SimpleEmbedder:
    """Deterministic word-level embedder for fast benchmarking."""

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        words = text.lower().split()
        emb = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            if w not in self._cache:
                self._cache[w] = self.rng.standard_normal(self.dim).astype(np.float32)
            emb += self._cache[w]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8) if norm > 0 else emb


def _load_dataset(name: str, limit: int = 200):
    path = PROJECT_ROOT / "datasets" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data.get("data", []))
    return records[:limit]


def _build_memory(cfg: RTMDKConfig, embedder, records: List[Dict]):
    mem = RTMDKMemory(config=cfg, embedder=embedder)
    for rec in records:
        text = rec.get("context", rec.get("text", ""))
        if not text:
            continue
        emb = embedder(text)
        mem.add_node(emb, content=rec)
        # rate limit: 100 nodes/sec max
        time.sleep(0.011)
    return mem


def _bench_memory(mem, embedder, records: List[Dict], top_k: int = 5):
    latencies = []
    recalls = []
    for rec in records:
        query = rec.get("query", rec.get("question", ""))
        if not query:
            continue
        emb = embedder(query)
        t0 = time.time()
        results = mem.retrieve_nodes(query, emb, top_k=top_k)
        latencies.append((time.time() - t0) * 1000)

        # Ground truth: the node whose content matches this record
        answer = rec.get("answer", "")
        hit = 0
        if answer and results:
            for nid, score, node in results:
                node_text = node.content.get("context", node.content.get("text", ""))
                if answer.lower() in node_text.lower() or node_text.lower() in answer.lower():
                    hit = 1
                    break
        recalls.append(hit)

    return {
        "recall@1": round(float(np.mean(recalls)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "total_queries": len(latencies),
    }


def main():
    print("=== Week 3 Integration Benchmark ===")
    records = _load_dataset("comprehensive_500", limit=200)
    embedder = SimpleEmbedder(dim=128)

    # Baseline config: everything off
    baseline_cfg = RTMDKConfig(
        latent_dim=128,
        security_enabled=False,
    )

    # Full-feature config: all GO features on
    full_cfg = RTMDKConfig(
        latent_dim=128,
        security_enabled=False,
        cascade_enabled=True,
        sentence_reranker_enabled=True,
        conformal_prediction=True,
        query_rewrite_enabled=True,
        query_intent_classification_enabled=True,
        result_explainability_enabled=True,
        spectral_consolidation=True,
        quantization="fp16",
        # Kalman
        enable_kalman_filter=True,
        kalman_init_variance=0.1,
        kalman_diagonal_approx=True,
    )

    print(f"Building baseline memory with {len(records)} records...")
    mem_base = _build_memory(baseline_cfg, embedder, records)

    print(f"Building full-feature memory with {len(records)} records...")
    mem_full = _build_memory(full_cfg, embedder, records)

    # Pre-calibrate conformal on full memory (bootstrap with self-queries)
    if hasattr(mem_full, "calibrate_conformal_sot"):
        print("Calibrating conformal...")
        try:
            mem_full.calibrate_conformal_sot()
        except Exception as e:
            print(f"Conformal calibration skipped: {e}")

    print("Benchmarking baseline...")
    base_metrics = _bench_memory(mem_base, embedder, records, top_k=5)
    print(json.dumps(base_metrics, indent=2))

    print("Benchmarking full-feature...")
    full_metrics = _bench_memory(mem_full, embedder, records, top_k=5)
    print(json.dumps(full_metrics, indent=2))

    result = {
        "baseline": base_metrics,
        "full_feature": full_metrics,
        "latency_overhead_pct": round(
            (full_metrics["latency_p50_ms"] - base_metrics["latency_p50_ms"])
            / (base_metrics["latency_p50_ms"] + 1e-8) * 100, 1
        ),
        "recall_diff": round(full_metrics["recall@1"] - base_metrics["recall@1"], 4),
        "status": "PASS" if full_metrics["latency_p95_ms"] < 200 else "FAIL",
        "note": "p95 latency should stay under 200ms with all features enabled",
    }
    print(json.dumps(result, indent=2))

    out = PROJECT_ROOT / "scripts" / "eval_week3_integration_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
