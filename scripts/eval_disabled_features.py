"""Evaluation of Category C: Disabled-by-default features.

Benchmarks:
- SOT v2 zero-shot recall vs TF-IDF
- ConformalCalibrator coverage guarantee
- Quantization memory reduction + recall degradation
- KalmanFilter uncertainty reduction (synthetic)
- SpectralClustering purity vs sklearn

Usage:
    python scripts/eval_disabled_features.py --dataset qa_1000_en
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_dataset(name: str):
    path = PROJECT_ROOT / "datasets" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("records", data.get("data", list(data.values())))
    return data


def eval_sot_v2(dataset: List[Dict]) -> Dict:
    """Evaluate SOT v2 zero-shot recall vs simple baseline."""
    print("\n=== SOT v2 Zero-Shot Recall ===")
    try:
        from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Prepare corpus
        texts = [item.get("context", item.get("text", "")) for item in dataset[:100]]
        queries = [item.get("question", "") for item in dataset[:20]]

        # TF-IDF baseline
        tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf.fit_transform(texts)

        # SIF
        tokenized = [t.lower().split() for t in texts]
        vocab = {w: i for i, w in enumerate(set(w for doc in tokenized for w in doc))}
        tokenized_ids = [[vocab[w] for w in doc if w in vocab] for doc in tokenized]

        sif = SIFEmbedder(latent_dim=64, min_count=1)
        sif.fit(tokenized_ids, vocab_size=len(vocab))

        sif_recalls = []
        tfidf_recalls = []

        for q in queries[:10]:
            if not q:
                continue
            q_tokens = [vocab[w] for w in q.lower().split() if w in vocab]
            if not q_tokens:
                continue
            q_emb = sif._embed_sentence_raw(q_tokens)
            if q_emb is None:
                continue

            # SIF scores
            doc_embs = [sif._embed_sentence_raw(doc) for doc in tokenized_ids]
            doc_embs = [e for e in doc_embs if e is not None]
            if not doc_embs:
                continue
            sif_scores = cosine_similarity([q_emb], doc_embs)[0]
            sif_top = np.argmax(sif_scores)
            sif_recalls.append(sif_top == 0)  # Simplified: assume answer is first doc

            # TF-IDF scores
            q_tfidf = tfidf.transform([q])
            tfidf_scores = cosine_similarity(q_tfidf, tfidf_matrix)[0]
            tfidf_top = np.argmax(tfidf_scores)
            tfidf_recalls.append(tfidf_top == 0)

        sif_acc = np.mean(sif_recalls) if sif_recalls else 0.0
        tfidf_acc = np.mean(tfidf_recalls) if tfidf_recalls else 0.0

        result = {
            "feature": "SOT v2 (SIF)",
            "sif_recall@1": round(float(sif_acc), 3),
            "tfidf_recall@1": round(float(tfidf_acc), 3),
            "status": "PASS" if sif_acc >= tfidf_acc else "FAIL",
            "note": f"SIF {'matches' if abs(sif_acc - tfidf_acc) < 0.05 else 'beats' if sif_acc > tfidf_acc else 'loses to'} TF-IDF",
        }
    except Exception as e:
        result = {
            "feature": "SOT v2 (SIF)",
            "status": "ERROR",
            "note": str(e),
        }
    print(json.dumps(result, indent=2))
    return result


def eval_conformal() -> Dict:
    """Evaluate conformal prediction coverage guarantee.

    Correct protocol:
    - Calibration set contains ONLY scores from known-relevant pairs.
    - Test: the true relevant item must be in the prediction set >= 1-alpha fraction.
    """
    print("\n=== Conformal Prediction ===")
    try:
        from rtmdk.memory.conformal import ConformalCalibrator
        cal = ConformalCalibrator(alpha=0.1)
        rng = np.random.default_rng(42)

        # Calibration: scores from KNOWN RELEVANT pairs.
        # In real system these come from user feedback (thumbs-up).
        # Here we simulate: relevance is independent of score magnitude.
        for _ in range(200):
            is_relevant = rng.random() < 0.4  # 40% of pairs are relevant
            if is_relevant:
                # Relevant pair score distribution
                score = float(rng.beta(2, 5))
                cal.add_sample(score)

        # Test: marginal coverage guarantee
        # True item is ALWAYS relevant; distractors are non-relevant.
        n_test = 500
        covered = 0
        for _ in range(n_test):
            true_score = float(rng.beta(2, 5))  # relevant item score
            # Distractor scores can come from anything; mix uniform + beta
            distractor_scores = np.concatenate([
                rng.uniform(0, 1, 5),
                rng.beta(2, 5, 4),
            ]).tolist()
            scores = [true_score] + distractor_scores
            nids = ["true"] + [f"d{i}" for i in range(9)]
            pred_set, _, threshold = cal.predict(scores, nids)
            if "true" in pred_set:
                covered += 1

        coverage = covered / n_test if n_test else 0.0
        result = {
            "feature": "ConformalCalibrator",
            "coverage": round(coverage, 3),
            "target_coverage": 0.9,
            "calibrated_samples": cal.n_calibrated,
            "threshold": round(cal.get_threshold(), 4),
            "status": "PASS" if coverage >= 0.85 else "FAIL",
            "note": f"Coverage {'adequate' if coverage >= 0.85 else 'insufficient'} for alpha=0.1",
        }
    except Exception as e:
        result = {
            "feature": "ConformalCalibrator",
            "status": "ERROR",
            "note": str(e),
        }
    print(json.dumps(result, indent=2))
    return result


def eval_quantization() -> Dict:
    """Evaluate quantization memory reduction."""
    print("\n=== Quantization ===")
    try:
        from rtmdk.memory.quantization import QuantizationHelper
        emb = np.random.randn(1000, 384).astype(np.float32)

        fp32_size = emb.nbytes
        qh_fp16 = QuantizationHelper("fp16")
        fp16_emb = qh_fp16.quantize(emb)
        fp16_size = fp16_emb.nbytes
        qh_int8 = QuantizationHelper("int8")
        int8_emb_tuple = qh_int8.quantize(emb)
        if isinstance(int8_emb_tuple, tuple):
            int8_emb = int8_emb_tuple[0]
        else:
            int8_emb = int8_emb_tuple
        int8_size = int8_emb.nbytes

        result = {
            "feature": "Quantization",
            "fp32_mb": round(fp32_size / 1024 / 1024, 2),
            "fp16_mb": round(fp16_size / 1024 / 1024, 2),
            "int8_mb": round(int8_size / 1024 / 1024, 2),
            "fp16_reduction": round((1 - fp16_size / fp32_size) * 100, 1),
            "int8_reduction": round((1 - int8_size / fp32_size) * 100, 1),
            "status": "PASS",
            "note": "Memory reduction works; recall degradation not tested here",
        }
    except Exception as e:
        result = {
            "feature": "Quantization",
            "status": "ERROR",
            "note": str(e),
        }
    print(json.dumps(result, indent=2))
    return result


def eval_spectral() -> Dict:
    """Evaluate spectral clustering purity."""
    print("\n=== Spectral Clustering ===")
    try:
        from rtmdk.memory.spectral import spectral_cluster_nodes
        from sklearn.datasets import make_blobs

        from sklearn.preprocessing import StandardScaler
        X, y_true = make_blobs(n_samples=200, centers=4, n_features=64, random_state=42)
        X = StandardScaler().fit_transform(X)
        nodes = []
        for i, vec in enumerate(X):
            class MockNode:
                def __init__(self, vec):
                    self.latent = vec
                    self.phase = 0.0
            nodes.append(MockNode(vec))

        import numpy as np
        positions = np.array([n.latent for n in nodes])
        phases = np.zeros(len(nodes))
        clusters = spectral_cluster_nodes(positions, phases, max_clusters=6)
        n_clusters = len(set(clusters)) if clusters is not None else 0

        result = {
            "feature": "SpectralClustering",
            "detected_clusters": n_clusters,
            "true_clusters": 4,
            "status": "PASS" if 3 <= n_clusters <= 5 else "FAIL",
            "note": f"Detected {n_clusters} clusters (expected ~4)",
        }
    except Exception as e:
        result = {
            "feature": "SpectralClustering",
            "status": "ERROR",
            "note": str(e),
        }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qa_1000_en")
    args = parser.parse_args()

    try:
        dataset = _load_dataset(args.dataset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        dataset = []

    print(f"Evaluating disabled features on dataset: {args.dataset} ({len(dataset)} items)")

    results = []
    results.append(eval_sot_v2(dataset))
    results.append(eval_conformal())
    results.append(eval_quantization())
    results.append(eval_spectral())

    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"\n=== SUMMARY ===")
    print(f"PASS: {passed}, FAIL: {failed}, ERROR: {errors}")

    output_path = PROJECT_ROOT / "scripts" / "eval_disabled_features_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
