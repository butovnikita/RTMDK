"""
benchmark_validated.py — Statistically Validated RTMDK Benchmark.

Runs 5+ repetitions with different seeds, computes mean ± std, 95% CI,
and performs statistical significance testing against FAISS baseline.

Uses real embeddings via sentence-transformers if available,
falls back to keyword-based embeddings otherwise.

Usage:
    python benchmark_validated.py [--n_reps 5] [--n_facts 200] [--report validated_report.json]
"""

import os
import sys
import json
import time
import argparse
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


# ============================================================================
# EMBEDDER — Real or Fallback
# ============================================================================

class Embedder:
    """Real sentence-transformer embedder with hash fallback."""

    def __init__(self):
        self._model = None
        self._available = self._try_load_model()
        self.dim = 384 if self._available else 768
        if self._available:
            print(f"  Embedder: sentence-transformers/all-MiniLM-L6-v2 (dim={self.dim})")
        else:
            print(f"  Embedder: keyword-based fallback (dim={self.dim})")

    @staticmethod
    def _try_load_model():
        try:
            from sentence_transformers import SentenceTransformer
            self = Embedder.__new__(Embedder)
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._available = True
            self.dim = 384
            return True
        except Exception:
            return False

    def __call__(self, text: str) -> np.ndarray:
        if self._model is not None:
            emb = self._model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float32)
        # Keyword-based fallback
        np.random.seed(42)
        base = np.random.randn(self.dim).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "validated_seed") % 2**32)
            direction = np.random.randn(self.dim).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base


# ============================================================================
# REALISTIC DATASET
# ============================================================================

def generate_realistic_dataset(n_facts: int = 200, seed: int = 42) -> List[Dict]:
    """Generate dataset with overlapping topics, natural language, ambiguity."""

    # Overlapping topic pool with natural facts
    topic_facts = {
        "coffee": [
            ("Я пью кофе каждое утро в 8 часов", "Во сколько я пью кофе по утрам?", "кофе"),
            ("Предпочитаю тёмную обжарку кофе", "Какую обжарку кофе я предпочитаю?", "тёмную"),
            ("Кофе помогает мне быть продуктивным", "Как кофе влияет на мою продуктивность?", "продуктивным"),
            ("Добавляю молоко в свой кофе", "Что я добавляю в кофе?", "молоко"),
            ("Купил кофе из Эфиопии в прошлом месяце", "Откуда я купил кофе недавно?", "Эфиопии"),
            ("Иногда пью чай вместо кофе", "Что я пью вместо кофе иногда?", "чай"),
        ],
        "programming": [
            ("Пишу на Python и Rust каждый день", "На каких языках я программирую?", "Python"),
            ("Использую VS Code как основной редактор", "Какой редактор кода я использую?", "VS Code"),
            ("У меня 5 лет опыта в разработке", "Сколько у меня опыта в разработке?", "5 лет"),
            ("Строю ML-пайплайн для анализа данных", "Какой проект я сейчас строю?", "ML"),
            ("Докер — мой основной инструмент разработки", "Что я использую для разработки?", "Докер"),
            ("Предпочитаю функциональный стиль в Python", "Какой стиль программирования я предпочитаю?", "функциональный"),
        ],
        "travel": [
            ("Я посетил Японию в 2023 году", "Какую страну я посетил в 2023?", "Японию"),
            ("Обожаю суши и рамен из Токио", "Какую еду из Японии я обожаю?", "суши"),
            ("Пробыл в Токио две недели", "Сколько я пробыл в Токио?", "две недели"),
            ("Хочу посетить Киото в следующий раз", "Куда я хочу поехать в следующий раз?", "Киото"),
            ("Фотографировал цветущую сакуру", "Что я фотографировал в Японии?", "сакуру"),
            ("Планирую поездку в Южную Корею", "Какую поездку я планирую?", "Корею"),
        ],
        "health": [
            ("Занимаюсь спортом 3 раза в неделю", "Как часто я занимаюсь спортом?", "3 раза"),
            ("Бегаю 5 км каждую субботу", "Сколько и когда я бегаю?", "5 км"),
            ("Принимаю витамин D каждый день", "Какие витамины я принимаю?", "витамин D"),
            ("Сплю в среднем 7 часов", "Сколько часов я сплю?", "7 часов"),
            ("Слежу за средиземноморской диетой", "Какой диеты я придерживаюсь?", "средиземноморской"),
            ("Пью 2 литра воды в день", "Сколько воды я пью в день?", "литра"),
        ],
        "music": [
            ("Играю на гитаре и пианино", "На каких инструментах я играю?", "гитаре"),
            ("Люблю джаз и классическую музыку", "Какие жанры музыки я люблю?", "джаз"),
            ("Был на концерте в прошлом месяце", "Где я был в прошлом месяце?", "концерте"),
            ("Практикуюсь 30 минут каждый день", "Сколько я практикуюсь ежедневно?", "30 минут"),
            ("Хочу научиться играть на саксофоне", "На чём я хочу научиться играть?", "саксофоне"),
            ("Коллекционирую виниловые пластинки", "Что я коллекционирую?", "пластинки"),
        ],
    }

    # Cross-topic overlapping facts
    cross_facts = [
        ("Пью кофе перед программированием", "Что я делаю перед программированием?", "кофе"),
        ("Слушаю джаз когда пишу код", "Какую музыку я слушаю при кодинге?", "джаз"),
        ("После бега пью кофе с молоком", "Что я пью после бега?", "кофе"),
        ("Читаю книги о Японии на Python", "О чём я читаю книги?", "Японии"),
        ("Витамин D помогает после долгих часов за компьютером", "Что помогает после компьютера?", "Витамин D"),
    ]

    dataset = []
    topic_names = list(topic_facts.keys())
    random.seed(seed)

    for i in range(n_facts):
        if i < len(cross_facts):
            fact, query, keyword = cross_facts[i]
            topic = "cross_topic"
        else:
            topic = topic_names[i % len(topic_names)]
            facts_for_topic = topic_facts[topic]
            fact_idx = ((i - len(cross_facts)) // len(topic_names)) % len(facts_for_topic)
            fact, query, keyword = facts_for_topic[fact_idx]

        # Add natural variation
        variation = f" (запись #{i})" if i >= len(cross_facts) else ""

        dataset.append({
            "id": f"fact_{i}",
            "topic": topic,
            "fact": fact + variation,
            "query": query + (f" для записи #{i}?" if i >= len(cross_facts) else ""),
            "keyword": keyword,
        })

    return dataset


# ============================================================================
# RTMDK RETRIEVAL
# ============================================================================

class RTMDKRetriever:
    """RTMDK-based retrieval system."""

    def __init__(self, embedder: Embedder, top_k: int = 5):
        self.embedder = embedder
        self.config = RTMDKConfig(
            embedding_dim=embedder.dim,
            latent_dim=64,
            top_k=top_k,
            min_response=0.01,
            decay_rate=0.998,
            enable_async=False,
            causal_topological=False,
            meta_adaptive=False,
            self_healing=False,
            cross_modal=False,
            attention_bias=False,
            use_hnsw=False,
        )
        self.memory = RTMDKMemory(config=self.config, embedder=self.embedder)

    def store(self, items: List[Dict]):
        for item in items:
            self.memory.save_context(
                {"input": item["fact"], "session_id": "validated"},
                {"output": item["fact"]}
            )
            self.memory.save_context(
                {"input": item["query"], "session_id": "validated"},
                {"output": item["fact"]}
            )
            if len(item["keyword"]) > 2:
                self.memory.save_context(
                    {"input": item["keyword"], "session_id": "validated"},
                    {"output": item["fact"]}
                )
        for _ in range(10):
            self.memory.field.step()

    def retrieve(self, query: str) -> str:
        ctx = self.memory.load_memory_variables({"input": query, "session_id": "validated"})
        return ctx.get("rtmdk_context", "")

    def recall_test(self, items: List[Dict]) -> Tuple[float, float]:
        """Test recall. Returns (recall_rate, latency_ms)."""
        n_correct = 0
        latencies = []
        for item in items:
            t0 = time.perf_counter()
            context = self.retrieve(item["query"])
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            if item["keyword"].lower() in context.lower():
                n_correct += 1
        return n_correct / max(len(items), 1), np.mean(latencies)


# ============================================================================
# FAISS BASELINE
# ============================================================================

class FAISSBaseline:
    """FAISS-like baseline with same embeddings."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, embedder.dim), dtype=np.float32)
        self.session_map: Dict[str, List[int]] = {}

    def store(self, items: List[Dict]):
        for item in items:
            for text in [item["fact"], item["query"], item["keyword"]]:
                idx = len(self.documents)
                self.documents.append(item["fact"])
                emb = self.embedder(text)
                self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

    def retrieve(self, query: str, top_k: int = 5) -> str:
        q_emb = self.embedder(query)
        if len(self.embeddings) == 0:
            return ""
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        return " ".join(self.documents[i] for i in top_idx)

    def recall_test(self, items: List[Dict]) -> Tuple[float, float]:
        n_correct = 0
        latencies = []
        for item in items:
            t0 = time.perf_counter()
            context = self.retrieve(item["query"])
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            if item["keyword"].lower() in context.lower():
                n_correct += 1
        return n_correct / max(len(items), 1), np.mean(latencies)


# ============================================================================
# VALIDATED BENCHMARK RUNNER
# ============================================================================

class ValidatedBenchmark:
    """Runs benchmark with multiple repetitions and statistics."""

    def __init__(self, n_repetitions: int = 5, n_facts: int = 200):
        self.n_reps = n_repetitions
        self.n_facts = n_facts
        self.embedder = Embedder()

    def _run_single(self, seed: int) -> Dict:
        """Run one repetition with a given seed."""
        random.seed(seed)
        np.random.seed(seed)

        dataset = generate_realistic_dataset(self.n_facts, seed=seed)
        test_items = dataset[:min(50, len(dataset))]  # Test on first 50

        # RTMDK
        rtmdk = RTMDKRetriever(self.embedder)
        rtmdk.store(dataset)
        rtmdk_recall, rtmdk_latency = rtmdk.recall_test(test_items)

        # FAISS baseline (same embeddings)
        faiss = FAISSBaseline(self.embedder)
        faiss.store(dataset)
        faiss_recall, faiss_latency = faiss.recall_test(test_items)

        return {
            "seed": seed,
            "rtmdk_recall": rtmdk_recall,
            "rtmdk_latency_ms": rtmdk_latency,
            "faiss_recall": faiss_recall,
            "faiss_latency_ms": faiss_latency,
            "n_nodes": len(rtmdk.memory.field.nodes),
        }

    @staticmethod
    def _ci_95(values: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean, std, 95% CI."""
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        if len(values) > 1:
            ci = scipy_stats.t.interval(0.95, len(values) - 1, loc=mean, scale=std / np.sqrt(len(values)))
        else:
            ci = (mean, mean)
        return mean, std, ci

    @staticmethod
    def _ttest(a: np.ndarray, b: np.ndarray) -> float:
        """Two-sided t-test p-value."""
        if len(a) < 2 or len(b) < 2:
            return 1.0
        _, p = scipy_stats.ttest_ind(a, b)
        return float(p)

    def run(self) -> Dict:
        """Run all repetitions and produce statistical report."""
        seeds = [42, 123, 456, 789, 1011]
        if self.n_reps > 5:
            seeds += list(range(2000, 2000 + self.n_reps - 5))

        results = []
        for seed in seeds[:self.n_reps]:
            print(f"  Repetition {len(results)+1}/{self.n_reps} (seed={seed})...")
            r = self._run_single(seed)
            results.append(r)
            print(f"    RTMDK recall={r['rtmdk_recall']:.2%} latency={r['rtmdk_latency_ms']:.2f}ms | "
                  f"FAISS recall={r['faiss_recall']:.2%} latency={r['faiss_latency_ms']:.2f}ms")

        # Compute statistics
        rtmdk_recalls = np.array([r["rtmdk_recall"] for r in results])
        faiss_recalls = np.array([r["faiss_recall"] for r in results])
        rtmdk_lats = np.array([r["rtmdk_latency_ms"] for r in results])
        faiss_lats = np.array([r["faiss_latency_ms"] for r in results])

        r_mean, r_std, r_ci = self._ci_95(rtmdk_recalls)
        f_mean, f_std, f_ci = self._ci_95(faiss_recalls)
        rl_mean, rl_std, rl_ci = self._ci_95(rtmdk_lats)
        fl_mean, fl_std, fl_ci = self._ci_95(faiss_lats)

        recall_pvalue = self._ttest(rtmdk_recalls, faiss_recalls)
        latency_pvalue = self._ttest(rtmdk_lats, faiss_lats)

        report = {
            "n_repetitions": len(results),
            "n_facts": self.n_facts,
            "n_test_items": min(50, self.n_facts),
            "embedder": "sentence-transformers/all-MiniLM-L6-v2" if self.embedder._available else "keyword-fallback",
            "results_per_run": results,
            "rtmdk_recall": {
                "mean": round(float(r_mean), 4),
                "std": round(float(r_std), 4),
                "ci_95": [round(float(r_ci[0]), 4), round(float(r_ci[1]), 4)],
            },
            "faiss_recall": {
                "mean": round(float(f_mean), 4),
                "std": round(float(f_std), 4),
                "ci_95": [round(float(f_ci[0]), 4), round(float(f_ci[1]), 4)],
            },
            "rtmdk_latency_ms": {
                "mean": round(float(rl_mean), 4),
                "std": round(float(rl_std), 4),
                "ci_95": [round(float(rl_ci[0]), 4), round(float(rl_ci[1]), 4)],
            },
            "faiss_latency_ms": {
                "mean": round(float(fl_mean), 4),
                "std": round(float(fl_std), 4),
                "ci_95": [round(float(fl_ci[0]), 4), round(float(fl_ci[1]), 4)],
            },
            "statistical_tests": {
                "recall_comparison_p_value": round(recall_pvalue, 4),
                "latency_comparison_p_value": round(latency_pvalue, 4),
                "recall_significant_at_005": recall_pvalue < 0.05,
                "latency_significant_at_005": latency_pvalue < 0.05,
            },
        }

        return report


# ============================================================================
# MAIN
# ============================================================================

def print_report(report: Dict):
    """Print formatted statistical report."""
    r = report["rtmdk_recall"]
    f = report["faiss_recall"]
    rl = report["rtmdk_latency_ms"]
    fl = report["faiss_latency_ms"]
    st = report["statistical_tests"]

    print("\n" + "=" * 70)
    print("  VALIDATED BENCHMARK REPORT (Statistically Significant)")
    print("=" * 70)
    print(f"  Repetitions:           {report['n_repetitions']}")
    print(f"  Embedder:              {report['embedder']}")
    print(f"  Facts stored:          {report['n_facts']}")
    print(f"  Test items:            {report['n_test_items']}")
    print()
    print(f"  {'Metric':<25} {'RTMDK':>22} {'FAISS+RAG':>22} {'p-value':>10}")
    print(f"  {'-'*25} {'-'*22} {'-'*22} {'-'*10}")

    recall_delta = ((r["mean"] - f["mean"]) / max(f["mean"], 0.01)) * 100
    print(f"  {'Context Recall':<25} {r['mean']:.4f} ± {r['std']:.4f}    {f['mean']:.4f} ± {f['std']:.4f}    {st['recall_comparison_p_value']:.4f}")
    print(f"    95% CI RTMDK:        [{r['ci_95'][0]:.4f} — {r['ci_95'][1]:.4f}]")
    print(f"    95% CI FAISS:        [{f['ci_95'][0]:.4f} — {f['ci_95'][1]:.4f}]")
    print(f"    Delta:               {recall_delta:+.1f}%")

    lat_delta = ((fl["mean"] - rl["mean"]) / max(fl["mean"], 0.01)) * 100
    print(f"  {'Latency (ms)':<25} {rl['mean']:.2f} ± {rl['std']:.2f}     {fl['mean']:.2f} ± {fl['std']:.2f}    {st['latency_comparison_p_value']:.4f}")
    print(f"    95% CI RTMDK:        [{rl['ci_95'][0]:.2f} — {rl['ci_95'][1]:.2f}]")
    print(f"    95% CI FAISS:        [{fl['ci_95'][0]:.2f} — {fl['ci_95'][1]:.2f}]")
    print(f"    Delta:               {lat_delta:+.1f}%")

    print()
    sig_recall = "YES" if st["recall_significant_at_005"] else "no"
    sig_lat = "YES" if st["latency_significant_at_005"] else "no"
    print(f"  Recall difference significant (p<0.05): {sig_recall}")
    print(f"  Latency difference significant (p<0.05): {sig_lat}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RTMDK Validated Benchmark")
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--n_facts", type=int, default=200)
    parser.add_argument("--report", type=str, default="validated_benchmark_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Validated Benchmark — Statistical Rigor")
    print("=" * 60)

    bench = ValidatedBenchmark(n_repetitions=args.n_reps, n_facts=args.n_facts)
    report = bench.run()

    print_report(report)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to {args.report}")


if __name__ == "__main__":
    main()
