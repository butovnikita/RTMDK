"""
SOT token_dim scaling benchmark: 64 -> 1536.
Measures embedding quality, memory, and latency after warm-start + contrastive learning.
Compares adaptive_lr (scaled by sqrt(token_dim/latent_dim)) vs fixed lr.
"""
from rtmdk.memory.self_organizing_field import SOTokenizer
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)

LATENT_DIM = 64
CORPUS = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing understands human text",
    "computer vision recognizes objects in images",
    "robotics combines sensors actuators and control systems",
    "reinforcement learning learns from rewards and penalties",
    "supervised learning requires labeled training data",
    "unsupervised learning finds patterns without labels",
    "transfer learning reuses pretrained models",
    "generative ai creates new content like text and images",
    "large language models are trained on vast text corpora",
    "transformer architecture uses self-attention mechanisms",
    "convolutional neural networks excel at image tasks",
    "recurrent neural networks process sequential data",
    "graph neural networks work on graph structured data",
    "machine learning research advances rapidly",
    "deep neural networks can approximate any function",
    "language understanding is key to chatbots",
    "object detection localizes items in scenes",
    "autonomous robots navigate complex environments",
]

POSITIVE_PAIRS = [
    ("machine learning", "deep learning uses neural networks"),
    ("computer vision", "object detection localizes items"),
    ("nlp", "language understanding is key to chatbots"),
    ("robotics", "autonomous robots navigate environments"),
    ("ai", "artificial intelligence research advances"),
]

NEGATIVE_PAIRS = [
    ("machine learning", "pizza is an italian dish"),
    ("robotics", "poetry expresses emotions"),
    ("vision", "chemistry studies molecules"),
]


def warm_start(tok, texts, n_rounds=3):
    for _ in range(n_rounds):
        for t in texts:
            tok.encode(t)
            tok.record_cooccurrence(tok.encode(t))


def contrastive_train(tok, pairs, lr=0.02, steps=30):
    for _ in range(steps):
        for query, positive in pairs:
            tok.contrastive_step(query, positive, [], lr=lr)


def evaluate(tok):
    def emb(text):
        return tok.embed(tok.encode(text))
    pos = [np.dot(emb(a), emb(b)) for a, b in POSITIVE_PAIRS]
    neg = [np.dot(emb(a), emb(b)) for a, b in NEGATIVE_PAIRS]
    margin = np.mean(pos) - np.mean(neg)
    return margin, np.mean(pos), np.mean(neg)


def benchmark(token_dim, adaptive_lr):
    tok = SOTokenizer(
        latent_dim=LATENT_DIM,
        token_dim=token_dim,
        tokenization_mode="word",
        max_vocab=2048,
        seed=42,
        adaptive_lr=adaptive_lr,
    )
    mem_tokens = len(tok.token_embeddings) * token_dim * 4 / (1024 ** 2)
    mem_proj = (token_dim * LATENT_DIM * 4) / (1024 ** 2)
    mem_total = mem_tokens + mem_proj

    warm_start(tok, CORPUS)
    margin_before, pos_b, neg_b = evaluate(tok)
    contrastive_train(tok, POSITIVE_PAIRS, lr=0.02, steps=30)
    margin_after, pos_a, neg_a = evaluate(tok)

    t0 = time.time()
    for _ in range(100):
        for t in CORPUS[:5]:
            tok.embed(tok.encode(t))
    t_embed = (time.time() - t0) / (100 * 5) * 1000

    return {
        "mem_mb": mem_total,
        "embed_ms": t_embed,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "pos_after": pos_a,
        "neg_after": neg_a,
    }


def main():
    dims = [64, 128, 256, 512, 768, 1024, 1536]
    print("token_dim | adaptive | mem_MB | embed_us | margin_before | margin_after | pos_after | neg_after")
    print("-" * 100)
    results = []
    for td in dims:
        for adaptive in [False, True]:
            r = benchmark(td, adaptive)
            results.append((td, adaptive, r))
            label = "yes" if adaptive else "no "
            print(
                f"{td:9d} | {label}    | {r['mem_mb']:6.2f} | {r['embed_ms']*1000:8.1f} | {r['margin_before']:13.3f} | {r['margin_after']:12.3f} | {r['pos_after']:9.3f} | {r['neg_after']:9.3f}")

    print("\n" + "=" * 100)
    print("ADAPTIVE LR IMPACT (delta margin vs fixed lr)")
    print("=" * 100)
    for td in dims:
        fixed = next(r for d, a, r in results if d == td and not a)
        adaptive = next(r for d, a, r in results if d == td and a)
        delta = adaptive["margin_after"] - fixed["margin_after"]
        print(
            f"  {td:4d}: fixed={fixed['margin_after']:.3f}  adaptive={adaptive['margin_after']:.3f}  delta={delta:+.3f}")


if __name__ == "__main__":
    main()
