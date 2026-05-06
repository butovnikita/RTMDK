"""
SOT token_dim vs latent_dim sweep.
Tests whether token_dim > latent_dim improves embedding quality.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtmdk.memory.self_organizing_field import SOTokenizer

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)

LATENT_DIM = 64
TEXTS = [
    "machine learning is amazing",
    "deep neural networks",
    "natural language processing",
    "computer vision and robotics",
    "artificial intelligence research",
    "machine learning research",
    "deep learning models",
    "language understanding",
    "robotics and vision",
    "ai and machine learning",
]

# Semantic pairs (should be close)
PAIRS_POS = [
    ("machine learning", "deep learning"),
    ("computer vision", "robotics"),
    ("nlp", "natural language processing"),
]
# Negative pairs (should be far)
PAIRS_NEG = [
    ("machine learning", "pizza"),
    ("robotics", "poetry"),
]


def evaluate(token_dim):
    tok = SOTokenizer(
        latent_dim=LATENT_DIM,
        token_dim=token_dim,
        tokenization_mode="word",
        max_vocab=512,
    )
    # Warm-up: encode all texts to build vocab
    for t in TEXTS:
        tok.encode(t)

    def emb(text):
        return tok.embed(tok.encode(text))

    pos_sims = []
    for a, b in PAIRS_POS:
        sim = np.dot(emb(a), emb(b))
        pos_sims.append(sim)

    neg_sims = []
    for a, b in PAIRS_NEG:
        sim = np.dot(emb(a), emb(b))
        neg_sims.append(sim)

    margin = np.mean(pos_sims) - np.mean(neg_sims)
    return margin, np.mean(pos_sims), np.mean(neg_sims)


print("token_dim | margin | avg_pos_sim | avg_neg_sim")
print("-" * 50)
for td in [16, 32, 64, 128, 256, 512]:
    margin, pos, neg = evaluate(td)
    print(f"{td:9d} | {margin:6.3f} | {pos:11.3f} | {neg:11.3f}")
