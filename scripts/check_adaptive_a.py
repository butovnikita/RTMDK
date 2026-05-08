"""Check what adaptive a was chosen."""
import json
from rtmdk.memory.sot_v2.integration import SOTv2Embedder

records = json.load(open("datasets/comprehensive_500.json", encoding="utf-8"))["records"]
corpus = list({r["context"] for r in records if r.get("language") == "en"})
sot = SOTv2Embedder(latent_dim=384, a=0.01, window_size=5)
sot.train(corpus)
print(f"Adaptive a = {sot._embedder.a}")
print(f"Vocab size = {len(sot._vocab)}")
print(f"Corpus size = {len(corpus)}")
probs = sorted(sot._embedder.word_probs.values())
print(f"Word prob percentiles: p10={probs[len(probs)//10]:.6f}, p50={probs[len(probs)//2]:.6f}, p90={probs[int(len(probs)*0.9)]:.6f}")
