import json
from rtmdk.memory.sot_v2.integration import SOTv2Embedder, _word_tokenize, _build_vocab

with open("datasets/comprehensive_500.json", "r", encoding="utf-8") as f:
    records = json.load(f)["records"]

# Filter English
records = [r for r in records if r.get("language") == "en"]
corpus_texts = list({r["context"] for r in records})
queries = [r["query"] for r in records]

embedder = SOTv2Embedder(latent_dim=384, window_size=5, a=0.01, remove_pc=True)
embedder.train(corpus_texts)

vocab = embedder._vocab
print(f"Vocab size: {len(vocab)}")

oov_counts = []
for q in queries[:20]:
    tokens = _word_tokenize(q)
    in_vocab = [t for t in tokens if t in vocab]
    oov = [t for t in tokens if t not in vocab]
    oov_counts.append(len(oov) / len(tokens) if tokens else 0)
    print(f"Q: {q[:60]}")
    print(f"  tokens: {tokens}")
    print(f"  OOV: {oov}")
    print(f"  OOV ratio: {len(oov) / len(tokens):.2f}")

print(f"\nMean OOV ratio: {sum(oov_counts) / len(oov_counts):.2f}")
