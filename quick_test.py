"""quick_test.py"""
from embedder_lmstudio import LMStudioEmbedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)

emb = LMStudioEmbedder()
print(f"Embedder available: {emb._available}")

config = RTMDKConfig(
    embedding_dim=768, latent_dim=256, top_k=5, min_response=0.001,
    decay_rate=0.999, enable_async=False, bm25_fallback=True,
    use_hnsw=True, learn_projection=True, projection_update_freq=300,
    phase_coupling=0.0, bandwidth=0.3,
)
print(f"Config: learn_projection={config.learn_projection}, bm25={config.bm25_fallback}")
mem = RTMDKMemory(config=config, embedder=emb)

facts = [
    ('Я пью чёрный кофе без сахара каждое утро в 8 часов', 'Что я пью по утрам?', 'кофе'),
    ('Мой любимый редактор кода — VS Code с темой Gruvbox', 'Какой редактор я использую?', 'VS Code'),
    ('В прошлом году я посетил Токио и был в восторге от суши в районе Синдзюку', 'Где я был в прошлом году?', 'Токио'),
    ('Я бегаю по 5 километров каждое субботнее утро в парке', 'Сколько я бегаю?', '5 километров'),
    ('Играю на акустической гитаре уже 10 лет, предпочитаю фингерстайл', 'На чём я играю?', 'гитаре'),
]
print(f"Storing {len(facts)} facts...")
for fact, query, kw in facts:
    mem.save_context({'input': fact, 'session_id': 't'}, {'output': fact})
    mem.save_context({'input': query, 'session_id': 't'}, {'output': fact})
    mem.save_context({'input': kw, 'session_id': 't'}, {'output': fact})

print(f"Nodes: {len(mem.field.nodes)}")
if mem.field.bm25_index:
    print(f"BM25 docs: {len(mem.field.bm25_index.documents)}")

# Check BM25 directly
if mem.field.bm25_index:
    r = mem.field.bm25_index.search('Что я пью по утрам?', top_k=3)
    print(f"BM25 direct search:")
    for doc_id, score in r:
        print(f"  {doc_id}: {mem.field.bm25_index.documents[doc_id][:80]}")

n_correct = 0
for fact, query, kw in facts:
    ctx = mem.load_memory_variables({'input': query, 'session_id': 't'})
    found = kw.lower() in ctx['rtmdk_context'].lower()
    if found:
        n_correct += 1
        print(f"  HIT: kw={kw}")
    else:
        print(f"  MISS: kw={kw} query={query[:40]}")
        print(f"    ctx: {ctx['rtmdk_context'][:100]}")

print(f"Recall: {n_correct}/{len(facts)} = {n_correct/len(facts):.0%}")
