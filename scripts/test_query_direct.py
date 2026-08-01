from rtmdk import RTMDKMemory
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
def emb(t):
    return model.encode(t, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

mem = RTMDKMemory.import_field(r"C:\Users\Никита\.rtmdk\memory.json", emb)
print("Loaded nodes:", len(mem.field.nodes))

for q in ["привет", "RTMDK", "первый коммит"]:
    emb_vec = emb(q)
    results = mem.retrieve_nodes(q, emb_vec, top_k=3)
    print(f"Query: {q}")
    for nid, score, node in results:
        text = node.content.get("text", "")[:200]
        print(f"  {nid}: score={score:.3f}, text={text}")
    print()
