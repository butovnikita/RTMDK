from rtmdk import RTMDKMemory
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def emb(t):
    return model.encode(t, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


mem = RTMDKMemory.import_field(r"C:\Users\Никита\.rtmdk\memory.json", emb)
print("Loaded nodes:", len(mem.field.nodes))

# Sort nodes by timestamp (earliest first)
nodes_list = list(mem.field.nodes.values())
nodes_list.sort(key=lambda n: n.content.get("timestamp", n.created_at))

print("\n=== 5 Earliest nodes ===")
for i, node in enumerate(nodes_list[:5]):
    ts = node.content.get("timestamp", node.created_at)
    title = node.content.get("title", "")
    text = node.content.get("text", "")[:300]
    source = node.content.get("source", "")
    print(f"\n#{i+1} [{source}] {title} (ts={ts})")
    print(text)
