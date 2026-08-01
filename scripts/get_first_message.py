from rtmdk import RTMDKMemory
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
def emb(t):
    return model.encode(t, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

mem = RTMDKMemory.import_field(r"C:\Users\Никита\.rtmdk\memory.json", emb)
print(f"Total nodes in memory: {len(mem.field.nodes)}")

# Sort by timestamp
nodes_list = list(mem.field.nodes.values())
nodes_list.sort(key=lambda n: n.content.get("timestamp", n.created_at))

print(f"\nFirst node timestamp: {nodes_list[0].content.get('timestamp', nodes_list[0].created_at)}")
print(f"First node source: {nodes_list[0].content.get('source', '')}")
print(f"First node title: {nodes_list[0].content.get('title', '')}")
print(f"\n=== FIRST MESSAGE (full text) ===")
print(nodes_list[0].content.get("text", ""))

# Also dump first 3 nodes to file for inspection
with open(r"C:\Users\Никита\Desktop\llm_lab\first_messages.txt", "w", encoding="utf-8") as f:
    for i, node in enumerate(nodes_list[:3]):
        f.write(f"\n{'='*60}\n")
        f.write(f"Node #{i+1}\n")
        f.write(f"Source: {node.content.get('source', '')}\n")
        f.write(f"Title: {node.content.get('title', '')}\n")
        f.write(f"Timestamp: {node.content.get('timestamp', node.created_at)}\n")
        f.write(f"Tags: {node.content.get('tags', [])}\n")
        f.write(f"Text:\n{node.content.get('text', '')}\n")

print("\nSaved first 3 nodes to first_messages.txt")
