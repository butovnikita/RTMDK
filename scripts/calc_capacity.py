"""RTMDK capacity calculator."""


def calc_capacity(ram_gb, dim, bytes_per_float, avg_tokens_per_doc, avg_doc_chars):
    ram_bytes = ram_gb * 1024**3
    emb_per_node = dim * bytes_per_float
    overhead = 500
    text_per_node = avg_doc_chars * 2
    total_per_node = emb_per_node + overhead + text_per_node
    max_nodes = int(ram_bytes / total_per_node)
    max_tokens = max_nodes * avg_tokens_per_doc
    return max_nodes, max_tokens, total_per_node


print("RTMDK Corpus Capacity")
print("=" * 80)
print(f"{'Config':<25} {'RAM':>8} {'Nodes':>12} {'Tokens':>15} {'Per node':>10}")
print("-" * 80)

configs = [
    ("Exact fp32", 4, 256, 100, 400),
    ("HNSW fp32", 4, 256, 100, 400),
    ("SOT fp32", 4, 256, 100, 400),
    ("Quant int8", 1, 256, 100, 400),
    ("Quant fp16", 2, 256, 100, 400),
]

for name, bpe, dim, tpd, chars in configs:
    for ram in [4, 16, 64]:
        nodes, tokens, pnode = calc_capacity(ram, dim, bpe, tpd, chars)
        print(f"{name:<25} {ram:>6}GB {nodes:>12,} {tokens:>15,} {pnode/1024:>9.1f}KB")
    print()

print("-" * 80)
print("Tiered Storage (1TB SSD)")
nodes = int(1e12 / 1024)
tokens = nodes * 100
print(f"{'Tiered SSD':<25} {'1 TB':>8} {nodes:>12,} {tokens:>15,} {'~1 KB':>10}")
