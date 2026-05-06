"""
Quantization recall vs dimension sweep.
Tests fp16 and int8 across 64d … 1536d.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)

N_NODES = 10_000
N_QUERIES = 200
TOP_K = 5


def build_field(dim, quantization="none"):
    cfg = RTMDKConfig(
        latent_dim=dim, top_k=TOP_K, min_response=0.001,
        decay_rate=0.999, use_hnsw=False, learn_projection=False,
        bm25_fallback=False, enable_async=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        quantization=quantization,
    )
    field = RTMDKField(cfg)
    positions = np.random.randn(N_NODES, dim).astype(np.float32)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    for i in range(N_NODES):
        field.add_node(positions[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
        field.nodes[f"n{i}"].amplitude = 1.0
        field.nodes[f"n{i}"].salience = 1.0
    return field


def recall_r1(field, queries):
    hits = 0
    for q in queries:
        r = field.query(q, top_k=1)
        if not r:
            continue
        # brute-force top1 for ground truth
        dots = np.array([np.dot(field.nodes[nid].latent_pos.astype(np.float32), q) for nid in field.node_index])
        true_top = field.node_index[int(np.argmax(dots))]
        if r[0][0] == true_top:
            hits += 1
    return hits / len(queries)


def estimate_emb_mem_mb(field):
    n = len(field.node_index)
    d = field.cfg.latent_dim
    itemsize = field._quant.itemsize
    # node embeddings + cached_positions
    return (n * d * itemsize * 2) / (1024 * 1024)


def main():
    dims = [64, 128, 256, 512, 768, 1024, 1536]
    print("Dim   | fp16-R@1 | fp16-MB | int8g-R@1 | int8g-MB | int8pd-R@1 | int8pd-MB | base-MB")
    print("-" * 95)

    for dim in dims:
        queries = np.random.randn(N_QUERIES, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        base = build_field(dim, "none")
        base_mem = estimate_emb_mem_mb(base)

        fp16 = build_field(dim, "fp16")
        fp16_r1 = recall_r1(fp16, queries)
        fp16_mem = estimate_emb_mem_mb(fp16)

        # manual int8 (positions passed pre-quantized; field stores fp32 because mode=none)
        positions = np.random.randn(N_NODES, dim).astype(np.float32)
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        # int8 global
        scale = 1.0 / 127.0
        pos_ig = np.round(positions / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
        # int8 per-dim
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        scales = (maxs - mins) / 255.0
        scales = np.maximum(scales, 1e-8)
        pos_ipd = np.round((positions - mins) / scales).clip(0, 255).astype(np.uint8).astype(np.float32) * scales + mins

        def field_from_pos(p):
            cfg = RTMDKConfig(latent_dim=dim, top_k=TOP_K, min_response=0.001,
                              decay_rate=0.999, use_hnsw=False, learn_projection=False,
                              bm25_fallback=False, enable_async=False,
                              resonance_kernel="cosine", phase_coupling=0.0)
            f = RTMDKField(cfg)
            for i in range(N_NODES):
                f.add_node(p[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
                f.nodes[f"n{i}"].amplitude = 1.0
                f.nodes[f"n{i}"].salience = 1.0
            return f

        f_ig = field_from_pos(pos_ig)
        ig_r1 = recall_r1(f_ig, queries)

        f_ipd = field_from_pos(pos_ipd)
        ipd_r1 = recall_r1(f_ipd, queries)

        print(f"{dim:4d}  | {fp16_r1:.4f}   | {fp16_mem:6.1f}  | {ig_r1:.4f}    | {base_mem:6.1f}   | {ipd_r1:.4f}     | {base_mem:6.1f}    | {base_mem:6.1f}")


if __name__ == "__main__":
    main()
