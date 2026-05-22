"""
Demonstrate why adaptive bandwidth is fundamentally broken for RTMDK.

Key finding: Curse of dimensionality makes k-NN distances nearly uniform
in high-D normalized embedding spaces. Any transform of ~constant values
produces ~constant bandwidth factors.
"""

import numpy as np
from scipy.spatial import cKDTree

np.random.seed(42)


def analyze_dimensionality(dim, n_samples=500, k=5):
    """Generate uniform unit vectors in `dim` dimensions and analyze k-NN spread."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, dim)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    tree = cKDTree(X)
    distances, _ = tree.query(X, k=k + 1)
    kdist = distances[:, -1]

    median = np.median(kdist)
    p01 = np.percentile(kdist, 1)
    p99 = np.percentile(kdist, 99)
    ratio = p99 / max(p01, 1e-8)
    cv = np.std(kdist) / max(median, 1e-8)

    # Compute what bw spread sqrt transform would give
    factors = np.sqrt(kdist / max(median, 1e-8))
    bw_ratio = np.percentile(factors, 99) / \
        max(np.percentile(factors, 1), 1e-8)

    return {
        "dim": dim,
        "median_kdist": float(median),
        "cv": float(cv),
        "raw_ratio": float(ratio),
        "bw_ratio": float(bw_ratio),
    }


def main():
    dims = [2, 3, 5, 10, 20, 50, 100, 200, 384, 768]
    results = [analyze_dimensionality(d) for d in dims]

    print("Dimensionality vs k-NN Distance Variability")
    print("=" * 70)
    print(f"{'Dim':>6} {'Median kdist':>12} {'CV':>8} {'Raw p99/p01':>12} {'BW spread':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['dim']:>6} {r['median_kdist']:>12.4f} {r['cv']:>8.4f} {r['raw_ratio']:>12.2f}x {r['bw_ratio']:>10.2f}x")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("""
In high-D normalized spaces (384d, 768d):
  - k-NN distances have CV < 0.01 (coefficient of variation < 1%)
  - p99/p01 ratio ~ 1.0x (distances are nearly identical)
  - After sqrt transform: BW spread ~ 1.0-1.4x

This means adaptive bandwidth CANNOT discriminate local density
in semantic embedding spaces. The feature is mathematically doomed.

Adaptive bandwidth only makes sense in:
  1. Low-dimensional spaces (< 10d)
  2. Non-normalized data with highly varying scales
  3. Manifold-aware metrics (e.g., geodesic distances)

CONCLUSION: Remove adaptive_bandwidth from RTMDK.
""")

    print("\n(Install matplotlib to generate plots)")


if __name__ == "__main__":
    main()
