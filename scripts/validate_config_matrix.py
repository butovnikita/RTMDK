"""
Config validation matrix: test key parameter combinations.

Usage:
    python scripts/validate_config_matrix.py
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField


def test_case(name: str, **kwargs):
    """Test that a config combination initializes without error."""
    try:
        cfg = RTMDKConfig(**kwargs)
        field = RTMDKField(cfg)
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


def main():
    print("RTMDK Config Validation Matrix")
    print("=" * 60)

    results = []

    # Core variants
    results.append(test_case("Default config"))
    results.append(test_case("High dim", latent_dim=256))
    results.append(test_case("Low dim", latent_dim=16))

    # SOT variants
    results.append(test_case("SOT byte default", sot_enabled=True))
    results.append(test_case("SOT word mode", sot_enabled=True, sot_tokenization_mode="word"))
    results.append(test_case("SOT word + warm-start", sot_enabled=True, sot_tokenization_mode="word", sot_warm_start_corpus=None))
    results.append(test_case("SOT word + cooccurrence limit", sot_enabled=True, sot_tokenization_mode="word", sot_max_cooccurrence=100))
    results.append(test_case("SOT byte + subword seed", sot_enabled=True, sot_subword_seed=True))
    results.append(test_case("SOT byte + attention", sot_enabled=True, sot_attention_pooling=True))
    results.append(test_case("SOT byte + skipgram", sot_enabled=True, sot_skipgram_window=5))
    results.append(test_case("SOT byte + hard negatives", sot_enabled=True, sot_hard_negatives=True))
    results.append(test_case("SOT byte + retrieval feedback", sot_enabled=True, sot_retrieval_feedback=True))

    # Mathematical features
    results.append(test_case("Hyperbolic", hyperbolic=True))
    results.append(test_case("Conformal prediction", conformal_prediction=True))
    results.append(test_case("Local bandwidth", adaptive_bandwidth=True))
    results.append(test_case("Spectral consolidation", spectral_consolidation=True))
    results.append(test_case("Kalman filter", enable_kalman_filter=True))

    # Production features
    results.append(test_case("Engrams", enable_engrams=True))
    results.append(test_case("Security", security_enabled=True))
    results.append(test_case("Version control", version_control=True))

    # Combined stress test
    results.append(test_case(
        "Full stress test",
        latent_dim=64,
        sot_enabled=True,
        sot_tokenization_mode="word",
        sot_subword_seed=True,
        sot_attention_pooling=True,
        sot_hard_negatives=True,
        sot_retrieval_feedback=True,
        sot_skipgram_window=3,
        sot_max_cooccurrence=50,
        hyperbolic=True,
        conformal_prediction=True,
        adaptive_bandwidth=True,
        spectral_consolidation=True,
        enable_kalman_filter=True,
        enable_engrams=True,
        security_enabled=True,
    ))

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("All configs validated successfully.")
        return 0
    else:
        print(f"{total - passed} config(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
