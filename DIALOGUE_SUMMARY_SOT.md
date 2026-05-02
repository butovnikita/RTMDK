# Dialogue Summary: Self-Organizing Tokenizer + Embedding Field + Architecture Audit

## Session Date
2026-05-01

## Changes Implemented

### 1. Self-Organizing Tokenizer (SOT) v1 + v2
- **New module**: `rtmdk/memory/self_organizing_field.py`
  - `SOTokenizer`: byte-to-subtoken tokenizer with co-retrieval merge
  - `ContrastiveHebbian`: online contrastive learning (positive pull, negative push)
  - `EmbeddingFieldSSM`: SSM momentum for smooth latent trajectories
- **Integration**: `RTMDKField.step()`, `query()`, `add_node()` in `core.py`
- **New config flags**: `sot_enabled`, `sot_max_vocab`, `sot_contrastive_lr`, `sot_negatives_per_query`, `sot_ssm_sync`, `sot_merge_freq`, `sot_merge_threshold`, `sot_min_cooccurrence`, `sot_use_for_query`

### 2. Diagonal SSM (Performance Unlock)
- **Modified**: `rtmdk/engines/ssm_dynamics.py`
- Added `diagonal=True/False` mode
- Reduces complexity from **O(N·d²)** to **O(N·d)**
- Enables scaling `latent_dim` to 512+ without performance collapse

### 3. Multi-Resolution Embeddings (Future-Proof)
- `token_dim` separate from `latent_dim` in `SOTokenizer`
- Learnable projection matrix `W: token_dim → latent_dim`
- Projection trained via contrastive Hebbian rule
- Enables high-capacity tokens (256d) with lightweight field (64d)

### 4. Tests
- `tests/test_sot_tokenizer.py` — 24 tests
- `tests/test_sot_hebbian.py` — 19 tests
- `tests/test_sot_integration.py` — 14 tests
- **Total: 57 new tests, 104 total tests — 100% pass**

### 5. Documentation
- Updated `docs/08_ARCHITECTURE.md` with Phase 21 section
- Updated `docs/01_API_REFERENCE.md` with new config flags and `query_by_text()`

### 6. Dual-Repo Sync
- All changes synced to `rtmdk_github/RTMDK/`
- Both copies pass 104 tests

## Architecture Audit Findings

### Critical (P0)
1. `core.py` is 6889-line God File — needs decomposition
2. Massive duplication: inline classes in `core.py` vs `engines/`/`support/`
3. `HNSWIndex` is misleading (not real HNSW)
4. `_safe_run` swallows all exceptions silently

### Serious (P1)
5. `RTMDKConfig` — 150-field config blob anti-pattern
6. Duplicate `load_memory_variables` methods
7. `TritonBackend` misleading (no Triton code)
8. Test coverage critically low for `engines/` and `support/`

### Strengths
- SSM Dynamics — best in class
- Serialization (`serialization.py`) — excellent
- Security layer — robust
- Graceful degradation — well done
- State serialization pattern (`get_state`/`load_state`) — universal

## Recommended Refactoring Roadmap
1. **P0**: Extract `RTMDKConfig` from `core.py`
2. **P0**: Make `core.py` import from `engines/`/`support/` instead of inline copies
3. **P1**: Merge `load_memory_variables` duplications
4. **P1**: Replace `_safe_run` with circuit breakers
5. **P2**: Add unit tests for `support/` and `engines/`
6. **P3**: Hierarchical `RTMDKConfig`

## Next Steps
Begin P0 refactoring: extract config + deduplicate inline classes.
