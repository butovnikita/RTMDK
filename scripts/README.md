# RTMDK Scripts Directory

> All actively maintained scripts for benchmarking, testing, validation, and CI.
> Debug, diagnostic, evaluation, and research scripts have been archived to `archive/scripts/`.

## Quick Reference

| Category | Scripts | Purpose |
|----------|---------|---------|
| **Benchmarks** | `bench_*.py` | Performance measurement and comparison |
| **Chaos Tests** | `chaos_*.py` | Resilience and failure injection |
| **Checks** | `check_*.py` | Regression and correctness verification |
| **E2E Tests** | `e2e_*.py` | End-to-end smoke tests |
| **Load Tests** | `load_test_*.py` | Throughput and concurrency testing |
| **Stress Tests** | `stress_*.py` | Load and stability testing |
| **Utility** | `calc_*.py`, `find_*.py`, `measure_*.py`, `microbench_*.py`, `profile_*.py`, `run-ci.py`, `validate_*.py` | Helpers, profilers, and validators |

---

## Benchmarks (`bench_*.py`)

| Script | What it benchmarks | Typical invocation |
|--------|-------------------|-------------------|
| `bench_adaptive_ab.py` | Adaptive A/B testing strategies | `python scripts/bench_adaptive_ab.py` |
| `bench_adaptive_sbert.py` | SBERT-based adaptive retrieval | `python scripts/bench_adaptive_sbert.py` |
| `bench_backlog_modules.py` | Backlog module performance | `python scripts/bench_backlog_modules.py` |
| `bench_batch_ingestion.py` | Batch ingestion throughput | `python scripts/bench_batch_ingestion.py --nodes 100000` |
| `bench_beir_grid.py` | BEIR dataset grid search | `python scripts/bench_beir_grid.py` |
| `bench_beir_rtmdk.py` | RTMDK on BEIR datasets | `python scripts/bench_beir_rtmdk.py` |
| `bench_graphql_websocket.py` | GraphQL and WebSocket latency | `python scripts/bench_graphql_websocket.py` |
| `bench_meta_adaptive.py` | Meta-adaptive kernel tuning | `python scripts/bench_meta_adaptive.py` |
| `bench_meta_adaptive_challenging.py` | Meta-adaptive on hard queries | `python scripts/bench_meta_adaptive_challenging.py` |
| `bench_meta_adaptive_v2.py` | Meta-adaptive v2 improvements | `python scripts/bench_meta_adaptive_v2.py` |
| `bench_pipeline_ab.py` | Pipeline vs legacy A/B test | `python scripts/bench_pipeline_ab.py --queries 100 --nodes 500` |
| `bench_pipeline_production.py` | Production pipeline benchmark | `python scripts/bench_pipeline_production.py` |
| `bench_planner_production.py` | Planner production benchmark | `python scripts/bench_planner_production.py` |
| `bench_planner_savings.py` | Planner token savings | `python scripts/bench_planner_savings.py --dataset datasets/comprehensive_500.json --n 200` |
| `bench_procrustes.py` | Procrustes alignment quality | `python scripts/bench_procrustes.py` |
| `bench_quantum_resonance.py` | Quantum-inspired resonance | `python scripts/bench_quantum_resonance.py` |
| `bench_query_expansion.py` | Query expansion effectiveness | `python scripts/bench_query_expansion.py` |
| `bench_rag_vs_rtmdk.py` | RAG baseline comparison | `python scripts/bench_rag_vs_rtmdk.py` |
| `bench_rtmdk_vs_baselines.py` | RTMDK vs FAISS/BM25/Chroma | `python scripts/bench_rtmdk_vs_baselines.py` |
| `bench_self_contained.py` | Self-contained benchmark | `python scripts/bench_self_contained.py` |
| `bench_sot_correct.py` | SOT correctness verification | `python scripts/bench_sot_correct.py` |
| `bench_sot_v2.py` | SOT v2 tokenizer quality | `python scripts/bench_sot_v2.py` |
| `bench_sot_v2_in_rtmdk.py` | SOT v2 integrated in RTMDK | `python scripts/bench_sot_v2_in_rtmdk.py` |
| `bench_sot_v2_recall_k.py` | SOT v2 Recall@K metrics | `python scripts/bench_sot_v2_recall_k.py` |
| `bench_wal_throughput.py` | Write-Ahead Log throughput | `python scripts/bench_wal_throughput.py` |

---

## Chaos & Resilience (`chaos_*.py`)

| Script | What it tests | Typical invocation |
|--------|--------------|-------------------|
| `chaos_test_pipeline.py` | Pipeline failure injection | `python scripts/chaos_test_pipeline.py --mode all --seed 42` |

---

## Regression Checks (`check_*.py`)

| Script | What it checks | Typical invocation |
|--------|---------------|-------------------|
| `check_adaptive_a.py` | Adaptive A correctness | `python scripts/check_adaptive_a.py` |
| `check_regression.py` | Performance regression | `python scripts/check_regression.py --run-benchmark --baseline benchmarks/baseline.json` |

---

## End-to-End Tests (`e2e_*.py`)

| Script | What it tests | Typical invocation |
|--------|--------------|-------------------|
| `e2e_smoke_test.py` | Full system smoke test | `python scripts/e2e_smoke_test.py` |

---

## Load & Stress Tests (`load_test_*.py`, `stress_*.py`)

| Script | What it tests | Typical invocation |
|--------|--------------|-------------------|
| `load_test_pipeline.py` | Pipeline under load | `python scripts/load_test_pipeline.py` |
| `stress_test_100k.py` | 100K node stress test | `python scripts/stress_test_100k.py` |
| `stress_test_pipeline.py` | Pipeline stress test | `python scripts/stress_test_pipeline.py --nodes 5000 --queries 50 --planner --cost-tracking` |

---

## Utilities & Helpers

| Script | Purpose | Typical invocation |
|--------|---------|-------------------|
| `calc_capacity.py` | Calculate memory capacity estimates | `python scripts/calc_capacity.py` |
| `find_orphaned_flags.py` | Find unused config flags | `python scripts/find_orphaned_flags.py` |
| `measure_memory_10k.py` | Measure RAM at 10K nodes | `python scripts/measure_memory_10k.py` |
| `microbench_numba.py` | Numba microbenchmarks | `python scripts/microbench_numba.py` |
| `profile_field_query.py` | Profile field query hot path | `python scripts/profile_field_query.py` |
| `run-ci.py` | Local CI runner | `python scripts/run-ci.py` |
| `validate_config_matrix.py` | Validate all config presets | `python scripts/validate_config_matrix.py` |

---

## Archived Scripts

Debug, diagnostic, evaluation, and research scripts have been moved to `archive/scripts/`.
They remain in git history but are no longer actively maintained.

| Original Location | Archive Location | Reason |
|-------------------|-----------------|--------|
| `scripts/debug_*.py` | `archive/scripts/` | One-off debugging sessions |
| `scripts/diagnose_*.py` | `archive/scripts/` | Diagnostic utilities (used once) |
| `scripts/eval_*.py` | `archive/scripts/` | Evaluation reports (completed) |
| `scripts/research_*.py` | `archive/scripts/` | Research prototypes (superseded) |

To restore an archived script:
```bash
git mv archive/scripts/<script>.py scripts/
```

---

## Running Benchmarks

All benchmarks support `--help` for available options:
```bash
python scripts/bench_pipeline_ab.py --help
```

Common options:
- `--queries N` — number of queries to run
- `--nodes N` — number of nodes in the test field
- `--dataset PATH` — path to JSON dataset
- `--output PATH` — write results to JSON file

---

*Last updated: 2026-05-22*
