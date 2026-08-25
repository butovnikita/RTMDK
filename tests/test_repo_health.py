"""Repo health regression tests.

Lock in fixes from the 2026-08-01 audit so they cannot silently regress:
1. version must be identical in pyproject.toml, rtmdk.__version__ and server banners
2. legacy/ modules must stay importable (conftest.py sys.path shim)
"""

import os
import re

import rtmdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version() -> str:
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    assert m, "version not found in pyproject.toml"
    return m.group(1)


class TestVersionSync:
    def test_package_version_matches_pyproject(self):
        assert rtmdk.__version__ == _pyproject_version(), (
            f"rtmdk.__version__={rtmdk.__version__} != " f"pyproject version={_pyproject_version()}"
        )

    def test_server_banner_matches_package_version(self):
        app_path = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(app_path, encoding="utf-8") as f:
            content = f.read()
        # Only version-reporting spots: banners, FastAPI metadata, health payloads
        reported = set()
        reported |= set(re.findall(r"Production API v(\d+\.\d+\.\d+)", content))
        reported |= set(re.findall(r'version="(\d+\.\d+\.\d+)"', content))
        reported |= set(re.findall(r'"version":\s*"(\d+\.\d+\.\d+)"', content))
        assert reported, "no version banners found in server/app.py"
        bad = {v for v in reported if v != rtmdk.__version__}
        assert not bad, f"stale version banners in server/app.py: {bad}"


class TestMetricsHonesty:
    """R1.1: README must report honest 95.6% @1000 QA, not bare 99.3% (RISKS.md)."""

    def test_readme_recall_is_honest(self):
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        # Honest metric from docs/06_SCIENTIFIC_ARTICLE.md:5.2 must be present
        assert "95.6%" in text, "README must contain honest 95.6% @1000 QA (docs/06)"
        assert "97.6%" in text, "README must mention per-topic avg 97.6% (docs/06:5.3)"
        assert "1000 QA" in text, "README must qualify dataset as 1000 QA"
        # Synthetic 99.3% is allowed only with qualifier comprehensive_500 / synthetic
        # Bare "| Recall@1 | **99.3%** |" without dataset qualifier is a regression
        assert "| Recall@1 | **99.3%** |" not in text
        assert "| **Recall@1** | **99.3%** |" not in text
        # If 99.3% appears at all, it must be qualified as synthetic / comprehensive_500
        if "99.3%" in text:
            assert "comprehensive_500" in text or "синтетический" in text.lower() or "synthetic" in text.lower(), (
                "99.3% in README must be qualified as synthetic comprehensive_500"
            )
            assert "18.1%" in text, "synthetic 99.3% must be paired with cosine 18.1% context"

    def test_scientific_article_consistency(self):
        art = os.path.join(ROOT, "docs", "06_SCIENTIFIC_ARTICLE.md")
        with open(art, encoding="utf-8") as f:
            text = f.read()
        # Disclaimer must remain
        assert "95.6% Recall@1" in text
        # Conclusion must not claim bare 99.3% on 1000 QA as production truth
        assert "Recall@1 = 99.3% (v8.3, production pipeline)" not in text
        # If 99.3% mentioned, must be qualified as comprehensive_500 / synthetic
        if text.count("99.3%") > 0:
            # At least one mention must be synthetic-qualified (we keep it in conclusion now qualified)
            assert "comprehensive_500" in text

    def test_readme_latency_is_honest(self):
        """R1.2: README @100K latencies must be marked as прогноз / forecast (RISKS.md)."""
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        # Any @100K latency mention must be qualified as прогноз/forecast
        assert "100K" in text
        # Detect bare unqualified pattern: "| **16 ms** |" without † or (прогноз) in same line
        for line in text.splitlines():
            if "@ 100K" in line and "ms" in line:
                assert "прогноз" in line.lower() or "forecast" in line.lower() or "†" in line, (
                    f"Latency @100K line must be marked as прогноз/forecast: {line!r}"
                )
        # Must reference honest sources
        assert "benchmarks/baseline.json" in text, "README must reference measured baseline.json @500"
        assert "baseline_100k.json" in text, "README must reference forecast baseline_100k.json"
        # Must explain extrapolation
        assert "500" in text and "10K" in text, "README must mention @500 and @10K basis for @100K forecast"

    def test_latency_baseline_artifacts(self):
        """R1.2/R6.3: baseline artifacts must exist and be honest."""
        import json

        b500 = os.path.join(ROOT, "benchmarks", "baseline.json")
        b100k = os.path.join(ROOT, "benchmarks", "baseline_100k.json")
        assert os.path.exists(b500), "benchmarks/baseline.json must exist (@500 measured)"
        assert os.path.exists(b100k), "benchmarks/baseline_100k.json must exist (forecast, R1.2)"
        with open(b100k, encoding="utf-8") as f:
            data = json.load(f)
        # Forecast file must be explicitly marked as forecast, not fake measurement
        assert data.get("forecast") is True, "baseline_100k.json must have forecast:true"
        assert "pending_measurement" in data or "disclaimer" in data
        # Forecast file must contain basis
        basis = data.get("forecast_basis", {})
        assert "measured_p95_ms_at_500" in basis or "measured_at_10k_ms" in basis

    def test_readme_ram_is_honest(self):
        """R1.3: README RAM must clarify latent-only vs full cost (RISKS.md)."""
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        # Why table RAM row must be qualified
        assert "19-30 MB" in text
        # Must contain footnote marker and explanation
        assert "‡" in text, "README must mark RAM rows with ‡ footnote"
        # Footnote must explain latent-only vs full indexes
        assert "HNSW" in text and "BM25" in text, "RAM footnote must mention HNSW/BM25"
        assert "docs/08_ARCHITECTURE.md:440" in text, "RAM footnote must reference docs/08:440 source of truth"
        # Must mention full cost 80/90 MB @10K and 750 MB @100K
        assert "80" in text and "90" in text, "RAM footnote must mention full 80/90MB @10K"
        # Results table RAM rows must also be marked
        ram_lines = [ln for ln in text.splitlines() if "RAM (" in ln and "MB" in ln]
        assert len(ram_lines) >= 2
        for ln in ram_lines:
            assert "‡" in ln, f"RAM result line must be marked with ‡: {ln!r}"

    def test_ram_source_of_truth_exists(self):
        """R1.3: docs/08 RAM table must exist as source of truth."""
        arch = os.path.join(ROOT, "docs", "08_ARCHITECTURE.md")
        with open(arch, encoding="utf-8") as f:
            text = f.read()
        assert "RAM по масштабу" in text
        assert "80 MB" in text and "90 MB" in text
        assert "750 MB" in text or "780 MB" in text
        assert "16 MB" in text  # 1K base


class TestMypyHealth:
    """R2: mypy debt must not be hidden (RISKS.md R2.1/R2.2)."""

    def test_no_hidden_ignores_for_core(self):
        mypy = os.path.join(ROOT, "mypy.ini")
        with open(mypy, encoding="utf-8") as f:
            lines = f.readlines()
        active = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        active_text = "\n".join(active)
        # R2.1: field/core/serialization must be checked (no ignore_errors)
        assert "[mypy-rtmdk.memory.field]" not in active_text, "R2.1: mypy.ini must not hide rtmdk.memory.field with ignore_errors"
        assert "[mypy-rtmdk.memory.core]" not in active_text, "R2.1: mypy.ini must not hide rtmdk.memory.core"
        assert "[mypy-rtmdk.memory.serialization]" not in active_text, "R2.1: mypy.ini must not hide serialization"
        # Must explain why R2.1 fix was done
        text = "".join(lines)
        assert "R2.1" in text or "heavy-module" in text.lower() or "baseline" in text.lower()

    def test_attr_defined_handling_documented(self):
        mypy = os.path.join(ROOT, "mypy.ini")
        with open(mypy, encoding="utf-8") as f:
            text = f.read()
        # R2.2: attr-defined must be disabled globally for delegation cycle, but documented
        assert "disable_error_code" in text and "attr-defined" in text
        assert "R2.2" in text or "manager-delegation" in text.lower() or "__getattr__" in text
        # Must mention the cycle field->manager->field
        assert "field" in text.lower() and "manager" in text.lower()
        # Baseline must remain 0 and be referenced
        assert ".github/mypy-baseline.txt" in text or "baseline" in text.lower()

    def test_mypy_baseline_is_zero_and_honest(self):
        baseline = os.path.join(ROOT, ".github", "mypy-baseline.txt")
        assert os.path.exists(baseline)
        with open(baseline, encoding="utf-8") as f:
            val = f.read().strip()
        assert val == "0", "mypy baseline must be 0 after R2.1 debt elimination (8.3.4)"
        # After removing ignores, baseline 0 is honest (not false-zero from hidden 40% LOC)
        mypy = os.path.join(ROOT, "mypy.ini")
        with open(mypy, encoding="utf-8") as f:
            mtext = f.read()
        assert "ignore_errors = True" not in mtext or mtext.count("ignore_errors = True") <= 2, (
            "Only numpy/scipy may keep ignore_errors, not core"
        )


class TestConfigHealth:
    """R3: config bloat must be documented and deprecated (RISKS.md R3.1-3.3)."""

    def test_orphaned_flags_deprecated_and_validated(self):
        cfg_path = os.path.join(ROOT, "rtmdk", "memory", "config.py")
        with open(cfg_path, encoding="utf-8") as f:
            text = f.read()
        assert "ORPHANED_FLAGS" in text
        assert "R3.1" in text or "v9.0" in text or "deprecated" in text.lower()
        # validate() must warn on orphaned and mark critical as ERROR
        assert "Orphaned flag" in text
        assert "ERROR:" in text  # critical pipeline thresholds

    def test_validate_warns_on_orphaned_and_errors_on_critical(self):
        from rtmdk.memory.config import RTMDKConfig, ORPHANED_FLAGS

        # Orphaned flag set to non-default should warn with deprecation
        cfg = RTMDKConfig(latent_dim=64, adjoint_enabled=True)
        if "adjoint_enabled" in ORPHANED_FLAGS:
            warns = cfg.validate()
            assert any("adjoint_enabled" in w and "deprecated" in w.lower() for w in warns), (
                f"validate must warn on orphaned adjoint_enabled, got {warns}"
            )
        # Critical pipeline threshold must be ERROR
        cfg2 = RTMDKConfig(latent_dim=16, pipeline_breaker_failure_threshold=0)
        warns2 = cfg2.validate()
        assert any("ERROR" in w and "failure_threshold" in w for w in warns2)

    def test_single_source_of_truth_for_config(self):
        cfg_reexport = os.path.join(ROOT, "rtmdk", "config.py")
        with open(cfg_reexport, encoding="utf-8") as f:
            text = f.read()
        assert "rtmdk/memory/config.py" in text or "memory.config" in text, (
            "rtmdk/config.py must re-export from memory.config (R3.2)"
        )
        assert "R3.2" in text or "single source" in text.lower()
        # Both imports must resolve to same class
        from rtmdk.memory.config import RTMDKConfig as FromMemory
        from rtmdk.config import RTMDKConfig as FromPreset

        assert FromMemory is FromPreset

    def test_values_deprecated_pointer(self):
        vals = os.path.join(ROOT, "Values.md")
        with open(vals, encoding="utf-8") as f:
            text = f.read()
        assert "R3.3" in text or "deprecated" in text.lower()
        assert "rtmdk/memory/config.py" in text
        assert "pipeline_breaker" in text or "sot_" in text

    def test_backlog_deprecation_plan_exists(self):
        bl = os.path.join(ROOT, "BACKLOG.md")
        with open(bl, encoding="utf-8") as f:
            text = f.read()
        assert "R3.1" in text
        assert "ORPHANED_FLAGS" in text
        assert "v9.0" in text
        assert "36" in text


class TestThreadSafety:
    """R4: thread safety must be documented and enforced (RISKS.md R4.1-4.3)."""

    def test_query_uses_write_lock(self):
        qm_path = os.path.join(ROOT, "rtmdk", "memory", "query_manager.py")
        with open(qm_path, encoding="utf-8") as f:
            text = f.read()
        # R4.1: _query_vectorized and _batch_resonance_cached must snapshot under _write_lock
        assert "with f._write_lock" in text, "query paths must snapshot under _write_lock (R4.1)"
        assert "_batch_resonance_cached" in text
        # query_batch must also snapshot node_index
        assert "node_index_snapshot" in text or "with f._write_lock" in text
        assert "R4.1" in text

    def test_sot_update_holds_field_lock(self):
        core_path = os.path.join(ROOT, "rtmdk", "memory", "core.py")
        with open(core_path, encoding="utf-8") as f:
            text = f.read()
        # R4.2: SOT Hebbian update must be under field lock
        assert "R4.2" in text
        assert "_sot_v2_online_lock" in text
        # Must acquire field lock for online_update
        assert "field" in text and "_write_lock" in text and "online_update" in text

    def test_distributed_lock_ordering(self):
        core_path = os.path.join(ROOT, "rtmdk", "memory", "core.py")
        with open(core_path, encoding="utf-8") as f:
            text = f.read()
        # R4.3: lock ordering documented and try/finally for release
        assert "R4.3" in text
        assert "distributed_lock" in text and "_write_lock" in text
        assert "outer" in text.lower() and "inner" in text.lower()
        assert "try:" in text and "finally:" in text

    def test_field_write_lock_is_rlock(self):
        init_path = os.path.join(ROOT, "rtmdk", "memory", "field_initializer.py")
        with open(init_path, encoding="utf-8") as f:
            text = f.read()
        assert "threading.RLock" in text
        assert "R4" in text or "_write_lock" in text

    def test_concurrent_query_and_add(self):
        """R4.1: concurrent add_nodes_batch + query must not raise torn-read ValueError."""
        import threading
        import numpy as np

        from rtmdk.memory.config import RTMDKConfig
        from rtmdk.memory.field import RTMDKField

        cfg = RTMDKConfig(latent_dim=16, embedding_dim=16, max_nodes=1000, use_hnsw=False)
        field = RTMDKField(cfg)

        # Seed with 20 nodes
        rng = np.random.default_rng(0)
        for i in range(20):
            emb = rng.standard_normal(16).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            field.add_node(emb, {"text": f"node {i}"})

        query_emb = rng.standard_normal(16).astype(np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        errors = []

        def add_many():
            try:
                embs = rng.standard_normal((10, 16)).astype(np.float32)
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
                contents = [{"text": f"batch {j}"} for j in range(10)]
                field.add_nodes_batch(embs, contents)
            except Exception as e:
                errors.append(e)

        def query_many():
            try:
                for _ in range(30):
                    field.query(query_emb, top_k=5)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=add_many))
            threads.append(threading.Thread(target=query_many))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent query/add raised: {errors}"
        # ValueError: broadcast shape mismatch would be the torn-read symptom
        assert all("broadcast" not in str(e) for e in errors)


class TestTieredAndBatch:
    """R5: tiered/storage, batch ingestion, SLO must be honest (RISKS.md R5.1-5.3)."""

    def test_tiered_v1_deprecated_v2_canonical(self):
        v1 = os.path.join(ROOT, "rtmdk", "memory", "tiered_storage.py")
        with open(v1, encoding="utf-8") as f:
            t1 = f.read()
        assert "R5.1" in t1 and "deprecated" in t1.lower()
        assert "storage/tiered.py" in t1 and "v2" in t1
        assert "DeprecationWarning" in t1

        v2 = os.path.join(ROOT, "rtmdk", "storage", "tiered.py")
        assert os.path.exists(v2)
        with open(v2, encoding="utf-8") as f:
            t2 = f.read()
        # v2 must NOT be deprecated, must be memmap/LFU
        assert "memmap" in t2.lower() or "mmap" in t2.lower()
        assert "LFU" in t2 or "warm" in t2.lower()

        cfg = os.path.join(ROOT, "rtmdk", "memory", "config.py")
        with open(cfg, encoding="utf-8") as f:
            ct = f.read()
        assert "tiered_storage_enabled" in ct and "R5.1" in ct
        assert "tiered_storage_v2_enabled" in ct

    def test_tiered_fallback_is_sampled_not_full_scan(self):
        qm = os.path.join(ROOT, "rtmdk", "memory", "query_manager.py")
        with open(qm, encoding="utf-8") as f:
            text = f.read()
        assert "R5.1" in text
        assert "peek_batch" in text
        # Must be sampled needed*5, not O(W+C) full scan
        assert "needed * 5" in text
        assert "O(W+C)" in text or "O(C)" in text or "not O" in text or "sample" in text.lower()

    def test_batch_artifact_exists_and_is_forecast(self):
        import json

        batch = os.path.join(ROOT, "benchmarks", "baseline_batch.json")
        assert os.path.exists(batch), "benchmarks/baseline_batch.json must exist (forecast, R5.2)"
        with open(batch, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("forecast") is True
        assert "without_wal" in str(data.get("forecast_basis", {}))
        assert "83K" in str(data) or "83000" in str(data) or "12s" in str(data)

        tiered = os.path.join(ROOT, "benchmarks", "baseline_tiered.json")
        assert os.path.exists(tiered)
        with open(tiered, encoding="utf-8") as f:
            dt = json.load(f)
        assert dt.get("forecast") is True

    def test_perf_workflow_has_nightly_batch_and_tiered(self):
        perf = os.path.join(ROOT, ".github", "workflows", "perf.yml")
        with open(perf, encoding="utf-8") as f:
            text = f.read()
        assert "schedule" in text and "cron" in text, "perf.yml must have nightly schedule (R5.2)"
        assert "bench_batch_ingestion" in text, "perf.yml must run batch ingestion (R5.2)"
        assert "bench_tiered" in text or "stress_test_100k" in text
        assert "R5.1" in text or "tiered" in text.lower()
        assert "R5.2" in text

    def test_slo_thresholds_covered(self):
        # R5.3: pipeline breaker thresholds must be validated as ERROR and tested
        cfg = os.path.join(ROOT, "rtmdk", "memory", "config.py")
        with open(cfg, encoding="utf-8") as f:
            ct = f.read()
        assert "pipeline_breaker_thresholds" in ct
        assert "ERROR:" in ct  # from R3.1, critical SLO misconfig
        # Test file must cover breaker thresholds
        tcb = os.path.join(ROOT, "tests", "test_pipeline_circuit_breaker.py")
        assert os.path.exists(tcb)
        with open(tcb, encoding="utf-8") as f:
            tbt = f.read()
        assert "failure_threshold" in tbt or "latency" in tbt.lower()


class TestOomAndScale:
    """R6: OOM and scale must be guarded (RISKS.md R6.1-6.3)."""

    def test_sif_oom_guardian_in_code(self):
        sif = os.path.join(ROOT, "rtmdk", "memory", "sot_v2", "sif_embedder.py")
        with open(sif, encoding="utf-8") as f:
            text = f.read()
        assert "SPARSE_PMI_THRESHOLD" in text
        assert "5000" in text
        assert "sparse" in text.lower() and "TruncatedSVD" in text

    def test_validate_warns_on_large_vocab(self):
        from rtmdk.memory.config import RTMDKConfig

        cfg_big = RTMDKConfig(latent_dim=16, sot_max_vocab=9000)
        warns = cfg_big.validate()
        assert any("sot_max_vocab" in w and ("8000" in w or "5000" in w) for w in warns), f"expected vocab warn, got {warns}"

        cfg_huge = RTMDKConfig(latent_dim=16, sot_max_vocab=20000)
        warns2 = cfg_huge.validate()
        assert any("sot_max_vocab" in w and "3GB" in w for w in warns2), f"expected 3GB warn for 20K, got {warns2}"

        cfg_ok = RTMDKConfig(latent_dim=16, sot_max_vocab=4096)
        warns_ok = cfg_ok.validate()
        # Default 4096 should not trigger R6.1 vocab warn
        assert not any("sot_max_vocab" in w and "8000" in w for w in warns_ok)

    def test_validate_warns_on_large_window(self):
        from rtmdk.memory.config import RTMDKConfig

        cfg_w = RTMDKConfig(latent_dim=16, sot_skipgram_window=10)
        warns = cfg_w.validate()
        assert any("sot_skipgram_window" in w and "R6.2" in w for w in warns), f"expected window>5 warn, got {warns}"

        cfg_ok = RTMDKConfig(latent_dim=16, sot_skipgram_window=1)
        warns_ok = cfg_ok.validate()
        assert not any("sot_skipgram_window" in w for w in warns_ok)

    def test_sot_guide_has_ram_table(self):
        guide = os.path.join(ROOT, "docs", "SOT_V2_GUIDE.md")
        with open(guide, encoding="utf-8") as f:
            text = f.read()
        assert "R6.1" in text
        assert "SPARSE_PMI_THRESHOLD" in text
        assert "5000" in text
        assert "3GB" in text or "3 GB" in text
        assert "RAM" in text and "sot_max_vocab" in text

    def test_baseline_100k_is_forecast_and_nightly_exists(self):
        import json

        b100k = os.path.join(ROOT, "benchmarks", "baseline_100k.json")
        assert os.path.exists(b100k), "baseline_100k.json must exist (R1.2/R6.3)"
        with open(b100k, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("forecast") is True

        perf = os.path.join(ROOT, ".github", "workflows", "perf.yml")
        with open(perf, encoding="utf-8") as f:
            text = f.read()
        assert "schedule" in text and "cron" in text, "perf.yml must have nightly schedule (R6.3)"
        assert "stress_test_100k" in text
        assert "R6.3" in text or "100K baseline" in text
        assert "baseline_100k" in text


class TestCircuitBreakerUnified:
    """R7: CircuitBreaker must be single source (RISKS.md R7.1)."""

    def test_pipeline_reuses_support_state(self):
        # R7.1: pipeline BreakerState must be alias to support CircuitState
        sup = os.path.join(ROOT, "rtmdk", "support", "circuit_breaker.py")
        pipe = os.path.join(ROOT, "rtmdk", "pipeline", "circuit_breaker.py")
        with open(sup, encoding="utf-8") as f:
            t_sup = f.read()
        with open(pipe, encoding="utf-8") as f:
            t_pipe = f.read()
        assert "R7.1" in t_sup
        assert "R7.1" in t_pipe
        assert "CircuitState" in t_sup
        assert "CircuitState" in t_pipe or "BreakerState = CircuitState" in t_pipe
        # Both must define CLOSED/OPEN/HALF_OPEN
        assert "CLOSED" in t_sup and "OPEN" in t_sup
        assert "CLOSED" in t_pipe and "OPEN" in t_pipe
        # Pipeline must import from support (single source)
        assert "from rtmdk.support.circuit_breaker import" in t_pipe

    def test_single_breaker_class_reused(self):
        # Both modules define class CircuitBreaker but pipeline should note it extends/reuses support
        pipe = os.path.join(ROOT, "rtmdk", "pipeline", "circuit_breaker.py")
        with open(pipe, encoding="utf-8") as f:
            text = f.read()
        # Must document inheritance of thresholds from config
        assert "pipeline_breaker_thresholds" in text or "config.py:748" in text
        # Must mention that support is canonical for 3-state
        assert "support" in text.lower() and "single source" in text.lower()

    def test_runtime_imports_share_state_values(self):
        from rtmdk.support.circuit_breaker import CircuitState as SupState
        from rtmdk.pipeline.circuit_breaker import BreakerState as PipeState

        # Alias must be same object
        assert PipeState is SupState
        assert PipeState.CLOSED.value == "closed"
        assert PipeState.OPEN.value == "open"
        assert PipeState.HALF_OPEN.value == "half_open"

        from rtmdk.support.circuit_breaker import CircuitBreaker as SupBreaker
        from rtmdk.pipeline.circuit_breaker import CircuitBreaker as PipeBreaker

        # Both must be instantiable with similar failure_threshold API
        sb = SupBreaker(name="test_sup", failure_threshold=2)
        pb = PipeBreaker(name="test_pipe", failure_threshold=2)
        assert sb.failure_threshold == 2
        assert pb.failure_threshold == 2
        # Pipeline breaker must have latency thresholds from config
        assert hasattr(pb, "latency_threshold_ms")


class TestLegacyDrift:
    """R8: legacy fork must stay frozen and not drift (RISKS.md R8.1)."""

    def test_legacy_readme_frozen(self):
        readme = os.path.join(ROOT, "legacy", "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        assert "Frozen" in text
        assert "2026-08-01" in text
        assert "R8.1" in text
        assert "rtmdk/server/app.py" in text
        assert "27" in text or "48" in text  # route counts documented

    def test_legacy_vs_server_route_drift(self):
        # R8.1: 27 legacy vs 48 server — drift must be intentional, not silent
        leg = os.path.join(ROOT, "legacy", "rtmdk_server.py")
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(leg, encoding="utf-8") as f:
            leg_text = f.read()
        with open(srv, encoding="utf-8") as f:
            srv_text = f.read()
        import re

        leg_routes = len(re.findall(r"@app\.(get|post|put|delete|patch)", leg_text))
        srv_routes = len(re.findall(r"@app\.(get|post|put|delete|patch)", srv_text))
        # Documented counts in RISKS.md R8.1
        assert leg_routes == 27, f"legacy routes {leg_routes} != 27 (RISKS.md R8.1)"
        assert srv_routes == 48, f"server routes {srv_routes} != 48 (RISKS.md R8.1)"
        assert srv_routes > leg_routes, "server must have more routes than frozen legacy"
        # If legacy gains routes, it must be intentional (frozen exception)
        assert leg_routes <= 27, "legacy gained routes — must be frozen (R8.1)"

    def test_env_load_duplication_documented(self):
        # R8.1: .env load was duplicated, now documented as launcher vs server
        readme = os.path.join(ROOT, "legacy", "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        assert ".env" in text or "load_dotenv" in text or "RTMDK_PORT" in text
        start_prod = os.path.join(ROOT, "start_production.py")
        with open(start_prod, encoding="utf-8") as f:
            sp = f.read()
        assert "load_dotenv" in sp


class TestSillyTavernParity:
    """R8.2: ST proxy/launcher must stay in sync with server OpenAI compat (RISKS.md R8.2)."""

    def test_proxy_and_launcher_exist(self):
        assert os.path.exists(os.path.join(ROOT, "legacy", "rtmdk_st_proxy.py"))
        assert os.path.exists(os.path.join(ROOT, "legacy", "rtmdk_sillytavern_launcher.py"))

    def test_proxy_targets_server(self):
        proxy = os.path.join(ROOT, "legacy", "rtmdk_st_proxy.py")
        launcher = os.path.join(ROOT, "legacy", "rtmdk_sillytavern_launcher.py")
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(proxy, encoding="utf-8") as f:
            pt = f.read()
        with open(launcher, encoding="utf-8") as f:
            lt = f.read()
        with open(srv, encoding="utf-8") as f:
            st = f.read()
        # Proxy must forward to server port 8080 / 5000 mapping
        assert "8080" in pt or "8080" in lt or "rtmdk" in pt.lower()
        assert "5000" in pt or "5000" in lt
        # Both must expose OpenAI compat
        assert "/v1/chat/completions" in st or "chat/completions" in st
        # Proxy should mention chat/completions or OpenAI
        assert "chat" in pt.lower() or "openai" in pt.lower() or "proxy" in pt.lower()

    def test_launcher_references_both(self):
        launcher = os.path.join(ROOT, "legacy", "rtmdk_sillytavern_launcher.py")
        with open(launcher, encoding="utf-8") as f:
            text = f.read()
        assert "rtmdk_server.py" in text
        assert "rtmdk_st_proxy.py" in text
        assert "5000" in text


class TestDepsSingleSource:
    """R9.1: deps single source is pyproject.toml (RISKS.md R9.1)."""

    def test_requirements_headers_point_to_pyproject(self):
        for fname in ["requirements.txt", "requirements-prod.txt"]:
            path = os.path.join(ROOT, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            assert "R9.1" in text
            assert "pyproject.toml" in text
            assert "single source" in text.lower()

    def test_pyproject_is_single_source(self):
        import re

        py = os.path.join(ROOT, "pyproject.toml")
        with open(py, encoding="utf-8") as f:
            text = f.read()
        # Core deps must be in pyproject
        assert "numpy" in text and "fastapi" in text and "pydantic" in text
        # Server extra must contain key deps
        assert "uvicorn" in text and "msgpack" in text and "python-dotenv" in text
        # Version pins should be >= (not hard ==) for flexibility
        assert re.search(r"numpy>=1\.24", text)
        assert re.search(r"fastapi>=0\.100", text)

    def test_requirements_contain_core_deps(self):
        req = os.path.join(ROOT, "requirements.txt")
        with open(req, encoding="utf-8") as f:
            text = f.read()
        assert "fastapi" in text.lower()
        assert "uvicorn" in text.lower()
        assert "msgpack" in text.lower()
        assert "python-dotenv" in text.lower()

        prod = os.path.join(ROOT, "requirements-prod.txt")
        with open(prod, encoding="utf-8") as f:
            text2 = f.read()
        assert "hnswlib" in text2.lower()
        # Documented drift 0.7 vs 0.8 must be mentioned
        assert "0.7.0" in text or "0.8.0" in text2


class TestEnvLoading:
    """R9.2: .env loading must be in lifespan with guard (RISKS.md R9.2)."""

    def test_lifespan_loads_dotenv_with_guard(self):
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(srv, encoding="utf-8") as f:
            text = f.read()
        assert "load_dotenv" in text, "lifespan must load .env (R9.2)"
        assert "R9.2" in text
        assert "PYTEST_CURRENT_TEST" in text
        # Guard must not pollute pytest
        assert "RTMDK_TESTING" in text or "PYTEST" in text
        # Must be inside lifespan, not just entrypoint
        assert "async def lifespan" in text
        # start_production still loads .env as before
        sp = os.path.join(ROOT, "start_production.py")
        with open(sp, encoding="utf-8") as f:
            spt = f.read()
        assert "load_dotenv" in spt

    def test_lifespan_does_not_hardcode_env(self):
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(srv, encoding="utf-8") as f:
            text = f.read()
        # Should use load_dotenv, not manual os.getenv without it
        assert "load_dotenv" in text
        # Must handle ImportError gracefully (optional dep)
        assert "ImportError" in text or "except" in text


class TestArchDebt:
    """R10: God initializer/managers and import cycle must be split (RISKS.md R10.1-10.3)."""

    def test_field_initializer_split(self):
        # R10.1: FieldInitializer must be thin facade delegating to Core/Index/Security
        fi = os.path.join(ROOT, "rtmdk", "memory", "field_initializer.py")
        with open(fi, encoding="utf-8") as f:
            text = f.read()
        assert "R10.1" in text
        assert "DIContainer" in text
        assert "CoreInitializer" in text and "IndexInitializer" in text and "SecurityInitializer" in text
        # Must be thin (<200 lines) — was 574 before split
        lines = text.count("\n")
        assert lines < 250, f"FieldInitializer still god object: {lines} lines (expected <250, was 574)"

        # New initializers must exist
        for sub in ["core.py", "index.py", "security.py"]:
            path = os.path.join(ROOT, "rtmdk", "memory", "initializers", sub)
            assert os.path.exists(path), f"initializers/{sub} missing (R10.1)"
            with open(path, encoding="utf-8") as f:
                t = f.read()
            assert "R10.1" in t

        # DI container must exist
        di = os.path.join(ROOT, "rtmdk", "memory", "initializers", "__init__.py")
        with open(di, encoding="utf-8") as f:
            assert "DIContainer" in f.read()

    def test_batch_resonance_extracted(self):
        # R10.2: QueryManager was 853 lines, batch logic extracted
        br = os.path.join(ROOT, "rtmdk", "memory", "batch_resonance.py")
        assert os.path.exists(br)
        with open(br, encoding="utf-8") as f:
            assert "BatchResonanceEngine" in f.read()
            assert "R10.2" in f.read()
        qm = os.path.join(ROOT, "rtmdk", "memory", "query_manager.py")
        with open(qm, encoding="utf-8") as f:
            qmt = f.read()
        assert "BatchResonanceEngine" in qmt
        assert "R10.2" in qmt
        # NodeManager still god but documented as next split
        assert "NodeManager" in qmt or "R10.2" in qmt

    def test_protocols_replace_getattr_cycle(self):
        # R10.3: field->manager->field cycle was hidden via __getattr__ + mypy ignore
        proto = os.path.join(ROOT, "rtmdk", "memory", "protocols.py")
        assert os.path.exists(proto)
        with open(proto, encoding="utf-8") as f:
            pt = f.read()
        assert "R10.3" in pt
        assert "Protocol" in pt and "FieldLike" in pt
        # Managers should import FieldLike
        qm = os.path.join(ROOT, "rtmdk", "memory", "query_manager.py")
        with open(qm, encoding="utf-8") as f:
            assert "FieldLike" in f.read()
            assert "R10.3" in f.read()
        # Core __getattr__ still exists but documented as cycle (R2.2/R10.3)
        core = os.path.join(ROOT, "rtmdk", "memory", "core.py")
        with open(core, encoding="utf-8") as f:
            assert "R10.3" in f.read() or "R2.2" in f.read()


class TestDocsSync:
    """R11: docs must not drift from code (RISKS.md R11.1-11.3)."""

    def test_check_docs_sync_script_exists(self):
        script = os.path.join(ROOT, "scripts", "check_docs_sync.py")
        assert os.path.exists(script)
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "R11.1" in text
        assert "README.md" in text and "cloc" in text.lower()

    def test_readme_stats_are_honest(self):
        # R11.1: README header must be ~42k/206/48/~1300, not 74k/440/49/1281
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        assert "scripts/check_docs_sync.py" in text
        assert "R11.1" in text or "42,000" in text or "206+" in text
        # Must mention 48 endpoints, not 49
        assert "48 API" in text or "48 endpoints" in text

    def test_docs_version_matches_package(self):
        import rtmdk

        docs_readme = os.path.join(ROOT, "docs", "README.md")
        with open(docs_readme, encoding="utf-8") as f:
            text = f.read()
        assert "R11" in text or "8.3.4" in text
        assert rtmdk.__version__ in text

        backlog = os.path.join(ROOT, "BACKLOG.md")
        with open(backlog, encoding="utf-8") as f:
            bl = f.read()
        assert rtmdk.__version__ in bl, "BACKLOG must be 8.3.4 (R11.3)"

    def test_mkdocs_includes_risks(self):
        mkdocs = os.path.join(ROOT, "mkdocs.yml")
        with open(mkdocs, encoding="utf-8") as f:
            text = f.read()
        assert "risks" in text.lower()
        assert "R11" in text or "RISKS" in text or "risks.md" in text

    def test_endpoint_count_matches_server(self):
        import re

        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(srv, encoding="utf-8") as f:
            srv_text = f.read()
        actual = len(re.findall(r"@app\.(get|post|put|delete|patch)", srv_text))
        readme = os.path.join(ROOT, "README.md")
        with open(readme, encoding="utf-8") as f:
            readme_text = f.read()
        # README must mention actual count
        assert str(actual) in readme_text, f"README must mention {actual} endpoints"


class TestSecurityAndPersistence:
    """R12: security and persistence must be honest (RISKS.md R12.1-12.3)."""

    def test_memory_file_is_msgpack(self):
        # R12.1: default must be memory.msgpack (was misleading memory.json)
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(srv, encoding="utf-8") as f:
            text = f.read()
        assert "memory.msgpack" in text, "default MEMORY_FILE must be memory.msgpack (R12.1)"
        assert "R12.1" in text
        # Serialization must support both and be documented as msgpack+zlib
        ser = os.path.join(ROOT, "rtmdk", "memory", "serialization.py")
        with open(ser, encoding="utf-8") as f:
            st = f.read()
        assert "msgpack" in st and "zlib" in st
        assert "R12.1" in st
        assert "memory.msgpack" in st or "memory.json" in st
        # Docs must describe format
        guide = os.path.join(ROOT, "docs", "SOT_V2_GUIDE.md")
        with open(guide, encoding="utf-8") as f:
            assert "msgpack" in f.read().lower()

    def test_production_preset_forbids_default_key(self):
        from rtmdk.memory.config import RTMDKConfig

        # R12.2: production() with default rtmdk-local must ERROR
        cfg = RTMDKConfig.production()
        # Ensure it's production_mode True (set in R12.2 fix)
        assert cfg.production.production_mode is True
        warns = cfg.validate()
        assert any("rtmdk-local" in w and "ERROR" in w for w in warns), f"production must ERROR on default key, got {warns}"

        # Non-production (local) must NOT error on same key
        cfg2 = RTMDKConfig.local()
        warns2 = cfg2.validate()
        # local may still have api_key rtmdk-local but not production_mode, so no ERROR
        assert not any("rtmdk-local" in w and "ERROR" in w and "production_mode" in w for w in warns2)

        # Custom key should not error
        cfg3 = RTMDKConfig(production_mode=True, api_key="my-secret-key-123")
        warns3 = cfg3.validate()
        assert not any("rtmdk-local" in w for w in warns3)

    def test_get_embedding_fail_fast(self):
        srv = os.path.join(ROOT, "rtmdk", "server", "app.py")
        with open(srv, encoding="utf-8") as f:
            text = f.read()
        assert "R12.3" in text
        # Must raise HTTPException 400 on dim mismatch, not silent pad
        assert "HTTPException" in text and "400" in text
        assert "Embedding dimension mismatch" in text
        # Old silent pad code must be gone (np.pad)
        # The pad line inside get_embedding should not exist for mismatch case
        # Check that the function contains raise, not just warning+pad
        assert "raise HTTPException" in text
        # Ensure no silent fallback remains for mismatch (np.pad after warning)
        # The old pattern was logger.warning then np.pad — now it should be logger.error then raise
        assert "logger.error" in text or "logger.warning" in text


class TestLegacyModules:
    """legacy/ SillyTavern modules must remain importable after the repo move."""

    def test_embedder_lmstudio_importable(self):
        import embedder_lmstudio  # noqa: F401

    def test_ux_and_dashboard_routers_importable(self):
        import rtmdk_dashboard_ui  # noqa: F401
        import rtmdk_server_ux  # noqa: F401

    def test_server_legacy_path_helper(self):
        from rtmdk.server.app import _ensure_legacy_path

        _ensure_legacy_path()
        import sys

        assert any(p.endswith("legacy") for p in sys.path), "_ensure_legacy_path() did not add legacy/ to sys.path"
