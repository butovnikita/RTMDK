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
