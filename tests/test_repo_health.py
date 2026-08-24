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
