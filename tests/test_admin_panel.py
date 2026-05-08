"""E2E smoke test for React Admin Panel.

Verifies that the admin panel builds and serves correctly.
Does NOT require a running server — only checks static assets.
"""

import os
import subprocess
import sys

import pytest

ADMIN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "admin")


class TestAdminPanelStructure:
    def test_admin_directory_exists(self):
        assert os.path.isdir(ADMIN_DIR), f"admin/ directory not found at {ADMIN_DIR}"

    def test_package_json_exists(self):
        assert os.path.isfile(os.path.join(ADMIN_DIR, "package.json"))

    def test_vite_config_exists(self):
        assert os.path.isfile(os.path.join(ADMIN_DIR, "vite.config.js"))

    def test_source_files_exist(self):
        src = os.path.join(ADMIN_DIR, "src")
        assert os.path.isfile(os.path.join(src, "App.jsx"))
        assert os.path.isfile(os.path.join(src, "App.css"))
        assert os.path.isfile(os.path.join(src, "main.jsx"))
        assert os.path.isdir(os.path.join(src, "components"))

    def test_components_exist(self):
        comp = os.path.join(ADMIN_DIR, "src", "components")
        for name in ["Dashboard.jsx", "MemoryNodes.jsx", "QueryInterface.jsx", "SOTPanel.jsx"]:
            assert os.path.isfile(os.path.join(comp, name)), f"Missing component: {name}"

    def test_builds_successfully(self):
        """Run npm run build and verify dist/ is created."""
        if not os.path.isdir(ADMIN_DIR):
            pytest.skip("admin/ directory not found")

        # Check if node_modules exists — if not, npm install is needed
        node_modules = os.path.join(ADMIN_DIR, "node_modules")
        if not os.path.isdir(node_modules):
            pytest.skip("node_modules not found — run 'npm install' in admin/ first")

        # Check npm is available in PATH
        npm_cmd = "npm"
        if sys.platform == "win32":
            npm_cmd = "npm.cmd"
        try:
            subprocess.run(
                [npm_cmd, "--version"],
                cwd=ADMIN_DIR,
                capture_output=True,
                timeout=10,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("npm not found in PATH")

        # Run build
        result = subprocess.run(
            [npm_cmd, "run", "build"],
            cwd=ADMIN_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Build failed:\n{result.stderr}"

        # Verify dist/index.html exists
        dist = os.path.join(ADMIN_DIR, "dist")
        assert os.path.isdir(dist), "dist/ directory not created"
        assert os.path.isfile(os.path.join(dist, "index.html")), "dist/index.html not found"

        # Verify JS bundle exists
        assets = os.path.join(dist, "assets")
        assert os.path.isdir(assets), "dist/assets/ not found"
        js_files = [f for f in os.listdir(assets) if f.endswith(".js")]
        css_files = [f for f in os.listdir(assets) if f.endswith(".css")]
        assert len(js_files) >= 1, "No JS bundle in dist/assets/"
        assert len(css_files) >= 1, "No CSS bundle in dist/assets/"

    def test_api_base_configurable(self):
        """Verify API base URL is defined in App.jsx."""
        app_path = os.path.join(ADMIN_DIR, "src", "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "API_BASE" in content or "apiBase" in content, "API_BASE not configurable"

    def test_all_tabs_defined(self):
        """Verify all 4 tabs are defined in App.jsx."""
        app_path = os.path.join(ADMIN_DIR, "src", "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        for tab in ["dashboard", "nodes", "query", "sot"]:
            assert tab in content, f"Tab '{tab}' not found in App.jsx"
