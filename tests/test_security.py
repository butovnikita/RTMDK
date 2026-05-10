"""
tests/test_security.py — Security hardening tests for RTMDK.

Covers:
1. Path sanitization blocks traversal attacks
2. Path sanitization blocks absolute paths
3. Rate limiting on add_node
4. JSON size limits
"""

import pytest
import numpy as np
import os
import tempfile

from rtmdk.memory.core import RTMDKMemory, RTMDKConfig, SecurityViolationError
from rtmdk.memory.utils import _sanitize_path, _safe_json_load


class TestPathSanitization:
    def test_blocks_parent_directory_traversal(self):
        with pytest.raises(SecurityViolationError):
            _sanitize_path("../../../etc/passwd")

    def test_blocks_dotdot_in_middle(self):
        with pytest.raises(SecurityViolationError):
            _sanitize_path("data/../secret.txt")

    def test_allows_absolute_unix_path(self):
        # Absolute paths are allowed (blocked by OS/container policy, not by
        # us)
        assert _sanitize_path("/etc/passwd") == "\\etc\\passwd"

    def test_allows_absolute_windows_path(self):
        assert _sanitize_path(
            "C:\\Windows\\system.ini") == "C:\\Windows\\system.ini"

    def test_allows_safe_relative_path(self):
        assert _sanitize_path("memory/state.json") == "memory\\state.json"
        assert _sanitize_path("backup_2024.json") == "backup_2024.json"

    def test_blocks_dotdot_in_absolute_path(self):
        with pytest.raises(SecurityViolationError):
            _sanitize_path("C:\\Users\\..\\secret.txt")
        with pytest.raises(SecurityViolationError):
            _sanitize_path("/home/../secret.txt")


class TestRateLimiting:
    def test_add_node_rate_limit(self, monkeypatch):
        monkeypatch.setenv("RTMDK_ADD_RATE_LIMIT", "100")
        config = RTMDKConfig.local()
        config.max_nodes = 10000

        def dummy_embedder(text: str) -> np.ndarray:
            return np.ones(config.embedding_dim, dtype=np.float32) * 0.1

        memory = RTMDKMemory(config=config, embedder=dummy_embedder)

        # First 100 adds should succeed
        for i in range(100):
            memory.add_node(
                dummy_embedder(f"node_{i}"),
                {"text": f"content {i}"},
            )

        # 101st add within 1 second should fail
        with pytest.raises(SecurityViolationError, match="Rate limit exceeded"):
            memory.add_node(
                dummy_embedder("overflow"),
                {"text": "overflow content"},
            )

    def test_rate_limit_resets_after_window(self):
        import time
        config = RTMDKConfig.local()
        config.max_nodes = 10000

        def dummy_embedder(text: str) -> np.ndarray:
            return np.ones(config.embedding_dim, dtype=np.float32) * 0.1

        memory = RTMDKMemory(config=config, embedder=dummy_embedder)

        # Add 100 nodes
        for i in range(100):
            memory.add_node(dummy_embedder(f"n{i}"), {"text": str(i)})

        # Manually backdate timestamps to simulate time passing
        memory.field._add_node_timestamps.clear()
        for _ in range(100):
            memory.field._add_node_timestamps.appendleft(time.time() - 2.0)

        # Should succeed again because all timestamps are > 1 sec old
        memory.add_node(dummy_embedder("after_window"), {"text": "ok"})


class TestJsonSizeLimit:
    def test_safe_json_load_blocks_oversized_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('"' + "x" * (101 * 1024 * 1024) + '"')
            path = f.name
        try:
            with pytest.raises(ValueError, match="File too large"):
                _safe_json_load(path)
        finally:
            os.unlink(path)

    def test_safe_json_load_allows_normal_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            import json
            json.dump({"config": {}, "nodes": []}, f)
            path = f.name
        try:
            result = _safe_json_load(path)
            assert "config" in result
        finally:
            os.unlink(path)
