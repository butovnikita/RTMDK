"""Tests for pipeline metrics persistence."""

import os


from rtmdk.pipeline.persistence import PipelineMetricsStore


class TestPipelineMetricsStore:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        store.write({"query_text": "q1", "total_latency_ms": 10.5, "stages": []})
        store.write({"query_text": "q2", "total_latency_ms": 20.0, "stages": []})

        records = store.read_all()
        assert len(records) == 2
        assert records[0]["query_text"] == "q1"
        assert records[1]["query_text"] == "q2"
        assert "ts" in records[0]

    def test_summary(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        store.write(
            {
                "query_text": "q1",
                "total_latency_ms": 10.0,
                "stages": [
                    {"stage": "embed", "latency_ms": 5.0, "error": None, "degraded": False},
                    {"stage": "retrieve", "latency_ms": 5.0, "error": None, "degraded": False},
                ],
            }
        )
        store.write(
            {
                "query_text": "q2",
                "total_latency_ms": 20.0,
                "stages": [
                    {"stage": "embed", "latency_ms": 10.0, "error": None, "degraded": False},
                    {"stage": "retrieve", "latency_ms": 10.0, "error": "boom", "degraded": True},
                ],
            }
        )

        summary = store.summary()
        assert summary["queries"] == 2
        assert summary["total_latency_ms"]["mean"] == 15.0
        assert summary["stages"]["embed"]["latency_ms"]["mean"] == 7.5
        assert summary["stages"]["retrieve"]["errors"] == 1
        assert summary["stages"]["retrieve"]["degraded"] == 1

    def test_empty_summary(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        summary = store.summary()
        assert summary["queries"] == 0

    def test_summary_stage_filter(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        store.write(
            {
                "query_text": "q1",
                "total_latency_ms": 10.0,
                "stages": [
                    {"stage": "embed", "latency_ms": 5.0, "error": None, "degraded": False},
                    {"stage": "retrieve", "latency_ms": 5.0, "error": None, "degraded": False},
                ],
            }
        )
        store.write(
            {
                "query_text": "q2",
                "total_latency_ms": 20.0,
                "stages": [
                    {"stage": "embed", "latency_ms": 10.0, "error": None, "degraded": False},
                    {"stage": "retrieve", "latency_ms": 10.0, "error": "boom", "degraded": True},
                ],
            }
        )

        summary = store.summary(stage_filter="embed")
        assert summary["queries"] == 2
        assert "embed" in summary["stages"]
        assert "retrieve" not in summary["stages"]
        assert summary["stages"]["embed"]["latency_ms"]["mean"] == 7.5

    def test_summary_since_filter(self, tmp_path):
        import time

        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        store.write({"query_text": "old", "total_latency_ms": 1.0, "stages": []})
        time.sleep(0.05)
        now = time.time()
        time.sleep(0.05)
        store.write({"query_text": "new", "total_latency_ms": 2.0, "stages": []})

        summary = store.summary(since=now)
        assert summary["queries"] == 1
        assert summary["total_latency_ms"]["mean"] == 2.0

    def test_rotation(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path), max_size_mb=0.001)  # ~1KB
        # Write enough to trigger rotation
        big_record = {"query_text": "x" * 500, "stages": []}
        store.write(big_record)
        store.write(big_record)
        store.write(big_record)

        # Should have rotated
        assert os.path.exists(str(path) + ".1")
        records = store.read_all()
        assert len(records) >= 2

    def test_thread_safety(self, tmp_path):
        import threading

        path = tmp_path / "metrics.jsonl"
        store = PipelineMetricsStore(str(path))
        errors = []

        def worker(i):
            try:
                store.write({"query_text": f"q{i}", "stages": []})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        records = store.read_all()
        assert len(records) == 20
