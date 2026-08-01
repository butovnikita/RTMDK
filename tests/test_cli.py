"""Tests for rtmdk/cli.py — CLI entry point (status/query/stats/export/recommend/presets/bootstrap/diagnose)."""

import json
import sys

import pytest

import rtmdk.cli as cli


def run_cli(monkeypatch, argv):
    """Invoke cli.main() with patched argv."""
    monkeypatch.setattr(sys, "argv", ["rtmdk", *argv])
    cli.main()


class TestMainEntryPoint:
    def test_python_dash_m_dispatches_to_cli(self, monkeypatch, capsys):
        """`python -m rtmdk status` must route to the CLI, not the server."""
        import os
        import runpy

        monkeypatch.setattr(sys, "argv", ["rtmdk", "status"])
        # __main__ calls load_dotenv(): keep .env vars from leaking into os.environ
        saved_env = dict(os.environ)
        try:
            runpy.run_module("rtmdk.__main__", run_name="__main__")
        finally:
            os.environ.clear()
            os.environ.update(saved_env)

        assert "RTMDK Status" in capsys.readouterr().out


class TestSimpleCommands:
    def test_status(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["status"])
        out = capsys.readouterr().out
        assert "RTMDK Status" in out
        assert "memory.field.stats" in out

    def test_query_echoes_query(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["query", "What do I know about coffee?"])
        out = capsys.readouterr().out
        assert "Query: What do I know about coffee?" in out
        assert "create_rtmdk" in out

    def test_query_missing_arg_exits_2(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["query"])
        assert exc.value.code == 2

    def test_stats(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["stats"])
        out = capsys.readouterr().out
        assert "RTMDK Statistics" in out

    def test_export_default_output(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["export"])
        out = capsys.readouterr().out
        assert "rtmdk_backup.json" in out

    def test_export_custom_output(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["export", "--output", "my_backup.json"])
        out = capsys.readouterr().out
        assert "Export to: my_backup.json" in out

    def test_no_command_prints_help(self, monkeypatch, capsys):
        run_cli(monkeypatch, [])
        out = capsys.readouterr().out
        assert "usage" in out.lower()


class TestRecommend:
    def test_local_preset(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["recommend", "--nodes", "1000", "--ram", "256"])
        out = capsys.readouterr().out
        assert "Preset:        local" in out
        assert "Est. RAM:      12.0 MB" in out
        # No overrides expected for this configuration
        assert "Overrides" not in out

    def test_latency_overrides_printed(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["recommend", "--nodes", "1000", "--latency", "10"])
        out = capsys.readouterr().out
        assert "Overrides:" in out
        assert "offline_dreaming" in out

    def test_agent_use_case(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["recommend", "--use-case", "agent"])
        out = capsys.readouterr().out
        assert "Preset:        agent" in out

    def test_enterprise_for_huge_scale(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["recommend", "--nodes", "400000", "--ram", "1024"])
        out = capsys.readouterr().out
        assert "Preset:        enterprise" in out


class TestPresets:
    def test_lists_all_presets(self, monkeypatch, capsys):
        from rtmdk import list_presets

        run_cli(monkeypatch, ["presets"])
        out = capsys.readouterr().out
        assert "RTMDK Available Presets" in out
        for name in list_presets():
            assert name in out
        # Column header sanity
        assert "Dim" in out and "Engrams" in out


class TestPipelineDiagnose:
    def test_default_smoke_test(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["pipeline-diagnose"])
        out = capsys.readouterr().out
        assert "No memory file" in out
        assert "Pipeline stages (6):" in out
        assert "embed" in out and "retrieve" in out
        assert "[OK] Pipeline is operational" in out

    def test_unknown_preset_falls_back(self, monkeypatch, capsys):
        run_cli(monkeypatch, ["pipeline-diagnose", "--preset", "does-not-exist"])
        out = capsys.readouterr().out
        assert "Unknown preset 'does-not-exist', falling back to 'local'" in out
        assert "[OK] Pipeline is operational" in out

    def test_existing_unloadable_memory_file_warns(self, monkeypatch, capsys, tmp_path):
        bad = tmp_path / "memory.json"
        bad.write_text("{not a valid memory dump", encoding="utf-8")

        run_cli(monkeypatch, ["pipeline-diagnose", "--memory", str(bad)])
        out = capsys.readouterr().out
        assert f"Loading memory from: {bad}" in out
        assert "[WARN] Could not load:" in out
        assert "[OK] Pipeline is operational" in out


class TestBootstrapSbert:
    def test_missing_corpus_exits_1(self, monkeypatch, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["bootstrap", str(tmp_path / "nope.json")])
        assert exc.value.code == 1
        assert "corpus file not found" in capsys.readouterr().out

    def test_invalid_corpus_shape_exits_1(self, monkeypatch, capsys, tmp_path):
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["bootstrap", str(corpus)])
        assert exc.value.code == 1
        assert "must be a list of strings" in capsys.readouterr().out

    def test_happy_path_list_corpus(self, monkeypatch, capsys, tmp_path):
        import rtmdk.memory.bootstrap_sbert as bs_mod

        captured = {}

        def fake_run_bootstrap(texts, output_path, model_name):
            captured["texts"] = texts
            captured["output_path"] = output_path
            captured["model_name"] = model_name

        monkeypatch.setattr(bs_mod, "run_bootstrap", fake_run_bootstrap)

        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps(["alpha", "beta", "gamma"]), encoding="utf-8")
        out_file = tmp_path / "proj.npz"

        run_cli(monkeypatch, ["bootstrap", str(corpus), "--output", str(out_file), "--model", "fake-model"])

        out = capsys.readouterr().out
        assert captured["texts"] == ["alpha", "beta", "gamma"]
        assert captured["output_path"] == str(out_file)
        assert captured["model_name"] == "fake-model"
        assert "Texts:  3" in out
        assert f"Bootstrap projection saved to: {out_file}" in out

    def test_happy_path_records_corpus(self, monkeypatch, capsys, tmp_path):
        import rtmdk.memory.bootstrap_sbert as bs_mod

        captured = {}
        monkeypatch.setattr(
            bs_mod,
            "run_bootstrap",
            lambda texts, output_path, model_name: captured.update(texts=texts),
        )

        records = [{"context": "ctx one", "answer": "ans one"}, {"context": "ctx two"}]
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps({"records": records}), encoding="utf-8")

        run_cli(monkeypatch, ["bootstrap", str(corpus)])

        assert captured["texts"] == ["ctx one ans one", "ctx two "]
        assert "Texts:  2" in capsys.readouterr().out


class TestBootstrapFasttext:
    def test_missing_model_exits_1(self, monkeypatch, capsys, tmp_path):
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps(["a"]), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["bootstrap-fasttext", str(tmp_path / "no.model"), str(corpus)])
        assert exc.value.code == 1
        assert "model file not found" in capsys.readouterr().out

    def test_missing_corpus_exits_1(self, monkeypatch, capsys, tmp_path):
        model = tmp_path / "m.model"
        model.write_bytes(b"fake")

        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["bootstrap-fasttext", str(model), str(tmp_path / "no.json")])
        assert exc.value.code == 1
        assert "corpus file not found" in capsys.readouterr().out

    def test_invalid_corpus_shape_exits_1(self, monkeypatch, capsys, tmp_path):
        model = tmp_path / "m.model"
        model.write_bytes(b"fake")
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps(42), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, ["bootstrap-fasttext", str(model), str(corpus)])
        assert exc.value.code == 1
        assert "must be a list of strings" in capsys.readouterr().out

    def test_happy_path(self, monkeypatch, capsys, tmp_path):
        import rtmdk.memory.bootstrap_fasttext as bf_mod

        captured = {}
        monkeypatch.setattr(
            bf_mod,
            "run_bootstrap",
            lambda tok, texts, model_path: captured.update(texts=texts, model_path=model_path),
        )

        model = tmp_path / "m.model"
        model.write_bytes(b"fake")
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps(["hello world", "goodbye world"]), encoding="utf-8")
        out_file = tmp_path / "state.json"

        run_cli(monkeypatch, ["bootstrap-fasttext", str(model), str(corpus), "--output", str(out_file)])

        out = capsys.readouterr().out
        assert captured["texts"] == ["hello world", "goodbye world"]
        assert captured["model_path"] == str(model)
        assert f"Bootstrap state saved to: {out_file}" in out

        state = json.loads(out_file.read_text(encoding="utf-8"))
        assert state["latent_dim"] == 64
        assert state["tokenization_mode"] == "word"
