"""Tests for rtmdk/dashboard.py — HTML dashboard generator."""

from types import SimpleNamespace

import pytest

from rtmdk.dashboard import DashboardGenerator


def make_node(text, salience, amplitude, tier="semantic"):
    return SimpleNamespace(content={"text": text}, salience=salience, amplitude=amplitude, tier=tier)


@pytest.fixture
def memory():
    return SimpleNamespace(
        field=SimpleNamespace(
            stats={"total_queries": 7, "consolidations": 2, "bm25_fallbacks": 1},
            nodes={
                "n1": make_node("alpha memory", 0.9, 1.5, tier="episodic"),
                "n2": make_node("beta memory", 0.1, 0.4, tier="semantic"),
                "n3": make_node("gamma memory", 0.5, 1.0, tier="episodic"),
            },
        )
    )


class TestGenerate:
    def test_writes_html_file(self, memory, tmp_path, capsys):
        out = tmp_path / "dash.html"
        path = DashboardGenerator(memory).generate(str(out))

        assert path == str(out)
        html = out.read_text(encoding="utf-8")
        assert html.startswith("<!DOCTYPE html>")
        assert "<title>RTMDK Memory Dashboard</title>" in html
        assert "Dashboard saved to:" in capsys.readouterr().out


class TestTierRows:
    def test_counts_and_percentages(self, memory):
        gen = DashboardGenerator(memory)
        rows = gen._tier_rows({"episodic": 2, "semantic": 1}, 3)

        lines = rows.splitlines()
        assert len(lines) == 2
        assert "<td>episodic</td><td>2</td>" in lines[0]
        assert "67%" in lines[0]
        assert "<td>semantic</td><td>1</td>" in lines[1]
        assert "33%" in lines[1]

    def test_zero_total_no_division_error(self, memory):
        gen = DashboardGenerator(memory)
        rows = gen._tier_rows({"episodic": 1}, 0)
        assert "100%" in rows


class TestTopNodeRows:
    def test_row_content_and_truncation(self, memory):
        gen = DashboardGenerator(memory)
        node = make_node("x" * 100, 0.123, 4.567)
        rows = gen._top_node_rows([node])

        assert "<td>1</td><td>0.123</td><td>4.567</td>" in rows
        assert "x" * 80 in rows
        assert "x" * 81 not in rows

    def test_enumeration(self, memory):
        gen = DashboardGenerator(memory)
        nodes = [make_node(f"t{i}", 0.1 * i, 1.0) for i in range(3)]
        rows = gen._top_node_rows(nodes).splitlines()
        assert rows[0].startswith("<tr><td>1</td>")
        assert rows[2].startswith("<tr><td>3</td>")


class TestFormatStats:
    def test_sorted_key_value_lines(self, memory):
        gen = DashboardGenerator(memory)
        text = gen._format_stats({"b": 2, "a": 1})

        assert text.splitlines() == ["  a: 1", "  b: 2"]
