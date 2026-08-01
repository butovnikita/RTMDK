"""
rtmdk/dashboard.py — HTML Dashboard Generator.

Generates a self-contained HTML report with memory statistics,
visualizations, and analysis.
"""

import time
from typing import Dict
from pathlib import Path


class DashboardGenerator:
    """Generates HTML dashboard for RTMDK memory visualization.

    Usage:
        dashboard = DashboardGenerator(memory)
        dashboard.generate("dashboard.html")
    """

    def __init__(self, memory):
        self.memory = memory

    def generate(self, output_path: str = "rtmdk_dashboard.html"):
        """Generate HTML dashboard."""
        stats = self.memory.field.stats
        nodes = self.memory.field.nodes
        node_count = len(nodes)

        # Collect node data for visualization
        tiers = {}
        for nid, node in nodes.items():
            tier = getattr(node, "tier", "unknown")
            if tier not in tiers:
                tiers[tier] = 0
            tiers[tier] += 1

        # Top-10 by salience
        top_nodes = sorted(
            nodes.values(),
            key=lambda n: getattr(n, "salience", 0.0),
            reverse=True,
        )[:10]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RTMDK Memory Dashboard</title>
    <style>
        body {{ font-family: -apple-system, sans-serif;\n            max-width: 1200px; margin: 0 auto;\n            padding: 20px; background: #f5f5f5; }}  # noqa: E501
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px;\n            padding: 20px; margin: 10px 0;\n            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}  # noqa: E501
        .stat {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .bar {{ background: #4CAF50; height: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>RTMDK Memory Dashboard</h1>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="grid">
        <div class="card">
            <h3>Total Nodes</h3>
            <div class="stat">{node_count}</div>
        </div>
        <div class="card">
            <h3>Total Queries</h3>
            <div class="stat">{stats.get('total_queries', 0)}</div>
        </div>
        <div class="card">
            <h3>Consolidations</h3>
            <div class="stat">{stats.get('consolidations', 0)}</div>
        </div>
        <div class="card">
            <h3>BM25 Fallbacks</h3>
            <div class="stat">{stats.get('bm25_fallbacks', 0)}</div>
        </div>
    </div>

    <div class="card">
        <h2>Memory Tiers</h2>
        <table>
            <tr><th>Tier</th><th>Count</th><th>Distribution</th></tr>
            {self._tier_rows(tiers, node_count)}
        </table>
    </div>

    <div class="card">
        <h2>Top 10 Nodes by Salience</h2>
        <table>
            <tr><th>#</th><th>Salience</th><th>Amplitude</th><th>Tier</th><th>Text Preview</th></tr>
            {self._top_node_rows(top_nodes)}
        </table>
    </div>

    <div class="card">
        <h2>Full Statistics</h2>
        <pre>{self._format_stats(stats)}</pre>
    </div>
</body>
</html>"""

        path = Path(output_path)
        path.write_text(html, encoding="utf-8")
        print(f"Dashboard saved to: {output_path}")
        return str(path)

    def _tier_rows(self, tiers: Dict, total: int) -> str:
        rows = []
        for tier, count in sorted(tiers.items()):
            pct = count / max(total, 1) * 100
            rows.append(
                f"<tr><td>{tier}</td><td>{count}</td>"
                f"<td><div class='bar' style='width:{pct}%'></div> {pct:.0f}%</td></tr>"
            )
        return "\n".join(rows)

    def _top_node_rows(self, nodes) -> str:
        rows = []
        for i, node in enumerate(nodes, 1):
            text = node.content.get("text", "")[:80]
            rows.append(
                f"<tr><td>{i}</td><td>{node.salience:.3f}</td>"
                f"<td>{node.amplitude:.3f}</td><td>{getattr(node, 'tier', '?')}</td>"
                f"<td>{text}</td></tr>"
            )
        return "\n".join(rows)

    def _format_stats(self, stats: Dict) -> str:
        lines = []
        for key, value in sorted(stats.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
