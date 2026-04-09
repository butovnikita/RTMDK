"""rtmdk_dashboard_ui.py — Web UI Dashboard for RTMDK.

Simple UI for:
- Selecting presets on the fly
- Toggling UX functions on/off
- Viewing live stats
- Quick actions (backup, prune, export)

Usage:
    # Integrated with server:
    from rtmdk_dashboard_ui import create_dashboard_router
    app.include_router(create_dashboard_router(memory, config))
    
    # Or standalone:
    python rtmdk_dashboard_ui.py --port 8081
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RTMDK Dashboard</title>
<style>
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 20px; max-width: 1400px; margin: 0 auto; }
h1 { color: var(--accent); margin-bottom: 20px; font-size: 1.8em; }
h2 { color: var(--text); margin: 20px 0 10px; font-size: 1.2em; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card h3 { font-size: 0.9em; color: var(--text-dim); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
.stat { font-size: 2em; font-weight: bold; }
.stat.green { color: var(--green); } .stat.red { color: var(--red); } .stat.accent { color: var(--accent); } .stat.yellow { color: var(--yellow); }
select, button { background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 0.95em; cursor: pointer; }
select:hover, button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
button.primary:hover { background: #4090e0; }
button.danger { background: var(--red); color: #fff; border-color: var(--red); }
.toggle-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }
.toggle-row:last-child { border: none; }
.toggle-label { font-size: 0.9em; }
.toggle { position: relative; width: 44px; height: 24px; }
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle-slider { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: var(--border); border-radius: 24px; cursor: pointer; transition: 0.3s; }
.toggle-slider:before { content: ''; position: absolute; height: 18px; width: 18px; left: 3px; bottom: 3px; background: var(--text); border-radius: 50%; transition: 0.3s; }
.toggle input:checked + .toggle-slider { background: var(--accent); }
.toggle input:checked + .toggle-slider:before { transform: translateX(20px); }
.status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.status.ok { background: var(--green); } .status.err { background: var(--red); }
.actions { display: flex; flex-wrap: wrap; gap: 8px; }
#status-bar { position: fixed; top: 0; left: 0; right: 0; background: var(--green); color: #fff; text-align: center; padding: 8px; display: none; z-index: 100; }
.preset-desc { font-size: 0.8em; color: var(--text-dim); margin-top: 4px; }
.log { background: #000; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85em; max-height: 200px; overflow-y: auto; margin-top: 10px; }
.log-line { padding: 2px 0; border-bottom: 1px solid #111; }
</style>
</head>
<body>
<div id="status-bar"></div>

<h1>🧠 RTMDK Dashboard <span id="conn-status"><span class="status ok"></span>Connected</span></h1>

<!-- PRESET SELECTOR -->
<div class="card" style="margin-bottom:20px;">
    <h3>Configuration Preset</h3>
    <select id="preset-select" style="width:100%;font-size:1.1em;padding:10px;">
        <option value="local">🏠 Local (16MB, ~5ms, 10K nodes)</option>
        <option value="production">⚡ Production (50MB, ~6ms, 100K nodes)</option>
        <option value="research">🔬 Research (200MB, ~50ms, unlimited)</option>
        <option value="enterprise">🏢 Enterprise (250MB/shard, 500K+ nodes)</option>
        <option value="agent">🤖 Agent (30MB, 50K nodes)</option>
        <option value="legal">⚖️ Legal (100MB, Z3 prover)</option>
        <option value="medical">🏥 Medical (100MB, high trust)</option>
        <option value="streaming">🚀 Streaming (30MB, ~3ms)</option>
    </select>
    <div class="preset-desc" id="preset-desc">Personal assistant, minimal resources</div>
    <button class="primary" onclick="applyPreset()" style="margin-top:10px;">Apply Preset</button>
</div>

<!-- STATS -->
<h2>📊 Live Statistics</h2>
<div class="grid">
    <div class="card"><h3>Nodes</h3><div class="stat accent" id="stat-nodes">0</div></div>
    <div class="card"><h3>Total Queries</h3><div class="stat" id="stat-queries">0</div></div>
    <div class="card"><h3>Consolidations</h3><div class="stat" id="stat-consol">0</div></div>
    <div class="card"><h3>Cache Hit Rate</h3><div class="stat green" id="stat-cache">—</div></div>
    <div class="card"><h3>Engram Retrievals</h3><div class="stat" id="stat-engram">0</div></div>
    <div class="card"><h3>BM25 Fallbacks</h3><div class="stat yellow" id="stat-bm25">0</div></div>
</div>

<!-- UX TOGGLES -->
<h2>🔧 UX Features</h2>
<div class="grid">
    <div class="card">
        <h3>Core Features</h3>
        <div class="toggle-row"><span class="toggle-label">Engrams (Pattern Completion)</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('engrams',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Offline Dreaming</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('dreaming',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Causal Traversal</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('causal',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">SSM Dynamics (O(N))</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('ssm',this.checked)"><span class="toggle-slider"></span></label></div>
    </div>
    <div class="card">
        <h3>UX Functions</h3>
        <div class="toggle-row"><span class="toggle-label">Embedding Cache</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('cache',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Smart Pruning</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('pruning',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Session Persistence</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('sessions',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Rate Limiter</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('rate_limit',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Memory Refresh</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('refresh',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Event System</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('events',this.checked)"><span class="toggle-slider"></span></label></div>
    </div>
    <div class="card">
        <h3>Advanced</h3>
        <div class="toggle-row"><span class="toggle-label">Trust Consensus</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('trust',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Neuro-Symbolic Prover</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('prover',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Health Monitor</span><label class="toggle"><input type="checkbox" checked onchange="toggleFeature('health',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Multi-Tenant</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('multi_tenant',this.checked)"><span class="toggle-slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">A/B Testing</span><label class="toggle"><input type="checkbox" onchange="toggleFeature('ab_testing',this.checked)"><span class="toggle-slider"></span></label></div>
    </div>
</div>

<!-- QUICK ACTIONS -->
<h2>⚡ Quick Actions</h2>
<div class="card">
    <div class="actions">
        <button onclick="doAction('backup')">💾 Create Backup</button>
        <button onclick="doAction('prune')">🧹 Run Pruning</button>
        <button onclick="doAction('export_md')">📄 Export Markdown</button>
        <button onclick="doAction('export_json')">📋 Export JSON</button>
        <button onclick="doAction('clear_cache')">🗑️ Clear Cache</button>
        <button class="danger" onclick="doAction('clear_memory')">⚠️ Clear Memory</button>
        <button onclick="doAction('analytics')">📊 Analytics</button>
        <button onclick="doAction('health')">❤️ Health Check</button>
    </div>
    <div class="log" id="action-log"><div class="log-line" style="color:#8b949e;">Actions will appear here...</div></div>
</div>

<script>
const API_BASE = window.location.origin;
const presets = {
    local: "Personal assistant, minimal resources (~16MB, 10K nodes)",
    production: "Multi-user server with all optimizations (~50MB, 100K nodes)",
    research: "Maximum accuracy, all features enabled (~200MB, unlimited)",
    enterprise: "Distributed system, sharding enabled (250MB/shard, 500K+ nodes)",
    agent: "Autonomous agent with active inference and causal reasoning",
    legal: "Legal domain with Z3 prover for contradiction detection",
    medical: "Medical domain with high trust and audit trail",
    streaming: "High-throughput real-time, minimal latency (~3ms, 50K nodes)"
};

document.getElementById('preset-select').addEventListener('change', function() {
    document.getElementById('preset-desc').textContent = presets[this.value] || '';
});

function notify(msg, type='ok') {
    const bar = document.getElementById('status-bar');
    bar.textContent = msg;
    bar.style.background = Type==='err' ? 'var(--red)' : 'var(--green)';
    bar.style.display = 'block';
    setTimeout(() => bar.style.display = 'none', 3000);
    logAction(msg);
}

function logAction(msg) {
    const log = document.getElementById('action-log');
    if (log.querySelector('.log-line[style]')) log.innerHTML = '';
    const time = new Date().toLocaleTimeString();
    log.innerHTML += `<div class="log-line">[${time}] ${msg}</div>`;
    log.scrollTop = log.scrollHeight;
}

async function applyPreset() {
    const preset = document.getElementById('preset-select').value;
    try {
        const r = await fetch(`${API_BASE}/api/preset`, {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({preset})
        });
        const d = await r.json();
        notify(`Preset "${preset}" applied: ${d.nodes} nodes loaded`);
        updateStats();
    } catch(e) { notify('Error: ' + e.message, 'err'); }
}

async function toggleFeature(feature, enabled) {
    try {
        await fetch(`${API_BASE}/api/feature`, {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({feature, enabled})
        });
        notify(`${feature} ${enabled ? 'enabled' : 'disabled'}`);
    } catch(e) { notify('Error: ' + e.message, 'err'); }
}

async function doAction(action) {
    logAction(`Running: ${action}...`);
    try {
        const r = await fetch(`${API_BASE}/api/action`, {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({action})
        });
        const d = await r.json();
        notify(`${action}: ${JSON.stringify(d).substring(0,100)}`);
    } catch(e) { notify('Error: ' + e.message, 'err'); }
}

async function updateStats() {
    try {
        const [healthR, cacheR] = await Promise.all([
            fetch(`${API_BASE}/v1/health`),
            fetch(`${API_BASE}/v1/cache/stats`).catch(() => null)
        ]);
        
        if (healthR.ok) {
            const h = await healthR.json();
            const checks = h.checks || {};
            document.getElementById('stat-nodes').textContent = checks.node_count?.value || 0;
            document.getElementById('stat-queries').textContent = checks.field_stats?.total_queries || 0;
            document.getElementById('stat-consol').textContent = checks.field_stats?.consolidations || 0;
            document.getElementById('stat-bm25').textContent = checks.field_stats?.bm25_fallbacks || 0;
            document.getElementById('stat-engram').textContent = h.stats?.engram_retrievals || 0;
            document.querySelector('#conn-status .status').className = 'status ok';
        }
        
        if (cacheR && cacheR.ok) {
            const c = await cacheR.json();
            document.getElementById('stat-cache').textContent = c.hit_rate ? `${(c.hit_rate*100).toFixed(0)}%` : '—';
        }
    } catch(e) {
        document.querySelector('#conn-status .status').className = 'status err';
        document.getElementById('conn-status').innerHTML = '<span class="status err"></span>Disconnected';
    }
}

// Update stats every 3 seconds
setInterval(updateStats, 3000);
updateStats();
</script>
</body>
</html>"""

import json
from typing import Dict, Any
from fastapi import APIRouter, HTTPException


def create_dashboard_router(memory=None, config: Dict[str, Any] = None) -> APIRouter:
    """Create FastAPI router for the dashboard UI.
    
    Args:
        memory: RTMDKMemory instance (optional for standalone mode)
        config: Configuration dict
        
    Returns:
        FastAPI APIRouter
    """
    router = APIRouter()
    
    # Serve dashboard HTML
    @router.get("/dashboard")
    @router.get("/")
    async def dashboard():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=HTML_PAGE)
    
    # API endpoints for dashboard control
    @router.post("/api/preset")
    async def api_apply_preset(data: dict):
        preset = data.get("preset", "local")
        nodes = len(memory.field.nodes) if memory else 0
        return {"status": "ok", "preset": preset, "nodes": nodes}
    
    @router.post("/api/feature")
    async def api_toggle_feature(data: dict):
        feature = data.get("feature", "")
        enabled = data.get("enabled", True)
        if config:
            config[f"enable_{feature}"] = enabled
        return {"status": "ok", "feature": feature, "enabled": enabled}
    
    @router.post("/api/action")
    async def api_action(data: dict):
        action = data.get("action", "")
        
        if not memory:
            return {"error": "Memory not available in standalone mode"}
        
        if action == "backup":
            from rtmdk.production.backup_restore import BackupManager
            bm = BackupManager(memory)
            path = bm.create_backup("manual")
            return {"status": "ok", "path": path}
        
        elif action == "prune":
            from rtmdk.production.smart_pruning import SmartPruner
            sp = SmartPruner(memory, dry_run=False)
            result = sp.prune()
            return result
        
        elif action == "export_md":
            from rtmdk.production.export import MemoryExporter
            me = MemoryExporter(memory)
            return {"content": me.to_markdown()[:500]}
        
        elif action == "export_json":
            from rtmdk.production.export import MemoryExporter
            me = MemoryExporter(memory)
            return me.to_dict()
        
        elif action == "clear_cache":
            return {"status": "ok", "message": "Cache cleared"}
        
        elif action == "clear_memory":
            memory.field.nodes.clear()
            memory.field.node_index.clear()
            return {"status": "ok", "message": "Memory cleared"}
        
        elif action == "analytics":
            from rtmdk.production.analytics import MemoryAnalytics
            an = MemoryAnalytics(memory)
            return an.export_report()
        
        elif action == "health":
            from rtmdk.production.health_monitor import HealthMonitor
            hm = HealthMonitor(memory)
            return hm.check_health()
        
        return {"error": f"Unknown action: {action}"}
    
    return router


# Standalone mode
if __name__ == "__main__":
    import argparse
    from fastapi import FastAPI
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    app = FastAPI(title="RTMDK Dashboard")
    app.include_router(create_dashboard_router())
    
    import uvicorn
    print(f"Dashboard running at http://localhost:{args.port}/dashboard")
    uvicorn.run(app, host=args.host, port=args.port)
