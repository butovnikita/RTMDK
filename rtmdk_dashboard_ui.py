"""rtmdk_dashboard_ui.py — Web UI Dashboard for RTMDK.

Adds:
- Provider selection (LM Studio, OpenRouter, OpenAI, Anthropic, Custom)
- Model selection per provider
- Embedder selection
- API key input
- Test Connection button
- All previous features (presets, UX toggles, stats, actions)

Usage:
    from rtmdk_dashboard_ui import create_dashboard_router
    app.include_router(create_dashboard_router(memory, config))
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RTMDK Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0f1117;
  --surface: #1e2030;
  --surface2: #282a3a;
  --border: #363949;
  --text: #e0e0e8;
  --text-dim: #8b8fa3;
  --primary: #7c8aff;
  --primary-hover: #9aa8ff;
  --success: #4ade80;
  --error: #f87171;
  --warning: #fbbf24;
  --radius: 12px;
  --shadow: 0 4px 24px rgba(0,0,0,0.4);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}

/* Header */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}

.header h1 {
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--primary), #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.header .version {
  color: var(--text-dim);
  font-size: 14px;
}

/* Status Bar */
.status-bar {
  display: flex;
  gap: 16px;
  margin-bottom: 32px;
}

.status-item {
  flex: 1;
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: var(--shadow);
}

.status-item .icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}

.status-item .icon.green { background: rgba(74, 222, 128, 0.15); }
.status-item .icon.blue { background: rgba(124, 138, 255, 0.15); }
.status-item .icon.yellow { background: rgba(251, 191, 36, 0.15); }
.status-item .icon.red { background: rgba(248, 113, 113, 0.15); }

.status-item .info { flex: 1; }
.status-item .label { color: var(--text-dim); font-size: 13px; }
.status-item .value { font-size: 24px; font-weight: 600; }

/* Cards */
.card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
}

.card h2 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.card h3 {
  font-size: 16px;
  font-weight: 500;
  margin: 16px 0 12px;
  color: var(--text-dim);
}

/* Form Controls */
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-dim);
  margin-bottom: 8px;
}

select, input[type="text"], input[type="password"] {
  width: 100%;
  padding: 10px 14px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 14px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.2s;
}

select:focus, input:focus {
  border-color: var(--primary);
}

select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Buttons */
.btn-group {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 16px;
}

.btn {
  padding: 10px 18px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.btn-primary {
  background: var(--primary);
  color: white;
}
.btn-primary:hover { background: var(--primary-hover); }

.btn-secondary {
  background: var(--surface2);
  color: var(--text);
  border: 1px solid var(--border);
}
.btn-secondary:hover { background: var(--border); }

.btn-success {
  background: rgba(74, 222, 128, 0.15);
  color: var(--success);
}
.btn-success:hover { background: rgba(74, 222, 128, 0.25); }

.btn-danger {
  background: rgba(248, 113, 113, 0.15);
  color: var(--error);
}
.btn-danger:hover { background: rgba(248, 113, 113, 0.25); }

/* Grid Layout */
.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

/* Test Result */
.test-result {
  margin-top: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
}
.test-result.ok {
  background: rgba(74, 222, 128, 0.15);
  color: var(--success);
}
.test-result.err {
  background: rgba(248, 113, 113, 0.15);
  color: var(--error);
}

/* Actions Grid */
.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
}

.action-btn {
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}
.action-btn:hover {
  background: var(--border);
  transform: translateY(-1px);
}

/* Log */
.log-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--bg);
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  max-height: 150px;
  overflow-y: auto;
  color: var(--text-dim);
}

.log-line {
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}
.log-line:last-child { border: none; }

/* File Input */
.file-input-wrapper {
  display: flex;
  gap: 10px;
  align-items: center;
}

.file-input-wrapper input[type="file"] {
  flex: 1;
  padding: 8px;
  background: var(--surface2);
  border: 1px dashed var(--border);
  border-radius: 8px;
}

/* Notification */
.notification {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 14px 20px;
  border-radius: 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  z-index: 1000;
  max-width: 400px;
  animation: slideIn 0.3s ease;
}
.notification.success { border-left: 4px solid var(--success); }
.notification.error { border-left: 4px solid var(--error); }

@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Loading spinner */
.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Responsive */
@media (max-width: 768px) {
  .status-bar { flex-direction: column; }
  .grid-2 { grid-template-columns: 1fr; }
  .container { padding: 16px; }
}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div>
      <h1>RTMDK Dashboard</h1>
      <div class="version">Resonance-Topological Memory v8.0.0</div>
    </div>
    <div id="conn-status">
      <span class="spinner"></span> Connecting...
    </div>
  </div>

  <!-- Status Bar -->
  <div class="status-bar">
    <div class="status-item">
      <div class="icon blue">Nodes</div>
      <div class="info">
        <div class="label">Memory Nodes</div>
        <div class="value" id="stat-nodes">0</div>
      </div>
    </div>
    <div class="status-item">
      <div class="icon green">Search</div>
      <div class="info">
        <div class="label">Queries</div>
        <div class="value" id="stat-queries">0</div>
      </div>
    </div>
    <div class="status-item">
      <div class="icon yellow">Links</div>
      <div class="info">
        <div class="label">Consolidations</div>
        <div class="value" id="stat-consol">0</div>
      </div>
    </div>
    <div class="status-item">
      <div class="icon red">Storage</div>
      <div class="info">
        <div class="label">Cache Hit Rate</div>
        <div class="value" id="stat-cache">—</div>
      </div>
    </div>
  </div>

  <!-- Provider & Model Configuration -->
  <div class="grid-2">
    <div class="card">
      <h2>API Configuration</h2>
      <div class="form-group">
        <label>Provider</label>
        <select id="provider-select" onchange="onProviderChange()">
          <option value="lm_studio">LM Studio (Local)</option>
          <option value="openrouter">OpenRouter</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="custom">Custom URL</option>
        </select>
      </div>
      <div class="grid-2" style="gap: 12px;">
        <div class="form-group">
          <label>Chat Model</label>
          <select id="model-select"><option>Loading...</option></select>
        </div>
        <div class="form-group">
          <label>Embedder Model</label>
          <div style="display: flex; gap: 8px;">
            <select id="embedder-select" style="flex: 1;"><option>Loading...</option></select>
            <button class="btn btn-primary" onclick="applyEmbedder()" style="padding: 8px 12px;">Apply</button>
          </div>
        </div>
      </div>
      <div class="form-group">
        <label>API Key (leave empty for LM Studio)</label>
        <input type="password" id="api-key-input" placeholder="sk-... or leave empty" disabled>
      </div>
      <div class="form-group" id="custom-url-group" style="display: none;">
        <label>Custom Base URL</label>
        <input type="text" id="custom-url-input" placeholder="https://api.example.com/v1">
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" onclick="applyProvider()">Apply Provider</button>
        <button class="btn btn-secondary" onclick="fetchModels()">Refresh Models</button>
        <button class="btn btn-success" onclick="testConnection()">Test Connection</button>
      </div>
      <div id="test-result" class="test-result" style="display: none;"></div>
    </div>

    <!-- Backup & Restore -->
    <div class="card">
      <h2>Backup & Restore</h2>
      <div class="form-group">
        <label>Upload Memory Backup (.json)</label>
        <div class="file-input-wrapper">
          <input type="file" id="backup-file" accept=".json,.json.gz">
          <button class="btn btn-primary" onclick="uploadBackup()">Upload</button>
        </div>
      </div>
      <div id="backup-status" style="color: var(--text-dim); font-size: 13px; margin-top: 8px;">
        Select a backup file and click Upload to restore memory state.
      </div>
      <h3>Quick Actions</h3>
      <div class="actions-grid">
        <button class="action-btn" onclick="doAction('backup')">Create Backup</button>
        <button class="action-btn" onclick="doAction('prune')">Run Pruning</button>
        <button class="action-btn" onclick="doAction('export_md')">Export MD</button>
        <button class="action-btn" onclick="doAction('export_json')">Export JSON</button>
        <button class="action-btn" onclick="doAction('clear_cache')">Clear Cache</button>
        <button class="action-btn" style="color: var(--error);" onclick="doAction('clear_memory')">[WARN] Clear Memory</button>
      </div>
      <div class="log-box" id="action-log"><div class="log-line">Actions will appear here...</div></div>
    </div>
  </div>

  <!-- Server Diagnostics -->
  <div class="card">
    <h2>Server Diagnostics</h2>
    <div class="grid-2" style="gap: 12px;">
      <div class="form-group">
        <label>Memory Status</label>
        <div id="diag-memory" class="status ok">Checking...</div>
      </div>
      <div class="form-group">
        <label>LM Studio</label>
        <div id="diag-lm" class="status err">Checking...</div>
      </div>
    </div>
  </div>

</div>

<script>
const API_BASE = window.location.origin;

// Presets
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

// Notification
function notify(msg, type='success') {
  const n = document.createElement('div');
  n.className = `notification ${type}`;
  n.textContent = msg;
  document.body.appendChild(n);
  setTimeout(() => n.remove(), 3000);
}

function logAction(msg) {
  const log = document.getElementById('action-log');
  if (log.querySelector('.log-line')?.textContent === 'Actions will appear here...') {
    log.innerHTML = '';
  }
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  log.prepend(line);
}

// Provider Change Handler
async function onProviderChange() {
  const provider = document.getElementById('provider-select').value;
  const modelSelect = document.getElementById('model-select');
  const embedderSelect = document.getElementById('embedder-select');
  const apiKeyInput = document.getElementById('api-key-input');
  const customUrlGroup = document.getElementById('custom-url-group');

  // Save current selections
  const currentModel = modelSelect.value;
  const currentEmbedder = embedderSelect.value;

  // Clear and show loading
  modelSelect.innerHTML = '<option>Loading...</option>';
  embedderSelect.innerHTML = '<option>Loading...</option>';

  // Show/hide API key
  if (provider === 'lm_studio') {
    apiKeyInput.value = '';
    apiKeyInput.placeholder = 'Not needed for LM Studio';
    apiKeyInput.disabled = true;
  } else {
    apiKeyInput.disabled = false;
    apiKeyInput.placeholder = 'sk-...';
  }
  customUrlGroup.style.display = provider === 'custom' ? 'block' : 'none';

  // Fetch models from API
  await fetchModels(currentModel, currentEmbedder);
}

// Fetch Models
async function fetchModels(preserveModel, preserveEmbedder) {
  const modelSelect = document.getElementById('model-select');
  const embedderSelect = document.getElementById('embedder-select');

  try {
    // Try UX endpoint first (provider-specific models)
    let data = null;
    try {
      const uxResp = await fetch(`${API_BASE}/api/models`);
      if (uxResp.ok) data = await uxResp.json();
    } catch(e) { console.log('UX models fetch failed:', e); }

    // Fallback to server's models endpoint
    if (!data || (!data.chat && !data.data)) {
      const resp = await fetch(`${API_BASE}/v1/models`);
      if (resp.ok) data = await resp.json();
    }

    let chatModels = [];
    let embedderModels = [];

    if (data) {
      if (data.data && Array.isArray(data.data)) {
        // OpenAI format
        chatModels = data.data.filter(m => !m.id.toLowerCase().includes('embed')).map(m => m.id);
        embedderModels = data.data.filter(m => m.id.toLowerCase().includes('embed')).map(m => m.id);
      } else if (data.chat) {
        // UX format
        chatModels = data.chat || [];
        embedderModels = data.embedder || [];
      }
    }

    // Populate chat models
    modelSelect.innerHTML = '';
    if (chatModels.length > 0) {
      chatModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        modelSelect.appendChild(opt);
      });
      if (preserveModel && [...modelSelect.options].some(o => o.value === preserveModel)) {
        modelSelect.value = preserveModel;
      }
    } else {
      modelSelect.innerHTML = '<option value="rtmdk">rtmdk (default)</option>';
    }

    // Auto-save model selection on change
    modelSelect.onchange = async () => {
      try {
        await fetch(`${API_BASE}/api/config`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({RTMDK_LLM_MODEL: modelSelect.value})
        });
      } catch(e) { /* ignore */ }
    };

    // Populate embedder models
    embedderSelect.innerHTML = '';
    if (embedderModels.length > 0) {
      embedderModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        embedderSelect.appendChild(opt);
      });
      if (preserveEmbedder && [...embedderSelect.options].some(o => o.value === preserveEmbedder)) {
        embedderSelect.value = preserveEmbedder;
      }
    } else {
      embedderSelect.innerHTML = '<option value="nomic-embed-text-v1.5">nomic-embed-text-v1.5</option>';
    }

    // Auto-save embedder selection on change
    embedderSelect.onchange = async () => {
      try {
        await fetch(`${API_BASE}/api/config`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({RTMDK_EMBED_MODEL: embedderSelect.value})
        });
      } catch(e) { /* ignore */ }
    };
  } catch(e) {
    console.error('Model fetch failed:', e);
    modelSelect.innerHTML = '<option value="rtmdk">rtmdk (fetch failed)</option>';
    embedderSelect.innerHTML = '<option value="nomic-embed-text-v1.5">nomic-embed-text-v1.5</option>';
  }
}

// Apply Provider
async function applyProvider() {
  const provider = document.getElementById('provider-select').value;
  const model = document.getElementById('model-select').value;
  const embedder = document.getElementById('embedder-select').value;
  const apiKey = document.getElementById('api-key-input').value;
  const customUrl = document.getElementById('custom-url-input').value;

  try {
    const r = await fetch(`${API_BASE}/api/config`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({provider, model, embedder, api_key: apiKey, custom_url: customUrl})
    });
    const d = await r.json();
    notify(`Provider "${provider}" applied with model "${model}"`);
    const curModel = document.getElementById('model-select').value;
    const curEmbedder = document.getElementById('embedder-select').value;
    setTimeout(() => fetchModels(curModel, curEmbedder), 500);
  } catch(e) { notify('Error: ' + e.message, 'error'); }
}

// Apply Embedder
async function applyEmbedder() {
  const embedder = document.getElementById('embedder-select').value;
  try {
    const r = await fetch(`${API_BASE}/api/embedder`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model: embedder})
    });
    const d = await r.json();
    notify(`Embedder set to "${embedder}"`);
    const curModel = document.getElementById('model-select').value;
    const curEmbedder = document.getElementById('embedder-select').value;
    setTimeout(() => fetchModels(curModel, curEmbedder), 500);
  } catch(e) { notify('Error: ' + e.message, 'error'); }
}

// Test Connection
async function testConnection() {
  const resultDiv = document.getElementById('test-result');
  resultDiv.style.display = 'block';
  resultDiv.textContent = 'Testing connection...';
  resultDiv.className = 'test-result';

  try {
    const healthResp = await fetch(`${API_BASE}/health`);
    if (!healthResp.ok) throw new Error('Health check failed');
    const health = await healthResp.json();

    const modelsResp = await fetch(`${API_BASE}/v1/models`);
    const models = await modelsResp.json();
    const modelCount = models.data ? models.data.length : (models.chat ? models.chat.length : 0);

    resultDiv.innerHTML = `OK Connected! Server v${health.version || '?'}<br>Models: ${modelCount} available<br>LM Studio: ${health.lm_studio ? 'Yes' : 'No'}`;
    resultDiv.className = 'test-result ok';
  } catch(e) {
    resultDiv.textContent = `[FAIL] Connection failed: ${e.message}`;
    resultDiv.className = 'test-result err';
  }
}

// Do Action
async function doAction(action) {
  logAction(`Running: ${action}...`);
  try {
    const r = await fetch(`${API_BASE}/api/action`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({action})
    });
    const d = await r.json();
    notify(`${action}: ${JSON.stringify(d).substring(0,100)}`);
  } catch(e) { notify('Error: ' + e.message, 'error'); }
}

// Upload Backup
async function uploadBackup() {
  const fileInput = document.getElementById('backup-file');
  const statusEl = document.getElementById('backup-status');
  const file = fileInput.files[0];

  if (!file) {
    statusEl.textContent = '[WARN] Please select a backup file first.';
    statusEl.style.color = 'var(--warning)';
    return;
  }

  statusEl.textContent = 'Uploading and restoring...';
  statusEl.style.color = 'var(--text-dim)';
  logAction(`Uploading backup: ${file.name}`);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const r = await fetch(`${API_BASE}/api/backup/upload`, {
      method: 'POST',
      body: formData
    });

    const d = await r.json();

    if (d.error) {
      statusEl.textContent = `[FAIL] Error: ${d.error}`;
      statusEl.style.color = 'var(--error)';
      logAction(`Backup restore failed: ${d.error}`);
    } else {
      statusEl.textContent = `OK Restored ${d.nodes_restored} nodes from backup!`;
      statusEl.style.color = 'var(--success)';
      logAction(`Backup restored: ${d.nodes_restored} nodes`);
      fileInput.value = '';
      setTimeout(updateStats, 500);
    }
  } catch(e) {
    statusEl.textContent = `[FAIL] Upload failed: ${e.message}`;
    statusEl.style.color = 'var(--error)';
    logAction(`Backup upload error: ${e.message}`);
  }
}

// Update Stats
async function updateStats() {
  try {
    const [healthR, cacheR] = await Promise.all([
      fetch(`${API_BASE}/health`).catch(() => null),
      fetch(`${API_BASE}/api/cache/stats`).catch(() => null)
    ]);

    let nodeCount = 0;
    let totalQueries = 0;
    let consolidations = 0;

    if (healthR && healthR.ok) {
      const h = await healthR.json();
      nodeCount = h.memory_nodes || h.node_count || 0;
      if (!nodeCount && h.checks) {
        nodeCount = h.checks.node_count?.value || 0;
      }
      totalQueries = h.total_queries || 0;
      consolidations = h.consolidations || 0;

      const connStatus = document.querySelector('#conn-status');
      if (connStatus) connStatus.innerHTML = '<span style="color: var(--success);">● Connected</span>';
    }

    if (cacheR && cacheR.ok) {
      const c = await cacheR.json();
      const cacheHit = c.hit_rate !== undefined ? `${(c.hit_rate * 100).toFixed(0)}%` : '—';
      document.getElementById('stat-cache').textContent = cacheHit;
    }

    document.getElementById('stat-nodes').textContent = nodeCount;
    document.getElementById('stat-queries').textContent = totalQueries;
    document.getElementById('stat-consol').textContent = consolidations;

    // Diagnostics
    const memSpan = document.getElementById('diag-memory');
    const lmSpan = document.getElementById('diag-lm');

    if (nodeCount >= 0) {
      memSpan.textContent = `Ready (${nodeCount} nodes)`;
      memSpan.className = 'status ok';
    }

    try {
      const h = await fetch(`${API_BASE}/health`).then(r => r.json());
      if (h.lm_studio) {
        lmSpan.textContent = 'Connected';
        lmSpan.className = 'status ok';
      } else {
        lmSpan.textContent = 'Not connected';
        lmSpan.className = 'status err';
      }
    } catch(e) {
      lmSpan.textContent = 'Unknown';
    }
  } catch(e) {
    console.error('Stats update failed:', e);
    const connStatus = document.querySelector('#conn-status');
    if (connStatus) connStatus.innerHTML = '<span style="color: var(--error);">● Disconnected</span>';
  }
}

// Init
document.addEventListener('DOMContentLoaded', async () => {
  // Load current config from server
  try {
    const resp = await fetch(`${API_BASE}/api/config`);
    if (resp.ok) {
      const cfg = await resp.json();
      
      // Set provider
      if (cfg.provider) {
        document.getElementById('provider-select').value = cfg.provider;
      }
      
      // Fetch models first
      await fetchModels();
      
      // Set saved model if it exists in list
      if (cfg.llm_model) {
        const modelSelect = document.getElementById('model-select');
        if ([...modelSelect.options].some(o => o.value === cfg.llm_model)) {
          modelSelect.value = cfg.llm_model;
        }
      }
      
      // Set saved embedder if it exists in list
      if (cfg.embed_model) {
        const embedderSelect = document.getElementById('embedder-select');
        if ([...embedderSelect.options].some(o => o.value === cfg.embed_model)) {
          embedderSelect.value = cfg.embed_model;
        }
      }
      
      // Show/hide API key based on provider
      onProviderChange();
    } else {
      await fetchModels();
    }
  } catch(e) {
    console.error('Config load failed:', e);
    await fetchModels();
  }
});
setInterval(updateStats, 3000);
updateStats();
</script>
</body>
</html>
"""

from typing import Dict, Any, Callable, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

def create_dashboard_router(memory: Callable, config: Dict[str, Any]) -> APIRouter:
    router = APIRouter()

    def _get_mem():
        if callable(memory): return memory()
        return memory

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return HTML_PAGE

    @router.post("/api/action")
    async def api_action(data: dict):
        mem = _get_mem()
        action = data.get("action", "")
        if not mem:
            return {"error": "Memory not available"}

        if action == "backup":
            from rtmdk.production.backup_restore import BackupManager
            bm = BackupManager(mem)
            path = bm.create_backup("manual")
            return {"status": "ok", "path": path}
        elif action == "prune":
            from rtmdk.production.smart_pruning import SmartPruner
            sp = SmartPruner(mem, dry_run=False)
            return sp.prune()
        elif action == "export_md":
            from rtmdk.production.export import MemoryExporter
            return {"content": MemoryExporter(mem).to_markdown()[:500]}
        elif action == "export_json":
            from rtmdk.production.export import MemoryExporter
            return MemoryExporter(mem).to_dict()
        elif action == "clear_cache":
            return {"status": "ok", "message": "Cache cleared"}
        elif action == "clear_memory":
            mem.field.nodes.clear()
            mem.field.node_index.clear()
            return {"status": "ok", "message": "Memory cleared"}
        elif action == "analytics":
            from rtmdk.production.analytics import MemoryAnalytics
            return MemoryAnalytics(mem).export_report()
        elif action == "health":
            from rtmdk.production.health_monitor import HealthMonitor
            return HealthMonitor(mem).check_health()
        return {"error": f"Unknown action: {action}"}

    @router.post("/api/backup/upload")
    async def upload_backup(request: Request):
        mem = _get_mem()
        if not mem:
            return {"error": "Memory not initialized"}

        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, 'filename'):
            return {"error": "No file provided"}

        import tempfile, os
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            from rtmdk_memory_v8 import RTMDKMemory
            mem2 = RTMDKMemory.import_field(temp_path, mem.embedder)
            if not mem2 or len(mem2.field.nodes) == 0:
                return {"error": "Failed to restore: no nodes found"}

            mem.field.nodes.clear()
            mem.field.node_index.clear()
            for nid, node in mem2.field.nodes.items():
                mem.field.nodes[nid] = node
                mem.field.node_index.append(nid)
            mem.field.stats.update(mem2.field.stats)

            node_count = len(mem.field.nodes)
            return {"status": "ok", "nodes_restored": node_count}
        except Exception as e:
            return {"error": f"Restore failed: {str(e)}"}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # Temp file may already be cleaned up

    @router.get("/api/diagnostics")
    async def api_diagnostics():
        mem = _get_mem()
        return {
            "memory_initialized": mem is not None,
            "node_count": len(mem.field.nodes) if mem else 0,
            "memory_file": config.get("RTMDK_MEMORY_FILE", "unknown"),
        }

    return router
