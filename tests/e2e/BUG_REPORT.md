# RTMDK Web Interface — Bug Report

> Full functional audit conducted on 2026-05-22 via Playwright E2E testing across all 11 admin UI pages.

> **✅ РЕШЕНО (2026-08-01):** после внедрения admin panel v2 повторный полный E2E-аудит (`admin/e2e-audit.mjs`, все 11 страниц) — **0 замечаний**. Critical-баг «Backend API mismatch» устранён унификацией через `/api/rtmdk` proxy + `use-api.js`; добавлены маршруты страниц (react-router). Отчёт сохранён как исторический артефакт.

## Test Environment

| Component | Version / Port |
|-----------|---------------|
| Admin UI | React 19 + Vite 8, built from `admin/` |
| Proxy | Express `server.cjs` on port 3000 |
| Backend | `rtmdk_server.py` (FastAPI) on port 8081 |
| Browser | Chromium (Playwright 1.60.0) |
| Memory | 413 nodes loaded |

---

## Summary

| Severity | Count | Categories |
|----------|-------|-----------|
| 🔴 Critical | 1 | Backend API mismatch breaks 4+ pages |
| 🟠 Medium | 1 | Missing intuitive URL routes |
| 🟡 Low | 2 | UI inconsistency, connection leak |

**Cross-cutting results**: Zero JavaScript console errors. Zero HTTP 5xx errors. No React crashes. The UI framework itself is stable.

---

## 🔴 Critical: Backend API Mismatch (`rtmdk_server.py` vs Admin UI)

### Problem
The running backend (`rtmdk_server.py`) exposes only 17 endpoints. The Admin UI expects the full production API from `rtmdk/server/app.py` (40+ endpoints). This causes **404 Not Found** errors on multiple pages.

### Affected Pages & Symptoms

| Page | Expected Endpoint | Actual Response | User Impact |
|------|------------------|-----------------|-------------|
| **Memory Nodes** (`/nodes`) | `GET /v1/memory/nodes` | `404 Not Found` | Toast: "Failed to load nodes — Not Found". Table always empty. |
| **Memory Nodes** (`/nodes`) | `GET /v1/memory/nodes/{id}` | `404 Not Found` | Cannot view/edit/delete individual nodes. |
| **Analytics** (`/analytics`) | `GET /v1/analytics/overview` | `404 Not Found` | All metrics show 0 (queries, latency, cache hits). |
| **Pipeline** (`/pipeline`) | `GET /v1/memory/pipeline/dag` | `404 Not Found` | "Pipeline data not available" + "unknown" health badge. |
| **SOT** (`/sot`) | `GET /v1/sot/status` | `404 Not Found` | Shows "Enabled: No", "Vocab Size: 0". |
| **Settings** (`/settings`) | `GET /v1/admin/config` | `404 Not Found` | May not persist config changes to backend. |
| **AI Connection** (`/ai`) | `GET /v1/models` | Works | Model discovery works, but embedder test may fail. |
| **Query** (`/query`) | `POST /v1/memory/query` | Works | Simple query works. Pipeline query may fail. |
| **Dashboard** (`/`) | `GET /health` | Works | Health cards render correctly. |

### Missing Endpoints in `rtmdk_server.py`

```
/v1/memory/nodes                    (GET, POST)
/v1/memory/nodes/{node_id}          (GET, PUT, DELETE)
/v1/memory/batch_ingest             (POST)
/v1/memory/pipeline/dag             (GET)
/v1/memory/pipeline/health          (GET)
/v1/memory/pipeline/metrics         (GET)
/v1/memory/pipeline/plan            (GET)
/v1/memory/pipeline/stream          (GET)
/v1/analytics/overview              (GET)
/v1/analytics/memory                (GET)
/v1/analytics/events                (GET)
/v1/analytics/pipeline              (GET)
/v1/analytics/report                (GET)
/v1/admin/api-keys                  (GET, POST)
/v1/admin/audit-log                 (GET)
/v1/admin/retention                 (GET)
/v1/admin/cache                     (GET)
/v1/admin/encryption                (GET)
/v1/admin/telemetry                 (GET)
/v1/sot/status                      (GET)
/v1/sot/vocab                       (GET)
/v1/sot/bootstrap                   (POST)
/v1/replication/mutation            (POST)
/v1/replication/wal                 (GET)
```

### Reproduction
1. Start backend: `python rtmdk_server.py`
2. Start proxy: `cd admin && npm start`
3. Open `/nodes` in browser
4. Observe "Failed to load nodes" toast

### Recommended Fix
**Option A (Recommended)**: Start the production server instead:
```bash
python -m rtmdk.server.app
```
Or update the launcher to use `rtmdk.server.app:app` instead of `rtmdk_server.py`.

**Option B**: Port the missing endpoints from `rtmdk/server/app.py` into `rtmdk_server.py`.

**Option C**: Add a startup check in the admin UI that warns users when connected to a limited API backend.

---

## 🟠 Medium: Missing Intuitive URL Routes

### Problem
Users typing or bookmarking intuitive URLs based on sidebar labels get **blank white pages** because the React Router only registers short routes.

| Intuitive URL (broken) | Actual Route (works) | Sidebar Label |
|------------------------|---------------------|---------------|
| `/ai-connection` | `/ai` | AI Connection |
| `/memory-nodes` | `/nodes` | Memory Nodes |
| `/server-control` | `/server` | Server Control |

### Root Cause
`App.jsx` inner `<Routes>` has no catch-all fallback. When `path="/*"` matches the outer route but the inner route doesn't match any defined path, nothing renders inside `<Layout>` — resulting in a blank white main content area.

### Reproduction
1. Navigate directly to `http://localhost:3000/ai-connection`
2. Page shows sidebar but blank white main area

### Recommended Fix
Add redirect routes or aliases in `App.jsx`:

```jsx
<Route path="/ai-connection" element={<Navigate to="/ai" replace />} />
<Route path="/memory-nodes" element={<Navigate to="/nodes" replace />} />
<Route path="/server-control" element={<Navigate to="/server" replace />} />
```

Or add a catch-all fallback that redirects to Dashboard:
```jsx
<Route path="*" element={<Navigate to="/" replace />} />
```

---

## 🟡 Low: AI Connection Provider / URL Mismatch

### Problem
The AI Connection page can display a **selected provider that doesn't match the endpoint URL**.

**Screenshot evidence**: Provider shows "OpenAI" selected (with checkmark) but Endpoint URL displays `https://openrouter.ai/api/v1`.

### Impact
Users may accidentally save mismatched configuration (e.g., sending OpenAI API key to OpenRouter endpoint).

### Recommended Fix
When the provider selection changes, automatically update the Endpoint URL to the default for that provider:
- LM Studio → `http://localhost:12345/v1`
- OpenAI → `https://api.openai.com/v1`
- OpenRouter → `https://openrouter.ai/api/v1`

Also validate on save that the URL domain matches the selected provider.

---

## 🟡 Low: Server Control EventSource Never Closes

### Problem
The Server Control page opens an `EventSource("/api/server/logs")` connection on mount. This connection is **never explicitly closed** when the component unmounts or when navigating away.

### Impact
1. Browser keeps a persistent SSE connection open indefinitely.
2. Prevents `networkidle` state in automation/testing.
3. Potential memory leak if user navigates back and forth multiple times (multiple EventSource instances).

### Code Location
`admin/src/pages/server-control.jsx` — `connect()` function creates `EventSource` but cleanup only happens in `useEffect` return if `esRef.current` is set at the exact moment of unmount.

### Recommended Fix
Ensure EventSource is closed on unmount:

```jsx
useEffect(() => {
  connect()
  return () => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }
}, [])
```

Also close the old connection before opening a new one in `connect()`.

---

## Pages Verified Working

| Page | Route | Status |
|------|-------|--------|
| Dashboard | `/` | ✅ Fully functional |
| Query | `/query` | ✅ Simple query works |
| Settings | `/settings` | ✅ Renders and edits work |
| Import/Export | `/import-export` | ✅ Renders correctly |
| SOT | `/sot` | ⚠️ Renders but data is 0 (API missing) |
| Pipeline | `/pipeline` | ⚠️ Renders but no data (API missing) |
| Analytics | `/analytics` | ⚠️ Renders but all metrics 0 (API missing) |
| Server Control | `/server` | ⚠️ Renders, shows ON badge, logs blank |
| AI Connection | `/ai` | ⚠️ Renders, provider/URL mismatch |
| Memory Nodes | `/nodes` | ❌ Broken (404 from backend) |

---

## Test Artifacts

- Screenshots: `tests/e2e/test-results/*/test-failed-1.png` and `test-finished-1.png`
- Videos: `tests/e2e/test-results/*/video.webm`
- Traces: `tests/e2e/test-results/*/trace.zip`
- HTML Report: `tests/e2e/playwright-report/`

---

*Report generated by Playwright E2E audit on 2026-05-22*
