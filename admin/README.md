# RTMDK Admin Panel

React-based visual administration interface for RTMDK v8.3.1.

## Architecture

```
Browser → Vite dev :5173 (dev only)
        → Express server.cjs :3000  →  RTMDK server :8081
           ├─ serves dist/ (prod)
           ├─ /api/rtmdk/*  — proxy to RTMDK API
           ├─ /api/server/* — launcher (start/stop/status RTMDK, SSE log stream)
           └─ config: ~/.rtmdk/admin-config.json
```

## Features

- **Dashboard** — health status, node count, version, uptime, deep-health checks
- **Memory Nodes** — paginated table with CRUD, salience/phase/amplitude
- **Query** — interactive memory search (simple + pipeline modes)
- **SOT** — Self-Organizing Tokenizer status, vocabulary inspection, bootstrap
- **Analytics** — query/pipeline analytics overview
- **Pipeline** — DAG visualization, pipeline health and metrics
- **Import/Export** — memory export, import, batch ingestion
- **Settings** — API keys, webhooks, runtime config
- **Server Control** — start/stop RTMDK server, live logs (SSE), port management
- **AI Connection** — provider setup (LM Studio / OpenAI / OpenRouter)
- **Welcome** — first-run configuration wizard

## Quick Start (development)

```bash
cd admin
npm install
npm run dev        # concurrently: node server.cjs (:3000) + vite (:5173)
```

Open `http://localhost:5173` — Vite proxies `/api` → `localhost:3000`.

## Production

```bash
npm run prod       # vite build + node server.cjs
# or, if dist/ is already built:
npm start          # node server.cjs → http://localhost:3000
```

The RTMDK server can be started externally (`python -m rtmdk`, port from `.env`, default 8081) or directly from the **Server Control** page — the launcher spawns it with env from `~/.rtmdk/admin-config.json`.

## E2E tests

```bash
node e2e-audit.mjs   # full UI audit of all pages (needs stack on :3000)
node e2e-ai.mjs      # AI Connection page flow
# Playwright specs: ../tests/e2e/ (npx playwright test)
```

## Configuration

The frontend talks to `/api/rtmdk` (see `src/hooks/use-api.js`); the API key is taken from the admin config (`env.RTMDK_API_KEY`) and sent as `X-API-Key`. Ports: `ADMIN_PORT` (default 3000), `RTMDK_PORT` (default from `.env`, 8081).
