# RTMDK Admin Panel

React-based visual administration interface for RTMDK v8.3.0.

## Features

- **Dashboard** — Health status, node count, version, uptime
- **Memory Nodes** — Paginated table view of all memory nodes with salience/phase/amplitude
- **Query Interface** — Interactive memory search with live results
- **SOT Panel** — Self-Organizing Tokenizer status and vocabulary inspection

## Quick Start

```bash
cd admin
npm install
npm run dev
```

Then open `http://localhost:5173` (or the URL shown in terminal).

Make sure the RTMDK server is running on `http://localhost:8080`.

## Build for Production

```bash
npm run build
```

Static files will be in `dist/`. Serve them with any static file server.

## Configuration

The API base URL is hardcoded to `http://localhost:8080` in `src/App.jsx`. Change it if your server runs on a different host/port.
