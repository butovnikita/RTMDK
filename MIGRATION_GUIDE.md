# Migration Guide: v8.2.0 → v8.2.1

> Minimal breaking changes. Most users can upgrade seamlessly.

---

## Breaking Changes

None. v8.2.1 is backward-compatible with v8.2.0 memory files, configs, and API clients.

## New Features Available After Upgrade

| Feature | How to Enable |
|---------|--------------|
| **GraphQL API** | No config needed. Available at `POST /graphql` automatically if `strawberry-graphql` installed. |
| **WebSocket Streaming** | No config needed. Available at `WS /ws/memory` automatically. |
| **SOT Vocabulary Endpoint** | No config needed. Requires `sot_enabled=True` in config (already default in `production` preset). |
| **SOT Persistence** | No config needed. SOT state auto-saved/loaded with `export_field()` / `import_field()`. |
| **React Admin Panel** | `cd admin && npm install && npm run dev` (optional). |

## Upgrade Steps

### Python Users

```bash
# 1. Pull latest code
git pull origin main

# 2. Install new dependency (GraphQL)
pip install strawberry-graphql[fastapi]

# 3. Restart server
python rtmdk_server.py
```

### Docker Users

```bash
# 1. Pull latest image
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# 2. Verify health
curl http://localhost:8080/health
# → look for "version": "8.2.1"
```

### SillyTavern Users

No changes needed. Proxy and monolith modes work identically.

## Data Migration

### Memory files (`.json` / `.msgpack`)

v8.2.1 adds `sot_tokenizer` key to exported memory files. Old files load fine — SOT will re-bootstrap automatically if `sot_enabled=True`.

```python
# If you want to preserve SOT state from v8.2.1:
# 1. Load old file
memory = RTMDKMemory.load("memory_v820.json")

# 2. Re-export (now includes SOT state)
memory.export_field("memory_v821.json")
```

### Config files

No changes. All v8.2.0 configs work unchanged. New optional flags:

```python
RTMDKConfig(
    # ... existing config ...
    # New optional flags (all have sensible defaults):
    # sot_use_for_query=True,   # Use SOT for query embeddings
    # sot_retrieval_feedback=True,  # Enable retrieval feedback loop
)
```

## API Changes

### New Endpoints

```
POST /graphql              # GraphQL API
WS   /ws/memory            # WebSocket streaming
GET  /v1/sot/status        # SOT status
POST /v1/sot/bootstrap     # SOT bootstrap from corpus
GET  /v1/sot/vocab         # SOT vocabulary (paginated)
```

### No Removed Endpoints

All v8.2.0 endpoints remain functional.

## Verification Checklist

After upgrade, verify:

```bash
# 1. Version
curl http://localhost:8080/health | grep version
# → "8.2.1"

# 2. GraphQL
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ health { status } }"}'
# → {"data": {"health": {"status": "ok"}}}

# 3. SOT status (if production preset)
curl http://localhost:8080/v1/sot/status
# → {"enabled": true, "vocab_size": ...}

# 4. Memory load/save roundtrip still works
python -c "from rtmdk import RTMDKMemory; m = RTMDKMemory.load('memory.json'); m.save()"
```

## Rollback

If issues occur:

```bash
# 1. Stop server
# 2. Restore memory file from backup (v8.2.0 file works)
# 3. Checkout v8.2.0
git checkout v8.2.0  # or specific commit
# 4. Restart
```

## Support

- GitHub Issues: https://github.com/butovnikita/RTMDK/issues
- Docs: https://github.com/butovnikita/RTMDK/tree/main/docs
