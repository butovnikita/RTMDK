# RTMDK Security Checklist

> Version: 8.2.1 | Last updated: 2026-05-07
> Status: Production-ready for single-node deployment

---

## 1. Authentication & Authorization

| Check | Status | Notes |
|-------|:------:|-------|
| API key validation on all endpoints | ✅ | `ENABLE_API_AUTH` (default: true) |
| Tenant isolation | ✅ | `TenantRateLimiter` per `X-API-Key` |
| Admin endpoints require admin key | ✅ | `/v1/admin/*` gated by `api_key_manager` |
| Key revocation | ✅ | `/v1/admin/api-keys/revoke` |
| Key expiration | ✅ | TTL support in `APIKeyManager` |

## 2. Input Validation

| Check | Status | Notes |
|-------|:------:|-------|
| Path traversal protection | ✅ | `safe_join` in export/import paths |
| JSON size limits | ✅ | `MAX_PAYLOAD_SIZE` (default 1MB) |
| Rate limiting | ✅ | Per-tenant + global rate limiters |
| Query parameter bounds | ✅ | `top_k` clamped to `[1, 50]`, `threshold ≥ 0` |
| Content security validation | ✅ | `SecurityValidator` in `add_node()` |
| SQL injection (N/A) | ✅ | No SQL database — in-memory + JSON |

## 3. Data Protection

| Check | Status | Notes |
|-------|:------:|-------|
| Encryption at rest | ✅ | AES-256-GCM via `EncryptionManager` (opt-in) |
| Encryption key from env | ✅ | `RTMDK_ENCRYPTION_KEY` (base64) |
| Memory file permissions | ⚠️ | User must set `chmod 600 ~/.rtmdk/memory.json` |
| Audit logging | ✅ | All admin actions logged to `audit_log` |
| PII scrubbing | ⚠️ | No automatic PII detection — user responsibility |

## 4. Network Security

| Check | Status | Notes |
|-------|:------:|-------|
| CORS middleware | ✅ | `ALLOWED_ORIGINS` configurable |
| Request ID tracing | ✅ | Every request gets unique `X-Request-ID` |
| Graceful shutdown | ✅ | Drain in-flight requests before exit |
| HTTP client timeouts | ✅ | `RTMDK_LM_STUDIO_TIMEOUT` (default 30s) |
| gRPC TLS | ⚠️ | Not implemented — use reverse proxy (nginx/traefik) |

## 5. Operational Security

| Check | Status | Notes |
|-------|:------:|-------|
| Health probes (liveness/readiness) | ✅ | `/health` + deep checks |
| Circuit breakers | ✅ | LM Studio + SOT bootstrap |
| Automatic retry with backoff | ✅ | `AsyncCircuitBreaker` recovery |
| Resource limits | ✅ | `max_nodes`, `max_vocab`, rate limits |
| Log sanitization | ⚠️ | No automatic secret redaction |

## 6. Known Limitations

| Risk | Mitigation | Priority |
|------|-----------|:--------:|
| No built-in HTTPS | Use reverse proxy (nginx, traefik, Caddy) | Medium |
| No RBAC (roles beyond admin/tenant) | Implement at proxy layer or fork | Low |
| No request signing | API keys only | Low |
| gRPC without TLS | Use mTLS at service mesh level | Low |
| WebSocket without auth | Same-origin policy + reverse proxy | Low |

## 7. Deployment Hardening Checklist

Before exposing RTMDK to the internet:

```bash
# 1. Enable encryption
export RTMDK_ENCRYPTION_KEY=$(openssl rand -base64 32)

# 2. Set strong API key (not default)
export RTMDK_API_KEY=$(openssl rand -hex 16)

# 3. Restrict CORS
export RTMDK_ALLOWED_ORIGINS=https://your-domain.com

# 4. Enable auth
export RTMDK_ENABLE_API_AUTH=true

# 5. Lock memory file permissions
chmod 600 ~/.rtmdk/memory.json

# 6. Run behind reverse proxy with HTTPS
# nginx/traefik → http://localhost:8080

# 7. Enable audit logging
export RTMDK_AUDIT_LOG_PATH=/var/log/rtmdk/audit.log
```

## 8. Vulnerability Reporting

Found a security issue? Please report privately:
- Email: security@rtmdk.dev (placeholder)
- Do NOT open public GitHub issues for security bugs
