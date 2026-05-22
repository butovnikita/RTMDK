# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 8.3.x   | ✅ Yes |
| 8.2.x   | ⚠️ Security fixes only |
| < 8.2   | ❌ No |

## Reporting a Vulnerability

Please report security vulnerabilities privately via GitHub Security Advisories:

1. Go to **Security → Advisories → New draft advisory**
2. Describe the vulnerability
3. We will respond within 48 hours

Please do NOT open public issues for security bugs.

## Security Measures

- SSRF protection
- Path traversal prevention
- Prompt injection filtering
- API key management
- Rate limiting
- Audit logging
- Input validation (Pydantic)

## Bug Bounty

We offer $50-$200 for critical vulnerabilities. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
