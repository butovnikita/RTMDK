# RTMDK v8.3.0 Release Checklist

## Pre-Release Verification
- [x] All tests pass (1112 passed, 2 skipped)
- [x] Smoke test passes (3 consecutive runs)
- [x] E2E smoke test passes (ingestion ~920 nodes/sec, p50 latency 0ms, p95 1ms, p99 1.4ms)
- [x] Build produces valid sdist + wheel (`python -m build`)
- [x] CHANGELOG.md updated with all changes
- [x] README.md stats refreshed (1112 tests, 109s suite)
- [x] `pyproject.toml` version = `8.3.0`

## Known Issues (Documented)
- Pre-existing flake8 style issues in `core.py` (F401/F811/E203/E501) — cosmetic, no functional impact
- `pytest --cov` crashes on Windows (C-extensions access violation) — CI runs on Ubuntu, not blocking

## Post-Release
- [ ] Create git tag `v8.3.0`
- [ ] Push tag to origin
- [ ] Create GitHub Release with CHANGELOG notes
- [ ] Upload sdist + wheel to PyPI (or attach to GitHub Release)
