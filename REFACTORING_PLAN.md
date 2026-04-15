# RTMDK Refactoring Plan (v8.1)

## Recent Milestones (Completed ✅)
- **Phase 20: Domain Memory** implemented (`domain_classifier.py`, `nodes.py` fields, `core.py` guards)
- **O(n²) Performance Fix** in trajectory simulation (moved index lookup out of loop)
- **Security Cleanup** (removed API key pattern from docstring, fixed bare excepts)

## Current Status

## Identified Issues

### 1. SillyTavern Streaming
**Problem:** Stream starts but text doesn't appear in chat
**Root cause:** Format mismatch - ST expects `{"choices":[{"text":"delta"}]}` 
**Status:** Format fixed, but needs testing with real LM Studio streaming

### 2. Dashboard Node Count
**Problem:** Shows 0 nodes sometimes
**Root cause:** Multiple response format variations
**Status:** Fixed with multi-location checking

### 3. Error Handling Consistency
**Problem:** Different endpoints have different error handling patterns
**Solution:** Standardize on try/except + proper HTTP status codes

### 4. Logging
**Problem:** Inconsistent logging across endpoints
**Solution:** Add structured logging for all critical operations

## Refactoring Priority

### High Priority (User-facing issues)
1. **SillyTavern Streaming** - Test and fix with real LM Studio
2. **Dashboard Node Count** - Ensure 100% accuracy
3. **Backup Upload** - Make 100% reliable

### Medium Priority (Developer experience)
4. **Error handling standardization**
5. **Logging improvements**
6. **Code deduplication**

### Low Priority (Nice to have)
7. **Performance optimizations**
8. **Documentation updates**

## Next Steps

1. Test SillyTavern streaming with actual LM Studio streaming enabled
2. Verify node count shows correctly in all scenarios
3. Test backup upload with various file types
4. Standardize error responses across all endpoints
5. Add request/response logging for debugging

## Testing Checklist

- [ ] SillyTavern message generation works
- [ ] SillyTavern streaming shows text in real-time
- [ ] Dashboard shows correct node count
- [ ] Backup upload works with .json files
- [ ] Health endpoint returns valid data
- [ ] Memory persists across server restarts
- [ ] All API endpoints return proper error codes
