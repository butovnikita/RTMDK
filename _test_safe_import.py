import sys
sys.path.insert(0, '.')

def _safe_import(module_path, **kwargs):
    try:
        parts = module_path.rsplit('.', 1)
        if len(parts) != 2:
            return None
        module_path_str, class_name = parts
        mod = __import__(module_path_str, fromlist=[class_name])
        cls = getattr(mod, class_name, None)
        if cls:
            return cls(**kwargs)
    except Exception as e:
        print(f'  Error: {type(e).__name__}: {e}', flush=True)
    return None

print('Testing EmbeddingCache...', flush=True)
result = _safe_import('rtmdk.production.embedding_cache.EmbeddingCache', cache_dir='test', max_size=100)
print(f'Result: {result}', flush=True)

print('Testing BackupManager...', flush=True)
result = _safe_import('rtmdk.production.backup_restore.BackupManager')
print(f'Result: {result}', flush=True)

print('Testing HealthMonitor...', flush=True)
result = _safe_import('rtmdk.production.health_monitor.HealthMonitor', memory=None)
print(f'Result: {result}', flush=True)

print('Done!', flush=True)
