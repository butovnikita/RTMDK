import sys
print('starting', flush=True)
try:
    from rtmdk.production.embedding_cache import EmbeddingCache
    print('EmbeddingCache loaded OK', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)
