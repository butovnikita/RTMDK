"""rtmdk/production/vector_storage.py — Vector-native storage stub.

Future: SQLite-VSS or pgvector backend for persistent vector storage.
"""

from typing import List, Optional, Tuple

import numpy as np


class VectorStorage:
    """Stub for vector-native storage backend.

    When a vector DB is available, this class replaces in-memory node
    storage with persistent vector operations.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn
        self._available = False
        if dsn:
            if dsn.startswith("sqlite"):
                try:
                    import sqlite_vss  # noqa: F401
                    self._available = True
                except ImportError:
                    pass
            elif dsn.startswith("postgresql"):
                try:
                    import pgvector  # noqa: F401
                    self._available = True
                except ImportError:
                    pass

    @property
    def available(self) -> bool:
        return self._available

    def insert(self, node_id: str, vector: np.ndarray, metadata: dict) -> bool:
        """Insert vector into storage (stub)."""
        return self._available

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """ANN search (stub)."""
        return []

    def delete(self, node_id: str) -> bool:
        """Delete vector (stub)."""
        return self._available
