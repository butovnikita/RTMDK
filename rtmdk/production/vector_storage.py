"""rtmdk/production/vector_storage.py — Persistent vector storage backends.

Supports:
  - SQLite (brute-force numpy search, no extension required)
  - In-memory fallback
  - pgvector stub (requires psycopg2 + pgvector server)

Future: SQLite-VSS when sqlite-vss is available.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_F32_FMT = struct.Struct("f")


def _encode_vector(vec: NDArray) -> bytes:
    """Pack float32 vector into bytes."""
    return vec.astype(np.float32).tobytes()


def _decode_vector(blob: bytes, dim: int) -> NDArray:
    """Unpack bytes into float32 vector."""
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorStorage:
    """Factory / abstract façade for vector-native storage.

    Use ``VectorStorage.create(dsn)`` to obtain a concrete backend.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn

    @classmethod
    def create(cls, dsn: Optional[str] = None, dim: int = 64) -> "VectorStorage":
        """Create the best available backend for *dsn*."""
        if dsn is None or dsn == ":memory:" or dsn == "":
            return InMemoryVectorStorage(dim=dim)
        if dsn.startswith("sqlite"):
            return SQLiteVectorStorage(dsn=dsn, dim=dim)
        if dsn.startswith("postgresql"):
            try:
                return PGVectorStorage(dsn=dsn, dim=dim)
            except Exception as exc:
                logger.warning("pgvector unavailable (%s), falling back to memory", exc)
                return InMemoryVectorStorage(dim=dim)
        return InMemoryVectorStorage(dim=dim)

    # -- interface ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return True

    def insert(self, node_id: str, vector: NDArray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError

    async def ainsert(self, node_id: str, vector: NDArray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        import asyncio
        return await asyncio.to_thread(self.insert, node_id, vector, metadata)

    def search(self, query: NDArray, top_k: int = 5) -> List[Tuple[str, float]]:
        raise NotImplementedError

    async def asearch(self, query: NDArray, top_k: int = 5) -> List[Tuple[str, float]]:
        import asyncio
        return await asyncio.to_thread(self.search, query, top_k)

    def delete(self, node_id: str) -> bool:
        raise NotImplementedError

    async def adelete(self, node_id: str) -> bool:
        import asyncio
        return await asyncio.to_thread(self.delete, node_id)

    def get(self, node_id: str) -> Optional[NDArray]:
        raise NotImplementedError

    async def aget(self, node_id: str) -> Optional[NDArray]:
        import asyncio
        return await asyncio.to_thread(self.get, node_id)

    def count(self) -> int:
        raise NotImplementedError

    async def acount(self) -> int:
        import asyncio
        return await asyncio.to_thread(self.count)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

class InMemoryVectorStorage(VectorStorage):
    """Brute-force in-memory storage with numpy batch search."""

    def __init__(self, dim: int = 64):
        super().__init__(dsn=None)
        self.dim = dim
        self._vectors: Dict[str, NDArray] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def insert(self, node_id: str, vector: NDArray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        self._vectors[node_id] = vec
        self._meta[node_id] = metadata or {}
        return True

    def search(self, query: NDArray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._vectors:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        ids = list(self._vectors.keys())
        mat = np.vstack([self._vectors[i] for i in ids])  # (N, D)
        # Cosine similarity
        qn = q / (np.linalg.norm(q) + 1e-12)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mn @ qn  # (N,)
        idx = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in idx]

    def delete(self, node_id: str) -> bool:
        self._vectors.pop(node_id, None)
        self._meta.pop(node_id, None)
        return True

    def get(self, node_id: str) -> Optional[NDArray]:
        return self._vectors.get(node_id)

    def count(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class SQLiteVectorStorage(VectorStorage):
    """SQLite-backed vector storage.

    Vectors are stored as BLOBs.  Search is performed in-memory via numpy
    after loading candidate rows.  This avoids any extension dependency.
    """

    def __init__(self, dsn: str, dim: int = 64):
        super().__init__(dsn=dsn)
        self.dim = dim
        # Parse dsn: sqlite:///path/to/file.db
        path = dsn.replace("sqlite://", "").lstrip("/")
        if path == ":memory:":
            path = ":memory:"
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rtmdk_vectors (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vec_created ON rtmdk_vectors(created_at)"
        )
        self._conn.commit()

    def insert(self, node_id: str, vector: NDArray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        blob = _encode_vector(vec)
        meta_json = json.dumps(metadata or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO rtmdk_vectors (node_id, vector, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    vector=excluded.vector,
                    metadata=excluded.metadata,
                    created_at=julianday('now')
                """,
                (node_id, blob, meta_json),
            )
        return True

    def search(self, query: NDArray, top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        qn = q / (np.linalg.norm(q) + 1e-12)

        cur = self._conn.execute(
            "SELECT node_id, vector FROM rtmdk_vectors"
        )
        rows = cur.fetchall()
        if not rows:
            return []

        ids = []
        mat = np.empty((len(rows), self.dim), dtype=np.float32)
        for i, (nid, blob) in enumerate(rows):
            ids.append(nid)
            mat[i] = _decode_vector(blob, self.dim)

        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mn @ qn
        idx = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in idx]

    def delete(self, node_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM rtmdk_vectors WHERE node_id = ?", (node_id,)
            )
            return cur.rowcount > 0

    def get(self, node_id: str) -> Optional[NDArray]:
        cur = self._conn.execute(
            "SELECT vector FROM rtmdk_vectors WHERE node_id = ?", (node_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _decode_vector(row[0], self.dim)

    def count(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM rtmdk_vectors"
        )
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# pgvector backend (optional)
# ---------------------------------------------------------------------------

class PGVectorStorage(VectorStorage):
    """pgvector-backed storage (requires psycopg2 and pgvector extension)."""

    def __init__(self, dsn: str, dim: int = 64):
        super().__init__(dsn=dsn)
        import psycopg2  # type: ignore[import-untyped]
        self.dim = dim
        self._conn = psycopg2.connect(dsn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS rtmdk_vectors (
                        node_id TEXT PRIMARY KEY,
                        embedding vector({self.dim}),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vec_embedding
                    ON rtmdk_vectors USING ivfflat (embedding vector_cosine_ops)
                    """
                )

    def insert(self, node_id: str, vector: NDArray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
        meta_json = json.dumps(metadata or {})
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rtmdk_vectors (node_id, embedding, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                    """,
                    (node_id, vec.tolist(), meta_json),
                )
        return True

    def search(self, query: NDArray, top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM rtmdk_vectors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q.tolist(), q.tolist(), top_k),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]

    def delete(self, node_id: str) -> bool:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rtmdk_vectors WHERE node_id = %s", (node_id,)
                )
                return cur.rowcount > 0

    def get(self, node_id: str) -> Optional[NDArray]:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM rtmdk_vectors WHERE node_id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return np.array(row[0], dtype=np.float32)

    def count(self) -> int:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rtmdk_vectors")
                return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()
