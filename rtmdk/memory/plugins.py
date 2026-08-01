"""rtmdk/memory/plugins.py — Plugin architecture for RTMDK memory subsystems.

Phase 5 Architecture: Introduces FieldPlugin Protocol and MemoryPort ABC
to enable modular, testable, and swappable memory subsystems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Any, runtime_checkable
from numpy.typing import NDArray


@runtime_checkable
class FieldPlugin(Protocol):
    """Protocol for pluggable RTMDK field subsystems.

    Any subsystem that wants to hook into RTMDKField lifecycle
    (add_node, query, consolidate, export) can implement this protocol.
    """

    name: str

    def on_node_added(self, node_id: str, latent_pos: NDArray, content: Dict) -> None:
        """Called after a node is added to the field."""
        ...

    def on_query(self, query_latent: NDArray, results: List[Tuple[str, float, Any]]) -> List[Tuple[str, float, Any]]:
        """Called after query results are computed. May modify or filter results."""
        return results

    def on_consolidate(self, updated_nodes: List[str]) -> None:
        """Called after consolidation completes."""
        ...

    def get_state(self) -> Optional[Dict]:
        """Return serializable state for checkpointing."""
        return None

    def load_state(self, state: Dict) -> None:
        """Restore state from checkpoint."""
        ...


class MemoryPort(ABC):
    """Abstract base class for memory backend implementations.

    Decouples high-level RTMDKMemory API from concrete field implementations,
    allowing alternative backends (redis, sqlite, distributed, etc.).
    """

    @abstractmethod
    def add(self, embedding: NDArray, content: Dict, **kwargs) -> str:
        """Store a memory vector with metadata. Returns node id."""
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: NDArray, top_k: int = 10, session_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Retrieve top-k memories by vector similarity. Returns (id, score) pairs."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, node_id: str) -> bool:
        """Remove a memory node. Returns True if existed."""
        raise NotImplementedError

    @abstractmethod
    def export(self, path: str, fmt: Optional[str] = None) -> None:
        """Persist memory state to disk."""
        raise NotImplementedError

    @abstractmethod
    def import_(self, path: str) -> None:
        """Load memory state from disk."""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        raise NotImplementedError
