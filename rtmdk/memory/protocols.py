"""Protocols for field ↔ manager interaction — R10.3.

R10.3 (2026-08-24, audit/risks-2026-08-24): field -> manager -> field cycle
was hidden via `__getattr__` and mypy `disable_error_code=attr-defined`
(mypy.ini:25). This module provides explicit Protocol/ABC so managers
depend on an interface, not concrete RTMDKField, and `__getattr__` can be
gradually removed. See docs/RISKS.md R10.3, mypy.ini R2.2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, List, Any, Optional

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FieldLike(Protocol):
    """Minimal interface that managers need from RTMDKField."""

    nodes: Dict[str, Any]
    node_index: List[str]
    cfg: Any
    _write_lock: Any
    _cached_positions: Optional[NDArray]
    _cached_phases: Optional[NDArray]
    # ... extend as needed, keep explicit to avoid __getattr__ cycle
    def _build_node_cache(self) -> None: ...
    def _project(self, emb: NDArray) -> NDArray: ...


@runtime_checkable
class QueryManagerLike(Protocol):
    def query(self, embedding: NDArray, phase: float = 0.0, top_k: Optional[int] = None, **kwargs): ...
    def query_batch(self, embeddings: NDArray, **kwargs): ...


@runtime_checkable
class NodeManagerLike(Protocol):
    def add_node(self, embedding: NDArray, content: Dict, **kwargs) -> str: ...
    def add_nodes_batch(self, embeddings: NDArray, contents: List[Dict], **kwargs) -> List[str]: ...


# Re-export for convenience
__all__ = ["FieldLike", "QueryManagerLike", "NodeManagerLike"]
