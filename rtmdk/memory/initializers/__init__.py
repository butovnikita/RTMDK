"""Initializers package — R10.1 split of FieldInitializer God object.

R10.1 (2026-08-24, audit/risks-2026-08-24): FieldInitializer (574 lines, 30+ _init_*)
was sequential coupling, order-critical. Now split into:

- CoreInitializer   — core field state, caches, projection, adaptive, lifecycle
- IndexInitializer  — tiered storage, HNSW/BM25/shard routing, crystallization
- SecurityInitializer — production, federated, engrams, version/role, managers

DIContainer wires them via constructor injection (field, cfg, projection_matrix, wal_path).
FieldInitializer.initialize() now delegates to the three (see field_initializer.py:92).
See docs/RISKS.md R10.1 and BACKLOG.md R3.1/R10.1.
"""

from __future__ import annotations

from rtmdk.memory.initializers.core import CoreInitializer
from rtmdk.memory.initializers.index import IndexInitializer
from rtmdk.memory.initializers.security import SecurityInitializer


class DIContainer:
    """Minimal DI container for field wiring — R10.1."""

    def __init__(self, field, config, projection_matrix=None, wal_path=None):
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path
        self.core = CoreInitializer(field, config, projection_matrix, wal_path)
        self.index = IndexInitializer(field, config, projection_matrix, wal_path)
        self.security = SecurityInitializer(field, config, projection_matrix, wal_path)

    def initialize(self) -> None:
        """Delegate to the three initializers in dependency order."""
        # Core first (quant, rng, wal, caches, etc. — other inits depend on them)
        self.core.initialize()
        # Index second (needs field.cfg, rng, quant)
        self.index.initialize()
        # Security/production last (needs indexes, caches)
        self.security.initialize()


__all__ = ["DIContainer", "CoreInitializer", "IndexInitializer", "SecurityInitializer"]
