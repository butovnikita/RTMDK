"""FieldInitializer — R10.1 split into Core/Index/Security + DI.

R10.1 (2026-08-24, audit/risks-2026-08-24): was God initializer (574 lines, 30+ _init_*
sequential coupling, order-critical). Now delegates to:

- CoreInitializer   (rtmdk/memory/initializers/core.py)    — core state, caches, projection, lifecycle
- IndexInitializer  (rtmdk/memory/initializers/index.py)   — tiered, HNSW/BM25, sparse routing
- SecurityInitializer (rtmdk/memory/initializers/security.py) — production, federated, engrams, managers

DIContainer (initializers/__init__.py) wires them via constructor injection.
FieldInitializer stays as thin facade for backward compat (rtmdk/memory/field.py
still does `FieldInitializer(field, cfg).initialize()`). New code should use
DIContainer directly. See docs/RISKS.md R10.1, BACKLOG.md R10.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from numpy.typing import NDArray

from rtmdk.memory.config import RTMDKConfig

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField


class FieldInitializer:
    """Thin facade — delegates to DIContainer (Core/Index/Security)."""

    def __init__(
        self,
        field: "RTMDKField",
        config: RTMDKConfig,
        projection_matrix: Optional[NDArray] = None,
        wal_path: Optional[str] = None,
    ) -> None:
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path

    def initialize(self) -> None:
        """Delegate to DIContainer (R10.1)."""
        from rtmdk.memory.initializers import DIContainer

        DIContainer(self.field, self.cfg, self.projection_matrix, self.wal_path).initialize()

    # ------------------------------------------------------------------
    # Backward-compat wrappers — old code may call _init_* directly.
    # They now delegate to the appropriate sub-initializer.
    # ------------------------------------------------------------------
    def _normalize_identity_projection(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._normalize_identity_projection()

    def _init_tiered_storage(self) -> None:
        from rtmdk.memory.initializers.index import IndexInitializer

        IndexInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_tiered_storage()

    def _init_wal(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_wal()

    def _init_caches(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_caches()

    def _init_conformal_and_learned(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_conformal_and_learned()

    def _init_adaptive_bandwidth(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_adaptive_bandwidth()

    def _init_adaptive_phase_coupling(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_adaptive_phase_coupling()

    def _init_kalman(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_kalman()

    def _init_projection_manager(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_projection_manager()

    def _init_adaptive_tda_gpu(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_adaptive_tda_gpu()

    def _init_index_manager(self) -> None:
        from rtmdk.memory.initializers.index import IndexInitializer

        IndexInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_index_manager()

    def _init_learnable_and_diff(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_learnable_and_diff()

    def _init_meta_and_healer(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_meta_and_healer()

    def _init_lazy_engines(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_lazy_engines()

    def _init_agent_orchestration(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_agent_orchestration()

    def _init_production_mode(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_production_mode()

    def _init_federated(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_federated()

    def _init_predictive_counterfactual_dp(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_predictive_counterfactual_dp()

    def _init_sparse_routing(self) -> None:
        from rtmdk.memory.initializers.index import IndexInitializer

        IndexInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_sparse_routing()

    def _init_crystallization_counters(self) -> None:
        from rtmdk.memory.initializers.index import IndexInitializer

        IndexInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_crystallization_counters()

    def _init_lifecycle_controls(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_lifecycle_controls()

    def _init_circuit_breakers(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_circuit_breakers()

    def _init_tension_cache(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_tension_cache()

    def _init_async_pipeline(self) -> None:
        from rtmdk.memory.initializers.index import IndexInitializer

        IndexInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_async_pipeline()

    def _init_goal_rl_event_lowrank(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_goal_rl_event_lowrank()

    def _init_engram(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_engram()

    def _init_meta_memory_security(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_meta_memory_security()

    def _init_version_and_role(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_version_and_role()

    def _init_stats(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_stats()

    def _init_deques_and_counters(self) -> None:
        from rtmdk.memory.initializers.core import CoreInitializer

        CoreInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_deques_and_counters()

    def _init_managers(self) -> None:
        from rtmdk.memory.initializers.security import SecurityInitializer

        SecurityInitializer(self.field, self.cfg, self.projection_matrix, self.wal_path)._init_managers()
