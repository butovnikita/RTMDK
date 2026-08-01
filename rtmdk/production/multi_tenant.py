"""
rtmdk/production/multi_tenant.py — Multi-Tenant Memory Isolation.

Provides isolated memory instances per tenant/user.
Features:
- One RTMDK → isolated memories per tenant
- Configurable max tenants
- Resource limits per tenant (max_nodes, RAM)
- Per-tenant stats and billing
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class TenantConfig:
    """Configuration for a single tenant."""

    max_nodes: int = 10000
    max_ram_mb: float = 100.0
    enable_engrams: bool = True
    enable_dreaming: bool = False
    auto_save: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class TenantStats:
    """Statistics for a single tenant."""

    tenant_id: str
    node_count: int = 0
    query_count: int = 0
    avg_latency_ms: float = 0.0
    created_at: float = 0.0
    last_active: float = 0.0
    ram_mb: float = 0.0


class TenantRouter:
    """Routes requests to tenant-isolated memory instances.

    Usage:
        router = TenantRouter(embedder_factory, max_tenants=100)

        # Get or create tenant memory:
        memory = router.get_tenant_memory("user123")

        # Use as normal:
        memory.save_context({"input": "Hello"}, {"output": "Hi!"})

        # Get tenant stats:
        stats = router.get_tenant_stats("user123")

        # List all tenants:
        tenants = router.list_tenants()
    """

    def __init__(
        self,
        embedder_factory: Callable,
        max_tenants: int = 100,
        default_config: Optional[TenantConfig] = None,
        on_tenant_created: Optional[Callable] = None,
        on_tenant_limit: Optional[Callable] = None,
    ):
        self.embedder_factory = embedder_factory
        self.max_tenants = max_tenants
        self.default_config = default_config or TenantConfig()
        self.on_tenant_created = on_tenant_created
        self.on_tenant_limit = on_tenant_limit

        self._memories: Dict[str, Any] = {}  # tenant_id → RTMDKMemory
        self._configs: Dict[str, TenantConfig] = {}
        self._stats: Dict[str, TenantStats] = {}

    def get_tenant_memory(
        self,
        tenant_id: str,
        config: Optional[TenantConfig] = None,
    ) -> Any:
        """Get or create isolated memory for a tenant.

        Args:
            tenant_id: Unique tenant identifier
            config: Optional custom config (uses default if None)

        Returns:
            RTMDKMemory instance for this tenant
        """
        if tenant_id in self._memories:
            # Update last active
            self._stats[tenant_id].last_active = time.time()
            self._stats[tenant_id].query_count += 1
            return self._memories[tenant_id]

        # Check tenant limit
        if len(self._memories) >= self.max_tenants:
            if self.on_tenant_limit:
                self.on_tenant_limit(tenant_id)
            raise ValueError(f"Max tenants ({self.max_tenants}) reached")

        # Create new tenant memory
        tenant_config = config or self.default_config
        embedder = self.embedder_factory()

        # Import here to avoid circular imports
        from rtmdk import create_rtmdk

        memory = create_rtmdk(
            preset="production",
            embedder=embedder,
            max_nodes=tenant_config.max_nodes,  # type: ignore[call-arg]
            enable_engrams=tenant_config.enable_engrams,  # type: ignore[call-arg]
            offline_dreaming=tenant_config.enable_dreaming,  # type: ignore[call-arg]
        )

        self._memories[tenant_id] = memory
        self._configs[tenant_id] = tenant_config
        self._stats[tenant_id] = TenantStats(
            tenant_id=tenant_id,
            created_at=time.time(),
            last_active=time.time(),
        )

        if self.on_tenant_created:
            self.on_tenant_created(tenant_id)

        return memory

    def remove_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant and free its resources."""
        if tenant_id not in self._memories:
            return False

        del self._memories[tenant_id]
        self._configs.pop(tenant_id, None)
        self._stats.pop(tenant_id, None)
        return True

    def list_tenants(self) -> List[Dict]:
        """List all active tenants."""
        result = []
        for tenant_id, stats in self._stats.items():
            result.append(
                {
                    "tenant_id": tenant_id,
                    "node_count": stats.node_count,
                    "query_count": stats.query_count,
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                    "created_at": stats.created_at,
                    "last_active": stats.last_active,
                    "ram_mb": round(stats.ram_mb, 2),
                }
            )
        return result

    def get_tenant_stats(self, tenant_id: str) -> Optional[Dict]:
        """Get statistics for a specific tenant."""
        if tenant_id not in self._memories:
            return None

        memory = self._memories[tenant_id]
        stats = self._stats[tenant_id]

        # Update live stats
        stats.node_count = len(memory.field.nodes)

        return {
            "tenant_id": tenant_id,
            "node_count": stats.node_count,
            "query_count": stats.query_count,
            "avg_latency_ms": round(stats.avg_latency_ms, 2),
            "config": {
                "max_nodes": self._configs[tenant_id].max_nodes,
                "max_ram_mb": self._configs[tenant_id].max_ram_mb,
                "engrams": self._configs[tenant_id].enable_engrams,
            },
        }

    def get_global_stats(self) -> Dict:
        """Get global stats across all tenants."""
        total_nodes = sum(s.node_count for s in self._stats.values())
        total_queries = sum(s.query_count for s in self._stats.values())

        return {
            "active_tenants": len(self._memories),
            "max_tenants": self.max_tenants,
            "total_nodes": total_nodes,
            "total_queries": total_queries,
            "avg_nodes_per_tenant": total_nodes / max(len(self._memories), 1),
        }
