"""rtmdk/production/tenant_rate_limiter.py — Per-tenant rate limiter.

Wraps the base RateLimiter with tenant-specific overrides from APIKeyManager.
"""

import os
from typing import Dict, Optional

from rtmdk.production.api_key_manager import APIKeyManager
from rtmdk.production.rate_limiter import RateLimiter


class TenantRateLimiter:
    """Rate limiter keyed by tenant_id with optional per-tenant overrides.

    Usage:
        trl = TenantRateLimiter(api_key_manager=mgr)
        trl.allow_request(tenant_id="t1") -> bool
        trl.get_remaining(tenant_id="t1") -> {"minute": 45, ...}
    """

    def __init__(
        self,
        api_key_manager: Optional[APIKeyManager] = None,
        default_per_minute: int = None,
        default_per_hour: int = None,
        default_per_day: int = None,
        pipeline_per_minute: int = None,
    ):
        self._api_key_manager = api_key_manager
        # Global defaults from env
        self._defaults = {
            "per_minute": default_per_minute
            or int(os.getenv("RTMDK_RATE_LIMIT_PER_MINUTE", "60")),
            "per_hour": default_per_hour
            or int(os.getenv("RTMDK_RATE_LIMIT_PER_HOUR", "1000")),
            "per_day": default_per_day
            or int(os.getenv("RTMDK_RATE_LIMIT_PER_DAY", "10000")),
        }
        self._pipeline_per_minute = pipeline_per_minute or int(
            os.getenv("RTMDK_PIPELINE_RATE_LIMIT_PER_MINUTE", "30")
        )
        # tenant_id -> RateLimiter instance
        self._limiters: Dict[str, RateLimiter] = {}
        # tenant_id -> RateLimiter instance for pipeline endpoints
        self._pipeline_limiters: Dict[str, RateLimiter] = {}

    def _get_limiter(self, tenant_id: str) -> RateLimiter:
        """Get or create a RateLimiter for tenant with optional override."""
        if tenant_id not in self._limiters:
            limits = dict(self._defaults)
            # Check for per-tenant override via API key records
            if self._api_key_manager is not None:
                keys = self._api_key_manager.list_keys(tenant_id=tenant_id)
                for key_rec in keys:
                    override = key_rec.get("rate_limit_override")
                    if override:
                        limits.update(override)
                        break  # use first override found
            self._limiters[tenant_id] = RateLimiter(
                max_per_minute=limits["per_minute"],
                max_per_hour=limits["per_hour"],
                max_per_day=limits["per_day"],
            )
        return self._limiters[tenant_id]

    def _get_pipeline_limiter(self, tenant_id: str) -> RateLimiter:
        """Get or create a stricter RateLimiter for pipeline endpoints."""
        if tenant_id not in self._pipeline_limiters:
            limits = dict(self._defaults)
            limits["per_minute"] = self._pipeline_per_minute
            if self._api_key_manager is not None:
                keys = self._api_key_manager.list_keys(tenant_id=tenant_id)
                for key_rec in keys:
                    override = key_rec.get("rate_limit_override")
                    if override and "pipeline_per_minute" in override:
                        limits["per_minute"] = override["pipeline_per_minute"]
                        break
            self._pipeline_limiters[tenant_id] = RateLimiter(
                max_per_minute=limits["per_minute"],
                max_per_hour=limits["per_hour"],
                max_per_day=limits["per_day"],
            )
        return self._pipeline_limiters[tenant_id]

    def allow_request(self, tenant_id: str) -> bool:
        limiter = self._get_limiter(tenant_id)
        return limiter.allow_request(tenant_id)

    def allow_pipeline_request(self, tenant_id: str) -> bool:
        """Stricter rate limit for pipeline endpoints."""
        limiter = self._get_pipeline_limiter(tenant_id)
        return limiter.allow_request(tenant_id)

    def get_remaining(self, tenant_id: str) -> Dict[str, int]:
        limiter = self._get_limiter(tenant_id)
        return limiter.get_remaining(tenant_id)

    def reset_tenant(self, tenant_id: str):
        """Clear rate-limit state for a tenant (e.g. on plan upgrade)."""
        self._limiters.pop(tenant_id, None)
        self._pipeline_limiters.pop(tenant_id, None)
