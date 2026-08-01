"""rtmdk/production/api_key_manager.py — API Key Registry with tenant isolation.

HMAC-SHA256 hashed keys. JSON-backed persistent storage.
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class APIKeyRecord:
    """Stored metadata for an API key (raw key is NEVER stored)."""

    key_hash: str
    tenant_id: str
    name: str = ""
    created_at: float = field(default_factory=time.time)
    revoked: bool = False
    revoked_at: Optional[float] = None
    rate_limit_override: Optional[Dict] = None  # {"per_minute": 100, ...}


class APIKeyManager:
    """Registry for API keys with tenant isolation.

    Usage:
        mgr = APIKeyManager()
        raw_key, record = mgr.create_key(tenant_id="t1", name="prod")
        mgr.validate_key(raw_key) -> "t1" or None
        mgr.revoke_key(key_hash)
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = str(Path.home() / ".rtmdk" / "api_keys.json")
        self.storage_path = storage_path
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        # key_hash -> APIKeyRecord
        self._keys: Dict[str, APIKeyRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key_hash, rec in data.items():
                self._keys[key_hash] = APIKeyRecord(**rec)
        except Exception:
            pass

    def _save(self):
        payload = {h: asdict(r) for h, r in self._keys.items()}
        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Key lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """HMAC-SHA256 with a fixed secret derived from process env."""
        secret = os.getenv("RTMDK_KEY_SECRET", "rtmdk-default-secret-change-me").encode()
        return hmac.new(secret, raw_key.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _generate_raw_key() -> str:
        return "rtmdk_" + secrets.token_urlsafe(32)

    def create_key(
        self,
        tenant_id: str,
        name: str = "",
        rate_limit_override: Optional[Dict] = None,
    ) -> tuple:
        """Create a new API key. Returns (raw_key, record)."""
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        record = APIKeyRecord(
            key_hash=key_hash,
            tenant_id=tenant_id,
            name=name,
            rate_limit_override=rate_limit_override,
        )
        self._keys[key_hash] = record
        self._save()
        return raw_key, record

    def validate_key(self, raw_key: str) -> Optional[str]:
        """Validate raw key. Returns tenant_id or None."""
        key_hash = self._hash_key(raw_key)
        record = self._keys.get(key_hash)
        if record is None or record.revoked:
            return None
        return record.tenant_id

    def revoke_key(self, key_hash: str) -> bool:
        """Revoke an API key by hash. Returns True if found."""
        record = self._keys.get(key_hash)
        if record is None:
            return False
        record.revoked = True
        record.revoked_at = time.time()
        self._save()
        return True

    def delete_key(self, key_hash: str) -> bool:
        """Permanently delete a key. Returns True if found."""
        if key_hash in self._keys:
            del self._keys[key_hash]
            self._save()
            return True
        return False

    def list_keys(
        self,
        tenant_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> List[Dict]:
        """List key metadata (never includes raw keys)."""
        results = []
        for rec in self._keys.values():
            if tenant_id is not None and rec.tenant_id != tenant_id:
                continue
            if rec.revoked and not include_revoked:
                continue
            results.append(asdict(rec))
        return results

    def get_tenant_for_key(self, raw_key: str) -> Optional[str]:
        """Alias for validate_key."""
        return self.validate_key(raw_key)
