"""rtmdk/production/webhooks.py — Webhook subscription manager.

Manage HTTP callback subscriptions for memory events.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx


@dataclass
class WebhookSubscription:
    """A webhook subscription record."""

    id: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    active: bool = True
    tenant_id: Optional[str] = None


class WebhookManager:
    """Manage webhook subscriptions and dispatch events.

    Usage:
        mgr = WebhookManager()
        sub = mgr.subscribe("https://example.com/hook", ["node_created"])
        mgr.dispatch("node_created", {"node_id": "n1"})
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = str(Path.home() / ".rtmdk" / "webhooks.json")
        self.storage_path = storage_path
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self._subs: Dict[str, WebhookSubscription] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for sid, rec in data.items():
                self._subs[sid] = WebhookSubscription(**rec)
        except Exception:
            pass

    def _save(self):
        payload = {sid: asdict(sub) for sid, sub in self._subs.items()}
        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def subscribe(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> WebhookSubscription:
        """Create a new subscription."""
        sub = WebhookSubscription(
            id=str(uuid.uuid4())[:12],
            url=url,
            events=list(events),
            secret=secret,
            tenant_id=tenant_id,
        )
        self._subs[sub.id] = sub
        self._save()
        return sub

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscription."""
        if sub_id in self._subs:
            del self._subs[sub_id]
            self._save()
            return True
        return False

    def list_subscriptions(
        self,
        event_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[WebhookSubscription]:
        """List subscriptions with optional filtering."""
        results = []
        for sub in self._subs.values():
            if not sub.active:
                continue
            if event_type is not None and event_type not in sub.events:
                continue
            if tenant_id is not None and sub.tenant_id != tenant_id:
                continue
            results.append(sub)
        return results

    def dispatch(
        self,
        event_type: str,
        payload: Dict,
        tenant_id: Optional[str] = None,
    ) -> List[Dict]:
        """Dispatch event to all matching subscriptions.

        Returns list of delivery results for observability.
        """
        subs = self.list_subscriptions(event_type=event_type, tenant_id=tenant_id)
        results = []
        for sub in subs:
            result = self._send(sub, event_type, payload)
            results.append(result)
        return results

    def _send(
        self,
        sub: WebhookSubscription,
        event_type: str,
        payload: Dict,
    ) -> Dict:
        """Send a single webhook payload."""
        body = {
            "event": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }
        headers = {"Content-Type": "application/json"}
        if sub.secret:
            import hmac
            import hashlib
            sig = hmac.new(
                sub.secret.encode(),
                json.dumps(body, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={sig}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(sub.url, json=body, headers=headers)
            return {
                "sub_id": sub.id,
                "url": sub.url,
                "status_code": resp.status_code,
                "success": resp.status_code < 400,
            }
        except Exception as exc:
            return {
                "sub_id": sub.id,
                "url": sub.url,
                "status_code": None,
                "success": False,
                "error": str(exc),
            }
