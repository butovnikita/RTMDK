"""
rtmdk/production/events.py — Webhook & Event System.

Emits events for important memory operations with webhook support.
"""

import time
from typing import Dict, List, Callable, Any, Optional


class EventSystem:
    """Event system for RTMDK with webhook support.

    Usage:
        events = EventSystem()

        # Register handler
        events.on("node_added", lambda e: print(f"Node added: {e['node_id']}"))

        # Emit event
        events.emit("node_added", {"node_id": "n_123"})

        # Webhook
        events.add_webhook("https://example.com/hook", events=["node_added", "memory_full"])
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._webhooks: List[Dict[str, Any]] = []
        self._event_log: List[Dict] = []

    def on(self, event_name: str, handler: Callable):
        """Register event handler."""
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)

    def emit(self, event_name: str, data: Optional[Dict] = None):
        """Emit an event."""
        event = {
            "name": event_name,
            "data": data or {},
            "timestamp": time.time(),
        }
        self._event_log.append(event)

        # Call handlers
        for handler in self._handlers.get(event_name, []):
            try:
                handler(event)
            except Exception:
                pass

        # Send webhooks
        for webhook in self._webhooks:
            if event_name in webhook.get("events", []):
                try:
                    import requests
                    requests.post(webhook["url"], json=event, timeout=10)
                except Exception:
                    pass

    def add_webhook(self, url: str, events: Optional[List[str]] = None):
        """Add webhook URL."""
        self._webhooks.append({"url": url, "events": events or []})

    def get_event_log(self, limit: int = 100) -> List[Dict]:
        """Get recent events."""
        return self._event_log[-limit:]
