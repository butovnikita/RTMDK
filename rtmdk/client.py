"""rtmdk/client.py — Official Python client for RTMDK REST API.

Usage:
    from rtmdk.client import RTMDKClient

    client = RTMDKClient(base_url="http://localhost:8080", api_key="rtmdk_xxx")
    client.query_memory("What is the capital of France?")
    client.create_node("New knowledge")
"""

from typing import Any, Dict, List, Optional

import httpx

__all__ = ["RTMDKClientError", "RTMDKClient", "AsyncRTMDKClient"]


class RTMDKClientError(Exception):
    """Base exception for RTMDK client errors."""

    def __init__(self, message: str, status_code: int = None, response_body: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RTMDKClient:
    """Typed Python client for RTMDK Production API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        resp = self._client.request(method, url, headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = None
            raise RTMDKClientError(
                f"HTTP {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=body,
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "rtmdk",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        session_id: str = "default",
    ) -> dict:
        """Send chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "session_id": session_id,
        }
        return self._request("POST", "/v1/chat/completions", json=payload)

    # ------------------------------------------------------------------
    # Memory Query
    # ------------------------------------------------------------------
    def query_memory(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        session_id: str = "default",
    ) -> dict:
        """Query memory field."""
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "session_id": session_id,
        }
        return self._request("POST", "/v1/memory/query", json=payload)

    def batch_query_memory(
        self,
        queries: List[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict:
        """Batch query memory field."""
        payload = {
            "queries": queries,
            "top_k": top_k,
            "threshold": threshold,
        }
        return self._request("POST", "/v1/memory/batch_query", json=payload)

    # ------------------------------------------------------------------
    # Memory Node CRUD
    # ------------------------------------------------------------------
    def create_node(
        self,
        content: str,
        node_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> dict:
        """Create a memory node."""
        payload: Dict[str, Any] = {"content": content}
        if node_id:
            payload["node_id"] = node_id
        if metadata:
            payload["metadata"] = metadata
        return self._request("POST", "/v1/memory/nodes", json=payload)

    def get_node(self, node_id: str) -> dict:
        """Get a memory node by ID."""
        return self._request("GET", f"/v1/memory/nodes/{node_id}")

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> dict:
        """Update a memory node."""
        payload: Dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        if metadata is not None:
            payload["metadata"] = metadata
        return self._request("PUT", f"/v1/memory/nodes/{node_id}", json=payload)

    def delete_node(self, node_id: str) -> dict:
        """Delete a memory node."""
        return self._request("DELETE", f"/v1/memory/nodes/{node_id}")

    def list_nodes(self, limit: int = 50, offset: int = 0) -> dict:
        """List memory nodes with pagination."""
        return self._request(
            "GET", "/v1/memory/nodes", params={"limit": limit, "offset": offset}
        )

    # ------------------------------------------------------------------
    # Batch Ingest
    # ------------------------------------------------------------------
    def batch_ingest(
        self,
        documents: List[str],
        metadata: Optional[Dict] = None,
        node_ids: Optional[List[str]] = None,
    ) -> dict:
        """Batch ingest documents into memory."""
        payload: Dict[str, Any] = {"documents": documents}
        if metadata:
            payload["metadata"] = metadata
        if node_ids:
            payload["node_ids"] = node_ids
        return self._request("POST", "/v1/memory/batch_ingest", json=payload)

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------
    def export_memory(self) -> dict:
        """Export all memory nodes."""
        return self._request("GET", "/v1/memory/export")

    def import_memory(
        self,
        nodes: List[Dict],
        clear_existing: bool = False,
    ) -> dict:
        """Import memory nodes from JSON."""
        payload = {"nodes": nodes, "clear_existing": clear_existing}
        return self._request("POST", "/v1/memory/import", json=payload)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def analytics_overview(self) -> dict:
        """Get dashboard overview."""
        return self._request("GET", "/v1/analytics/overview")

    def analytics_memory(self) -> dict:
        """Get memory analytics."""
        return self._request("GET", "/v1/analytics/memory")

    def analytics_events(
        self,
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> dict:
        """Get recent events."""
        params: Dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return self._request("GET", "/v1/analytics/events", params=params)

    def track_event(
        self,
        event_type: str,
        properties: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Track a custom analytics event."""
        payload: Dict[str, Any] = {"event_type": event_type}
        if properties:
            payload["properties"] = properties
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/v1/analytics/track", json=payload)

    # ------------------------------------------------------------------
    # Admin / API Keys
    # ------------------------------------------------------------------
    def create_api_key(
        self,
        tenant_id: str,
        name: str = "",
        rate_limit_override: Optional[Dict] = None,
    ) -> dict:
        """Create API key (admin only)."""
        payload: Dict[str, Any] = {"tenant_id": tenant_id, "name": name}
        if rate_limit_override:
            payload["rate_limit_override"] = rate_limit_override
        return self._request("POST", "/v1/admin/api-keys", json=payload)

    def list_api_keys(self, tenant_id: Optional[str] = None) -> dict:
        """List API keys (admin only)."""
        params: Dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        return self._request("GET", "/v1/admin/api-keys", params=params)

    def revoke_api_key(self, key_hash: str) -> dict:
        """Revoke an API key (admin only)."""
        return self._request(
            "POST", "/v1/admin/api-keys/revoke", json={"key_hash": key_hash}
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Check server health."""
        return self._request("GET", "/health")

    def close(self):
        """Close underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AsyncRTMDKClient:
    """Async Python client for RTMDK Production API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        resp = await self._client.request(method, url, headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = None
            raise RTMDKClientError(
                f"HTTP {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=body,
            )
        return resp.json()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "rtmdk",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        session_id: str = "default",
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "session_id": session_id,
        }
        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def query_memory(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        session_id: str = "default",
    ) -> dict:
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "session_id": session_id,
        }
        return await self._request("POST", "/v1/memory/query", json=payload)

    async def create_node(
        self,
        content: str,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        payload: Dict[str, Any] = {"content": content}
        if node_id:
            payload["node_id"] = node_id
        if metadata:
            payload["metadata"] = metadata
        return await self._request("POST", "/v1/memory/nodes", json=payload)

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
