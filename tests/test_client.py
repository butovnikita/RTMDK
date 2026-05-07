"""Tests for RTMDK Python client."""

from unittest.mock import MagicMock, patch

import pytest

from rtmdk.client import RTMDKClient, RTMDKClientError


class TestRTMDKClientUnit:
    """Unit tests for RTMDKClient using mocked HTTP responses."""

    def _mock_response(self, json_data=None, status_code=200, text=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = text or str(json_data)
        return resp

    @patch("rtmdk.client.httpx.Client")
    def test_health(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"status": "ok", "version": "8.2.0"}
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test", api_key="key")
        data = client.health()
        assert data["status"] == "ok"
        mock_client.request.assert_called_once()
        _, kwargs = mock_client.request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer key"
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_query_memory(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"query": "hello", "total": 2, "results": []}
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test")
        data = client.query_memory("hello", top_k=3)
        assert data["query"] == "hello"
        assert data["total"] == 2
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_create_node(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"id": "n1", "status": "created"}
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test")
        data = client.create_node("content", node_id="n1", metadata={"tag": "x"})
        assert data["status"] == "created"
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_error_raises_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            json_data={"error": "not found"}, status_code=404, text="not found"
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test")
        with pytest.raises(RTMDKClientError) as exc_info:
            client.get_node("missing")
        assert exc_info.value.status_code == 404
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_batch_ingest(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"ingested": 3, "node_ids": ["a", "b", "c"]}
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test")
        data = client.batch_ingest(["d1", "d2", "d3"])
        assert data["ingested"] == 3
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_export_memory(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"total": 5, "nodes": []}
        )
        mock_client_cls.return_value = mock_client

        client = RTMDKClient(base_url="http://test")
        data = client.export_memory()
        assert data["total"] == 5
        client.close()

    @patch("rtmdk.client.httpx.Client")
    def test_context_manager(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.request.return_value = self._mock_response(
            {"status": "ok"}
        )
        mock_client_cls.return_value = mock_client

        with RTMDKClient(base_url="http://test") as client:
            data = client.health()
            assert data["status"] == "ok"
        mock_client.close.assert_called_once()

    def test_api_key_header(self):
        client = RTMDKClient(base_url="http://test", api_key="test_key")
        assert client._headers()["Authorization"] == "Bearer test_key"
        client.close()
