"""Tests for gRPC service."""

import pytest

# Skip entire module if grpc is unavailable
grpc = pytest.importorskip("grpc")

from rtmdk.server.proto import rtmdk_pb2  # noqa: E402
from rtmdk.server.proto import rtmdk_pb2_grpc  # noqa: E402
from rtmdk.server.grpc_service import create_server  # noqa: E402

app_mod = pytest.importorskip("rtmdk.server.app")


@pytest.fixture(scope="module")
def grpc_server():
    app_mod.ENABLE_API_AUTH = False
    server = create_server(port=0)  # ephemeral port
    port = server.add_insecure_port("localhost:0")
    server.start()
    yield server, port
    server.stop(5)


@pytest.fixture
def grpc_channel(grpc_server):
    _, port = grpc_server
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel
    channel.close()


@pytest.fixture
def grpc_stub(grpc_channel):
    return rtmdk_pb2_grpc.RTMDKStub(grpc_channel)


class TestGRPCHelath:
    def test_health(self, grpc_stub):
        resp = grpc_stub.Health(rtmdk_pb2.HealthRequest())
        assert resp.status == "ok"
        assert resp.version == "8.3.0"

    def test_query_memory_unavailable(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.QueryMemory(rtmdk_pb2.QueryMemoryRequest(query="test"))
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_create_node_unavailable(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.CreateNode(rtmdk_pb2.CreateNodeRequest(content="hello"))
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_get_node_unavailable(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.GetNode(rtmdk_pb2.GetNodeRequest(node_id="missing"))
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

    def test_delete_node_unavailable(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as exc_info:
            grpc_stub.DeleteNode(rtmdk_pb2.DeleteNodeRequest(node_id="missing"))
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
