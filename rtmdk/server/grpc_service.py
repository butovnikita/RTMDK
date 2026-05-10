"""rtmdk/server/grpc_service.py — gRPC service for RTMDK.

Runs alongside the FastAPI HTTP server.
"""

import asyncio
from concurrent import futures
import grpc

from rtmdk.server.proto import rtmdk_pb2
from rtmdk.server.proto import rtmdk_pb2_grpc


def _memory_from_globals():
    """Access global memory from app module."""
    import rtmdk.server.app as app_mod
    return getattr(app_mod, "memory", None)


def _require_key(context) -> bool:
    """Validate API key from gRPC metadata."""
    import rtmdk.server.app as app_mod
    if not getattr(app_mod, "ENABLE_API_AUTH", False):
        return True
    metadata = dict(context.invocation_metadata() or [])
    api_key = metadata.get("x-api-key", "")
    mgr = getattr(app_mod, "api_key_manager", None)
    if mgr is not None and mgr.validate_key(api_key):
        return True
    if api_key == getattr(app_mod, "API_KEY", ""):
        return True
    return False


class RTMDKServicer(rtmdk_pb2_grpc.RTMDKServicer):
    """gRPC servicer wrapping RTMDK memory operations."""

    def Health(self, request, context):
        memory = _memory_from_globals()
        node_count = 0
        if memory and memory.field is not None:
            node_count = len(memory.field.nodes)
        return rtmdk_pb2.HealthResponse(
            status="ok",
            version="8.3.0",
            memory_nodes=node_count,
        )

    def QueryMemory(self, request, context):
        if not _require_key(context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid API key")
            return rtmdk_pb2.QueryMemoryResponse()
        memory = _memory_from_globals()
        if memory is None or memory.field is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Memory not initialized")
            return rtmdk_pb2.QueryMemoryResponse()
        try:
            top_k = max(1, min(request.top_k or 5, 100))
            results = memory.query(
                request.query,
                top_k=top_k,
                session_id=request.session_id or "default",
            )
            out = []
            for r in results:
                node_id = getattr(r, "id", "")
                score = getattr(r, "score", 0.0)
                content = ""
                if hasattr(r, "content"):
                    if isinstance(r.content, dict):
                        content = r.content.get("text", str(r.content))
                    else:
                        content = str(r.content)
                out.append(rtmdk_pb2.MemoryResult(
                    node_id=node_id, score=float(score), content=content))
            return rtmdk_pb2.QueryMemoryResponse(results=out, total=len(out))
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return rtmdk_pb2.QueryMemoryResponse()

    def CreateNode(self, request, context):
        if not _require_key(context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid API key")
            return rtmdk_pb2.CreateNodeResponse()
        memory = _memory_from_globals()
        if memory is None or memory.field is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Memory not initialized")
            return rtmdk_pb2.CreateNodeResponse()
        try:
            nid = memory.add_node(
                request.content,
                node_id=request.node_id or None,
                metadata=dict(request.metadata),
            )
            return rtmdk_pb2.CreateNodeResponse(node_id=nid, status="created")
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return rtmdk_pb2.CreateNodeResponse()

    def GetNode(self, request, context):
        memory = _memory_from_globals()
        if memory is None or memory.field is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Memory not initialized")
            return rtmdk_pb2.Node()
        node = memory.field.nodes.get(request.node_id)
        if node is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Node not found")
            return rtmdk_pb2.Node()
        content = ""
        if isinstance(node.content, dict):
            content = node.content.get("text", str(node.content))
        else:
            content = str(node.content)
        return rtmdk_pb2.Node(
            id=node.id,
            content=content,
            salience=float(node.salience),
            phase=float(node.phase),
            amplitude=float(node.amplitude),
        )

    def DeleteNode(self, request, context):
        if not _require_key(context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid API key")
            return rtmdk_pb2.DeleteNodeResponse()
        memory = _memory_from_globals()
        if memory is None or memory.field is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Memory not initialized")
            return rtmdk_pb2.DeleteNodeResponse()
        node = memory.field.nodes.get(request.node_id)
        if node is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Node not found")
            return rtmdk_pb2.DeleteNodeResponse()
        memory.field.delete_nodes([request.node_id])
        return rtmdk_pb2.DeleteNodeResponse(status="deleted")

    def BatchIngest(self, request, context):
        if not _require_key(context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid API key")
            return rtmdk_pb2.BatchIngestResponse()
        memory = _memory_from_globals()
        if memory is None or memory.field is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Memory not initialized")
            return rtmdk_pb2.BatchIngestResponse()
        try:
            ids = memory.add_nodes_batch(
                list(request.contents),
                node_ids=list(request.node_ids) if request.node_ids else None,
            )
            return rtmdk_pb2.BatchIngestResponse(
                ingested=len(ids), node_ids=ids)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return rtmdk_pb2.BatchIngestResponse()


def create_server(port: int = 50051, max_workers: int = 10) -> grpc.Server:
    """Create and return a gRPC server instance."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    rtmdk_pb2_grpc.add_RTMDKServicer_to_server(RTMDKServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    return server


async def serve_grpc(port: int = 50051):
    """Start gRPC server in an asyncio-compatible way."""
    server = create_server(port=port)
    server.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        server.stop(5)
